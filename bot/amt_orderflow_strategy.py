from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from collections import Counter
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import logging
from bot.logger_config import log as bot_log

from bot.exchange.bybit_client import BybitClient
from bot.strategy import Signal, Action

logger = logging.getLogger(__name__)

_DEBUG_ENABLED: bool = False
_DEBUG_COUNTER: Counter = Counter()

def set_amt_debug(enabled: bool = True, reset: bool = True) -> None:
    global _DEBUG_ENABLED, _DEBUG_COUNTER
    _DEBUG_ENABLED = bool(enabled)
    if reset:
        _DEBUG_COUNTER = Counter()

def get_amt_debug_stats(reset: bool = False) -> Dict[str, int]:
    global _DEBUG_COUNTER
    out = dict(_DEBUG_COUNTER)
    if reset:
        _DEBUG_COUNTER = Counter()
    return out

def _dbg(key: str, n: int = 1, symbol: Optional[str] = None) -> None:
    if not _DEBUG_ENABLED:
        return
    if symbol:
        _DEBUG_COUNTER[f"{symbol}:{key}"] += n
    else:
        _DEBUG_COUNTER[key] += n

def _safe_utc_timestamp(dt: datetime) -> pd.Timestamp:
    """Return UTC-aware pandas Timestamp from naive/aware datetime without tz constructor errors."""
    return pd.Timestamp(dt).tz_convert("UTC") if dt.tzinfo else pd.Timestamp(dt, tz="UTC")

def _candle_context_ok_for_absorption(df_ohlcv: pd.DataFrame, vol_mult: float = 1.3, max_body_ratio: float = 0.35) -> bool:
    """Absorption works best on high volume + small body (pin/doji) candles."""
    if df_ohlcv.empty or len(df_ohlcv) < 20:
        return True
    last = df_ohlcv.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return False
    body = float(abs(last["close"] - last["open"]))
    body_ratio = body / rng
    v_avg = float(df_ohlcv["volume"].tail(20).mean())
    v_ok = float(last["volume"]) >= (v_avg * vol_mult) if v_avg > 0 else True
    return v_ok and (body_ratio <= max_body_ratio)

def _breakout_quality_ok(df_ohlcv: pd.DataFrame, min_body_ratio: float = 0.6, min_vol_mult: float = 1.2) -> bool:
    """Breakout needs a strong body + elevated volume."""
    if df_ohlcv.empty or len(df_ohlcv) < 20:
        return True
    last = df_ohlcv.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return False
    body = float(abs(last["close"] - last["open"]))
    body_ratio = body / rng
    v_avg = float(df_ohlcv["volume"].tail(20).mean())
    vol_ok = float(last["volume"]) >= (v_avg * min_vol_mult) if v_avg > 0 else True
    return (body_ratio >= min_body_ratio) and vol_ok

def _breakout_close_near_extreme_ok(df_ohlcv: pd.DataFrame, action: Action, close_pos_thresh: float = 0.8) -> bool:
    """
    Extra breakout confirmation:
    - LONG: close should be in top X% of candle range
    - SHORT: close should be in bottom X% of candle range
    """
    if df_ohlcv.empty:
        return True
    last = df_ohlcv.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return False
    close_pos = (float(last["close"]) - float(last["low"])) / rng  # 0..1
    if action == Action.LONG:
        return close_pos >= close_pos_thresh
    return close_pos <= (1.0 - close_pos_thresh)

def _breakout_candle_direction_ok(df_ohlcv: pd.DataFrame, action: Action) -> bool:
    if df_ohlcv.empty:
        return True
    last = df_ohlcv.iloc[-1]
    return (last["close"] >= last["open"]) if action == Action.LONG else (last["close"] <= last["open"])

def _cvd_pressure_matches_reversal(trades_df: pd.DataFrame, action: Action) -> bool:
    """
    For absorption reversal:
    - LONG: expect net selling pressure (CVD slope < 0) being absorbed near support
    - SHORT: expect net buying pressure (CVD slope > 0) being absorbed near resistance
    """
    if trades_df.empty or len(trades_df) < 20:
        return True
    recent = trades_df.tail(40).copy()
    recent["signed_vol"] = np.where(recent["side"].str.upper() == "BUY", recent["qty"], -recent["qty"])
    cvd = recent["signed_vol"].cumsum()
    slope = float(cvd.iloc[-1] - cvd.iloc[0])
    return slope < 0 if action == Action.LONG else slope > 0

@dataclass
class AbsorptionConfig:
    lookback_seconds: int = 180
    min_total_volume: float = 1000.0
    min_buy_sell_ratio: float = 1.8
    max_price_drift_pct: float = 0.3
    min_cvd_delta: float = 800.0

@dataclass
class VolumeProfileConfig:
    price_step: float = 10.0
    value_area_pct: float = 0.70

@dataclass
class LhOrderflowConfig:
    cvd_spike_mult: float = 2.5

@dataclass
class SymbolSettings:
    absorption: AbsorptionConfig
    volume_profile: VolumeProfileConfig
    liquidation_hunter: LhOrderflowConfig

AMT_CONFIG_REGISTRY = {
    "BTCUSDT": SymbolSettings(
        absorption=AbsorptionConfig(lookback_seconds=180, min_total_volume=1500.0, min_cvd_delta=500.0, min_buy_sell_ratio=2.2, max_price_drift_pct=0.5),
        volume_profile=VolumeProfileConfig(price_step=10.0),
        liquidation_hunter=LhOrderflowConfig(cvd_spike_mult=3.0),
    ),
    "ETHUSDT": SymbolSettings(
        absorption=AbsorptionConfig(lookback_seconds=180, min_total_volume=300.0, min_cvd_delta=100.0, min_buy_sell_ratio=2.2, max_price_drift_pct=0.5),
        volume_profile=VolumeProfileConfig(price_step=1.0),
        liquidation_hunter=LhOrderflowConfig(cvd_spike_mult=3.0),
    ),
    "SOLUSDT": SymbolSettings(
        absorption=AbsorptionConfig(lookback_seconds=180, min_total_volume=800.0, min_cvd_delta=300.0, min_buy_sell_ratio=2.2, max_price_drift_pct=0.6),
        volume_profile=VolumeProfileConfig(price_step=0.1),
        liquidation_hunter=LhOrderflowConfig(cvd_spike_mult=3.0),
    ),
}

def _resolve_symbol_settings(symbol: str, settings: Optional["SymbolSettings"] = None) -> "SymbolSettings":
    """
    Backwards-compatible helper for live-trading code.

    Historically `bot/live.py` imported `_resolve_symbol_settings` from this module.
    Some refactors moved settings resolution inline, but we keep this function to
    avoid breaking imports and to provide a single place to apply defaults.

    Args:
        symbol: e.g. "BTCUSDT"
        settings: optional pre-built SymbolSettings; if provided, returned as-is.

    Returns:
        SymbolSettings resolved from registry or defaults.
    """
    if settings is not None:
        return settings
    return AMT_CONFIG_REGISTRY.get(
        symbol,
        SymbolSettings(AbsorptionConfig(), VolumeProfileConfig(), LhOrderflowConfig()),
    )

def resolve_final_amt_configs(
    symbol: str,
    strategy_settings: Any,
    current_time_utc: Optional[datetime] = None,
    use_adaptive_volume: bool = False
) -> tuple[VolumeProfileConfig, AbsorptionConfig]:
    """
    Resolve final AMT configs for live trading.
    
    Args:
        symbol: Trading symbol (e.g. "BTCUSDT")
        strategy_settings: Strategy settings object (may contain AMT-specific params)
        current_time_utc: Current UTC time for adaptive volume adjustments
        use_adaptive_volume: Whether to apply time-of-day volume adjustments
    
    Returns:
        Tuple of (VolumeProfileConfig, AbsorptionConfig)
    """
    sym_settings = AMT_CONFIG_REGISTRY.get(
        symbol,
        SymbolSettings(AbsorptionConfig(), VolumeProfileConfig(), LhOrderflowConfig()),
    )
    
    vp_cfg = sym_settings.volume_profile
    abs_cfg = sym_settings.absorption
    
    # Apply adaptive volume adjustments if requested
    if use_adaptive_volume and current_time_utc:
        hour = current_time_utc.hour
        # Asian session (0-8 UTC): lower liquidity -> higher thresholds
        # European/American session (8-24 UTC): higher liquidity -> lower thresholds
        if 0 <= hour < 8:
            # Asian session: increase volume thresholds by 20%
            abs_cfg = replace(
                abs_cfg,
                min_total_volume=abs_cfg.min_total_volume * 1.2,
                min_cvd_delta=abs_cfg.min_cvd_delta * 1.2,
            )
        # else: use default thresholds for European/American sessions
    
    return vp_cfg, abs_cfg

def calculate_atr(df_ohlcv: pd.DataFrame, period: int = 14) -> float:
    if len(df_ohlcv) < period + 1: return 0.0
    h, l, c = df_ohlcv['high'], df_ohlcv['low'], df_ohlcv['close'].shift(1)
    tr = pd.concat([h - l, abs(h - c), abs(l - c)], axis=1).max(axis=1)
    return float(tr.rolling(window=period).mean().iloc[-1])

def calculate_rsi(df_ohlcv: pd.DataFrame, period: int = 14) -> float:
    if len(df_ohlcv) < period + 1: return 50.0
    delta = df_ohlcv['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    if loss.iloc[-1] == 0: return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))

def enhanced_trend_filter(df_ohlcv: pd.DataFrame, current_price: float) -> tuple[bool, bool]:
    """Trend votes (2/3) using EMA 50/100/200 when available."""
    if len(df_ohlcv) < 50:
        return True, True
    e50 = df_ohlcv["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    e100 = df_ohlcv["close"].ewm(span=100, adjust=False).mean().iloc[-1] if len(df_ohlcv) >= 100 else e50
    e200 = df_ohlcv["close"].ewm(span=200, adjust=False).mean().iloc[-1] if len(df_ohlcv) >= 200 else e100
    bull_votes = int(current_price > e50) + int(current_price > e100) + int(current_price > e200)
    bear_votes = int(current_price < e50) + int(current_price < e100) + int(current_price < e200)
    return bull_votes >= 2, bear_votes >= 2

def generate_amt_signals(
    client: BybitClient,
    symbol: str,
    current_price: float,
    df_ohlcv: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    current_time: Optional[datetime] = None,
    vp_config: Optional[VolumeProfileConfig] = None,
    abs_config: Optional[AbsorptionConfig] = None,
    delta_aggr_mult: Optional[float] = None,
) -> List[Signal]:
    signals: List[Signal] = []
    if df_ohlcv.empty or len(df_ohlcv) < 50:
        _dbg("reject:ohlcv_short", symbol=symbol)
        return signals
    
    sym_c = AMT_CONFIG_REGISTRY.get(symbol, SymbolSettings(AbsorptionConfig(), VolumeProfileConfig(), LhOrderflowConfig()))
    can_long, can_short = enhanced_trend_filter(df_ohlcv, current_price)
    rsi = calculate_rsi(df_ohlcv)
    # IMPORTANT: build VP on PREVIOUS candles to avoid “moving boundary” on breakout candle
    if len(df_ohlcv) >= 101:
        vp_df = df_ohlcv.tail(101).iloc[:-1]
    else:
        vp_df = df_ohlcv.tail(100).iloc[:-1] if len(df_ohlcv) > 1 else df_ohlcv
    vp = build_volume_profile_from_ohlcv(vp_df, sym_c.volume_profile)
    
    if trades_df is None:
        _dbg("reject:no_trades_df", symbol=symbol)
        return []
    
    cvd_m = _compute_cvd_metrics(trades_df, 180, current_time)
    if not cvd_m:
        _dbg("reject:no_cvd_metrics", symbol=symbol)
        return []
    dv, avg_d = cvd_m["delta_velocity"], cvd_m["avg_abs_delta"]
    atr14 = calculate_atr(df_ohlcv, period=14)
    
    now_dt = current_time or datetime.now(timezone.utc)
    now = _safe_utc_timestamp(now_dt)

    # ABSORPTION: disabled for BTC/ETH/SOL in backtest (synthetic ticks -> noisy, no edge seen)
    # In live trading you may want to re-enable with real tick data.
    if vp and symbol not in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        if not _candle_context_ok_for_absorption(df_ohlcv, vol_mult=1.25, max_body_ratio=0.40):
            _dbg("abs:reject:candle_context", symbol=symbol)
        else:
            _dbg("abs:candidate", symbol=symbol)
            p_rng = float(vp["vah"] - vp["val"])
            if p_rng > 0:
                is_near_vah = current_price >= (vp["vah"] - p_rng * 0.20)
                is_near_val = current_price <= (vp["val"] + p_rng * 0.20)
            else:
                is_near_vah = is_near_val = True

            abs_s = detect_absorption_squeeze_short(
                client, symbol, current_price, sym_c.absorption, trades_df, current_time
            )
            if abs_s:
                _dbg("abs:raw_hit", symbol=symbol)
                if not is_near_vah:
                    _dbg("abs:reject:not_near_vah", symbol=symbol)
                elif not (rsi > 35):
                    _dbg("abs:reject:rsi_short", symbol=symbol)
                elif not can_short:
                    _dbg("abs:reject:trend_short", symbol=symbol)
                elif not _cvd_pressure_matches_reversal(trades_df, Action.SHORT):
                    _dbg("abs:reject:cvd_pressure_short", symbol=symbol)
                else:
                    _dbg("abs:accepted", symbol=symbol)
                    signals.append(abs_s)

            abs_l = detect_absorption_squeeze_long(
                client, symbol, current_price, sym_c.absorption, trades_df, current_time
            )
            if abs_l:
                _dbg("abs:raw_hit", symbol=symbol)
                if not is_near_val:
                    _dbg("abs:reject:not_near_val", symbol=symbol)
                elif not (rsi < 65):
                    _dbg("abs:reject:rsi_long", symbol=symbol)
                elif not can_long:
                    _dbg("abs:reject:trend_long", symbol=symbol)
                elif not _cvd_pressure_matches_reversal(trades_df, Action.LONG):
                    _dbg("abs:reject:cvd_pressure_long", symbol=symbol)
                else:
                    _dbg("abs:accepted", symbol=symbol)
                    signals.append(abs_l)

    # BREAKOUT: ETHUSDT disabled completely (0% WinRate in backtest)
    # BTC and SOL have stricter filters based on backtest results
    if vp and symbol != "ETHUSDT":
        # make BTC strictest (was losing), SOL looser (was winning)
        if symbol == "BTCUSDT":
            # BTC: longs underperform -> make LONG stricter than SHORT
            mult_long = 7.0
            mult_short = 6.0
            q_vol_mult_long = 1.50
            q_vol_mult_short = 1.40
            clearance_long = 0.40
            clearance_short = 0.35
            rsi_long_max = 65   # was 75
            rsi_short_min = 25  # keep
            close_pos_thresh_long = 0.85
            close_pos_thresh_short = 0.80
        else:  # SOLUSDT and others
            # Slightly stricter for SOL (it was marginally losing): filter more fakeouts
            mult_long = 4.5
            mult_short = 4.5
            q_vol_mult_long = 1.25
            q_vol_mult_short = 1.25
            clearance_long = 0.30
            clearance_short = 0.30
            rsi_long_max = 75
            rsi_short_min = 25
            close_pos_thresh_long = 0.80
            close_pos_thresh_short = 0.80
        # breakout long candidate
        if current_price > vp["vah"]:
            _dbg("brk:above_vah", symbol=symbol)
            if not (dv > avg_d * mult_long):
                _dbg("brk:reject:dv_long", symbol=symbol)
            elif not can_long:
                _dbg("brk:reject:trend_long", symbol=symbol)
            elif not (rsi < rsi_long_max):
                _dbg("brk:reject:rsi_long", symbol=symbol)
            elif not _breakout_candle_direction_ok(df_ohlcv, Action.LONG):
                _dbg("brk:reject:dir_long", symbol=symbol)
            elif not _breakout_close_near_extreme_ok(df_ohlcv, Action.LONG, close_pos_thresh=close_pos_thresh_long):
                _dbg("brk:reject:closepos_long", symbol=symbol)
            elif atr14 > 0 and (current_price - float(vp["vah"])) < (atr14 * clearance_long):
                _dbg("brk:reject:clearance_long", symbol=symbol)
            elif not _breakout_quality_ok(df_ohlcv, min_body_ratio=0.6, min_vol_mult=q_vol_mult_long):
                _dbg("brk:reject:quality_long", symbol=symbol)
        else:
                _dbg("brk:accepted", symbol=symbol)
                signals.append(Signal(timestamp=now, action=Action.LONG, price=current_price, reason=f"brk_long_dv{int(dv)}"))
        # breakout short candidate
        if current_price < vp["val"]:
            _dbg("brk:below_val", symbol=symbol)
            if not (dv < -avg_d * mult_short):
                _dbg("brk:reject:dv_short", symbol=symbol)
            elif not can_short:
                _dbg("brk:reject:trend_short", symbol=symbol)
            elif not (rsi > rsi_short_min):
                _dbg("brk:reject:rsi_short", symbol=symbol)
            elif not _breakout_candle_direction_ok(df_ohlcv, Action.SHORT):
                _dbg("brk:reject:dir_short", symbol=symbol)
            elif not _breakout_close_near_extreme_ok(df_ohlcv, Action.SHORT, close_pos_thresh=close_pos_thresh_short):
                _dbg("brk:reject:closepos_short", symbol=symbol)
            elif atr14 > 0 and (float(vp["val"]) - current_price) < (atr14 * clearance_short):
                _dbg("brk:reject:clearance_short", symbol=symbol)
            elif not _breakout_quality_ok(df_ohlcv, min_body_ratio=0.6, min_vol_mult=q_vol_mult_short):
                _dbg("brk:reject:quality_short", symbol=symbol)
    else:
                _dbg("brk:accepted", symbol=symbol)
                signals.append(Signal(timestamp=now, action=Action.SHORT, price=current_price, reason=f"brk_short_dv{int(dv)}"))

    if len(signals) > 1:
        signals.sort(key=lambda s: 2 if "brk" in s.reason else 1, reverse=True)
        signals = signals[:1]
    return signals

def build_volume_profile_from_ohlcv(df: pd.DataFrame, config: Optional[VolumeProfileConfig] = None) -> Optional[Dict[str, float]]:
    if df.empty: return None
    cfg = config or VolumeProfileConfig(); step = cfg.price_step; profile = {}
    for _, row in df.iterrows():
        bin_p = round(row["close"] / step) * step
        profile[bin_p] = profile.get(bin_p, 0.0) + row["volume"]
    if not profile: return None
    poc = max(profile, key=profile.get); sorted_p = sorted(profile.items(), key=lambda x: abs(x[0] - poc))
    total_v = sum(profile.values()); target_v = total_v * 0.7; cur_v = 0; va_p = []
    for p, v in sorted_p:
        cur_v += v; va_p.append(p)
        if cur_v >= target_v: break
    return {"poc": poc, "vah": max(va_p), "val": min(va_p), "total_volume": total_v}

def _compute_cvd_metrics(trades_df: pd.DataFrame, lookback_seconds: int = 60, current_time: Optional[datetime] = None) -> Optional[Dict[str, float]]:
    if trades_df.empty: return None
    now_utc = current_time or datetime.now(timezone.utc)
    if now_utc.tzinfo is None: now_utc = now_utc.replace(tzinfo=timezone.utc)
    dt_lb = now_utc - timedelta(seconds=lookback_seconds)
    # Исправлено: используем tz_convert вместо передачи tz в конструктор
    start_ts = _safe_utc_timestamp(dt_lb)
    if trades_df["time"].dt.tz is None:
        trades_df = trades_df.copy()
        trades_df["time"] = pd.to_datetime(trades_df["time"], utc=True)
    w_df = trades_df[trades_df["time"] >= start_ts]
    if w_df.empty: return None
    w_df = w_df.copy(); w_df["signed_vol"] = np.where(w_df["side"].str.upper() == "BUY", w_df["qty"], -w_df["qty"])
    cvd_s = w_df["signed_vol"].cumsum()
    return {"delta_velocity": float(cvd_s.iloc[-1] - cvd_s.iloc[0]), "avg_abs_delta": float(w_df["signed_vol"].abs().mean()), "total_volume": float(w_df["qty"].sum())}

def _parse_trades(raw: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in raw:
        try:
            ts = int(t.get("time") or t.get("T") or t.get("timestamp") or 0)
            p = float(t.get("price") or t.get("p") or 0); q = float(t.get("qty") or t.get("q") or 0); s = str(t.get("side") or t.get("S") or "").upper()
            if ts == 0 or p <= 0 or q <= 0 or s not in ["BUY", "SELL"]: continue
            rows.append({"time": datetime.fromtimestamp(ts/1000.0, tz=timezone.utc), "price": p, "qty": q, "side": s})
        except: continue
    df = pd.DataFrame(rows)
    if not df.empty: df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time")

def detect_absorption_squeeze_short(client: BybitClient, symbol: str, current_price: float, config: Optional[AbsorptionConfig] = None, trades_df: Optional[pd.DataFrame] = None, current_time: Optional[datetime] = None) -> Optional[Signal]:
    cfg = config or AbsorptionConfig(); df_t = trades_df if trades_df is not None else _parse_trades(client.get_recent_trades(symbol, limit=1000))
    if df_t.empty: return None
    b_v = float(df_t.loc[df_t["side"].str.upper() == "BUY", "qty"].sum())
    s_v = float(df_t.loc[df_t["side"].str.upper() == "SELL", "qty"].sum())
    total_v = b_v + s_v; cvd_d = b_v - s_v; ratio = b_v / max(s_v, 0.001)
    if total_v >= cfg.min_total_volume and cvd_d >= cfg.min_cvd_delta and ratio >= cfg.min_buy_sell_ratio:
        now_dt = current_time or datetime.now(timezone.utc)
        now_ts = pd.Timestamp(now_dt).tz_convert('UTC') if now_dt.tzinfo else pd.Timestamp(now_dt, tz='UTC')
        p_chg = abs(df_t["price"].iloc[-1] - df_t["price"].iloc[0]) / df_t["price"].iloc[0] * 100
        if p_chg > cfg.max_price_drift_pct: return None
        return Signal(timestamp=now_ts, action=Action.SHORT, price=current_price, reason=f"abs_short_cvd_{int(cvd_d)}")
    return None

def detect_absorption_squeeze_long(client: BybitClient, symbol: str, current_price: float, config: Optional[AbsorptionConfig] = None, trades_df: Optional[pd.DataFrame] = None, current_time: Optional[datetime] = None) -> Optional[Signal]:
    cfg = config or AbsorptionConfig(); df_t = trades_df if trades_df is not None else _parse_trades(client.get_recent_trades(symbol, limit=1000))
    if df_t.empty: return None
    b_v = float(df_t.loc[df_t["side"].str.upper() == "BUY", "qty"].sum())
    s_v = float(df_t.loc[df_t["side"].str.upper() == "SELL", "qty"].sum())
    total_v = b_v + s_v; cvd_d = b_v - s_v; ratio = s_v / max(b_v, 0.001)
    if total_v >= cfg.min_total_volume and -cvd_d >= cfg.min_cvd_delta and ratio >= cfg.min_buy_sell_ratio:
        now_dt = current_time or datetime.now(timezone.utc)
        now_ts = pd.Timestamp(now_dt).tz_convert('UTC') if now_dt.tzinfo else pd.Timestamp(now_dt, tz='UTC')
        p_chg = abs(df_t["price"].iloc[-1] - df_t["price"].iloc[0]) / df_t["price"].iloc[0] * 100
        if p_chg > cfg.max_price_drift_pct: return None
        return Signal(timestamp=now_ts, action=Action.LONG, price=current_price, reason=f"abs_long_cvd_{int(cvd_d)}")
    return None

def get_signals_for_symbol(client: BybitClient, symbol: str, current_price: float, df_ohlcv: pd.DataFrame, settings: Any, trades_df: Optional[pd.DataFrame] = None, current_time: Optional[datetime] = None) -> List[Signal]:
    return generate_amt_signals(client, symbol, current_price, df_ohlcv, trades_df, current_time)

__all__ = [
    "AbsorptionConfig",
    "VolumeProfileConfig",
    "LhOrderflowConfig",
    "SymbolSettings",
    "build_volume_profile_from_ohlcv",
    "detect_absorption_squeeze_short",
    "detect_absorption_squeeze_long",
    "generate_amt_signals",
    "get_signals_for_symbol",
    "AMT_CONFIG_REGISTRY",
    "calculate_atr",
    "_parse_trades",
    "_compute_cvd_metrics",
    "_resolve_symbol_settings",
    "resolve_final_amt_configs",
]
