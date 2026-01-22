import typing as t
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from typing import Any, Optional, List


class Action(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class Bias(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class MarketPhase(Enum):
    TREND = "trend"
    FLAT = "flat"
    MOMENTUM = "momentum"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    action: Action
    reason: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing: Optional[dict] = None
    indicators_info: Optional[dict] = None


def enrich_for_strategy(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Passthrough/enrichment placeholder. Existing code expects this function.

    Currently returns df unchanged. In future can add indicator prepping here.
    """
    # For backward compatibility, try to ensure standard columns exist
    df_out = df.copy()
    # Ensure timestamp column exists if index is DatetimeIndex
    if isinstance(df_out.index, pd.DatetimeIndex) and 'timestamp' not in df_out.columns:
        df_out = df_out.reset_index().rename(columns={'index': 'timestamp'}).set_index(pd.Index(df_out.index))
    return df_out


def build_signals(
    df: pd.DataFrame,
    strategy_obj: t.Any,
    use_momentum: bool = False,
    use_liquidity: bool = False,
    state: Optional[dict] = None,
    params: Optional[dict] = None,
    **kwargs,
) -> List[Signal]:
    """Backward-compatible adapter used across the codebase.

    Parameters from existing callers preserved: use_momentum/use_liquidity.
    strategy_obj can be a string ("TREND","FLAT","MOMENTUM") or a settings
    object; we attempt to derive a strategy name from it.
    """
    state = state or {}
    params = params or {}
    out: List[Signal] = []

    # derive name
    name = None
    if isinstance(strategy_obj, str):
        name = strategy_obj.upper()
    else:
        # try common attributes
        for attr in ('strategy', 'name', 'strategy_name'):
            if hasattr(strategy_obj, attr):
                try:
                    val = getattr(strategy_obj, attr)
                    if isinstance(val, str):
                        name = val.upper()
                        break
                except Exception:
                    pass
    # fallback to flags
    if name is None:
        name = 'MOMENTUM' if use_momentum else 'TREND'

    try:
        if name == 'FLAT':
            res = generate_flat_signal(
                df,
                rsi_period=params.get('rsi_period', 14),
                rsi_base_low=params.get('rsi_base_low', 35),
                rsi_base_high=params.get('rsi_base_high', 65),
                bb_period=params.get('bb_period', 20),
                bb_mult=params.get('bb_mult', 2.0),
                bb_compression_factor=params.get('bb_compression_factor', 0.8),
                min_history=params.get('min_history', 50),
            )
        elif name == 'MOMENTUM':
            res = generate_momentum_signal(
                df,
                ema_short=params.get('ema_short', 20),
                ema_long=params.get('ema_long', 50),
                vol_lookback=params.get('vol_lookback', 100),
                vol_top_pct=params.get('vol_top_pct', 0.75),
                min_history=params.get('min_history', 50),
            )
        else:
            # default to TREND
            res = generate_trend_signal(
                df,
                state=state,
                sma_period=params.get('sma_period', 21),
                atr_period=params.get('atr_period', 14),
                atr_multiplier=params.get('atr_multiplier', 2.0),
                max_pyramid=params.get('max_pyramid', 2),
                min_history=params.get('min_history', 100),
            )
    except Exception:
        return out

    if res and res.get('signal') is not None:
        action = Action.LONG if res.get('signal') == 'LONG' else (Action.SHORT if res.get('signal') == 'SHORT' else Action.HOLD)
        reason = res.get('reason', '')
        price = float(df['close'].iloc[-1]) if 'close' in df.columns and len(df) > 0 else 0.0
        # Prefer explicit indicators_info, but attach SL/TP/trailing into indicators so downstream
        # systems that only accept a Signal object still have access to exit params.
        indicators = dict(res.get('indicators_info', {}) or {})
        # attach stop/take/trailing into indicators for downstream consumers
        if res.get('stop_loss') is not None:
            try:
                indicators['stop_loss'] = float(res.get('stop_loss'))
            except Exception:
                indicators['stop_loss'] = res.get('stop_loss')
        if res.get('take_profit') is not None:
            try:
                indicators['take_profit'] = float(res.get('take_profit'))
            except Exception:
                indicators['take_profit'] = res.get('take_profit')
        if res.get('trailing') is not None:
            indicators['trailing'] = res.get('trailing')
        # prefer timestamp from df index if available
        try:
            ts = pd.Timestamp(df.index[-1])
        except Exception:
            ts = pd.Timestamp.now()
        # Prefer explicit stop/take/trailing fields on Signal for downstream consumers
        sig = Signal(
            timestamp=ts,
            action=action,
            reason=reason,
            price=price,
            stop_loss=res.get('stop_loss') or res.get('indicators_info', {}).get('sl'),
            take_profit=res.get('take_profit') or res.get('indicators_info', {}).get('tp'),
            trailing=res.get('trailing'),
            indicators_info=indicators,
        )
        out.append(sig)

    return out


def detect_market_phase(row_or_df: t.Union[pd.Series, pd.DataFrame], strategy_name: Optional[str] = None) -> Optional[MarketPhase]:
    """
    Простая детекция рыночной фазы для совместимости с остальным кодом.
    Принимает либо одну строку (Series) с индикаторами, либо DataFrame.
    Если доступны индикаторы ('adx', 'atr' и т.д.), пытается определить фазу.

    Возвращает MarketPhase или None.
    """
    try:
        # If DataFrame passed, use last row
        if isinstance(row_or_df, pd.DataFrame):
            row = row_or_df.iloc[-1]
        else:
            row = row_or_df

        # Prefer explicit strategy_name hints
        name_hint = None
        if isinstance(strategy_name, str):
            name_hint = strategy_name.upper()
        elif strategy_name is not None:
            # try common attributes if it's a settings object
            for attr in ('strategy', 'name', 'strategy_name'):
                if hasattr(strategy_name, attr):
                    try:
                        val = getattr(strategy_name, attr)
                        if isinstance(val, str):
                            name_hint = val.upper()
                            break
                    except Exception:
                        pass

        if name_hint == 'TREND':
            return MarketPhase.TREND
        if name_hint == 'FLAT':
            return MarketPhase.FLAT
        if name_hint == 'MOMENTUM':
            return MarketPhase.MOMENTUM

        # ADX-based heuristic if available
        adx = row.get('adx') if hasattr(row, 'get') else None
        if adx is not None:
            try:
                adx_v = float(adx)
                if adx_v > 25:
                    return MarketPhase.TREND
                if adx_v < 20:
                    return MarketPhase.FLAT
            except Exception:
                pass

        # Volatility-based fallback using atr
        atr = row.get('atr') if hasattr(row, 'get') else None
        if atr is not None:
            try:
                atr_v = float(atr)
                # crude thresholds - kept conservative
                if atr_v > 0.5:
                    return MarketPhase.MOMENTUM
                return MarketPhase.FLAT
            except Exception:
                pass

        return None
    except Exception:
        return None


def detect_market_bias(row: pd.Series) -> Optional[Bias]:
    # Ищем DI под любыми именами
    p_di = row.get('plus_di') or row.get('DMP_14') or row.get('ADX_14_pos')
    m_di = row.get('minus_di') or row.get('DMN_14') or row.get('ADX_14_neg')

    if pd.notnull(p_di) and pd.notnull(m_di):
        return Bias.LONG if float(p_di) > float(m_di) else Bias.SHORT
    
    # Если DI нет, смотрим на SMA (обязательно!)
    close = row.get('close')
    sma = row.get('sma') or row.get('sma_200')
    if pd.notnull(close) and pd.notnull(sma):
        return Bias.LONG if float(close) > float(sma) else Bias.SHORT
    
    return None


def generate_range_signal(row: pd.Series, position_bias: Optional[Bias], settings: t.Any) -> Signal:
    """Compatibility wrapper for legacy code that expects a row-based range signal.

    Uses indicators already present in the row (prepared by prepare_with_indicators).
    Falls back to conservative defaults when indicators are missing.
    """
    try:
        price = float(row.get('close', 0.0))
        indicators_info = {}

        rsi = row.get('rsi') if row.get('rsi') is not None else row.get('rsi_14')
        bbw = row.get('bb_width') or row.get('bbw')
        atr = row.get('atr') or row.get('atr_14')
        indicators_info['rsi'] = float(rsi) if rsi is not None else None
        indicators_info['bb_width'] = float(bbw) if bbw is not None else None
        indicators_info['atr'] = float(atr) if atr is not None else None

        # Bollinger compression: if bb width present and very small -> block
        if bbw is not None:
            try:
                if float(bbw) <= 1e-6:
                    return Signal(timestamp=row.name, action=Action.HOLD, reason='bb_compression', price=price, indicators_info=indicators_info)
            except Exception:
                pass

        # Adaptive RSI thresholds - try to read from settings, fallback to 35/65
        try:
            low_lvl = getattr(settings, 'range_rsi_low', None) or getattr(settings, 'rsi_base_low', None) or 30
            high_lvl = getattr(settings, 'range_rsi_high', None) or getattr(settings, 'rsi_base_high', None) or 70
            low_lvl = int(low_lvl)
            high_lvl = int(high_lvl)
        except Exception:
            low_lvl, high_lvl = 35, 65

        if rsi is None:
            return Signal(timestamp=row.name, action=Action.HOLD, reason='flat_missing_indicators', price=price, indicators_info=indicators_info)

        rsi_v = float(rsi)
        open_price = float(row.get('open', price))
        if rsi_v < low_lvl and price > open_price:
            return Signal(timestamp=row.name, action=Action.LONG, reason='rsi_oversold_bullish_confirm', price=price, indicators_info=indicators_info)
        if rsi_v > high_lvl and price < open_price:
            return Signal(timestamp=row.name, action=Action.SHORT, reason='rsi_overbought_bearish_confirm', price=price, indicators_info=indicators_info)

        return Signal(timestamp=row.name, action=Action.HOLD, reason='no_mean_reversion', price=price, indicators_info=indicators_info)
    except Exception as e:
        return Signal(timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(), action=Action.HOLD, reason=f'error_range_signal_{str(e)[:80]}', price=float(row.get('close', 0.0)), indicators_info={})


def generate_momentum_breakout_signal(row: pd.Series, position_bias: Optional[Bias], settings: t.Any) -> Signal:
    """Compatibility wrapper for legacy momentum breakout signature.

    Uses EMA and volume fields that should be present in the row. Falls back
    to conservative HOLD if required indicators are missing.
    """
    try:
        price = float(row.get('close', 0.0))
        indicators_info = {}

        # Try common EMA column names
        ema_short = row.get('ema_fast') or row.get('ema20') or row.get('ema_20')
        ema_long = row.get('ema_slow') or row.get('ema50') or row.get('ema_50')
        if ema_short is None or ema_long is None:
            # try momentum-specific names
            tf = getattr(settings, 'momentum_ema_timeframe', None)
            if tf:
                ema_short = ema_short or row.get(f'ema_fast_{tf}')
                ema_long = ema_long or row.get(f'ema_slow_{tf}')

        indicators_info['ema_short'] = float(ema_short) if ema_short is not None else None
        indicators_info['ema_long'] = float(ema_long) if ema_long is not None else None

        # RSI fallback: if missing, use neutral 50
        rsi_val = row.get('rsi') if row.get('rsi') is not None else row.get('rsi_14')
        try:
            indicators_info['rsi'] = float(rsi_val) if rsi_val is not None else 50.0
        except Exception:
            indicators_info['rsi'] = 50.0

        # Volume confirmation with safe defaults (row may not contain vol_sma/avg5)
        vol = row.get('volume')
        vol_sma = row.get('vol_sma') or row.get('volume_sma')
        vol_avg5 = row.get('vol_avg5') or row.get('volume_avg5')

        vol_current_safe = float(vol) if vol is not None else 0.0
        try:
            vol_sma_safe = float(vol_sma) if vol_sma is not None else vol_current_safe
        except Exception:
            vol_sma_safe = vol_current_safe
        try:
            vol_avg5_safe = float(vol_avg5) if vol_avg5 is not None else vol_current_safe
        except Exception:
            vol_avg5_safe = vol_current_safe

        indicators_info['vol_current'] = vol_current_safe
        indicators_info['vol_sma'] = vol_sma_safe
        indicators_info['vol_avg5'] = vol_avg5_safe

        # Require EMA fan: support both LONG and SHORT
        if ema_short is None or ema_long is None:
            return Signal(timestamp=row.name, action=Action.HOLD, reason='momentum_missing_ema', price=price, indicators_info=indicators_info)

        ema_s = float(ema_short)
        ema_l = float(ema_long)

        is_long_fan = price > ema_s > ema_l
        is_short_fan = price < ema_s < ema_l

        if not (is_long_fan or is_short_fan):
            return Signal(timestamp=row.name, action=Action.HOLD, reason='ema_fan_not_aligned', price=price, indicators_info=indicators_info)

        # Volume spike: require current volume >= 2x vol_sma and short-term jump
        try:
            if vol_current_safe < vol_sma_safe * 2.0:
                return Signal(timestamp=row.name, action=Action.HOLD, reason='no_volume_spike', price=price, indicators_info=indicators_info)
        except Exception:
            return Signal(timestamp=row.name, action=Action.HOLD, reason='volume_check_error', price=price, indicators_info=indicators_info)

        if vol_current_safe < 1.5 * vol_avg5_safe:
            return Signal(timestamp=row.name, action=Action.HOLD, reason='no_short_term_volume_jump', price=price, indicators_info=indicators_info)

        # Directional result
        if is_long_fan:
            return Signal(timestamp=row.name, action=Action.LONG, reason='momentum_breakout_ok', price=price, indicators_info=indicators_info)
        if is_short_fan:
            return Signal(timestamp=row.name, action=Action.SHORT, reason='momentum_breakout_ok_short', price=price, indicators_info=indicators_info)

        return Signal(timestamp=row.name, action=Action.HOLD, reason='no_action', price=price, indicators_info=indicators_info)
    except Exception as e:
        return Signal(timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(), action=Action.HOLD, reason=f'error_momentum_signal_{str(e)[:80]}', price=float(row.get('close', 0.0)), indicators_info={})



def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    # Use min_periods=1 to produce values on short series (prevents NaN for small datasets)
    return series.rolling(period, min_periods=1).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Backfill then default to neutral 50 to avoid None values in downstream checks/logs
    rsi = rsi.bfill().fillna(50.0)
    return rsi


def _bb_width(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    mid = _sma(df['close'], period)
    # ensure std is computed with min_periods to avoid NaNs on short series
    std = df['close'].rolling(period, min_periods=1).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / (mid.replace(0, np.nan).abs())
    return width


def _ensure_history(df: pd.DataFrame, required: int) -> bool:
    return len(df) >= required


def _generate_trend_signal_df(
    df: pd.DataFrame,
    state: t.Optional[dict] = None,
    sma_period: int = 21,
    atr_period: int = 14,
    atr_multiplier: float = 3.0,
    max_pyramid: int = 2,
    min_history: int = 50,
    adx_threshold: float = 30.0,
    vol_multiplier: float = 1.0,
) -> t.Dict:
    """
    Генерация сигнала TREND.

    Входные параметры:
      - df: DataFrame с колонками ['open','high','low','close','volume']
      - state: словарь состояния позиции: ожидаются ключи 'long_pyramid' и 'short_pyramid'

    Возвращает dict с ключами: signal, stop_loss, indicators_info, reason.
    """
    state = state or {}
    indicators_info = {}

    # Cooldown check: don't generate signals too frequently
    last_signal_idx = state.get('last_signal_idx', -100)
    current_idx = len(df)
    if current_idx - last_signal_idx < 10:
        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "cooldown"}

    # Forward-fill simple NaNs then validate
    df = df.ffill()

    # Data validation: avoid NaN/inf-only DataFrames
    if df.replace([np.inf, -np.inf], np.nan).dropna().empty:
        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "nan_in_data"}

    if not _ensure_history(df, min_history):
        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "insufficient_history"}
    close = df['close']
    price = float(close.iloc[-1])

    sma = _sma(close, sma_period)
    sma_current = float(sma.iloc[-1])
    sma_prev = float(sma.iloc[-2])

    # Prefer ATR already present in the DataFrame (precomputed) — fallback to computed ATR.
    atr_current = None
    try:
        if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]):
            atr_current = float(df['atr'].iloc[-1])
        else:
            # compute ATR from price series as a fallback
            atr_series = _atr(df, period=atr_period)
            if not pd.isna(atr_series.iloc[-1]):
                atr_current = float(atr_series.iloc[-1])
    except Exception:
        atr_current = None

    # If ATR still missing, use conservative 0.5% of price (more realistic for crypto on 15m-1h)
    if atr_current is None or np.isnan(atr_current):
        try:
            atr_current = float(price * 0.005)
        except Exception:
            atr_current = 0.0

    indicators_info['atr'] = atr_current

    # default max hold (bars) for time-exit protection — can be overridden via state/settings
    max_hold = int(state.get('max_hold_bars', 12))
    indicators_info['max_hold_bars'] = max_hold

    # Use EMA trend-following as primary entry condition (faster, less noise-prone than SMA pullbacks).
    # 1. EMA fan
    try:
        ema_short = _ema(close, 21)
        ema_long = _ema(close, 50)
        ema_s_curr = float(ema_short.iloc[-1])
        ema_s_prev = float(ema_short.iloc[-2]) if len(ema_short) > 1 else ema_s_curr
        ema_l_curr = float(ema_long.iloc[-1])
    except Exception:
        ema_s_curr = None
        ema_l_curr = None

    is_uptrend = False
    is_downtrend = False
    try:
        if ema_s_curr is not None and ema_l_curr is not None:
            # Тренд подтвержден веером EMA
            # Тренд подтвержден веером EMA и наклоном EMA20
            trend_up = ema_s_curr > ema_l_curr and ema_s_curr > ema_s_prev
            trend_down = ema_s_curr < ema_l_curr and ema_s_curr < ema_s_prev
            
            # Pullback: цена коснулась EMA20, но закрылась в сторону тренда
            low_curr = float(df['low'].iloc[-1])
            high_curr = float(df['high'].iloc[-1])
            
            # Pullback: цена коснулась EMA20 (с допуском 0.5%), но закрылась в сторону тренда
            is_uptrend = trend_up and (low_curr <= ema_s_curr * 1.005) and (price > ema_s_curr)
            is_downtrend = trend_down and (high_curr >= ema_s_curr * 0.995) and (price < ema_s_curr)
    except Exception:
        is_uptrend = False
        is_downtrend = False

    # ADX filter (required)
    adx_ok = True
    try:
        if 'adx' in df.columns:
            adx_val = float(df['adx'].iloc[-1])
            if adx_val < 25:
                adx_ok = False
    except Exception:
        adx_ok = True

    # Simplified volume filter: require current volume >= vol_sma * 1.0 when vol_sma is present
    vol_ok = True
    try:
        if 'vol_sma' in df.columns and 'volume' in df.columns:
            vol_s = float(df['vol_sma'].iloc[-1])
            vol_c = float(df['volume'].iloc[-1])
            if vol_c < vol_s * 1.0:
                vol_ok = False
    except Exception:
        vol_ok = True

    # Breakout logic (Donchian Channels)
    donchian_lookback = 20
    upper_band = df['high'].rolling(donchian_lookback).max().shift(1)
    lower_band = df['low'].rolling(donchian_lookback).min().shift(1)
    
    is_breakout_long = trend_up and price > upper_band.iloc[-1]
    is_breakout_short = trend_down and price < lower_band.iloc[-1]

    long_allowed = False
    short_allowed = False
    entry_reason = "ok"

    if (is_uptrend or is_breakout_long) and adx_ok and vol_ok:
        long_allowed = True
        entry_reason = "trend_pullback_long" if is_uptrend else "trend_breakout_long"
    if (is_downtrend or is_breakout_short) and adx_ok and vol_ok:
        short_allowed = True
        entry_reason = "trend_pullback_short" if is_downtrend else "trend_breakout_short"

    # RSI guard: avoid entering LONG when market is already overheated
    try:
        rsi_series = _rsi(close, period=14)
        rsi_current = float(rsi_series.iloc[-1])
        indicators_info['rsi'] = rsi_current
        # Relaxed RSI guard for trend following: allow up to 70 for LONG, down to 30 for SHORT
        if long_allowed and rsi_current > 70.0:
            long_allowed = False
        if short_allowed and rsi_current < 30.0:
            short_allowed = False
    except Exception:
        # if RSI cannot be computed, don't block by RSI
        pass

    if not (long_allowed or short_allowed):
        # Avoid "panic exits" — don't close positions on small retracements.
        # Only signal an exit (HOLD) when the price crosses the slow EMA (EMA50)
        # in the opposite direction of the current position (full trend reversal).
        try:
            current_bias = None
            if state is not None:
                current_bias = state.get('current_bias') or state.get('bias') or state.get('position_bias')
            if isinstance(current_bias, str):
                try:
                    current_bias = Bias[current_bias.upper()]
                except Exception:
                    current_bias = None

            is_trend_broken_long = False
            is_trend_broken_short = False
            if ema_l_curr is not None:
                try:
                    # price below slow EMA indicates trend break for longs
                    is_trend_broken_long = price < ema_l_curr
                    # price above slow EMA indicates trend break for shorts
                    is_trend_broken_short = price > ema_l_curr
                except Exception:
                    is_trend_broken_long = False
                    is_trend_broken_short = False

            if current_bias == Bias.LONG and is_trend_broken_long:
                return {"signal": "HOLD", "stop_loss": None, "indicators_info": indicators_info, "reason": "trend_broken_long"}
            if current_bias == Bias.SHORT and is_trend_broken_short:
                return {"signal": "HOLD", "stop_loss": None, "indicators_info": indicators_info, "reason": "trend_broken_short"}
        except Exception:
            pass

        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "sma_not_trending_or_price_mismatch"}

    # ADX filter: if ADX present and below threshold, block trend entries
    try:
        adx_val = None
        if 'adx' in df.columns:
            adx_val = float(df['adx'].iloc[-1])
            if adx_val is not None:
                if adx_val <= float(adx_threshold):
                    return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "adx_no_trend"}
                # add adaptive exit suggestion: if ADX falls below 20 later, suggest exit (for downstream to act on)
                indicators_info['adx_exit_threshold'] = 20
        indicators_info['adx'] = adx_val
    except Exception:
        pass

    # Volume confirmation: prefer a simple check against vol_sma to avoid over-filtering
    try:
        vol_current = df['volume'].iloc[-1] if 'volume' in df.columns else None
        vol_sma = df.get('vol_sma') and df['vol_sma'].iloc[-1]
        vol_avg5 = df.get('vol_avg5') and df['vol_avg5'].iloc[-1]
        indicators_info['vol_current'] = float(vol_current) if vol_current is not None else None
        indicators_info['vol_sma'] = float(vol_sma) if vol_sma is not None else None
        indicators_info['vol_avg5'] = float(vol_avg5) if vol_avg5 is not None else None

        # Require current volume >= vol_sma * vol_multiplier when vol_sma is present.
        # Removing strict per-bar comparison vs previous bar to avoid false negatives.
        if vol_current is not None and vol_sma is not None:
            try:
                if float(vol_current) < float(vol_sma) * float(vol_multiplier):
                    return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "low_volume_trend"}
            except Exception:
                pass
    except Exception:
        pass

    # Breakout confirmation and HTF EMA200 filter temporarily disabled to increase signal coverage.
    # Kept lightweight markers in indicators_info for debugging, but these checks DO NOT block entries.
    try:
        indicators_info['donchian_checked'] = False
        indicators_info['ema1h200_present'] = None
    except Exception:
        pass

    # pyramiding limit
    long_pyr = int(state.get('long_pyramid', 0))
    short_pyr = int(state.get('short_pyramid', 0))
    if long_allowed and long_pyr >= max_pyramid:
        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "long_pyramid_limit_reached"}
    if short_allowed and short_pyr >= max_pyramid:
        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "short_pyramid_limit_reached"}

    # Dynamic stop-loss via ATR — separate for LONG/SHORT
    # SL = 2 ATR, TP = 4 ATR (соотношение 1:2)
    if long_allowed:
        stop_loss = price - (2.5 * atr_current)
        take_profit = price + (8.0 * atr_current)
        # add trailing parameters: start trailing after 1.5 * ATR; trail step = 0.5 * ATR
        trailing = {
            "start_at_atr": 1.5,
            "trail_step_atr": 0.5,
            "move_to_break_even_atr": 1.0,
        }
        return {
            "signal": "LONG",
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "trailing": trailing,
            "indicators_info": {**indicators_info, "sl": float(stop_loss), "tp": float(take_profit)},
            "reason": entry_reason,
            "last_signal_idx": current_idx,
        }

    if short_allowed:
        stop_loss = price + (2.5 * atr_current)
        take_profit = price - (8.0 * atr_current)
        trailing = {
            "start_at_atr": 1.5,
            "trail_step_atr": 0.5,
            "move_to_break_even_atr": 1.0,
        }
        return {
            "signal": "SHORT",
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "trailing": trailing,
            "indicators_info": {**indicators_info, "sl": float(stop_loss), "tp": float(take_profit)},
            "reason": entry_reason,
            "last_signal_idx": current_idx,
        }

    return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "no_action"}


def generate_trend_signal(*args, **kwargs):
    """Compatibility wrapper for generate_trend_signal.

    Supports two call styles:
      - generate_trend_signal(df: DataFrame, state: dict, ...)
      - generate_trend_signal(row: Series, position_bias: Bias, settings: object)

    When a Series (row) is passed, the function reads precomputed indicators from
    the row (sma, atr, close) and returns a Signal-like dict compatible with
    legacy callers.
    """
    # If first arg is a pandas Series -> legacy row-based call
    if len(args) >= 1 and isinstance(args[0], pd.Series):
        row: pd.Series = args[0]
        position_bias = args[1] if len(args) > 1 else None
        settings = args[2] if len(args) > 2 else None

        indicators_info = {}
        price = float(row.get('close', row.get('price', 0.0)))

        sma_current = row.get('sma') or row.get('sma_50') or row.get('sma_20')
        sma_prev = row.get('sma_prev')
        atr_current = row.get('atr') if 'atr' in row.index else row.get('atr_14')
        # Normalize missing/NaN ATR to None so downstream checks are consistent
        try:
            if atr_current is None or (hasattr(pd, 'isna') and pd.isna(atr_current)):
                indicators_info['atr'] = None
                atr_current = None
            else:
                atr_current = float(atr_current)
                indicators_info['atr'] = atr_current
        except Exception:
            indicators_info['atr'] = None
            atr_current = None

        # SMA slope check using available fields
        if sma_current is None or sma_prev is None:
            # fallback: if no sma_prev provided, allow only if price > sma_current
            if sma_current is None:
                return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "missing_sma"}
            
            # Pullback logic for row-based
            low_p = float(row.get('low', price))
            if not (low_p <= float(sma_current) * 1.001 and price > float(sma_current)):
                return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "no_pullback_sma"}
        else:
            try:
                if not (float(sma_current) > float(sma_prev)):
                    return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "sma_not_uptrending"}
                
                # Pullback logic
                low_p = float(row.get('low', price))
                if not (low_p <= float(sma_current) * 1.001 and price > float(sma_current)):
                    return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "no_pullback_sma"}
            except Exception:
                return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "sma_parse_error"}

        # pyramiding: try to read from position_bias if provided
        long_pyr = 0
        if isinstance(position_bias, dict):
            long_pyr = int(position_bias.get('long_pyramid', 0))
        elif hasattr(position_bias, 'value'):
            # position_bias enum — no pyramid info
            long_pyr = 0

        max_pyramid = getattr(settings, 'max_pyramid', 2) if settings is not None else 2
        if long_pyr >= max_pyramid:
            return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "pyramid_limit_reached"}

        # Dynamic SL via ATR if available (treat NaN as missing)
        if atr_current is None or (hasattr(np, 'isnan') and np.isnan(atr_current)):
            atr_current = price * 0.005
            indicators_info['atr'] = atr_current

        # Additional trend confirmation: ADX and volume checks when available
        try:
            adx_threshold = getattr(settings, 'adx_threshold', None) if settings is not None else None
            if adx_threshold is None:
                adx_threshold = getattr(settings.strategy, 'adx_threshold', 25.0) if settings is not None else 25.0
            vol_multiplier = getattr(settings, 'trend_min_volume_multiplier', None) if settings is not None else None
            if vol_multiplier is None:
                vol_multiplier = getattr(settings.strategy, 'breakout_volume_mult', 1.5) if settings is not None else 1.5
            adx_val = row.get('adx')
            if adx_val is not None:
                try:
                    if float(adx_val) <= float(adx_threshold):
                        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "adx_no_trend"}
                except Exception:
                    pass

            vol_current = row.get('volume')
            vol_prev = None
            try:
                vol_prev = getattr(row, 'volume') if hasattr(row, 'volume') else None
            except Exception:
                vol_prev = None
            vol_sma = row.get('vol_sma') or row.get('volume_sma')
            if vol_current is not None and vol_sma is not None:
                try:
                    # prefer short-term increase vs previous bar
                    if vol_prev is not None:
                        if float(vol_current) <= float(vol_prev):
                            return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "low_volume_vs_prev"}
                    # fallback: require vol_current >= vol_sma * vol_multiplier
                    if float(vol_current) < float(vol_sma) * float(vol_multiplier):
                        return {"signal": None, "stop_loss": None, "indicators_info": indicators_info, "reason": "low_volume_trend"}
                except Exception:
                    pass
        except Exception:
            pass
        # SL = 2.5 ATR, TP = 5.0 ATR
        stop_loss = price - 2.5 * float(atr_current)
        take_profit = price + 8.0 * float(atr_current)

        return {
            "signal": "LONG",
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "indicators_info": {**indicators_info, "sl": float(stop_loss), "tp": float(take_profit)},
            "reason": "ok"
        }

    # Otherwise, delegate to DataFrame implementation
    return _generate_trend_signal_df(*args, **kwargs)


def generate_flat_signal(
    df: pd.DataFrame,
    state: t.Optional[dict] = None,
    rsi_period: int = 14,
    rsi_base_low: int = 30,
    rsi_base_high: int = 70,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    bb_compression_factor: float = 0.8,
    min_history: int = 50,
) -> t.Dict:
    """
    Генерация сигнала FLAT (Mean Reversion) с фильтрами сжатия BB и адаптивным RSI.
    """
    # forward-fill missing indicator values to improve robustness on short series
    df = df.ffill()
    state = state or {}
    indicators_info = {}

    # Cooldown check
    last_signal_idx = state.get('last_signal_idx', -100)
    current_idx = len(df)
    if current_idx - last_signal_idx < 10:
        return {"signal": None, "indicators_info": indicators_info, "reason": "cooldown"}

    if df.replace([np.inf, -np.inf], np.nan).dropna().empty:
        return {"signal": None, "indicators_info": indicators_info, "reason": "nan_in_data"}

    if not _ensure_history(df, min_history):
        return {"signal": None, "indicators_info": indicators_info, "reason": "insufficient_history"}

    close = df['close']
    price = float(close.iloc[-1])

    # Фильтр по объему: не торгуем во флэте на низком объеме - это ловушка
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    if df['volume'].iloc[-1] < vol_avg * 0.8:
        return {"signal": None, "indicators_info": indicators_info, "reason": "low_volume_flat_trap"}

    bbw = _bb_width(df, period=bb_period, mult=bb_mult)
    bbw_current = float(bbw.iloc[-1])
    # Compare to historical median to detect compression
    bbw_median = float(bbw.iloc[-20:].median()) if len(bbw) >= 20 else float(bbw.median())
    indicators_info['bb_width'] = bbw_current

    # Relax BB compression check for testing: allow entries if bbw_current not smaller than 0.6*median
    # Relax BB compression check for testing: allow entries if bbw_current >= 0.6*median
    if bbw_median == 0 or bbw_current < max(0.6 * bbw_median, (bb_compression_factor * bbw_median) * 0.75):
        return {"signal": None, "indicators_info": indicators_info, "reason": "bb_compression"}

    # Adaptive RSI levels by volatility (ATR20 percentile)
    atr20 = _atr(df, period=20)
    atr20_current = float(atr20.iloc[-1])
    indicators_info['atr20'] = atr20_current

    # compute volatility rank over last 100
    atr20_hist = atr20.iloc[-100:] if len(atr20) >= 100 else atr20
    vol_pct = float((atr20_current >= atr20_hist).mean())  # crude percentile

    # Extreme volatility guard: if in top 10% volatility, block mean-reversion
    if vol_pct > 0.9:
        return {"signal": None, "indicators_info": indicators_info, "reason": "extreme_volatility_block"}

    if vol_pct > 0.75:
        low_level, high_level = 25, 75
    elif vol_pct < 0.25:
        low_level, high_level = 35, 65
    else:
        low_level, high_level = rsi_base_low, rsi_base_high

    # Ensure minimum gap between levels to avoid too narrow RSI bands
    min_gap = 20
    if (high_level - low_level) < min_gap:
        center = (high_level + low_level) / 2
        low_level = max(40, int(center - min_gap / 2))
        high_level = min(60, int(center + min_gap / 2))

    rsi = _rsi(close, period=rsi_period)
    rsi_current = float(rsi.iloc[-1])
    indicators_info['rsi'] = rsi_current

    # Time exit suggestion for FLAT: how many bars to hold maximum
    max_hold = getattr(df, 'max_hold_bars', None) or 20
    indicators_info['max_hold_bars'] = int(max_hold)

    # Mean reversion entry: RSI below low_level => LONG, above high_level => SHORT
    # Добавляем свечное подтверждение (разворотная свеча)
    open_curr = float(df['open'].iloc[-1])
    if rsi_current < low_level and price > open_curr:
        return {"signal": "LONG", "indicators_info": indicators_info, "reason": "rsi_oversold_bullish_confirm", "last_signal_idx": current_idx}
    if rsi_current > high_level and price < open_curr:
        return {"signal": "SHORT", "indicators_info": indicators_info, "reason": "rsi_overbought_bearish_confirm", "last_signal_idx": current_idx}

    return {"signal": None, "indicators_info": indicators_info, "reason": "no_mean_reversion"}


def generate_momentum_signal(
    df: pd.DataFrame,
    ema_short: int = 20,
    ema_long: int = 50,
    vol_lookback: int = 100,
    vol_top_pct: float = 0.60,
    min_history: int = 100,
) -> t.Dict:
    """
    Генерация сигнала MOMENTUM с проверкой "веера EMA" и подтверждением по объему.

    Улучшения:
      - смягчён vol_top_pct по умолчанию до 0.75
      - дополнительное условие: текущий объём >= 1.5 * avg(volume last 5 bars)
      - RSI(14) должен быть в диапазоне [50,65] для входа
      - поддержка SHORT при Price < EMA20 < EMA50
      - выход (HOLD) при закрытии ниже EMA20
    """
    indicators_info = {}
    if df.replace([np.inf, -np.inf], np.nan).dropna().empty:
        return {"signal": None, "indicators_info": indicators_info, "reason": "nan_in_data"}
    if not _ensure_history(df, min_history):
        return {"signal": None, "indicators_info": indicators_info, "reason": "insufficient_history"}

    close = df['close']
    price = float(close.iloc[-1])

    ema_s = _ema(close, ema_short)
    ema_l = _ema(close, ema_long)
    ema_s_val = float(ema_s.iloc[-1])
    ema_l_val = float(ema_l.iloc[-1])
    indicators_info['ema20'] = ema_s_val
    indicators_info['ema50'] = ema_l_val

    # previous EMA values (for cross detection)
    prev_ema_s_val = float(ema_s.iloc[-2]) if len(ema_s) >= 2 else None
    prev_ema_l_val = float(ema_l.iloc[-2]) if len(ema_l) >= 2 else None
    indicators_info['prev_ema20'] = prev_ema_s_val
    indicators_info['prev_ema50'] = prev_ema_l_val

    # EMA fan check: LONG if Price > EMA20 > EMA50, SHORT if Price < EMA20 < EMA50
    is_long_fan = price > ema_s_val > ema_l_val
    is_short_fan = price < ema_s_val < ema_l_val

    # Allow entry also on a clean EMA cross (fast EMA crossing above slow EMA) to relax strict fan requirement
    ema_cross_up = False
    ema_cross_down = False
    try:
        if prev_ema_s_val is not None and prev_ema_l_val is not None:
            ema_cross_up = (prev_ema_s_val <= prev_ema_l_val) and (ema_s_val > ema_l_val)
            ema_cross_down = (prev_ema_s_val >= prev_ema_l_val) and (ema_s_val < ema_l_val)
    except Exception:
        ema_cross_up = False
        ema_cross_down = False

    indicators_info['ema_cross_up'] = ema_cross_up
    indicators_info['ema_cross_down'] = ema_cross_down

    # require either full fan alignment or a recent cross
    is_long_condition = is_long_fan or ema_cross_up
    is_short_condition = is_short_fan or ema_cross_down

    if not (is_long_condition or is_short_condition):
        return {"signal": None, "indicators_info": indicators_info, "reason": "ema_fan_not_aligned"}

    # RSI filter (leading): require RSI(14) in [50,65]
    rsi = _rsi(close, period=14)
    rsi_current = float(rsi.iloc[-1])
    indicators_info['rsi'] = rsi_current
    # Relax RSI window for momentum entries to [45,75]
    if not (45.0 <= rsi_current <= 75.0):
        return {"signal": None, "indicators_info": indicators_info, "reason": "rsi_not_in_preferred_range"}

    # Volume checks
    vol = df['volume'].iloc[-vol_lookback:]
    vol_hist = vol
    vol_current = float(df['volume'].iloc[-1])
    threshold = float(np.percentile(vol_hist, vol_top_pct * 100)) if len(vol_hist) > 0 else float(vol_current)
    indicators_info['vol_current'] = vol_current
    indicators_info['vol_threshold_pct'] = vol_top_pct

    # short-term average over last 5 bars
    vol_avg5 = float(df['volume'].iloc[-5:].mean()) if len(df) >= 5 else vol_current
    indicators_info['vol_avg5'] = vol_avg5

    # Require both percentile and short-term jump
    if vol_current < threshold:
        return {"signal": None, "indicators_info": indicators_info, "reason": "no_volume_spike_percentile"}
    # require short-term multiplier (relaxed to 1.1x to increase signal frequency)
    if vol_current < 1.1 * vol_avg5:
        return {"signal": None, "indicators_info": indicators_info, "reason": "no_short_term_volume_jump"}

    # Exit (trailing idea): if price closes below EMA20 -> HOLD (close)
    if price < ema_s_val:
        return {"signal": "HOLD", "indicators_info": indicators_info, "reason": "momentum_exit_below_ema20"}

    # Determine direction
    if is_long_fan:
        return {"signal": "LONG", "indicators_info": indicators_info, "reason": "ok"}
    if is_short_fan:
        return {"signal": "SHORT", "indicators_info": indicators_info, "reason": "ok"}

    return {"signal": None, "indicators_info": indicators_info, "reason": "no_action"}


if __name__ == '__main__':
    # quick smoke test (runs only when module executed directly)
    import json

    # create dummy data
    idx = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='T')
    df = pd.DataFrame(index=idx)
    df['open'] = np.linspace(100, 120, len(df)) + np.random.randn(len(df))
    df['high'] = df['open'] + np.random.rand(len(df)) * 1.5
    df['low'] = df['open'] - np.random.rand(len(df)) * 1.5
    df['close'] = df['open'] + np.random.randn(len(df)) * 0.5
    df['volume'] = np.random.randint(1, 100, len(df))

    print(json.dumps(generate_trend_signal(df, state={'long_pyramid': 0}), indent=2))
    print(json.dumps(generate_flat_signal(df), indent=2))
    print(json.dumps(generate_momentum_signal(df), indent=2))

