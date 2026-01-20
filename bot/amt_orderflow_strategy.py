from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from bot.exchange.bybit_client import BybitClient
from bot.strategy import Signal, Action


@dataclass
class AbsorptionConfig:
    """Параметры детекции сквиза на поглощении (упрощённая версия по order flow)."""

    lookback_seconds: int = 60            # окно анализа по тикам
    min_total_volume: float = 10_000.0    # минимум суммарного объёма в окне
    min_buy_sell_ratio: float = 2.0       # сколько раз покупок больше продаж для bear-squeeze
    max_price_drift_pct: float = 0.05     # максимум движения цены в % при сильном CVD (стоим / топчемся)
    min_cvd_delta: float = 5_000.0        # минимальный прирост CVD за окно


@dataclass
class VolumeProfileConfig:
    """
    Конфиг для расчёта Volume Profile / POC / VAH / VAL.

    Профиль считается по OHLCV‑свечам (например, 15m) с биннингом цены.
    """

    price_step: float                     # размер ценового шага (бин), например 10 для BTCUSDT, 1 для ETHUSDT, 0.05 для SOLUSDT
    value_area_pct: float = 0.7           # доля объёма в зоне стоимости (обычно 70%)
    session_start_utc: Optional[int] = None  # час начала сессии по UTC (если None — используем все бары)
    session_end_utc: Optional[int] = None    # час конца сессии по UTC (исключительно)


def _filter_session(df: pd.DataFrame, cfg: VolumeProfileConfig) -> pd.DataFrame:
    """
    Фильтрует DataFrame по торговой сессии, если заданы session_start_utc / session_end_utc.
    Ожидается, что индекс df — DatetimeIndex в UTC.
    """
    if cfg.session_start_utc is None or cfg.session_end_utc is None:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    hours = df.index.hour
    if cfg.session_start_utc < cfg.session_end_utc:
        mask = (hours >= cfg.session_start_utc) & (hours < cfg.session_end_utc)
    else:
        # Сессия через полночь, например 22–06
        mask = (hours >= cfg.session_start_utc) | (hours < cfg.session_end_utc)
    return df[mask]


def build_volume_profile_from_ohlcv(
    df: pd.DataFrame,
    config: VolumeProfileConfig,
) -> Optional[Dict[str, Any]]:
    """
    Строит volume profile по OHLCV‑свечам и возвращает:
      - poc: цена POC
      - vah: верхняя граница value area (≈70% объёма)
      - val: нижняя граница value area
      - prices: массив цен уровней
      - volumes: массив объёмов по уровням
      - total_volume: суммарный объём в профиле
    """
    if df is None or df.empty:
        return None

    step = float(config.price_step)
    if step <= 0:
        raise ValueError("price_step must be positive for VolumeProfileConfig")

    df_use = _filter_session(df, config)
    if df_use.empty:
        return None

    bins: Dict[int, float] = {}

    # 1. Распределяем объём каждой свечи по ценовым уровням (бины)
    for row in df_use.itertuples():
        high = float(getattr(row, "high", np.nan))
        low = float(getattr(row, "low", np.nan))
        vol = float(getattr(row, "volume", getattr(row, "Volume", np.nan)))

        if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(vol):
            continue
        if vol <= 0:
            continue

        # Гарантируем, что low <= high
        if low > high:
            low, high = high, low

        start_bin = int(np.floor(low / step))
        end_bin = int(np.ceil(high / step))
        if end_bin < start_bin:
            end_bin = start_bin

        bin_count = max(end_bin - start_bin + 1, 1)
        vol_per_bin = vol / bin_count

        for b in range(start_bin, end_bin + 1):
            bins[b] = bins.get(b, 0.0) + vol_per_bin

    if not bins:
        return None

    # Сортируем бины по цене
    sorted_bins = sorted(bins.items(), key=lambda x: x[0])
    bin_indices = [b for b, _ in sorted_bins]
    volumes = np.array([v for _, v in sorted_bins], dtype=float)
    prices = np.array([b * step for b in bin_indices], dtype=float)

    total_volume = float(volumes.sum())
    if total_volume <= 0:
        return None

    # 2. POC — уровень с максимальным объёмом
    poc_idx = int(np.argmax(volumes))
    poc_price = float(prices[poc_idx])

    # 3. Value Area (VAH/VAL) методом расширения вокруг POC
    target_volume = total_volume * float(config.value_area_pct)
    current_volume = float(volumes[poc_idx])
    lower_idx = poc_idx
    upper_idx = poc_idx

    n = len(volumes)

    def _sum_range(i_start: int, i_end: int) -> float:
        """Сумма объёма по индексам [i_start, i_end] с учётом границ массива."""
        i_start = max(i_start, 0)
        i_end = min(i_end, n - 1)
        if i_end < i_start:
            return 0.0
        return float(volumes[i_start : i_end + 1].sum())

    while current_volume < target_volume and (lower_idx > 0 or upper_idx < n - 1):
        # Две "строки" сверху и снизу
        vol_up = _sum_range(upper_idx + 1, upper_idx + 2) if upper_idx < n - 1 else 0.0
        vol_down = _sum_range(lower_idx - 2, lower_idx - 1) if lower_idx > 0 else 0.0

        if vol_up <= 0 and vol_down <= 0:
            break

        if vol_up >= vol_down:
            current_volume += vol_up
            upper_idx = min(upper_idx + 2, n - 1)
        else:
            current_volume += vol_down
            lower_idx = max(lower_idx - 2, 0)

    val_price = float(prices[lower_idx])
    vah_price = float(prices[upper_idx])

    return {
        "poc": poc_price,
        "vah": vah_price,
        "val": val_price,
        "prices": prices,
        "volumes": volumes,
        "total_volume": total_volume,
        "price_step": step,
    }


def _compute_cvd_metrics(
    trades_df: pd.DataFrame,
    lookback_seconds: int,
) -> Optional[Dict[str, float]]:
    """
    Считает CVD и Delta Velocity на последнем окне тиков.

    Возвращает:
      - cvd_now: текущее значение CVD
      - delta_velocity: суммарное изменение CVD за окно
      - avg_abs_delta: среднее |ΔCVD| (для сравнения с delta_velocity)
    """
    if trades_df.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - pd.Timedelta(seconds=lookback_seconds)
    window_df = trades_df[trades_df["time"] >= window_start]
    if window_df.empty:
        return None

    side_upper = window_df["side"].str.upper()
    window_df = window_df.copy()
    window_df["signed_vol"] = np.where(side_upper == "BUY", window_df["qty"], -window_df["qty"])
    window_df["cvd"] = window_df["signed_vol"].cumsum()

    cvd_series = window_df["cvd"]
    cvd_now = float(cvd_series.iloc[-1])

    cvd_diff = cvd_series.diff().fillna(0.0)
    if cvd_diff.empty:
        return None

    avg_abs_delta = float(cvd_diff.abs().mean())

    # Используем последние 20% точек как "последний импульс"
    tail_len = max(1, int(len(cvd_diff) * 0.2))
    delta_velocity = float(cvd_diff.iloc[-tail_len:].sum())

    return {
        "cvd_now": cvd_now,
        "delta_velocity": delta_velocity,
        "avg_abs_delta": avg_abs_delta,
    }


def _parse_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует список тиков Bybit в DataFrame с колонками:
    time (datetime UTC), price, qty, side ('Buy'/'Sell').
    """
    if not raw_trades:
        return pd.DataFrame(columns=["time", "price", "qty", "side"])

    rows = []
    for t in raw_trades:
        try:
            # unified trading формат: time, side, qty/execQty, price/execPrice
            ts = int(t.get("time", t.get("T", 0)))
            price_str = t.get("price") or t.get("execPrice") or t.get("p")
            qty_str = t.get("execQty") or t.get("qty") or t.get("q")
            side = t.get("side") or t.get("S") or ""

            if not price_str or not qty_str or not side:
                continue

            price = float(price_str)
            qty = float(qty_str)
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)

            rows.append({"time": dt, "price": price, "qty": qty, "side": side})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["time", "price", "qty", "side"])

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


@dataclass
class LhOrderflowConfig:
    """
    Настройки orderflow‑варианта Liquidation Hunter (через CVD + Volume Profile).
    """
    cvd_lookback_sec: int = 60
    cvd_spike_mult: float = 2.0
    max_price_drift_pct: float = 0.2
    min_rvol: float = 1.5
    rr_to_poc_min: float = 1.5


def generate_lh_orderflow_signals(
    client: BybitClient,
    symbol: str,
    df_ohlcv: pd.DataFrame,
    vp_config: VolumeProfileConfig,
    cfg: Optional[LhOrderflowConfig] = None,
) -> List[Signal]:
    """
    Orderflow‑вариант Liquidation Hunter:
    - ищем выбросы CVD за VAL/VAH,
    - вход по возврату внутрь VA,
    - TP предполагается по POC (Poc включаем в reason).
    """
    cfg = cfg or LhOrderflowConfig()
    signals: List[Signal] = []
    if df_ohlcv is None or df_ohlcv.empty:
        return signals

    # --- Volume Profile для VAL/VAH/POC ---
    df_vp = df_ohlcv.copy()
    if "timestamp" in df_vp.columns:
        df_vp["timestamp"] = pd.to_datetime(df_vp["timestamp"], unit="ms", utc=True)
        df_vp = df_vp.set_index("timestamp")

    vp = build_volume_profile_from_ohlcv(df_vp, vp_config)
    if not vp:
        return signals

    poc = float(vp["poc"])
    vah = float(vp["vah"])
    val = float(vp["val"])

    last = df_ohlcv.iloc[-1]
    prev = df_ohlcv.iloc[-2] if len(df_ohlcv) >= 2 else last
    close = float(last["close"])
    volume = float(last["volume"])
    vol_sma_series = df_ohlcv["volume"].rolling(window=20, min_periods=1).mean()
    vol_sma = float(vol_sma_series.iloc[-2]) if len(vol_sma_series) >= 2 else float(vol_sma_series.iloc[-1])

    rvol = volume / vol_sma if vol_sma > 0 else 0.0
    if rvol < cfg.min_rvol:
        return signals

    # --- CVD / Delta Velocity ---
    trades = client.get_recent_trades(symbol, limit=400)
    trades_df = _parse_trades(trades)
    if trades_df.empty:
        return signals

    cvd_metrics = _compute_cvd_metrics(trades_df, lookback_seconds=cfg.cvd_lookback_sec)
    if not cvd_metrics:
        return signals

    delta_velocity = cvd_metrics["delta_velocity"]
    avg_abs_delta = cvd_metrics["avg_abs_delta"]
    if avg_abs_delta <= 0:
        return signals

    cvd_spike_thr = avg_abs_delta * cfg.cvd_spike_mult

    # --- LONG: выброс за VAL и возврат внутрь VA при затухании продаж ---
    long_setup = (
        prev["low"] < val
        and close > val
        and delta_velocity < 0
        and abs(delta_velocity) <= cvd_spike_thr
        and abs(close - float(prev["close"])) / float(prev["close"]) * 100.0 <= cfg.max_price_drift_pct
    )

    if long_setup:
        entry = close
        tp = poc
        sl = min(float(prev["low"]), val * 0.998)
        rr = (tp - entry) / (entry - sl) if entry > sl else 0.0
        if rr >= cfg.rr_to_poc_min:
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"lh_of_long_cvd_absorption_val_{val:.2f}"
                f"_rvol_{rvol:.2f}_dv_{int(delta_velocity)}"
                f"_poc_{poc:.2f}"
            )
            signals.append(Signal(timestamp=ts, action=Action.LONG, reason=reason, price=entry))

    # --- SHORT: выброс за VAH и возврат внутрь VA при затухании покупок ---
    short_setup = (
        prev["high"] > vah
        and close < vah
        and delta_velocity > 0
        and abs(delta_velocity) <= cvd_spike_thr
        and abs(close - float(prev["close"])) / float(prev["close"]) * 100.0 <= cfg.max_price_drift_pct
    )

    if short_setup:
        entry = close
        tp = poc
        sl = max(float(prev["high"]), vah * 1.002)
        rr = (entry - tp) / (sl - entry) if sl > entry else 0.0
        if rr >= cfg.rr_to_poc_min:
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"lh_of_short_cvd_absorption_vah_{vah:.2f}"
                f"_rvol_{rvol:.2f}_dv_{int(delta_velocity)}"
                f"_poc_{poc:.2f}"
            )
            signals.append(Signal(timestamp=ts, action=Action.SHORT, reason=reason, price=entry))

    return signals


def detect_absorption_squeeze_short(
    client: BybitClient,
    symbol: str,
    current_price: float,
    config: Optional[AbsorptionConfig] = None,
) -> Optional[Signal]:
    """
    Детекция сценария B: Сквиз на поглощении (Absorption Squeeze) в сторону SHORT.
    """
    cfg = config or AbsorptionConfig()

    raw_trades = client.get_recent_trades(symbol, limit=500)
    df_trades = _parse_trades(raw_trades)
    if df_trades.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - pd.Timedelta(seconds=cfg.lookback_seconds)
    window_df = df_trades[df_trades["time"] >= window_start]

    if window_df.empty:
        return None

    buy_mask = window_df["side"].str.upper() == "BUY"
    sell_mask = window_df["side"].str.upper() == "SELL"

    buy_volume = float(window_df.loc[buy_mask, "qty"].sum())
    sell_volume = float(window_df.loc[sell_mask, "qty"].sum())
    total_volume = buy_volume + sell_volume

    if total_volume < cfg.min_total_volume:
        return None

    cvd_delta = buy_volume - sell_volume  # положительный → агрессивные покупки
    buy_sell_ratio = (buy_volume / max(sell_volume, 1e-6)) if sell_volume > 0 else np.inf

    first_price = float(window_df["price"].iloc[0])
    last_price = float(window_df["price"].iloc[-1])
    price_change_pct = (last_price - first_price) / first_price * 100.0

    # Условия "абсорбции" для SHORT:
    if cvd_delta < cfg.min_cvd_delta:
        return None

    if buy_sell_ratio < cfg.min_buy_sell_ratio:
        return None

    if abs(price_change_pct) > cfg.max_price_drift_pct:
        return None

    ts = pd.Timestamp(now_utc)
    reason = (
        f"amt_of_absorption_short_cvd_{int(cvd_delta)}"
        f"_buy_vs_sell_{int(buy_volume)}/{int(sell_volume)}"
        f"_price_chg_{price_change_pct:.2f}%"
    )

    return Signal(
        timestamp=ts,
        action=Action.SHORT,
        reason=reason,
        price=current_price,
    )


def detect_absorption_squeeze_long(
    client: BybitClient,
    symbol: str,
    current_price: float,
    config: Optional[AbsorptionConfig] = None,
) -> Optional[Signal]:
    """
    Зеркальный сценарий B: Сквиз на поглощении (Absorption Squeeze) в сторону LONG.
    """
    cfg = config or AbsorptionConfig()

    raw_trades = client.get_recent_trades(symbol, limit=500)
    df_trades = _parse_trades(raw_trades)
    if df_trades.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - pd.Timedelta(seconds=cfg.lookback_seconds)
    window_df = df_trades[df_trades["time"] >= window_start]

    if window_df.empty:
        return None

    buy_mask = window_df["side"].str.upper() == "BUY"
    sell_mask = window_df["side"].str.upper() == "SELL"

    buy_volume = float(window_df.loc[buy_mask, "qty"].sum())
    sell_volume = float(window_df.loc[sell_mask, "qty"].sum())
    total_volume = buy_volume + sell_volume

    if total_volume < cfg.min_total_volume:
        return None

    cvd_delta = buy_volume - sell_volume  # отрицательный → агрессивные продажи
    sell_buy_ratio = (sell_volume / max(buy_volume, 1e-6)) if buy_volume > 0 else np.inf

    first_price = float(window_df["price"].iloc[0])
    last_price = float(window_df["price"].iloc[-1])
    price_change_pct = (last_price - first_price) / first_price * 100.0

    # Условия "абсорбции" для LONG:
    if -cvd_delta < cfg.min_cvd_delta:
        return None

    if sell_buy_ratio < cfg.min_buy_sell_ratio:
        return None

    if abs(price_change_pct) > cfg.max_price_drift_pct:
        return None

    ts = pd.Timestamp(now_utc)
    reason = (
        f"amt_of_absorption_long_cvd_{int(cvd_delta)}"
        f"_sell_vs_buy_{int(sell_volume)}/{int(buy_volume)}"
        f"_price_chg_{price_change_pct:.2f}%"
    )

    return Signal(
        timestamp=ts,
        action=Action.LONG,
        reason=reason,
        price=current_price,
    )


def generate_amt_signals(
    client: BybitClient,
    symbol: str,
    current_price: float,
    df_ohlcv: pd.DataFrame,
    vp_config: VolumeProfileConfig,
    abs_config: AbsorptionConfig,
    delta_aggr_mult: float = 2.5,
) -> List[Signal]:
    """
    Генерирует AMT‑сигналы на основе:
      - Volume Profile (POC / VAH / VAL)
      - Delta Velocity (наклон CVD за последнее окно)
      - сценариев Breakout / Squeeze
      - Absorption Squeeze (detect_absorption_squeeze_*)
    """
    signals: List[Signal] = []

    # --- Volume Profile ---
    vp = None
    try:
        vp = build_volume_profile_from_ohlcv(df_ohlcv, vp_config)
    except Exception:
        vp = None

    vah = val = poc = None
    if vp:
        poc = vp["poc"]
        vah = vp["vah"]
        val = vp["val"]

    # --- Order flow / CVD ---
    trades = client.get_recent_trades(symbol, limit=500)
    trades_df = _parse_trades(trades)

    cvd_metrics = _compute_cvd_metrics(trades_df, lookback_seconds=abs_config.lookback_seconds)
    delta_velocity = None
    avg_abs_delta = None
    if cvd_metrics:
        delta_velocity = cvd_metrics["delta_velocity"]
        avg_abs_delta = cvd_metrics["avg_abs_delta"]

    # --- Breakout / Squeeze сценарии ---
    if vp and cvd_metrics and avg_abs_delta and avg_abs_delta > 0:
        threshold = avg_abs_delta * max(delta_aggr_mult, 1.0)
        last_row = df_ohlcv.iloc[-1]
        prev_row = df_ohlcv.iloc[-2] if len(df_ohlcv) >= 2 else None

        # TRUE BREAKOUT LONG: цена устойчиво выше VAH + сильный положительный наклон CVD
        if vah is not None and current_price > vah and delta_velocity is not None and delta_velocity > threshold:
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"amt_of_breakout_long_price_{current_price:.2f}_above_vah_{vah:.2f}"
                f"_dv_{int(delta_velocity)}_thr_{int(threshold)}"
            )
            signals.append(Signal(timestamp=ts, action=Action.LONG, reason=reason, price=current_price))

        # TRUE BREAKOUT SHORT: цена устойчиво ниже VAL + сильный отрицательный наклон CVD
        if val is not None and current_price < val and delta_velocity is not None and delta_velocity < -threshold:
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"amt_of_breakout_short_price_{current_price:.2f}_below_val_{val:.2f}"
                f"_dv_{int(delta_velocity)}_thr_{int(threshold)}"
            )
            signals.append(Signal(timestamp=ts, action=Action.SHORT, reason=reason, price=current_price))

        # SQUEEZE SHORT: ложный пробой выше VAH и возврат внутрь при отрицательном наклоне CVD
        if (
            vah is not None
            and prev_row is not None
            and prev_row.high > vah
            and last_row.close < vah
            and delta_velocity is not None
            and delta_velocity < 0
        ):
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"amt_of_squeeze_short_false_break_vah_{vah:.2f}"
                f"_price_{current_price:.2f}_dv_{int(delta_velocity)}"
            )
            signals.append(Signal(timestamp=ts, action=Action.SHORT, reason=reason, price=current_price))

        # SQUEEZE LONG: ложный пробой ниже VAL и возврат внутрь при положительном наклоне CVD
        if (
            val is not None
            and prev_row is not None
            and prev_row.low < val
            and last_row.close > val
            and delta_velocity is not None
            and delta_velocity > 0
        ):
            ts = pd.Timestamp(datetime.now(timezone.utc))
            reason = (
                f"amt_of_squeeze_long_false_break_val_{val:.2f}"
                f"_price_{current_price:.2f}_dv_{int(delta_velocity)}"
            )
            signals.append(Signal(timestamp=ts, action=Action.LONG, reason=reason, price=current_price))

    # --- Absorption Squeeze (существующие сценарии) ---
    try:
        short_abs = detect_absorption_squeeze_short(client, symbol, current_price, abs_config)
        if short_abs:
            signals.append(short_abs)
        long_abs = detect_absorption_squeeze_long(client, symbol, current_price, abs_config)
        if long_abs:
            signals.append(long_abs)
    except Exception:
        pass

    return signals


__all__ = [
    "AbsorptionConfig",
    "VolumeProfileConfig",
    "build_volume_profile_from_ohlcv",
    "detect_absorption_squeeze_short",
    "detect_absorption_squeeze_long",
    "generate_amt_signals",
    "LhOrderflowConfig",
    "generate_lh_orderflow_signals",
]

