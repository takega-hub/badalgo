"""zscore_strategy_v2.py

Векторизированная реализация стратегии Z-Score (Mean Reversion).
Цель: высокая производительность, типизация, логирование причин пропуска сигналов,
и вынос всех магических чисел в класс параметров для последующей оптимизации.

Файл возвращает функцию generate_signals(df, params) -> pd.DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    window: int = 20  # период для SMA и StdDev
    z_long: float = -2.0  # УЖЕСТОЧЕНО: было -2.5, теперь -2.0 для более сильных сигналов
    z_short: float = 2.0  # УЖЕСТОЧЕНО: было 2.5, теперь 2.0 для более сильных сигналов
    z_exit: float = 0.5
    vol_factor: float = 0.85  # УЖЕСТОЧЕНО: было 0.8, теперь 0.85 для большего объема
    adx_threshold: float = 20.0  # УЖЕСТОЧЕНО: было 25.0, теперь 20.0 для фильтрации сильных трендов
    epsilon: float = 1e-9
    rsi_enabled: bool = True
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    sma_slope_threshold: float = 0.0008  # УЖЕСТОЧЕНО: было 0.001, теперь 0.0008 для более плоского рынка
    atr_window: int = 14
    adx_window: int = 14
    stop_loss_atr: float = 1.0  # ОПТИМИЗИРОВАНО: было 3.0, теперь 1.0 для меньших потерь
    take_profit_atr: float = 2.0  # ОПТИМИЗИРОВАНО: было 2.0, остается 2.0 для лучшего соотношения TP/SL = 2.0
    # Новые параметры для дополнительных фильтров
    min_volatility: float = 0.001  # Минимальная относительная волатильность (ATR/Close) = 0.1%
    exclude_hours: list = field(default_factory=lambda: [0, 1, 2])  # Часы для исключения (по умолчанию [0, 1, 2] - ночные часы)
    use_dynamic_sl_tp: bool = True  # Использовать динамические SL/TP на основе Z-Score
    require_confirmation: bool = True  # Требовать подтверждение сигнала на следующей свече


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Wilder's smoothing: use EMA with alpha = 1/window (approximation)
    return _ema(tr.fillna(0), span=window)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    # Directional Movement
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_series = _ema(tr.fillna(0), span=window)

    plus_di = 100 * _ema(plus_dm, span=window) / (atr_series.replace(0, np.nan))
    minus_di = 100 * _ema(minus_dm, span=window) / (atr_series.replace(0, np.nan))

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx_series = _ema(dx.fillna(0), span=window)
    return adx_series.fillna(0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def generate_signals(df: pd.DataFrame, params: Optional[StrategyParams] = None) -> pd.DataFrame:
    """Generate vectorized signals for Z-Score mean reversion strategy.

    Input DataFrame must contain columns: ['open','high','low','close','volume'].

    Returns DataFrame with added columns: ['sma','std','z','atr','adx','rsi','signal','reason']
    """
    if params is None:
        params = StrategyParams()

    # Validate input
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        # return empty df with expected columns
        cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        out = pd.DataFrame(columns=cols + ["sma", "std", "z", "atr", "adx", "rsi", "signal", "reason"])
        return out

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Copy to avoid changing original
    data = df.copy()

    close = data["close"].astype(float)

    # SMA and STD
    data["sma"] = close.rolling(window=params.window, min_periods=1).mean()
    data["std"] = close.rolling(window=params.window, min_periods=1).std(ddof=0).fillna(0.0)

    # Z-Score with epsilon protection
    data["z"] = (close - data["sma"]) / (data["std"] + params.epsilon)

    # ATR and ADX for volatility and market state
    data["atr"] = atr(data["high"], data["low"], data["close"], params.atr_window)
    data["adx"] = adx(data["high"], data["low"], data["close"], params.adx_window)

    # RSI (optional)
    data["rsi"] = rsi(close)

    # Volume filter: compare to rolling mean volume
    # Исправлено: для mean reversion нужен нормальный объем, но не обязательно высокий
    # vol_factor = 0.8 означает, что объем должен быть >= 80% от среднего (не слишком низкий)
    # Если vol_factor = 0, отключаем фильтр (всегда True)
    data["vol_mean"] = data["volume"].rolling(window=params.window, min_periods=1).mean()
    if params.vol_factor <= 0:
        data["vol_ok"] = True
    else:
        data["vol_ok"] = data["volume"] >= (data["vol_mean"] * params.vol_factor)

    # SMA slope filter to detect flat market
    sma_diff = data["sma"].diff().abs() / (data["sma"].shift(1).replace(0, np.nan).abs() + params.epsilon)
    data["sma_flat"] = sma_diff.fillna(0) < params.sma_slope_threshold

    # Dynamic thresholds scaled by current volatility (atr relative to rolling mean)
    atr_mean = data["atr"].rolling(window=params.window, min_periods=1).mean().replace(0, np.nan)
    vol_scale = (data["atr"] / (atr_mean + params.epsilon)).fillna(1.0)
    data["z_long_thr"] = params.z_long * vol_scale
    data["z_short_thr"] = params.z_short * vol_scale

    # Prepare output columns
    data["signal"] = ""
    data["reason"] = ""

    # Determine market state: if ADX too high, pause mean reversion
    data["market_allowed"] = data["adx"] < params.adx_threshold

    # Фильтр по волатильности (исключить слишком спокойные рынки)
    data["atr_relative"] = data["atr"] / data["close"]
    # Если min_volatility = 0, отключаем фильтр (всегда True)
    if params.min_volatility <= 0:
        data["volatility_ok"] = True
    else:
        data["volatility_ok"] = data["atr_relative"] > params.min_volatility

    # Фильтр по времени (избегать открытия перед важными новостями/ночью)
    # Пытаемся извлечь час из индекса
    try:
        if isinstance(data.index, pd.DatetimeIndex):
            data["hour"] = data.index.hour
        else:
            # Если индекс не DatetimeIndex, пытаемся преобразовать
            data["hour"] = pd.to_datetime(data.index).hour
    except:
        # Если не удалось извлечь час, считаем что время подходит (не фильтруем)
        data["hour"] = pd.Series([12] * len(data), index=data.index)  # Устанавливаем полдень (не в exclude_hours)
    data["time_ok"] = ~data["hour"].isin(params.exclude_hours)

    # Entry conditions с новыми фильтрами
    long_entry = (data["z"] <= data["z_long_thr"]) & data["market_allowed"] & data["sma_flat"] & data["vol_ok"] & data["volatility_ok"] & data["time_ok"]
    short_entry = (data["z"] >= data["z_short_thr"]) & data["market_allowed"] & data["sma_flat"] & data["vol_ok"] & data["volatility_ok"] & data["time_ok"]

    # Apply RSI filter if enabled
    if params.rsi_enabled:
        long_entry &= data["rsi"] < params.rsi_long_threshold
        short_entry &= data["rsi"] > params.rsi_short_threshold

    # Mark skipped reasons for entries
    skip_reasons = []
    skip_reasons.append((~data["market_allowed"], "skipped: market trending (ADX)"))
    skip_reasons.append((~data["sma_flat"], "skipped: SMA slope indicates strong trend"))
    skip_reasons.append((~data["vol_ok"], "skipped: low volume"))
    skip_reasons.append((~data["volatility_ok"], "skipped: low volatility"))
    skip_reasons.append((~data["time_ok"], "skipped: excluded hours"))
    if params.rsi_enabled:
        skip_reasons.append((data["rsi"] >= params.rsi_long_threshold, "skipped: RSI not oversold for LONG"))
        skip_reasons.append((data["rsi"] <= params.rsi_short_threshold, "skipped: RSI not overbought for SHORT"))

    # Fill entry signals and reasons (пока без подтверждения)
    data.loc[long_entry, "signal"] = "LONG"
    data.loc[short_entry, "signal"] = "SHORT"
    
    # Механизм подтверждения сигналов (требовать подтверждение на следующей свече)
    if params.require_confirmation:
        data["signal_confirmed"] = False
        data.loc[data.index[0], "signal_confirmed"] = True  # Первая строка всегда подтверждена
        
        for i in range(1, len(data)):
            if data.iloc[i]["signal"] in ["LONG", "SHORT"]:
                # Проверяем, что сигнал подтверждается на текущей свече
                prev_z = data.iloc[i-1]["z"]
                curr_z = data.iloc[i]["z"]
                
                if data.iloc[i]["signal"] == "LONG":
                    # Для LONG: Z-score должен оставаться низким или ухудшаться
                    confirmed = curr_z <= prev_z or curr_z <= -1.8
                else:  # SHORT
                    # Для SHORT: Z-score должен оставаться высоким или увеличиваться
                    confirmed = curr_z >= prev_z or curr_z >= 1.8
                
                data.iloc[i, data.columns.get_loc("signal_confirmed")] = confirmed
                
                # Если не подтвержден - отменяем сигнал
                if not confirmed:
                    data.iloc[i, data.columns.get_loc("signal")] = ""
                    data.iloc[i, data.columns.get_loc("reason")] = "skipped: no confirmation"
            else:
                data.iloc[i, data.columns.get_loc("signal_confirmed")] = True
    else:
        data["signal_confirmed"] = True

    # For rows that are candidate entries but were blocked, log reason
    candidates = (data["z"] <= data["z_long_thr"]) | (data["z"] >= data["z_short_thr"])  # price deviated
    blocked = candidates & (data["signal"] == "")
    # collect reasons per row (first matching reason)
    reasons = np.full(len(data), "", dtype=object)
    for mask, text in skip_reasons:
        idx = mask & blocked & (reasons == "")
        reasons[idx] = text
    data.loc[blocked, "reason"] = reasons[blocked]
    # Also log debug messages for blocked entries (только для последних 5 строк для производительности)
    blocked_indices = np.where(blocked)[0]
    if len(blocked_indices) > 0:
        # Логируем только последние 5 заблокированных сигналов
        for i in blocked_indices[-5:]:
            logger.debug("Signal blocked at %s: %s (z=%.2f, adx=%.2f, rsi=%.2f, sma_flat=%s, vol_ok=%s)", 
                        data.index[i], reasons[i],
                        float(data.iloc[i].get("z", 0)),
                        float(data.iloc[i].get("adx", 0)),
                        float(data.iloc[i].get("rsi", 0)),
                        bool(data.iloc[i].get("sma_flat", False)),
                        bool(data.iloc[i].get("vol_ok", False)))

    # Динамические SL/TP на основе Z-Score и волатильности
    def calculate_dynamic_sl_tp(z_abs: float, volatility_scale: float, params: StrategyParams) -> tuple:
        """Рассчитывает динамические SL/TP на основе Z-Score и волатильности"""
        if not params.use_dynamic_sl_tp:
            return params.stop_loss_atr, params.take_profit_atr
        
        # Чем сильнее отклонение, тем больше TP и меньше SL
        if z_abs > 2.5:
            tp_multiplier = 2.5
            sl_multiplier = 0.8
        elif z_abs > 2.0:
            tp_multiplier = 2.0
            sl_multiplier = 0.9
        else:
            tp_multiplier = 1.5
            sl_multiplier = 1.0
        
        # Корректировка по волатильности
        if volatility_scale > 0.02:  # Высокая волатильность
            tp_multiplier *= 0.8
            sl_multiplier *= 1.2
        
        return sl_multiplier, tp_multiplier
    
    # Вычисляем динамические SL/TP для каждой строки
    data["z_abs"] = data["z"].abs()
    data["volatility_scale"] = data["atr_relative"]
    sl_tp_pairs = data.apply(
        lambda row: calculate_dynamic_sl_tp(row["z_abs"], row["volatility_scale"], params),
        axis=1
    )
    data["sl_multiplier"] = [pair[0] for pair in sl_tp_pairs]
    data["tp_multiplier"] = [pair[1] for pair in sl_tp_pairs]
    
    # Exit logic using Z crossing towards mean and ATR-based SL/TP (динамические или фиксированные)
    z_exit = params.z_exit
    
    # Используем динамические множители для SL/TP
    long_exit = (data["z"] >= -z_exit) | (
        (data["close"] >= data["sma"] + data["tp_multiplier"] * data["atr"]) |
        (data["close"] <= data["sma"] - data["sl_multiplier"] * data["atr"])
    )

    short_exit = (data["z"] <= z_exit) | (
        (data["close"] <= data["sma"] - data["tp_multiplier"] * data["atr"]) |
        (data["close"] >= data["sma"] + data["sl_multiplier"] * data["atr"])
    )

    # Mark exits
    # For LONG exit when z >= -z_exit (closer to 0)
    data.loc[(data["z"] >= -z_exit) & (data["signal"] == ""), "signal"] = "EXIT_LONG"
    # For SHORT exit when z <= z_exit (closer to 0)
    data.loc[(data["z"] <= z_exit) & (data["signal"] == ""), "signal"] = "EXIT_SHORT"
    
    # Check ATR based exits if they are closer (используем динамические множители)
    long_atr_exit = (
        (data["close"] >= data["sma"] + data["tp_multiplier"] * data["atr"]) |
        (data["close"] <= data["sma"] - data["sl_multiplier"] * data["atr"])
    )
    short_atr_exit = (
        (data["close"] <= data["sma"] - data["tp_multiplier"] * data["atr"]) |
        (data["close"] >= data["sma"] + data["sl_multiplier"] * data["atr"])
    )
    
    data.loc[long_atr_exit & (data["signal"] == ""), "signal"] = "EXIT_LONG"
    data.loc[short_atr_exit & (data["signal"] == ""), "signal"] = "EXIT_SHORT"

    # Provide reasons for exit types
    data.loc[data["signal"] == "EXIT_LONG", "reason"] = "exit long: z target or ATR SL/TP"
    data.loc[data["signal"] == "EXIT_SHORT", "reason"] = "exit short: z target or ATR SL/TP"

    # Any non-empty signal should have a reason
    data.loc[(data["signal"] != "") & (data["reason"] == ""), "reason"] = data["signal"]

    # Keep only relevant columns for downstream systems
    out_cols = list(df.columns) + [
        "sma", "std", "z", "atr", "adx", "rsi", "signal", "reason",
        "z_long_thr", "z_short_thr", "market_allowed", "sma_flat", "vol_ok",
        "volatility_ok", "time_ok", "signal_confirmed", "sl_multiplier", "tp_multiplier"
    ]
    return data[out_cols]


__all__ = ["StrategyParams", "generate_signals"]

