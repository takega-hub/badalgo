"""zscore_strategy_v2.py

Векторизированная реализация стратегии Z-Score (Mean Reversion).
Цель: высокая производительность, типизация, логирование причин пропуска сигналов,
и вынос всех магических чисел в класс параметров для последующей оптимизации.

Файл возвращает функцию generate_signals(df, params) -> pd.DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    window: int = 20  # период для SMA и StdDev
    z_long: float = -2.5
    z_short: float = 2.5
    z_exit: float = 0.5
    vol_factor: float = 0.8
    adx_threshold: float = 25.0
    epsilon: float = 1e-9
    rsi_enabled: bool = True
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    sma_slope_threshold: float = 0.001
    atr_window: int = 14
    adx_window: int = 14
    stop_loss_atr: float = 3.0  # StopLoss в единицах ATR
    take_profit_atr: float = 2.0  # TakeProfit в единицах ATR


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
    data["vol_mean"] = data["volume"].rolling(window=params.window, min_periods=1).mean()
    data["vol_ok"] = data["volume"] > (data["vol_mean"] * params.vol_factor)

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

    # Entry conditions
    long_entry = (data["z"] <= data["z_long_thr"]) & data["market_allowed"] & data["sma_flat"] & data["vol_ok"]
    short_entry = (data["z"] >= data["z_short_thr"]) & data["market_allowed"] & data["sma_flat"] & data["vol_ok"]

    # Apply RSI filter if enabled
    if params.rsi_enabled:
        long_entry &= data["rsi"] < params.rsi_long_threshold
        short_entry &= data["rsi"] > params.rsi_short_threshold

    # Mark skipped reasons for entries
    skip_reasons = []
    skip_reasons.append((~data["market_allowed"], "skipped: market trending (ADX)"))
    skip_reasons.append((~data["sma_flat"], "skipped: SMA slope indicates strong trend"))
    skip_reasons.append((~data["vol_ok"], "skipped: low volume"))
    if params.rsi_enabled:
        skip_reasons.append((data["rsi"] >= params.rsi_long_threshold, "skipped: RSI not oversold for LONG"))
        skip_reasons.append((data["rsi"] <= params.rsi_short_threshold, "skipped: RSI not overbought for SHORT"))

    # Fill entry signals and reasons
    data.loc[long_entry, "signal"] = "LONG"
    data.loc[short_entry, "signal"] = "SHORT"

    # For rows that are candidate entries but were blocked, log reason
    candidates = (data["z"] <= data["z_long_thr"]) | (data["z"] >= data["z_short_thr"])  # price deviated
    blocked = candidates & (data["signal"] == "")
    # collect reasons per row (first matching reason)
    reasons = np.full(len(data), "", dtype=object)
    for mask, text in skip_reasons:
        idx = mask & blocked & (reasons == "")
        reasons[idx] = text
    data.loc[blocked, "reason"] = reasons[blocked]
    # Also log debug messages for blocked entries
    for i in np.where(blocked)[0]:
        logger.debug("Signal blocked at %s: %s", data.index[i], reasons[i])

    # Exit logic using Z crossing towards mean and ATR-based SL/TP
    # Exit LONG when z >= -z_exit OR price reaches TP/SL based on ATR
    z_exit = params.z_exit
    long_exit = (data["z"] >= -z_exit) | (
        (data["close"] >= data["sma"] + params.take_profit_atr * data["atr"]) |
        (data["close"] <= data["sma"] - params.stop_loss_atr * data["atr"])
    )

    short_exit = (data["z"] <= z_exit) | (
        (data["close"] <= data["sma"] - params.take_profit_atr * data["atr"]) |
        (data["close"] >= data["sma"] + params.stop_loss_atr * data["atr"])
    )

    # Mark exits
    data.loc[long_exit & (data["signal"] == ""), "signal"] = "EXIT_LONG"
    data.loc[short_exit & (data["signal"] == ""), "signal"] = "EXIT_SHORT"

    # Provide reasons for exit types
    data.loc[long_exit & (data["signal"] == "EXIT_LONG"), "reason"] = (
        "exit: z>=-z_exit or TP/SL hit"
    )
    data.loc[short_exit & (data["signal"] == "EXIT_SHORT"), "reason"] = (
        "exit: z<=z_exit or TP/SL hit"
    )

    # Any non-empty signal should have a reason
    data.loc[(data["signal"] != "") & (data["reason"] == ""), "reason"] = data["signal"]

    # Keep only relevant columns for downstream systems
    out_cols = list(df.columns) + ["sma", "std", "z", "atr", "adx", "rsi", "signal", "reason"]
    return data[out_cols]


__all__ = ["StrategyParams", "generate_signals"]

