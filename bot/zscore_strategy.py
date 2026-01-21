"""Compatibility wrapper for the new vectorized Z-Score strategy.

Этот модуль адаптирует новую векторизованную реализацию
[`bot/zscore_strategy_v2.py`](bot/zscore_strategy_v2.py:1) к устаревшему интерфейсу,
используемому в `live.py` — функцию `build_zscore_signals(df, params, symbol)`,
а также экспортирует `Action` и `Signal`, чтобы старый код мог импортировать их
как раньше.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from bot.strategy import Action, Signal
from bot.config import StrategyParams as ConfigStrategyParams

# Импортируем векторизированную реализацию
from bot.zscore_strategy_v2 import generate_signals as v2_generate_signals, StrategyParams as V2StrategyParams


def _map_config_to_v2(params: ConfigStrategyParams) -> V2StrategyParams:
    """Map existing config StrategyParams to V2StrategyParams with safe fallbacks."""
    # Берём значения, если они есть, иначе используем значения по умолчанию из V2
    v2 = V2StrategyParams()
    # window / sma length
    v2.window = int(getattr(params, "sma_length", getattr(params, "window", v2.window)))
    # thresholds (try multiple possible field names)
    v2.z_long = float(getattr(params, "z_long", getattr(params, "z_threshold_long", v2.z_long)))
    v2.z_short = float(getattr(params, "z_short", getattr(params, "z_threshold_short", v2.z_short)))
    v2.z_exit = float(getattr(params, "z_exit", v2.z_exit))
    v2.vol_factor = float(getattr(params, "vol_factor", v2.vol_factor))
    v2.adx_threshold = float(getattr(params, "adx_threshold", getattr(params, "adx_length", v2.adx_threshold)))
    v2.epsilon = float(getattr(params, "epsilon", v2.epsilon))
    # RSI options
    v2.rsi_enabled = bool(getattr(params, "rsi_enabled", True))
    v2.rsi_long_threshold = float(getattr(params, "rsi_long_threshold", getattr(params, "range_rsi_oversold", v2.rsi_long_threshold)))
    v2.rsi_short_threshold = float(getattr(params, "rsi_short_threshold", getattr(params, "range_rsi_overbought", v2.rsi_short_threshold)))
    # SMA slope / volatility windows
    v2.sma_slope_threshold = float(getattr(params, "sma_slope_threshold", getattr(params, "trend_slope_threshold", v2.sma_slope_threshold)))
    v2.atr_window = int(getattr(params, "atr_window", getattr(params, "atr_length", v2.atr_window)))
    v2.adx_window = int(getattr(params, "adx_window", getattr(params, "adx_length", v2.adx_window)))
    # risk related mapped to ATR multipliers (if present)
    v2.stop_loss_atr = float(getattr(params, "zscore_stop_loss_atr", getattr(params, "stop_loss_atr", v2.stop_loss_atr)))
    v2.take_profit_atr = float(getattr(params, "zscore_take_profit_atr", getattr(params, "take_profit_atr", v2.take_profit_atr)))
    return v2


def build_zscore_signals(df: pd.DataFrame, params: Optional[ConfigStrategyParams], symbol: str = "Unknown") -> List[Signal]:
    """Compatibility function used by live.py.

    Принимает привычный DataFrame и параметры стратегии из `bot.config` и возвращает
    список объектов `bot.strategy.Signal` (LONG/SHORT). Возвращаем только входные сигналы
    (LONG/SHORT). Причина и цена заполняются из столбцов, сгенерированных v2.
    """
    if params is None:
        # Если параметров нет — используем дефолтные параметры v2
        v2_params = V2StrategyParams()
    else:
        v2_params = _map_config_to_v2(params)

    try:
        df_signals = v2_generate_signals(df, v2_params)
    except Exception as e:
        # В случае ошибки в v2 не рушим систему — логируем и возвращаем пустой список
        import logging

        logging.getLogger(__name__).exception("ZScore v2 failed: %s", e)
        return []

    results: List[Signal] = []

    if df_signals is None or df_signals.empty:
        return results

    # df_signals содержит колонку 'signal' с значениями "LONG"/"SHORT"/"EXIT_*"
    for idx, row in df_signals.iterrows():
        sig = str(row.get("signal", "")).upper()
        if sig == "LONG":
            action = Action.LONG
        elif sig == "SHORT":
            action = Action.SHORT
        else:
            continue

        reason = row.get("reason") or f"zscore_{sig.lower()}"
        price = float(row.get("close", row.get("price", float('nan'))))

        try:
            ts = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
        except Exception:
            ts = pd.Timestamp.now()

        results.append(Signal(timestamp=ts, action=action, reason=str(reason), price=price))

    return results


__all__ = ["build_zscore_signals", "Action", "Signal"]
