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
    v2 = V2StrategyParams()
    # window / sma length
    v2.window = int(getattr(params, "zscore_window", getattr(params, "sma_length", v2.window)))
    # thresholds
    v2.z_long = float(getattr(params, "zscore_long", v2.z_long))
    v2.z_short = float(getattr(params, "zscore_short", v2.z_short))
    v2.z_exit = float(getattr(params, "zscore_exit", v2.z_exit))
    v2.vol_factor = float(getattr(params, "zscore_vol_factor", v2.vol_factor))
    v2.adx_threshold = float(getattr(params, "zscore_adx_threshold", v2.adx_threshold))
    v2.epsilon = float(getattr(params, "epsilon", v2.epsilon))
    # RSI options
    v2.rsi_enabled = bool(getattr(params, "zscore_rsi_enabled", True))
    v2.rsi_long_threshold = float(getattr(params, "zscore_rsi_long", v2.rsi_long_threshold))
    v2.rsi_short_threshold = float(getattr(params, "zscore_rsi_short", v2.rsi_short_threshold))
    # risk
    v2.stop_loss_atr = float(getattr(params, "zscore_stop_loss_atr", v2.stop_loss_atr))
    v2.take_profit_atr = float(getattr(params, "zscore_take_profit_atr", v2.take_profit_atr))
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
        
        # Диагностика: логируем параметры и статистику сигналов
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"[Z-Score] {symbol} Parameters: window={v2_params.window}, z_long={v2_params.z_long}, "
            f"z_short={v2_params.z_short}, adx_threshold={v2_params.adx_threshold}, "
            f"vol_factor={v2_params.vol_factor}, rsi_enabled={v2_params.rsi_enabled}, "
            f"sma_slope_threshold={v2_params.sma_slope_threshold}"
        )
        
        if df_signals is not None and not df_signals.empty:
            # Проверяем последнюю строку для диагностики
            last_row = df_signals.iloc[-1]
            if "z" in df_signals.columns:
                # Безопасное извлечение значений из Series
                def safe_get_float(row, col, default=0):
                    if col not in df_signals.columns:
                        return default
                    val = row[col]
                    # СНАЧАЛА проверяем, является ли это Series
                    if isinstance(val, pd.Series):
                        if len(val) == 0:
                            return default
                        val = val.iloc[0]
                    # Теперь проверяем на NaN (val уже скаляр)
                    if pd.isna(val):
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                def safe_get_bool(row, col, default=False):
                    if col not in df_signals.columns:
                        return default
                    val = row[col]
                    # СНАЧАЛА проверяем, является ли это Series
                    if isinstance(val, pd.Series):
                        if len(val) == 0:
                            return default
                        val = val.iloc[0]
                    # Теперь проверяем на NaN (val уже скаляр)
                    if pd.isna(val):
                        return default
                    try:
                        return bool(val)
                    except (ValueError, TypeError):
                        return default
                
                def safe_get_str(row, col, default=""):
                    if col not in df_signals.columns:
                        return default
                    val = row[col]
                    # СНАЧАЛА проверяем, является ли это Series
                    if isinstance(val, pd.Series):
                        if len(val) == 0:
                            return default
                        val = val.iloc[0]
                    # Теперь проверяем на NaN (val уже скаляр)
                    if pd.isna(val):
                        return default
                    try:
                        return str(val)
                    except (ValueError, TypeError):
                        return default
                
                last_z = safe_get_float(last_row, "z", 0)
                last_adx = safe_get_float(last_row, "adx", 0)
                last_rsi = safe_get_float(last_row, "rsi", 0)
                last_sma_flat = safe_get_bool(last_row, "sma_flat", False)
                last_vol_ok = safe_get_bool(last_row, "vol_ok", False)
                last_market_allowed = safe_get_bool(last_row, "market_allowed", False)
                last_signal = safe_get_str(last_row, "signal", "")
                last_reason = safe_get_str(last_row, "reason", "")
                
                logger.debug(
                    f"[Z-Score] {symbol} Last row diagnostics: z={last_z:.2f}, adx={last_adx:.2f}, "
                    f"rsi={last_rsi:.2f}, sma_flat={last_sma_flat}, vol_ok={last_vol_ok}, "
                    f"market_allowed={last_market_allowed}, signal={last_signal}, "
                    f"reason={last_reason}"
                )
                
                # Подсчитываем количество сигналов
                long_signals = len(df_signals[df_signals["signal"] == "LONG"])
                short_signals = len(df_signals[df_signals["signal"] == "SHORT"])
                logger.debug(
                    f"[Z-Score] {symbol} Signals count: LONG={long_signals}, SHORT={short_signals}, "
                    f"Total rows={len(df_signals)}"
                )
    except Exception as e:
        # В случае ошибки в v2 не рушим систему — логируем и возвращаем пустой список
        import logging

        logging.getLogger(__name__).exception("ZScore v2 failed: %s", e)
        return []

    results: List[Signal] = []

    if df_signals is None or df_signals.empty:
        return results

    # ВАЖНО: Фильтруем сигналы только для последних свечей (последние 3 свечи)
    # Это предотвращает генерацию сигналов для старых данных
    # Используем только последние 3 свечи, чтобы гарантировать свежесть сигналов
    MAX_SIGNAL_AGE_CANDLES = 3
    if len(df_signals) > MAX_SIGNAL_AGE_CANDLES:
        # Берем только последние N свечей для генерации сигналов
        df_signals_filtered = df_signals.tail(MAX_SIGNAL_AGE_CANDLES)
    else:
        df_signals_filtered = df_signals
    
    # ДИАГНОСТИКА: Логируем информацию о свечах для отладки
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"[Z-Score] {symbol} Filtered signals DataFrame shape: {df_signals_filtered.shape}, "
        f"columns: {list(df_signals_filtered.columns)}"
    )
    
    # Проверяем уникальность цен в отфильтрованных свечах
    if 'close' in df_signals_filtered.columns:
        unique_prices = df_signals_filtered['close'].unique()
        logger.info(
            f"[Z-Score] {symbol} Unique prices in filtered candles: {unique_prices.tolist()}, "
            f"count: {len(unique_prices)}"
        )
        # Логируем каждую свечу с ее ценой и сигналом
        for candle_idx, candle_row in df_signals_filtered.iterrows():
            candle_price = candle_row.get('close', 'N/A')
            candle_signal = candle_row.get('signal', '')
            logger.info(
                f"[Z-Score] {symbol} Candle timestamp={candle_idx}, price={candle_price}, signal={candle_signal}"
            )
    
    # Определяем индекс последней строки для обновления timestamp свежих сигналов
    last_row_idx = len(df_signals_filtered) - 1
    
    # df_signals содержит колонку 'signal' с значениями "LONG"/"SHORT"/"EXIT_*"
    for position, (idx, row) in enumerate(df_signals_filtered.iterrows()):
        sig = str(row.get("signal", "")).upper()
        if sig == "LONG":
            action = Action.LONG
        elif sig == "SHORT":
            action = Action.SHORT
        else:
            continue

        # ВАЖНО: Убеждаемся, что reason всегда начинается с префикса "zscore_"
        # Это необходимо для правильной фильтрации сигналов по стратегиям
        raw_reason = row.get("reason") or ""
        if raw_reason and not raw_reason.startswith("zscore_"):
            # Если reason есть, но не начинается с "zscore_", добавляем префикс
            reason = f"zscore_{raw_reason}"
        elif raw_reason:
            # Если reason уже начинается с "zscore_", используем как есть
            reason = raw_reason
        else:
            # Если reason нет, создаем стандартный
            reason = f"zscore_{sig.lower()}"
        
        # ВАЖНО: Берем цену из колонки 'close' для конкретной свечи (idx)
        # Это гарантирует, что каждый сигнал имеет цену соответствующей свечи
        # Используем прямое обращение к колонке через индекс, чтобы гарантировать правильное значение
        if 'close' in df_signals_filtered.columns:
            # Берем цену напрямую из DataFrame по индексу свечи
            price_from_df = df_signals_filtered.loc[idx, 'close']
            # Также проверяем значение из row для сравнения
            price_from_row = row.get("close", row.get("price", float('nan')))
            
            # Используем значение из DataFrame, так как оно гарантированно соответствует индексу
            if pd.notna(price_from_df):
                price = float(price_from_df)
            elif pd.notna(price_from_row):
                price = float(price_from_row)
            else:
                price = float('nan')
            
            # ДИАГНОСТИКА: Сравниваем значения из разных источников
            logger.info(
                f"[Z-Score] {symbol} Signal price check: idx={idx}, "
                f"price_from_df={price_from_df:.2f}, price_from_row={price_from_row:.2f}, "
                f"final_price={price:.2f}, position={position}/{last_row_idx}"
            )
        else:
            # Fallback: используем значение из row
            price = float(row.get("close", row.get("price", float('nan'))))
            logger.warning(
                f"[Z-Score] {symbol} 'close' column not found in DataFrame, using row value: {price:.2f}"
            )
        
        # ВАЖНО: Добавляем уникальный идентификатор к reason на основе timestamp свечи и цены
        # Это гарантирует, что каждый сигнал из разных свечей имеет уникальный reason
        # даже если они имеют одинаковый базовый reason (например, "LONG_poc_130.00")
        # Это также помогает отличить сигналы от разных свечей с одинаковой ценой
        import hashlib
        # Используем timestamp свечи и цену для создания уникального идентификатора
        ts_str = str(idx) if hasattr(idx, 'isoformat') else str(idx)
        # Нормализуем timestamp для генерации ID (убираем микросекунды)
        ts_normalized = ts_str.split('.')[0] if '.' in ts_str else ts_str
        unique_id = hashlib.md5(f"{ts_normalized}_{price:.2f}".encode()).hexdigest()[:6]
        
        # Добавляем уникальный идентификатор к reason только если его там еще нет
        if f"_{unique_id}" not in reason:
            reason = f"{reason}_{unique_id}"
            logger.debug(
                f"[Z-Score] {symbol} Added unique ID to reason: {reason} (timestamp={ts_normalized}, price={price:.2f})"
            )

        try:
            ts = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
        except Exception:
            ts = pd.Timestamp.now()

        # КРИТИЧЕСКИ ВАЖНО: Если сигнал соответствует последней строке DataFrame,
        # обновляем timestamp на текущее время, чтобы сигнал считался свежим
        # Это гарантирует немедленное исполнение сигналов от Z-Score стратегии
        if position == last_row_idx:
            # Сигнал на последней свече - обновляем timestamp на текущее время
            ts = pd.Timestamp.now(tz='UTC')
        
        results.append(Signal(timestamp=ts, action=action, reason=str(reason), price=price))

    return results


__all__ = ["build_zscore_signals", "Action", "Signal"]
