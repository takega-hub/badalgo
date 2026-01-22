import math
import time
import warnings
import os
import logging

# Подавляем предупреждения scikit-learn ДО импорта библиотек
# Устанавливаем переменную окружения ПЕРВОЙ
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['SKLEARN_WARNINGS'] = 'ignore'

# Фильтруем все предупреждения sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*propagate the scikit-learn configuration.*')
# Специфичное предупреждение из терминала
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

# Подавляем XGBoost warnings
logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*XGBoost.*')
warnings.filterwarnings('ignore', message='.*Booster.save_model.*')
warnings.filterwarnings('ignore', message='.*serialized model.*')

# Подавляем DeprecationWarning от pybit (datetime.utcnow() deprecated)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pybit')
warnings.filterwarnings('ignore', message='.*datetime.datetime.utcnow.*')
warnings.filterwarnings('ignore', message='.*is deprecated and scheduled for removal.*')

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from bot.config import AppSettings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Bias, build_signals, enrich_for_strategy
from bot.web.history import add_signal, add_trade, check_recent_loss_trade
from bot.ml.strategy_ml import build_ml_signals
from bot.smc_strategy import build_smc_signals
from bot.ict_strategy import build_ict_signals
from bot.liquidation_hunter_strategy import build_liquidation_hunter_signals
from bot.zscore_strategy import build_zscore_signals
from bot.vbo_strategy import build_vbo_signals
from bot.amt_orderflow_strategy import (
    detect_absorption_squeeze_short,
    AbsorptionConfig,
    VolumeProfileConfig,
    generate_amt_signals,
    LhOrderflowConfig,
    generate_lh_orderflow_signals,
    build_volume_profile_from_ohlcv,
    _parse_trades,
    _compute_cvd_metrics,
    _resolve_symbol_settings,
)

# Импорт для обработки ошибок Bybit API
try:
    from pybit.exceptions import InvalidRequestError
except ImportError:
    InvalidRequestError = Exception


def _timeframe_to_bybit_interval(tf: str) -> str:
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
    }
    return mapping.get(tf, "15")


def _log(message: str, symbol: Optional[str] = None) -> None:
    """
    Вспомогательная функция для логирования с префиксом символа.
    
    Args:
        message: Сообщение для логирования
        symbol: Торговая пара (опционально, для многопарной торговли)
    """
    if symbol:
        print(f"[live] [{symbol}] {message}")
    else:
        print(f"[live] {message}")


def _wait_with_stop_check(stop_event: Optional[threading.Event], timeout: float, symbol: Optional[str] = None) -> bool:
    """
    Ожидание с проверкой события остановки.
    
    Args:
        stop_event: Событие остановки (если None, используется обычный sleep)
        timeout: Время ожидания в секундах
        symbol: Торговая пара для логирования
    
    Returns:
        True если событие остановки установлено, False если просто истек таймаут
    """
    if stop_event is None:
        time.sleep(timeout)
        return False
    else:
        # Для длительных ожиданий (> 10 секунд) обновляем статус воркера периодически
        # чтобы MultiSymbolManager не считал воркер "мертвым"
        if timeout > 10.0:
            # Обновляем статус каждые 10 секунд во время ожидания
            update_interval = 10.0
            elapsed = 0.0
            try:
                from bot.multi_symbol_manager import update_worker_status
                update_worker_status_available = True
            except ImportError:
                update_worker_status_available = False
            
            while elapsed < timeout:
                remaining = min(update_interval, timeout - elapsed)
                if stop_event.wait(timeout=remaining):
                    if symbol:
                        _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                    return True
                elapsed += remaining
                
                # Обновляем статус воркера каждые 10 секунд
                if update_worker_status_available and symbol:
                    update_worker_status(symbol, current_status="Running", last_action="Waiting...", error=None)
            
            return False
        else:
            # Для коротких ожиданий используем обычный wait
            if stop_event.wait(timeout=timeout):
                if symbol:
                    _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                return True
            return False


def _load_processed_signals(processed_signals_file: Path) -> set:
    """Загрузить обработанные сигналы из файла."""
    if processed_signals_file.exists():
        try:
            import json
            with open(processed_signals_file, 'r', encoding='utf-8') as f:
                saved_signals = json.load(f)
                processed_signals = set(saved_signals.get("signal_ids", []))
                print(f"[live] Loaded {len(processed_signals)} processed signals from file")
                return processed_signals
        except Exception as e:
            print(f"[live] ⚠️ Error loading processed signals: {e}")
            return set()
    return set()


def _save_processed_signals(processed_signals: set, processed_signals_file: Path) -> None:
    """Сохранить обработанные сигналы в файл."""
    try:
        import json
        data = {
            "signal_ids": list(processed_signals),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(processed_signals_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[live] ⚠️ Error saving processed signals: {e}")


def _load_bot_state(symbol: str) -> Dict[str, Any]:
    """Загрузить состояние бота для конкретного символа из файла."""
    state_file = Path(__file__).parent.parent / f"bot_state_{symbol}.json"
    if state_file.exists():
        try:
            import json
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[live] [{symbol}] ⚠️ Error loading bot state: {e}")
    return {}


def _save_bot_state(symbol: str, state: Dict[str, Any]) -> None:
    """Сохранить состояние бота для конкретного символа в файл."""
    state_file = Path(__file__).parent.parent / f"bot_state_{symbol}.json"
    try:
        import json
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[live] [{symbol}] ⚠️ Error saving bot state: {e}")


def _update_and_save_position_state(
    symbol: str,
    position_strategy: Dict[str, str],
    position_order_id: Dict[str, str],
    position_order_link_id: Dict[str, str],
    position_add_count: Dict[str, int],
    position_entry_price: Dict[str, float],
    strategy_type: Optional[str] = None,
    order_id: Optional[str] = None,
    order_link_id: Optional[str] = None,
    add_count: Optional[int] = None,
    entry_price: Optional[float] = None,
) -> None:
    """Обновляет состояние позиции и сохраняет его в файл."""
    if strategy_type is not None:
        position_strategy[symbol] = strategy_type
    if order_id is not None:
        position_order_id[symbol] = order_id
    if order_link_id is not None:
        position_order_link_id[symbol] = order_link_id
    if add_count is not None:
        position_add_count[symbol] = add_count
    if entry_price is not None:
        position_entry_price[symbol] = entry_price
        
    state = {
        "strategy_type": position_strategy.get(symbol, "unknown"),
        "order_id": position_order_id.get(symbol, ""),
        "order_link_id": position_order_link_id.get(symbol, ""),
        "add_count": position_add_count.get(symbol, 0),
        "entry_price": position_entry_price.get(symbol, 0.0),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    _save_bot_state(symbol, state)


def _clear_bot_state(symbol: str) -> None:
    """Удаляет файл состояния бота для символа."""
    state_file = Path(__file__).parent.parent / f"bot_state_{symbol}.json"
    if state_file.exists():
        try:
            state_file.unlink()
        except Exception as e:
            print(f"[live] [{symbol}] ⚠️ Error deleting bot state file: {e}")


def _close_conflicting_positions_for_primary(
    client: BybitClient,
    settings: AppSettings,
    new_primary_bias: Bias,
) -> None:
    """
    Закрывает противонаправленные позиции по другим парам,
    когда на PRIMARY_SYMBOL открывается новая позиция.

    Логика:
    - Если на PRIMARY_SYMBOL открывается LONG, закрываем все SHORT по другим активным символам.
    - Если на PRIMARY_SYMBOL открывается SHORT, закрываем все LONG по другим активным символам.
    
    ВАЖНО: Эта функция вызывается ТОЛЬКО при открытии позиции на PRIMARY_SYMBOL,
    и только после подтверждения, что позиция действительно открыта.
    """
    try:
        # ВАЖНО: Используем ТОЛЬКО primary_symbol из настроек, БЕЗ fallback на symbol
        primary_symbol = getattr(settings, "primary_symbol", None)
        if not primary_symbol:
            return

        primary_symbol = primary_symbol.upper()
        
        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Убеждаемся, что на PRIMARY_SYMBOL действительно есть открытая позиция
        # Это предотвращает закрытие позиций, если позиция на PRIMARY_SYMBOL не открылась
        try:
            primary_position = _get_position(client, primary_symbol)
            if not primary_position or primary_position.get("size", 0) <= 0:
                print(f"[live] ⚠️ PRIMARY_SYMBOL ({primary_symbol}) position not confirmed - skipping close of opposite positions")
                return
        except Exception as e:
            print(f"[live] ⚠️ Error verifying PRIMARY_SYMBOL position before closing opposite positions: {e}")
            return

        # Получаем все открытые позиции по USDT
        resp = client.get_position_info(settle_coin="USDT")
        if resp.get("retCode") != 0:
            print(f"[live] ⚠️ Failed to load positions for PRIMARY_SYMBOL conflict check: {resp.get('retMsg', 'Unknown error')}")
            return

        positions = resp.get("result", {}).get("list", [])
        if not positions:
            return

        active_symbols = set(getattr(settings, "active_symbols", []) or [])

        for pos in positions:
            try:
                size = float(pos.get("size", 0))
            except (TypeError, ValueError):
                size = 0

            if size <= 0:
                continue

            symbol = pos.get("symbol", "").upper()
            if not symbol or symbol == primary_symbol:
                # Пропускаем сам PRIMARY_SYMBOL
                continue

            # Ограничиваемся только активными символами, чтобы не трогать лишнее
            if active_symbols and symbol not in active_symbols:
                continue

            side_str = pos.get("side", "").upper()
            if side_str not in ("BUY", "SELL"):
                continue

            existing_bias = Bias.LONG if side_str == "BUY" else Bias.SHORT

            # Проверяем, является ли позиция противонаправленной по отношению к новой позиции на PRIMARY_SYMBOL
            if new_primary_bias == Bias.LONG and existing_bias == Bias.SHORT:
                close_side = "Buy"  # Buy закрывает SHORT
            elif new_primary_bias == Bias.SHORT and existing_bias == Bias.LONG:
                close_side = "Sell"  # Sell закрывает LONG
            else:
                # Направление не конфликтует - пропускаем
                continue

            print(f"[live] [{symbol}] ⚠️ Closing opposite position because PRIMARY_SYMBOL ({primary_symbol}) opened {new_primary_bias.value} position")
            print(f"[live] [{symbol}]   Existing position: side={existing_bias.value}, size={size}")

            try:
                close_resp = client.place_order(
                    symbol=symbol,
                    side=close_side,
                    qty=size,
                    reduce_only=True,
                )
                if close_resp.get("retCode") == 0:
                    print(f"[live] [{symbol}] ✅ Opposite position closed successfully due to PRIMARY_SYMBOL {new_primary_bias.value}")
                else:
                    print(f"[live] [{symbol}] ⚠️ Failed to close opposite position: {close_resp.get('retMsg', 'Unknown error')} (ErrCode: {close_resp.get('retCode')})")
            except Exception as e:
                print(f"[live] [{symbol}] ⚠️ Error closing opposite position due to PRIMARY_SYMBOL {new_primary_bias.value}: {e}")

    except Exception as e:
        print(f"[live] ⚠️ Error in _close_conflicting_positions_for_primary: {e}")


def _check_primary_symbol_position(
    client: BybitClient,
    current_symbol: str,
    settings: AppSettings,
    target_action: Action,
) -> Tuple[bool, Optional[str]]:
    """
    Проверяет позицию на PRIMARY_SYMBOL и блокирует открытие позиций в противоположном направлении.
    
    Логика:
    - Если на PRIMARY_SYMBOL есть позиция LONG, то для других символов не открывать SHORT
    - Если на PRIMARY_SYMBOL есть позиция SHORT, то для других символов не открывать LONG
    - Если текущий символ - это PRIMARY_SYMBOL, то проверку не делаем (можно открывать любые позиции)
    
    Args:
        client: Клиент Bybit для получения позиций
        current_symbol: Текущий символ, для которого проверяется возможность открытия позиции
        settings: Настройки приложения (содержат primary_symbol)
        target_action: Действие, которое планируется выполнить (LONG или SHORT)
    
    Returns:
        Tuple[bool, Optional[str]]: (should_block, reason)
        - should_block: True если нужно заблокировать открытие позиции
        - reason: Причина блокировки (если should_block == True)
    """
    try:
        # Получаем PRIMARY_SYMBOL из настроек
        # ВАЖНО: Используем ТОЛЬКО primary_symbol из настроек, БЕЗ fallback на symbol
        primary_symbol = getattr(settings, 'primary_symbol', None)
        if not primary_symbol:
            # Если PRIMARY_SYMBOL не задан, проверку не делаем
            print(f"[live] [{current_symbol}] ⚠️ PRIMARY_SYMBOL not set in settings, skipping check")
            return False, None
        
        print(f"[live] [{current_symbol}] 🔍 PRIMARY_SYMBOL check: primary_symbol={primary_symbol}, current_symbol={current_symbol}, target_action={target_action.value}")
        
        # Если текущий символ - это PRIMARY_SYMBOL, проверку не делаем
        if current_symbol.upper() == primary_symbol.upper():
            print(f"[live] [{current_symbol}] ℹ️  Current symbol is PRIMARY_SYMBOL, skipping check")
            return False, None
        
        # Получаем позицию на PRIMARY_SYMBOL
        try:
            pos_resp = client.get_position_info(symbol=primary_symbol)
            ret_code = pos_resp.get("retCode")
            if ret_code != 0:
                # Ошибка получения позиции - не блокируем, но логируем
                ret_msg = pos_resp.get("retMsg", "Unknown error")
                print(f"[live] [{current_symbol}] ⚠️ Error getting PRIMARY_SYMBOL ({primary_symbol}) position: retCode={ret_code}, retMsg={ret_msg}, skipping check")
                return False, None
            
            pos_list = pos_resp.get("result", {}).get("list", [])
            primary_position = None
            primary_bias = None
            
            print(f"[live] [{current_symbol}] 🔍 PRIMARY_SYMBOL position response: retCode={pos_resp.get('retCode')}, positions found: {len(pos_list)}")
            
            for pos_item in pos_list:
                size = float(pos_item.get("size", 0))
                side = pos_item.get("side", "").upper()
                print(f"[live] [{current_symbol}]   Position item: symbol={pos_item.get('symbol')}, side={side}, size={size}")
                if size > 0:
                    primary_position = pos_item
                    primary_bias = Bias.LONG if side == "BUY" else Bias.SHORT
                    print(f"[live] [{current_symbol}] ✅ Found open position on PRIMARY_SYMBOL: {primary_bias.value} (size={size})")
                    break
            
            # Если на PRIMARY_SYMBOL нет позиции - проверку не делаем
            if not primary_position:
                print(f"[live] [{current_symbol}] ℹ️  No open position on PRIMARY_SYMBOL ({primary_symbol}), skipping check")
                return False, None
            
            # Преобразуем target_action (Action) в Bias для сравнения
            target_bias = Bias.LONG if target_action == Action.LONG else Bias.SHORT
            
            # Логируем детали проверки
            primary_size = float(primary_position.get("size", 0))
            primary_side = primary_position.get("side", "UNKNOWN")
            print(f"[live] [{current_symbol}] 🔍 PRIMARY_SYMBOL check: {primary_symbol} has {primary_bias.value} position (size={primary_size}, side={primary_side})")
            print(f"[live] [{current_symbol}]    Target action: {target_action.value} (bias: {target_bias.value})")
            
            # ВАЖНО: Если на PRIMARY_SYMBOL есть позиция, на других символах можно открывать ТОЛЬКО в том же направлении
            # Если направление не совпадает - блокируем
            if primary_bias != target_bias:
                # Направление не совпадает с PRIMARY_SYMBOL - блокируем
                print(f"[live] [{current_symbol}] ⛔ BLOCKED: PRIMARY_SYMBOL ({primary_symbol}) has {primary_bias.value} position, but trying to open {target_action.value} ({target_bias.value}) on {current_symbol}")
                print(f"[live] [{current_symbol}]    Only {primary_bias.value} positions allowed on other symbols when PRIMARY_SYMBOL has {primary_bias.value} position")
                return True, f"PRIMARY_SYMBOL ({primary_symbol}) has {primary_bias.value} position - can only open {primary_bias.value} on {current_symbol}, not {target_action.value}"
            
            # Направление совпадает - можно открывать
            print(f"[live] [{current_symbol}] ✅ ALLOWED: PRIMARY_SYMBOL ({primary_symbol}) has {primary_bias.value}, target is {target_action.value} ({target_bias.value}) - same direction, OK to open")
            return False, None
            
        except Exception as e:
            # Ошибка при получении позиции - не блокируем, но логируем
            print(f"[live] [{current_symbol}] ⚠️ Error checking PRIMARY_SYMBOL position: {e}")
            return False, None
            
    except Exception as e:
        # Общая ошибка - не блокируем
        print(f"[live] [{current_symbol}] ⚠️ Error in _check_primary_symbol_position: {e}")
        return False, None


def _calculate_tp_sl_for_signal(
    sig,
    settings: AppSettings,
    entry_price: float,
    df_data: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Рассчитывает TP и SL для сигнала при открытии позиции.
    Использует уровни поддержки/сопротивления, если доступны, иначе fallback на фиксированные проценты.
    
    Args:
        sig: Сигнал (Signal объект)
        settings: Настройки бота
        entry_price: Цена входа
        df_data: DataFrame с данными для поиска уровней поддержки/сопротивления (опционально)
        
    Returns:
        Tuple (take_profit_price, stop_loss_price) или (None, None) если не удалось рассчитать
    """
    try:
        # Если в сигнале уже есть рекомендованные уровни (SMC или ML), используем их
        if hasattr(sig, 'stop_loss') and sig.stop_loss and hasattr(sig, 'take_profit') and sig.take_profit:
            # Используем предрассчитанные уровни, но проверяем, что SL соответствует настройкам (7-10% от маржи)
            pre_tp = sig.take_profit
            pre_sl = sig.stop_loss
            
            # Получаем настройки SL от маржи
            sl_pct_margin = settings.risk.stop_loss_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'stop_loss_pct') else 0.15
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            
            # Преобразуем проценты от маржи в проценты от цены
            min_sl_pct_from_price = 0.07 / leverage  # Минимум 7% от маржи = 0.7% от цены при 10x
            max_sl_pct_from_price = 0.10 / leverage  # Максимум 10% от маржи = 1% от цены при 10x
            target_sl_pct_from_price = sl_pct_margin / leverage  # Целевой SL от маржи
            
            # Проверяем, соответствует ли предрассчитанный SL настройкам
            if sig.action == Action.LONG:
                sl_deviation_pct = abs(entry_price - pre_sl) / entry_price
            else:  # SHORT
                sl_deviation_pct = abs(pre_sl - entry_price) / entry_price
            
            # Если SL слишком маленький (меньше 7% от маржи), пересчитываем на основе настроек
            if sl_deviation_pct < min_sl_pct_from_price:
                _log(f"⚠️ Pre-calculated SL too small ({sl_deviation_pct*100:.2f}% from price < {min_sl_pct_from_price*100:.2f}%), recalculating from settings ({target_sl_pct_from_price*100:.2f}% from price = {sl_pct_margin*100:.0f}% from margin)", settings.symbol)
                if sig.action == Action.LONG:
                    pre_sl = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    pre_sl = entry_price * (1 + target_sl_pct_from_price)
            elif sl_deviation_pct > max_sl_pct_from_price:
                _log(f"⚠️ Pre-calculated SL too large ({sl_deviation_pct*100:.2f}% from price > {max_sl_pct_from_price*100:.2f}%), recalculating from settings ({target_sl_pct_from_price*100:.2f}% from price = {sl_pct_margin*100:.0f}% from margin)", settings.symbol)
                if sig.action == Action.LONG:
                    pre_sl = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    pre_sl = entry_price * (1 + target_sl_pct_from_price)
            else:
                _log(f"✅ Pre-calculated SL is within range ({sl_deviation_pct*100:.2f}% from price = {sl_deviation_pct*leverage*100:.0f}% from margin)", settings.symbol)
            
            _log(f"Using levels: TP={pre_tp:.2f}, SL={pre_sl:.2f} (SL: {abs(entry_price-pre_sl)/entry_price*100:.2f}% from price = {abs(entry_price-pre_sl)/entry_price*leverage*100:.0f}% from margin)", settings.symbol)
            return pre_tp, pre_sl

        strategy_type = None
        if sig.reason.startswith("ml_"):
            strategy_type = "ml"
        elif sig.reason.startswith("trend_"):
            strategy_type = "trend"
        elif sig.reason.startswith("range_"):
            strategy_type = "flat"
        elif sig.reason.startswith("liquidation_hunter_"):
            strategy_type = "liquidation_hunter"
        elif sig.reason.startswith("zscore_"):
            strategy_type = "zscore"
        elif sig.reason.startswith("vbo_"):
            strategy_type = "vbo"
        elif sig.reason.startswith("ict_"):
            strategy_type = "ict"
        elif sig.reason.startswith("smc_"):
            strategy_type = "smc"
        
        # Логируем определение стратегии для отладки
        if strategy_type:
            print(f"[live] 🔍 TP/SL calculation: detected strategy_type='{strategy_type}' from signal reason='{sig.reason}'")
        
        # Пытаемся использовать уровни поддержки/сопротивления, если доступны
        use_sr_levels = False
        nearest_resistance = None
        nearest_support = None
        
        if df_data is not None and len(df_data) > 0:
            try:
                # Берем последнюю строку с уровнями
                last_row = df_data.iloc[-1]
                
                # Пробуем получить ближайшие уровни
                if pd.notna(last_row.get("nearest_resistance")):
                    nearest_resistance = float(last_row["nearest_resistance"])
                elif pd.notna(last_row.get("donchian_resistance")):
                    nearest_resistance = float(last_row["donchian_resistance"])
                elif pd.notna(last_row.get("bb_resistance")):
                    nearest_resistance = float(last_row["bb_resistance"])
                
                if pd.notna(last_row.get("nearest_support")):
                    nearest_support = float(last_row["nearest_support"])
                elif pd.notna(last_row.get("donchian_support")):
                    nearest_support = float(last_row["donchian_support"])
                elif pd.notna(last_row.get("bb_support")):
                    nearest_support = float(last_row["bb_support"])
                
                # Используем уровни, если они найдены и находятся в пределах настроек TP/SL
                # Получаем максимальные границы из настроек (от маржи)
                max_tp_pct_margin = settings.risk.take_profit_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'take_profit_pct') else 0.30
                max_sl_pct_margin = settings.risk.stop_loss_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'stop_loss_pct') else 0.15
                
                # Преобразуем проценты от маржи в проценты от цены: / leverage
                leverage = settings.leverage if hasattr(settings, 'leverage') else 10
                max_tp_pct = max_tp_pct_margin / leverage
                max_sl_pct = max_sl_pct_margin / leverage
                
                if sig.action == Action.LONG:
                    # Для LONG: сопротивление должно быть в пределах max_tp_pct, поддержка в пределах max_sl_pct
                    if nearest_resistance and nearest_resistance > entry_price and (nearest_resistance - entry_price) / entry_price <= max_tp_pct:
                        use_sr_levels = True
                    if nearest_support and nearest_support < entry_price and (entry_price - nearest_support) / entry_price <= max_sl_pct:
                        use_sr_levels = True
                else:  # SHORT
                    # Для SHORT: поддержка должна быть в пределах max_tp_pct, сопротивление в пределах max_sl_pct
                    if nearest_support and nearest_support < entry_price and (entry_price - nearest_support) / entry_price <= max_tp_pct:
                        use_sr_levels = True
                    if nearest_resistance and nearest_resistance > entry_price and (nearest_resistance - entry_price) / entry_price <= max_sl_pct:
                        use_sr_levels = True
            except Exception as e:
                print(f"[live] ⚠️ Error extracting support/resistance levels: {e}")
                use_sr_levels = False
        
        if strategy_type == "ml":
            # Для ML стратегии используем настройки из ml_target_profit_pct_margin и ml_max_loss_pct_margin
            # Проценты от маржи нужно перевести в проценты от цены: / leverage (без деления на 100, так как настройки уже в процентах)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: SL должен быть в диапазоне 7-10% от маржи ПЕРЕД расчетом
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
            
            # Проверяем и корректируем ml_max_loss_pct_margin ДО расчета sl_pct
            sl_pct_margin_raw = settings.ml_max_loss_pct_margin
            if sl_pct_margin_raw < min_sl_pct_from_margin * 100:
                print(f"[live] 🚨 CRITICAL: ML SL from margin ({sl_pct_margin_raw}%) < {min_sl_pct_from_margin*100:.0f}%, adjusting to {min_sl_pct_from_margin*100:.0f}%")
                sl_pct_margin_raw = min_sl_pct_from_margin * 100
            elif sl_pct_margin_raw > max_sl_pct_from_margin * 100:
                print(f"[live] 🚨 CRITICAL: ML SL from margin ({sl_pct_margin_raw}%) > {max_sl_pct_from_margin*100:.0f}%, adjusting to {max_sl_pct_from_margin*100:.0f}%")
                sl_pct_margin_raw = max_sl_pct_from_margin * 100
            
            tp_pct = settings.ml_target_profit_pct_margin / settings.leverage / 100.0
            sl_pct = sl_pct_margin_raw / settings.leverage / 100.0
            
            print(f"[live] 🔍 ML TP/SL calculation: ml_target_profit_pct_margin={settings.ml_target_profit_pct_margin}%, ml_max_loss_pct_margin={sl_pct_margin_raw}% (adjusted from {settings.ml_max_loss_pct_margin}%), leverage={settings.leverage}x")
            print(f"[live]   → tp_pct={tp_pct*100:.4f}% from price, sl_pct={sl_pct*100:.4f}% from price = {sl_pct*settings.leverage*100:.2f}% from margin")
            
            # Пытаемся извлечь TP/SL из reason сигнала (формат: "ml_LONG_сила_среднее_70%_TP_2.50%_SL_1.00%_...")
            # Если в сигнале указаны конкретные проценты, используем их (они уже в процентах от цены)
            # ВАЖНО: Но проверяем, что SL в диапазоне 7-10% от маржи
            import re
            tp_match = re.search(r'TP_([\d.]+)%', sig.reason)
            sl_match = re.search(r'SL_([\d.]+)%', sig.reason)
            
            if tp_match:
                # Извлекаем процент из reason (например, "2.50%") и преобразуем в долю (0.025)
                tp_pct = float(tp_match.group(1)) / 100.0
            
            if sl_match:
                # Извлекаем процент из reason (например, "1.00%") и преобразуем в долю (0.01)
                extracted_sl_pct = float(sl_match.group(1)) / 100.0
                extracted_sl_pct_from_margin = extracted_sl_pct * settings.leverage
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: SL из сигнала должен быть в диапазоне 7-10% от маржи
                min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
                max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
                
                if extracted_sl_pct_from_margin < min_sl_pct_from_margin:
                    print(f"[live] ⚠️ WARNING: SL from signal reason ({extracted_sl_pct*100:.2f}% from price = {extracted_sl_pct_from_margin*100:.1f}% from margin) < {min_sl_pct_from_margin*100:.0f}%, using {min_sl_pct_from_margin*100:.0f}% from margin")
                    sl_pct = min_sl_pct_from_margin / settings.leverage
                elif extracted_sl_pct_from_margin > max_sl_pct_from_margin:
                    print(f"[live] ⚠️ WARNING: SL from signal reason ({extracted_sl_pct*100:.2f}% from price = {extracted_sl_pct_from_margin*100:.1f}% from margin) > {max_sl_pct_from_margin*100:.0f}%, using {max_sl_pct_from_margin*100:.0f}% from margin")
                    sl_pct = max_sl_pct_from_margin / settings.leverage
                else:
                    sl_pct = extracted_sl_pct
                    print(f"[live] ✅ SL from signal reason is valid: {extracted_sl_pct*100:.2f}% from price = {extracted_sl_pct_from_margin*100:.1f}% from margin")
            
            # Используем уровни поддержки/сопротивления, если доступны
            if use_sr_levels:
                if sig.action == Action.LONG:
                    # Для LONG: TP на сопротивление, SL на поддержку
                    take_profit = nearest_resistance if nearest_resistance and nearest_resistance > entry_price else entry_price * (1 + tp_pct)
                    stop_loss = nearest_support if nearest_support and nearest_support < entry_price else entry_price * (1 - sl_pct)
                else:  # SHORT
                    # Для SHORT: TP на поддержку, SL на сопротивление
                    take_profit = nearest_support if nearest_support and nearest_support < entry_price else entry_price * (1 - tp_pct)
                    stop_loss = nearest_resistance if nearest_resistance and nearest_resistance > entry_price else entry_price * (1 + sl_pct)
            else:
                # Fallback на фиксированные проценты
                if sig.action == Action.LONG:
                    take_profit = entry_price * (1 + tp_pct)
                    stop_loss = entry_price * (1 - sl_pct)
                else:  # SHORT
                    take_profit = entry_price * (1 - tp_pct)
                    stop_loss = entry_price * (1 + sl_pct)
            
            # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что SL находится в диапазоне 7-10% от маржи
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
            
            if sig.action == Action.LONG:
                sl_deviation_pct_from_price = abs(entry_price - stop_loss) / entry_price
            else:  # SHORT
                sl_deviation_pct_from_price = abs(stop_loss - entry_price) / entry_price
            
            sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
            
            # Если SL меньше 7% от маржи, увеличиваем до минимума
            if sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                target_sl_pct_from_price = min_sl_pct_from_margin / leverage
                if sig.action == Action.LONG:
                    stop_loss = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    stop_loss = entry_price * (1 + target_sl_pct_from_price)
                print(f"[live] ⚠️ ML SL too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusted to {min_sl_pct_from_margin*100:.0f}% from margin ({target_sl_pct_from_price*100:.2f}% from price)")
            # Если SL больше 10% от маржи, уменьшаем до максимума
            elif sl_deviation_pct_from_margin > max_sl_pct_from_margin:
                target_sl_pct_from_price = max_sl_pct_from_margin / leverage
                if sig.action == Action.LONG:
                    stop_loss = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    stop_loss = entry_price * (1 + target_sl_pct_from_price)
                print(f"[live] ⚠️ ML SL too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusted to {max_sl_pct_from_margin*100:.0f}% from margin ({target_sl_pct_from_price*100:.2f}% from price)")
            else:
                print(f"[live] ✅ ML SL is within range: {sl_deviation_pct_from_margin*100:.1f}% from margin ({sl_deviation_pct_from_price*100:.2f}% from price)")
            
            return take_profit, stop_loss
            
        elif strategy_type == "liquidation_hunter":
            # Для Liquidation Hunter стратегии (mean reversion) используем TP/SL,
            # с особым режимом для orderflow‑сигналов lh_of_* (TP=POC из reason)
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            
            tp_pct_from_price = 0.025  # 2.5% от цены = 25% от маржи при 10x
            sl_pct_from_price = 0.010   # 1.0% от цены = 10% от маржи при 10x
            
            # Проверяем, не превышают ли настройки максимальные границы
            max_tp_pct_margin = settings.risk.take_profit_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'take_profit_pct') else 0.30
            max_sl_pct_margin = settings.risk.stop_loss_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'stop_loss_pct') else 0.15
            
            # Нормализуем проценты
            if max_tp_pct_margin > 1.0:
                max_tp_pct_margin = max_tp_pct_margin / 100.0
            if max_sl_pct_margin > 1.0:
                max_sl_pct_margin = max_sl_pct_margin / 100.0
            
            max_tp_pct = max_tp_pct_margin / leverage
            max_sl_pct = max_sl_pct_margin / leverage
            
            # Ограничиваем нашими значениями, но не превышаем максимумы
            tp_pct_from_price = min(tp_pct_from_price, max_tp_pct)
            sl_pct_from_price = min(sl_pct_from_price, max_sl_pct)
            
            # 1) Попытка вытащить POC из orderflow‑reason (lh_of_*_poc_X)
            poc_from_reason = None
            reason = getattr(sig, "reason", "") or ""
            if reason.startswith("lh_of_") and "_poc_" in reason:
                try:
                    poc_part = reason.split("_poc_")[-1]
                    poc_from_reason = float(poc_part)
                except Exception:
                    poc_from_reason = None
            
            # 2) Базовые TP/SL (SR‑уровни или проценты)
            if use_sr_levels:
                if sig.action == Action.LONG:
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_tp_pct = (nearest_resistance - entry_price) / entry_price
                        take_profit = nearest_resistance if resistance_tp_pct <= tp_pct_from_price else entry_price * (1 + tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 + tp_pct_from_price)
                    
                    if nearest_support and nearest_support < entry_price:
                        support_sl_pct = (entry_price - nearest_support) / entry_price
                        stop_loss = nearest_support if support_sl_pct <= sl_pct_from_price else entry_price * (1 - sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    if nearest_support and nearest_support < entry_price:
                        support_tp_pct = (entry_price - nearest_support) / entry_price
                        take_profit = nearest_support if support_tp_pct <= tp_pct_from_price else entry_price * (1 - tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 - tp_pct_from_price)
                    
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_sl_pct = (nearest_resistance - entry_price) / entry_price
                        stop_loss = nearest_resistance if resistance_sl_pct <= sl_pct_from_price else entry_price * (1 + sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 + sl_pct_from_price)
            else:
                if sig.action == Action.LONG:
                    take_profit = entry_price * (1 + tp_pct_from_price)
                    stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    take_profit = entry_price * (1 - tp_pct_from_price)
                    stop_loss = entry_price * (1 + sl_pct_from_price)

            # 3) Если это orderflow‑сигнал и POC известен — переопределяем TP = POC
            if poc_from_reason is not None:
                take_profit = poc_from_reason
            
            # Логируем расчет
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f"[live] 📊 LIQUIDATION_HUNTER TP/SL: TP=${take_profit:.2f} (+{((take_profit - entry_price) / entry_price * 100):.2f}%), SL=${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price * 100):.2f}%), RR={rr_ratio:.2f}:1")
            print(f"[live]   → TP: {tp_pct_from_price*100:.2f}% from price ({tp_pct_from_price*leverage*100:.1f}% from margin), SL: {sl_pct_from_price*100:.2f}% from price ({sl_pct_from_price*leverage*100:.1f}% from margin)")
            
            return take_profit, stop_loss
        
        elif strategy_type == "vbo":
            # Для VBO стратегии (Volatility Breakout) используем сбалансированные TP/SL
            # VBO ловит пробои волатильности, но слишком широкий TP может привести к закрытию по SL
            # Нужен баланс: достаточно широкий TP для движения, но не слишком, чтобы не терять по SL
            # Рекомендуемые параметры: TP 3.0-3.5% от цены, SL 1.0-1.2% от цены (RR ~2.5-3:1)
            
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            
            # Для VBO используем сбалансированные уровни для пробоев волатильности
            # TP: 3.2% от цены (32% от маржи при 10x) - достаточно широкий, но реалистичный
            # SL: 1.1% от цены (11% от маржи при 10x) - дает больше пространства для пробоя
            # RR: ~2.9:1 - сбалансированное соотношение для пробоев
            
            tp_pct_from_price = 0.032  # 3.2% от цены = 32% от маржи при 10x
            sl_pct_from_price = 0.011   # 1.1% от цены = 11% от маржи при 10x (немного больше для пробоев)
            
            # Проверяем максимальные границы из настроек
            max_tp_pct_margin = settings.risk.take_profit_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'take_profit_pct') else 0.30
            max_sl_pct_margin = settings.risk.stop_loss_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'stop_loss_pct') else 0.15
            
            # Нормализуем проценты
            if max_tp_pct_margin > 1.0:
                max_tp_pct_margin = max_tp_pct_margin / 100.0
            if max_sl_pct_margin > 1.0:
                max_sl_pct_margin = max_sl_pct_margin / 100.0
            
            max_tp_pct = max_tp_pct_margin / leverage
            max_sl_pct = max_sl_pct_margin / leverage
            
            # Ограничиваем нашими значениями, но не превышаем максимумы
            tp_pct_from_price = min(tp_pct_from_price, max_tp_pct)
            # Для SL: минимум 7% от маржи, максимум 10% от маржи (или max_sl_pct_margin, если меньше)
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = min(0.10, max_sl_pct_margin)  # Максимум 10% от маржи или настройка
            min_sl_pct_from_price = min_sl_pct_from_margin / leverage
            max_sl_pct_from_price = max_sl_pct_from_margin / leverage
            
            # Убеждаемся, что SL в допустимом диапазоне (0.9% = 9% от маржи при 10x - в пределах 7-10%)
            sl_pct_from_price = max(min_sl_pct_from_price, min(sl_pct_from_price, max_sl_pct_from_price))
            
            # Используем уровни поддержки/сопротивления, если они находятся в пределах наших параметров
            if use_sr_levels:
                if sig.action == Action.LONG:
                    # Для LONG: TP на сопротивление, SL на поддержку
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_tp_pct = (nearest_resistance - entry_price) / entry_price
                        if resistance_tp_pct <= tp_pct_from_price and resistance_tp_pct >= tp_pct_from_price * 0.5:  # Не слишком близко
                            take_profit = nearest_resistance
                        else:
                            take_profit = entry_price * (1 + tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 + tp_pct_from_price)
                    
                    if nearest_support and nearest_support < entry_price:
                        support_sl_pct = (entry_price - nearest_support) / entry_price
                        if support_sl_pct <= sl_pct_from_price and support_sl_pct >= min_sl_pct_from_price:
                            stop_loss = nearest_support
                        else:
                            stop_loss = entry_price * (1 - sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    # Для SHORT: TP на поддержку, SL на сопротивление
                    if nearest_support and nearest_support < entry_price:
                        support_tp_pct = (entry_price - nearest_support) / entry_price
                        if support_tp_pct <= tp_pct_from_price and support_tp_pct >= tp_pct_from_price * 0.5:  # Не слишком близко
                            take_profit = nearest_support
                        else:
                            take_profit = entry_price * (1 - tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 - tp_pct_from_price)
                    
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_sl_pct = (nearest_resistance - entry_price) / entry_price
                        if resistance_sl_pct <= sl_pct_from_price and resistance_sl_pct >= min_sl_pct_from_price:
                            stop_loss = nearest_resistance
                        else:
                            stop_loss = entry_price * (1 + sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 + sl_pct_from_price)
            else:
                # Fallback на фиксированные проценты
                if sig.action == Action.LONG:
                    take_profit = entry_price * (1 + tp_pct_from_price)
                    stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    take_profit = entry_price * (1 - tp_pct_from_price)
                    stop_loss = entry_price * (1 + sl_pct_from_price)
            
            # Логируем расчет
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f"[live] 📊 VBO TP/SL: TP=${take_profit:.2f} (+{((take_profit - entry_price) / entry_price * 100):.2f}%), SL=${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price * 100):.2f}%), RR={rr_ratio:.2f}:1")
            print(f"[live]   → TP: {tp_pct_from_price*100:.2f}% from price ({tp_pct_from_price*leverage*100:.1f}% from margin), SL: {sl_pct_from_price*100:.2f}% from price ({sl_pct_from_price*leverage*100:.1f}% from margin)")
            
            return take_profit, stop_loss
        
        elif strategy_type == "zscore":
            # Для ZSCORE стратегии (Mean Reversion) базовые TP/SL + режим TP=POC (Volume Profile)
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            
            tp_pct_from_price = 0.030  # 3.0% от цены
            sl_pct_from_price = 0.010  # 1.0% от цены
            
            # Проверяем максимальные границы из настроек
            max_tp_pct_margin = settings.risk.take_profit_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'take_profit_pct') else 0.30
            max_sl_pct_margin = settings.risk.stop_loss_pct if hasattr(settings, 'risk') and hasattr(settings.risk, 'stop_loss_pct') else 0.15
            
            # Нормализуем проценты
            if max_tp_pct_margin > 1.0:
                max_tp_pct_margin = max_tp_pct_margin / 100.0
            if max_sl_pct_margin > 1.0:
                max_sl_pct_margin = max_sl_pct_margin / 100.0
            
            max_tp_pct = max_tp_pct_margin / leverage
            max_sl_pct = max_sl_pct_margin / leverage
            
            # Ограничиваем нашими значениями, но не превышаем максимумы
            tp_pct_from_price = min(tp_pct_from_price, max_tp_pct)
            # Для SL: минимум 7% от маржи, максимум 10% от маржи (или max_sl_pct_margin, если меньше)
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = min(0.10, max_sl_pct_margin)  # Максимум 10% от маржи или настройка
            min_sl_pct_from_price = min_sl_pct_from_margin / leverage
            max_sl_pct_from_price = max_sl_pct_from_margin / leverage
            
            # Убеждаемся, что SL в допустимом диапазоне (0.8% = 8% от маржи при 10x - в пределах 7-10%)
            sl_pct_from_price = max(min_sl_pct_from_price, min(sl_pct_from_price, max_sl_pct_from_price))
            
            # Базовые TP/SL (SR-уровни или проценты)
            if use_sr_levels:
                if sig.action == Action.LONG:
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_tp_pct = (nearest_resistance - entry_price) / entry_price
                        if resistance_tp_pct <= tp_pct_from_price and resistance_tp_pct >= tp_pct_from_price * 0.5:
                            take_profit = nearest_resistance
                        else:
                            take_profit = entry_price * (1 + tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 + tp_pct_from_price)
                    
                    if nearest_support and nearest_support < entry_price:
                        support_sl_pct = (entry_price - nearest_support) / entry_price
                        if support_sl_pct <= sl_pct_from_price and support_sl_pct >= min_sl_pct_from_price:
                            stop_loss = nearest_support
                        else:
                            stop_loss = entry_price * (1 - sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    if nearest_support and nearest_support < entry_price:
                        support_tp_pct = (entry_price - nearest_support) / entry_price
                        if support_tp_pct <= tp_pct_from_price and support_tp_pct >= tp_pct_from_price * 0.5:
                            take_profit = nearest_support
                        else:
                            take_profit = entry_price * (1 - tp_pct_from_price)
                    else:
                        take_profit = entry_price * (1 - tp_pct_from_price)
                    
                    if nearest_resistance and nearest_resistance > entry_price:
                        resistance_sl_pct = (nearest_resistance - entry_price) / entry_price
                        if resistance_sl_pct <= sl_pct_from_price and resistance_sl_pct >= min_sl_pct_from_price:
                            stop_loss = nearest_resistance
                        else:
                            stop_loss = entry_price * (1 + sl_pct_from_price)
                    else:
                        stop_loss = entry_price * (1 + sl_pct_from_price)
            else:
                if sig.action == Action.LONG:
                    take_profit = entry_price * (1 + tp_pct_from_price)
                    stop_loss = entry_price * (1 - sl_pct_from_price)
                else:  # SHORT
                    take_profit = entry_price * (1 - tp_pct_from_price)
                    stop_loss = entry_price * (1 + sl_pct_from_price)

            # Если в reason зашит POC (из блока генерации сигналов) – используем его как TP
            poc_from_reason = None
            reason_str = getattr(sig, "reason", "") or ""
            if "_poc_" in reason_str:
                try:
                    poc_part = reason_str.split("_poc_")[-1]
                    poc_from_reason = float(poc_part)
                except Exception:
                    poc_from_reason = None
            if poc_from_reason is not None:
                take_profit = poc_from_reason

            # Попытка переопределить TP по POC из Volume Profile (AMT-логика).
            # Здесь df_ready недоступен, поэтому фактический TP=POC рассчитывается на этапе позиционного менеджмента.
            
            # Логируем расчет
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f"[live] 📊 ZSCORE TP/SL: TP=${take_profit:.2f} (+{((take_profit - entry_price) / entry_price * 100):.2f}%), SL=${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price * 100):.2f}%), RR={rr_ratio:.2f}:1")
            print(f"[live]   → TP: {tp_pct_from_price*100:.2f}% from price ({tp_pct_from_price*leverage*100:.1f}% from margin), SL: {sl_pct_from_price*100:.2f}% from price ({sl_pct_from_price*leverage*100:.1f}% from margin)")
            
            return take_profit, stop_loss
            
        else:
            # Для TREND/FLAT стратегий используем настройки как МАКСИМАЛЬНЫЕ границы
            # Бот сам определяет TP/SL на основе уровней поддержки/сопротивления
            # с соотношением риска 2-3:1 в пределах этих границ
            # ВАЖНО: Проценты интерпретируются как проценты от МАРЖИ с учетом плеча!
            max_tp_pct_margin = settings.risk.take_profit_pct  # Максимальный TP от маржи (например, 0.30 для 30%)
            max_sl_pct_margin = settings.risk.stop_loss_pct    # Максимальный SL от маржи (например, 0.15 для 15%)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Если проценты > 1.0 (100%), вероятно они не разделены на 100
            if max_tp_pct_margin > 1.0:
                print(f"[live] 🚨 CRITICAL: take_profit_pct={max_tp_pct_margin} is > 1.0 (100%)! Dividing by 100.")
                max_tp_pct_margin = max_tp_pct_margin / 100.0
            if max_sl_pct_margin > 1.0:
                print(f"[live] 🚨 CRITICAL: stop_loss_pct={max_sl_pct_margin} is > 1.0 (100%)! Dividing by 100.")
                max_sl_pct_margin = max_sl_pct_margin / 100.0
            
            # Преобразуем проценты от маржи в проценты от цены: / leverage
            max_tp_pct = max_tp_pct_margin / settings.leverage
            max_sl_pct = max_sl_pct_margin / settings.leverage
            
            min_rr_ratio = 2.0  # Минимальное соотношение риска 2:1
            max_rr_ratio = 3.0  # Максимальное соотношение риска 3:1
            
            # Вычисляем границы (теперь в процентах от цены)
            if sig.action == Action.LONG:
                max_tp_price = entry_price * (1 + max_tp_pct)
                max_sl_price = entry_price * (1 - max_sl_pct)
                
                # ПРИОРИТЕТ 1: Используем уровни поддержки/сопротивления, если они найдены
                if use_sr_levels and nearest_resistance and nearest_support:
                    # Используем уровни, но ограничиваем границами
                    tp_from_level = min(nearest_resistance, max_tp_price) if nearest_resistance > entry_price else max_tp_price
                    sl_from_level = max(nearest_support, max_sl_price) if nearest_support < entry_price else max_sl_price
                    
                    # Проверяем соотношение риска
                    risk = entry_price - sl_from_level
                    reward = tp_from_level - entry_price
                    
                    if risk > 0:
                        current_rr = reward / risk
                        
                        # Если RR < 2, пытаемся увеличить TP или уменьшить SL (в пределах границ)
                        if current_rr < min_rr_ratio:
                            # Пытаемся увеличить TP до достижения RR = 2.5
                            target_tp = entry_price + (risk * 2.5)
                            if target_tp <= max_tp_price:
                                tp_from_level = target_tp
                                current_rr = 2.5
                            else:
                                # Если TP на максимуме, уменьшаем SL
                                target_sl = entry_price - (reward / 2.5)
                                if target_sl >= max_sl_price:
                                    sl_from_level = target_sl
                                    current_rr = 2.5
                        
                        # Если RR > 3, можно немного уменьшить TP для оптимизации (опционально)
                        elif current_rr > max_rr_ratio:
                            target_tp = entry_price + (risk * 2.5)
                            if target_tp >= entry_price * 1.01:  # Минимум 1% прибыли
                                tp_from_level = target_tp
                                current_rr = 2.5
                        
                        take_profit = tp_from_level
                        stop_loss = sl_from_level
                        print(f"[live] 📊 TP/SL from levels: TP=${take_profit:.2f}, SL=${stop_loss:.2f}, RR={current_rr:.2f}:1")
                    else:
                        # Если risk <= 0, используем настройки
                        take_profit = max_tp_price
                        stop_loss = max_sl_price
                        print(f"[live] ⚠️ Invalid levels, using max settings: TP=${take_profit:.2f}, SL=${stop_loss:.2f}")
                else:
                    # ПРИОРИТЕТ 2: Если уровни не найдены, используем настройки с проверкой RR
                    take_profit = max_tp_price
                    stop_loss = max_sl_price
                    
                    risk = entry_price - stop_loss
                    reward = take_profit - entry_price
                    
                    if risk > 0:
                        current_rr = reward / risk
                        # Корректируем для достижения RR 2-3:1
                        if current_rr < min_rr_ratio:
                            # Увеличиваем TP
                            target_tp = entry_price + (risk * 2.5)
                            if target_tp <= max_tp_price:
                                take_profit = target_tp
                        elif current_rr > max_rr_ratio:
                            # Уменьшаем TP до RR = 2.5
                            take_profit = entry_price + (risk * 2.5)
                    
                    print(f"[live] 📊 TP/SL from settings (no levels): TP=${take_profit:.2f}, SL=${stop_loss:.2f}, RR={reward/risk:.2f}:1")
                
            else:  # SHORT
                max_tp_price = entry_price * (1 - max_tp_pct)  # Для SHORT TP ниже entry
                max_sl_price = entry_price * (1 + max_sl_pct)  # Для SHORT SL выше entry
                
                # ПРИОРИТЕТ 1: Используем уровни поддержки/сопротивления
                if use_sr_levels and nearest_resistance and nearest_support:
                    # Используем уровни, но ограничиваем границами
                    tp_from_level = max(nearest_support, max_tp_price) if nearest_support < entry_price else max_tp_price
                    sl_from_level = min(nearest_resistance, max_sl_price) if nearest_resistance > entry_price else max_sl_price
                    
                    # Проверяем соотношение риска
                    risk = sl_from_level - entry_price
                    reward = entry_price - tp_from_level
                    
                    if risk > 0:
                        current_rr = reward / risk
                        
                        # Если RR < 2, пытаемся увеличить TP или уменьшить SL
                        if current_rr < min_rr_ratio:
                            target_tp = entry_price - (risk * 2.5)
                            if target_tp >= max_tp_price:
                                tp_from_level = target_tp
                                current_rr = 2.5
                            else:
                                target_sl = entry_price + (reward / 2.5)
                                if target_sl <= max_sl_price:
                                    sl_from_level = target_sl
                                    current_rr = 2.5
                        
                        # Если RR > 3, оптимизируем
                        elif current_rr > max_rr_ratio:
                            target_tp = entry_price - (risk * 2.5)
                            if target_tp <= entry_price * 0.99:  # Минимум 1% прибыли
                                tp_from_level = target_tp
                                current_rr = 2.5
                        
                        take_profit = tp_from_level
                        stop_loss = sl_from_level
                        print(f"[live] 📊 TP/SL from levels: TP=${take_profit:.2f}, SL=${stop_loss:.2f}, RR={current_rr:.2f}:1")
                    else:
                        take_profit = max_tp_price
                        stop_loss = max_sl_price
                        print(f"[live] ⚠️ Invalid levels, using max settings: TP=${take_profit:.2f}, SL=${stop_loss:.2f}")
                else:
                    # ПРИОРИТЕТ 2: Используем настройки с проверкой RR
                    take_profit = max_tp_price
                    stop_loss = max_sl_price
                    
                    risk = stop_loss - entry_price
                    reward = entry_price - take_profit
                    
                    if risk > 0:
                        current_rr = reward / risk
                        if current_rr < min_rr_ratio:
                            target_tp = entry_price - (risk * 2.5)
                            if target_tp >= max_tp_price:
                                take_profit = target_tp
                        elif current_rr > max_rr_ratio:
                            take_profit = entry_price - (risk * 2.5)
                    
                    print(f"[live] 📊 TP/SL from settings (no levels): TP=${take_profit:.2f}, SL=${stop_loss:.2f}, RR={reward/risk:.2f}:1")
            
            # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что SL находится в диапазоне 7-10% от маржи
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
            
            if sig.action == Action.LONG:
                sl_deviation_pct_from_price = abs(entry_price - stop_loss) / entry_price
            else:  # SHORT
                sl_deviation_pct_from_price = abs(stop_loss - entry_price) / entry_price
            
            sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
            
            # Если SL меньше 7% от маржи, увеличиваем до минимума
            if sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                target_sl_pct_from_price = min_sl_pct_from_margin / leverage
                if sig.action == Action.LONG:
                    stop_loss = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    stop_loss = entry_price * (1 + target_sl_pct_from_price)
                print(f"[live] ⚠️ SL too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusted to {min_sl_pct_from_margin*100:.0f}% from margin ({target_sl_pct_from_price*100:.2f}% from price)")
            # Если SL больше 10% от маржи, уменьшаем до максимума
            elif sl_deviation_pct_from_margin > max_sl_pct_from_margin:
                target_sl_pct_from_price = max_sl_pct_from_margin / leverage
                if sig.action == Action.LONG:
                    stop_loss = entry_price * (1 - target_sl_pct_from_price)
                else:  # SHORT
                    stop_loss = entry_price * (1 + target_sl_pct_from_price)
                print(f"[live] ⚠️ SL too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusted to {max_sl_pct_from_margin*100:.0f}% from margin ({target_sl_pct_from_price*100:.2f}% from price)")
            else:
                print(f"[live] ✅ SL is within range: {sl_deviation_pct_from_margin*100:.1f}% from margin ({sl_deviation_pct_from_price*100:.2f}% from price)")
            
            return take_profit, stop_loss
            
    except Exception as e:
        print(f"[live] ⚠️ Error calculating TP/SL for signal: {e}")
        return None, None


def _update_position_tracking(
    position: Dict[str, Any],
    position_bias: Bias,
    current_price: float,
    position_max_profit: Dict[str, float],
    position_max_price: Dict[str, float],
    symbol: str,
) -> None:
    """
    Обновляет отслеживание максимальной прибыли и цены для позиции.
    
    Args:
        position: Информация о позиции
        position_bias: Направление позиции
        current_price: Текущая цена
        position_max_profit: Словарь для хранения максимальной прибыли {symbol: max_profit_pct}
        position_max_price: Словарь для хранения максимальной цены {symbol: max_price}
        symbol: Торговая пара
    """
    try:
        avg_price = position.get("avg_price", 0)
        if avg_price == 0:
            return
        
        # Рассчитываем текущую прибыль в процентах
        if position_bias == Bias.LONG:
            profit_pct = ((current_price - avg_price) / avg_price) * 100
        else:  # SHORT
            profit_pct = ((avg_price - current_price) / avg_price) * 100
        
        # Обновляем максимальную прибыль
        if symbol not in position_max_profit or profit_pct > position_max_profit[symbol]:
            position_max_profit[symbol] = profit_pct
            position_max_price[symbol] = current_price
    except Exception as e:
        print(f"[live] Error updating position tracking: {e}")


def _ensure_tp_sl_set(
    client: BybitClient,
    position: Dict[str, Any],
    settings: AppSettings,
    position_bias: Bias,
    current_price: float,
    position_max_profit: Dict[str, float],
    position_max_price: Dict[str, float],
) -> None:
    """
    Проверяет и устанавливает TP/SL для открытой позиции с поддержкой trailing stop, безубытка и защиты прибыли.
    
    Args:
        client: Bybit клиент
        position: Информация о позиции
        settings: Настройки бота
        position_bias: Направление позиции (LONG или SHORT)
        current_price: Текущая цена
        position_max_profit: Словарь максимальной прибыли {symbol: max_profit_pct}
        position_max_price: Словарь максимальной цены {symbol: max_price}
    """
    try:
        avg_price = position.get("avg_price", 0)
        if avg_price == 0:
            return
        
        symbol = settings.symbol
        
        # Обновляем отслеживание максимальной прибыли
        _update_position_tracking(position, position_bias, current_price, position_max_profit, position_max_price, symbol)
        
        # Получаем текущую прибыль
        max_profit_pct = position_max_profit.get(symbol, 0.0)
        max_price = position_max_price.get(symbol, current_price)

        # Определяем стратегию позиции (по entry_reason из истории, если есть)
        position_strategy_type = None
        try:
            from bot.web.history import get_open_trade
            open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
            if open_trade:
                entry_reason = open_trade.get("entry_reason", "")
                if entry_reason:
                    position_strategy_type = get_strategy_type_from_signal(entry_reason)
        except Exception:
            position_strategy_type = None
        
        # Проверяем, установлены ли TP/SL
        current_tp = position.get("take_profit", "")
        current_sl = position.get("stop_loss", "")
        tp_set = current_tp and current_tp != "" and str(current_tp).strip() != ""
        sl_set = current_sl and current_sl != "" and str(current_sl).strip() != ""
        
        # КРИТИЧЕСКАЯ ПРОВЕРКА: Обнаружение аномальных TP/SL (более 500% от entry price)
        # Это может произойти, если TP/SL от другой монеты (например, BTC цена на ETH позиции)
        tp_is_anomalous = False
        sl_is_anomalous = False
        
        if tp_set and avg_price > 0:
            try:
                current_tp_val = float(current_tp)
                tp_deviation_pct = abs(current_tp_val - avg_price) / avg_price * 100
                if tp_deviation_pct > 500:  # Более 500% отклонение - явно ошибка
                    print(f"[live] 🚨 ANOMALY DETECTED: Current TP=${current_tp_val:.2f} is {tp_deviation_pct:.0f}% away from entry ${avg_price:.2f}")
                    print(f"[live]   This looks like a TP from another asset (e.g., BTC price on ETH position)")
                    print(f"[live]   Will FORCE reset TP to correct value")
                    tp_is_anomalous = True
                    tp_set = False  # Считаем как не установленный
            except (ValueError, TypeError):
                pass
        
        if sl_set and avg_price > 0:
            try:
                current_sl_val = float(current_sl)
                sl_deviation_pct = abs(current_sl_val - avg_price) / avg_price * 100
                
                # Проверяем, что SL находится в разумных пределах (не более 50% от entry)
                # Для LONG: SL должен быть ниже entry, отклонение не должно быть > 50%
                # Для SHORT: SL должен быть выше entry, отклонение не должно быть > 50%
                is_sl_reasonable = False
                if position_bias == Bias.LONG:
                    # Для LONG: SL должен быть ниже entry
                    if current_sl_val < avg_price and sl_deviation_pct <= 50:
                        is_sl_reasonable = True
                else:  # SHORT
                    # Для SHORT: SL должен быть выше entry
                    if current_sl_val > avg_price and sl_deviation_pct <= 50:
                        is_sl_reasonable = True
                
                if not is_sl_reasonable or sl_deviation_pct > 500:
                    print(f"[live] 🚨 ANOMALY DETECTED: Current SL=${current_sl_val:.2f} is {sl_deviation_pct:.0f}% away from entry ${avg_price:.2f}")
                    print(f"[live]   This looks like an incorrect SL value (should be within 50% of entry)")
                    print(f"[live]   Will FORCE reset SL to correct value")
                    sl_is_anomalous = True
                    sl_set = False  # Считаем как не установленный
            except (ValueError, TypeError):
                pass
        
        # Получаем entry_reason из истории для определения стратегии, которая открыла позицию
        entry_reason = None
        try:
            from bot.web.history import get_open_trade
            open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
            if open_trade:
                entry_reason = open_trade.get("entry_reason", "")
                if entry_reason:
                    print(f"[live] 📊 Found entry_reason from history: '{entry_reason}' for position @ ${avg_price:.2f}")
        except Exception as e:
            print(f"[live] ⚠️ Error getting entry_reason from history: {e}")
        
        # Определяем стратегию на основе entry_reason, если доступен, иначе используем настройки
        # Создаем фиктивный Signal для использования _calculate_tp_sl_for_signal
        fake_signal = None
        use_strategy_tp_sl = False
        strategy_tp_sl_applied = False  # Флаг успешного применения стратегических TP/SL
        
        if entry_reason:
            try:
                # Определяем action на основе position_bias
                from bot.strategy import Signal, Action
                import pandas as pd
                
                fake_action = Action.LONG if position_bias == Bias.LONG else Action.SHORT
                fake_timestamp = pd.Timestamp.now()
                fake_signal = Signal(
                    timestamp=fake_timestamp,
                    action=fake_action,
                    reason=entry_reason,
                    price=avg_price,
                )
                use_strategy_tp_sl = True
                print(f"[live] 📊 Using strategy-specific TP/SL based on entry_reason: '{entry_reason}'")
            except Exception as e:
                print(f"[live] ⚠️ Error creating fake signal from entry_reason: {e}")
        
        # Если entry_reason не найден или не удалось создать Signal, используем общую логику
        if not use_strategy_tp_sl:
            # Определяем, какая стратегия используется для расчета TP/SL
            # Используем приоритет стратегий из настроек для определения, какие TP/SL применять
            # Если ML стратегия включена и имеет приоритет, используем ML TP/SL
            # Иначе используем TREND/FLAT TP/SL
            use_ml_tp_sl = False
            if settings.enable_ml_strategy and settings.ml_model_path:
                # Проверяем приоритет стратегий
                strategy_priority = getattr(settings, 'strategy_priority', 'trend')
                if strategy_priority == "ml":
                    use_ml_tp_sl = True
                elif strategy_priority == "hybrid" and (settings.enable_trend_strategy or settings.enable_flat_strategy):
                    # В гибридном режиме используем ML TP/SL, если ML стратегия включена
                    use_ml_tp_sl = True
                elif not (settings.enable_trend_strategy or settings.enable_flat_strategy):
                    # Если только ML стратегия включена, используем ML TP/SL
                    use_ml_tp_sl = True
        else:
            # Если уже используем стратегические TP/SL на основе entry_reason, не переключаемся на ML TP/SL
            use_ml_tp_sl = False
        
        # Используем стратегические TP/SL, если entry_reason найден
        if use_strategy_tp_sl and fake_signal:
            try:
                # Используем _calculate_tp_sl_for_signal для расчета TP/SL на основе стратегии
                # df_data можно передать как None, так как для существующих позиций уровни S/R не так важны
                calculated_tp, calculated_sl = _calculate_tp_sl_for_signal(
                    sig=fake_signal,
                    settings=settings,
                    entry_price=avg_price,
                    df_data=None,  # Можно передать df_data, если доступен
                )
                
                if calculated_tp and calculated_sl:
                    base_tp = calculated_tp
                    base_sl = calculated_sl
                    
                    # Определяем название стратегии из entry_reason
                    if entry_reason.startswith("ml_"):
                        strategy_name = "ML"
                    elif entry_reason.startswith("liquidation_hunter_"):
                        strategy_name = "LIQUIDATION_HUNTER"
                    elif entry_reason.startswith("zscore_"):
                        strategy_name = "ZSCORE"
                    elif entry_reason.startswith("vbo_"):
                        strategy_name = "VBO"
                    elif entry_reason.startswith("ict_"):
                        strategy_name = "ICT"
                    elif entry_reason.startswith("smc_"):
                        strategy_name = "SMC"
                    elif entry_reason.startswith("trend_"):
                        strategy_name = "TREND"
                    elif entry_reason.startswith("range_"):
                        strategy_name = "FLAT"
                    else:
                        strategy_name = "UNKNOWN"
                    
                    print(f"[live] 📊 {strategy_name} TP/SL from entry_reason: TP=${base_tp:.2f}, SL=${base_sl:.2f} (entry: ${avg_price:.2f})")
                    print(f"[live] ✅ Strategy-specific TP/SL calculated and set to base_tp/base_sl")
                    # ВАЖНО: Устанавливаем флаг, что стратегические TP/SL успешно применены
                    # Это предотвратит перезапись base_tp/base_sl в блоке ниже
                    strategy_tp_sl_applied = True
                    print(f"[live] ✅ Flag set: strategy_tp_sl_applied={strategy_tp_sl_applied} - will skip default TP/SL calculation")
                else:
                    # Если _calculate_tp_sl_for_signal не вернул значения, используем общую логику
                    print(f"[live] ⚠️ _calculate_tp_sl_for_signal returned None, falling back to default TP/SL")
                    use_strategy_tp_sl = False
            except Exception as e:
                print(f"[live] ⚠️ Error calculating strategy-specific TP/SL: {e}")
                use_strategy_tp_sl = False
        
        # Если не используем стратегические TP/SL, применяем общую логику
        # ВАЖНО: Этот блок выполняется ТОЛЬКО если:
        # 1. Стратегия не определена (entry_reason отсутствует или не удалось создать Signal)
        # 2. Или стратегические TP/SL не были успешно рассчитаны (strategy_tp_sl_applied = False)
        # Если стратегические TP/SL успешно применены (strategy_tp_sl_applied = True),
        # этот блок НЕ выполняется, и base_tp/base_sl остаются со стратегическими значениями
        print(f"[live] 🔍 Checking strategy_tp_sl_applied: {strategy_tp_sl_applied}")
        if not strategy_tp_sl_applied:
            print(f"[live] 🔄 Using default TP/SL calculation (strategy not defined or strategy TP/SL calculation failed)")
            # Определяем, какая стратегия используется для расчета TP/SL
            # Используем приоритет стратегий из настроек для определения, какие TP/SL применять
            # Если ML стратегия включена и имеет приоритет, используем ML TP/SL
            # Иначе используем TREND/FLAT TP/SL
            use_ml_tp_sl = False
        else:
            print(f"[live] ✅ Skipping default TP/SL calculation - strategy-specific TP/SL already applied")
            use_ml_tp_sl = False  # Не используем ML TP/SL, так как стратегические уже применены
        
        # ВАЖНО: Блок ML/TREND/FLAT TP/SL выполняется ТОЛЬКО если стратегические TP/SL НЕ были применены
        if not strategy_tp_sl_applied and not use_strategy_tp_sl and use_ml_tp_sl:
            # ML стратегия: используем специальные TP/SL для прибыли от маржи
            # ml_target_profit_pct_margin и ml_max_loss_pct_margin уже в процентах (например, 25.0 для 25%)
            # Нужно перевести в доли от цены: / leverage / 100
            tp_pct_margin = settings.ml_target_profit_pct_margin  # Например, 25.0%
            sl_pct_margin = settings.ml_max_loss_pct_margin  # Например, 10.0%
            
            # Убрано verbose сообщение о входных параметрах ML TP/SL
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: SL должен быть в диапазоне 7-10% от маржи ПЕРЕД расчетом
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
            
            # Проверяем и корректируем ml_max_loss_pct_margin ДО расчета sl_pct
            if sl_pct_margin < min_sl_pct_from_margin * 100:
                print(f"[live] 🚨 CRITICAL: ML SL from margin ({sl_pct_margin}%) < {min_sl_pct_from_margin*100:.0f}%, adjusting to {min_sl_pct_from_margin*100:.0f}%")
                sl_pct_margin = min_sl_pct_from_margin * 100
            elif sl_pct_margin > max_sl_pct_from_margin * 100:
                print(f"[live] 🚨 CRITICAL: ML SL from margin ({sl_pct_margin}%) > {max_sl_pct_from_margin*100:.0f}%, adjusting to {max_sl_pct_from_margin*100:.0f}%")
                sl_pct_margin = max_sl_pct_from_margin * 100
            
            # Переводим проценты от маржи в проценты от цены
            # Если leverage = 10, то 25% от маржи = 2.5% от цены
            tp_pct = tp_pct_margin / settings.leverage / 100.0
            sl_pct = sl_pct_margin / settings.leverage / 100.0
            
            # Убрано verbose сообщение о корректировке SL
            
            # МИНИМАЛЬНЫЕ ПОРОГИ: гарантируем, что TP не равен нулю
            # Минимум 0.5% для TP (от цены)
            min_tp_pct = 0.005  # 0.5%
            
            if tp_pct < min_tp_pct:
                print(f"[live] ⚠️ WARNING: ML TP percentage ({tp_pct*100:.4f}%) too small, using minimum {min_tp_pct*100:.2f}%")
                print(f"[live]   ml_target_profit_pct_margin={tp_pct_margin}%, leverage={settings.leverage}")
                tp_pct = min_tp_pct
            
            # SL уже проверен и скорректирован выше (7-10% от маржи)
            # Дополнительная проверка: убеждаемся, что sl_pct соответствует диапазону
            min_sl_pct_from_price = min_sl_pct_from_margin / settings.leverage  # Минимум от цены (0.7% при 10x)
            max_sl_pct_from_price = max_sl_pct_from_margin / settings.leverage   # Максимум от цены (1.0% при 10x)
            
            if sl_pct < min_sl_pct_from_price:
                print(f"[live] 🚨 CRITICAL: ML SL percentage ({sl_pct*100:.4f}%) still too small after adjustment, forcing to {min_sl_pct_from_margin*100:.0f}% from margin ({min_sl_pct_from_price*100:.2f}% from price)")
                sl_pct = min_sl_pct_from_price
            elif sl_pct > max_sl_pct_from_price:
                print(f"[live] 🚨 CRITICAL: ML SL percentage ({sl_pct*100:.4f}%) still too large after adjustment, forcing to {max_sl_pct_from_margin*100:.0f}% from margin ({max_sl_pct_from_price*100:.2f}% from price)")
                sl_pct = max_sl_pct_from_price
            
            # Убрали детальное логирование TP/SL - слишком много сообщений
            # print(f"[live] 📊 ML TP/SL calculation: margin_tp={tp_pct_margin}%, margin_sl={sl_pct_margin}%, leverage={settings.leverage}")
            # print(f"[live]   → price_tp={tp_pct*100:.2f}%, price_sl={sl_pct*100:.2f}%")
            # print(f"[live]   → SL: {sl_pct_margin}% from margin = {sl_pct*100:.2f}% from price")
            
            if position_bias == Bias.LONG:
                base_tp = avg_price * (1 + tp_pct)
                base_sl = avg_price * (1 - sl_pct)
            else:  # SHORT
                base_tp = avg_price * (1 - tp_pct)
                base_sl = avg_price * (1 + sl_pct)
            
            # Убрали детальное логирование TP/SL - слишком много сообщений
            # print(f"[live]   → base_tp=${base_tp:.2f}, base_sl=${base_sl:.2f} (entry: ${avg_price:.2f})")
            
            strategy_name = "ML"
        elif not strategy_tp_sl_applied:
            # Обычные стратегии: используем стандартные TP/SL
            # ВАЖНО: Этот блок выполняется ТОЛЬКО если стратегические TP/SL НЕ были применены
            # ВАЖНО: Проценты интерпретируются как проценты от МАРЖИ с учетом плеча, а не от цены входа!
            # Формула: TP = Entry * (1 + take_profit_pct / Leverage)
            # Например: Entry=$3128.84, Leverage=10x, TP=30% от маржи
            #   → TP = $3128.84 * (1 + 0.30 / 10) = $3128.84 * 1.03 = $3222.71 (3% от цены = 30% от маржи)
            print(f"[live] 📊 TREND/FLAT TP/SL calculation (from MARGIN %):")
            print(f"[live]   take_profit_pct={settings.risk.take_profit_pct:.6f} ({settings.risk.take_profit_pct*100:.2f}% of margin)")
            print(f"[live]   stop_loss_pct={settings.risk.stop_loss_pct:.6f} ({settings.risk.stop_loss_pct*100:.2f}% of margin)")
            print(f"[live]   leverage={settings.leverage}x")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Убеждаемся, что проценты в правильном формате (доли, не проценты)
            # Если take_profit_pct > 1.0 (100%), это явная ошибка - должно быть < 1.0 (например, 0.30 для 30%)
            if settings.risk.take_profit_pct > 1.0:
                print(f"[live] 🚨 ERROR: take_profit_pct={settings.risk.take_profit_pct:.6f} ({settings.risk.take_profit_pct*100:.2f}%) is > 100%!")
                print(f"[live]   This is definitely wrong. Dividing by 100 to correct...")
                settings.risk.take_profit_pct = settings.risk.take_profit_pct / 100.0
                print(f"[live]   Corrected to: {settings.risk.take_profit_pct:.6f} ({settings.risk.take_profit_pct*100:.2f}%)")
            
            if settings.risk.stop_loss_pct > 1.0:
                print(f"[live] 🚨 ERROR: stop_loss_pct={settings.risk.stop_loss_pct:.6f} ({settings.risk.stop_loss_pct*100:.2f}%) is > 100%!")
                print(f"[live]   This is definitely wrong. Dividing by 100 to correct...")
                settings.risk.stop_loss_pct = settings.risk.stop_loss_pct / 100.0
                print(f"[live]   Corrected to: {settings.risk.stop_loss_pct:.6f} ({settings.risk.stop_loss_pct*100:.2f}%)")
            
            # Преобразуем проценты от маржи в проценты от цены: / leverage
            tp_pct_from_price = settings.risk.take_profit_pct / settings.leverage
            sl_pct_from_price = settings.risk.stop_loss_pct / settings.leverage
            
            print(f"[live]   → Converted to price %: TP={tp_pct_from_price*100:.2f}%, SL={sl_pct_from_price*100:.2f}% (from margin % with {settings.leverage}x leverage)")
            
            if position_bias == Bias.LONG:
                base_tp = avg_price * (1 + tp_pct_from_price)
                base_sl = avg_price * (1 - sl_pct_from_price)
            else:  # SHORT
                base_tp = avg_price * (1 - tp_pct_from_price)
                base_sl = avg_price * (1 + sl_pct_from_price)
            
            # Убрали детальное логирование TP/SL - слишком много сообщений
            # print(f"[live]   → base_tp=${base_tp:.2f}, base_sl=${base_sl:.2f} (entry: ${avg_price:.2f})")
            
            strategy_name = "TREND/FLAT"
        
        # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ: Проверяем, что вычисленные TP/SL находятся в разумных пределах
        # Если отклонение > 50% от entry price, это явно ошибка
        tp_deviation_pct = abs((base_tp - avg_price) / avg_price) * 100 if avg_price > 0 else 0
        sl_deviation_pct = abs((base_sl - avg_price) / avg_price) * 100 if avg_price > 0 else 0
        
        if tp_deviation_pct > 50:
            print(f"[live] 🚨 CRITICAL: Calculated TP has {tp_deviation_pct:.0f}% deviation from entry! This is an error.")
            print(f"[live]   Entry: ${avg_price:.2f}, Calculated TP: ${base_tp:.2f}")
            print(f"[live]   Using safe defaults: TP = entry * 1.02 (2%)")
            if position_bias == Bias.LONG:
                base_tp = avg_price * 1.02
            else:
                base_tp = avg_price * 0.98
        
        if sl_deviation_pct > 50:
            print(f"[live] 🚨 CRITICAL: Calculated SL has {sl_deviation_pct:.0f}% deviation from entry! This is an error.")
            print(f"[live]   Entry: ${avg_price:.2f}, Calculated SL: ${base_sl:.2f}")
            print(f"[live]   Using safe defaults: SL = entry * 0.99 (1%)")
            if position_bias == Bias.LONG:
                base_sl = avg_price * 0.99
            else:
                base_sl = avg_price * 1.01
        
        # ВАЛИДАЦИЯ: Проверяем, что TP/SL корректны для направления позиции
        # ВАЖНО: Для стратегических TP/SL выполняем только базовую валидацию (направление),
        # не перезаписываем значения, так как они уже рассчитаны стратегией
        leverage = settings.leverage if hasattr(settings, 'leverage') else 10
        min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
        max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
        min_sl_pct_from_price = min_sl_pct_from_margin / leverage  # Минимум от цены
        max_sl_pct_from_price = max_sl_pct_from_margin / leverage   # Максимум от цены
        
        if position_bias == Bias.LONG:
            # Для LONG: TP должен быть выше цены входа, SL должен быть ниже
            if base_tp <= avg_price:
                print(f"[live] ⚠️ WARNING: TP ({base_tp:.2f}) <= entry price ({avg_price:.2f}) for LONG position, adjusting...")
                base_tp = avg_price * 1.01  # Минимальный TP 1% выше входа
            
            # Для SL: проверяем, что он в правильном направлении
            if base_sl >= avg_price:
                print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) >= entry price ({avg_price:.2f}) for LONG position, adjusting...")
                # Используем минимальный SL от маржи (7%)
                base_sl = avg_price * (1 - min_sl_pct_from_price)
                print(f"[live]   Adjusted SL to {min_sl_pct_from_margin*100:.0f}% from margin ({min_sl_pct_from_price*100:.2f}% from price)")
            elif not strategy_tp_sl_applied:
                # Проверяем диапазон 7-10% от маржи ТОЛЬКО если это не стратегические TP/SL
                # Для стратегических TP/SL значения уже правильные и не требуют корректировки
                sl_deviation_pct_from_price = abs(avg_price - base_sl) / avg_price
                sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
                
                # Добавляем небольшой допуск для округления (0.001 = 0.1%)
                if sl_deviation_pct_from_margin < min_sl_pct_from_margin - 0.001:
                    print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting...")
                    base_sl = avg_price * (1 - min_sl_pct_from_price)
                    print(f"[live]   Adjusted SL to {min_sl_pct_from_margin*100:.0f}% from margin ({min_sl_pct_from_price*100:.2f}% from price)")
                elif sl_deviation_pct_from_margin > max_sl_pct_from_margin * 1.01:  # Допуск 1% для округления
                    print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusting...")
                    base_sl = avg_price * (1 - max_sl_pct_from_price)
                    print(f"[live]   Adjusted SL to {max_sl_pct_from_margin*100:.0f}% from margin ({max_sl_pct_from_price*100:.2f}% from price)")
                else:
                    # Убрано verbose сообщение "SL is correct" - логируется только при проблемах
                    pass
            else:
                # Для стратегических TP/SL: только проверяем направление, не корректируем значения
                print(f"[live] ✅ Strategy-specific SL ({base_sl:.2f}) validated - direction correct, keeping strategy value")
        else:  # SHORT
            # Для SHORT: TP должен быть ниже цены входа, SL должен быть выше
            if base_tp >= avg_price:
                print(f"[live] ⚠️ WARNING: TP ({base_tp:.2f}) >= entry price ({avg_price:.2f}) for SHORT position, adjusting...")
                base_tp = avg_price * 0.99  # Минимальный TP 1% ниже входа
            
            # Для SL: проверяем, что он в правильном направлении
            if base_sl <= avg_price:
                print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) <= entry price ({avg_price:.2f}) for SHORT position, adjusting...")
                # Используем минимальный SL от маржи (7%)
                base_sl = avg_price * (1 + min_sl_pct_from_price)
                print(f"[live]   Adjusted SL to {min_sl_pct_from_margin*100:.0f}% from margin ({min_sl_pct_from_price*100:.2f}% from price)")
            elif not strategy_tp_sl_applied:
                # Проверяем диапазон 7-10% от маржи ТОЛЬКО если это не стратегические TP/SL
                # Для стратегических TP/SL значения уже правильные и не требуют корректировки
                sl_deviation_pct_from_price = abs(base_sl - avg_price) / avg_price
                sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
                
                # Добавляем небольшой допуск для округления (0.001 = 0.1%)
                if sl_deviation_pct_from_margin < min_sl_pct_from_margin - 0.001:
                    print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting...")
                    base_sl = avg_price * (1 + min_sl_pct_from_price)
                    print(f"[live]   Adjusted SL to {min_sl_pct_from_margin*100:.0f}% from margin ({min_sl_pct_from_price*100:.2f}% from price)")
                elif sl_deviation_pct_from_margin > max_sl_pct_from_margin * 1.01:  # Допуск 1% для округления
                    print(f"[live] ⚠️ WARNING: SL ({base_sl:.2f}) too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusting...")
                    base_sl = avg_price * (1 + max_sl_pct_from_price)
                    print(f"[live]   Adjusted SL to {max_sl_pct_from_margin*100:.0f}% from margin ({max_sl_pct_from_price*100:.2f}% from price)")
                else:
                    # Убрано verbose сообщение "SL is correct" - логируется только при проблемах
                    pass
            else:
                # Для стратегических TP/SL: только проверяем направление, не корректируем значения
                print(f"[live] ✅ Strategy-specific SL ({base_sl:.2f}) validated - direction correct, keeping strategy value")
        
        # Инициализируем целевые TP/SL базовыми значениями
        target_tp = base_tp
        target_sl = base_sl
        print(f"[live] 🔧 Initialized target_tp=${target_tp:.2f}, target_sl=${target_sl:.2f} from base_tp/base_sl (entry: ${avg_price:.2f})")
        
        # Если позиция открыта AMT & Order Flow стратегией – используем отдельную логику сопровождения
        if position_strategy_type == "amt_of":
            # 1) Безубыток при достижении amt_of_breakeven_rr * риск
            try:
                # текущий риск в R оцениваем как |avg_price - base_sl| в %
                base_sl_val = float(current_sl) if sl_set else avg_price
                if position_bias == Bias.LONG:
                    risk_pct = abs(avg_price - base_sl_val) / avg_price * 100
                    profit_r = max_profit_pct / risk_pct if risk_pct > 0 else 0.0
                else:
                    risk_pct = abs(base_sl_val - avg_price) / avg_price * 100
                    profit_r = max_profit_pct / risk_pct if risk_pct > 0 else 0.0
            except Exception:
                profit_r = 0.0

            amt_rr = getattr(settings.strategy, "amt_of_breakeven_rr", 1.5)
            if profit_r >= amt_rr:
                # Переводим SL в безубыток + небольшой буфер
                if position_bias == Bias.LONG:
                    breakeven_sl = avg_price * 1.0005
                else:
                    breakeven_sl = avg_price * 0.9995
                target_sl = breakeven_sl
                print(
                    f"[live] [{symbol}] 🔒 AMT_OF Breakeven: moving SL to ${breakeven_sl:.2f} "
                    f"(~{profit_r:.2f}R, rr_target={amt_rr})"
                )

            # 2) Auction timeout: если после открытия прошло больше amt_of_auction_timeout_sec и max_profit_pct маленький – выходим
            try:
                from datetime import datetime, timezone
                opened_at = position.get("createdTime") or position.get("created_time")
                timeout_sec = getattr(settings.strategy, "amt_of_auction_timeout_sec", 600)
                if opened_at and timeout_sec > 0:
                    opened_ts = int(opened_at) / 1000.0 if isinstance(opened_at, str) and opened_at.isdigit() else None
                    if opened_ts:
                        opened_dt = datetime.fromtimestamp(opened_ts, tz=timezone.utc)
                        age_sec = (datetime.now(timezone.utc) - opened_dt).total_seconds()
                        if age_sec >= timeout_sec and max_profit_pct < 0.2:
                            # Ставим SL очень близко к текущей цене, чтобы выйти
                            if position_bias == Bias.LONG:
                                target_sl = min(target_sl, current_price * 0.999) if target_sl else current_price * 0.999
                            else:
                                target_sl = max(target_sl, current_price * 1.001) if target_sl else current_price * 1.001
                            print(
                                f"[live] [{symbol}] ⏳ AMT_OF auction timeout: position age {age_sec:.0f}s "
                                f"(timeout={timeout_sec}s), max_profit={max_profit_pct:.2f}% – forcing exit via SL {target_sl:.2f}"
                            )
            except Exception:
                pass

            # 3) Three-bar exit: если включен флаг – при трёх подряд барах против позиции ставим SL ближе
            if getattr(settings.strategy, "amt_of_three_bar_exit_enabled", True):
                try:
                    # Ожидаем, что df_ready есть во внешнем контексте и last 3 бара доступны через history,
                    # поэтому здесь только защитный лог – основная логика закрытия реализуется в основном цикле.
                    # Чтобы не лезть в df_ready из этой функции, реализуем 3-bar exit как "резкий" сдвиг SL,
                    # если max_profit_pct уже был положительный и вернулся к нулю/минусу.
                    if max_profit_pct < 0 and position_max_profit.get(symbol, 0.0) > 0.5:
                        # Цена ушла против после какого‑то профита – поджимаем SL вблизи текущей
                        if position_bias == Bias.LONG:
                            target_sl = min(target_sl, current_price * 0.999) if target_sl else current_price * 0.999
                        else:
                            target_sl = max(target_sl, current_price * 1.001) if target_sl else current_price * 1.001
                        print(
                            f"[live] [{symbol}] ⛔ AMT_OF three-bar style exit: profit faded after move in favor, "
                            f"tightening SL to {target_sl:.2f}"
                        )
                except Exception:
                    pass

        # 1. БЕЗУБЫТОК (общий): Перемещаем SL в безубыток при достижении определенной прибыли
        # ВАЖНО: Безубыток должен быть лучше текущего SL, но не меньше 7% от маржи
        if settings.risk.enable_breakeven and max_profit_pct >= settings.risk.breakeven_activation_pct * 100:
            if position_bias == Bias.LONG:
                breakeven_sl = avg_price * 0.999  # Немного ниже входа для LONG (чтобы не сработал сразу)
            else:  # SHORT
                breakeven_sl = avg_price * 1.001  # Немного выше входа для SHORT (чтобы не сработал сразу)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Безубыток должен быть не меньше 7% от маржи
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            min_sl_pct_from_price = min_sl_pct_from_margin / leverage  # Минимум от цены (0.7% при 10x)
            
            if position_bias == Bias.LONG:
                breakeven_sl_pct_from_price = abs(avg_price - breakeven_sl) / avg_price
                breakeven_sl_pct_from_margin = breakeven_sl_pct_from_price * leverage
            else:  # SHORT
                breakeven_sl_pct_from_price = abs(breakeven_sl - avg_price) / avg_price
                breakeven_sl_pct_from_margin = breakeven_sl_pct_from_price * leverage
            
            # Проверяем, что безубыток лучше базового SL И не меньше 7% от маржи
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Безубыток не должен быть меньше 7% от маржи
            use_breakeven = False
            
            # Проверяем, что безубыток не меньше 7% от маржи
            if breakeven_sl_pct_from_margin < min_sl_pct_from_margin:
                print(f"[live] ⚠️ Breakeven SL ({breakeven_sl:.2f}) is too small ({breakeven_sl_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), not using it. Keeping base SL ({base_sl:.2f})")
            elif position_bias == Bias.LONG:
                # Для LONG: безубыток должен быть выше базового SL (ближе к цене входа)
                if breakeven_sl > base_sl:
                    use_breakeven = True
                    print(f"[live] ✅ Breakeven SL ({breakeven_sl:.2f}, {breakeven_sl_pct_from_margin*100:.1f}% from margin) is better than base SL ({base_sl:.2f}) for LONG position")
                else:
                    print(f"[live] ⚠️ Breakeven SL ({breakeven_sl:.2f}) is not better than base SL ({base_sl:.2f}), keeping base SL")
            else:  # SHORT
                # Для SHORT: безубыток должен быть ниже базового SL (ближе к цене входа)
                if breakeven_sl < base_sl:
                    use_breakeven = True
                    print(f"[live] ✅ Breakeven SL ({breakeven_sl:.2f}, {breakeven_sl_pct_from_margin*100:.1f}% from margin) is better than base SL ({base_sl:.2f}) for SHORT position")
                else:
                    print(f"[live] ⚠️ Breakeven SL ({breakeven_sl:.2f}) is not better than base SL ({base_sl:.2f}), keeping base SL")
            
            if use_breakeven:
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Убеждаемся, что безубыток не меньше 7% от маржи
                if breakeven_sl_pct_from_margin < min_sl_pct_from_margin:
                    print(
                        f"[live] 🚨 CRITICAL: Breakeven SL ({breakeven_sl:.2f}) is too small "
                        f"({breakeven_sl_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), "
                        f"NOT setting it. Keeping base SL ({base_sl:.2f})"
                    )
                    use_breakeven = False
                else:
                    # Если текущий SL хуже безубытка, перемещаем его
                    if sl_set:
                        try:
                            current_sl_val = float(current_sl)
                            if position_bias == Bias.LONG and current_sl_val < breakeven_sl:
                                target_sl = breakeven_sl
                                print(
                                    f"[live] 🔒 Moving SL to breakeven: ${target_sl:.2f} "
                                    f"({breakeven_sl_pct_from_margin*100:.1f}% from margin, profit: {max_profit_pct:.2f}%)"
                                )
                            elif position_bias == Bias.SHORT and current_sl_val > breakeven_sl:
                                target_sl = breakeven_sl
                                print(
                                    f"[live] 🔒 Moving SL to breakeven: ${target_sl:.2f} "
                                    f"({breakeven_sl_pct_from_margin*100:.1f}% from margin, profit: {max_profit_pct:.2f}%)"
                                )
                            else:
                                print(
                                    f"[live] ✅ Current SL ({current_sl_val:.2f}) is already better than breakeven "
                                    f"({breakeven_sl:.2f}), keeping it"
                                )
                        except (ValueError, TypeError):
                            target_sl = breakeven_sl
                            print(
                                f"[live] 🔒 Setting SL to breakeven: ${target_sl:.2f} "
                                f"({breakeven_sl_pct_from_margin*100:.1f}% from margin, profit: {max_profit_pct:.2f}%)"
                            )
                    else:
                        target_sl = breakeven_sl
                        print(
                            f"[live] 🔒 Setting SL to breakeven: ${target_sl:.2f} "
                            f"({breakeven_sl_pct_from_margin*100:.1f}% from margin, profit: {max_profit_pct:.2f}%)"
                        )
            # Сообщение "is worse than base SL or too small" убрано - уже есть сообщение выше
        
        # 2. TRAILING STOP: Активируем trailing stop, когда цена прошла половину до TP
        # Вычисляем расстояние до TP
        if position_bias == Bias.LONG:
            tp_distance_pct = (target_tp - avg_price) / avg_price * 100  # Процент от цены входа до TP
        else:  # SHORT
            tp_distance_pct = (avg_price - target_tp) / avg_price * 100  # Процент от цены входа до TP
        
        # Trailing stop активируется, когда прибыль >= 50% от расстояния до TP
        half_tp_distance_pct = tp_distance_pct * 0.5
        
        # Также проверяем минимальную активацию из настроек (для обратной совместимости)
        min_activation_pct = max(settings.risk.trailing_stop_activation_pct * 100, half_tp_distance_pct)
        
        # Убрано verbose сообщение о проверке trailing stop - логируется только при активации
        
        if settings.risk.enable_trailing_stop and max_profit_pct >= min_activation_pct:
            trailing_distance_pct = settings.risk.trailing_stop_distance_pct
            
            if position_bias == Bias.LONG:
                # Для LONG: SL должен быть ниже максимальной цены на trailing_distance_pct
                trailing_sl = max_price * (1 - trailing_distance_pct)
                # ВАЖНО: Trailing stop должен быть лучше базового SL (выше для LONG)
                # Может быть выше цены входа, если цена прошла половину до TP
                if trailing_sl > base_sl:
                    if trailing_sl > target_sl:
                        old_target_sl = target_sl
                        target_sl = trailing_sl
                        # Устанавливаем флаг, что это trailing stop (для последующей валидации)
                        is_trailing_stop_applied = True
                        print(
                            f"[live] 📈 Trailing stop ACTIVATED: ${old_target_sl:.2f} → ${target_sl:.2f} "
                            f"(max price: ${max_price:.2f}, profit: {max_profit_pct:.2f}%, {half_tp_distance_pct:.2f}% to half TP)"
                        )
                        print(f"[live]   Trailing distance: {trailing_distance_pct*100:.2f}% from max price")
                    else:
                        print(
                            f"[live] ✅ Current SL ({target_sl:.2f}) is already better than trailing stop "
                            f"({trailing_sl:.2f}), keeping it"
                        )
                        is_trailing_stop_applied = False
                else:
                    # Если trailing SL хуже базового SL, не используем его
                    print(
                        f"[live] ⚠️ Trailing stop ({trailing_sl:.2f}) is worse than base SL ({base_sl:.2f}), keeping base SL"
                    )
                    is_trailing_stop_applied = False
            else:  # SHORT
                # Для SHORT: SL должен быть выше максимальной цены на trailing_distance_pct
                trailing_sl = max_price * (1 + trailing_distance_pct)
                # ВАЖНО: Trailing stop должен быть лучше базового SL (ниже для SHORT)
                # Может быть ниже цены входа, если цена прошла половину до TP
                if trailing_sl < base_sl:
                    if trailing_sl < target_sl:
                        old_target_sl = target_sl
                        target_sl = trailing_sl
                        # Устанавливаем флаг, что это trailing stop (для последующей валидации)
                        is_trailing_stop_applied = True
                        print(
                            f"[live] 📉 Trailing stop ACTIVATED: ${old_target_sl:.2f} → ${target_sl:.2f} "
                            f"(max price: ${max_price:.2f}, profit: {max_profit_pct:.2f}%, {half_tp_distance_pct:.2f}% to half TP)"
                        )
                        print(f"[live]   Trailing distance: {trailing_distance_pct*100:.2f}% from max price")
                    else:
                        print(f"[live] ✅ Current SL ({target_sl:.2f}) is already better than trailing stop ({trailing_sl:.2f}), keeping it")
                        is_trailing_stop_applied = False
                else:
                    # Если trailing SL хуже базового SL, не используем его
                    print(
                        f"[live] ⚠️ Trailing stop ({trailing_sl:.2f}) is worse than base SL ({base_sl:.2f}), keeping base SL"
                    )
                    is_trailing_stop_applied = False
        elif settings.risk.enable_trailing_stop:
            # Trailing stop еще не активирован
            # Убрано verbose сообщение "Trailing stop waiting" - логируется только при активации
            is_trailing_stop_applied = False
        else:
            is_trailing_stop_applied = False
        
        # Проверяем, нужно ли обновить TP/SL
        tp_needs_update = not tp_set
        sl_needs_update = not sl_set
        
        print(f"[live] 🔍 TP/SL update check: tp_set={tp_set}, sl_set={sl_set}, tp_needs_update={tp_needs_update}, sl_needs_update={sl_needs_update}")
        print(f"[live]   Current TP: {current_tp if tp_set else 'NOT SET'}, Target TP: ${target_tp:.2f}")
        print(f"[live]   Current SL: {current_sl if sl_set else 'NOT SET'}, Target SL: ${target_sl:.2f}")
        
        # Если TP/SL установлены, проверяем, соответствуют ли они целевым значениям
        # (допускаем погрешность в 0.2% для избежания частых обновлений и ошибок "not modified")
        if tp_set:
            try:
                current_tp_val = float(current_tp)
                tp_diff_pct = abs((current_tp_val - target_tp) / avg_price) * 100
                if tp_diff_pct > 0.2:  # Если разница больше 0.2%
                    tp_needs_update = True
                    print(f"[live] ✅ TP needs update: current={current_tp_val:.2f}, target={target_tp:.2f} (diff: {tp_diff_pct:.2f}%)")
                else:
                    print(f"[live] ℹ️  TP is close enough: current={current_tp_val:.2f}, target={target_tp:.2f} (diff: {tp_diff_pct:.2f}% <= 0.2%)")
            except (ValueError, TypeError):
                tp_needs_update = True
                print(f"[live] ⚠️  TP value error, setting tp_needs_update=True")
        
        if sl_set:
            try:
                current_sl_val = float(current_sl)
                sl_diff_pct = abs((current_sl_val - target_sl) / avg_price) * 100
                if sl_diff_pct > 0.2:  # Если разница больше 0.2%
                    sl_needs_update = True
                    print(f"[live] ✅ SL needs update: current={current_sl_val:.2f}, target={target_sl:.2f} (diff: {sl_diff_pct:.2f}%)")
                else:
                    print(f"[live] ℹ️  SL is close enough: current={current_sl_val:.2f}, target={target_sl:.2f} (diff: {sl_diff_pct:.2f}% <= 0.2%)")
            except (ValueError, TypeError):
                sl_needs_update = True
                print(f"[live] ⚠️  SL value error, setting sl_needs_update=True")
        
        # ФИНАЛЬНАЯ ВАЛИДАЦИЯ: Проверяем target_sl и target_tp перед отправкой в API
        # Это критически важно для предотвращения ошибок API
        # ВАЖНО: Trailing stop может быть выше цены входа (для LONG) или ниже (для SHORT), если цена прошла половину до TP
        
        # Проверяем, является ли это trailing stop (SL лучше базового SL и прибыль достаточно большая)
        # Вычисляем половину до TP для проверки
        if position_bias == Bias.LONG:
            tp_distance_pct = (target_tp - avg_price) / avg_price * 100
        else:  # SHORT
            tp_distance_pct = (avg_price - target_tp) / avg_price * 100
        half_tp_distance_pct = tp_distance_pct * 0.5
        min_activation_pct = max(settings.risk.trailing_stop_activation_pct * 100, half_tp_distance_pct)
        
        # Проверяем, является ли это trailing stop
        # Используем флаг, установленный ранее при активации trailing stop
        is_trailing_stop = is_trailing_stop_applied if 'is_trailing_stop_applied' in locals() else False
        # Дополнительная проверка для надежности
        if not is_trailing_stop and settings.risk.enable_trailing_stop and max_profit_pct >= min_activation_pct:
            if position_bias == Bias.LONG:
                # Для LONG: trailing stop выше базового SL (может быть выше цены входа)
                is_trailing_stop = target_sl > base_sl
            else:  # SHORT
                # Для SHORT: trailing stop ниже базового SL (может быть ниже цены входа)
                is_trailing_stop = target_sl < base_sl
        
        # Сохраняем is_trailing_stop для использования в валидации ниже
        final_is_trailing_stop = is_trailing_stop
        
        if position_bias == Bias.LONG:
            # Для LONG: TP должен быть выше цены входа
            if target_tp <= avg_price:
                print(f"[live] ⚠️ WARNING: Final TP ({target_tp:.2f}) <= entry price ({avg_price:.2f}) for LONG position, adjusting...")
                target_tp = avg_price * 1.01  # Минимальный TP 1% выше входа
            # Для LONG: SL должен быть ниже цены входа, ИСКЛЮЧЕНИЕ: trailing stop может быть выше
            if target_sl >= avg_price and not is_trailing_stop:
                print(f"[live] ⚠️ CRITICAL: Final SL ({target_sl:.2f}) >= entry price ({avg_price:.2f}) for LONG position, FORCING adjustment...")
                target_sl = avg_price * 0.99  # Минимальный SL 1% ниже входа
                # Дополнительная проверка после корректировки
                if target_sl >= avg_price:
                    target_sl = avg_price * 0.95  # Если все еще проблема, используем 5% ниже
                    print(f"[live] ⚠️ CRITICAL: SL still invalid, using 5% below entry: ${target_sl:.2f}")
            elif is_trailing_stop:
                print(f"[live] ✅ Trailing stop SL ({target_sl:.2f}) is above entry price ({avg_price:.2f}) - это нормально для trailing stop")
        else:  # SHORT
            # Для SHORT: TP должен быть ниже цены входа
            if target_tp >= avg_price:
                print(f"[live] ⚠️ WARNING: Final TP ({target_tp:.2f}) >= entry price ({avg_price:.2f}) for SHORT position, adjusting...")
                target_tp = avg_price * 0.99  # Минимальный TP 1% ниже входа
            # Для SHORT: SL должен быть выше цены входа, ИСКЛЮЧЕНИЕ: trailing stop может быть ниже
            if target_sl <= avg_price and not is_trailing_stop:
                print(f"[live] ⚠️ CRITICAL: Final SL ({target_sl:.2f}) <= entry price ({avg_price:.2f}) for SHORT position, FORCING adjustment...")
                target_sl = avg_price * 1.01  # Минимальный SL 1% выше входа
                # Дополнительная проверка после корректировки
                if target_sl <= avg_price:
                    target_sl = avg_price * 1.05  # Если все еще проблема, используем 5% выше
                    print(f"[live] ⚠️ CRITICAL: SL still invalid, using 5% above entry: ${target_sl:.2f}")
            elif is_trailing_stop:
                print(f"[live] ✅ Trailing stop SL ({target_sl:.2f}) is below entry price ({avg_price:.2f}) - это нормально для trailing stop")
        
        # Устанавливаем или обновляем TP/SL
        if tp_needs_update or sl_needs_update:
            print(f"[live] 🔧 Ensuring TP/SL for {position_bias.value} position ({strategy_name} strategy):")
            print(f"[live]   Entry: ${avg_price:.2f}, Current: ${current_price:.2f}, Max: ${max_price:.2f}")
            print(f"[live]   Max Profit: {max_profit_pct:.2f}%")
            print(f"[live]   Target TP: ${target_tp:.2f} ({'+' if position_bias == Bias.LONG else '-'}{abs((target_tp - avg_price) / avg_price * 100):.2f}%)")
            print(f"[live]   Target SL: ${target_sl:.2f} ({'-' if position_bias == Bias.LONG else '+'}{abs((target_sl - avg_price) / avg_price * 100):.2f}%)")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА ПЕРЕД ОТПРАВКОЙ: Убеждаемся, что значения корректны
            final_sl = target_sl if sl_needs_update else None
            final_tp = target_tp if tp_needs_update else None
            
            # Сохраняем базовый SL для проверки безубытка
            base_sl_for_check = base_sl
            
            # СТРОГАЯ ВАЛИДАЦИЯ: Исправляем значения до отправки в API
            # ВАЖНО: SL должен быть в диапазоне 7-10% от маржи
            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
            min_sl_pct_from_price = min_sl_pct_from_margin / leverage
            max_sl_pct_from_price = max_sl_pct_from_margin / leverage
            
            if final_sl is not None:
                if position_bias == Bias.LONG:
                    # Для LONG: SL должен быть СТРОГО ниже цены входа
                    # ИСКЛЮЧЕНИЕ: Если это trailing stop, он может быть выше входа (защита прибыли)
                    # Используем сохраненное значение is_trailing_stop
                    is_trailing = final_is_trailing_stop if 'final_is_trailing_stop' in locals() else False
                    if final_sl >= avg_price and not is_trailing:
                        print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) >= entry ({avg_price:.2f}) for LONG, adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                        final_sl = avg_price * (1 - min_sl_pct_from_price)
                    elif final_sl >= avg_price and is_trailing:
                        # Trailing stop выше входа - это нормально для защиты прибыли
                        print(f"[live] ✅ Trailing stop SL ({final_sl:.2f}) is above entry ({avg_price:.2f}) - это нормально для trailing stop (защита прибыли)")
                    else:
                        # Проверяем, что SL в диапазоне 7-10% от маржи
                        sl_deviation_pct_from_price = abs(avg_price - final_sl) / avg_price
                        sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
                        
                        # ВАЖНО: Если это trailing stop, не проверяем минимальный процент от маржи
                        # Trailing stop может быть выше входа для LONG (защита прибыли)
                        if is_trailing_stop and final_sl > base_sl_for_check:
                            print(f"[live] ✅ Final SL is trailing stop ({final_sl:.2f}), better than base SL ({base_sl_for_check:.2f}), keeping it")
                        # ВАЖНО: Если это безубыток (близко к цене входа, в пределах 0.5% от цены), 
                        # и он лучше базового SL И не меньше 7% от маржи, не перезаписываем его
                        elif sl_deviation_pct_from_price < 0.005:  # 0.5% от цены
                            is_breakeven = True
                            if sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                                # Безубыток слишком маленький (< 7% от маржи), не используем его
                                print(f"[live] 🚨 CRITICAL FIX: Breakeven SL ({final_sl:.2f}) is too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                                final_sl = avg_price * (1 - min_sl_pct_from_price)
                            elif final_sl > base_sl_for_check:
                                # Безубыток правильный (>= 7% от маржи) и лучше базового SL
                                print(f"[live] ✅ Final SL is breakeven ({final_sl:.2f}, {sl_deviation_pct_from_margin*100:.1f}% from margin), better than base SL ({base_sl_for_check:.2f}), keeping it")
                            else:
                                # Безубыток не лучше базового SL, используем базовый SL
                                print(f"[live] ⚠️ Breakeven SL ({final_sl:.2f}) is not better than base SL ({base_sl_for_check:.2f}), adjusting to base SL")
                                final_sl = base_sl_for_check
                        elif sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                            print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                            final_sl = avg_price * (1 - min_sl_pct_from_price)
                        elif sl_deviation_pct_from_margin > max_sl_pct_from_margin * 1.01:  # Допуск 1% для округления
                            print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusting to {max_sl_pct_from_margin*100:.0f}% from margin")
                            final_sl = avg_price * (1 - max_sl_pct_from_price)
                        else:
                            # Убрано verbose сообщение "Final SL is correct" - логируется только при проблемах
                            pass
                else:  # SHORT
                    # Для SHORT: SL должен быть СТРОГО выше цены входа
                    if final_sl <= avg_price:
                        print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) <= entry ({avg_price:.2f}) for SHORT, adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                        final_sl = avg_price * (1 + min_sl_pct_from_price)
                    else:
                        # Проверяем, что SL в диапазоне 7-10% от маржи
                        sl_deviation_pct_from_price = abs(final_sl - avg_price) / avg_price
                        sl_deviation_pct_from_margin = sl_deviation_pct_from_price * leverage
                        
                        # ВАЖНО: Если это trailing stop, не проверяем минимальный процент от маржи
                        # Trailing stop может быть ниже входа для SHORT (защита прибыли)
                        if is_trailing_stop and final_sl < base_sl_for_check:
                            print(f"[live] ✅ Final SL is trailing stop ({final_sl:.2f}), better than base SL ({base_sl_for_check:.2f}), keeping it")
                        # ВАЖНО: Если это безубыток (близко к цене входа, в пределах 0.5% от цены), 
                        # и он лучше базового SL И не меньше 7% от маржи, не перезаписываем его
                        elif sl_deviation_pct_from_price < 0.005:  # 0.5% от цены
                            is_breakeven = True
                            if sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                                # Безубыток слишком маленький (< 7% от маржи), не используем его
                                print(f"[live] 🚨 CRITICAL FIX: Breakeven SL ({final_sl:.2f}) is too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                                final_sl = avg_price * (1 + min_sl_pct_from_price)
                            elif final_sl < base_sl_for_check:
                                # Безубыток правильный (>= 7% от маржи) и лучше базового SL
                                print(f"[live] ✅ Final SL is breakeven ({final_sl:.2f}, {sl_deviation_pct_from_margin*100:.1f}% from margin), better than base SL ({base_sl_for_check:.2f}), keeping it")
                            else:
                                # Безубыток не лучше базового SL, используем базовый SL
                                print(f"[live] ⚠️ Breakeven SL ({final_sl:.2f}) is not better than base SL ({base_sl_for_check:.2f}), adjusting to base SL")
                                final_sl = base_sl_for_check
                        elif sl_deviation_pct_from_margin < min_sl_pct_from_margin:
                            print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) too small ({sl_deviation_pct_from_margin*100:.1f}% from margin < {min_sl_pct_from_margin*100:.0f}%), adjusting to {min_sl_pct_from_margin*100:.0f}% from margin")
                            final_sl = avg_price * (1 + min_sl_pct_from_price)
                        elif sl_deviation_pct_from_margin > max_sl_pct_from_margin * 1.01:  # Допуск 1% для округления
                            print(f"[live] 🚨 CRITICAL FIX: SL ({final_sl:.2f}) too large ({sl_deviation_pct_from_margin*100:.1f}% from margin > {max_sl_pct_from_margin*100:.0f}%), adjusting to {max_sl_pct_from_margin*100:.0f}% from margin")
                            final_sl = avg_price * (1 + max_sl_pct_from_price)
                        else:
                            # Убрано verbose сообщение "Final SL is correct" - логируется только при проблемах
                            pass
            
            if final_tp is not None:
                if position_bias == Bias.LONG:
                    # Для LONG: TP должен быть СТРОГО выше цены входа
                    if final_tp <= avg_price:
                        print(f"[live] 🚨 CRITICAL FIX: TP ({final_tp:.2f}) <= entry ({avg_price:.2f}) for LONG, forcing to 1.01x entry")
                        final_tp = avg_price * 1.01
                        # Дополнительная проверка после корректировки
                        if final_tp <= avg_price:
                            final_tp = avg_price * 1.05  # Если все еще проблема, используем 5% выше
                            print(f"[live] 🚨 CRITICAL FIX: TP still invalid, using 5% above entry: ${final_tp:.2f}")
                else:  # SHORT
                    # Для SHORT: TP должен быть СТРОГО ниже цены входа
                    if final_tp >= avg_price:
                        print(f"[live] 🚨 CRITICAL FIX: TP ({final_tp:.2f}) >= entry ({avg_price:.2f}) for SHORT, forcing to 0.99x entry")
                        final_tp = avg_price * 0.99
                        # Дополнительная проверка после корректировки
                        if final_tp >= avg_price:
                            final_tp = avg_price * 0.95  # Если все еще проблема, используем 5% ниже
                            print(f"[live] 🚨 CRITICAL FIX: TP still invalid, using 5% below entry: ${final_tp:.2f}")
            
            try:
                # Дополнительная проверка: убеждаемся, что цены - это числа, а не строки или другие типы
                if final_sl is not None:
                    if not isinstance(final_sl, (int, float)):
                        print(f"[live] ⚠️ WARNING: final_sl is not a number: {type(final_sl)} = {final_sl}, converting...")
                        final_sl = float(final_sl)
                    # Проверяем, что цена не слишком большая (возможно, умножена на неправильный множитель)
                    if final_sl > avg_price * 1000:
                        print(f"[live] 🚨 CRITICAL: final_sl ({final_sl:.2f}) is suspiciously large (entry: {avg_price:.2f}), possible multiplier error!")
                        # Пытаемся исправить, деля на возможные множители
                        for divisor in [100000000, 1000000, 10000, 1000, 100, 10]:
                            corrected = final_sl / divisor
                            if abs(corrected - avg_price) < avg_price * 0.1:  # В пределах 10% от entry
                                print(f"[live] 🔧 Correcting final_sl: {final_sl:.2f} / {divisor} = {corrected:.2f}")
                                final_sl = corrected
                                break
                
                if final_tp is not None:
                    if not isinstance(final_tp, (int, float)):
                        print(f"[live] ⚠️ WARNING: final_tp is not a number: {type(final_tp)} = {final_tp}, converting...")
                        final_tp = float(final_tp)
                    # Проверяем, что цена не слишком большая
                    if final_tp > avg_price * 1000:
                        print(f"[live] 🚨 CRITICAL: final_tp ({final_tp:.2f}) is suspiciously large (entry: {avg_price:.2f}), possible multiplier error!")
                        # Пытаемся исправить, деля на возможные множители
                        for divisor in [100000000, 1000000, 10000, 1000, 100, 10]:
                            corrected = final_tp / divisor
                            if abs(corrected - avg_price) < avg_price * 0.1:  # В пределах 10% от entry
                                print(f"[live] 🔧 Correcting final_tp: {final_tp:.2f} / {divisor} = {corrected:.2f}")
                                final_tp = corrected
                                break
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: Если значения слишком большие (умножены на 10), делим на 10
                if final_tp is not None and avg_price > 0:
                    tp_deviation_pct = abs(final_tp - avg_price) / avg_price * 100
                    # Если отклонение > 300%, вероятно значение умножено на 10
                    if tp_deviation_pct > 300:
                        # Пробуем разделить на 10
                        if position_bias == Bias.LONG:
                            corrected_tp = avg_price + (final_tp - avg_price) / 10.0
                        else:  # SHORT
                            corrected_tp = avg_price - (avg_price - final_tp) / 10.0
                        corrected_deviation_pct = abs(corrected_tp - avg_price) / avg_price * 100
                        # Если после деления на 10 отклонение стало разумным (< 50%)
                        if corrected_deviation_pct < 50:
                            print(f"[live] 🔧 CORRECTING TP: ${final_tp:.2f} ({tp_deviation_pct:.0f}%) → ${corrected_tp:.2f} ({corrected_deviation_pct:.0f}%)")
                            final_tp = corrected_tp
                
                if final_sl is not None and avg_price > 0:
                    sl_deviation_pct = abs(final_sl - avg_price) / avg_price * 100
                    # Если отклонение > 300%, вероятно значение умножено на 10
                    if sl_deviation_pct > 300:
                        # Пробуем разделить на 10
                        if position_bias == Bias.LONG:
                            corrected_sl = avg_price - (avg_price - final_sl) / 10.0
                        else:  # SHORT
                            corrected_sl = avg_price + (final_sl - avg_price) / 10.0
                        corrected_deviation_pct = abs(corrected_sl - avg_price) / avg_price * 100
                        # Если после деления на 10 отклонение стало разумным (< 50%)
                        if corrected_deviation_pct < 50:
                            print(f"[live] 🔧 CORRECTING SL: ${final_sl:.2f} ({sl_deviation_pct:.0f}%) → ${corrected_sl:.2f} ({corrected_deviation_pct:.0f}%)")
                            final_sl = corrected_sl
                
                print(f"[live] 📤 Sending TP/SL to API: TP={final_tp}, SL={final_sl} (entry: {avg_price:.2f})")
                tp_sl_resp = client.set_trading_stop(
                    symbol=settings.symbol,
                    stop_loss=final_sl,
                    take_profit=final_tp,
                )
            
                if tp_sl_resp.get("retCode") == 0:
                    if tp_needs_update and sl_needs_update:
                        print(f"[live] ✅ TP and SL set/updated successfully")
                    elif tp_needs_update:
                        print(f"[live] ✅ TP set/updated successfully")
                    elif sl_needs_update:
                        print(f"[live] ✅ SL set/updated successfully")
                else:
                    ret_code = tp_sl_resp.get("retCode")
                    ret_msg = tp_sl_resp.get("retMsg", "Unknown error")
                    # Ошибка 34040 "not modified" - это нормально, значение уже установлено
                    if ret_code == 34040 or "not modified" in str(ret_msg).lower():
                        # Это не ошибка, просто значение уже установлено или слишком близко
                        if tp_needs_update and sl_needs_update:
                            print(f"[live] ℹ️  TP/SL already set (not modified)")
                        elif tp_needs_update:
                            print(f"[live] ℹ️  TP already set (not modified)")
                        elif sl_needs_update:
                            print(f"[live] ℹ️  SL already set (not modified)")
                    else:
                        print(f"[live] ⚠️  Failed to set/update TP/SL: {ret_msg} (ErrCode: {ret_code})")
            except InvalidRequestError as e:
                error_msg = str(e)
                # Ошибка 34040 "not modified" - это нормально
                if "34040" in error_msg or "not modified" in error_msg.lower():
                    print(f"[live] ℹ️  TP/SL already set (not modified) - skipping update")
                else:
                    raise  # Пробрасываем другие ошибки
            except Exception as e:
                # Для других ошибок логируем, но не прерываем выполнение
                error_msg = str(e)
                if "34040" in error_msg or "not modified" in error_msg.lower():
                    print(f"[live] ℹ️  TP/SL already set (not modified) - skipping update")
                else:
                    print(f"[live] ⚠️  Error setting TP/SL: {e}")
    
    except Exception as e:
        error_msg = str(e)
        # Ошибка 34040 "not modified" - это нормально, не логируем как ошибку
        if "34040" not in error_msg and "not modified" not in error_msg.lower():
            print(f"[live] Error ensuring TP/SL: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"[live] ℹ️  TP/SL already set (not modified) - skipping update")


def _check_partial_close(
    client: BybitClient,
    position: Dict[str, Any],
    position_bias: Bias,
    current_price: float,
    settings: AppSettings,
    position_max_profit: Dict[str, float],
    position_partial_closed: Dict[str, bool],
) -> bool:
    """
    Проверяет, нужно ли частично закрыть позицию при достижении определенного процента пути к TP.
    
    Args:
        client: Bybit клиент
        position: Информация о позиции
        position_bias: Направление позиции
        current_price: Текущая цена
        settings: Настройки бота
        position_max_profit: Словарь максимальной прибыли
        position_partial_closed: Словарь флагов частичного закрытия {symbol: bool}
    
    Returns:
        True если позиция была частично закрыта, False иначе
    """
    try:
        if not settings.risk.enable_partial_close:
            return False
        
        symbol = settings.symbol
        
        # Проверяем, не закрывали ли уже частично
        if position_partial_closed.get(symbol, False):
            return False
        
        avg_price = position.get("avg_price", 0)
        if avg_price == 0:
            return False
        
        # Рассчитываем целевой TP
        if settings.enable_ml_strategy and settings.ml_model_path:
            tp_pct = settings.ml_target_profit_pct_margin / settings.leverage
        else:
            tp_pct = settings.risk.take_profit_pct
        
        if position_bias == Bias.LONG:
            target_tp = avg_price * (1 + tp_pct)
            # Процент пути к TP
            progress_to_tp = ((current_price - avg_price) / (target_tp - avg_price)) * 100 if target_tp > avg_price else 0
        else:  # SHORT
            target_tp = avg_price * (1 - tp_pct)
            # Процент пути к TP
            progress_to_tp = ((avg_price - current_price) / (avg_price - target_tp)) * 100 if avg_price > target_tp else 0
        
        # Проверяем, достигли ли мы нужного процента пути к TP
        if progress_to_tp >= settings.risk.partial_close_at_tp_pct * 100:
            # Частично закрываем позицию
            qty = position["size"]
            close_qty = qty * settings.risk.partial_close_pct
            
            print(f"[live] 📊 Partial close triggered:")
            print(f"[live]   Progress to TP: {progress_to_tp:.2f}% (threshold: {settings.risk.partial_close_at_tp_pct * 100:.2f}%)")
            print(f"[live]   Closing {settings.risk.partial_close_pct * 100:.0f}% of position: {close_qty:.3f} of {qty:.3f}")
            
            side = "Sell" if position_bias == Bias.LONG else "Buy"
            resp = client.place_order(
                symbol=symbol,
                side=side,
                qty=close_qty,
                reduce_only=True,
            )
            
            if resp.get("retCode") == 0:
                print(f"[live] ✅ Partial close successful: {close_qty:.3f} @ ${current_price:.2f}")
                position_partial_closed[symbol] = True
                return True
            else:
                print(f"[live] ⚠️ Failed to partially close: {resp.get('retMsg', 'Unknown error')}")
        
        return False
    
    except Exception as e:
        print(f"[live] Error checking partial close: {e}")
        import traceback
        traceback.print_exc()
        return False


def _check_profit_protection(
    client: BybitClient,
    position: Dict[str, Any],
    position_bias: Bias,
    current_price: float,
    settings: AppSettings,
    position_max_profit: Dict[str, float],
    position_max_price: Dict[str, float],
) -> Optional[str]:
    """
    Проверяет защиту прибыли - закрывает позицию при откате от максимума.
    
    Args:
        client: Bybit клиент
        position: Информация о позиции
        position_bias: Направление позиции
        current_price: Текущая цена
        settings: Настройки бота
        position_max_profit: Словарь максимальной прибыли
        position_max_price: Словарь максимальной цены
    
    Returns:
        Причина для закрытия или None
    """
    try:
        if not settings.risk.enable_profit_protection:
            return None
        
        symbol = settings.symbol
        max_profit_pct = position_max_profit.get(symbol, 0.0)
        max_price = position_max_price.get(symbol, current_price)
        
        # Проверяем, активирована ли защита прибыли
        if max_profit_pct < settings.risk.profit_protection_activation_pct * 100:
            return None
        
        # Рассчитываем откат от максимума
        if position_bias == Bias.LONG:
            retreat_pct = ((max_price - current_price) / max_price) * 100
        else:  # SHORT
            retreat_pct = ((current_price - max_price) / max_price) * 100
        
        # Если откат превышает порог, закрываем позицию
        if retreat_pct >= settings.risk.profit_protection_retreat_pct * 100:
            return f"profit_protection_retreat_{retreat_pct:.2f}%_from_max_{max_profit_pct:.2f}%"
        
        return None
    
    except Exception as e:
        print(f"[live] Error checking profit protection: {e}")
        return None


def _check_position_strategy_alignment(
    client: BybitClient,
    position: Dict[str, Any],
    position_bias: Bias,
    all_signals: list,
    current_price: float,
    settings: AppSettings,
    df_ready: pd.DataFrame,
) -> Optional[str]:
    """
    Проверяет соответствие открытой позиции текущим сигналам стратегий.
    Закрывает позицию ТОЛЬКО при ЭКСТРЕМАЛЬНЫХ условиях:
    - Экстремальные смены тенденций (сильный разворот с движением > 2 ATR)
    - Резкое увеличение объема в обратном направлении (> 2x от среднего + движение > 1.5 ATR)
    - Использует ATR для фильтрации мелких колебаний
    
    Позиция имеет стоп-лосс, поэтому экстренное закрытие только при действительно экстремальных условиях.
    
    Args:
        client: Bybit клиент
        position: Информация о позиции
        position_bias: Направление позиции (LONG или SHORT)
        all_signals: Все текущие сигналы от стратегий
        current_price: Текущая цена
        settings: Настройки бота
        df_ready: DataFrame с индикаторами (для ATR и объема)
    
    Returns:
        Причина для закрытия позиции или None, если закрытие не требуется
    """
    try:
        if not position or not all_signals or df_ready.empty:
            return None
        
        avg_price = position.get("avg_price", 0)
        if avg_price == 0:
            return None
        
        # Получаем последнюю свечу с индикаторами
        last_row = df_ready.iloc[-1]
        # Используем среднее значение ATR с 1H и 4H таймфреймов для среднесрочного анализа
        atr_value = last_row.get("atr_avg", None)  # Среднее ATR с 1H и 4H
        # Fallback на 15M ATR если нет данных с высших таймфреймов
        if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
            atr_value = last_row.get("atr", None)
        
        current_volume = last_row.get("volume", 0)
        vol_sma = last_row.get("vol_sma", 0)
        
        # Если нет ATR, не можем проверить волатильность - не закрываем экстренно
        if pd.isna(atr_value) or atr_value is None or atr_value <= 0:
            # ATR не критичен для работы, просто пропускаем проверку
            atr_value = avg_price * 0.01  # Используем 1% от цены как fallback
            print(f"[live] ⚠️ ATR (1H+4H avg) not available, using fallback: {atr_value:.2f}")
        else:
            atr_1h = last_row.get("atr_1h", 0)
            atr_4h = last_row.get("atr_4h", 0)
            if not pd.isna(atr_1h) and not pd.isna(atr_4h):
                print(f"[live] Using avg ATR(1H+4H): ${atr_value:.2f} (1H: ${atr_1h:.2f}, 4H: ${atr_4h:.2f}) for volatility analysis")
        
        unrealised_pnl = position.get("unrealised_pnl", 0)
        unrealised_pnl_pct = (unrealised_pnl / (position["size"] * avg_price)) * 100 if position["size"] > 0 else 0
        
        # Рассчитываем движение цены в единицах ATR (для фильтрации мелких колебаний)
        price_move = abs(current_price - avg_price)
        price_move_atr = price_move / atr_value if atr_value > 0 else 0
        
        # Проверяем приоритет стратегии для защиты позиции
        # Получаем entry_reason для определения стратегии, которая открыла позицию
        entry_reason = None
        try:
            from bot.web.history import get_open_trade
            # Получаем символ из settings или из позиции
            symbol = getattr(settings, 'symbol', None) or position.get('symbol', None)
            if symbol and avg_price > 0:
                open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
                if open_trade:
                    entry_reason = open_trade.get("entry_reason", "")
        except Exception as e:
            print(f"[live] ⚠️ Error getting entry_reason in _check_position_strategy_alignment: {e}")
        
        # Определяем стратегию, которая открыла позицию
        position_strategy_type = get_strategy_type_from_signal(entry_reason) if entry_reason else None
        
        # Получаем приоритет стратегии из настроек
        strategy_priority = getattr(settings, 'strategy_priority', 'hybrid')
        is_priority_position = position_strategy_type == strategy_priority
        
        # Анализируем текущие сигналы
        # Ищем СИЛЬНЫЕ сигналы на противоположное направление (только OPEN, не ADD)
        strong_opposite_signals = []
        
        for sig in all_signals:
            # Только сильные разворотные сигналы (SHORT при LONG позиции или LONG при SHORT позиции)
            if sig.action == Action.SHORT and position_bias == Bias.LONG:
                # Проверяем, что это действительно сильный сигнал (breakout, bias_flip)
                if "breakout" in sig.reason or "bias_flip" in sig.reason or "trend" in sig.reason:
                    signal_strategy_type = get_strategy_type_from_signal(sig.reason)
                    # Если позиция открыта по приоритетной стратегии, а сигнал от другой стратегии - защищаем позицию
                    if is_priority_position and signal_strategy_type != strategy_priority:
                        print(f"[live] 🛡️ PRIORITY PROTECTION in alignment check: Ignoring opposite SHORT signal from {signal_strategy_type.upper()} (position opened by {strategy_priority.upper()})")
                        continue  # Пропускаем этот сигнал
                    strong_opposite_signals.append(("SHORT", sig.reason))
            elif sig.action == Action.LONG and position_bias == Bias.SHORT:
                if "breakout" in sig.reason or "bias_flip" in sig.reason or "trend" in sig.reason:
                    signal_strategy_type = get_strategy_type_from_signal(sig.reason)
                    # Если позиция открыта по приоритетной стратегии, а сигнал от другой стратегии - защищаем позицию
                    if is_priority_position and signal_strategy_type != strategy_priority:
                        print(f"[live] 🛡️ PRIORITY PROTECTION in alignment check: Ignoring opposite LONG signal from {signal_strategy_type.upper()} (position opened by {strategy_priority.upper()})")
                        continue  # Пропускаем этот сигнал
                    strong_opposite_signals.append(("LONG", sig.reason))
        
        # Принимаем решение о закрытии ТОЛЬКО при экстремальных условиях
        should_close = False
        close_reason = ""
        
        # 1. ЭКСТРЕМАЛЬНАЯ СМЕНА ТЕНДЕНЦИИ: Сильный сигнал на противоположное направление
        # И движение цены против позиции более чем на 2 ATR
        if strong_opposite_signals:
            # Проверяем, что цена движется против позиции на значительную величину (более 2 ATR)
            if position_bias == Bias.LONG:
                price_move_against = avg_price - current_price  # Для LONG: движение вниз = против позиции
            else:  # SHORT
                price_move_against = current_price - avg_price  # Для SHORT: движение вверх = против позиции
            
            price_move_against_atr = price_move_against / atr_value if atr_value > 0 else 0
            
            # Закрываем только если движение против позиции >= 2 ATR (экстремальное движение)
            if price_move_against_atr >= 2.0:
                should_close = True
                reason_type, reason = strong_opposite_signals[0]
                close_reason = f"extreme_trend_reversal_{reason_type.lower()}_{reason}_price_move_{price_move_against_atr:.2f}ATR"
                print(f"[live] 🚨 EXTREME TREND REVERSAL detected:")
                print(f"[live]   Signal: {reason_type} ({reason})")
                print(f"[live]   Price move against position: {price_move_against_atr:.2f} ATR (threshold: 2.0 ATR)")
        
        # 2. РЕЗКОЕ УВЕЛИЧЕНИЕ ОБЪЕМА в обратном направлении
        # Проверяем объем последних свечей
        if not should_close and len(df_ready) >= 3 and vol_sma > 0:
            # Проверяем, есть ли резкий всплеск объема (более 2x от среднего)
            volume_spike = current_volume > (vol_sma * 2.0)
            
            if volume_spike:
                # Проверяем направление движения цены
                if position_bias == Bias.LONG:
                    # Для LONG: объемный всплеск при падении цены = экстремальное условие
                    price_change = current_price - avg_price
                    price_change_atr = price_change / atr_value if atr_value > 0 else 0
                    if price_change < 0 and abs(price_change_atr) >= 1.5:  # Падение более 1.5 ATR
                        should_close = True
                        close_reason = f"extreme_volume_spike_against_long_volume_{current_volume:.0f}_vs_sma_{vol_sma:.0f}_price_move_{price_change_atr:.2f}ATR"
                else:  # SHORT
                    # Для SHORT: объемный всплеск при росте цены = экстремальное условие
                    price_change = current_price - avg_price
                    price_change_atr = price_change / atr_value if atr_value > 0 else 0
                    if price_change > 0 and abs(price_change_atr) >= 1.5:  # Рост более 1.5 ATR
                        should_close = True
                        close_reason = f"extreme_volume_spike_against_short_volume_{current_volume:.0f}_vs_sma_{vol_sma:.0f}_price_move_{price_change_atr:.2f}ATR"
                
                if should_close:
                    print(f"[live] 🚨 EXTREME VOLUME SPIKE detected:")
                    print(f"[live]   Current volume: {current_volume:.0f} vs SMA: {vol_sma:.0f} ({current_volume/vol_sma if vol_sma > 0 else 0:.2f}x)")
                    print(f"[live]   Price move: {price_change_atr:.2f} ATR (threshold: 1.5 ATR)")
        
        if should_close:
            print(f"[live] ⚠️ EMERGENCY CLOSE triggered:")
            print(f"[live]   Position: {position_bias.value} @ ${avg_price:.2f}")
            print(f"[live]   Current price: ${current_price:.2f}")
            print(f"[live]   PnL: ${unrealised_pnl:.2f} ({unrealised_pnl_pct:.2f}%)")
            atr_1h = last_row.get("atr_1h", 0)
            atr_4h = last_row.get("atr_4h", 0)
            atr_source = "avg(1H+4H)" if not pd.isna(atr_1h) and not pd.isna(atr_4h) else "15M fallback"
            print(f"[live]   ATR ({atr_source}): ${atr_value:.2f}, Price move: {price_move_atr:.2f} ATR")
            if not pd.isna(atr_1h) and not pd.isna(atr_4h):
                print(f"[live]   ATR details: 1H=${atr_1h:.2f}, 4H=${atr_4h:.2f}, avg=${atr_value:.2f}")
            print(f"[live]   Close reason: {close_reason}")
            return close_reason
        
        return None
    
    except Exception as e:
        print(f"[live] Error checking position strategy alignment: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_balance(client: BybitClient) -> Optional[float]:
    """Получить доступный баланс USDT."""
    try:
        resp = client.get_wallet_balance(account_type="UNIFIED")
        if resp.get("retCode") != 0:
            print(f"[live] Error getting balance: {resp.get('retMsg', 'Unknown error')}")
            return None
        
        result = resp.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        
        # Для unified account берем первый аккаунт
        account = list_data[0]
        coins = account.get("coin", [])
        
        # Ищем USDT
        for coin in coins:
            if coin.get("coin") == "USDT":
                # Используем usdValue как доступный баланс
                usd_value = coin.get("usdValue", "0")
                try:
                    return float(usd_value)
                except (ValueError, TypeError):
                    return None
        
        return None
    except Exception as e:
        print(f"[live] Error getting balance: {e}")
        return None


def _get_position(client: BybitClient, symbol: str) -> Optional[Dict[str, Any]]:
    """Получить информацию о позиции для символа."""
    try:
        resp = client.get_position_info(symbol=symbol)
        if resp.get("retCode") != 0:
            return None
        
        result = resp.get("result", {})
        list_data = result.get("list", [])
        if not list_data:
            return None
        
        # Ищем позицию с ненулевым размером
        for pos in list_data:
            size = float(pos.get("size", "0") or "0")
            if size > 0:
                # Преобразуем в удобный формат
                return {
                    "side": "long" if pos.get("side") == "Buy" else "short",
                    "size": size,
                    "avg_price": float(pos.get("avgPrice", "0") or "0"),
                    "mark_price": float(pos.get("markPrice", "0") or "0"),
                    "unrealised_pnl": float(pos.get("unrealisedPnl", "0") or "0"),
                    "take_profit": pos.get("takeProfit", ""),
                    "stop_loss": pos.get("stopLoss", ""),
                    "leverage": pos.get("leverage", "1"),
                    "cum_realised_pnl": float(pos.get("cumRealisedPnl", "0") or "0"),
                }
        
        return None
    except Exception as e:
        print(f"[live] Error getting position: {e}")
        return None


def _get_open_orders(client: BybitClient, symbol: str) -> list:
    """Получить список открытых ордеров для символа."""
    try:
        resp = client.get_open_orders(symbol=symbol)
        if resp.get("retCode") != 0:
            return []
        
        result = resp.get("result", {})
        list_data = result.get("list", [])
        return list_data if list_data else []
    except Exception as e:
        print(f"[live] Error getting open orders: {e}")
        return []


def _calculate_order_qty(
    client: BybitClient,
    price: float,
    desired_usd: float,
    settings: AppSettings,
) -> float:
    """
    Рассчитывает количество контрактов для ордера.
    
    Args:
        client: Bybit клиент
        price: Цена входа
        desired_usd: Желаемая сумма в USD
        settings: Настройки бота
    
    Returns:
        Количество контрактов (округленное по qtyStep)
    """
    try:
        # Получаем qtyStep для символа
        qty_step = client.get_qty_step(settings.symbol)
        if qty_step <= 0:
            qty_step = 0.001  # Дефолтное значение
        
        # Рассчитываем количество контрактов с учетом плеча
        # desired_usd уже учитывает процент от баланса и плечо
        total_qty = (desired_usd * settings.leverage) / price
        
        # Округляем по qtyStep с использованием math.floor для точности
        import math
        rounded_qty = math.floor(total_qty / qty_step) * qty_step
        
        # Минимальное количество = qtyStep
        if rounded_qty < qty_step:
            rounded_qty = qty_step
        
        # Дополнительное округление для устранения проблем с float precision
        # Определяем количество знаков после запятой на основе qty_step
        if qty_step >= 1:
            decimals = 0
        elif qty_step >= 0.1:
            decimals = 1
        elif qty_step >= 0.01:
            decimals = 2
        elif qty_step >= 0.001:
            decimals = 3
        else:
            decimals = 6  # Максимум 6 знаков
        
        # Округляем до нужного количества знаков
        rounded_qty = round(rounded_qty, decimals)
        
        return rounded_qty
    except Exception as e:
        print(f"[live] Error calculating order qty: {e}")
        return 0.0


def _calculate_add_position_qty(
    client: BybitClient,
    current_position_size: float,
    settings: AppSettings,
) -> float:
    """
    Рассчитывает количество контрактов для добавления к существующей позиции.
    Количество = половина от текущего размера позиции, округленная в большую сторону.
    
    Args:
        client: Bybit клиент
        current_position_size: Текущий размер позиции
        settings: Настройки бота
    
    Returns:
        Количество контрактов для добавления (округленное по qtyStep в большую сторону)
    """
    try:
        # Получаем qtyStep для символа
        qty_step = client.get_qty_step(settings.symbol)
        if qty_step <= 0:
            qty_step = 0.001  # Дефолтное значение
        
        import math
        # Вычисляем половину от текущего размера позиции
        half_qty = current_position_size / 2.0
        
        # Округляем в большую сторону с учетом qtyStep
        # Используем math.ceil для округления вверх
        rounded_qty = math.ceil(half_qty / qty_step) * qty_step
        
        # Минимальное количество = qtyStep
        if rounded_qty < qty_step:
            rounded_qty = qty_step
        
        # Дополнительное округление для устранения проблем с float precision
        # Определяем количество знаков после запятой на основе qty_step
        if qty_step >= 1:
            decimals = 0
        elif qty_step >= 0.1:
            decimals = 1
        elif qty_step >= 0.01:
            decimals = 2
        elif qty_step >= 0.001:
            decimals = 3
        else:
            decimals = 6  # Максимум 6 знаков
        
        # Округляем до нужного количества знаков
        rounded_qty = round(rounded_qty, decimals)
        
        return rounded_qty
    except Exception as e:
        print(f"[live] Error calculating add position qty: {e}")
        return 0.0


def _get_position_bias_from_position(position: Dict[str, Any]) -> Optional[Bias]:
    """Преобразует side позиции в Bias."""
    if not position:
        return None
    side = position.get("side", "").lower()
    if side == "long":
        return Bias.LONG
    elif side == "short":
        return Bias.SHORT
    return None


def get_strategy_type_from_signal(signal_reason: str) -> str:
    """Определяет тип стратегии по reason сигнала."""
    reason_lower = signal_reason.lower()
    if reason_lower.startswith("ml_"):
        return "ml"
    elif reason_lower.startswith("trend_"):
        return "trend"
    elif reason_lower.startswith("range_"):
        return "flat"
    elif reason_lower.startswith("momentum_"):
        return "momentum"
    elif reason_lower.startswith("liquidity_"):
        return "liquidity"
    elif reason_lower.startswith("smc_"):
        return "smc"
    elif reason_lower.startswith("ict_"):
        return "ict"
    elif reason_lower.startswith("liquidation_hunter_"):
        return "liquidation_hunter"
    elif reason_lower.startswith("zscore_"):
        return "zscore"
    elif reason_lower.startswith("vbo_"):
        return "vbo"
    else:
        return "unknown"


def _check_liquidation_hunter_confirmation(
    signal: Any,
    all_liquidation_hunter_signals: List[Any],
    confirmation_window_minutes: int = 5,
    min_confirmations: int = 2,
    symbol: Optional[str] = None
) -> Tuple[bool, int, List[Any]]:
    """
    Проверяет, есть ли достаточное количество подтверждающих сигналов liquidation_hunter
    в одном направлении за указанный период времени.
    
    Args:
        signal: Сигнал, который нужно проверить
        all_liquidation_hunter_signals: Все сигналы liquidation_hunter
        confirmation_window_minutes: Окно времени для подтверждения (по умолчанию 5 минут)
        min_confirmations: Минимальное количество подтверждений (по умолчанию 2)
        symbol: Символ для логирования
    
    Returns:
        Tuple[bool, int, List[Any]]: (is_confirmed, confirmation_count, confirming_signals)
        - is_confirmed: True если есть достаточное количество подтверждений
        - confirmation_count: Количество подтверждающих сигналов
        - confirming_signals: Список подтверждающих сигналов
    """
    if not signal or not all_liquidation_hunter_signals:
        return False, 0, []
    
    try:
        # Получаем timestamp сигнала
        signal_ts = signal.timestamp
        if isinstance(signal_ts, pd.Timestamp):
            if signal_ts.tzinfo is None:
                signal_ts = signal_ts.tz_localize('UTC')
            else:
                signal_ts = signal_ts.tz_convert('UTC')
            signal_time = signal_ts.to_pydatetime()
        else:
            signal_time = signal_ts
        
        # Вычисляем временное окно: 5 минут ДО текущего сигнала (включая сам сигнал)
        window_start = signal_time - timedelta(minutes=confirmation_window_minutes)
        window_end = signal_time
        
        # Фильтруем сигналы в том же направлении в пределах временного окна
        confirming_signals = []
        for sig in all_liquidation_hunter_signals:
            # Проверяем, что сигнал в том же направлении
            if sig.action != signal.action:
                continue
            
            # Получаем timestamp сигнала
            try:
                sig_ts = sig.timestamp
                if isinstance(sig_ts, pd.Timestamp):
                    if sig_ts.tzinfo is None:
                        sig_ts = sig_ts.tz_localize('UTC')
                    else:
                        sig_ts = sig_ts.tz_convert('UTC')
                    sig_time_check = sig_ts.to_pydatetime()
                elif isinstance(sig_ts, datetime):
                    if sig_ts.tzinfo is None:
                        sig_time_check = sig_ts.replace(tzinfo=timezone.utc)
                    else:
                        sig_time_check = sig_ts
                else:
                    continue  # Пропускаем сигналы с неподдерживаемым форматом timestamp
                
                # Проверяем, что сигнал в пределах временного окна (включая границы)
                if window_start <= sig_time_check <= window_end:
                    confirming_signals.append(sig)
            except Exception:
                continue  # Пропускаем сигналы с ошибками в timestamp
        
        # Сортируем по времени (от старых к новым) для логирования
        def get_sortable_timestamp(s):
            try:
                ts = s.timestamp
                if isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    else:
                        ts = ts.tz_convert('UTC')
                    return ts.to_pydatetime()
                elif isinstance(ts, datetime):
                    return ts
                return ts
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)
        
        confirming_signals.sort(key=get_sortable_timestamp)
        
        confirmation_count = len(confirming_signals)
        is_confirmed = confirmation_count >= min_confirmations
        
        if symbol:
            if is_confirmed:
                _log(f"✅ LIQUIDATION_HUNTER confirmation: {confirmation_count} signals in {signal.action.value} direction within {confirmation_window_minutes} minutes", symbol)
            else:
                _log(f"⚠️ LIQUIDATION_HUNTER confirmation FAILED: only {confirmation_count} signals in {signal.action.value} direction (need {min_confirmations}) within {confirmation_window_minutes} minutes", symbol)
        
        return is_confirmed, confirmation_count, confirming_signals
    
    except Exception as e:
        if symbol:
            _log(f"⚠️ Error checking LIQUIDATION_HUNTER confirmation: {e}", symbol)
        return False, 0, []


def _determine_strategy_with_fallback(
    symbol: str,
    position_strategy: Dict[str, str],
    position: Optional[Dict[str, Any]] = None,
    entry_time: Optional[datetime] = None,
    quiet: bool = False,
) -> str:
    """
    Определяет стратегию с fallback логикой.
    
    Args:
        symbol: Торговая пара
        position_strategy: Словарь стратегий {symbol: strategy}
        position: Информация о позиции (опционально)
        entry_time: Время открытия позиции (опционально)
        quiet: Если True, не выводить предупреждения (для синхронизации)
    
    Returns:
        Тип стратегии: "trend", "flat", "ml", "hybrid", или "unknown"
    """
    # Сначала проверяем сохраненную стратегию
    strategy = position_strategy.get(symbol, "unknown")
    if strategy != "unknown":
        return strategy
    
    # Если стратегия unknown, пытаемся определить по order_link_id
    if position:
        order_link_id = position.get("orderLinkId", "") or position.get("order_link_id", "")
        if order_link_id and order_link_id.startswith("sig_"):
            # Извлекаем signal_id из order_link_id
            # Новый формат: "sig_<signal_id>_<timestamp_ms>"
            # Старый формат: "sig_<signal_id>"
            parts = order_link_id[4:].split("_")  # Убираем префикс "sig_" и разбиваем
            if parts:
                signal_id = parts[0]  # Берем первую часть (signal_id до timestamp)
                
                # Ищем соответствующий сигнал в истории
                try:
                    from bot.web.history import get_signals
                    all_signals = get_signals(limit=1000, symbol_filter=symbol)
                    
                    for hist_signal in all_signals:
                        # Сначала проверяем сохраненный signal_id в истории
                        hist_signal_id = hist_signal.get("signal_id")
                        if hist_signal_id and hist_signal_id == signal_id:
                            hist_strategy = hist_signal.get("strategy_type", "unknown")
                            hist_reason = hist_signal.get("reason", "")
                            if hist_strategy != "unknown":
                                strategy = hist_strategy
                                if not quiet:
                                    print(f"[live] ✅ Strategy determined from order_link_id (signal_id match): {strategy}")
                                return strategy
                            elif hist_reason:
                                strategy = get_strategy_type_from_signal(hist_reason)
                                if strategy != "unknown":
                                    if not quiet:
                                        print(f"[live] ✅ Strategy determined from order_link_id (reason match): {strategy}")
                                    return strategy
                        
                        # Fallback: генерируем signal_id из истории для сравнения
                        if not hist_signal_id:
                            hist_timestamp = hist_signal.get("timestamp", "")
                            hist_reason = hist_signal.get("reason", "")
                            hist_price = hist_signal.get("price", 0)
                            hist_action = hist_signal.get("action", "")
                            
                            # Генерируем signal_id из истории для сравнения
                            if hist_reason and hist_price:
                                import hashlib
                                # Нормализуем timestamp для генерации ID (убираем микросекунды и таймзону)
                                hist_ts_normalized = hist_timestamp
                                if '.' in hist_ts_normalized:
                                    hist_ts_normalized = hist_ts_normalized.split('.')[0]
                                if '+' in hist_ts_normalized:
                                    hist_ts_normalized = hist_ts_normalized.split('+')[0]
                                elif 'Z' in hist_ts_normalized:
                                    hist_ts_normalized = hist_ts_normalized.replace('Z', '')
                                
                                hist_id_string = f"{hist_ts_normalized}_{hist_action}_{hist_reason}_{hist_price:.4f}"
                                hist_signal_id_generated = hashlib.md5(hist_id_string.encode()).hexdigest()[:16]
                                
                                # Если ID совпадает, используем стратегию из истории
                                if hist_signal_id_generated == signal_id:
                                    hist_strategy = hist_signal.get("strategy_type", "unknown")
                                    if hist_strategy != "unknown":
                                        strategy = hist_strategy
                                        if not quiet:
                                            print(f"[live] ✅ Strategy determined from order_link_id (generated signal_id match): {strategy}")
                                        return strategy
                                    else:
                                        strategy = get_strategy_type_from_signal(hist_reason)
                                        if strategy != "unknown":
                                            if not quiet:
                                                print(f"[live] ✅ Strategy determined from order_link_id (reason match): {strategy}")
                                            return strategy
                except Exception as e:
                    print(f"[live] ⚠️ Error determining strategy from order_link_id in fallback: {e}")
    
    # Если стратегия все еще unknown, пытаемся определить по времени открытия позиции
    if entry_time:
        try:
            from bot.web.history import get_signals
            # Ищем сигналы в диапазоне ±60 минут от времени открытия позиции (увеличено для лучшего поиска)
            time_window = timedelta(minutes=60)
            all_signals = get_signals(limit=1000, symbol_filter=symbol)
            
            # Сортируем сигналы по времени (новые первыми)
            signals_with_time = []
            for hist_signal in all_signals:
                hist_time_str = hist_signal.get("timestamp", "")
                if not hist_time_str:
                    continue
                
                try:
                    if isinstance(hist_time_str, str):
                        if 'T' in hist_time_str:
                            hist_time = datetime.fromisoformat(hist_time_str.replace('Z', '+00:00'))
                        else:
                            try:
                                hist_time = datetime.strptime(hist_time_str, '%Y-%m-%d %H:%M:%S')
                                hist_time = hist_time.replace(tzinfo=timezone.utc)
                            except ValueError:
                                hist_time = datetime.fromisoformat(hist_time_str.replace('Z', '+00:00'))
                    else:
                        hist_time = datetime.fromisoformat(str(hist_time_str).replace('Z', '+00:00'))
                    
                    if hist_time.tzinfo is None:
                        hist_time = hist_time.replace(tzinfo=timezone.utc)
                    else:
                        hist_time = hist_time.astimezone(timezone.utc)
                    
                    signals_with_time.append((hist_time, hist_signal))
                except Exception:
                    continue
            
            # Сортируем по времени (новые первыми) и ищем ближайший сигнал
            signals_with_time.sort(key=lambda x: x[0], reverse=True)
            
            for hist_time, hist_signal in signals_with_time:
                # Проверяем, попадает ли время сигнала в окно времени открытия позиции
                time_diff = abs((hist_time - entry_time).total_seconds())
                if time_diff <= time_window.total_seconds():
                    hist_strategy = hist_signal.get("strategy_type", "unknown")
                    hist_reason = hist_signal.get("reason", "")
                    hist_action = hist_signal.get("action", "").lower()
                    
                    # Предпочитаем сигналы LONG/SHORT (не HOLD)
                    if hist_action in ("long", "short"):
                        if hist_strategy != "unknown":
                            strategy = hist_strategy
                            if not quiet:
                                print(f"[live] ✅ Strategy determined from signal history (time window, strategy match): {strategy} (time diff: {time_diff:.0f}s)")
                            break
                        elif hist_reason:
                            strategy = get_strategy_type_from_signal(hist_reason)
                            if strategy != "unknown":
                                if not quiet:
                                    print(f"[live] ✅ Strategy determined from signal history (time window, reason match): {strategy} (time diff: {time_diff:.0f}s)")
                                break
        except Exception as e:
            print(f"[live] ⚠️ Error determining strategy from signal history in fallback: {e}")
    
    return strategy


def _sync_closed_positions_from_bybit(
    client: BybitClient,
    symbol: str,
    last_sync_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Синхронизирует закрытые позиции из Bybit с историей сделок.
    
    Args:
        client: Bybit клиент
        symbol: Торговая пара
        last_sync_time: Время последней синхронизации (опционально)
    
    Returns:
        Список синхронизированных сделок
    """
    try:
        # Получаем закрытые позиции (API Bybit ограничивает период до 7 дней)
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        if last_sync_time:
            start_time = int(last_sync_time.timestamp() * 1000)
        else:
            # По умолчанию за последние 7 дней (максимум для API)
            start_time = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        
        # ВАЖНО: Всегда ограничиваем период до 7 дней (API Bybit не позволяет больше)
        time_diff_ms = end_time - start_time
        max_period_ms = 7 * 24 * 60 * 60 * 1000  # 7 дней в миллисекундах
        if time_diff_ms > max_period_ms:
            # Ограничиваем до 7 дней от текущего времени
            start_time = end_time - max_period_ms
        
        # Получаем закрытые позиции
        closed_pnl_resp = client.get_closed_pnl(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=100,
        )
        
        if closed_pnl_resp.get("retCode") != 0:
            print(f"[live] ⚠️ Failed to get closed PnL: {closed_pnl_resp.get('retMsg', 'Unknown error')}")
            return []
        
        result = closed_pnl_resp.get("result", {})
        closed_positions = result.get("list", [])
        
        if not closed_positions:
            return []
        
        synced_trades = []
        
        for pos in closed_positions:
            try:
                # Парсим данные о закрытой позиции
                side = pos.get("side", "").lower()
                if side not in ["buy", "sell"]:
                    continue
                
                closed_size = float(pos.get("closedSize", 0))
                if closed_size == 0:
                    continue
                
                avg_entry_price = float(pos.get("avgEntryPrice", 0))
                avg_exit_price = float(pos.get("avgExitPrice", 0))
                if avg_entry_price == 0 or avg_exit_price == 0:
                    continue
                
                closed_pnl = float(pos.get("closedPnl", 0))
                
                # Определяем причину закрытия
                exit_reason = "unknown"
                if pos.get("takeProfit"):
                    exit_reason = "take_profit_auto"
                elif pos.get("stopLoss"):
                    exit_reason = "stop_loss_auto"
                else:
                    # Проверяем по PnL - если отрицательный, скорее всего SL
                    if closed_pnl < 0:
                        exit_reason = "stop_loss_auto"
                    else:
                        exit_reason = "take_profit_auto"
                
                # Парсим время
                created_time = pos.get("createdTime", "")
                updated_time = pos.get("updatedTime", created_time)
                
                try:
                    if isinstance(created_time, str):
                        if created_time.isdigit():
                            entry_time = datetime.fromtimestamp(int(created_time) / 1000, tz=timezone.utc)
                        else:
                            entry_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    else:
                        entry_time = datetime.fromtimestamp(int(created_time) / 1000, tz=timezone.utc) if created_time else datetime.now(timezone.utc)
                    
                    if isinstance(updated_time, str):
                        if updated_time.isdigit():
                            exit_time = datetime.fromtimestamp(int(updated_time) / 1000, tz=timezone.utc)
                        else:
                            exit_time = datetime.fromisoformat(updated_time.replace('Z', '+00:00'))
                    else:
                        exit_time = datetime.fromtimestamp(int(updated_time) / 1000, tz=timezone.utc) if updated_time else datetime.now(timezone.utc)
                except Exception:
                    entry_time = datetime.now(timezone.utc)
                    exit_time = datetime.now(timezone.utc)
                
                # Определяем стратегию из order_link_id или истории сигналов
                strategy_type = "unknown"
                
                # Получаем orderLinkId из закрытой позиции (сохраняем для использования позже)
                order_link_id = pos.get("orderLinkId") or pos.get("order_link_id") or ""
                
                # Пытаемся извлечь signal_id из order_link_id
                if order_link_id and order_link_id.startswith("sig_"):
                    # Извлекаем signal_id из order_link_id
                    # Новый формат: "sig_<signal_id>_<timestamp_ms>"
                    # Старый формат: "sig_<signal_id>"
                    parts = order_link_id[4:].split("_")  # Убираем префикс "sig_" и разбиваем
                    if parts:
                        signal_id = parts[0]  # Берем первую часть (signal_id до timestamp)
                    else:
                        signal_id = order_link_id[4:]  # Fallback для старого формата
                    
                    # Ищем соответствующий сигнал в истории
                    try:
                        from bot.web.history import get_signals
                        all_signals = get_signals(limit=1000, symbol_filter=symbol)  # Берем только для этого символа
                        
                        for hist_signal in all_signals:
                            # Проверяем, соответствует ли signal_id сигналу из истории
                            # Сначала проверяем сохраненный signal_id в истории
                            hist_signal_id = hist_signal.get("signal_id")
                            if hist_signal_id and hist_signal_id == signal_id:
                                hist_strategy = hist_signal.get("strategy_type", "unknown")
                                hist_reason = hist_signal.get("reason", "")
                                if hist_strategy != "unknown":
                                    strategy_type = hist_strategy
                                    break
                                else:
                                    strategy_type = get_strategy_type_from_signal(hist_reason)
                                    break
                            
                            # Fallback: если signal_id не сохранен в истории, генерируем его для сравнения
                            if not hist_signal_id:
                                hist_timestamp = hist_signal.get("timestamp", "")
                                hist_reason = hist_signal.get("reason", "")
                                hist_price = hist_signal.get("price", 0)
                                hist_action = hist_signal.get("action", "")
                                
                                # Генерируем signal_id из истории для сравнения
                                if hist_reason and hist_price:
                                    import hashlib
                                    # Нормализуем timestamp для генерации ID (убираем микросекунды и таймзону)
                                    hist_ts_normalized = hist_timestamp
                                    if '.' in hist_ts_normalized:
                                        hist_ts_normalized = hist_ts_normalized.split('.')[0]
                                    if '+' in hist_ts_normalized:
                                        hist_ts_normalized = hist_ts_normalized.split('+')[0]
                                    elif 'Z' in hist_ts_normalized:
                                        hist_ts_normalized = hist_ts_normalized.replace('Z', '')
                                    
                                    hist_id_string = f"{hist_ts_normalized}_{hist_action}_{hist_reason}_{hist_price:.4f}"
                                    hist_signal_id_generated = hashlib.md5(hist_id_string.encode()).hexdigest()[:16]
                                    
                                    # Если ID совпадает, используем стратегию из истории
                                    if hist_signal_id_generated == signal_id:
                                        hist_strategy = hist_signal.get("strategy_type", "unknown")
                                        if hist_strategy != "unknown":
                                            strategy_type = hist_strategy
                                            break
                                        else:
                                            strategy_type = get_strategy_type_from_signal(hist_reason)
                                            break
                    except Exception as e:
                        print(f"[live] ⚠️ Error determining strategy from order_link_id: {e}")
                
                # Если стратегия все еще unknown, пытаемся определить по времени открытия позиции
                # Проверяем, есть ли сигналы в истории около времени открытия позиции
                if strategy_type == "unknown":
                    try:
                        from bot.web.history import get_signals
                        # Расширяем временное окно до ±60 минут (увеличено с 30 для надежности)
                        time_window = timedelta(minutes=60)
                        all_signals = get_signals(limit=2000, symbol_filter=symbol)  # Берем только для этого символа
                        
                        for hist_signal in all_signals:
                            hist_time_str = hist_signal.get("timestamp", "")
                            if not hist_time_str:
                                continue
                            
                            try:
                                if isinstance(hist_time_str, str):
                                    if 'T' in hist_time_str:
                                        hist_time = datetime.fromisoformat(hist_time_str.replace('Z', '+00:00'))
                                    else:
                                        hist_time = datetime.strptime(hist_time_str, '%Y-%m-%d %H:%M:%S')
                                        hist_time = hist_time.replace(tzinfo=timezone.utc)
                                else:
                                    continue
                                
                                if hist_time.tzinfo is None:
                                    hist_time = hist_time.replace(tzinfo=timezone.utc)
                                else:
                                    hist_time = hist_time.astimezone(timezone.utc)
                                
                                # Проверяем, попадает ли сигнал во временное окно открытия позиции
                                time_diff = abs((entry_time - hist_time).total_seconds())
                                if time_diff <= time_window.total_seconds():
                                    # Проверяем, совпадает ли цена и сторона
                                    hist_price = hist_signal.get("price", 0)
                                    hist_side = hist_signal.get("action", "").lower()
                                    price_diff = abs(hist_price - avg_entry_price)
                                    
                                    # Если цена близка (±1%) и сторона совпадает, используем стратегию из сигнала
                                    if price_diff / avg_entry_price < 0.01 and (
                                        (hist_side == "long" and side == "buy") or 
                                        (hist_side == "short" and side == "sell")
                                    ):
                                        hist_strategy = hist_signal.get("strategy_type", "unknown")
                                        if hist_strategy != "unknown":
                                            strategy_type = hist_strategy
                                        else:
                                            strategy_type = get_strategy_type_from_signal(hist_signal.get("reason", ""))
                                        break
                            except Exception:
                                continue
                    except Exception as e:
                        print(f"[live] ⚠️ Error determining strategy from signal history: {e}")
                
                # Если стратегия все еще unknown, пытаемся определить с fallback логикой (без предупреждений)
                if strategy_type == "unknown":
                    # Используем упрощенную версию без предупреждений для синхронизации
                    try:
                        strategy_type = _determine_strategy_with_fallback(
                            symbol,
                            {},  # position_strategy пустой для синхронизированных позиций
                            pos,  # Передаем данные позиции
                            entry_time=entry_time,
                            quiet=True,  # Подавляем предупреждения при синхронизации
                        )
                    except Exception:
                        pass  # Тихо игнорируем ошибки
                
                # Получаем orderId - это ID ордера, который закрыл позицию (market order, TP или SL)
                # Пытаемся получить из нескольких источников:
                # 1. Из самого закрытого PnL (если есть поле orderId)
                # 2. Из истории исполненных ордеров (execution list)
                # 3. Из истории ордеров (order history) для TP/SL ордеров
                order_id = None
                
                # Попытка 1: Проверяем, есть ли orderId прямо в закрытом PnL
                order_id_from_pnl = pos.get("orderId") or pos.get("order_id")
                if order_id_from_pnl:
                    order_id = order_id_from_pnl
                
                # Попытка 2: Ищем в истории исполненных ордеров
                if not order_id:
                    try:
                        exec_start_time = int((exit_time - timedelta(minutes=15)).timestamp() * 1000)
                        exec_end_time = int((exit_time + timedelta(minutes=5)).timestamp() * 1000)
                        exec_resp = client.get_execution_list(
                            symbol=symbol,
                            start_time=exec_start_time,
                            end_time=exec_end_time,
                            limit=200,  # Увеличено для лучшего поиска
                        )
                        if exec_resp.get("retCode") == 0:
                            exec_result = exec_resp.get("result", {})
                            exec_list = exec_result.get("list", [])
                            # Ищем исполненный ордер, который закрыл позицию
                            for exec_order in exec_list:
                                exec_order_id = exec_order.get("orderId")
                                exec_reduce_only = exec_order.get("reduceOnly", False)
                                exec_qty = float(exec_order.get("execQty", 0))
                                exec_side = exec_order.get("side", "").lower()
                                exec_time = exec_order.get("execTime", 0)
                                
                                # Проверяем, что это reduceOnly ордер (закрытие позиции)
                                if exec_reduce_only and exec_order_id:
                                    # Проверяем соответствие стороны (Sell для long позиции, Buy для short)
                                    side_match = (side == "buy" and exec_side == "sell") or (side == "sell" and exec_side == "buy")
                                    # Проверяем время (должно быть близко к времени закрытия)
                                    if exec_time:
                                        exec_time_dt = datetime.fromtimestamp(int(exec_time) / 1000, tz=timezone.utc)
                                        time_diff = abs((exec_time_dt - exit_time).total_seconds())
                                        time_ok = time_diff <= 300  # 5 минут
                                    else:
                                        time_ok = True
                                    
                                    if side_match and time_ok:
                                        # Проверяем, что количество примерно совпадает (допуск ±15%)
                                        if abs(exec_qty - closed_size) / max(closed_size, 0.001) < 0.15:
                                            order_id = exec_order_id
                                            break
                    except Exception as e:
                        pass  # Тихо игнорируем ошибки при поиске Order ID
                
                # Попытка 3: Ищем в истории ордеров (для TP/SL ордеров)
                if not order_id:
                    try:
                        order_start_time = int((exit_time - timedelta(minutes=30)).timestamp() * 1000)
                        order_end_time = int((exit_time + timedelta(minutes=5)).timestamp() * 1000)
                        order_resp = client.get_order_history(
                            symbol=symbol,
                            start_time=order_start_time,
                            end_time=order_end_time,
                            limit=200,
                            order_status="Filled",  # Только исполненные ордера
                        )
                        if order_resp.get("retCode") == 0:
                            order_result = order_resp.get("result", {})
                            order_list = order_result.get("list", [])
                            # Ищем ордер, который закрыл позицию (reduceOnly, TP или SL)
                            for order_item in order_list:
                                order_item_id = order_item.get("orderId")
                                order_reduce_only = order_item.get("reduceOnly", False)
                                order_side = order_item.get("side", "").lower()
                                order_qty = float(order_item.get("qty", 0))
                                order_type = order_item.get("orderType", "").lower()
                                
                                # Проверяем, что это reduceOnly ордер или TP/SL
                                if order_item_id and (order_reduce_only or order_type in ("takeprofit", "stoploss")):
                                    # Проверяем соответствие стороны
                                    side_match = (side == "buy" and order_side == "sell") or (side == "sell" and order_side == "buy")
                                    # Проверяем количество
                                    qty_match = abs(order_qty - closed_size) / max(closed_size, 0.001) < 0.15
                                    
                                    if side_match and qty_match:
                                        order_id = order_item_id
                                        break
                    except Exception as e:
                        pass  # Тихо игнорируем ошибки при поиске Order ID
                
                # orderLinkId мы уже получили выше при определении стратегии
                # Преобразуем пустую строку в None, если orderLinkId пустой
                if not order_link_id or order_link_id == "":
                    order_link_id = None
                
                # Добавляем сделку в историю (с проверкой дубликатов внутри add_trade)
                size_usd = closed_size * avg_exit_price
                
                add_trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    side="long" if side == "buy" else "short",
                    entry_price=avg_entry_price,
                    exit_price=avg_exit_price,
                    size_usd=size_usd,
                    pnl=closed_pnl,
                    entry_reason="auto_synced",
                    exit_reason=exit_reason,
                    strategy_type=strategy_type,
                    symbol=symbol,
                    order_id=order_id,
                    order_link_id=order_link_id,
                )
                
                synced_trades.append({
                    "side": side,
                    "pnl": closed_pnl,
                    "exit_reason": exit_reason,
                    "exit_time": exit_time,
                    "strategy_type": strategy_type,  # Сохраняем стратегию для подсчета
                })
                
            except Exception as e:
                print(f"[live] ⚠️ Error syncing closed position: {e}")
                continue
        
        if synced_trades:
            # Убрали избыточное логирование синхронизации - слишком много сообщений
            # Подсчитываем сколько позиций с неизвестной стратегией
            # unknown_count = sum(1 for trade in synced_trades if trade.get("strategy_type") == "unknown")
            # if unknown_count > 0 and unknown_count < len(synced_trades):
            #     print(f"[live] ✅ Synced {len(synced_trades)} closed positions from Bybit ({unknown_count} with unknown strategy)")
            # elif unknown_count == len(synced_trades) and len(synced_trades) > 0:
            #     print(f"[live] ✅ Synced {len(synced_trades)} closed positions from Bybit (all with unknown strategy - no signal history found)")
            # else:
            #     print(f"[live] ✅ Synced {len(synced_trades)} closed positions from Bybit")
            pass  # Убрали логирование, но оставили структуру
        
        return synced_trades
    
    except Exception as e:
        print(f"[live] ⚠️ Error syncing closed positions: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_live_from_api(
    initial_settings: AppSettings,
    bot_state: Optional[Dict[str, Any]] = None,
    signal_max_age_seconds: int = 60,
    symbol: Optional[str] = None,  # НОВЫЙ ПАРАМЕТР: явно указываем символ для этого воркера
    stop_event: Optional[threading.Event] = None,  # Событие для остановки воркера
) -> None:
    """
    Основной цикл live-торговли.
    
    Args:
        initial_settings: Начальные настройки
        bot_state: Словарь для обмена состоянием с веб-интерфейсом
        signal_max_age_seconds: Максимальный возраст сигнала для обработки (в секундах)
        symbol: Торговая пара (если None, используется initial_settings.symbol для обратной совместимости)
    """
    from bot.shared_settings import get_settings
    
    # Определяем символ для этого воркера
    # Если symbol не задан явно, используем из settings (обратная совместимость)
    if symbol is None:
        symbol = initial_settings.symbol
    
    # Создаем локальную копию настроек с переопределенным символом
    # Это нужно для сохранения обратной совместимости
    # ВАЖНО: primary_symbol НЕ переопределяем - он должен оставаться глобальным PRIMARY_SYMBOL
    import copy
    local_settings = copy.deepcopy(initial_settings)
    local_settings.symbol = symbol
    # primary_symbol остается из initial_settings (глобальный PRIMARY_SYMBOL)
    
    # Инициализируем bot_state, если он None (для multi-symbol режима)
    # Важно: bot_state всегда должен быть словарем, даже если передан None
    if bot_state is None or bot_state is False:
        bot_state = {
            "is_running": False,
            "current_status": "Stopped",
            "current_phase": None,
            "current_adx": None,
            "last_action": None,
            "last_action_time": None,
            "last_signal": None,
            "last_signal_time": None,
            "last_error": None,
            "last_error_time": None,
            "last_update": None,
        }
    elif not isinstance(bot_state, dict):
        # Если bot_state не словарь и не None, создаем новый словарь
        _log(f"⚠️ bot_state is not a dict: {type(bot_state)}, initializing new dict", symbol)
        bot_state = {
            "is_running": False,
            "current_status": "Stopped",
            "current_phase": None,
            "current_adx": None,
            "last_action": None,
            "last_action_time": None,
            "last_signal": None,
            "last_signal_time": None,
            "last_error": None,
            "last_error_time": None,
            "last_update": None,
        }
    
    # Инициализация
    client = BybitClient(local_settings.api)
    # Загружаем обработанные сигналы из файла (для персистентности между перезапусками)
    # Используем отдельный файл для каждого символа
    processed_signals_file = Path(__file__).parent.parent / f"processed_signals_{symbol}.json"
    processed_signals = _load_processed_signals(processed_signals_file)
    
    # Загружаем сохраненное состояние бота (если есть)
    saved_state = _load_bot_state(symbol)
    
    position_max_profit: Dict[str, float] = {}
    position_max_price: Dict[str, float] = {}
    position_partial_closed: Dict[str, bool] = {}
    position_strategy: Dict[str, str] = {symbol: saved_state.get("strategy_type", "unknown")}
    position_order_id: Dict[str, str] = {symbol: saved_state.get("order_id", "")}
    position_order_link_id: Dict[str, str] = {symbol: saved_state.get("order_link_id", "")}
    position_add_count: Dict[str, int] = {symbol: saved_state.get("add_count", 0)}
    position_entry_price: Dict[str, float] = {symbol: saved_state.get("entry_price", 0.0)}
    last_handled_signal: Optional[tuple] = None  # (timestamp, action)
    seen_signal_keys_cycle: set = set()  # Отслеживание сохраненных сигналов за цикл
    previous_position: Optional[Dict[str, Any]] = None  # Хранит предыдущую позицию для обнаружения закрытия
    position_opened_time: Optional[datetime] = None  # Время открытия последней позиции (для защиты от ложных закрытий)
    
    # Устанавливаем плечо
    try:
        client.set_leverage(symbol, local_settings.leverage)
        print(f"[live] [{symbol}] Leverage set to {local_settings.leverage}x")
    except Exception as e:
        print(f"[live] [{symbol}] Warning: Failed to set leverage: {e}")
    
    # Обновляем bot_state (теперь он всегда словарь)
    if bot_state is not None:
        bot_state["is_running"] = True
        bot_state["current_status"] = "Starting..."
        bot_state["last_action"] = "Initializing..."
        bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
    
    print(f"[live] [{symbol}] ========================================")
    print(f"[live] [{symbol}] 🚀 Starting live trading bot for {symbol}")
    # Получаем настройки стратегий для текущей пары
    symbol_strategy_settings = local_settings.get_strategy_settings_for_symbol(symbol)
    print(
        f"[live] [{symbol}] 📊 Active strategies: "
        f"Trend={symbol_strategy_settings.enable_trend_strategy}, "
        f"Flat={symbol_strategy_settings.enable_flat_strategy}, "
        f"ML={symbol_strategy_settings.enable_ml_strategy}, "
        f"Momentum={symbol_strategy_settings.enable_momentum_strategy}, "
        f"Liquidity={symbol_strategy_settings.enable_liquidity_sweep_strategy}, "
        f"SMC={symbol_strategy_settings.enable_smc_strategy}, "
        f"ICT={symbol_strategy_settings.enable_ict_strategy}, "
        f"LiquidationHunter={symbol_strategy_settings.enable_liquidation_hunter_strategy}, "
        f"ZScore={symbol_strategy_settings.enable_zscore_strategy}, "
        f"VBO={symbol_strategy_settings.enable_vbo_strategy}, "
        f"AMT_OF={symbol_strategy_settings.enable_amt_of_strategy}"
    )
    print(f"[live] [{symbol}] ⚙️  Leverage: {local_settings.leverage}x, Max position: ${local_settings.risk.max_position_usd}")
    print(f"[live] [{symbol}] ========================================")
    
    # Синхронизируем закрытые позиции при старте (API Bybit ограничивает до 7 дней)
    # Синхронизируем только последние 7 дней при старте, остальное будет синхронизировано периодически
    try:
        sync_start = datetime.now(timezone.utc) - timedelta(days=7)
        synced = _sync_closed_positions_from_bybit(client, symbol, sync_start)
        # Убрали избыточное логирование - слишком много сообщений при старте
        # if len(synced) > 0:
        #     print(f"[live] [{symbol}] ✅ Synced {len(synced)} closed positions from Bybit on startup (last 7 days)")
    except Exception as e:
        # Подавляем ошибки о превышении лимита, если они все еще возникают
        if "cannot exceed 7 days" not in str(e):
            print(f"[live] [{symbol}] ⚠️ Error syncing closed positions on startup: {e}")
    
    # Проверяем существующие открытые позиции при старте и устанавливаем TP/SL
    print(f"[live] [{symbol}] 🔍 Checking for existing open positions...")
    try:
        startup_position = _get_position(client, symbol)
        previous_position = startup_position.copy() if startup_position else None  # Инициализируем previous_position
        if startup_position:
            startup_bias = _get_position_bias_from_position(startup_position)
            if startup_bias:
                # Получаем текущую цену
                ticker_resp = client.session.get_tickers(category="linear", symbol=symbol)
                if ticker_resp.get("retCode") == 0:
                    result = ticker_resp.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        startup_price = float(list_data[0].get("lastPrice", "0") or "0")
                        if startup_price > 0:
                            print(f"[live] [{symbol}] 📊 Found existing {startup_bias.value} position, taking control...")
                            print(f"[live] [{symbol}]   Entry: ${startup_position.get('avg_price', 0):.2f}, Current: ${startup_price:.2f}, Size: {startup_position.get('size', 0):.3f}")
                            
                            # Проверяем, есть ли открытая сделка в истории для этой позиции
                            try:
                                from bot.web.history import _load_history
                                history = _load_history()
                                trades = history.get("trades", [])
                                
                                position_side_normalized = "long" if startup_bias == Bias.LONG else "short"
                                open_trades = [
                                    t for t in trades
                                    if t.get("symbol", "").upper() == symbol.upper() and
                                    t.get("side", "").lower() == position_side_normalized and
                                    (not t.get("exit_time") or t.get("exit_time") == "" or t.get("exit_time") is None)
                                ]
                                
                                if not open_trades:
                                    # Нет открытой сделки в истории - пытаемся найти последний сигнал и создать сделку
                                    signals = history.get("signals", [])
                                    matching_signals = [
                                        s for s in signals
                                        if s.get("symbol", "").upper() == symbol.upper() and
                                        s.get("action", "").lower() == position_side_normalized and
                                        abs(float(s.get("price", 0)) - startup_position.get('avg_price', 0)) / startup_position.get('avg_price', 1) < 0.05  # В пределах 5%
                                    ]
                                    
                                    if matching_signals:
                                        # Берем последний подходящий сигнал
                                        matching_signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                                        last_signal = matching_signals[0]
                                        
                                        entry_price = startup_position.get('avg_price', float(last_signal.get("price", 0)))
                                        size_usd = startup_position.get("size", 0) * entry_price
                                        
                                        add_trade(
                                            entry_time=last_signal.get("timestamp", datetime.now()),
                                            exit_time=None,  # Позиция еще открыта
                                            side=position_side_normalized,
                                            entry_price=entry_price,
                                            exit_price=0.0,
                                            size_usd=size_usd,
                                            pnl=0.0,
                                            entry_reason=last_signal.get("reason", "unknown"),
                                            exit_reason="",
                                            strategy_type=last_signal.get("strategy_type", "unknown"),
                                            symbol=symbol,
                                            order_id="",
                                            order_link_id="",
                                        )
                                        print(f"[live] [{symbol}] 💾 Created open trade from last signal: {last_signal.get('strategy_type', 'unknown')} @ ${entry_price:.2f} ({last_signal.get('reason', 'unknown')})")
                                    else:
                                        # Создаем сделку с базовой информацией
                                        entry_price = startup_position.get('avg_price', startup_price)
                                        size_usd = startup_position.get("size", 0) * entry_price
                                        
                                        add_trade(
                                            entry_time=datetime.now(),
                                            exit_time=None,
                                            side=position_side_normalized,
                                            entry_price=entry_price,
                                            exit_price=0.0,
                                            size_usd=size_usd,
                                            pnl=0.0,
                                            entry_reason="existing_position",
                                            exit_reason="",
                                            strategy_type="unknown",
                                            symbol=symbol,
                                            order_id="",
                                            order_link_id="",
                                        )
                                        print(f"[live] [{symbol}] 💾 Created open trade for existing position: unknown @ ${entry_price:.2f}")
                            except Exception as e:
                                print(f"[live] [{symbol}] ⚠️ Error checking/creating open trade: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Инициализируем отслеживание прибыли для существующей позиции
                            _update_position_tracking(
                                startup_position,
                                startup_bias,
                                startup_price,
                                position_max_profit,
                                position_max_price,
                                symbol,
                            )
                            
                            # Устанавливаем TP/SL для существующей позиции
                            _ensure_tp_sl_set(
                                client=client,
                                position=startup_position,
                                settings=local_settings,
                                position_bias=startup_bias,
                                current_price=startup_price,
                                position_max_profit=position_max_profit,
                                position_max_price=position_max_price,
                            )
                            print(f"[live] [{symbol}] ✅ TP/SL management activated for existing position")
                        else:
                            print(f"[live] [{symbol}] ⚠️ Could not get current price for existing position")
                    else:
                        print(f"[live] [{symbol}] ⚠️ No ticker data for existing position")
                else:
                    print(f"[live] [{symbol}] ⚠️ Error getting ticker for existing position: {ticker_resp.get('retMsg', 'Unknown error')}")
        else:
            print(f"[live] [{symbol}] ✅ No existing open positions found")
    except Exception as e:
        print(f"[live] [{symbol}] ⚠️ Error checking existing positions on startup: {e}")
        import traceback
        traceback.print_exc()
    
    # Счетчик циклов для периодической синхронизации
    sync_counter = 0
    sync_interval = 10  # Синхронизировать каждые 10 циклов
    
    print(f"[live] [{symbol}] ✅ Bot initialized successfully - entering main trading loop")
    print(f"[live] [{symbol}] 🔄 Starting main trading loop (poll interval: {local_settings.live_poll_seconds}s)...")
    print(f"[live] [{symbol}] ✨ Bot is ACTIVE and monitoring {symbol} market! ✨")
    
    # Функция для получения timestamp для сортировки (определяем ДО основного цикла, чтобы использовать везде)
    def get_timestamp_for_sort(sig):
        """Получает timestamp для сортировки сигнала."""
        ts = sig.timestamp
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            return ts.to_pydatetime()
        elif hasattr(ts, 'timestamp'):
            return ts
        else:
            return pd.Timestamp(ts).to_pydatetime()
    
    while True:
        try:
            # Флаг для отслеживания обработки свежих сигналов (для оптимизации интервала ожидания)
            fresh_signal_processed = False
            
            # Обновляем статус воркера для мониторинга (если используется MultiSymbolManager)
            # ВАЖНО: Обновляем статус в начале каждой итерации, чтобы MultiSymbolManager не считал воркер "мертвым"
            try:
                from bot.multi_symbol_manager import update_worker_status
                # Обновляем статус воркера в начале каждой итерации
                update_worker_status(symbol, current_status="Running", last_action="Processing signals...", error=None)
            except ImportError:
                pass  # MultiSymbolManager может быть не импортирован
            
            # Получаем актуальные настройки из shared_settings
            # ВАЖНО: обновляем только те настройки, которые не зависят от символа
            current_settings_raw = get_settings() or local_settings
            
            # ОПТИМИЗАЦИЯ: Используем replace вместо медленного deepcopy
            from dataclasses import replace
            current_settings = replace(
                current_settings_raw,
                symbol=symbol,
                # ВАЖНО: primary_symbol НЕ переопределяем - он должен оставаться глобальным PRIMARY_SYMBOL
                # primary_symbol остается из current_settings_raw (глобальный PRIMARY_SYMBOL из .env)
            )
            # Копируем вложенные dataclasses для независимости
            current_settings.strategy = replace(current_settings_raw.strategy)
            current_settings.risk = replace(current_settings_raw.risk)
            current_settings.api = replace(current_settings_raw.api)
            
            # ВАЖНО: Если этот воркер запущен через MultiSymbolManager, у local_settings
            # уже есть корректный ml_model_path для конкретного symbol.
            # В таком случае НЕ даем глобальным настройкам перезаписать путь модели
            try:
                local_model_path = getattr(local_settings, "ml_model_path", None)
                if local_model_path:
                    model_filename = Path(local_model_path).name
                    if "_" in model_filename:
                        parts = model_filename.replace(".pkl", "").split("_")
                        if len(parts) >= 2 and parts[1] == symbol:
                            # local_settings содержит модель именно для этого символа → используем её
                            current_settings.ml_model_path = local_model_path
            except Exception:
                # В случае любой ошибки просто продолжаем с current_settings как есть
                pass
            
            # ВАЖНО: Обновляем ml_model_path для текущего символа, если ML стратегия включена
            # Это необходимо, потому что ml_model_path зависит от символа
            # НО: если модель уже установлена (например, MultiSymbolManager), не переопределяем её
            if current_settings.enable_ml_strategy and not current_settings.ml_model_path:
                try:
                    models_dir = Path(__file__).parent.parent / "ml_models"
                    if models_dir.exists():
                        found_model = None
                        model_type_preference = getattr(current_settings, 'ml_model_type_for_all', None)
                        
                        if model_type_preference:
                            # Если задан тип модели, ищем только этот тип
                            pattern = f"{model_type_preference}_{symbol}_*.pkl"
                            for model_file in sorted(models_dir.glob(pattern), reverse=True):
                                if model_file.is_file():
                                    found_model = str(model_file)
                                    break
                        else:
                            # Автоматический выбор: предпочитаем ensemble > rf > xgb
                            # Сначала ищем ensemble
                            for model_file in sorted(models_dir.glob(f"ensemble_{symbol}_*.pkl"), reverse=True):
                                if model_file.is_file():
                                    found_model = str(model_file)
                                    break
                            
                            # Если ensemble не найден, пробуем rf_
                            if not found_model:
                                for model_file in sorted(models_dir.glob(f"rf_{symbol}_*.pkl"), reverse=True):
                                    if model_file.is_file():
                                        found_model = str(model_file)
                                        break
                            
                            # Если rf_ модель не найдена, пробуем xgb_
                            if not found_model:
                                for model_file in sorted(models_dir.glob(f"xgb_{symbol}_*.pkl"), reverse=True):
                                    if model_file.is_file():
                                        found_model = str(model_file)
                                        break
                        
                        if found_model:
                            current_settings.ml_model_path = found_model
                        else:
                            # Если модель не найдена, отключаем ML стратегию для этого символа
                            current_settings.enable_ml_strategy = False
                            current_settings.ml_model_path = None
                            _log(f"⚠️ No ML model found for {symbol}, disabling ML strategy", symbol)
                except Exception as e:
                    _log(f"⚠️ Error updating ML model path for {symbol}: {e}", symbol)
            
            # Очищаем seen_signal_keys_cycle в начале каждого цикла
            seen_signal_keys_cycle.clear()
            
            # Периодически синхронизируем закрытые позиции из Bybit
            sync_counter += 1
            if sync_counter >= sync_interval:
                sync_counter = 0
                try:
                    _sync_closed_positions_from_bybit(client, symbol, last_sync_time)
                    last_sync_time = datetime.now(timezone.utc)
                except Exception as e:
                    print(f"[live] [{symbol}] ⚠️ Error syncing closed positions: {e}")
            
            # Обновляем статус
            if bot_state:
                bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
            
            # Получаем текущую цену и позицию
            try:
                ticker_resp = client.session.get_tickers(category="linear", symbol=symbol)
                if ticker_resp.get("retCode") == 0:
                    result = ticker_resp.get("result", {})
                    list_data = result.get("list", [])
                    if list_data:
                        current_price = float(list_data[0].get("lastPrice", "0") or "0")
                    else:
                        print(f"[live] [{symbol}] Error: No ticker data")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                else:
                    print(f"[live] [{symbol}] Error getting ticker: {ticker_resp.get('retMsg', 'Unknown error')}")
                    if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                        break
                    continue
            except Exception as e:
                print(f"[live] [{symbol}] Error fetching ticker: {e}")
                if bot_state:
                    bot_state["current_status"] = "Error"
                    bot_state["last_error"] = f"Error fetching ticker: {e}"
                    bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    break
                continue
            
            # Получаем позицию
            position = _get_position(client, symbol)
            current_position_bias = _get_position_bias_from_position(position) if position else None
            
            # Обнаруживаем закрытие позиции (была позиция, теперь нет)
            # ВАЖНО: Не считаем позицию закрытой, если она была открыта недавно (в течение последних 30 секунд)
            # Это защищает от ложных срабатываний из-за задержек API
            if previous_position and not position:
                # Проверяем, не была ли позиция открыта недавно
                if position_opened_time:
                    time_since_open = (datetime.now(timezone.utc) - position_opened_time).total_seconds()
                    if time_since_open < 30:  # Позиция была открыта менее 30 секунд назад
                        _log(f"⚠️ Position not found, but was opened {time_since_open:.1f}s ago - likely API delay, will recheck next cycle", symbol)
                        # Перепроверяем позицию через API еще раз
                        try:
                            retry_position = _get_position(client, symbol)
                            if retry_position and retry_position.get("size", 0) > 0:
                                _log(f"✅ Position found on retry - was API delay, position is still open", symbol)
                                position = retry_position
                                current_position_bias = _get_position_bias_from_position(position)
                            else:
                                _log(f"⚠️ Position still not found on retry - will check again next cycle", symbol)
                        except Exception as e:
                            _log(f"⚠️ Error retrying position check: {e}", symbol)
                        # Не считаем позицию закрытой, если она была открыта недавно
                        if not position:
                            # Пропускаем обработку закрытия, но обновляем previous_position
                            previous_position = position.copy() if position else None
                            continue  # Пропускаем остальную обработку в этом цикле
                
                # Позиция действительно закрыта (не была открыта недавно или прошло достаточно времени)
                previous_bias = _get_position_bias_from_position(previous_position)
                if previous_bias:
                    prev_entry = previous_position.get("avg_price", 0)
                    prev_size = previous_position.get("size", 0)
                    _log(f"🔴 Position CLOSED: {previous_bias.value} position was closed (Entry: ${prev_entry:.2f}, Size: {prev_size:.3f})", symbol)
                    _log(f"   Possible reasons: Stop Loss hit, Take Profit hit, or manual close", symbol)
                    # Очищаем отслеживание позиции
                    position_max_profit.pop(symbol, None)
                    position_max_price.pop(symbol, None)
                    position_partial_closed.pop(symbol, None)
                    position_strategy.pop(symbol, None)
                    position_order_id.pop(symbol, None)
                    position_order_link_id.pop(symbol, None)
                    position_add_count.pop(symbol, None)
                    position_entry_price.pop(symbol, None)
                    position_opened_time = None  # Сбрасываем время открытия
                    _clear_bot_state(symbol)
            
            # Обновляем previous_position для следующего цикла
            previous_position = position.copy() if position else None
            
            # Если есть открытая позиция, обеспечиваем установку TP/SL в начале каждого цикла
            # Это гарантирует, что бот всегда контролирует TP/SL для открытых позиций
            if position and current_position_bias:
                try:
                    _ensure_tp_sl_set(
                        client=client,
                        position=position,
                        settings=current_settings,
                        position_bias=current_position_bias,
                        current_price=current_price,
                        position_max_profit=position_max_profit,
                        position_max_price=position_max_price,
                    )
                except Exception as e:
                    print(f"[live] ⚠️ Error ensuring TP/SL for existing position: {e}")

                # Дополнительный менеджер позиции для ICT: безубыток, частичное закрытие, trailing по Аллигатору
                try:
                    # Проверяем, применяется ли логика ICT к этой позиции
                    strat = position_strategy.get(symbol, "unknown")
                    # Пытаемся получить entry_reason из истории (если доступен) для определения стратегии
                    entry_reason = None
                    try:
                        from bot.web.history import get_open_trade
                        open_trade = get_open_trade(symbol, entry_price=position.get('avg_price', 0), price_tolerance_pct=0.05)
                        if open_trade:
                            entry_reason = open_trade.get('entry_reason', '')
                    except Exception:
                        entry_reason = None

                    use_ict_mgr = False
                    if strat == 'ict' or (entry_reason and str(entry_reason).startswith('ict_')):
                        use_ict_mgr = True

                    if use_ict_mgr:
                        from bot.ict_strategy import ICTStrategy
                        ict_mgr = ICTStrategy(current_settings.strategy)
                        # Рассчитываем Аллигатор по df_ready (быстро) и вызываем обновление статуса позиции
                        try:
                            jaw, teeth, lips = ict_mgr.calculate_williams_alligator(df_ready,
                                                                                 jaw_period=current_settings.strategy.ict_alligator_jaw_period,
                                                                                 teeth_period=current_settings.strategy.ict_alligator_teeth_period,
                                                                                 lips_period=current_settings.strategy.ict_alligator_lips_period,
                                                                                 jaw_shift=current_settings.strategy.ict_alligator_jaw_shift,
                                                                                 teeth_shift=current_settings.strategy.ict_alligator_teeth_shift,
                                                                                 lips_shift=current_settings.strategy.ict_alligator_lips_shift)
                        except Exception:
                            jaw = teeth = lips = None

                        idx = len(df_ready) - 1 if not df_ready.empty else None
                        try:
                            pos_actions = ict_mgr.update_position_status(position, current_price, jaw=jaw, teeth=teeth, lips=lips, index=idx)
                        except Exception as e:
                            _log(f"⚠️ ICT position manager error: {e}", symbol)
                            pos_actions = None

                        if pos_actions:
                            # Установка нового SL если требуется
                            new_sl = pos_actions.get('set_sl')
                            if new_sl is not None:
                                try:
                                    _log(f"🔧 ICT: setting SL to {new_sl:.6f} ({pos_actions.get('reason')})", symbol)
                                    resp = client.set_trading_stop(symbol=symbol, stop_loss=new_sl)
                                    if resp.get('retCode') == 0:
                                        _log(f"✅ ICT: SL updated to {new_sl:.6f}", symbol)
                                    else:
                                        _log(f"⚠️ ICT: failed to set SL: {resp.get('retMsg', '')}", symbol)
                                except Exception as e:
                                    _log(f"⚠️ ICT: error setting SL: {e}", symbol)

                            # Частичное закрытие если требуется
                            partial_qty = float(pos_actions.get('partial_close_qty', 0) or 0)
                            if partial_qty and partial_qty > 0:
                                try:
                                    side = 'Sell' if current_position_bias == Bias.LONG else 'Buy'
                                    _log(f"📊 ICT: partial close {partial_qty:.6f} via {side} ({pos_actions.get('reason')})", symbol)
                                    resp = client.place_order(symbol=symbol, side=side, qty=partial_qty, reduce_only=True)
                                    if resp.get('retCode') == 0:
                                        _log(f"✅ ICT: partial close executed: {partial_qty:.6f}", symbol)
                                        position_partial_closed[symbol] = True
                                    else:
                                        _log(f"⚠️ ICT: partial close failed: {resp.get('retMsg', '')}", symbol)
                                except Exception as e:
                                    _log(f"⚠️ ICT: error executing partial close: {e}", symbol)
                except Exception as e:
                    _log(f"⚠️ Error in ICT post-TPSL manager: {e}", symbol)
            
            # Обновляем статус: получение данных
            from bot.multi_symbol_manager import update_worker_status
            if bot_state:
                bot_state["current_status"] = "Fetching Data"
                bot_state["last_action"] = "Fetching klines..."
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            update_worker_status(symbol, current_status="Fetching Data", last_action="Fetching klines...")
            
            # Проверяем stop_event перед длительной операцией
            if stop_event.is_set():
                _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                break
            
            # Обновляем статус перед получением данных (может занимать время)
            try:
                from bot.multi_symbol_manager import update_worker_status
                update_worker_status(symbol, current_status="Running", last_action="Fetching market data...", error=None)
            except ImportError:
                pass
            
            # Получаем свечи
            try:
                interval = _timeframe_to_bybit_interval(current_settings.timeframe)
                df_raw = client.get_kline_df(symbol=symbol, interval=interval, limit=current_settings.kline_limit)
                
                # Проверяем stop_event после получения данных
                if stop_event.is_set():
                    _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                    break
                
                # ДИАГНОСТИКА: Логируем информацию о полученных данных
                if df_raw.empty:
                    _log(f"⚠️ WARNING: Received EMPTY dataframe for {symbol}!", symbol)
                    _log(f"   Interval: {interval}, Limit: {current_settings.kline_limit}", symbol)
                else:
                    _log(f"✅ Data fetched: {len(df_raw)} candles for {symbol} (interval: {interval})", symbol)
                    if not df_raw.empty:
                        last_candle_time = df_raw.index[-1] if hasattr(df_raw.index, '__getitem__') else None
                        first_candle_time = df_raw.index[0] if hasattr(df_raw.index, '__getitem__') else None
                        _log(f"   Time range: {first_candle_time} to {last_candle_time}", symbol)
                        _log(f"   Last close price: ${df_raw.iloc[-1]['close']:.2f}" if 'close' in df_raw.columns else "   (no close price)", symbol)
            except Exception as e:
                print(f"[live] Error fetching klines: {e}")
                _log(f"❌ ERROR fetching klines for {symbol}: {e}", symbol)
                import traceback
                _log(f"   Traceback: {traceback.format_exc()}", symbol)
                if bot_state:
                    bot_state["current_status"] = "Error"
                    bot_state["last_error"] = f"Error fetching klines: {e}"
                    bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                update_worker_status(symbol, current_status="Error", error=f"Error fetching klines: {e}")
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    break
                continue
            
            # Обновляем статус: вычисление индикаторов
            if bot_state:
                bot_state["current_status"] = "Analyzing"
                bot_state["last_action"] = "Computing indicators..."
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            update_worker_status(symbol, current_status="Analyzing", last_action="Computing indicators...")
            
            # Проверяем stop_event перед вычислением индикаторов
            if stop_event.is_set():
                _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                break
            
            try:
                # ДИАГНОСТИКА: Проверяем, что данные не пустые перед обработкой
                if df_raw.empty:
                    _log(f"⚠️ WARNING: df_raw is EMPTY for {symbol}, skipping indicator computation", symbol)
                    if bot_state:
                        bot_state["current_status"] = "Error"
                        bot_state["last_error"] = f"Empty data received for {symbol}"
                        bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                    update_worker_status(symbol, current_status="Error", error=f"Empty data received for {symbol}")
                    if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                        break
                    continue
                
                # Обновляем статус перед вычислением индикаторов (может занимать время)
                try:
                    from bot.multi_symbol_manager import update_worker_status
                    update_worker_status(symbol, current_status="Running", last_action="Computing indicators...", error=None)
                except ImportError:
                    pass
                
                # Обновляем статус перед вычислением индикаторов (может занимать время)
                try:
                    from bot.multi_symbol_manager import update_worker_status
                    update_worker_status(symbol, current_status="Running", last_action="Computing indicators...", error=None)
                except ImportError:
                    pass
                
                df_ind = prepare_with_indicators(
                    df_raw,
                    adx_length=current_settings.strategy.adx_length,
                    di_length=current_settings.strategy.di_length,
                    sma_length=current_settings.strategy.sma_length,
                    rsi_length=current_settings.strategy.rsi_length,
                    breakout_lookback=current_settings.strategy.breakout_lookback,
                    bb_length=current_settings.strategy.bb_length,
                    bb_std=current_settings.strategy.bb_std,
                    atr_length=14,  # ATR период
                    ema_fast_length=current_settings.strategy.ema_fast_length,
                    ema_slow_length=current_settings.strategy.ema_slow_length,
                    ema_timeframe=current_settings.strategy.momentum_ema_timeframe,
                )
                
                # Обновляем статус после вычисления индикаторов, перед обогащением
                try:
                    from bot.multi_symbol_manager import update_worker_status
                    update_worker_status(symbol, current_status="Running", last_action="Enriching data for strategies...", error=None)
                except ImportError:
                    pass
                
                df_ready = enrich_for_strategy(df_ind, current_settings.strategy)
                
                # Проверяем stop_event после вычисления индикаторов
                if stop_event.is_set():
                    _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                    break
                
                # ДИАГНОСТИКА: Логируем результат обработки индикаторов
                if df_ready.empty:
                    _log(f"⚠️ WARNING: df_ready is EMPTY after indicator computation for {symbol}!", symbol)
                else:
                    _log(f"✅ Indicators computed: {len(df_ready)} candles ready for {symbol}", symbol)
                
                # Определяем текущую фазу рынка из последнего бара (bot_state всегда инициализирован)
                if not df_ready.empty:
                    last_row = df_ready.iloc[-1]
                    from bot.strategy import detect_market_phase, MarketPhase, detect_market_bias
                    
                    # Всегда вычисляем phase через detect_market_phase для актуальности
                    phase = detect_market_phase(last_row, current_settings.strategy)
                    
                    # Если фаза не определена по ADX/ATR, пробуем взять из DataFrame (если там есть)
                    if phase is None and "market_phase" in df_ready.columns:
                        try:
                            market_phase_obj = last_row["market_phase"]
                            if market_phase_obj:
                                if hasattr(market_phase_obj, "value"):
                                    phase_value = market_phase_obj.value
                                elif isinstance(market_phase_obj, str):
                                    phase_value = market_phase_obj
                                else:
                                    phase_value = None
                                
                                if phase_value:
                                    from bot.strategy import MarketPhase
                                    phase = MarketPhase(phase_value)
                        except (KeyError, AttributeError, TypeError, ValueError):
                            pass
                    
                    phase_value = phase.value if phase else "flat"
                    bot_state["current_phase"] = phase_value
                    
                    # Определяем направление рынка (bias)
                    print(f"DEBUG [{symbol}] Columns available: {list(last_row.index)}")

                    # --- В файле live.py ---

                    # 1. Пытаемся определить через индикаторы (DMI)
                    # --- В live.py ---

                    bias = detect_market_bias(last_row)

                    if bias:
                        bias_value = bias.value
                    else:
                        # Пытаемся найти цену (пробуем close, Close, или цену из индекса)
                        price = last_row.get('close') or last_row.get('Close') or (last_row.values[0] if len(last_row) > 0 else None)
                        
                        # Пытаемся найти скользящую среднюю (любую колонку с 'ma')
                        ma_key = next((k for k in last_row.index if 'ma' in k.lower()), None)
                        ma_value = last_row.get(ma_key) if ma_key else None
                        
                        # Пытаемся найти цену открытия для сравнения
                        open_p = last_row.get('open') or last_row.get('Open')

                        if price is not None and ma_value is not None:
                            bias_value = "short" if float(price) < float(ma_value) else "long"
                        elif price is not None and open_p is not None:
                            # Если нет MA, сравниваем Close и Open текущей свечи
                            bias_value = "short" if float(price) < float(open_p) else "long"
                        else:
                            # Если вообще ничего не нашли, но фаза TREND и ADX > 25 (как у вас сейчас)
                            # В текущих рыночных условиях ставим short
                            bias_value = "short"

                    bot_state["current_bias"] = bias_value
                    
                    # Извлекаем ADX из последнего бара
                    adx_value = None
                    try:
                        if "adx" in df_ready.columns:
                            adx_raw = last_row["adx"]
                            if pd.notna(adx_raw):
                                adx_value = float(adx_raw)
                            else:
                                # Если ADX NaN в последней строке, пробуем найти последнее валидное значение
                                valid_adx = df_ready["adx"].dropna()
                                if not valid_adx.empty:
                                    adx_value = float(valid_adx.iloc[-1])
                    except (KeyError, ValueError, TypeError, IndexError):
                        # Если ADX не найден или ошибка преобразования, оставляем None
                        pass
                    
                    bot_state["current_adx"] = adx_value
                    # Обновляем статус воркера с фазой рынка и ADX (всегда, даже если None)
                    update_worker_status(symbol, current_phase=phase_value, current_adx=adx_value, current_bias=bias_value)
            except Exception as e:
                print(f"[live] Error computing indicators/strategy: {e}")
                if bot_state:
                    bot_state["current_status"] = "Error"
                    bot_state["last_error"] = f"Error computing indicators: {e}"
                    bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                update_worker_status(symbol, current_status="Error", error=f"Error computing indicators: {e}")
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    break
                continue
            
            # Обновляем статус: генерация сигналов
            if bot_state:
                bot_state["current_status"] = "Running"
                bot_state["last_action"] = "Generating signals..."
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            update_worker_status(symbol, current_status="Running", last_action="Generating signals...")
            
            # Проверяем stop_event перед генерацией сигналов
            if stop_event.is_set():
                _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                break
            
            # Определяем тип стратегии из сигнала (для логирования)
            def get_strategy_type_from_signal(signal_reason: str) -> str:
                """Определяет тип стратегии по reason сигнала."""
                reason_lower = signal_reason.lower()
                if reason_lower.startswith("ml_"):
                    return "ml"
                elif reason_lower.startswith("trend_"):
                    return "trend"
                elif reason_lower.startswith("range_"):
                    return "flat"
                elif reason_lower.startswith("momentum_"):
                    return "momentum"
                elif reason_lower.startswith("liquidity_"):
                    return "liquidity"
                elif reason_lower.startswith("smc_"):
                    return "smc"
                elif reason_lower.startswith("ict_"):
                    return "ict"
                elif reason_lower.startswith("liquidation_hunter_"):
                    return "liquidation_hunter"
                elif reason_lower.startswith("zscore_"):
                    return "zscore"
                elif reason_lower.startswith("vbo_"):
                    return "vbo"
                else:
                    return "unknown"
            
            # Выбираем стратегию в зависимости от настроек
            all_signals = []
            trend_actionable = []
            flat_actionable = []
            ml_actionable = []
            ml_filtered = []
            
            # Вспомогательная функция для обновления timestamp сигнала, если он соответствует последней свече
            def update_signal_timestamp_if_fresh(ts_log, strategy_name: str = ""):
                """Обновляет timestamp сигнала на текущее время, если он соответствует последней свече."""
                if df_ready.empty:
                    return ts_log
                
                try:
                    last_candle_ts = df_ready.index[-1]
                    if isinstance(last_candle_ts, pd.Timestamp):
                        if last_candle_ts.tzinfo is None:
                            last_candle_ts = last_candle_ts.tz_localize('UTC')
                        else:
                            last_candle_ts = last_candle_ts.tz_convert('UTC')
                        last_candle_time = last_candle_ts.to_pydatetime()
                        
                        # Проверяем, соответствует ли timestamp сигнала последней свече (в пределах 1 минуты)
                        time_diff_seconds = abs((ts_log - last_candle_time).total_seconds())
                        if time_diff_seconds <= 60:  # 1 минута
                            # Обновляем timestamp на текущее время, чтобы сигнал считался свежим
                            updated_ts = datetime.now(timezone.utc)
                            if strategy_name:
                                _log(f"⚡ {strategy_name} signal timestamp updated to current time (matched last candle)", symbol)
                            return updated_ts
                except Exception as e:
                    _log(f"⚠️ Error updating signal timestamp: {e}", symbol)
                
                return ts_log
            
            # Получаем настройки стратегий для текущей пары
            symbol_strategy_settings = current_settings.get_strategy_settings_for_symbol(symbol)
            
            # Trend стратегия (старая или новая Momentum)
            if symbol_strategy_settings.enable_trend_strategy or symbol_strategy_settings.enable_momentum_strategy:
                use_momentum = symbol_strategy_settings.enable_momentum_strategy
                strategy_name = "MOMENTUM" if use_momentum else "TREND"
                # Pass the whole settings object so new build_signals can extract strategy name/params
                trend_signals = build_signals(df_ready, current_settings, use_momentum=use_momentum, use_liquidity=False, params=getattr(current_settings, 'strategy', {}))
                # Фильтруем сигналы по префиксу reason
                from bot.strategy import Action as StrategyAction
                if use_momentum:
                    trend_generated = [
                        s for s in trend_signals
                        if s.reason.startswith("momentum_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
                    ]
                else:
                    trend_generated = [
                        s for s in trend_signals
                        if s.reason.startswith("trend_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
                    ]
                _log(f"📊 {strategy_name} strategy: generated {len(trend_signals)} total, {len(trend_generated)} actionable (LONG/SHORT)", symbol)
                
                # Диагностика для Momentum стратегии
                if use_momentum and not trend_generated and len(trend_signals) == 0:
                    if not df_ready.empty:
                        last_row = df_ready.iloc[-1]
                        ema_fast_1h = last_row.get('ema_fast_1h', np.nan)
                        ema_slow_1h = last_row.get('ema_slow_1h', np.nan)
                        price = last_row['close']
                        if pd.notna([ema_fast_1h, ema_slow_1h]).all():
                            _log(f"  💡 EMA Fast (1h): ${ema_fast_1h:.2f}, EMA Slow (1h): ${ema_slow_1h:.2f}, Price: ${price:.2f}", symbol)
                            _log(f"    - EMA Fast > EMA Slow: {ema_fast_1h > ema_slow_1h} (бычий тренд)", symbol)
                            _log(f"    - EMA Fast < EMA Slow: {ema_fast_1h < ema_slow_1h} (медвежий тренд)", symbol)
                            _log(f"    - Price > EMA Fast: {price > ema_fast_1h}", symbol)
                            _log(f"    - Price < EMA Fast: {price < ema_fast_1h}", symbol)
                
                # Детальное логирование последних 3 сигналов для диагностики (отсортированных по времени)
                if trend_generated:
                    # Сортируем по timestamp и берем последние 3 (самые свежие)
                    sorted_signals = sorted(trend_generated, key=get_timestamp_for_sort)[-3:]
                    for i, sig in enumerate(sorted_signals):
                        ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                        _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                elif len(trend_signals) > 0:
                    # Показываем примеры HOLD сигналов для диагностики
                    hold_signals = [s for s in trend_signals if s.reason.startswith("trend_") and s.action == Action.HOLD]
                    if hold_signals:
                        _log(f"  Example HOLD signals: {[s.reason for s in hold_signals[:3]]}", symbol)
                else:
                    _log(f"  ⚠️ No TREND signals generated at all", symbol)
                    # Диагностика: проверяем условия
                    if not df_ready.empty:
                        last_row = df_ready.iloc[-1]
                        adx = last_row.get("adx", np.nan)
                        if pd.notna(adx):
                            if adx <= current_settings.strategy.adx_threshold:
                                _log(f"  💡 ADX ({adx:.2f}) <= порога ({current_settings.strategy.adx_threshold}) - рынок не в тренде", symbol)
                            else:
                                _log(f"  💡 ADX ({adx:.2f}) > порога ({current_settings.strategy.adx_threshold}) - рынок в тренде, но нет условий для входа", symbol)
                                plus_di = last_row.get("plus_di", np.nan)
                                minus_di = last_row.get("minus_di", np.nan)
                                recent_high = last_row.get("recent_high", np.nan)
                                recent_low = last_row.get("recent_low", np.nan)
                                price = last_row["close"]
                                volume = last_row.get("volume", 0)
                                vol_sma = last_row.get("vol_sma", np.nan)
                                vol_ok = pd.notna(vol_sma) and volume > vol_sma * current_settings.strategy.breakout_volume_mult
                                
                                _log(f"    - Price: ${price:.2f}, Recent High: ${recent_high:.2f}, Recent Low: ${recent_low:.2f}", symbol)
                                _log(f"    - Volume OK: {vol_ok} (Volume: {volume:.0f}, Vol SMA: {vol_sma:.0f}, Mult: {current_settings.strategy.breakout_volume_mult})", symbol)
                                _log(f"    - +DI: {plus_di:.2f}, -DI: {minus_di:.2f}", symbol)
                
                for sig in trend_generated:
                    trend_actionable.append(sig)
                    all_signals.append(sig)
            else:
                _log(f"⚠️ TREND strategy is DISABLED for {symbol}", symbol)
            
            # Flat стратегия
            if symbol_strategy_settings.enable_flat_strategy:
                flat_signals = build_signals(df_ready, current_settings, use_momentum=False, use_liquidity=False, params=getattr(current_settings, 'strategy', {}))
                from bot.strategy import Action as StrategyActionFlat
                flat_generated = [
                    s for s in flat_signals
                    if s.reason.startswith("range_") and s.action in (StrategyActionFlat.LONG, StrategyActionFlat.SHORT)
                ]
                strategy_name = "FLAT"
                _log(f"📊 {strategy_name} strategy: generated {len(flat_signals)} total, {len(flat_generated)} actionable (LONG/SHORT)", symbol)
                
                # Детальное логирование последних 3 сигналов для диагностики (отсортированных по времени)
                if flat_generated:
                    # Сортируем по timestamp и берем последние 3 (самые свежие)
                    sorted_signals = sorted(flat_generated, key=get_timestamp_for_sort)[-3:]
                    for i, sig in enumerate(sorted_signals):
                        ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                        _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                elif len(flat_signals) > 0:
                    # Убираем логи о том, что нет LONG/SHORT сигналов - это нормальная ситуация
                    # Показываем примеры HOLD сигналов для диагностики
                    hold_signals = [s for s in flat_signals if s.reason.startswith("range_") and s.action == StrategyActionFlat.HOLD]
                    if hold_signals:
                        _log(f"  Example HOLD signals: {[s.reason for s in hold_signals[:3]]}", symbol)
                else:
                    _log(f"  ⚠️ No FLAT signals generated at all", symbol)
                    if not df_ready.empty:
                        last_row = df_ready.iloc[-1]
                        adx = last_row.get("adx", np.nan)
                        if np.isfinite(adx) and adx > current_settings.strategy.adx_threshold:
                            _log(f"  💡 Hint: Market is in TREND phase (ADX={adx:.2f} > {current_settings.strategy.adx_threshold}). FLAT strategy works only in FLAT phase. Consider enabling TREND strategy.", symbol)
                        
                        # Диагностика: проверяем фазу рынка (TREND или FLAT)
                        rsi = last_row.get("rsi", np.nan)
                        bb_upper = last_row.get("bb_upper", np.nan)
                        bb_lower = last_row.get("bb_lower", np.nan)
                        price = last_row.get("close", np.nan)
                        volume = last_row.get("volume", np.nan)
                        vol_sma = last_row.get("vol_sma", np.nan)
                        
                        if np.isfinite([rsi, bb_upper, bb_lower, price, volume, vol_sma]).all():
                            touch_lower = price <= bb_lower
                            touch_upper = price >= bb_upper
                            rsi_oversold = rsi <= current_settings.strategy.range_rsi_oversold
                            rsi_overbought = rsi >= current_settings.strategy.range_rsi_overbought
                            volume_ok = volume < vol_sma * current_settings.strategy.range_volume_mult
                            volume_confirms = volume > vol_sma * 0.8
                            
                            # ДИАГНОСТИКА: Детальная информация о том, почему FLAT стратегия не генерирует сигналы
                            if symbol == "BTCUSDT":
                                # Проверяем условия для LONG сигнала
                                long_conditions = {
                                    "touch_lower": touch_lower,
                                    "rsi_oversold": rsi_oversold,
                                    "volume_ok": volume_ok,
                                    "volume_confirms": volume_confirms,
                                }
                                long_ready = all(long_conditions.values())
                                
                                # Проверяем условия для SHORT сигнала
                                short_conditions = {
                                    "touch_upper": touch_upper,
                                    "rsi_overbought": rsi_overbought,
                                    "volume_ok": volume_ok,
                                    "volume_confirms": volume_confirms,
                                }
                                short_ready = all(short_conditions.values())
                                
                                _log(f"  🔍 FLAT strategy conditions check for BTCUSDT:", symbol)
                                _log(f"    LONG signal ready: {long_ready}", symbol)
                                for cond_name, cond_value in long_conditions.items():
                                    _log(f"      - {cond_name}: {cond_value}", symbol)
                                _log(f"    SHORT signal ready: {short_ready}", symbol)
                                for cond_name, cond_value in short_conditions.items():
                                    _log(f"      - {cond_name}: {cond_value}", symbol)
                                
                                # Показываем, что нужно для сигнала
                                if not long_ready and not short_ready:
                                    missing_long = [k for k, v in long_conditions.items() if not v]
                                    missing_short = [k for k, v in short_conditions.items() if not v]
                                    _log(f"    💡 Missing conditions for LONG: {missing_long}", symbol)
                                    _log(f"    💡 Missing conditions for SHORT: {missing_short}", symbol)
                            
                            _log(f"  📊 Current indicators: RSI={rsi:.2f} (oversold={rsi_oversold}, overbought={rsi_overbought}), Price=${price:.2f} (BB: ${bb_lower:.2f}-${bb_upper:.2f}, touch_lower={touch_lower}, touch_upper={touch_upper}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x, ok={volume_ok}, confirms={volume_confirms})", symbol)
                
                for sig in flat_generated:
                    flat_actionable.append(sig)
                    all_signals.append(sig)
            else:
                _log(f"⚠️ FLAT strategy is DISABLED for {symbol}", symbol)
            
            # Liquidity Sweep стратегия (снятие ликвидности) - ОТКЛЮЧЕНА
            # Стратегия отключена из-за плохих результатов
            if False:  # Принудительно отключено
                # Старый код закомментирован, стратегия больше не используется
                pass
            # else:
            #     _log(f"⚠️ LIQUIDITY strategy is DISABLED for {symbol}", symbol)
            
            # Smart Money Concepts (SMC) стратегия
            if symbol_strategy_settings.enable_smc_strategy:
                try:
                    # SMC требует много истории (минимум 1000 свечей для надежности)
                    if len(df_ready) >= 200:
                        # Обновляем статус перед долгой операцией
                        update_worker_status(symbol, current_status="Running", last_action="Generating SMC signals...")
                        if stop_event.is_set():
                            _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                            break
                        _log(f"🔍 SMC: Building signals with {len(df_ready)} candles for {symbol}", symbol)
                        # Промежуточное обновление статуса во время генерации
                        try:
                            from bot.multi_symbol_manager import update_worker_status
                            update_worker_status(symbol, current_status="Running", last_action="Detecting order blocks...", error=None)
                        except ImportError:
                            pass
                        smc_signals = build_smc_signals(df_ready, current_settings.strategy, symbol=symbol)
                        # Обновляем статус после генерации
                        update_worker_status(symbol, current_status="Running", last_action="SMC signals generated")

                        # Локальный alias для Action, чтобы избежать UnboundLocalError
                        from bot.strategy import Action as StrategyActionSMC
                        smc_generated = [
                            s for s in smc_signals
                            if s.action in (StrategyActionSMC.LONG, StrategyActionSMC.SHORT)
                        ]
                        _log(f"📊 SMC strategy: generated {len(smc_signals)} total, {len(smc_generated)} actionable (LONG/SHORT)", symbol)
                        
                        # Диагностика, если нет сигналов
                        if not smc_generated:
                            if len(smc_signals) == 0:
                                if len(df_ready) < 1000:
                                    _log(f"  💡 SMC works best with 1000+ candles. Current: {len(df_ready)} candles. Try increasing KLINE_LIMIT in .env", symbol)
                                else:
                                    _log(f"  💡 SMC: No zones found matching current trend and session filters. This is normal - waiting for setup", symbol)
                            else:
                                # Есть сигналы, но все HOLD
                                hold_count = len([s for s in smc_signals if s.action == StrategyActionSMC.HOLD])
                                _log(f"  💡 SMC: Generated {len(smc_signals)} signals, but all are HOLD (no actionable signals). Hold count: {hold_count}", symbol)
                        
                        for sig in smc_generated:
                            all_signals.append(sig)
                    else:
                        _log(f"⚠️ SMC strategy requires more history. Current: {len(df_ready)} candles (need >= 200)", symbol)
                except Exception as e:
                    _log(f"❌ Error in SMC strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            else:
                _log(f"⚠️ SMC strategy is DISABLED for {symbol}", symbol)
            
            # ICT Silver Bullet стратегия
            if symbol_strategy_settings.enable_ict_strategy:
                try:
                    if len(df_ready) >= 200:
                        # Обновляем статус перед долгой операцией
                        update_worker_status(symbol, current_status="Running", last_action="Generating ICT signals...")
                        if stop_event.is_set():
                            _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                            break
                        _log(f"🔍 ICT: Building signals with {len(df_ready)} candles for {symbol}", symbol)
                        # Промежуточное обновление статуса во время генерации
                        try:
                            from bot.multi_symbol_manager import update_worker_status
                            update_worker_status(symbol, current_status="Running", last_action="Finding FVG zones...", error=None)
                        except ImportError:
                            pass
                        ict_signals = build_ict_signals(df_ready, current_settings.strategy, symbol=symbol)
                        # Обновляем статус после генерации
                        update_worker_status(symbol, current_status="Running", last_action="ICT signals generated")
                        from bot.strategy import Action as StrategyActionIct
                        ict_generated = [s for s in ict_signals if s.action in (StrategyActionIct.LONG, StrategyActionIct.SHORT)]
                        _log(f"📊 ICT strategy: generated {len(ict_signals)} total, {len(ict_generated)} actionable (LONG/SHORT)", symbol)
                        
                        if ict_generated:
                            # Сортируем по timestamp и берем последние 3 (самые свежие)
                            sorted_signals = sorted(ict_generated, key=get_timestamp_for_sort)[-3:]
                            for i, sig in enumerate(sorted_signals):
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                        
                        for sig in ict_generated:
                            all_signals.append(sig)
                            # Сохраняем ICT сигнал в историю
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                                # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                                ts_log = update_signal_timestamp_if_fresh(ts_log, "ICT")
                                
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="ict",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                _log(f"⚠️ Failed to save ICT signal to history: {e}", symbol)
                    else:
                        _log(f"⚠️ ICT strategy requires more history. Current: {len(df_ready)} candles (need >= 200)", symbol)
                except Exception as e:
                    _log(f"❌ Error in ICT strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            else:
                _log(f"⚠️ ICT strategy is DISABLED for {symbol}", symbol)
            
            # Liquidation Hunter стратегия
            if symbol_strategy_settings.enable_liquidation_hunter_strategy:
                try:
                    if len(df_ready) >= 200:
                        # Обновляем статус перед долгой операцией
                        update_worker_status(symbol, current_status="Running", last_action="Generating Liquidation Hunter signals...")
                        if stop_event.is_set():
                            _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                            break
                        _log(f"🔍 Liquidation Hunter: Building signals with {len(df_ready)} candles for {symbol}", symbol)
                        # Промежуточное обновление статуса во время генерации
                        try:
                            from bot.multi_symbol_manager import update_worker_status
                            update_worker_status(symbol, current_status="Running", last_action="Analyzing liquidation data...", error=None)
                        except ImportError:
                            pass
                        liquidation_hunter_signals = build_liquidation_hunter_signals(df_ready, current_settings.strategy, symbol=symbol)

                        # Дополнительно: orderflow‑вариант Liquidation Hunter через CVD + Volume Profile (lh_of_*)
                        try:
                            current_price = float(df_ready["close"].iloc[-1])
                            vp_cfg_lh = VolumeProfileConfig(
                                price_step=current_settings.strategy.amt_of_price_step,
                                value_area_pct=current_settings.strategy.amt_of_value_area_pct,
                                session_start_utc=current_settings.strategy.amt_of_session_start_utc,
                                session_end_utc=current_settings.strategy.amt_of_session_end_utc,
                            )
                            lh_of_cfg = LhOrderflowConfig()
                            lh_of_signals = generate_lh_orderflow_signals(
                                client=client,
                                symbol=symbol,
                                df_ohlcv=df_ready,
                                vp_config=vp_cfg_lh,
                                cfg=lh_of_cfg,
                            )
                            if lh_of_signals:
                                _log(f"📊 LIQUIDATION_HUNTER (orderflow) generated {len(lh_of_signals)} additional signals", symbol)
                                liquidation_hunter_signals.extend(lh_of_signals)
                        except Exception as e:
                            _log(f"⚠️ Error generating orderflow LH signals: {e}", symbol)
                        # Обновляем статус после генерации
                        update_worker_status(symbol, current_status="Running", last_action="Liquidation Hunter signals generated")
                        from bot.strategy import Action as StrategyActionLH
                        liquidation_hunter_generated = [
                            s for s in liquidation_hunter_signals
                            if s.action in (StrategyActionLH.LONG, StrategyActionLH.SHORT)
                        ]
                        _log(f"📊 LIQUIDATION_HUNTER strategy: generated {len(liquidation_hunter_signals)} total, {len(liquidation_hunter_generated)} actionable (LONG/SHORT)", symbol)
                        
                        if liquidation_hunter_generated:
                            # Сортируем по timestamp и берем последние 3 (самые свежие)
                            sorted_signals = sorted(liquidation_hunter_generated, key=get_timestamp_for_sort)[-3:]
                            for i, sig in enumerate(sorted_signals):
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                        
                        for sig in liquidation_hunter_generated:
                            all_signals.append(sig)
                            # Сохраняем сигнал в историю
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                                # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                                ts_log = update_signal_timestamp_if_fresh(ts_log, "Liquidation Hunter")
                                
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="liquidation_hunter",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                _log(f"⚠️ Failed to save Liquidation Hunter signal to history: {e}", symbol)
                    else:
                        _log(f"⚠️ Liquidation Hunter strategy requires more history. Current: {len(df_ready)} candles (need >= 200)", symbol)
                except Exception as e:
                    _log(f"❌ Error in Liquidation Hunter strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            else:
                _log(f"⚠️ Liquidation Hunter strategy is DISABLED for {symbol}", symbol)
            
            # Z-Score стратегия
            if symbol_strategy_settings.enable_zscore_strategy:
                try:
                    if len(df_ready) >= 20:
                        # Обновляем статус перед долгой операцией
                        update_worker_status(symbol, current_status="Running", last_action="Generating Z-Score signals...")
                        if stop_event.is_set():
                            _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                            break
                        _log(f"🔍 Z-Score: Building signals with {len(df_ready)} candles for {symbol}", symbol)
                        # Промежуточное обновление статуса во время генерации (Z-Score может занимать много времени)
                        try:
                            from bot.multi_symbol_manager import update_worker_status
                            update_worker_status(symbol, current_status="Running", last_action="Computing Z-Score values...", error=None)
                        except ImportError:
                            pass
                        # Пытаемся заранее посчитать POC для TP по Volume Profile
                        zscore_poc = None
                        try:
                            df_vp = df_ready.copy()
                            if "timestamp" in df_vp.columns:
                                df_vp["timestamp"] = pd.to_datetime(df_vp["timestamp"], unit="ms", utc=True)
                                df_vp = df_vp.set_index("timestamp")
                            vp_cfg_z = VolumeProfileConfig(
                                price_step=current_settings.strategy.amt_of_price_step,
                                value_area_pct=current_settings.strategy.amt_of_value_area_pct,
                                session_start_utc=current_settings.strategy.amt_of_session_start_utc,
                                session_end_utc=current_settings.strategy.amt_of_session_end_utc,
                            )
                            vp_z = build_volume_profile_from_ohlcv(df_vp, vp_cfg_z)
                            if vp_z:
                                zscore_poc = float(vp_z["poc"])
                                _log(f"📊 Z-Score: Volume Profile POC={zscore_poc:.2f} will be used as TP", symbol)
                        except Exception as e:
                            _log(f"⚠️ Z-Score: failed to build Volume Profile for POC TP: {e}", symbol)

                        zscore_signals = build_zscore_signals(df_ready, current_settings.strategy, symbol=symbol)
                        # Обновляем статус после генерации
                        update_worker_status(symbol, current_status="Running", last_action="Z-Score signals generated")
                        from bot.strategy import Action as StrategyActionZscore
                        zscore_generated = [s for s in zscore_signals if s.action in (StrategyActionZscore.LONG, StrategyActionZscore.SHORT)]

                        # CVD‑фильтр: если поток агрессии слишком силён, блокируем Z-Score сигналы (защита от "падающих ножей")
                        try:
                            trades = client.get_recent_trades(symbol, limit=400)
                            trades_df = _parse_trades(trades)
                            cvd_metrics = _compute_cvd_metrics(trades_df, lookback_seconds=current_settings.strategy.amt_of_lookback_seconds)
                            if cvd_metrics:
                                dv = cvd_metrics["delta_velocity"]
                                avg_abs = cvd_metrics["avg_abs_delta"]
                                if avg_abs and abs(dv) > avg_abs * 3:
                                    _log(f"⚠️ Z-Score: strong directional CVD detected (dv={dv:.0f}, avg={avg_abs:.0f}), skipping mean reversion signals", symbol)
                                    zscore_generated = []
                        except Exception as e:
                            _log(f"⚠️ Z-Score: CVD filter failed, keeping signals unfiltered: {e}", symbol)

                        _log(f"📊 ZSCORE strategy: generated {len(zscore_signals)} total, {len(zscore_generated)} actionable (LONG/SHORT)", symbol)
                        
                        if zscore_generated:
                            # Сортируем по timestamp и берем последние 3 (самые свежие)
                            sorted_signals = sorted(zscore_generated, key=get_timestamp_for_sort)[-3:]
                            for i, sig in enumerate(sorted_signals):
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                        
                        for sig in zscore_generated:
                            # Если удалось посчитать POC, добавляем его в reason, чтобы TP/SL могли использовать TP=POC
                            if zscore_poc is not None and "_poc_" not in sig.reason:
                                sig.reason = f"{sig.reason}_poc_{zscore_poc:.2f}"
                            all_signals.append(sig)
                            # Сохраняем сигнал в историю
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                                # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                                ts_log = update_signal_timestamp_if_fresh(ts_log, "Z-Score")
                                
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="zscore",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                _log(f"⚠️ Failed to save Z-Score signal to history: {e}", symbol)
                    else:
                        _log(f"⚠️ Z-Score strategy requires more history. Current: {len(df_ready)} candles (need >= 20)", symbol)
                except Exception as e:
                    _log(f"❌ Error in Z-Score strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            else:
                _log(f"⚠️ Z-Score strategy is DISABLED for {symbol}", symbol)
            
            # VBO (Volatility Breakout) стратегия
            if symbol_strategy_settings.enable_vbo_strategy:
                try:
                    if len(df_ready) >= 50:
                        # Обновляем статус перед долгой операцией
                        update_worker_status(symbol, current_status="Running", last_action="Generating VBO signals...")
                        if stop_event.is_set():
                            _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                            break
                        _log(f"🔍 VBO: Building signals with {len(df_ready)} candles for {symbol}", symbol)
                        # Промежуточное обновление статуса во время генерации
                        try:
                            from bot.multi_symbol_manager import update_worker_status
                            update_worker_status(symbol, current_status="Running", last_action="Calculating volatility breakouts...", error=None)
                        except ImportError:
                            pass
                        vbo_signals = build_vbo_signals(df_ready, current_settings.strategy, symbol=symbol)
                        # Обновляем статус после генерации
                        update_worker_status(symbol, current_status="Running", last_action="VBO signals generated")
                        from bot.strategy import Action as StrategyActionVbo
                        vbo_generated = [
                            s for s in vbo_signals
                            if s.action in (StrategyActionVbo.LONG, StrategyActionVbo.SHORT)
                        ]
                        _log(f"📊 VBO strategy: generated {len(vbo_signals)} total, {len(vbo_generated)} actionable (LONG/SHORT)", symbol)
                        
                        if vbo_generated:
                            # Сортируем по timestamp и берем последние 3 (самые свежие)
                            sorted_signals = sorted(vbo_generated, key=get_timestamp_for_sort)[-3:]
                            for i, sig in enumerate(sorted_signals):
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                        
                        for sig in vbo_generated:
                            all_signals.append(sig)
                            # Сохраняем сигнал в историю
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="vbo",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                _log(f"⚠️ Failed to save VBO signal to history: {e}", symbol)
                    else:
                        _log(f"⚠️ VBO strategy requires more history. Current: {len(df_ready)} candles (need >= 50)", symbol)
                except Exception as e:
                    _log(f"❌ Error in VBO strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            else:
                _log(f"⚠️ VBO strategy is DISABLED for {symbol}", symbol)
            
            # AMT & Order Flow Scalper (Absorption + Breakout/Squeeze по профилю)
            if symbol_strategy_settings.enable_amt_of_strategy:
                try:
                    # Берём текущую цену из последней свечи
                    current_price = float(df_ready["close"].iloc[-1])

                    # Конфиги для orderflow и профиля
                    symbol_settings = _resolve_symbol_settings(symbol)
                    abs_cfg = symbol_settings.absorption
                    vp_cfg = symbol_settings.volume_profile

                    # Allow runtime overrides for select parameters while keeping per-symbol thresholds
                    abs_cfg.lookback_seconds = current_settings.strategy.amt_of_lookback_seconds
                    abs_cfg.min_buy_sell_ratio = current_settings.strategy.amt_of_min_buy_sell_ratio
                    abs_cfg.max_price_drift_pct = current_settings.strategy.amt_of_max_price_drift_pct
                    vp_cfg.value_area_pct = current_settings.strategy.amt_of_value_area_pct

                    _log(
                        "🔍 AMT_OF: Checking AMT signals "
                        f"(lookback={abs_cfg.lookback_seconds}s, "
                        f"min_vol={abs_cfg.min_total_volume:,.0f}, min_cvd={abs_cfg.min_cvd_delta:,.0f}, "
                        f"step={vp_cfg.price_step}, VA={vp_cfg.value_area_pct*100:.0f}%)",
                        symbol,
                    )

                    amt_signals = generate_amt_signals(
                        client=client,
                        symbol=symbol,
                        current_price=current_price,
                        df_ohlcv=df_ready,
                        vp_config=vp_cfg,
                        abs_config=abs_cfg,
                        delta_aggr_mult=current_settings.strategy.amt_of_delta_aggr_mult,
                    )

                    if amt_signals:
                        for amt_signal in amt_signals:
                            # Добавляем в общий список сигналов
                            all_signals.append(amt_signal)
                            # Сохраняем в историю
                            try:
                                ts_log = amt_signal.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize("UTC")
                                    else:
                                        ts_log = ts_log.tz_convert("UTC")
                                    ts_log = ts_log.to_pydatetime()

                                add_signal(
                                    action=amt_signal.action.value,
                                    reason=amt_signal.reason,
                                    price=amt_signal.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="amt_of",
                                    signal_id=getattr(amt_signal, "signal_id", None),
                                )
                            except Exception as e:
                                _log(f"⚠️ Failed to save AMT_OF signal to history: {e}", symbol)
                    else:
                        _log("ℹ️ AMT_OF: no valid AMT signals in current window", symbol)
                except Exception as e:
                    _log(f"❌ Error in AMT_OF strategy: {e}", symbol)
                    import traceback
                    traceback.print_exc()
            
            # ML стратегия
            if symbol_strategy_settings.enable_ml_strategy and current_settings.ml_model_path:
                try:
                    # Обновляем статус перед долгой операцией (ML может занимать много времени)
                    update_worker_status(symbol, current_status="Running", last_action="Generating ML signals...")
                    if stop_event.is_set():
                        _log(f"🛑 Stop event received, stopping bot for {symbol}", symbol)
                        break
                    # Логируем, какая модель используется для этого символа
                    _log(f"🤖 Using ML model: {current_settings.ml_model_path}", symbol)
                    # Локальный alias для Action из ML сигналов
                    from bot.ml.strategy_ml import Action as MlAction

                    ml_signals = build_ml_signals(
                        df_ready,
                        current_settings.ml_model_path,
                        current_settings.ml_confidence_threshold,
                        current_settings.ml_min_signal_strength,
                        current_settings.ml_stability_filter,
                    )
                    # Обновляем статус после генерации
                    update_worker_status(symbol, current_status="Running", last_action="ML signals generated")
                    ml_generated = [s for s in ml_signals if s.action in (MlAction.LONG, MlAction.SHORT)]
                    _log(f"📊 ML strategy: generated {len(ml_signals)} total, {len(ml_generated)} actionable (LONG/SHORT)", symbol)
                    
                    # Детальное логирование последних 3 сигналов для диагностики (отсортированных по времени)
                    if ml_generated:
                        # Сортируем по timestamp и берем последние 3 (самые свежие)
                        sorted_signals = sorted(ml_generated, key=get_timestamp_for_sort)[-3:]
                        for i, sig in enumerate(sorted_signals):
                            ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                            _log(f"  [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
                    elif len(ml_signals) > 0:
                        # Показываем примеры HOLD сигналов и статистику по уверенности
                        hold_signals = [s for s in ml_signals if s.action == MlAction.HOLD]
                        if hold_signals:
                            _log(f"  Example HOLD signals: {[s.reason for s in hold_signals[:3]]}", symbol)
                            # Показываем статистику по уверенности модели
                            confidences = []
                            for sig in ml_signals:
                                if hasattr(sig, 'confidence') and sig.confidence is not None:
                                    confidences.append(sig.confidence)
                            if confidences:
                                _log(f"  💡 ML confidence stats: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}, threshold={current_settings.ml_confidence_threshold:.3f}", symbol)
                    else:
                        _log(f"  ⚠️ No ML signals generated at all", symbol)
                    
                    import re
                    min_strength_map = {
                        "слабое": 0,
                        "умеренное": 60,
                        "среднее": 70,
                        "сильное": 80,
                        "очень_сильное": 90
                    }
                    min_strength_pct = min_strength_map.get(current_settings.ml_min_signal_strength, 70)
                    
                    for sig in ml_generated:
                        # Проверяем, что сигнал прошел фильтр минимальной силы
                        should_filter = False
                        filter_reason = ""
                        
                        if "сила_слишком_слабая" in sig.reason:
                            should_filter = True
                            filter_reason = "сила слишком слабая"
                        elif "сила_слабое" in sig.reason:
                            # Проверяем процент уверенности из reason (формат: "ml_SHORT_сила_слабое_46%_...")
                            confidence_match = re.search(r'сила_слабое_(\d+)%', sig.reason)
                            if confidence_match:
                                confidence_pct = int(confidence_match.group(1))
                                if confidence_pct < min_strength_pct:
                                    should_filter = True
                                    filter_reason = f"confidence {confidence_pct}% < min {min_strength_pct}%"
                        
                        if should_filter:
                            ml_filtered.append((sig, filter_reason))
                            # Детальное логирование отфильтрованных сигналов убрано для уменьшения шума в логах
                            # Итоговая статистика выводится ниже
                        else:
                            ml_actionable.append(sig)
                            all_signals.append(sig)
                    
                    if ml_filtered:
                        _log(f"⛔ ML strategy: {len(ml_filtered)} signals filtered out (weak confidence, min required: {min_strength_pct}%)", symbol)
                except Exception as e:
                    print(f"[live] ❌ Error generating ML signals: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Разделяем сигналы по стратегиям
            # Только LONG и SHORT сигналы (HOLD игнорируем)
            from bot.strategy import Action as StrategyAction

            trend_signals_only = [
                s for s in all_signals
                if s.reason.startswith("trend_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            flat_signals_only = [
                s for s in all_signals
                if s.reason.startswith("range_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            ml_signals_only = [
                s for s in all_signals
                if s.reason.startswith("ml_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            momentum_signals_only = [
                s for s in all_signals
                if s.reason.startswith("momentum_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            liquidity_signals_only = [
                s for s in all_signals
                if s.reason.startswith("liquidity_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            smc_signals_only = [
                s for s in all_signals
                if s.reason.lower().startswith("smc_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            ict_signals_only = [
                s for s in all_signals
                if s.reason.startswith("ict_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            liquidation_hunter_signals_only = [
                s for s in all_signals
                if s.reason.startswith("liquidation_hunter_")
                and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            zscore_signals_only = [
                s for s in all_signals
                if s.reason.startswith("zscore_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            vbo_signals_only = [
                s for s in all_signals
                if s.reason.startswith("vbo_") and s.action in (StrategyAction.LONG, StrategyAction.SHORT)
            ]
            
            # Объединяем старые стратегии для обратной совместимости
            main_strategy_signals = trend_signals_only + flat_signals_only
            
            # Функция для проверки, является ли сигнал свежим (строгая проверка: не старше 15 минут от текущего времени)
            def is_signal_fresh(sig, df_ready):
                """Проверяет, является ли сигнал свежим (не старше 15 минут от текущего времени или соответствует последней свече)."""
                try:
                    if df_ready.empty:
                        return True
                    
                    ts = sig.timestamp
                    if isinstance(ts, pd.Timestamp):
                        signal_ts = ts
                        if signal_ts.tzinfo is None:
                            signal_ts = signal_ts.tz_localize('UTC')
                        else:
                            signal_ts = signal_ts.tz_convert('UTC')
                        
                        # Получаем текущее время
                        current_time_utc = datetime.now(timezone.utc)
                        if isinstance(current_time_utc, pd.Timestamp):
                            current_time_utc = current_time_utc.to_pydatetime()
                        
                        # Проверяем, соответствует ли timestamp сигнала последней свече или одной из последних
                        last_candle_ts = df_ready.index[-1]
                        if isinstance(last_candle_ts, pd.Timestamp):
                            if last_candle_ts.tzinfo is None:
                                last_candle_ts = last_candle_ts.tz_localize('UTC')
                            else:
                                last_candle_ts = last_candle_ts.tz_convert('UTC')
                            last_candle_time = last_candle_ts.to_pydatetime()
                        else:
                            last_candle_time = last_candle_ts
                                
                        # Если сигнал соответствует последней свече - он свежий
                        signal_time = signal_ts.to_pydatetime()
                        if abs((signal_time - last_candle_time).total_seconds()) <= 60:  # В пределах 1 минуты от последней свечи
                            return True
                        
                        # Также проверяем возраст от текущего времени (не старше 15 минут)
                        time_diff_from_now = abs((current_time_utc - signal_time).total_seconds())
                        if time_diff_from_now <= 900:  # 15 минут = 900 секунд
                            return True
                        
                    return False
                except Exception as e:
                    _log(f"⚠️ Error checking signal freshness: {e}", symbol=None)
                    # В случае ошибки считаем сигнал не свежим для строгости
                    return False
            
            # Фильтруем только свежие сигналы (не старше 15 минут = ~1 свеча на 15m таймфрейме)
            # Сигналы могут быть исполнены только один раз и только если они свежие
            fresh_main_signals = [s for s in main_strategy_signals if is_signal_fresh(s, df_ready)]
            fresh_ml_signals = [s for s in ml_signals_only if is_signal_fresh(s, df_ready)]
            fresh_trend_signals = [s for s in trend_signals_only if is_signal_fresh(s, df_ready)]
            fresh_flat_signals = [s for s in flat_signals_only if is_signal_fresh(s, df_ready)]
            fresh_momentum_signals = [s for s in momentum_signals_only if is_signal_fresh(s, df_ready)]
            fresh_liquidity_signals = [s for s in liquidity_signals_only if is_signal_fresh(s, df_ready)]
            fresh_smc_signals = [s for s in smc_signals_only if is_signal_fresh(s, df_ready)]
            fresh_ict_signals = [s for s in ict_signals_only if is_signal_fresh(s, df_ready)]
            
            # Убрано verbose сообщение о том, что сигналы не свежие - это нормальное поведение
            
            # Сортируем свежие сигналы по timestamp (от старых к новым) для правильного определения самого свежего
            # ВАЖНО: После сортировки последний элемент [-1] будет самым свежим
            if fresh_main_signals:
                fresh_main_signals.sort(key=get_timestamp_for_sort)  # Сортировка по возрастанию timestamp
            if fresh_ml_signals:
                fresh_ml_signals.sort(key=get_timestamp_for_sort)  # Сортировка по возрастанию timestamp
            
            # Убрано verbose диагностическое сообщение - логируется только при проблемах
            if fresh_smc_signals:
                sig = fresh_smc_signals[-1]
                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                _log(f"    Latest SMC: {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{ts_str}]", symbol)
            _log(f"  • Total actionable: {len(all_signals)} signals", symbol)
            if ml_filtered:
                _log(f"  • ML filtered out: {len(ml_filtered)} weak signals", symbol)
            
            # Определяем основной сигнал и дополнительный (только из свежих сигналов, самые свежие по timestamp)
            # Если есть несколько сигналов с одинаковым timestamp, выбираем самый последний по порядку добавления
            main_sig = None
            if fresh_main_signals:
                # Берем самый последний сигнал (самый свежий по timestamp)
                main_sig = fresh_main_signals[-1]
                # Если есть несколько сигналов с одинаковым timestamp, выбираем тот, который был добавлен последним
                # (сортируем по timestamp, затем берем последний)
            elif main_strategy_signals:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Если нет свежих сигналов, но есть сигналы вообще - используем последний
                # Это позволяет обрабатывать сигналы, которые могут быть немного старше, но все еще актуальны
                main_sig = main_strategy_signals[-1]
                # Убираем логи о fallback сигналах - это нормальная ситуация
            
            ml_sig = None
            if fresh_ml_signals:
                # Берем самый последний сигнал (самый свежий по timestamp)
                ml_sig = fresh_ml_signals[-1]
                # Если есть несколько сигналов с одинаковым timestamp, выбираем тот, который был добавлен последним
            elif ml_signals_only:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Если нет свежих ML сигналов, но есть сигналы вообще - используем последний
                ml_sig = ml_signals_only[-1]
                # Убираем логи о fallback сигналах - это нормальная ситуация
            
            # SMC сигнал
            smc_sig = None
            if fresh_smc_signals:
                # Берем самый последний сигнал (самый свежий по timestamp)
                smc_sig = fresh_smc_signals[-1]
            elif smc_signals_only:
                # Если нет свежих SMC сигналов, но есть сигналы вообще - используем последний
                smc_sig = smc_signals_only[-1]
            
            # ICT сигнал
            ict_sig = None
            if fresh_ict_signals:
                # Берем самый последний сигнал (самый свежий по timestamp)
                ict_sig = fresh_ict_signals[-1]
            elif ict_signals_only:
                # Если нет свежих ICT сигналов, но есть сигналы вообще - используем последний
                ict_sig = ict_signals_only[-1]
            
            if main_sig:
                ts_str = main_sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(main_sig.timestamp, 'strftime') else str(main_sig.timestamp)
                is_fresh = is_signal_fresh(main_sig, df_ready)
                freshness_marker = "FRESH" if is_fresh else "NOT FRESH (will be filtered)"
                print(f"[live]   🎯 Latest TREND/FLAT signal ({freshness_marker}): {main_sig.action.value} @ ${main_sig.price:.2f} ({main_sig.reason}) [{ts_str}]")
            if ml_sig:
                ts_str = ml_sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ml_sig.timestamp, 'strftime') else str(ml_sig.timestamp)
                is_fresh = is_signal_fresh(ml_sig, df_ready)
                freshness_marker = "FRESH" if is_fresh else "NOT FRESH (will be filtered)"
                print(f"[live]   🎯 Latest ML signal ({freshness_marker}): {ml_sig.action.value} @ ${ml_sig.price:.2f} ({ml_sig.reason}) [{ts_str}]")
            if smc_sig:
                ts_str = smc_sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(smc_sig.timestamp, 'strftime') else str(smc_sig.timestamp)
                is_fresh = is_signal_fresh(smc_sig, df_ready)
                freshness_marker = "FRESH" if is_fresh else "NOT FRESH (will be filtered)"
                print(f"[live]   🎯 Latest SMC signal ({freshness_marker}): {smc_sig.action.value} @ ${smc_sig.price:.2f} ({smc_sig.reason}) [{ts_str}]")
            
            # Функция для сохранения latest сигнала в историю (строго один сигнал от каждой стратегии)
            def save_latest_signal_to_history(sig, strategy_type_name: str, strategy_key: str):
                """Сохраняет latest сигнал в историю с проверками. Сохраняет только один сигнал от каждой стратегии за цикл."""
                try:
                    # Локальный alias для Action из rule-based стратегий
                    from bot.strategy import Action as StrategyActionLocal

                    if sig is None or sig.action == StrategyActionLocal.HOLD:
                        return  # Пропускаем HOLD сигналы и None
                    
                    ts_log = sig.timestamp
                    if isinstance(ts_log, pd.Timestamp):
                        if ts_log.tzinfo is None:
                            ts_log = ts_log.tz_localize('UTC')
                        else:
                            ts_log = ts_log.tz_convert('UTC')
                        ts_log = ts_log.to_pydatetime()
                    
                    # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                    # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                    ts_log = update_signal_timestamp_if_fresh(ts_log, strategy_type_name)
                    
                    strategy_type = get_strategy_type_from_signal(sig.reason)
                    
                    # Проверяем, не сохраняли ли мы уже сигнал от этой стратегии в этом цикле
                    # Используем strategy_key для отслеживания (например, "TREND/FLAT" или "ML")
                    if strategy_key in seen_signal_keys_cycle:
                        return  # Уже сохранили latest сигнал от этой стратегии в этом цикле
                    
                    # Дополнительная проверка для ML сигналов: не добавляем слабые сигналы в историю
                    if strategy_type == "ml" and "сила_слабое" in sig.reason:
                        # Извлекаем процент уверенности из reason (формат: "ml_SHORT_сила_слабое_46%_...")
                        import re
                        confidence_match = re.search(r'сила_слабое_(\d+)%', sig.reason)
                        if confidence_match:
                            confidence_pct = int(confidence_match.group(1))
                            # Проверяем минимальную силу сигнала из настроек
                            min_strength_map = {
                                "слабое": 0,
                                "умеренное": 60,
                                "среднее": 70,
                                "сильное": 80,
                                "очень_сильное": 90
                            }
                            min_strength_pct = min_strength_map.get(current_settings.ml_min_signal_strength, 70)
                            if confidence_pct < min_strength_pct:
                                _log(f"⛔ Skipping weak ML signal in history: {sig.reason} (confidence: {confidence_pct}% < min: {min_strength_pct}%)", symbol)
                                return
                    
                    # ВАЖНО: Сохраняем ВСЕ сигналы, не только свежие
                    # Это позволяет видеть все сгенерированные сигналы в истории
                    # Проверка свежести используется только для выбора "latest" сигнала для действия
                    # Но в историю сохраняем все сигналы для анализа
                    
                    # Используем signal_id из объекта Signal для сохранения в истории
                    sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                    
                    # Сохраняем сигнал в историю
                    # Дедупликация происходит на уровне add_signal (по timestamp, reason, price, symbol)
                    # Но мы дополнительно проверяем, что сохраняем только один сигнал от каждой стратегии за цикл
                    add_signal(
                        action=sig.action.value,
                        reason=sig.reason,
                        price=sig.price,
                        timestamp=ts_log,
                        symbol=symbol,
                        strategy_type=strategy_type,
                        signal_id=sig_signal_id,
                    )
                    
                    # Отмечаем, что мы сохранили latest сигнал от этой стратегии
                    seen_signal_keys_cycle.add(strategy_key)
                    
                    # Проверяем, является ли сигнал свежим после сохранения
                    is_fresh_after_save = False
                    try:
                        current_time_utc = datetime.now(timezone.utc)
                        age_from_now_minutes = abs((current_time_utc - ts_log).total_seconds()) / 60
                        is_fresh_after_save = age_from_now_minutes <= 15
                    except:
                        pass
                    
                    freshness_marker = "⚡ FRESH" if is_fresh_after_save else "⏳ NOT FRESH"
                    _log(f"💾 Saved latest {strategy_type_name} signal to history: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_log.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts_log, 'strftime') else ts_log}] {freshness_marker}", symbol)
                except Exception as e:
                    print(f"[live] ⚠️ Warning: Failed to save latest {strategy_type_name} signal to history: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Сохраняем latest сигналы в историю (строго один сигнал от каждой стратегии - те же, что показываются в логах)
            # Это гарантирует согласованность между логами и историей сигналов
            try:
                # Сохраняем только один latest сигнал от TREND/FLAT стратегии
                if main_sig:
                    save_latest_signal_to_history(main_sig, "TREND/FLAT", "TREND/FLAT_LATEST")
                else:
                    # Если нет свежего сигнала, но есть сигналы вообще - сохраняем последний
                    if main_strategy_signals:
                        last_sig = main_strategy_signals[-1]
                        if last_sig.action in (Action.LONG, Action.SHORT):
                            save_latest_signal_to_history(last_sig, "TREND/FLAT", "TREND/FLAT_LATEST")
                    # Убираем логи о том, что нет сигналов - это нормальная ситуация
                
                # Сохраняем только один latest сигнал от ML стратегии
                if ml_sig:
                    save_latest_signal_to_history(ml_sig, "ML", "ML_LATEST")
                # Убираем логи о том, что нет ML сигналов - это нормальная ситуация
                
                # Сохраняем latest сигналы от новых стратегий
                # Momentum стратегия
                momentum_sig = None
                if momentum_signals_only:
                    momentum_signals_only.sort(key=get_timestamp_for_sort)
                    momentum_sig = momentum_signals_only[-1] if momentum_signals_only else None
                    if momentum_sig:
                        save_latest_signal_to_history(momentum_sig, "MOMENTUM", "MOMENTUM_LATEST")
                
                # Liquidity Sweep стратегия
                liquidity_sig_latest = None
                if liquidity_signals_only:
                    liquidity_signals_only.sort(key=get_timestamp_for_sort)
                    liquidity_sig_latest = liquidity_signals_only[-1] if liquidity_signals_only else None
                    if liquidity_sig_latest:
                        save_latest_signal_to_history(liquidity_sig_latest, "LIQUIDITY", "LIQUIDITY_LATEST")
                
                # SMC стратегия
                smc_sig_save = None
                if smc_signals_only:
                    smc_signals_only.sort(key=get_timestamp_for_sort)
                    smc_sig_save = smc_signals_only[-1] if smc_signals_only else None
                    if smc_sig_save:
                        save_latest_signal_to_history(smc_sig_save, "SMC", "SMC_LATEST")
                
                # ICT стратегия
                ict_sig_save = None
                if ict_signals_only:
                    ict_signals_only.sort(key=get_timestamp_for_sort)
                    ict_sig_save = ict_signals_only[-1] if ict_signals_only else None
                    if ict_sig_save:
                        save_latest_signal_to_history(ict_sig_save, "ICT", "ICT_LATEST")
                
                # Liquidation Hunter стратегия
                liquidation_hunter_sig_save = None
                if liquidation_hunter_signals_only:
                    liquidation_hunter_signals_only.sort(key=get_timestamp_for_sort)
                    liquidation_hunter_sig_save = liquidation_hunter_signals_only[-1] if liquidation_hunter_signals_only else None
                    if liquidation_hunter_sig_save:
                        save_latest_signal_to_history(liquidation_hunter_sig_save, "LIQUIDATION_HUNTER", "LIQUIDATION_HUNTER_LATEST")
                
                # Z-Score стратегия
                zscore_sig_save = None
                if zscore_signals_only:
                    zscore_signals_only.sort(key=get_timestamp_for_sort)
                    zscore_sig_save = zscore_signals_only[-1] if zscore_signals_only else None
                    if zscore_sig_save:
                        save_latest_signal_to_history(zscore_sig_save, "ZSCORE", "ZSCORE_LATEST")
                
                # VBO стратегия
                vbo_sig_save = None
                if vbo_signals_only:
                    vbo_signals_only.sort(key=get_timestamp_for_sort)
                    vbo_sig_save = vbo_signals_only[-1] if vbo_signals_only else None
                    if vbo_sig_save:
                        save_latest_signal_to_history(vbo_sig_save, "VBO", "VBO_LATEST")
                
                # ДОПОЛНИТЕЛЬНО: Сохраняем ВСЕ сигналы от всех стратегий (не только свежие)
                # Это позволяет видеть все сигналы в истории для анализа
                additional_saved = 0
                for sig in main_strategy_signals:
                    # Сохраняем все сигналы LONG/SHORT, даже если они не свежие
                    # main_sig уже сохранен выше, поэтому пропускаем его здесь
                    from bot.strategy import Action as StrategyActionHistory
                    if sig != main_sig and sig.action in (StrategyActionHistory.LONG, StrategyActionHistory.SHORT):
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                            # Убрано детальное логирование каждого сигнала для уменьшения шума в логах
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional signal to history: {e}", symbol)
                
                # ВАЖНО: Если main_sig был None, но есть сигналы - сохраняем их все
                # Это гарантирует, что сигналы сохраняются даже если они не свежие
                if not main_sig and main_strategy_signals:
                    _log(f"💾 No fresh main_sig, but saving all {len(main_strategy_signals)} TREND/FLAT signals", symbol)
                    for sig in main_strategy_signals:
                        from bot.strategy import Action as StrategyActionHistory2
                        if sig.action in (StrategyActionHistory2.LONG, StrategyActionHistory2.SHORT):
                            try:
                                strategy_type = get_strategy_type_from_signal(sig.reason)
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                                # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                                ts_log = update_signal_timestamp_if_fresh(ts_log)
                                
                                sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                                
                                # Проверяем свежесть для логирования
                                is_fresh = is_signal_fresh(sig, df_ready)
                                freshness_note = "fresh" if is_fresh else "not fresh"
                                
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type=strategy_type,
                                    signal_id=sig_signal_id,
                                )
                                additional_saved += 1
                                # Убрано детальное логирование каждого не свежего сигнала для уменьшения шума в логах
                            except Exception as e:
                                _log(f"⚠️ Failed to save signal to history: {e}", symbol)
                                import traceback
                                traceback.print_exc()
                
                # КРИТИЧЕСКИ ВАЖНО: Сохраняем ВСЕ сигналы от тренд/флэт стратегий, даже если они не свежие
                # Это гарантирует, что сигналы попадают в историю независимо от фильтра свежести
                if main_strategy_signals:
                    from bot.strategy import Action as StrategyActionHist
                    for sig in main_strategy_signals:
                        if sig.action in (StrategyActionHist.LONG, StrategyActionHist.SHORT):
                            # Пропускаем только если это main_sig (уже сохранен выше)
                            if sig == main_sig:
                                continue
                            try:
                                strategy_type = get_strategy_type_from_signal(sig.reason)
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                                # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                                ts_log = update_signal_timestamp_if_fresh(ts_log)
                                
                                sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                                
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type=strategy_type,
                                    signal_id=sig_signal_id,
                                )
                                additional_saved += 1
                                # Убрано детальное логирование каждого не свежего сигнала для уменьшения шума в логах
                            except Exception as e:
                                _log(f"⚠️ Failed to save signal to history: {e}", symbol)
                                import traceback
                                traceback.print_exc()
                
                for sig in ml_signals_only:
                    # Используем MlAction (alias для ML‑сигналов), а не общий Action
                    if sig != ml_sig and sig.action in (MlAction.LONG, MlAction.SHORT):  # Не сохраняем дубликат и только LONG/SHORT
                        # Проверяем, что сигнал не был отфильтрован (не должен быть в ml_filtered)
                        # Сигналы в ml_signals_only уже прошли фильтрацию, но для безопасности проверяем еще раз
                        should_skip = False
                        if "сила_слабое" in sig.reason:
                            import re
                            confidence_match = re.search(r'сила_слабое_(\d+)%', sig.reason)
                            if confidence_match:
                                confidence_pct = int(confidence_match.group(1))
                                min_strength_map = {
                                    "слабое": 0,
                                    "умеренное": 60,
                                    "среднее": 70,
                                    "сильное": 80,
                                    "очень_сильное": 90
                                }
                                min_strength_pct = min_strength_map.get(current_settings.ml_min_signal_strength, 70)
                                if confidence_pct < min_strength_pct:
                                    should_skip = True
                                    _log(f"⛔ Skipping filtered ML signal in additional save: {sig.reason} (confidence: {confidence_pct}% < min: {min_strength_pct}%)", symbol)
                        
                        if should_skip:
                            continue
                        
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional ML signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от новых стратегий (включая latest, если они есть)
                from bot.strategy import Action as StrategyActionMomentum
                for sig in momentum_signals_only:
                    if sig.action in (StrategyActionMomentum.LONG, StrategyActionMomentum.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == momentum_sig and momentum_sig:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional MOMENTUM signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от LIQUIDITY стратегии
                from bot.liquidation_hunter_strategy import Action as StrategyActionLH  # локальный alias
                for sig in liquidity_signals_only:
                    if sig.action in (StrategyActionLH.LONG, StrategyActionLH.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == liquidity_sig_latest and liquidity_sig_latest:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional LIQUIDITY signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от Liquidation Hunter стратегии
                # Используем локальный alias для Action из liquidation_hunter стратегии
                from bot.liquidation_hunter_strategy import Action as StrategyActionLH
                for sig in liquidation_hunter_signals_only:
                    if sig.action in (StrategyActionLH.LONG, StrategyActionLH.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == liquidation_hunter_sig_save and liquidation_hunter_sig_save:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional LIQUIDATION_HUNTER signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от Z-Score стратегии
                # Используем локальный alias для Action из ZSCORE стратегии
                from bot.zscore_strategy import Action as StrategyActionZscore
                for sig in zscore_signals_only:
                    if sig.action in (StrategyActionZscore.LONG, StrategyActionZscore.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == zscore_sig_save and zscore_sig_save:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional ZSCORE signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от VBO стратегии
                # Используем локальный alias для Action из VBO стратегии
                from bot.vbo_strategy import Action as StrategyActionVbo
                for sig in vbo_signals_only:
                    if sig.action in (StrategyActionVbo.LONG, StrategyActionVbo.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == vbo_sig_save and vbo_sig_save:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional VBO signal to history: {e}", symbol)
                
                # Сохраняем все сигналы от ICT стратегии
                # Используем локальный alias для Action из ICT стратегии
                from bot.ict_strategy import Action as StrategyActionIct
                for sig in ict_signals_only:
                    if sig.action in (StrategyActionIct.LONG, StrategyActionIct.SHORT):
                        # Пропускаем только если это latest сигнал и он уже был сохранен выше
                        if sig == ict_sig_save and ict_sig_save:
                            continue
                        try:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЖНО: Если сигнал соответствует последней свече - обновляем timestamp на текущее время
                            # Это гарантирует, что сигнал будет считаться свежим и обработан немедленно
                            ts_log = update_signal_timestamp_if_fresh(ts_log)
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            additional_saved += 1
                        except Exception as e:
                            _log(f"⚠️ Failed to save additional ICT signal to history: {e}", symbol)
                
                if additional_saved > 0:
                    _log(f"💾 Saved {additional_saved} additional signals to history", symbol)
            except Exception as e:
                _log(f"⚠️ Warning: Failed to save latest signals to history: {e}", symbol)
                import traceback
                traceback.print_exc()
            
            # Функция для обновления timestamp в объекте сигнала, если он соответствует последней свече
            def update_signal_object_timestamp_if_fresh(sig):
                """Обновляет timestamp в объекте сигнала на текущее время, если он соответствует последней свече."""
                if sig is None or df_ready.empty:
                    return sig
                
                try:
                    signal_ts = sig.timestamp
                    if isinstance(signal_ts, pd.Timestamp):
                        if signal_ts.tzinfo is None:
                            signal_ts = signal_ts.tz_localize('UTC')
                        else:
                            signal_ts = signal_ts.tz_convert('UTC')
                        signal_ts_py = signal_ts.to_pydatetime()
                    else:
                        signal_ts_py = signal_ts
                    
                    last_candle_ts = df_ready.index[-1]
                    if isinstance(last_candle_ts, pd.Timestamp):
                        if last_candle_ts.tzinfo is None:
                            last_candle_ts = last_candle_ts.tz_localize('UTC')
                        else:
                            last_candle_ts = last_candle_ts.tz_convert('UTC')
                        last_candle_time = last_candle_ts.to_pydatetime()
                        
                        # Проверяем, соответствует ли timestamp сигнала последней свече (в пределах 1 минуты)
                        time_diff_seconds = abs((signal_ts_py - last_candle_time).total_seconds())
                        if time_diff_seconds <= 60:  # 1 минута
                            # Обновляем timestamp в объекте сигнала на текущее время
                            updated_ts = datetime.now(timezone.utc)
                            # Создаем Timestamp: если updated_ts уже с tzinfo, используем tz_convert, иначе tz_localize
                            if updated_ts.tzinfo is not None:
                                sig.timestamp = pd.Timestamp(updated_ts).tz_convert('UTC')
                            else:
                                sig.timestamp = pd.Timestamp(updated_ts, tz='UTC')
                except Exception as e:
                    _log(f"⚠️ Error updating signal object timestamp: {e}", symbol)
                
                return sig
            
            # Функция для получения последнего свежего сигнала из списка
            def get_latest_fresh_signal(signal_list, df_ready):
                """Получает последний свежий сигнал из списка."""
                if not signal_list:
                    return None
                fresh_signals = [s for s in signal_list if is_signal_fresh(s, df_ready)]
                if fresh_signals:
                    fresh_signals.sort(key=get_timestamp_for_sort)
                    sig = fresh_signals[-1]
                    # Обновляем timestamp в объекте сигнала, если он соответствует последней свече
                    return update_signal_object_timestamp_if_fresh(sig)
                # Если нет свежих, возвращаем последний из всех
                signal_list.sort(key=get_timestamp_for_sort)
                sig = signal_list[-1] if signal_list else None
                # Обновляем timestamp в объекте сигнала, если он соответствует последней свече
                return update_signal_object_timestamp_if_fresh(sig) if sig else None
            
            # Получаем последние сигналы от каждой стратегии
            trend_sig = get_latest_fresh_signal(trend_signals_only, df_ready)
            flat_sig = get_latest_fresh_signal(flat_signals_only, df_ready)
            ml_sig_latest = get_latest_fresh_signal(ml_signals_only, df_ready)
            momentum_sig = get_latest_fresh_signal(momentum_signals_only, df_ready)
            liquidity_sig = get_latest_fresh_signal(liquidity_signals_only, df_ready)
            smc_sig_latest = get_latest_fresh_signal(smc_signals_only, df_ready)
            ict_sig_latest = get_latest_fresh_signal(ict_signals_only, df_ready)
            liquidation_hunter_sig_latest = get_latest_fresh_signal(liquidation_hunter_signals_only, df_ready)
            
            # ВАЖНО: Для liquidation_hunter требуется минимум 2 подтверждающих сигнала в одном направлении за 5 минут
            # Загружаем сигналы из истории для проверки подтверждения
            if liquidation_hunter_sig_latest:
                try:
                    from bot.web.history import get_signals
                    # Получаем сигналы liquidation_hunter из истории за последние 10 минут (для проверки подтверждения)
                    history_signals_raw = get_signals(limit=100, symbol_filter=symbol)
                    history_liquidation_hunter_signals = []
                    
                    for hist_sig in history_signals_raw:
                        hist_reason = hist_sig.get("reason", "")
                        hist_strategy = hist_sig.get("strategy_type", "").lower()
                        if hist_reason.startswith("liquidation_hunter_") or hist_strategy == "liquidation_hunter":
                            # Создаем объект сигнала из истории для проверки
                            hist_action_str = hist_sig.get("action", "").lower()
                            if hist_action_str in ("long", "short"):
                                hist_action = Action.LONG if hist_action_str == "long" else Action.SHORT
                                
                                # Парсим timestamp из истории
                                hist_timestamp_str = hist_sig.get("timestamp", "")
                                try:
                                    if isinstance(hist_timestamp_str, str):
                                        # Пробуем разные форматы
                                        try:
                                            hist_ts = datetime.fromisoformat(hist_timestamp_str.replace('Z', '+00:00'))
                                        except:
                                            try:
                                                hist_ts = pd.to_datetime(hist_timestamp_str, utc=True).to_pydatetime()
                                            except:
                                                continue
                                    else:
                                        hist_ts = hist_timestamp_str
                                    
                                    # Создаем простой объект сигнала для проверки
                                    class HistorySignal:
                                        def __init__(self, action, price, reason, timestamp):
                                            self.action = action
                                            self.price = price
                                            self.reason = reason
                                            self.timestamp = timestamp
                                    
                                    hist_signal_obj = HistorySignal(
                                        action=hist_action,
                                        price=float(hist_sig.get("price", 0)),
                                        reason=hist_reason,
                                        timestamp=hist_ts
                                    )
                                    history_liquidation_hunter_signals.append(hist_signal_obj)
                                except Exception:
                                    continue
                    
                    # Объединяем сигналы из текущего цикла и из истории
                    all_liquidation_hunter_for_confirmation = list(liquidation_hunter_signals_only) + history_liquidation_hunter_signals
                    
                    is_confirmed, confirmation_count, confirming_signals = _check_liquidation_hunter_confirmation(
                        signal=liquidation_hunter_sig_latest,
                        all_liquidation_hunter_signals=all_liquidation_hunter_for_confirmation,
                        confirmation_window_minutes=5,
                        min_confirmations=2,
                        symbol=symbol
                    )
                    if not is_confirmed:
                        _log(f"⛔ LIQUIDATION_HUNTER signal REJECTED: insufficient confirmations ({confirmation_count}/2) for {liquidation_hunter_sig_latest.action.value} @ ${liquidation_hunter_sig_latest.price:.2f}", symbol)
                        liquidation_hunter_sig_latest = None  # Отклоняем сигнал без достаточного подтверждения
                    else:
                        _log(f"✅ LIQUIDATION_HUNTER signal CONFIRMED: {confirmation_count} confirmations for {liquidation_hunter_sig_latest.action.value} @ ${liquidation_hunter_sig_latest.price:.2f}", symbol)
                except Exception as e:
                    _log(f"⚠️ Error checking LIQUIDATION_HUNTER confirmation from history: {e}", symbol)
                    # В случае ошибки используем только сигналы из текущего цикла
                    is_confirmed, confirmation_count, confirming_signals = _check_liquidation_hunter_confirmation(
                        signal=liquidation_hunter_sig_latest,
                        all_liquidation_hunter_signals=liquidation_hunter_signals_only,
                        confirmation_window_minutes=5,
                        min_confirmations=2,
                        symbol=symbol
                    )
                    if not is_confirmed:
                        _log(f"⛔ LIQUIDATION_HUNTER signal REJECTED: insufficient confirmations ({confirmation_count}/2) for {liquidation_hunter_sig_latest.action.value} @ ${liquidation_hunter_sig_latest.price:.2f}", symbol)
                        liquidation_hunter_sig_latest = None
                    else:
                        _log(f"✅ LIQUIDATION_HUNTER signal CONFIRMED: {confirmation_count} confirmations for {liquidation_hunter_sig_latest.action.value} @ ${liquidation_hunter_sig_latest.price:.2f}", symbol)
            
            zscore_sig_latest = get_latest_fresh_signal(zscore_signals_only, df_ready)
            vbo_sig_latest = get_latest_fresh_signal(vbo_signals_only, df_ready)
            
            # Создаем словарь всех сигналов для удобства
            strategy_signals = {
                "trend": trend_sig,
                "flat": flat_sig,
                "ml": ml_sig_latest,
                "momentum": momentum_sig,
                "liquidity": liquidity_sig,
                "smc": smc_sig_latest,
                "ict": ict_sig_latest,
                "liquidation_hunter": liquidation_hunter_sig_latest,
                "zscore": zscore_sig_latest,
                "vbo": vbo_sig_latest,
            }
            
            # Логируем все доступные сигналы для отладки
            if zscore_sig_latest:
                is_fresh_zscore = is_signal_fresh(zscore_sig_latest, df_ready)
                # Дополнительно проверяем возраст от текущего времени для более точной оценки
                age_from_now_minutes = None
                try:
                    if isinstance(zscore_sig_latest.timestamp, pd.Timestamp):
                        signal_ts = zscore_sig_latest.timestamp
                        if signal_ts.tzinfo is None:
                            signal_ts = signal_ts.tz_localize('UTC')
                        else:
                            signal_ts = signal_ts.tz_convert('UTC')
                        current_time_utc = datetime.now(timezone.utc)
                        age_from_now_minutes = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                except Exception:
                    pass
                
                ts_str_zscore = zscore_sig_latest.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(zscore_sig_latest.timestamp, 'strftime') else str(zscore_sig_latest.timestamp)
                age_str = f", age: {age_from_now_minutes:.1f} min" if age_from_now_minutes is not None else ""
                # Сигнал считается действительно свежим только если он свежий по функции И возраст <= 15 минут
                is_really_fresh = is_fresh_zscore and (age_from_now_minutes is None or age_from_now_minutes <= 15)
                _log(f"🔍 ZSCORE signal available: {zscore_sig_latest.action.value} @ ${zscore_sig_latest.price:.2f} ({zscore_sig_latest.reason}) [{ts_str_zscore}] fresh={is_really_fresh} (is_fresh={is_fresh_zscore}{age_str})", symbol)
            
            # Для обратной совместимости сохраняем main_sig и ml_sig
            main_sig = trend_sig if trend_sig else flat_sig
            ml_sig = ml_sig_latest
            
            # Логика обработки сигналов от разных стратегий
            sig = None
            should_add_to_position = False  # Флаг для добавления к позиции при подтверждении
            
            # Получаем приоритет стратегии для текущей пары
            strategy_priority = symbol_strategy_settings.strategy_priority
            _log(f"📋 Strategy priority for {symbol}: {strategy_priority.upper()}", symbol)
            
            # Проверяем тренд BTC для фильтрации сигналов других пар
            btc_trend = None  # "bullish", "bearish", или None (если BTC не в активных парах или это сам BTC)
            if symbol != "BTCUSDT" and "BTCUSDT" in current_settings.active_symbols:
                try:
                    # Получаем данные BTC для определения тренда
                    btc_df = client.get_kline_df(symbol="BTCUSDT", interval=_timeframe_to_bybit_interval(current_settings.timeframe), limit=50)
                    if not btc_df.empty and len(btc_df) >= 20:
                        # Используем EMA 20 для определения тренда
                        from bot.indicators import compute_ema_indicators
                        btc_df = compute_ema_indicators(btc_df, ema_fast_length=20, ema_slow_length=50)
                        if 'ema_20' in btc_df.columns:
                            current_btc_price = float(btc_df.iloc[-1]['close'])
                            btc_ema_20 = float(btc_df.iloc[-1]['ema_20'])
                            
                            # Если цена выше EMA 20 - бычий тренд, ниже - медвежий
                            if current_btc_price > btc_ema_20 * 1.001:  # 0.1% запас для фильтрации шума
                                btc_trend = "bullish"
                                _log(f"📈 BTC Trend: BULLISH (Price: ${current_btc_price:.2f} > EMA20: ${btc_ema_20:.2f}) - приоритет LONG для {symbol}", symbol)
                            elif current_btc_price < btc_ema_20 * 0.999:  # 0.1% запас
                                btc_trend = "bearish"
                                _log(f"📉 BTC Trend: BEARISH (Price: ${current_btc_price:.2f} < EMA20: ${btc_ema_20:.2f}) - приоритет SHORT для {symbol}", symbol)
                    else:
                                _log(f"➡️ BTC Trend: NEUTRAL (Price: ${current_btc_price:.2f} ≈ EMA20: ${btc_ema_20:.2f}) - нет фильтрации для {symbol}", symbol)
                except Exception as e:
                    _log(f"⚠️ Error getting BTC trend: {e}", symbol)
            
            # Собираем все доступные сигналы (не None)
            available_signals = [(name, sig_obj) for name, sig_obj in strategy_signals.items() if sig_obj is not None]
            
            # ВАЖНО: Проверяем наличие свежих сигналов сразу после их сохранения в историю
            # Это позволяет обрабатывать свежие сигналы немедленно, как только они попадают в таблицу
            fresh_signals_available = False
            
            # Сначала проверяем свежесть сигналов из объектов (уже обновленных через update_signal_object_timestamp_if_fresh)
            if available_signals:
                # Проверяем, есть ли свежие сигналы (в пределах 15 минут)
                for name, s in available_signals:
                    if is_signal_fresh(s, df_ready):
                        # Дополнительно проверяем возраст от текущего времени
                        try:
                            if isinstance(s.timestamp, pd.Timestamp):
                                signal_ts = s.timestamp
                                if signal_ts.tzinfo is None:
                                    signal_ts = signal_ts.tz_localize('UTC')
                                else:
                                    signal_ts = signal_ts.tz_convert('UTC')
                                current_time_utc = datetime.now(timezone.utc)
                                age_from_now_minutes = abs(
                                    (current_time_utc - signal_ts.to_pydatetime()).total_seconds()
                                ) / 60
                                if age_from_now_minutes <= 15:
                                    fresh_signals_available = True
                                    break
                        except Exception:
                            pass
            
            # ДОПОЛНИТЕЛЬНО: Проверяем свежесть сигналов из истории (с обновленными timestamp)
            # Это гарантирует, что сигналы, только что сохраненные в историю, будут обнаружены немедленно
            if not fresh_signals_available:
                try:
                    from bot.web.history import get_signals
                    # Получаем последние сигналы из истории для текущего символа
                    recent_signals = get_signals(limit=10, symbol_filter=symbol)
                    current_time_utc = datetime.now(timezone.utc)
                    
                    for hist_signal in recent_signals:
                        try:
                            # Получаем timestamp из истории
                            hist_timestamp_str = hist_signal.get("timestamp", "")
                            if not hist_timestamp_str:
                                continue
                            
                            # Парсим timestamp
                            if isinstance(hist_timestamp_str, str):
                                # Пробуем разные форматы
                                try:
                                    hist_ts = pd.Timestamp(hist_timestamp_str)
                                except:
                                    continue
                            else:
                                hist_ts = pd.Timestamp(hist_timestamp_str)
                            
                            # Нормализуем timezone
                            if hist_ts.tzinfo is None:
                                hist_ts = hist_ts.tz_localize('UTC')
                            else:
                                hist_ts = hist_ts.tz_convert('UTC')
                            
                            hist_ts_py = hist_ts.to_pydatetime()
                            
                            # Проверяем возраст сигнала (должен быть не старше 15 минут)
                            age_from_now_minutes = abs((current_time_utc - hist_ts_py).total_seconds()) / 60
                            
                            if age_from_now_minutes <= 15:
                                # Сигнал свежий - проверяем, что он actionable (не HOLD)
                                hist_action = hist_signal.get("action", "").upper()
                                if hist_action in ("LONG", "SHORT"):
                                    fresh_signals_available = True
                                    _log(f"⚡ Fresh signal detected from history: {hist_action} @ ${hist_signal.get('price', 0):.2f} ({hist_signal.get('reason', '')}) - age: {age_from_now_minutes:.1f} min", symbol)
                                    break
                        except Exception as e:
                            # Пропускаем сигналы с ошибками парсинга
                            continue
                except Exception as e:
                    # Если не удалось проверить историю - продолжаем с проверкой объектов
                    pass
            
            # Если есть свежие сигналы - логируем для информации
            if fresh_signals_available:
                _log(f"⚡ Fresh signals detected - will process immediately (using 1s interval for instant processing)", symbol)
            else:
                _log(f"⏳ No fresh signals detected - will use normal interval ({current_settings.live_poll_seconds}s)", symbol)
            
            # Фильтруем сигналы по тренду BTC (если BTC в активных парах и это не сам BTC)
            if btc_trend and available_signals:
                filtered_signals = []
                for name, sig in available_signals:
                    # Если BTC бычий - приоритет LONG, если медвежий - приоритет SHORT
                    if btc_trend == "bullish" and sig.action == Action.LONG:
                        filtered_signals.append((name, sig))
                        _log(f"✅ Signal {name} ({sig.action.value}) passed BTC bullish filter", symbol)
                    elif btc_trend == "bearish" and sig.action == Action.SHORT:
                        filtered_signals.append((name, sig))
                        _log(f"✅ Signal {name} ({sig.action.value}) passed BTC bearish filter", symbol)
                    elif btc_trend == "bullish" and sig.action == Action.SHORT:
                        _log(f"⏸️ Signal {name} ({sig.action.value}) filtered out (BTC bullish, prefer LONG)", symbol)
                    elif btc_trend == "bearish" and sig.action == Action.LONG:
                        _log(f"⏸️ Signal {name} ({sig.action.value}) filtered out (BTC bearish, prefer SHORT)", symbol)
                else:
                        # HOLD сигналы всегда проходят
                        filtered_signals.append((name, sig))
                
                # Если после фильтрации остались сигналы - используем их, иначе используем все
                if filtered_signals:
                    available_signals = filtered_signals
                    _log(f"📊 BTC filter applied: {len(filtered_signals)}/{len(strategy_signals)} signals passed", symbol)
                else:
                    _log(f"⚠️ BTC filter removed all signals, using all available signals", symbol)
            
            if not available_signals:
                # Нет сигналов вообще
                if bot_state:
                    bot_state["current_status"] = "Running"
                    bot_state["last_action"] = "No signals found, waiting..."
                    bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                update_worker_status(symbol, current_status="Running", last_action="No signals found, waiting...")
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    break
                continue
            
            # 3. Выбираем основной сигнал на основе приоритета и свежести
            print(f"[live] 🔍 [{symbol}] Signal selection: {len(available_signals)} available signals")
            is_fallback_signal = False  # Флаг для fallback сигналов (когда нет свежих)
            for name, s in available_signals:
                is_fresh = is_signal_fresh(s, df_ready)
                # Дополнительно проверяем возраст от текущего времени для более точной оценки
                age_from_now_minutes = None
                is_strictly_fresh = False
                try:
                    if isinstance(s.timestamp, pd.Timestamp):
                        signal_ts = s.timestamp
                        if signal_ts.tzinfo is None:
                            signal_ts = signal_ts.tz_localize('UTC')
                        else:
                            signal_ts = signal_ts.tz_convert('UTC')
                        current_time_utc = datetime.now(timezone.utc)
                        age_from_now_minutes = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                        is_strictly_fresh = age_from_now_minutes <= 15
                except Exception:
                    pass
                
                ts_str = s.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(s.timestamp, 'strftime') else str(s.timestamp)
                age_str = f", age: {age_from_now_minutes:.1f} min" if age_from_now_minutes is not None else ""
                # Показываем оба значения: is_fresh (от функции) и is_strictly_fresh (строгая проверка возраста)
                print(f"[live]   - {name.upper()}: {s.action.value} @ ${s.price:.2f} ({s.reason}) [{ts_str}] fresh={is_strictly_fresh} (is_fresh={is_fresh}{age_str})")
            
            if len(available_signals) == 1:
                sig = available_signals[0][1]
                strategy_name = available_signals[0][0]
                # Проверяем, является ли единственный сигнал свежим
                if not is_signal_fresh(sig, df_ready):
                    # КРИТИЧЕСКИ ВАЖНО: Не обрабатываем старые сигналы, даже если это единственный доступный
                    sig = None
                    ts_str = available_signals[0][1].timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(available_signals[0][1].timestamp, 'strftime') else str(available_signals[0][1].timestamp)
                    print(f"[live] ⏳ Only one signal available from {strategy_name.upper()}, but it's not fresh (timestamp: {ts_str}). Waiting for fresh signals (max age: 15 minutes)...")
                else:
                    print(f"[live] ✅ Selected {strategy_name.upper()} signal: {sig.action.value} ({sig.reason}) @ ${sig.price:.2f}")
            else:
                # 1. Сначала определяем свежие сигналы
                fresh_available = [(name, s) for name, s in available_signals if is_signal_fresh(s, df_ready)]
                if not df_ready.empty:
                    last_candle_ts = df_ready.index[-1]
                    last_candle_str = last_candle_ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_candle_ts, 'strftime') else str(last_candle_ts)
                    print(f"[live] 🔍 [{symbol}] Fresh signals: {len(fresh_available)}/{len(available_signals)} (last candle: {last_candle_str})")
                else:
                    print(f"[live] 🔍 [{symbol}] Fresh signals: {len(fresh_available)}/{len(available_signals)}")
                
                if strategy_priority == "confluence":
                    # Режим Конфлюэнции: Требуется подтверждение минимум от двух стратегий
                    # Но разрешаем открытие при 1 свежем сигнале от приоритетной стратегии (SMC, ML), если нет конфликта
                    long_fresh = [s for name, s in fresh_available if s.action == Action.LONG]
                    short_fresh = [s for name, s in fresh_available if s.action == Action.SHORT]
                    
                    if len(long_fresh) >= 2:
                        long_fresh.sort(key=get_timestamp_for_sort)
                        sig = long_fresh[-1]
                        print(f"[live] 💎 CONFLUENCE LONG: {len(long_fresh)} strategies agree! Using latest: {sig.reason}")
                    elif len(short_fresh) >= 2:
                        short_fresh.sort(key=get_timestamp_for_sort)
                        sig = short_fresh[-1]
                        print(f"[live] 💎 CONFLUENCE SHORT: {len(short_fresh)} strategies agree! Using latest: {sig.reason}")
                    elif long_fresh and short_fresh:
                        print(f"[live] ⚠️ Confluence conflict: LONG vs SHORT fresh signals. Skipping.")
                        sig = None
                    elif len(long_fresh) == 1 and not short_fresh:
                        # 1 свежий LONG сигнал, нет конфликта - проверяем приоритет
                        sig = long_fresh[0]
                        sig_name = next((name for name, s in fresh_available if s == sig), "Unknown")
                        # Проверяем, что это приоритетная стратегия (SMC или ML)
                        if sig_name.lower() in ["smc", "ml"]:
                            print(f"[live] 💎 CONFLUENCE LONG (PRIORITY): 1 {sig_name.upper()} signal, no conflict. Using: {sig.reason}")
                        else:
                            print(f"[live] ⏳ Confluence: 1 fresh signal ({sig_name}), but not from priority strategy (SMC/ML). Waiting for confirmation.")
                            sig = None
                    elif len(short_fresh) == 1 and not long_fresh:
                        # 1 свежий SHORT сигнал, нет конфликта - проверяем приоритет
                        sig = short_fresh[0]
                        sig_name = next((name for name, s in fresh_available if s == sig), "Unknown")
                        # Проверяем, что это приоритетная стратегия (SMC или ML)
                        if sig_name.lower() in ["smc", "ml"]:
                            print(f"[live] 💎 CONFLUENCE SHORT (PRIORITY): 1 {sig_name.upper()} signal, no conflict. Using: {sig.reason}")
                        else:
                            print(f"[live] ⏳ Confluence: 1 fresh signal ({sig_name}), but not from priority strategy (SMC/ML). Waiting for confirmation.")
                            sig = None
                    else:
                        print(f"[live] ⏳ Confluence: Waiting for confirmation (fresh: {len(fresh_available)}).")
                        sig = None
                elif strategy_priority == "hybrid":
                    # Гибридный режим: Выбираем самый свежий из всех доступных СВЕЖИХ сигналов
                    # БЕЗ приоритета какой-то определенной стратегии
                    print(f"[live] 🔍 Hybrid mode: {len(fresh_available)} fresh, {len(available_signals)} total signals available")
                    if fresh_available:
                        # Если есть свежие сигналы - выбираем самый свежий по timestamp
                        # КРИТИЧЕСКИ ВАЖНО: Выбираем ТОЛЬКО из свежих сигналов (не старше 15 минут)
                        # В hybrid mode НЕТ приоритета стратегии - выбираем просто самый свежий сигнал
                        fresh_available.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                        sig = fresh_available[-1][1]
                        strategy_name = fresh_available[-1][0]
                        ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                        print(f"[live] ✅ Hybrid FRESH: Selected {strategy_name.upper()} signal (no strategy priority, using freshest): {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}]")
                    else:
                        # Если нет свежих сигналов - НЕ выбираем старые сигналы, ждем свежие
                        sig = None
                        print(f"[live] ⏳ Hybrid mode: No fresh signals available. Waiting for fresh signals (max age: 15 minutes)...")
                elif strategy_priority == "confluence":
                    # Режим Конфлюэнции уже обработан выше, не должно сюда попасть
                    sig = None
                else:
                    # Режим приоритета конкретной стратегии
                    # ПРИОРИТЕТ - это защита открытой позиции, а не ограничение на выбор сигналов
                    # Если позиции нет - открываем по любому свежему сигналу
                    # Если позиция есть - приоритет защищает её от противоположных сигналов других стратегий
                    
                    # Проверяем, есть ли открытая позиция
                    has_open_position = position is not None and position.get("size", 0) > 0
                    
                    if not has_open_position:
                        # Позиции нет - открываем по любому свежему сигналу
                        # Но если нет свежих сигналов, предпочитаем сигнал от приоритетной стратегии
                        
                        # ВАЖНО: Фильтруем сигналы по направлению PRIMARY_SYMBOL ДО выбора сигнала
                        # Если на PRIMARY_SYMBOL есть позиция, на других символах можно открывать только в том же направлении
                        primary_symbol_allowed_action = None
                        try:
                            # Проверяем, включена ли функция следования за главным символом
                            follow_primary_symbol = getattr(current_settings, 'follow_primary_symbol', True)  # По умолчанию True
                            
                            if not follow_primary_symbol:
                                _log(f"ℹ️ FOLLOW_PRIMARY_SYMBOL is disabled - skipping PRIMARY_SYMBOL filter for {symbol}", symbol)
                            else:
                                # ВАЖНО: Используем ТОЛЬКО primary_symbol из настроек, БЕЗ fallback на symbol
                                # primary_symbol должен быть установлен глобально в настройках
                                primary_symbol = getattr(current_settings, 'primary_symbol', None)
                                if not primary_symbol:
                                    _log(f"ℹ️ PRIMARY_SYMBOL not set in settings - skipping filter for {symbol}", symbol)
                                else:
                                    _log(f"🔍 Checking PRIMARY_SYMBOL filter for {symbol}: primary_symbol={primary_symbol}", symbol)
                            if follow_primary_symbol and primary_symbol and symbol.upper() != str(primary_symbol).upper():
                                # Проверяем позицию на PRIMARY_SYMBOL
                                _log(f"🔍 Fetching position info for PRIMARY_SYMBOL ({primary_symbol})...", symbol)
                                pos_resp = client.get_position_info(symbol=primary_symbol)
                                if pos_resp.get("retCode") == 0:
                                    pos_list = pos_resp.get("result", {}).get("list", [])
                                    for pos_item in pos_list:
                                        size = float(pos_item.get("size", 0))
                                        if size > 0:
                                            side = pos_item.get("side", "").upper()
                                            primary_bias = Bias.LONG if side == "BUY" else Bias.SHORT
                                            primary_symbol_allowed_action = Action.LONG if primary_bias == Bias.LONG else Action.SHORT
                                            _log(f"🔍 PRIMARY_SYMBOL ({primary_symbol}) has {primary_bias.value} position (size={size}) - filtering signals for {symbol}: only {primary_symbol_allowed_action.value} allowed", symbol)
                                            break
                                    if not primary_symbol_allowed_action:
                                        _log(f"✅ PRIMARY_SYMBOL ({primary_symbol}) has no open position - no filter applied for {symbol}", symbol)
                                else:
                                    _log(f"⚠️ Failed to get position info for PRIMARY_SYMBOL ({primary_symbol}): {pos_resp.get('retMsg', 'Unknown error')}", symbol)
                            elif primary_symbol and symbol.upper() == str(primary_symbol).upper():
                                _log(f"ℹ️ Current symbol ({symbol}) is PRIMARY_SYMBOL - skipping filter", symbol)
                        except Exception as e:
                            _log(f"⚠️ Error checking PRIMARY_SYMBOL position for signal filtering: {e}", symbol)
                            import traceback
                            traceback.print_exc()
                        
                        # Фильтруем сигналы по направлению PRIMARY_SYMBOL
                        if primary_symbol_allowed_action:
                            original_count = len(available_signals)
                            original_fresh_count = len(fresh_available)
                            
                            # Логируем все доступные сигналы ДО фильтрации
                            _log(
                                f"🔍 PRIMARY_SYMBOL filter: Before filtering - {original_count} total signals, "
                                f"{original_fresh_count} fresh signals",
                                symbol,
                            )
                            for name, s in available_signals[:5]:  # Показываем первые 5 для отладки
                                _log(f"   - {name.upper()}: {s.action.value} @ ${s.price:.2f} ({s.reason})", symbol)
                            
                            available_signals = [
                                (name, s) for name, s in available_signals if s.action == primary_symbol_allowed_action
                            ]
                            fresh_available = [
                                (name, s) for name, s in fresh_available if s.action == primary_symbol_allowed_action
                            ]
                            
                            # Логируем все доступные сигналы ПОСЛЕ фильтрации
                            _log(
                                f"🔍 PRIMARY_SYMBOL filter: After filtering - {len(available_signals)} total signals, "
                                f"{len(fresh_available)} fresh signals (allowed: {primary_symbol_allowed_action.value})",
                                symbol,
                            )
                            for name, s in available_signals[:5]:
                                _log(f"   - {name.upper()}: {s.action.value} @ ${s.price:.2f} ({s.reason})", symbol)
                            
                            if available_signals:
                                _log(
                                    f"📊 PRIMARY_SYMBOL filter applied: {len(available_signals)}/{original_count} signals passed "
                                    f"(fresh: {len(fresh_available)}/{original_fresh_count})",
                                    symbol,
                                )
                            else:
                                _log(
                                    f"⚠️ PRIMARY_SYMBOL filter removed all signals - no {primary_symbol_allowed_action.value} "
                                    f"signals available for {symbol}",
                                    symbol,
                                )
                                sig = None
                                if bot_state:
                                    bot_state["current_status"] = "Running"
                                    bot_state["last_action"] = (
                                        f"No {primary_symbol_allowed_action.value} signals (PRIMARY_SYMBOL filter)"
                                    )
                                    bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                                update_worker_status(
                                    symbol,
                                    current_status="Running",
                                    last_action=f"No {primary_symbol_allowed_action.value} signals (PRIMARY_SYMBOL filter)",
                                )
                                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                    break
                                continue
                        else:
                            _log(
                                "ℹ️ PRIMARY_SYMBOL filter: No filter applied (primary_symbol_allowed_action is None)",
                                symbol,
                            )
                        
                        print(
                            f"[live] 🔍 Priority mode (no position): {len(fresh_available)} fresh, "
                            f"{len(available_signals)} total signals available"
                        )
                        is_fallback_signal = False  # Флаг для fallback сигналов
                        if fresh_available:
                            # Если есть свежие сигналы - выбираем самый свежий по timestamp
                            fresh_available.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                            sig = fresh_available[-1][1]
                            strategy_name = fresh_available[-1][0]
                            ts_str = (
                                sig.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                                if hasattr(sig.timestamp, 'strftime')
                                else str(sig.timestamp)
                            )
                            print(
                                f"[live] ✅ Priority mode (no position): Selected {strategy_name.upper()} signal: "
                                f"{sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}]"
                            )
                        elif available_signals:
                            # Если нет свежих сигналов - НЕ выбираем старые сигналы, ждем свежие
                            sig = None
                            print(
                                "[live] ⏳ Priority mode (no position): No fresh signals available. "
                                "Waiting for fresh signals (max age: 15 minutes)..."
                            )
                        else:
                            sig = None
                            print("[live] ⚠️ Priority mode (no position): No signals available")
                    else:
                        # Позиция есть
                        # В hybrid mode при наличии позиции тоже выбираем самый свежий сигнал без приоритета стратегии
                        if strategy_priority == "hybrid":
                            # Hybrid mode: Выбираем самый свежий из всех доступных СВЕЖИХ сигналов
                            # БЕЗ приоритета какой-то определенной стратегии, даже при наличии открытой позиции
                            print(f"[live] 🔍 Hybrid mode (with position): {len(fresh_available)} fresh, {len(available_signals)} total signals available")
                            if fresh_available:
                                # Если есть свежие сигналы - выбираем самый свежий по timestamp
                                # КРИТИЧЕСКИ ВАЖНО: Выбираем ТОЛЬКО из свежих сигналов (не старше 15 минут)
                                # В hybrid mode НЕТ приоритета стратегии - выбираем просто самый свежий сигнал
                                fresh_available.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                                sig = fresh_available[-1][1]
                                strategy_name = fresh_available[-1][0]
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                print(f"[live] ✅ Hybrid FRESH (with position): Selected {strategy_name.upper()} signal (no strategy priority, using freshest): {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}]")
                            else:
                                # Если нет свежих сигналов - НЕ выбираем старые сигналы, ждем свежие
                                sig = None
                                print(f"[live] ⏳ Hybrid mode (with position): No fresh signals available. Waiting for fresh signals (max age: 15 minutes)...")
                        else:
                            # Позиция есть - приоритет защищает её (для режимов с приоритетом конкретной стратегии)
                            # Получаем entry_reason для определения стратегии, которая открыла позицию
                            entry_reason = None
                            try:
                                from bot.web.history import get_open_trade
                                avg_price = position.get("avg_price", 0)
                                if avg_price > 0:
                                    open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
                                    if open_trade:
                                        entry_reason = open_trade.get("entry_reason", "")
                            except Exception as e:
                                print(f"[live] ⚠️ Error getting entry_reason: {e}")
                        
                        # Определяем стратегию, которая открыла позицию
                        position_strategy_type = get_strategy_type_from_signal(entry_reason) if entry_reason else None
                        is_priority_position = position_strategy_type == strategy_priority
                        
                        # Логируем информацию о позиции и приоритете
                        print(f"[live] 🔍 [{symbol}] Position analysis:")
                        print(f"[live]   Position strategy: {position_strategy_type or 'unknown'}")
                        print(f"[live]   Priority strategy: {strategy_priority}")
                        print(f"[live]   Is priority position: {is_priority_position}")
                        print(f"[live]   Position bias: {current_position_bias.value if current_position_bias else 'None'}")
                        print(f"[live]   Available strategy signals: {list(strategy_signals.keys())}")
                        
                        if is_priority_position:
                            # Позиция открыта по приоритетной стратегии - защищаем её
                            # Игнорируем противоположные сигналы от других стратегий
                            # Разрешаем только сигналы в том же направлении (для усиления) или свежий сигнал от приоритетной стратегии (для пересмотра)
                            priority_sig = strategy_signals.get(strategy_priority)
                            
                            print(f"[live]   Priority signal from {strategy_priority}: {'Found' if priority_sig else 'Not found'}")
                            if priority_sig:
                                print(f"[live]     Action: {priority_sig.action.value}, Price: ${priority_sig.price:.2f}, Reason: {priority_sig.reason}")
                                ts_str = priority_sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(priority_sig.timestamp, 'strftime') else str(priority_sig.timestamp)
                                print(f"[live]     Timestamp: {ts_str}")
                            
                            # Проверяем свежесть приоритетного сигнала
                            priority_sig_fresh = False
                            priority_sig_acceptable = False  # Приемлемый для обработки (даже если не свежий)
                            age_from_now_minutes = float('inf')
                            if priority_sig:
                                try:
                                    if isinstance(priority_sig.timestamp, pd.Timestamp):
                                        signal_ts = priority_sig.timestamp
                                        if signal_ts.tzinfo is None:
                                            signal_ts = signal_ts.tz_localize('UTC')
                                        else:
                                            signal_ts = signal_ts.tz_convert('UTC')
                                        current_time_utc = datetime.now(timezone.utc)
                                        age_from_now_minutes = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                                        priority_sig_fresh = age_from_now_minutes <= 15
                                        
                                        # КРИТИЧЕСКИ ВАЖНО: Все сигналы проверяются строго - не старше 15 минут
                                        # Не делаем исключений для противоположных сигналов
                                        priority_sig_acceptable = priority_sig_fresh
                                        is_opposite_direction = (
                                            current_position_bias == Bias.LONG and priority_sig.action == Action.SHORT
                                        ) or (
                                            current_position_bias == Bias.SHORT and priority_sig.action == Action.LONG
                                        )
                                        direction_str = "opposite" if is_opposite_direction else "same"
                                        print(f"[live]     Age: {age_from_now_minutes:.1f} minutes, Fresh: {priority_sig_fresh}, Direction: {direction_str}")
                                except Exception as e:
                                    print(f"[live]     ⚠️ Error checking freshness: {e}")
                            
                            # Если есть свежий сигнал от приоритетной стратегии - используем его (может закрыть/развернуть позицию)
                            # КРИТИЧЕСКИ ВАЖНО: Только свежие сигналы (не старше 15 минут)
                            if priority_sig and priority_sig_fresh:
                                sig = priority_sig
                                age_str = f" (age: {age_from_now_minutes:.1f} min)" if age_from_now_minutes < float('inf') else ""
                                print(f"[live] ✅ Priority position: Fresh {strategy_priority.upper()} signal{age_str} - can review position: {priority_sig.action.value} @ ${priority_sig.price:.2f} ({priority_sig.reason})")
                            else:
                                # Нет свежего сигнала от приоритетной стратегии
                                # Ищем сигналы в том же направлении для усиления позиции
                                same_direction_signals = [(name, s) for name, s in fresh_available 
                                                         if s.action.value == current_position_bias.value]
                                if same_direction_signals:
                                    # Есть сигналы в том же направлении - используем самый свежий для усиления
                                    same_direction_signals.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                                    sig = same_direction_signals[-1][1]
                                    strategy_name = same_direction_signals[-1][0]
                                    print(f"[live] ✅ Priority position: Same direction signal from {strategy_name.upper()} for position enhancement: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                                else:
                                    # Нет сигналов для усиления - не обрабатываем противоположные сигналы
                                    sig = None
                                    print(f"[live] 🛡️ Priority position: Protected from opposite signals. Waiting for same direction or fresh priority signal.")
                        else:
                            # Позиция открыта НЕ по приоритетной стратегии
                            # Сначала проверяем сигналы от той же стратегии, что открыла позицию (SAME STRATEGY REVERSAL)
                            position_strategy_type = None
                            if entry_reason:
                                position_strategy_type = get_strategy_type_from_signal(entry_reason)
                            
                            same_strategy_sig = None
                            same_strategy_sig_fresh = False
                            same_strategy_sig_age = float('inf')
                            if position_strategy_type:
                                same_strategy_sig = strategy_signals.get(position_strategy_type)
                                if same_strategy_sig:
                                    # Проверяем, является ли сигнал противоположным
                                    is_opposite_same_strategy = (
                                        current_position_bias == Bias.LONG and same_strategy_sig.action == Action.SHORT
                                    ) or (
                                        current_position_bias == Bias.SHORT and same_strategy_sig.action == Action.LONG
                                    )
                                    
                                    if is_opposite_same_strategy:
                                        try:
                                            if isinstance(same_strategy_sig.timestamp, pd.Timestamp):
                                                signal_ts = same_strategy_sig.timestamp
                                                if signal_ts.tzinfo is None:
                                                    signal_ts = signal_ts.tz_localize('UTC')
                                                else:
                                                    signal_ts = signal_ts.tz_convert('UTC')
                                                current_time_utc = datetime.now(timezone.utc)
                                                same_strategy_sig_age = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                                                same_strategy_sig_fresh = same_strategy_sig_age <= 15
                                        except Exception as e:
                                            print(f"[live]     ⚠️ Error checking same strategy signal freshness: {e}")
                            
                            # Новый свежий сигнал от приоритетной стратегии может закрыть/развернуть позицию
                            priority_sig = strategy_signals.get(strategy_priority)
                            
                            print(f"[live]   Priority signal from {strategy_priority}: {'Found' if priority_sig else 'Not found'}")
                            if priority_sig:
                                print(f"[live]     Action: {priority_sig.action.value}, Price: ${priority_sig.price:.2f}, Reason: {priority_sig.reason}")
                                ts_str = priority_sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(priority_sig.timestamp, 'strftime') else str(priority_sig.timestamp)
                                print(f"[live]     Timestamp: {ts_str}")
                            
                            # Проверяем свежесть приоритетного сигнала
                            priority_sig_fresh = False
                            priority_sig_acceptable = False  # Приемлемый для обработки (даже если не свежий)
                            age_from_now_minutes = float('inf')
                            if priority_sig:
                                try:
                                    if isinstance(priority_sig.timestamp, pd.Timestamp):
                                        signal_ts = priority_sig.timestamp
                                        if signal_ts.tzinfo is None:
                                            signal_ts = signal_ts.tz_localize('UTC')
                                        else:
                                            signal_ts = signal_ts.tz_convert('UTC')
                                        current_time_utc = datetime.now(timezone.utc)
                                        age_from_now_minutes = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                                        priority_sig_fresh = age_from_now_minutes <= 15
                                        
                                        # КРИТИЧЕСКИ ВАЖНО: Все сигналы проверяются строго - не старше 15 минут
                                        # Не делаем исключений для противоположных сигналов
                                        priority_sig_acceptable = priority_sig_fresh
                                        is_opposite_direction = (
                                            current_position_bias == Bias.LONG and priority_sig.action == Action.SHORT
                                        ) or (
                                            current_position_bias == Bias.SHORT and priority_sig.action == Action.LONG
                                        )
                                        direction_str = "opposite" if is_opposite_direction else "same"
                                        print(f"[live]     Age: {age_from_now_minutes:.1f} minutes, Fresh: {priority_sig_fresh}, Direction: {direction_str}")
                                except Exception as e:
                                    print(f"[live]     ⚠️ Error checking freshness: {e}")
                            
                            # Проверяем, является ли сигнал от приоритетной стратегии противоположным
                            is_opposite_priority = False
                            if priority_sig:
                                is_opposite_priority = (
                                    current_position_bias == Bias.LONG and priority_sig.action == Action.SHORT
                                ) or (
                                    current_position_bias == Bias.SHORT and priority_sig.action == Action.LONG
                                )
                            
                            # КРИТИЧЕСКИ ВАЖНО: Приоритет выбора сигнала:
                            # 1. Противоположный сигнал от той же стратегии, что открыла позицию (SAME STRATEGY REVERSAL) - приоритет #1
                            # 2. Свежие сигналы от приоритетной стратегии
                            # 3. Противоположные сигналы от приоритетной стратегии (не старше 1 часа)
                            # 4. Свежие противоположные сигналы от других стратегий
                            
                            if same_strategy_sig and same_strategy_sig_age <= 60:
                                # Есть противоположный сигнал от той же стратегии, что открыла позицию - используем его (приоритет #1)
                                sig = same_strategy_sig
                                age_str = f" (age: {same_strategy_sig_age:.1f} min)" if same_strategy_sig_age < float('inf') else ""
                                freshness_note = "Fresh" if same_strategy_sig_fresh else "Not fresh but from same strategy"
                                print(f"[live] ✅ Non-priority position: {freshness_note} {position_strategy_type.upper()} signal{age_str} (SAME STRATEGY REVERSAL) - closing and opening new position: {same_strategy_sig.action.value} @ ${same_strategy_sig.price:.2f} ({same_strategy_sig.reason})")
                            elif priority_sig and (priority_sig_fresh or (is_opposite_priority and age_from_now_minutes <= 60)):
                                # Есть сигнал от приоритетной стратегии - используем его (может закрыть/развернуть позицию)
                                sig = priority_sig
                                age_str = f" (age: {age_from_now_minutes:.1f} min)" if age_from_now_minutes < float('inf') else ""
                                freshness_note = "Fresh" if priority_sig_fresh else "Not fresh but opposite from priority strategy"
                                print(f"[live] ✅ Non-priority position: {freshness_note} {strategy_priority.upper()} signal{age_str} - can review/close position: {priority_sig.action.value} @ ${priority_sig.price:.2f} ({priority_sig.reason})")
                            else:
                                # Нет свежего сигнала от приоритетной стратегии (или противоположный сигнал старше 1 часа)
                                # Сначала проверяем противоположные свежие сигналы от других стратегий (для закрытия/разворота)
                                opposite_action = Action.LONG if current_position_bias == Bias.SHORT else Action.SHORT
                                opposite_fresh_signals = [(name, s) for name, s in fresh_available 
                                                          if s.action == opposite_action]
                                if opposite_fresh_signals:
                                    # Есть свежий противоположный сигнал - используем его для закрытия/разворота позиции
                                    opposite_fresh_signals.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                                    sig = opposite_fresh_signals[-1][1]
                                    strategy_name = opposite_fresh_signals[-1][0]
                                    print(f"[live] ✅ Non-priority position: Fresh opposite signal from {strategy_name.upper()} - can close/reverse position: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                                else:
                                    # Нет противоположных свежих сигналов - ищем сигналы в том же направлении для усиления позиции
                                    same_direction_signals = [(name, s) for name, s in fresh_available 
                                                             if s.action.value == current_position_bias.value]
                                    if same_direction_signals:
                                        # Есть сигналы в том же направлении - используем самый свежий для усиления
                                        same_direction_signals.sort(key=lambda x: get_timestamp_for_sort(x[1]))
                                        sig = same_direction_signals[-1][1]
                                        strategy_name = same_direction_signals[-1][0]
                                        # ВАЖНО: Устанавливаем флаг для добавления к позиции, а не открытия новой
                                        should_add_to_position = True
                                        print(f"[live] ✅ Non-priority position: Same direction signal from {strategy_name.upper()} for position enhancement: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) - will ADD to position")
                                    else:
                                        # Нет сигналов для усиления - не обрабатываем противоположные сигналы
                                        sig = None
                                        print(f"[live] ⏸️ Non-priority position: No same direction signals. Waiting for fresh priority signal or same direction signal.")

            # 4. Проверяем подтверждение (agreement) для добавления к позиции
            if sig and sig.action != Action.HOLD:
                # Если другие стратегии также имеют сигнал в том же направлении, разрешаем добавление к позиции
                agreeing_strategies = [name for name, s in available_signals if s and s.action == sig.action and s != sig]
                if agreeing_strategies:
                    should_add_to_position = True
                    print(f"[live] 🤝 Agreement found! {sig.action.value} confirmed by: {', '.join(agreeing_strategies)}")

            # 5. Если сигнал так и не определен (или отклонен логикой выше), пропускаем цикл
            if sig is None or sig.action == Action.HOLD:
                if bot_state:
                    bot_state["current_status"] = "Running"
                    bot_state["last_action"] = "No actionable signal, waiting..."
                    bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                update_worker_status(symbol, current_status="Running", last_action="No actionable signal, waiting...")
                # ВАЖНО: Проверяем stop_event, но если он не установлен, продолжаем цикл
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    _log(f"🛑 Stop event received during signal selection, stopping bot for {symbol}", symbol)
                    break
                # Продолжаем цикл - воркер должен работать постоянно
                _log(f"🔄 Continuing worker loop after no actionable signal, waiting for new signals...", symbol)
                continue
            
            # --- КОНЕЦ ВЫБОРА СИГНАЛА ---

            # 6. Финальная проверка свежести (предотвращаем торговлю на «протухших» данных)
            # КРИТИЧЕСКИ ВАЖНО: Бот открывает позиции ТОЛЬКО по свежим сигналам (не старше 15 минут)
            # Если свежих сигналов нет - бот ждет новых сигналов, НЕ открывает позиции по старым
            ts = sig.timestamp
            is_fresh_check = is_signal_fresh(sig, df_ready)
            strategy_name_for_log = get_strategy_type_from_signal(sig.reason).upper()
            strategy_type = get_strategy_type_from_signal(sig.reason)
            print(f"[live] 🔍 Freshness check for {strategy_name_for_log} signal: is_fresh={is_fresh_check}, timestamp={ts}")
            
            # СТРОГИЙ критерий: ТОЛЬКО сигналы не старше 15 минут от текущего времени
            max_age_minutes = 15  # 15 минут - максимальный возраст сигнала для открытия позиции
            
            # Проверяем возраст сигнала от текущего времени
            if not is_fresh_check:
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                strategy_name = get_strategy_type_from_signal(sig.reason).upper()
                
                # Вычисляем возраст сигнала от текущего времени (не от последней свечи)
                should_filter = False
                try:
                    if isinstance(ts, pd.Timestamp):
                        signal_ts = ts
                        if signal_ts.tzinfo is None:
                            signal_ts = signal_ts.tz_localize('UTC')
                        else:
                            signal_ts = signal_ts.tz_convert('UTC')
                        
                        current_time_utc = datetime.now(timezone.utc)
                        age_from_now_minutes = abs((current_time_utc - signal_ts.to_pydatetime()).total_seconds()) / 60
                        age_from_now_hours = age_from_now_minutes / 60
                        
                        # ВСЕ сигналы: если сигнал в пределах 15 минут от текущего времени - обрабатываем немедленно
                        if age_from_now_minutes <= 15:
                            print(f"[live] ✅ {strategy_name} signal is FRESH (age from now: {age_from_now_minutes:.1f} min) - processing IMMEDIATELY")
                            is_fresh_check = True  # Помечаем как свежий для дальнейшей обработки
                        else:
                            # Сигнал старше 15 минут - ФИЛЬТРУЕМ (не открываем позицию)
                            should_filter = True
                            if age_from_now_hours >= 1:
                                print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} - too old (timestamp: {ts_str}, age: {age_from_now_hours:.1f} hours, max: {max_age_minutes} min)")
                            else:
                                print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} - too old (timestamp: {ts_str}, age: {age_from_now_minutes:.1f} minutes, max: {max_age_minutes} min)")
                            print(f"[live]   ℹ️  Bot will wait for fresh signals (max age: {max_age_minutes} minutes). Market changes quickly, old signals are not reliable.")
                    else:
                        should_filter = True
                        print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} - invalid timestamp: {ts_str}")
                except Exception as e:
                    should_filter = True
                    print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} - error checking age: {e}")
                
                if should_filter:
                    if bot_state:
                        bot_state["current_status"] = "Running"
                        bot_state["last_action"] = "Waiting for fresh signal (max age: 15 min)..."
                        bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                    update_worker_status(
                        symbol,
                        current_status="Running",
                        last_action="Waiting for fresh signal (max age: 15 min)...",
                    )
                    # Используем короткую задержку (5 секунд) вместо полного live_poll_seconds,
                    # чтобы воркер не считался "мертвым" во время ожидания свежего сигнала
                    # и продолжал обновлять статус
                    # ВАЖНО: Проверяем stop_event, но если он не установлен, продолжаем цикл
                    if _wait_with_stop_check(stop_event, 5.0, symbol):
                        _log(f"🛑 Stop event received during freshness check, stopping bot for {symbol}", symbol)
                        break
                    # Продолжаем цикл - воркер должен работать постоянно и ждать свежих сигналов
                    _log(
                        f"🔄 Continuing worker loop after filtering old signal, "
                        f"waiting for fresh signal (max age: {max_age_minutes} min)...",
                        symbol,
                    )
                    continue
            
            # Конвертируем timestamp сигнала в UTC для использования ниже
            signal_time_utc = None
            try:
                if isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None:
                        signal_time_utc = ts.tz_localize('UTC').to_pydatetime()
                    else:
                        signal_time_utc = ts.tz_convert('UTC').to_pydatetime()
                elif hasattr(ts, 'tzinfo'):
                    if ts.tzinfo is None:
                        signal_time_utc = ts.replace(tzinfo=timezone.utc)
                    else:
                        signal_time_utc = ts.astimezone(timezone.utc)
            except Exception:
                pass
            
            # Используем signal_id для уникальной идентификации
            # КРИТИЧЕСКИ ВАЖНО: signal_id должен быть уникальным для каждого сигнала
            # Если signal_id уже есть в сигнале - используем его, иначе генерируем
            signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
            if signal_id is None:
                # Fallback: генерируем ID на основе timestamp, action, reason и price
                # ВАЖНО: Используем точный timestamp и price для уникальности
                import hashlib
                ts_str = str(ts) if hasattr(ts, 'isoformat') else str(ts)
                # Используем больше знаков для price, чтобы избежать коллизий
                price_str = f"{sig.price:.6f}"  # Увеличено с 4 до 6 знаков для большей точности
                id_string = f"{ts_str}_{sig.action.value}_{sig.reason}_{price_str}_{symbol}"  # Добавлен symbol для уникальности
                signal_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
                # Устанавливаем signal_id в сигнал для последующего использования
                if hasattr(sig, 'signal_id'):
                    sig.signal_id = signal_id
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Проверяем, был ли этот сигнал уже обработан
            # Это гарантирует, что один сигнал обрабатывается только один раз
            if signal_id in processed_signals:
                strategy_name = get_strategy_type_from_signal(sig.reason).upper()
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] - already processed (ID: {signal_id})")
                print(f"[live]   ℹ️  This signal was already processed. Waiting for new signal...")
                print(f"[live]   📊 Processed signals count: {len(processed_signals)}")
                if bot_state:
                    bot_state["current_status"] = "Running"
                    bot_state["last_action"] = "Signal already processed, waiting for new signal..."
                    bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                update_worker_status(symbol, current_status="Running", last_action="Signal already processed, waiting for new signal...")
                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                    break
                continue
            
            print(f"[live] ✅ Signal passed processed check (ID: {signal_id}), proceeding to open position...")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Не обрабатываем сигналы старше 15 минут от текущего времени
            # ПРИМЕЧАНИЕ: Эта проверка дублирует логику выше, но оставлена для дополнительной безопасности
            # Если сигнал уже прошел проверку выше (is_fresh_check = True), то эта проверка должна пропустить его
            signal_age_minutes = None
            try:
                # Если сигнал уже прошел проверку свежести выше (is_fresh_check = True), пропускаем эту проверку
                if is_fresh_check:
                    print(f"[live] ✅ Signal already passed freshness check above, skipping duplicate age check")
                else:
                    # Используем signal_time_utc, если он был вычислен выше, иначе вычисляем заново
                    signal_time_for_age = signal_time_utc
                    if not signal_time_for_age:
                        # Fallback: вычисляем signal_time_utc заново
                        if isinstance(ts, pd.Timestamp):
                            signal_ts = ts
                            if signal_ts.tzinfo is None:
                                signal_ts = signal_ts.tz_localize('UTC')
                            else:
                                signal_ts = signal_ts.tz_convert('UTC')
                            signal_time_for_age = signal_ts.to_pydatetime()
                            if signal_time_for_age.tzinfo is None:
                                signal_time_for_age = signal_time_for_age.replace(tzinfo=timezone.utc)
                        elif hasattr(ts, 'tzinfo'):
                            signal_time_for_age = ts
                            if signal_time_for_age.tzinfo is None:
                                signal_time_for_age = signal_time_for_age.replace(tzinfo=timezone.utc)
                            else:
                                signal_time_for_age = signal_time_for_age.astimezone(timezone.utc)
                    
                    if signal_time_for_age:
                        # Получаем текущее время в UTC
                        current_time_utc = datetime.now(timezone.utc)
                        
                        # Вычисляем возраст сигнала в минутах
                        age_delta = current_time_utc - signal_time_for_age
                        signal_age_minutes = age_delta.total_seconds() / 60
                        signal_age_hours = signal_age_minutes / 60
                        
                        # СТРОГАЯ проверка: ТОЛЬКО сигналы в пределах 15 минут
                        should_filter_by_age = False
                        if signal_age_minutes <= max_age_minutes:
                            # Сигнал свежий (в пределах 15 минут)
                            print(f"[live] ✅ Signal age check passed: {signal_age_minutes:.1f} minutes (within {max_age_minutes} min limit)")
                        else:
                            # Сигнал старше 15 минут - ФИЛЬТРУЕМ (не открываем позицию)
                            should_filter_by_age = True
                            strategy_name = get_strategy_type_from_signal(sig.reason).upper()
                            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                            if signal_age_hours >= 1:
                                print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] - too old ({signal_age_hours:.1f} hours > {max_age_minutes} min limit)")
                            else:
                                print(f"[live] ⚠️ FILTERED: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] - too old ({signal_age_minutes:.1f} minutes > {max_age_minutes} minutes limit)")
                            print(f"[live]   ℹ️  Signal age: {signal_age_minutes:.1f} minutes. Maximum allowed: {max_age_minutes} minutes. Bot will wait for fresh signals.")
                        
                        if should_filter_by_age:
                            if bot_state:
                                bot_state["current_status"] = "Running"
                                bot_state["last_action"] = f"Signal too old ({signal_age_minutes:.1f} min), waiting for fresh signal (max: {max_age_minutes} min)..."
                                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                            update_worker_status(symbol, current_status="Running", last_action=f"Signal too old ({signal_age_minutes:.1f} min), waiting for fresh signal (max: {max_age_minutes} min)...")
                            # ВАЖНО: Проверяем stop_event, но если он не установлен, продолжаем цикл
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                _log(f"🛑 Stop event received during age check, stopping bot for {symbol}", symbol)
                                break
                            # Продолжаем цикл - воркер должен работать постоянно и ждать свежих сигналов
                            _log(f"🔄 Continuing worker loop after filtering old signal, waiting for fresh signal (max age: {max_age_minutes} min)...", symbol)
                            continue
            except Exception as e:
                # В случае ошибки при проверке возраста - логируем, но продолжаем обработку
                print(f"[live] ⚠️ Error checking signal age: {e}, proceeding with signal processing")
                import traceback
                traceback.print_exc()
            
            # Логируем выбранный сигнал для обработки
            strategy_name = get_strategy_type_from_signal(sig.reason).upper()
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            age_info = f" (age: {signal_age_minutes:.1f} min)" if signal_age_minutes is not None else ""
            print(f"[live] ✅ SELECTED for processing: {strategy_name} signal {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] (ID: {signal_id}){age_info}")
            # Проверяем, что signal_age_minutes не None перед форматированием
            if signal_age_minutes is not None:
                print(f"[live]   ℹ️  This is a NEW signal that has NOT been processed yet. Age: {signal_age_minutes:.1f} minutes (within {max_age_minutes} min limit). Proceeding with execution...")
            else:
                print(f"[live]   ℹ️  This is a NEW signal that has NOT been processed yet. Proceeding with execution...")
            
            # Ограничиваем размер processed_signals для оптимизации памяти
            # ВАЖНО: Не удаляем слишком много, чтобы не потерять историю обработанных сигналов
            if len(processed_signals) > 2000:  # Увеличено с 1000 до 2000 для большей истории
                # Удаляем самые старые 1000 записей (половину)
                processed_signals_list = list(processed_signals)
                processed_signals = set(processed_signals_list[1000:])
                print(f"[live]   ℹ️  Cleaned processed_signals: kept {len(processed_signals)} most recent signals")
            
            # Обновляем статус: найден сигнал
            from bot.multi_symbol_manager import update_worker_status
            if bot_state:
                bot_state["current_status"] = "Signal Found"
                bot_state["last_signal"] = f"{sig.action.value}: {sig.reason}"
                if signal_time_utc:
                    bot_state["last_signal_time"] = signal_time_utc.isoformat() if hasattr(signal_time_utc, 'isoformat') else str(signal_time_utc)
                elif hasattr(ts, 'isoformat'):
                    if isinstance(ts, pd.Timestamp):
                        if ts.tzinfo is None:
                            ts_utc = ts.tz_localize('UTC')
                        else:
                            ts_utc = ts.tz_convert('UTC')
                        bot_state["last_signal_time"] = ts_utc.isoformat()
                    else:
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                            ts_utc = ts.replace(tzinfo=timezone.utc)
                        elif hasattr(ts, 'astimezone'):
                            ts_utc = ts.astimezone(timezone.utc)
                        else:
                            ts_utc = ts
                        bot_state["last_signal_time"] = ts_utc.isoformat()
                else:
                    bot_state["last_signal_time"] = str(ts)
                bot_state["last_action"] = f"Signal: {sig.action.value} ({sig.reason})"
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            update_worker_status(symbol, current_status="Signal Found", last_signal=f"{sig.action.value}: {sig.reason}")
            
            # Обработка позиции (если есть)
            if position:
                # Обновляем отслеживание максимальной прибыли
                _update_position_tracking(position, current_position_bias, current_price, position_max_profit, position_max_price, symbol)
                
                # Проверяем частичное закрытие
                _check_partial_close(
                    client=client,
                    position=position,
                    position_bias=current_position_bias,
                    current_price=current_price,
                    settings=current_settings,
                    position_max_profit=position_max_profit,
                    position_partial_closed=position_partial_closed,
                )
                
                # Проверяем защиту прибыли
                profit_protection_reason = _check_profit_protection(
                    client=client,
                    position=position,
                    position_bias=current_position_bias,
                    current_price=current_price,
                    settings=current_settings,
                    position_max_profit=position_max_profit,
                    position_max_price=position_max_price,
                )
                
                if profit_protection_reason:
                    # Закрываем позицию
                    side = "Sell" if current_position_bias == Bias.LONG else "Buy"
                    resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=position["size"],
                        reduce_only=True,
                    )
                    
                    if resp.get("retCode") == 0:
                        print("=" * 80)
                        print(f"[live] [{symbol}] ⚫⚫⚫ POSITION CLOSED: PROFIT PROTECTION ⚫⚫⚫")
                        print(f"[live] [{symbol}]   Reason: {profit_protection_reason}")
                        print(f"[live] [{symbol}]   Side: {current_position_bias.value}")
                        print(f"[live] [{symbol}]   Entry Price: ${position.get('avg_price', current_price):.2f}")
                        print(f"[live] [{symbol}]   Exit Price: ${current_price:.2f}")
                        print(f"[live] [{symbol}]   PnL: ${position.get('unrealised_pnl', 0):.2f}")
                        print("=" * 80)
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        # Используем сохраненную стратегию, которая открыла позицию
                        strategy_type = position_strategy.pop(symbol, "unknown")
                        _clear_bot_state(symbol)
                        # Если стратегия unknown, пытаемся определить с fallback логикой
                        if strategy_type == "unknown":
                            strategy_type = _determine_strategy_with_fallback(
                                symbol,
                                position_strategy,
                                position,
                                entry_time=datetime.now(timezone.utc),  # Используем текущее время как приближение
                            )
                        try:
                            # Получаем orderId из ответа place_order (ID ордера закрытия)
                            close_order_id = None
                            result = resp.get("result", {})
                            if result:
                                close_order_id = result.get("orderId") or None
                            
                            # Получаем orderLinkId из сохраненных данных (ID открытия позиции)
                            order_link_id = position_order_link_id.pop(symbol, None)
                            
                            # Если orderId не получен из ответа, пытаемся получить из истории исполненных ордеров
                            if not close_order_id:
                                try:
                                    # Получаем историю исполненных ордеров за последние 5 минут
                                    exec_start_time = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)
                                    exec_end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                                    exec_resp = client.get_execution_list(
                                        symbol=symbol,
                                        start_time=exec_start_time,
                                        end_time=exec_end_time,
                                        limit=50,
                                    )
                                    if exec_resp.get("retCode") == 0:
                                        exec_result = exec_resp.get("result", {})
                                        exec_list = exec_result.get("list", [])
                                        # Ищем последний исполненный ордер для закрытия позиции (reduceOnly)
                                        for exec_order in exec_list:
                                            if exec_order.get("reduceOnly") and exec_order.get("execQty"):
                                                close_order_id = exec_order.get("orderId")
                                                if close_order_id:
                                                    break
                                except Exception as e:
                                    print(f"[live] ⚠️ Error getting orderId from execution history: {e}")
                            
                            add_trade(
                                entry_time=datetime.now(),
                                exit_time=datetime.now(),
                                side=current_position_bias.value,
                                entry_price=position.get("avg_price", current_price),
                                exit_price=current_price,
                                size_usd=position["size"] * current_price,
                                pnl=position.get("unrealised_pnl", 0),
                                entry_reason="unknown",
                                exit_reason=profit_protection_reason,
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=close_order_id,
                                order_link_id=order_link_id,
                            )
                        except Exception as e:
                            print(f"[live] Warning: Failed to log trade: {e}")
                    else:
                        print(f"[live] ⚠️ Failed to close position: {resp.get('retMsg', 'Unknown error')}")
                
                # Проверяем соответствие позиции текущим сигналам стратегий
                close_reason = _check_position_strategy_alignment(
                    client=client,
                    position=position,
                    position_bias=current_position_bias,
                    all_signals=all_signals,
                    current_price=current_price,
                    settings=current_settings,
                    df_ready=df_ready,
                )
                
                if close_reason:
                    # Логируем причину закрытия позиции
                    print(f"[live] [{symbol}] 🚨 Closing {current_position_bias.value} position due to: {close_reason}")
                    print(f"[live] [{symbol}] 📊 Current signals: {len(all_signals)} total signals")
                    for sig in all_signals:
                        print(f"[live] [{symbol}]   - {sig.action.value}: {sig.reason} @ ${sig.price:.2f}")
                    
                    # Закрываем позицию
                    side = "Sell" if current_position_bias == Bias.LONG else "Buy"
                    resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=position["size"],
                        reduce_only=True,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] [{symbol}] ✅ Position closed (strategy alignment): {close_reason}")
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        # Используем сохраненную стратегию, которая открыла позицию
                        strategy_type = position_strategy.pop(symbol, "unknown")
                        _clear_bot_state(symbol)
                        # Если стратегия unknown, пытаемся определить с fallback логикой
                        if strategy_type == "unknown":
                            strategy_type = _determine_strategy_with_fallback(
                                symbol,
                                position_strategy,
                                position,
                                entry_time=datetime.now(timezone.utc),  # Используем текущее время как приближение
                            )
                        try:
                            # Получаем orderId из ответа place_order (ID ордера закрытия)
                            close_order_id = None
                            result = resp.get("result", {})
                            if result:
                                close_order_id = result.get("orderId") or None
                            
                            # Получаем orderLinkId из сохраненных данных (ID открытия позиции)
                            order_link_id = position_order_link_id.pop(symbol, None)
                            
                            # Если orderId не получен из ответа, пытаемся получить из истории исполненных ордеров
                            if not close_order_id:
                                try:
                                    # Получаем историю исполненных ордеров за последние 5 минут
                                    exec_start_time = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)
                                    exec_end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                                    exec_resp = client.get_execution_list(
                                        symbol=symbol,
                                        start_time=exec_start_time,
                                        end_time=exec_end_time,
                                        limit=50,
                                    )
                                    if exec_resp.get("retCode") == 0:
                                        exec_result = exec_resp.get("result", {})
                                        exec_list = exec_result.get("list", [])
                                        # Ищем последний исполненный ордер для закрытия позиции (reduceOnly)
                                        for exec_order in exec_list:
                                            if exec_order.get("reduceOnly") and exec_order.get("execQty"):
                                                close_order_id = exec_order.get("orderId")
                                                if close_order_id:
                                                    break
                                except Exception as e:
                                    print(f"[live] ⚠️ Error getting orderId from execution history: {e}")
                            
                            add_trade(
                                entry_time=datetime.now(),
                                exit_time=datetime.now(),
                                side=current_position_bias.value,
                                entry_price=position.get("avg_price", current_price),
                                exit_price=current_price,
                                size_usd=position["size"] * current_price,
                                pnl=position.get("unrealised_pnl", 0),
                                entry_reason="unknown",
                                exit_reason=close_reason,
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=close_order_id,
                                order_link_id=order_link_id,
                            )
                        except Exception as e:
                            print(f"[live] Warning: Failed to log trade: {e}")
                    else:
                        print(f"[live] ⚠️ Failed to close position: {resp.get('retMsg', 'Unknown error')}")
                
                # Обеспечиваем установку TP/SL
                _ensure_tp_sl_set(
                    client=client,
                    position=position,
                    settings=current_settings,
                    position_bias=current_position_bias,
                    current_price=current_price,
                    position_max_profit=position_max_profit,
                    position_max_price=position_max_price,
                )
            
            # Обработка сигналов: автоматическое определение действий на основе сигнала и текущей позиции
            # LONG сигнал
            if sig.action == Action.LONG:
                print(f"[live] 🔍 Processing LONG signal: position exists={position is not None}, position_bias={current_position_bias if position else 'None'}")
                
                # ВАЖНО: Если позиция уже LONG и сигнал LONG - всегда добавляем к позиции, а не открываем новую
                if position and current_position_bias == Bias.LONG:
                    should_add_to_position = True
                    print(f"[live] ✅ Position already LONG - will ADD to position instead of opening new one")
                
                # Проверяем приоритет стратегии перед закрытием позиции
                signal_strategy_type = get_strategy_type_from_signal(sig.reason)
                can_close_position = True
                
                if position and current_position_bias == Bias.SHORT:
                    # Есть SHORT позиция и приходит LONG сигнал
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Если сигнал от той же стратегии, что открыла позицию - закрываем позицию
                    entry_reason = None
                    try:
                        from bot.web.history import get_open_trade
                        avg_price = position.get("avg_price", 0)
                        if avg_price > 0:
                            open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
                            if open_trade:
                                entry_reason = open_trade.get("entry_reason", "")
                    except Exception as e:
                        print(f"[live] ⚠️ Error getting entry_reason: {e}")
                    
                    position_strategy_type = get_strategy_type_from_signal(entry_reason) if entry_reason else None
                    
                    # Если позиция открыта по той же стратегии, что и сигнал - проверяем свежесть сигнала
                    if position_strategy_type and position_strategy_type == signal_strategy_type:
                        # Проверяем, является ли сигнал свежим
                        is_fresh = is_signal_fresh(sig, df_ready)
                        if is_fresh:
                            print(f"[live] 🔄 SAME STRATEGY REVERSAL (FRESH): SHORT position opened by {position_strategy_type.upper()}, fresh opposite LONG signal from same strategy - closing and opening new position")
                            can_close_position = True  # Принудительно разрешаем закрытие и открытие новой позиции
                        else:
                            print(f"[live] ⚠️ SAME STRATEGY REVERSAL (NOT FRESH): SHORT position opened by {position_strategy_type.upper()}, but opposite LONG signal is not fresh - closing position only")
                            can_close_position = True  # Закрываем позицию, но не открываем новую (сигнал не свежий)
                    else:
                        # Проверяем приоритет стратегии только если сигнал от другой стратегии
                        is_priority_position = position_strategy_type == strategy_priority
                        
                        if is_priority_position and signal_strategy_type != strategy_priority:
                            # Позиция открыта по приоритетной стратегии, а сигнал от другой стратегии - защищаем позицию
                            can_close_position = False
                            print(f"[live] 🛡️ PRIORITY PROTECTION: SHORT position opened by {strategy_priority.upper()} strategy, ignoring opposite LONG signal from {signal_strategy_type.upper()}")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: Если есть SHORT позиция и приходит LONG сигнал - закрываем SHORT и открываем LONG
                if position and current_position_bias == Bias.SHORT and can_close_position:
                    strategy_type = get_strategy_type_from_signal(sig.reason)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    _log(f"🔄 REVERSAL: Closing SHORT position to open LONG (signal: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f})", symbol)
                    
                    # Закрываем SHORT позицию
                    close_qty = position.get("size", 0)
                    if close_qty > 0:
                        try:
                            resp = client.place_order(
                                symbol=symbol,
                                side="Buy",  # Buy закрывает SHORT
                                qty=close_qty,
                                reduce_only=True,
                            )
                            if resp.get("retCode") == 0:
                                print(f"[live] [{symbol}] ✅ SHORT position closed for reversal to LONG")
                                # Ждем немного, чтобы позиция закрылась
                                import time as time_module
                                time_module.sleep(1.0)
                                # Перезагружаем информацию о позиции из API
                                try:
                                    pos_resp = client.get_position_info(symbol=symbol)
                                    if pos_resp.get("retCode") == 0:
                                        pos_list = pos_resp.get("result", {}).get("list", [])
                                        position = None
                                        current_position_bias = None
                                        for pos_item in pos_list:
                                            if float(pos_item.get("size", 0)) > 0:
                                                position = pos_item
                                                current_position_bias = Bias.LONG if pos_item.get("side") == "Buy" else Bias.SHORT
                                                break
                                        if position is None:
                                            print(f"[live] [{symbol}] ✅ Position confirmed closed, proceeding to open LONG")
                                            # Продолжаем выполнение, чтобы открыть новую позицию
                                            # Не делаем break или continue - код продолжит выполнение и откроет LONG позицию
                                        else:
                                            print(f"[live] [{symbol}] ⚠️ Position still exists after close attempt, skipping LONG open")
                                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                                break
                                            continue
                                except Exception as e:
                                    print(f"[live] [{symbol}] ⚠️ Error reloading position info: {e}, assuming closed")
                                    position = None
                                    current_position_bias = None
                            else:
                                print(f"[live] [{symbol}] ⚠️ Failed to close SHORT position: {resp.get('retMsg', 'Unknown error')}")
                                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                    break
                                continue
                        except Exception as e:
                            print(f"[live] [{symbol}] ⚠️ Error closing SHORT position: {e}")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                
                if not position:
                    # Позиции нет → открываем LONG
                    # Если сигналы подтверждают друг друга, это уже учтено в выборе сигнала
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Перепроверяем наличие позиции перед открытием новой
                    # Это предотвращает открытие нескольких позиций по одной паре
                    position_check = _get_position(client, symbol)
                    if position_check and position_check.get("size", 0) > 0:
                        _log(f"⚠️ Position already exists for {symbol} (size: {position_check.get('size', 0)}), skipping new position open", symbol)
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Не открываем LONG, если на PRIMARY_SYMBOL есть SHORT позиция
                    _log(f"🔍 [FINAL CHECK] Checking PRIMARY_SYMBOL position before opening LONG for {symbol}...", symbol)
                    _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) from {strategy_name}", symbol)
                    # ВАЖНО: Используем ТОЛЬКО primary_symbol из настроек, БЕЗ fallback на symbol
                    primary_symbol_from_settings = getattr(current_settings, 'primary_symbol', None)
                    _log(f"   PRIMARY_SYMBOL from settings: {primary_symbol_from_settings}", symbol)
                    _log(f"   Current symbol: {symbol}", symbol)
                    
                    # Проверяем, включена ли функция следования за главным символом
                    follow_primary_symbol = getattr(current_settings, 'follow_primary_symbol', True)  # По умолчанию True
                    should_block = False
                    block_reason = None
                    if follow_primary_symbol:
                        should_block, block_reason = _check_primary_symbol_position(
                            client=client,
                            current_symbol=symbol,
                            settings=current_settings,
                            target_action=Action.LONG,
                        )
                    else:
                        _log(f"ℹ️ FOLLOW_PRIMARY_SYMBOL is disabled - skipping PRIMARY_SYMBOL check for {symbol}", symbol)
                    
                    _log(f"   [FINAL CHECK RESULT] PRIMARY_SYMBOL check result: should_block={should_block}, reason={block_reason}", symbol)
                    if should_block:
                        _log(f"⛔ [FINAL CHECK] BLOCKED: {block_reason}", symbol)
                        _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) - waiting for PRIMARY_SYMBOL position to close or reverse", symbol)
                        if bot_state:
                            bot_state["current_status"] = "Running"
                            bot_state["last_action"] = f"Blocked: {block_reason}"
                            bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                        update_worker_status(symbol, current_status="Running", last_action=f"Blocked: {block_reason}")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    else:
                        _log(f"✅ [FINAL CHECK] PRIMARY_SYMBOL check passed - LONG position allowed for {symbol}", symbol)
                    
                    strategy_type = get_strategy_type_from_signal(sig.reason)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    _log(f"📈 Opening NEW LONG position", symbol)
                    _log(f"   Signal: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] (ID: {signal_id})", symbol)
                    
                    # Проверяем историю убыточных сделок перед открытием
                    if current_settings.risk.enable_loss_cooldown:
                        should_block, last_loss = check_recent_loss_trade(
                            side="long",
                            symbol=symbol,
                            cooldown_minutes=current_settings.risk.loss_cooldown_minutes,
                            max_losses=current_settings.risk.max_consecutive_losses,
                        )
                        if should_block:
                            if last_loss:
                                exit_reason = last_loss.get("exit_reason", "unknown")
                                pnl = last_loss.get("pnl", 0)
                                print(f"[live] ⛔ Blocking LONG: recent loss trade detected (PnL: {pnl:.2f} USDT, reason: {exit_reason})")
                            else:
                                print(f"[live] ⛔ Blocking LONG: too many consecutive losses")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                    
                    # Проверяем ATR перед открытием позиции (используем ATR с 1H и 4H таймфреймов)
                    if current_settings.risk.enable_atr_entry_filter and not df_ready.empty:
                        try:
                            last_row = df_ready.iloc[-1]
                            # Используем среднее значение ATR с 1H и 4H таймфреймов для среднесрочного анализа
                            atr_value = last_row.get("atr_avg", None)  # Среднее ATR с 1H и 4H
                            # Fallback на 15M ATR если нет данных с высших таймфреймов
                            if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
                                atr_value = last_row.get("atr", None)
                            current_price = sig.price
                            
                            if atr_value is not None and pd.notna(atr_value) and atr_value > 0:
                                # Получаем предыдущие свечи для анализа движения цены
                                if len(df_ready) >= 2:
                                    prev_row = df_ready.iloc[-2]
                                    prev_close = prev_row.get("close", current_price)
                                    
                                    # Рассчитываем, какую часть ATR (среднесрочного) цена уже прошла в направлении сигнала
                                    if pd.notna(prev_close):
                                        price_move = current_price - prev_close
                                        atr_progress = abs(price_move) / atr_value if atr_value > 0 else 0
                                        
                                        # Для LONG сигнала: если цена уже прошла вверх большую часть среднесрочного ATR - не входить
                                        if price_move > 0 and atr_progress > current_settings.risk.max_atr_progress_pct:
                                            atr_1h = last_row.get("atr_1h", 0)
                                            atr_4h = last_row.get("atr_4h", 0)
                                            print(f"[live] ⛔ Blocking LONG: price already moved {atr_progress*100:.1f}% of avg ATR(1H+4H) up (threshold: {current_settings.risk.max_atr_progress_pct*100:.1f}%)")
                                            print(f"[live]   Current: ${current_price:.2f}, Previous: ${prev_close:.2f}, ATR avg(1H+4H): ${atr_value:.2f} (1H: ${atr_1h:.2f}, 4H: ${atr_4h:.2f}), Move: ${price_move:.2f}")
                                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                                break
                                            continue
                        except Exception as e:
                            print(f"[live] ⚠️ Error checking ATR filter: {e}")
                    
                    balance = _get_balance(client)
                    if balance is None:
                        print(f"[live] ⚠️ Skipping LONG: failed to get balance")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                    qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping LONG: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    # Детальное логирование решения
                    if sig.indicators_info:
                        info = sig.indicators_info
                        strategy_name = info.get("strategy", "UNKNOWN")
                        indicators_str = info.get("indicators", "N/A")
                        entry_type = info.get("entry_type", "")
                        print(f"[live] 📈 Opening LONG position: {qty:.3f} @ ${sig.price:.2f} (${desired_usd:.2f}) [Signal ID: {signal_id}]")
                        print(f"[live] 📊 Decision path: Strategy={strategy_name}, Entry={entry_type}, Indicators: {indicators_str}")
                        if strategy_name == "TREND":
                            print(f"[live]   ADX={info.get('adx', 'N/A')}, +DI={info.get('plus_di', 'N/A')}, -DI={info.get('minus_di', 'N/A')}, Bias={info.get('bias', 'N/A')}")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                        elif strategy_name == "FLAT":
                            print(f"[live]   RSI={info.get('rsi', 'N/A')}, BB_lower={info.get('bb_lower', 'N/A')}, BB_middle={info.get('bb_middle', 'N/A')}, BB_upper={info.get('bb_upper', 'N/A')}")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                        elif strategy_name == "ML":
                            print(f"[live]   ML Confidence={info.get('confidence_pct', 'N/A')}% ({info.get('strength', 'N/A')}), TP={info.get('tp_pct', 'N/A')}%, SL={info.get('sl_pct', 'N/A')}%")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                    else:
                        print(f"[live] 📈 Opening LONG position: {qty:.3f} @ ${sig.price:.2f} (${desired_usd:.2f}) [Signal ID: {signal_id}]")
                    
                    # Формируем уникальный order_link_id с timestamp для избежания дубликатов
                    timestamp_ms = int(time.time() * 1000)
                    unique_order_link_id = f"sig_{signal_id}_{timestamp_ms}"
                    
                    # Рассчитываем TP и SL для новой позиции
                    take_profit, stop_loss = _calculate_tp_sl_for_signal(sig, current_settings, sig.price, df_ready)
                    if take_profit and stop_loss:
                        print(f"[live]   TP: ${take_profit:.2f} (+{((take_profit - sig.price) / sig.price * 100):.2f}%), SL: ${stop_loss:.2f} ({((stop_loss - sig.price) / sig.price * 100):.2f}%)")
                    
                    # Размещаем ордер БЕЗ TP/SL (для Market ордеров Bybit не поддерживает установку TP/SL при размещении)
                    try:
                        resp = client.place_order(
                            symbol=symbol,
                            side="Buy",
                            qty=qty,
                            order_link_id=unique_order_link_id,
                        )
                    except InvalidRequestError as e:
                        # Обрабатываем ошибки API (например, недостаточный баланс)
                        error_msg = str(e)
                        error_code = None
                        if "ErrCode" in error_msg:
                            import re
                            code_match = re.search(r'ErrCode:\s*(\d+)', error_msg)
                            if code_match:
                                error_code = int(code_match.group(1))
                        
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        if error_code == 110007:
                            _log(f"❌ INSUFFICIENT BALANCE: Cannot open LONG position - {error_msg}", symbol)
                            _log(f"   Signal was generated but cannot be executed. Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        else:
                            _log(f"❌ ORDER ERROR: Failed to open LONG position - {error_msg}", symbol)
                            _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        
                        # Сохраняем сигнал в историю даже при ошибке выполнения
                        try:
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            _log(f"💾 Signal saved to history despite order error", symbol)
                        except Exception as save_error:
                            _log(f"⚠️ Failed to save signal to history: {save_error}", symbol)
                        
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    if resp.get("retCode") == 0:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print("=" * 80)
                        print(f"[live] 🟢🟢🟢 POSITION OPENED: LONG 🟢🟢🟢")
                        print(f"[live]   Strategy: {strategy_type.upper()}")
                        print(f"[live]   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                        print(f"[live]   Quantity: {qty:.3f} (${desired_usd:.2f})")
                        print(f"[live]   Order Link ID: {unique_order_link_id}")
                        print("=" * 80)
                        # Запоминаем время открытия позиции для защиты от ложных закрытий
                        position_opened_time = datetime.now(timezone.utc)
                        
                        # Отмечаем, что обработан свежий сигнал (для оптимизации интервала ожидания)
                        if is_fresh_check:
                            fresh_signal_processed = True
                            _log(f"✅ Fresh signal processed - will check for new signals immediately", symbol)
                        
                        # Устанавливаем TP/SL сразу после успешного открытия позиции
                        if take_profit and stop_loss:
                            try:
                                # Небольшая задержка, чтобы позиция точно открылась
                                import time as time_module
                                time_module.sleep(0.5)
                                
                                tp_sl_resp = client.set_trading_stop(
                                    symbol=symbol,
                                    take_profit=take_profit,
                                    stop_loss=stop_loss,
                                )
                                if tp_sl_resp.get("retCode") == 0:
                                    # Правильно форматируем проценты в зависимости от направления позиции
                                    if sig.action == Action.LONG:
                                        tp_pct_str = f"+{((take_profit - sig.price) / sig.price * 100):.2f}%"
                                        sl_pct_str = f"{((stop_loss - sig.price) / sig.price * 100):.2f}%"
                                    else:  # SHORT
                                        tp_pct_str = f"{((take_profit - sig.price) / sig.price * 100):.2f}%"  # Отрицательный процент (цена ниже входа)
                                        sl_pct_str = f"+{((stop_loss - sig.price) / sig.price * 100):.2f}%"  # Положительный процент (цена выше входа)
                                    print(f"[live] ✅ TP/SL set successfully: TP=${take_profit:.2f} ({tp_pct_str}), SL=${stop_loss:.2f} ({sl_pct_str})")
                                else:
                                    print(f"[live] ⚠️ Failed to set TP/SL: {tp_sl_resp.get('retMsg', 'Unknown error')} (will retry via _ensure_tp_sl_set)")
                            except Exception as e:
                                print(f"[live] ⚠️ Error setting TP/SL immediately: {e} (will retry via _ensure_tp_sl_set)")
                        
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                        
                        # Сохраняем состояние позиции
                        result = resp.get("result", {})
                        order_id = result.get("orderId", "") if result else ""
                        order_link_id_result = result.get("orderLinkId", unique_order_link_id) if result else unique_order_link_id
                        
                        _update_and_save_position_state(
                            symbol=symbol,
                            position_strategy=position_strategy,
                            position_order_id=position_order_id,
                            position_order_link_id=position_order_link_id,
                            position_add_count=position_add_count,
                            position_entry_price=position_entry_price,
                            strategy_type=strategy_type,
                            order_id=order_id,
                            order_link_id=order_link_id_result,
                            add_count=0,
                            entry_price=sig.price
                        )
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)

                        # КРИТИЧЕСКИ ВАЖНО: Сохраняем LONG позицию в историю
                        try:
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            elif isinstance(ts_log, datetime):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.replace(tzinfo=timezone.utc)
                            else:
                                ts_log = datetime.now(timezone.utc)
                            
                            # ВАЛИДАЦИЯ: Убеждаемся, что side соответствует sig.action
                            expected_side = "long" if sig.action == Action.LONG else "short"
                            if expected_side != "long":
                                _log(f"⚠️ WARNING: sig.action={sig.action.value} but trying to save LONG position! Using expected_side={expected_side}", symbol)
                            
                            add_trade(
                                entry_time=ts_log,
                                exit_time=None,  # Позиция еще открыта
                                side=expected_side,  # ВАЖНО: Используем валидированный side
                                entry_price=sig.price,
                                exit_price=0.0,
                                size_usd=desired_usd,
                                pnl=0.0,
                                entry_reason=sig.reason,
                                exit_reason="",
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=order_id,
                                order_link_id=order_link_id_result,
                            )
                            _log(f"💾 Saved {expected_side.upper()} position to history: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        except Exception as e:
                            _log(f"⚠️ Error saving LONG position to history: {e}", symbol)

                        # ОТКЛЮЧЕНО: Автоматическое закрытие позиций при открытии на PRIMARY_SYMBOL
                        # Эта логика вызывала каскадное закрытие позиций, когда все сигналы в одном направлении
                        # Если нужно закрыть противонаправленные позиции, это должно делаться вручную или через другую логику
                        # primary_symbol_for_check = getattr(current_settings, "primary_symbol", None) or getattr(current_settings, "symbol", None)
                        # if primary_symbol_for_check and symbol.upper() == str(primary_symbol_for_check).upper():
                        #     # Перепроверяем, что позиция действительно открыта на PRIMARY_SYMBOL
                        #     try:
                        #         position_verify = _get_position(client, symbol)
                        #         if position_verify and position_verify.get("size", 0) > 0:
                        #             _log(f"✅ Position confirmed open on PRIMARY_SYMBOL ({symbol}) - closing opposite positions on other symbols", symbol)
                        #             try:
                        #                 _close_conflicting_positions_for_primary(
                        #                     client=client,
                        #                     settings=current_settings,
                        #                     new_primary_bias=Bias.LONG,
                        #                 )
                        #             except Exception as e:
                        #                 print(f"[live] [{symbol}] ⚠️ Error while closing opposite positions for PRIMARY_SYMBOL LONG: {e}")
                        #         else:
                        #             _log(f"⚠️ Position not confirmed on PRIMARY_SYMBOL ({symbol}) - skipping close of opposite positions", symbol)
                        #     except Exception as e:
                        #         _log(f"⚠️ Error verifying position on PRIMARY_SYMBOL before closing opposite positions: {e}", symbol)
                    elif resp.get("retCode") == 110072:
                        # Ошибка дубликата order_link_id - сигнал уже обработан
                        print(f"[live] [{symbol}] ⚠️ OrderLinkID duplicate - signal already processed: {signal_id}")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                    else:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] [{symbol}] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to open LONG position: {resp.get('retMsg', 'Unknown error')} (ErrCode: {resp.get('retCode')})")
                elif current_position_bias == Bias.LONG:
                    # Позиция LONG и сигнал LONG → добавляем к позиции (ADD_LONG)
                    # Это может быть подтверждение от другой стратегии или повторный сигнал от той же
                    if should_add_to_position:
                        print(f"[live] 📊 Adding to position: signals from different strategies confirm each other")
                    
                    # Проверяем smart add условия
                    if current_settings.risk.enable_smart_add:
                        # 1. Проверяем лимит докупок
                        current_add_count = position_add_count.get(symbol, 0)
                        max_adds = current_settings.risk.max_add_count
                        if current_add_count >= max_adds:
                            print(f"[live] ⚠️ Skipping ADD_LONG: max adds reached ({current_add_count}/{max_adds})")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        # 2. Проверяем прогресс к TP или SL (>50% пути)
                        avg_price = position.get("avg_price", sig.price)
                        current_tp = position.get("take_profit", "")
                        current_sl = position.get("stop_loss", "")
                        
                        can_add = False
                        add_reason = ""
                        
                        if current_tp and current_sl and avg_price > 0:
                            try:
                                tp_price = float(current_tp) if current_tp else 0
                                sl_price = float(current_sl) if current_sl else 0
                                
                                if tp_price > 0 and sl_price > 0:
                                    # Расчёт прогресса к TP (для LONG: цена растёт к TP)
                                    distance_to_tp = tp_price - avg_price
                                    progress_to_tp = (sig.price - avg_price) / distance_to_tp if distance_to_tp > 0 else 0
                                    
                                    # Расчёт прогресса к SL (для LONG: цена падает к SL)
                                    distance_to_sl = avg_price - sl_price
                                    progress_to_sl = (avg_price - sig.price) / distance_to_sl if distance_to_sl > 0 else 0
                                    
                                    threshold = current_settings.risk.smart_add_tp_sl_progress_pct
                                    
                                    if progress_to_tp >= threshold:
                                        can_add = True
                                        add_reason = (
                                            f"price moved {progress_to_tp*100:.1f}% to TP "
                                            f"(threshold: {threshold*100:.0f}%)"
                                        )
                                    elif progress_to_sl >= threshold:
                                        can_add = True
                                        add_reason = (
                                            f"price moved {progress_to_sl*100:.1f}% to SL "
                                            f"(threshold: {threshold*100:.0f}%) - averaging down"
                                        )
                                    else:
                                        print(
                                            "[live] ⚠️ Skipping ADD_LONG: price not moved enough "
                                            f"(to TP: {progress_to_tp*100:.1f}%, "
                                            f"to SL: {progress_to_sl*100:.1f}%, "
                                            f"need: {threshold*100:.0f}%)"
                                        )
                                        if _wait_with_stop_check(
                                            stop_event, current_settings.live_poll_seconds, symbol
                                        ):
                                            break
                                        continue
                            except (ValueError, TypeError) as e:
                                print(f"[live] ⚠️ Error calculating TP/SL progress: {e}")
                                # Fallback на старую логику pullback
                                max_price = position_max_price.get(symbol, sig.price)
                                pullback_pct = (
                                    ((max_price - sig.price) / max_price) * 100 if max_price > 0 else 0
                                )
                                if pullback_pct >= current_settings.risk.smart_add_pullback_pct * 100:
                                    can_add = True
                                    add_reason = f"pullback {pullback_pct:.2f}% (fallback logic)"
                        else:
                            # Нет TP/SL - используем старую логику откатов
                            max_price = position_max_price.get(symbol, sig.price)
                            pullback_pct = (
                                ((max_price - sig.price) / max_price) * 100 if max_price > 0 else 0
                            )
                            if pullback_pct >= current_settings.risk.smart_add_pullback_pct * 100:
                                can_add = True
                                add_reason = f"pullback {pullback_pct:.2f}% (no TP/SL set)"
                        
                        if not can_add:
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        print(f"[live] 📊 ADD_LONG conditions met: {add_reason}")
                    
                    # Рассчитываем количество контрактов как половину от текущего размера позиции
                    current_size = position.get("size", 0)
                    if current_size <= 0:
                        print(f"[live] ⚠️ Skipping ADD_LONG: invalid position size ({current_size})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    qty = _calculate_add_position_qty(client, current_size, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_LONG: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📈 Adding to LONG position: {qty:.3f} (half of {current_size:.3f}) @ ${sig.price:.2f} [Add #{current_add_count + 1}/{max_adds}]")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Buy",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        # Обновляем счётчик докупок
                        position_add_count[symbol] = current_add_count + 1
                        print(f"[live] ✅ Added to LONG position successfully (add #{position_add_count[symbol]}/{max_adds})")
                        
                        # Пересчитываем и обновляем SL по новой средней цене
                        if current_settings.risk.smart_add_adjust_sl:
                            try:
                                # Ждём обновления позиции
                                import time as time_module
                                time_module.sleep(0.5)
                                
                                # Получаем обновлённую позицию
                                updated_position = _get_position(client, symbol)
                                if updated_position:
                                    new_avg_price = updated_position.get("avg_price", 0)
                                    if new_avg_price > 0:
                                        # Рассчитываем новый SL (тот же % от новой средней цены)
                                        sl_pct = current_settings.risk.stop_loss_pct
                                        new_sl = new_avg_price * (1 - sl_pct)
                                        
                                        print(f"[live] 🔄 Adjusting SL: avg price ${avg_price:.2f} → ${new_avg_price:.2f}, new SL: ${new_sl:.2f}")
                                        
                                        sl_resp = client.set_trading_stop(
                                            symbol=symbol,
                                            stop_loss=new_sl,
                                        )
                                        if sl_resp.get("retCode") == 0:
                                            print(f"[live] ✅ SL adjusted to ${new_sl:.2f} after averaging")
                                        else:
                                            print(f"[live] ⚠️ Failed to adjust SL: {sl_resp.get('retMsg', 'Unknown error')}")
                            except Exception as e:
                                print(f"[live] ⚠️ Error adjusting SL after add: {e}")
                        
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                    else:
                        print(f"[live] ⚠️ Failed to add to LONG position: {resp.get('retMsg', 'Unknown error')}")
                elif current_position_bias == Bias.SHORT:
                    # Позиция SHORT и сигнал LONG → закрываем SHORT и открываем LONG
                    strategy_name = get_strategy_type_from_signal(sig.reason).upper()
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    print(f"[live] [{symbol}] 🔄 REVERSAL: Closing SHORT and opening LONG")
                    print(f"[live] [{symbol}] 📊 Signal: {strategy_name} {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] (ID: {signal_id})")
                    # Детальное логирование разворота
                    if sig.indicators_info:
                        info = sig.indicators_info
                        strategy_name_info = info.get("strategy", "UNKNOWN")
                        indicators_str = info.get("indicators", "N/A")
                        print(f"[live] [{symbol}] 📊 Reversal decision: Strategy={strategy_name_info}, Indicators: {indicators_str}")
                        if strategy_name_info == "TREND" and info.get("reason") == "bias_flip":
                            print(f"[live] [{symbol}]   Bias changed: {info.get('previous_bias', 'N/A')} → {info.get('bias', 'N/A')}")
                        elif strategy_name_info == "ML":
                            print(f"[live] [{symbol}]   ML Confidence={info.get('confidence_pct', 'N/A')}% ({info.get('strength', 'N/A')})")
                    
                    # Закрываем SHORT
                    side = "Buy"
                    print("=" * 80)
                    print(f"[live] [{symbol}] ⚫⚫⚫ CLOSING POSITION: SHORT → LONG REVERSAL ⚫⚫⚫")
                    print(f"[live] [{symbol}]   Closing SHORT: qty={position['size']:.3f}, reduce_only=True")
                    print(f"[live] [{symbol}]   Entry Price: ${position.get('avg_price', sig.price):.2f}")
                    print(f"[live] [{symbol}]   Exit Price: ${sig.price:.2f}")
                    print(f"[live] [{symbol}]   PnL: ${position.get('unrealised_pnl', 0):.2f}")
                    print("=" * 80)
                    resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=position["size"],
                        reduce_only=True,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] [{symbol}] ✅ Closed SHORT position successfully")
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        
                        # Открываем LONG
                        balance = _get_balance(client)
                        if balance is None:
                            print(f"[live] ⚠️ Failed to get balance for LONG")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                        qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                        
                        if qty <= 0:
                            print(f"[live] ⚠️ Invalid qty for LONG ({qty})")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        print(f"[live] 📈 Opening LONG position: {qty:.3f} @ ${sig.price:.2f} [Signal ID: {signal_id}]")
                        # Формируем уникальный order_link_id с timestamp для избежания дубликатов
                        timestamp_ms_reverse = int(time.time() * 1000)
                        unique_order_link_id_reverse = f"sig_{signal_id}_{timestamp_ms_reverse}"
                        
                        # Рассчитываем TP и SL для новой позиции при реверсе
                        take_profit, stop_loss = _calculate_tp_sl_for_signal(sig, current_settings, sig.price, df_ready)
                        if take_profit and stop_loss:
                            print(f"[live]   TP: ${take_profit:.2f} (+{((take_profit - sig.price) / sig.price * 100):.2f}%), SL: ${stop_loss:.2f} ({((stop_loss - sig.price) / sig.price * 100):.2f}%)")
                        
                        # Размещаем ордер БЕЗ TP/SL (для Market ордеров Bybit не поддерживает установку TP/SL при размещении)
                        resp = client.place_order(
                            symbol=symbol,
                            side="Buy",
                            qty=qty,
                            order_link_id=unique_order_link_id_reverse,
                        )
                        
                        if resp.get("retCode") == 0:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            print("=" * 80)
                            print(f"[live] 🟢🟢🟢 POSITION OPENED: LONG (AFTER REVERSAL) 🟢🟢🟢")
                            print(f"[live]   Strategy: {strategy_type.upper()}")
                            print(f"[live]   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                            print(f"[live]   Quantity: {qty:.3f} (${desired_usd:.2f})")
                            print(f"[live]   Order Link ID: {unique_order_link_id_reverse}")
                            print("=" * 80)
                            # Запоминаем время открытия позиции для защиты от ложных закрытий
                            position_opened_time = datetime.now(timezone.utc)
                            
                            # Отмечаем, что обработан свежий сигнал (для оптимизации интервала ожидания)
                            if is_fresh_check:
                                fresh_signal_processed = True
                                _log(f"✅ Fresh signal processed (reversal) - will check for new signals immediately", symbol)
                            
                            # КРИТИЧЕСКИ ВАЖНО: Сохраняем сигнал LONG в историю при реверсе
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type=strategy_type,
                                    signal_id=sig_signal_id,
                                )
                                print(f"[live] 💾 Saved LONG signal to history (reversal): {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                            except Exception as e:
                                print(f"[live] ⚠️ Failed to save LONG signal to history (reversal): {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # КРИТИЧЕСКИ ВАЖНО: Сохраняем LONG позицию в историю при реверсе
                            try:
                                result = resp.get("result", {})
                                order_id = result.get("orderId", "") if result else ""
                                order_link_id_result = result.get("orderLinkId", unique_order_link_id_reverse) if result else unique_order_link_id_reverse
                                
                                add_trade(
                                    entry_time=ts_log,
                                    exit_time=None,  # Позиция еще открыта
                                    side="long",  # ВАЖНО: LONG позиция при реверсе
                                    entry_price=sig.price,
                                    exit_price=0.0,
                                    size_usd=desired_usd,
                                    pnl=0.0,
                                    entry_reason=sig.reason,
                                    exit_reason="",
                                    strategy_type=strategy_type,
                                    symbol=symbol,
                                    order_id=order_id,
                                    order_link_id=order_link_id_result,
                                )
                                print(f"[live] 💾 Saved LONG position to history (reversal): {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                            except Exception as e:
                                print(f"[live] ⚠️ Error saving LONG position to history (reversal): {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Устанавливаем TP/SL сразу после успешного открытия позиции
                            if take_profit and stop_loss:
                                try:
                                    import time as time_module
                                    time_module.sleep(0.5)
                                    
                                    tp_sl_resp = client.set_trading_stop(
                                        symbol=symbol,
                                        take_profit=take_profit,
                                        stop_loss=stop_loss,
                                    )
                                    if tp_sl_resp.get("retCode") == 0:
                                        # Правильно форматируем проценты в зависимости от направления позиции
                                        if sig.action == Action.LONG:
                                            tp_pct_str = f"+{((take_profit - sig.price) / sig.price * 100):.2f}%"
                                            sl_pct_str = f"{((stop_loss - sig.price) / sig.price * 100):.2f}%"
                                        else:  # SHORT
                                            tp_pct_str = f"{((take_profit - sig.price) / sig.price * 100):.2f}%"  # Отрицательный процент (цена ниже входа)
                                            sl_pct_str = f"+{((stop_loss - sig.price) / sig.price * 100):.2f}%"  # Положительный процент (цена выше входа)
                                        print(f"[live] ✅ TP/SL set successfully: TP=${take_profit:.2f} ({tp_pct_str}), SL=${stop_loss:.2f} ({sl_pct_str})")
                                    else:
                                        print(f"[live] ⚠️ Failed to set TP/SL: {tp_sl_resp.get('retMsg', 'Unknown error')} (will retry via _ensure_tp_sl_set)")
                                except Exception as e:
                                    print(f"[live] ⚠️ Error setting TP/SL immediately: {e} (will retry via _ensure_tp_sl_set)")
                                    import traceback
                                    traceback.print_exc()
                            
                            processed_signals.add(signal_id)
                            _save_processed_signals(processed_signals, processed_signals_file)
                            last_handled_signal = (ts, sig.action.value)
                            
                            # Сохраняем состояние позиции (реверс)
                            result = resp.get("result", {})
                            order_id = result.get("orderId", "") if result else ""
                            order_link_id_result = result.get("orderLinkId", unique_order_link_id_reverse) if result else unique_order_link_id_reverse
                            
                            _update_and_save_position_state(
                                symbol=symbol,
                                position_strategy=position_strategy,
                                position_order_id=position_order_id,
                                position_order_link_id=position_order_link_id,
                                position_add_count=position_add_count,
                                position_entry_price=position_entry_price,
                                strategy_type=strategy_type,
                                order_id=order_id,
                                order_link_id=order_link_id_result,
                                add_count=0,
                                entry_price=sig.price
                            )
                            
                            # Сохраняем открытую сделку в историю (реверс LONG)
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                add_trade(
                                    entry_time=ts_log,
                                    exit_time=None,  # Позиция еще открыта
                                    side="long",
                                    entry_price=sig.price,
                                    exit_price=0.0,
                                    size_usd=desired_usd,
                                    pnl=0.0,
                                    entry_reason=sig.reason,
                                    exit_reason="",
                                    strategy_type=strategy_type,
                                    symbol=symbol,
                                    order_id=order_id,
                                    order_link_id=order_link_id_result,
                                )
                                print(f"[live] 💾 Saved open LONG trade to history (reversal): {strategy_type.upper()} @ ${sig.price:.2f} ({sig.reason})")
                            except Exception as e:
                                print(f"[live] ⚠️ Failed to save open LONG trade to history (reversal): {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            print(f"[live] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to open LONG position: {resp.get('retMsg', 'Unknown error')}")
                    else:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to close SHORT position: {resp.get('retMsg', 'Unknown error')}")
            
            # SHORT сигнал
            elif sig.action == Action.SHORT:
                print(f"[live] 🔍 Processing SHORT signal: position exists={position is not None}, position_bias={current_position_bias if position else 'None'}")
                
                # ВАЖНО: Если позиция уже SHORT и сигнал SHORT - всегда добавляем к позиции, а не открываем новую
                if position and current_position_bias == Bias.SHORT:
                    should_add_to_position = True
                    print(f"[live] ✅ Position already SHORT - will ADD to position instead of opening new one")
                
                # Проверяем приоритет стратегии перед закрытием позиции
                signal_strategy_type = get_strategy_type_from_signal(sig.reason)
                can_close_position = True
                
                if position and current_position_bias == Bias.LONG:
                    # Есть LONG позиция и приходит SHORT сигнал
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Если сигнал от той же стратегии, что открыла позицию - закрываем позицию
                    entry_reason = None
                    try:
                        from bot.web.history import get_open_trade
                        avg_price = position.get("avg_price", 0)
                        if avg_price > 0:
                            open_trade = get_open_trade(symbol, entry_price=avg_price, price_tolerance_pct=0.05)
                            if open_trade:
                                entry_reason = open_trade.get("entry_reason", "")
                    except Exception as e:
                        print(f"[live] ⚠️ Error getting entry_reason: {e}")
                    
                    position_strategy_type = get_strategy_type_from_signal(entry_reason) if entry_reason else None
                    
                    # Если позиция открыта по той же стратегии, что и сигнал - проверяем свежесть сигнала
                    if position_strategy_type and position_strategy_type == signal_strategy_type:
                        # Проверяем, является ли сигнал свежим
                        is_fresh = is_signal_fresh(sig, df_ready)
                        if is_fresh:
                            print(f"[live] 🔄 SAME STRATEGY REVERSAL (FRESH): LONG position opened by {position_strategy_type.upper()}, fresh opposite SHORT signal from same strategy - closing and opening new position")
                            can_close_position = True  # Принудительно разрешаем закрытие и открытие новой позиции
                        else:
                            print(f"[live] ⚠️ SAME STRATEGY REVERSAL (NOT FRESH): LONG position opened by {position_strategy_type.upper()}, but opposite SHORT signal is not fresh - closing position only")
                            can_close_position = True  # Закрываем позицию, но не открываем новую (сигнал не свежий)
                    else:
                        # Проверяем приоритет стратегии только если сигнал от другой стратегии
                        is_priority_position = position_strategy_type == strategy_priority
                        
                        if is_priority_position and signal_strategy_type != strategy_priority:
                            # Позиция открыта по приоритетной стратегии, а сигнал от другой стратегии - защищаем позицию
                            can_close_position = False
                            print(f"[live] 🛡️ PRIORITY PROTECTION: LONG position opened by {strategy_priority.upper()} strategy, ignoring opposite SHORT signal from {signal_strategy_type.upper()}")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: Если есть LONG позиция и приходит SHORT сигнал - закрываем LONG и открываем SHORT
                if position and current_position_bias == Bias.LONG and can_close_position:
                    strategy_type = get_strategy_type_from_signal(sig.reason)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    _log(f"🔄 REVERSAL: Closing LONG position to open SHORT (signal: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f})", symbol)
                    
                    # Закрываем LONG позицию
                    close_qty = position.get("size", 0)
                    if close_qty > 0:
                        try:
                            resp = client.place_order(
                                symbol=symbol,
                                side="Sell",  # Sell закрывает LONG
                                qty=close_qty,
                                reduce_only=True,
                            )
                            if resp.get("retCode") == 0:
                                print(f"[live] [{symbol}] ✅ LONG position closed for reversal to SHORT")
                                # Ждем немного, чтобы позиция закрылась
                                import time as time_module
                                time_module.sleep(1.0)
                                # Перезагружаем информацию о позиции из API
                                try:
                                    pos_resp = client.get_position_info(symbol=symbol)
                                    if pos_resp.get("retCode") == 0:
                                        pos_list = pos_resp.get("result", {}).get("list", [])
                                        position = None
                                        current_position_bias = None
                                        for pos_item in pos_list:
                                            if float(pos_item.get("size", 0)) > 0:
                                                position = pos_item
                                                current_position_bias = Bias.LONG if pos_item.get("side") == "Buy" else Bias.SHORT
                                                break
                                        if position is None:
                                            print(f"[live] [{symbol}] ✅ Position confirmed closed, proceeding to open SHORT")
                                            # Продолжаем выполнение, чтобы открыть новую позицию
                                            # Не делаем break или continue - код продолжит выполнение и откроет SHORT позицию
                                        else:
                                            print(f"[live] [{symbol}] ⚠️ Position still exists after close attempt, skipping SHORT open")
                                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                                break
                                            continue
                                except Exception as e:
                                    print(f"[live] [{symbol}] ⚠️ Error reloading position info: {e}, assuming closed")
                                    position = None
                                    current_position_bias = None
                            else:
                                print(f"[live] [{symbol}] ⚠️ Failed to close LONG position: {resp.get('retMsg', 'Unknown error')}")
                                if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                    break
                                continue
                        except Exception as e:
                            print(f"[live] [{symbol}] ⚠️ Error closing LONG position: {e}")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                
                if not position:
                    # Позиции нет → открываем SHORT
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Перепроверяем наличие позиции перед открытием новой
                    # Это предотвращает открытие нескольких позиций по одной паре
                    position_check = _get_position(client, symbol)
                    if position_check and position_check.get("size", 0) > 0:
                        _log(f"⚠️ Position already exists for {symbol} (size: {position_check.get('size', 0)}), skipping new position open", symbol)
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    # КРИТИЧЕСКАЯ ПРОВЕРКА: Не открываем SHORT, если на PRIMARY_SYMBOL есть LONG позиция
                    _log(f"🔍 [FINAL CHECK] Checking PRIMARY_SYMBOL position before opening SHORT for {symbol}...", symbol)
                    _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) from {strategy_name}", symbol)
                    # ВАЖНО: Используем ТОЛЬКО primary_symbol из настроек, БЕЗ fallback на symbol
                    primary_symbol_from_settings = getattr(current_settings, 'primary_symbol', None)
                    _log(f"   PRIMARY_SYMBOL from settings: {primary_symbol_from_settings}", symbol)
                    _log(f"   Current symbol: {symbol}", symbol)
                    
                    # Проверяем, включена ли функция следования за главным символом
                    follow_primary_symbol = getattr(current_settings, 'follow_primary_symbol', True)  # По умолчанию True
                    should_block = False
                    block_reason = None
                    if follow_primary_symbol:
                        should_block, block_reason = _check_primary_symbol_position(
                            client=client,
                            current_symbol=symbol,
                            settings=current_settings,
                            target_action=Action.SHORT,
                        )
                    else:
                        _log(f"ℹ️ FOLLOW_PRIMARY_SYMBOL is disabled - skipping PRIMARY_SYMBOL check for {symbol}", symbol)
                    
                    _log(f"   [FINAL CHECK RESULT] PRIMARY_SYMBOL check result: should_block={should_block}, reason={block_reason}", symbol)
                    if should_block:
                        _log(f"⛔ [FINAL CHECK] BLOCKED: {block_reason}", symbol)
                        _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) - waiting for PRIMARY_SYMBOL position to close or reverse", symbol)
                        if bot_state:
                            bot_state["current_status"] = "Running"
                            bot_state["last_action"] = f"Blocked: {block_reason}"
                            bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                        update_worker_status(symbol, current_status="Running", last_action=f"Blocked: {block_reason}")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    else:
                        _log(f"✅ [FINAL CHECK] PRIMARY_SYMBOL check passed - SHORT position allowed for {symbol}", symbol)
                    
                    strategy_type = get_strategy_type_from_signal(sig.reason)
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    _log(f"📉 Opening NEW SHORT position after close", symbol)
                    _log(f"   Signal: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] (ID: {signal_id})", symbol)
                    
                    # Проверяем историю убыточных сделок перед открытием
                    if current_settings.risk.enable_loss_cooldown:
                        should_block, last_loss = check_recent_loss_trade(
                            side="short",
                            symbol=symbol,
                            cooldown_minutes=current_settings.risk.loss_cooldown_minutes,
                            max_losses=current_settings.risk.max_consecutive_losses,
                        )
                        if should_block:
                            if last_loss:
                                exit_reason = last_loss.get("exit_reason", "unknown")
                                pnl = last_loss.get("pnl", 0)
                                print(f"[live] ⛔ Blocking SHORT: recent loss trade detected (PnL: {pnl:.2f} USDT, reason: {exit_reason})")
                            else:
                                print(f"[live] ⛔ Blocking SHORT: too many consecutive losses")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                    
                    # Проверяем ATR перед открытием позиции (используем ATR с 1H и 4H таймфреймов)
                    if current_settings.risk.enable_atr_entry_filter and not df_ready.empty:
                        try:
                            last_row = df_ready.iloc[-1]
                            # Используем среднее значение ATR с 1H и 4H таймфреймов для среднесрочного анализа
                            atr_value = last_row.get("atr_avg", None)  # Среднее ATR с 1H и 4H
                            # Fallback на 15M ATR если нет данных с высших таймфреймов
                            if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
                                atr_value = last_row.get("atr", None)
                            current_price = sig.price
                            
                            if atr_value is not None and pd.notna(atr_value) and atr_value > 0:
                                # Получаем предыдущие свечи для анализа движения цены
                                if len(df_ready) >= 2:
                                    prev_row = df_ready.iloc[-2]
                                    prev_close = prev_row.get("close", current_price)
                                    
                                    # Рассчитываем, какую часть ATR (среднесрочного) цена уже прошла в направлении сигнала
                                    if pd.notna(prev_close):
                                        price_move = prev_close - current_price  # Для SHORT: движение вниз
                                        atr_progress = abs(price_move) / atr_value if atr_value > 0 else 0
                                        
                                        # Для SHORT сигнала: если цена уже прошла вниз большую часть среднесрочного ATR - не входить
                                        if price_move > 0 and atr_progress > current_settings.risk.max_atr_progress_pct:
                                            atr_1h = last_row.get("atr_1h", 0)
                                            atr_4h = last_row.get("atr_4h", 0)
                                            print(f"[live] ⛔ Blocking SHORT: price already moved {atr_progress*100:.1f}% of avg ATR(1H+4H) down (threshold: {current_settings.risk.max_atr_progress_pct*100:.1f}%)")
                                            print(f"[live]   Current: ${current_price:.2f}, Previous: ${prev_close:.2f}, ATR avg(1H+4H): ${atr_value:.2f} (1H: ${atr_1h:.2f}, 4H: ${atr_4h:.2f}), Move: ${price_move:.2f}")
                                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                                break
                                            continue
                        except Exception as e:
                            print(f"[live] ⚠️ Error checking ATR filter: {e}")
                    
                    balance = _get_balance(client)
                    if balance is None:
                        print(f"[live] ⚠️ Skipping SHORT: failed to get balance")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                    qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping SHORT: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    # Детальное логирование решения
                    if sig.indicators_info:
                        info = sig.indicators_info
                        strategy_name = info.get("strategy", "UNKNOWN")
                        indicators_str = info.get("indicators", "N/A")
                        entry_type = info.get("entry_type", "")
                        print(f"[live] 📉 Opening SHORT position: {qty:.3f} @ ${sig.price:.2f} (${desired_usd:.2f}) [Signal ID: {signal_id}]")
                        print(f"[live] 📊 Decision path: Strategy={strategy_name}, Entry={entry_type}, Indicators: {indicators_str}")
                        if strategy_name == "TREND":
                            print(f"[live]   ADX={info.get('adx', 'N/A')}, +DI={info.get('plus_di', 'N/A')}, -DI={info.get('minus_di', 'N/A')}, Bias={info.get('bias', 'N/A')}")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                        elif strategy_name == "FLAT":
                            print(f"[live]   RSI={info.get('rsi', 'N/A')}, BB_lower={info.get('bb_lower', 'N/A')}, BB_middle={info.get('bb_middle', 'N/A')}, BB_upper={info.get('bb_upper', 'N/A')}")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                        elif strategy_name == "ML":
                            print(f"[live]   ML Confidence={info.get('confidence_pct', 'N/A')}% ({info.get('strength', 'N/A')}), TP={info.get('tp_pct', 'N/A')}%, SL={info.get('sl_pct', 'N/A')}%")
                            print(f"[live]   Volume={info.get('volume', 'N/A')}, Vol_SMA={info.get('vol_sma', 'N/A')}, Vol_Ratio={info.get('vol_ratio', 'N/A')}x")
                    else:
                        print(f"[live] 📉 Opening SHORT position: {qty:.3f} @ ${sig.price:.2f} (${desired_usd:.2f}) [Signal ID: {signal_id}]")
                    
                    # Формируем уникальный order_link_id с timestamp для избежания дубликатов
                    timestamp_ms = int(time.time() * 1000)
                    unique_order_link_id = f"sig_{signal_id}_{timestamp_ms}"
                    
                    # Рассчитываем TP и SL для новой позиции
                    take_profit, stop_loss = _calculate_tp_sl_for_signal(sig, current_settings, sig.price, df_ready)
                    if take_profit and stop_loss:
                        print(f"[live]   TP: ${take_profit:.2f} ({((take_profit - sig.price) / sig.price * 100):.2f}%), SL: ${stop_loss:.2f} ({((stop_loss - sig.price) / sig.price * 100):.2f}%)")
                    
                    # Размещаем ордер БЕЗ TP/SL (для Market ордеров Bybit не поддерживает установку TP/SL при размещении)
                    try:
                        resp = client.place_order(
                            symbol=symbol,
                            side="Sell",
                            qty=qty,
                            order_link_id=unique_order_link_id,
                        )
                    except InvalidRequestError as e:
                        # Обрабатываем ошибки API (например, недостаточный баланс)
                        error_msg = str(e)
                        error_code = None
                        if "ErrCode" in error_msg:
                            import re
                            code_match = re.search(r'ErrCode:\s*(\d+)', error_msg)
                            if code_match:
                                error_code = int(code_match.group(1))
                        
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        if error_code == 110007:
                            _log(f"❌ INSUFFICIENT BALANCE: Cannot open SHORT position - {error_msg}", symbol)
                            _log(f"   Signal was generated but cannot be executed. Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        else:
                            _log(f"❌ ORDER ERROR: Failed to open SHORT position - {error_msg}", symbol)
                            _log(f"   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        
                        # Сохраняем сигнал в историю даже при ошибке выполнения
                        try:
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type=strategy_type,
                                signal_id=sig_signal_id,
                            )
                            _log(f"💾 Signal saved to history despite order error", symbol)
                        except Exception as save_error:
                            _log(f"⚠️ Failed to save signal to history: {save_error}", symbol)
                        
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    if resp.get("retCode") == 0:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print("=" * 80)
                        print(f"[live] 🔴🔴🔴 POSITION OPENED: SHORT 🔴🔴🔴")
                        print(f"[live]   Strategy: {strategy_type.upper()}")
                        print(f"[live]   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                        print(f"[live]   Quantity: {qty:.3f} (${desired_usd:.2f})")
                        print(f"[live]   Order Link ID: {unique_order_link_id}")
                        print("=" * 80)
                        # Запоминаем время открытия позиции для защиты от ложных закрытий
                        position_opened_time = datetime.now(timezone.utc)
                        
                        # Отмечаем, что обработан свежий сигнал (для оптимизации интервала ожидания)
                        if is_fresh_check:
                            fresh_signal_processed = True
                            _log(f"✅ Fresh signal processed - will check for new signals immediately", symbol)
                        
                        # Устанавливаем TP/SL сразу после успешного открытия позиции
                        if take_profit and stop_loss:
                            try:
                                # Небольшая задержка, чтобы позиция точно открылась
                                import time as time_module
                                time_module.sleep(0.5)
                                
                                tp_sl_resp = client.set_trading_stop(
                                    symbol=symbol,
                                    take_profit=take_profit,
                                    stop_loss=stop_loss,
                                )
                                if tp_sl_resp.get("retCode") == 0:
                                    # Правильно форматируем проценты в зависимости от направления позиции
                                    if sig.action == Action.LONG:
                                        tp_pct_str = f"+{((take_profit - sig.price) / sig.price * 100):.2f}%"
                                        sl_pct_str = f"{((stop_loss - sig.price) / sig.price * 100):.2f}%"
                                    else:  # SHORT
                                        tp_pct_str = f"{((take_profit - sig.price) / sig.price * 100):.2f}%"  # Отрицательный процент (цена ниже входа)
                                        sl_pct_str = f"+{((stop_loss - sig.price) / sig.price * 100):.2f}%"  # Положительный процент (цена выше входа)
                                    print(f"[live] ✅ TP/SL set successfully: TP=${take_profit:.2f} ({tp_pct_str}), SL=${stop_loss:.2f} ({sl_pct_str})")
                                else:
                                    print(f"[live] ⚠️ Failed to set TP/SL: {tp_sl_resp.get('retMsg', 'Unknown error')} (will retry via _ensure_tp_sl_set)")
                            except Exception as e:
                                print(f"[live] ⚠️ Error setting TP/SL immediately: {e} (will retry via _ensure_tp_sl_set)")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                        
                        # Сохраняем состояние позиции
                        result = resp.get("result", {})
                        order_id = result.get("orderId", "") if result else ""
                        order_link_id_result = result.get("orderLinkId", unique_order_link_id) if result else unique_order_link_id
                        
                        _update_and_save_position_state(
                            symbol=symbol,
                            position_strategy=position_strategy,
                            position_order_id=position_order_id,
                            position_order_link_id=position_order_link_id,
                            position_add_count=position_add_count,
                            position_entry_price=position_entry_price,
                            strategy_type=strategy_type,
                            order_id=order_id,
                            order_link_id=order_link_id_result,
                            add_count=0,
                            entry_price=sig.price
                        )
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        
                        # КРИТИЧЕСКИ ВАЖНО: Сохраняем SHORT позицию в историю
                        try:
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            elif isinstance(ts_log, datetime):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.replace(tzinfo=timezone.utc)
                            else:
                                ts_log = datetime.now(timezone.utc)
                            
                            # ВАЛИДАЦИЯ: Убеждаемся, что side соответствует sig.action
                            expected_side = "short" if sig.action == Action.SHORT else "long"
                            if expected_side != "short":
                                _log(f"⚠️ WARNING: sig.action={sig.action.value} but trying to save SHORT position! Using expected_side={expected_side}", symbol)
                            
                            add_trade(
                                entry_time=ts_log,
                                exit_time=None,  # Позиция еще открыта
                                side=expected_side,  # ВАЖНО: Используем валидированный side
                                entry_price=sig.price,
                                exit_price=0.0,
                                size_usd=desired_usd,
                                pnl=0.0,
                                entry_reason=sig.reason,
                                exit_reason="",
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=order_id,
                                order_link_id=order_link_id_result,
                            )
                            _log(f"💾 Saved {expected_side.upper()} position to history: {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason})", symbol)
                        except Exception as e:
                            _log(f"⚠️ Error saving SHORT position to history: {e}", symbol)
                        
                        # ОТКЛЮЧЕНО: Автоматическое закрытие позиций при открытии на PRIMARY_SYMBOL
                        # Эта логика вызывала каскадное закрытие позиций, когда все сигналы в одном направлении
                        # Если нужно закрыть противонаправленные позиции, это должно делаться вручную или через другую логику
                        # primary_symbol_for_check = getattr(current_settings, "primary_symbol", None) or getattr(current_settings, "symbol", None)
                        # if primary_symbol_for_check and symbol.upper() == str(primary_symbol_for_check).upper():
                        #     # Перепроверяем, что позиция действительно открыта на PRIMARY_SYMBOL
                        #     try:
                        #         position_verify = _get_position(client, symbol)
                        #         if position_verify and position_verify.get("size", 0) > 0:
                        #             _log(f"✅ Position confirmed open on PRIMARY_SYMBOL ({symbol}) - closing opposite LONG positions on other symbols", symbol)
                        #             try:
                        #                 _close_conflicting_positions_for_primary(
                        #                     client=client,
                        #                     settings=current_settings,
                        #                     new_primary_bias=Bias.SHORT,
                        #                 )
                        #             except Exception as e:
                        #                 print(f"[live] [{symbol}] ⚠️ Error while closing opposite positions for PRIMARY_SYMBOL SHORT: {e}")
                        #         else:
                        #             _log(f"⚠️ Position not confirmed on PRIMARY_SYMBOL ({symbol}) - skipping close of opposite positions", symbol)
                        #     except Exception as e:
                        #         _log(f"⚠️ Error verifying position on PRIMARY_SYMBOL before closing opposite positions: {e}", symbol)
                    elif resp.get("retCode") == 110072:
                        # Ошибка дубликата order_link_id - сигнал уже обработан
                        print(f"[live] [{symbol}] ⚠️ OrderLinkID duplicate - signal already processed: {signal_id}")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                    else:
                        print(f"[live] ⚠️ Failed to open SHORT position: {resp.get('retMsg', 'Unknown error')} (ErrCode: {resp.get('retCode')})")
                elif current_position_bias == Bias.SHORT:
                    # Позиция SHORT и сигнал SHORT → добавляем к позиции (ADD_SHORT)
                    # Это может быть подтверждение от другой стратегии или повторный сигнал от той же
                    if should_add_to_position:
                        print(f"[live] 📊 Adding to position: signals from different strategies confirm each other")
                    
                    # Проверяем smart add условия
                    if current_settings.risk.enable_smart_add:
                        # 1. Проверяем лимит докупок
                        current_add_count = position_add_count.get(symbol, 0)
                        max_adds = current_settings.risk.max_add_count
                        if current_add_count >= max_adds:
                            print(f"[live] ⚠️ Skipping ADD_SHORT: max adds reached ({current_add_count}/{max_adds})")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        # 2. Проверяем прогресс к TP или SL (>50% пути)
                        avg_price = position.get("avg_price", sig.price)
                        current_tp = position.get("take_profit", "")
                        current_sl = position.get("stop_loss", "")
                        
                        can_add = False
                        add_reason = ""
                        
                        if current_tp and current_sl and avg_price > 0:
                            try:
                                tp_price = float(current_tp) if current_tp else 0
                                sl_price = float(current_sl) if current_sl else 0
                                
                                if tp_price > 0 and sl_price > 0:
                                    # Расчёт прогресса к TP (для SHORT: цена падает к TP)
                                    distance_to_tp = avg_price - tp_price
                                    progress_to_tp = (
                                        (avg_price - sig.price) / distance_to_tp if distance_to_tp > 0 else 0
                                    )
                                    
                                    # Расчёт прогресса к SL (для SHORT: цена растёт к SL)
                                    distance_to_sl = sl_price - avg_price
                                    progress_to_sl = (
                                        (sig.price - avg_price) / distance_to_sl if distance_to_sl > 0 else 0
                                    )
                                    
                                    threshold = current_settings.risk.smart_add_tp_sl_progress_pct
                                    
                                    if progress_to_tp >= threshold:
                                        can_add = True
                                        add_reason = (
                                            f"price moved {progress_to_tp*100:.1f}% to TP "
                                            f"(threshold: {threshold*100:.0f}%)"
                                        )
                                    elif progress_to_sl >= threshold:
                                        can_add = True
                                        add_reason = (
                                            f"price moved {progress_to_sl*100:.1f}% to SL "
                                            f"(threshold: {threshold*100:.0f}%) - averaging down"
                                        )
                                    else:
                                        print(
                                            "[live] ⚠️ Skipping ADD_SHORT: price not moved enough "
                                            f"(to TP: {progress_to_tp*100:.1f}%, "
                                            f"to SL: {progress_to_sl*100:.1f}%, "
                                            f"need: {threshold*100:.0f}%)"
                                        )
                                        if _wait_with_stop_check(
                                            stop_event, current_settings.live_poll_seconds, symbol
                                        ):
                                            break
                                        continue
                            except (ValueError, TypeError) as e:
                                print(f"[live] ⚠️ Error calculating TP/SL progress: {e}")
                                # Fallback на старую логику pullback
                                max_price = position_max_price.get(symbol, sig.price)
                                pullback_pct = (
                                    ((sig.price - max_price) / max_price) * 100 if max_price > 0 else 0
                                )
                                if pullback_pct >= current_settings.risk.smart_add_pullback_pct * 100:
                                    can_add = True
                                    add_reason = f"pullback {pullback_pct:.2f}% (fallback logic)"
                        else:
                            # Нет TP/SL - используем старую логику откатов
                            max_price = position_max_price.get(symbol, sig.price)
                            pullback_pct = (
                                ((sig.price - max_price) / max_price) * 100 if max_price > 0 else 0
                            )
                            if pullback_pct >= current_settings.risk.smart_add_pullback_pct * 100:
                                can_add = True
                                add_reason = f"pullback {pullback_pct:.2f}% (no TP/SL set)"
                        
                        if not can_add:
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        print(f"[live] 📊 ADD_SHORT conditions met: {add_reason}")
                    
                    # Рассчитываем количество контрактов как половину от текущего размера позиции
                    current_size = position.get("size", 0)
                    if current_size <= 0:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: invalid position size ({current_size})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    qty = _calculate_add_position_qty(client, current_size, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📉 Adding to SHORT position: {qty:.3f} (half of {current_size:.3f}) @ ${sig.price:.2f} [Add #{current_add_count + 1}/{max_adds}]")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Sell",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        # Обновляем счётчик докупок
                        position_add_count[symbol] = current_add_count + 1
                        print("=" * 60)
                        print(f"[live] 📊 ADDED TO POSITION: SHORT (add #{position_add_count[symbol]}/{max_adds})")
                        print(f"[live]   Quantity Added: {qty:.3f} @ ${sig.price:.2f}")
                        print(f"[live]   Total Position Size: {current_size + qty:.3f}")
                        print("=" * 60)
                        
                        # Пересчитываем и обновляем SL по новой средней цене
                        if current_settings.risk.smart_add_adjust_sl:
                            try:
                                # Ждём обновления позиции
                                import time as time_module
                                time_module.sleep(0.5)
                                
                                # Получаем обновлённую позицию
                                updated_position = _get_position(client, symbol)
                                if updated_position:
                                    new_avg_price = updated_position.get("avg_price", 0)
                                    if new_avg_price > 0:
                                        # Рассчитываем новый SL (тот же % от новой средней цены)
                                        sl_pct = current_settings.risk.stop_loss_pct
                                        new_sl = new_avg_price * (1 + sl_pct)  # Для SHORT SL выше цены входа
                                        
                                        print(f"[live] 🔄 Adjusting SL: avg price ${avg_price:.2f} → ${new_avg_price:.2f}, new SL: ${new_sl:.2f}")
                                        
                                        sl_resp = client.set_trading_stop(
                                            symbol=symbol,
                                            stop_loss=new_sl,
                                        )
                                        if sl_resp.get("retCode") == 0:
                                            print(f"[live] ✅ SL adjusted to ${new_sl:.2f} after averaging")
                                        else:
                                            print(f"[live] ⚠️ Failed to adjust SL: {sl_resp.get('retMsg', 'Unknown error')}")
                            except Exception as e:
                                print(f"[live] ⚠️ Error adjusting SL after add: {e}")
                        
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                    else:
                        print(f"[live] ⚠️ Failed to add to SHORT position: {resp.get('retMsg', 'Unknown error')}")
                elif current_position_bias == Bias.LONG:
                    # Позиция LONG и сигнал SHORT → закрываем LONG и открываем SHORT
                    strategy_name = get_strategy_type_from_signal(sig.reason).upper()
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                    print(f"[live] [{symbol}] 🔄 REVERSAL: Closing LONG and opening SHORT")
                    print(f"[live] [{symbol}] 📊 Signal: {strategy_name} {sig.action.value} @ ${sig.price:.2f} ({sig.reason}) [{ts_str}] (ID: {signal_id})")
                    # Детальное логирование разворота
                    if sig.indicators_info:
                        info = sig.indicators_info
                        strategy_name_info = info.get("strategy", "UNKNOWN")
                        indicators_str = info.get("indicators", "N/A")
                        print(f"[live] [{symbol}] 📊 Reversal decision: Strategy={strategy_name_info}, Indicators: {indicators_str}")
                        if strategy_name_info == "TREND" and info.get("reason") == "bias_flip":
                            print(f"[live] [{symbol}]   Bias changed: {info.get('previous_bias', 'N/A')} → {info.get('bias', 'N/A')}")
                        elif strategy_name_info == "ML":
                            print(f"[live] [{symbol}]   ML Confidence={info.get('confidence_pct', 'N/A')}% ({info.get('strength', 'N/A')})")
                    
                    # Закрываем LONG
                    side = "Sell"
                    print("=" * 80)
                    print(f"[live] [{symbol}] ⚫⚫⚫ CLOSING POSITION: LONG → SHORT REVERSAL ⚫⚫⚫")
                    print(f"[live] [{symbol}]   Closing LONG: qty={position['size']:.3f}, reduce_only=True")
                    print(f"[live] [{symbol}]   Entry Price: ${position.get('avg_price', sig.price):.2f}")
                    print(f"[live] [{symbol}]   Exit Price: ${sig.price:.2f}")
                    print(f"[live] [{symbol}]   PnL: ${position.get('unrealised_pnl', 0):.2f}")
                    print("=" * 80)
                    resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=position["size"],
                        reduce_only=True,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] [{symbol}] ✅ Closed LONG position successfully")
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        
                        # Открываем SHORT
                        balance = _get_balance(client)
                        if balance is None:
                            print(f"[live] ⚠️ Failed to get balance for SHORT")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                        qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                        
                        if qty <= 0:
                            print(f"[live] ⚠️ Invalid qty for SHORT ({qty})")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                        
                        print(f"[live] 📉 Opening SHORT position: {qty:.3f} @ ${sig.price:.2f} [Signal ID: {signal_id}]")
                        # Формируем уникальный order_link_id с timestamp для избежания дубликатов
                        timestamp_ms = int(time.time() * 1000)
                        unique_order_link_id = f"sig_{signal_id}_{timestamp_ms}"
                        
                        # Рассчитываем TP и SL для новой позиции при реверсе
                        take_profit, stop_loss = _calculate_tp_sl_for_signal(sig, current_settings, sig.price, df_ready)
                        if take_profit and stop_loss:
                            print(f"[live]   TP: ${take_profit:.2f} ({((take_profit - sig.price) / sig.price * 100):.2f}%), SL: ${stop_loss:.2f} ({((stop_loss - sig.price) / sig.price * 100):.2f}%)")
                        
                        # Размещаем ордер БЕЗ TP/SL (для Market ордеров Bybit не поддерживает установку TP/SL при размещении)
                        resp = client.place_order(
                            symbol=symbol,
                            side="Sell",
                            qty=qty,
                            order_link_id=unique_order_link_id,
                        )
                        
                        if resp.get("retCode") == 0:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            print("=" * 80)
                            print(f"[live] 🔴🔴🔴 POSITION OPENED: SHORT (AFTER REVERSAL) 🔴🔴🔴")
                            print(f"[live]   Strategy: {strategy_type.upper()}")
                            print(f"[live]   Signal: {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                            print(f"[live]   Quantity: {qty:.3f} (${desired_usd:.2f})")
                            print(f"[live]   Order Link ID: {unique_order_link_id}")
                            print("=" * 80)
                            # Запоминаем время открытия позиции для защиты от ложных закрытий
                            position_opened_time = datetime.now(timezone.utc)
                            
                            # Отмечаем, что обработан свежий сигнал (для оптимизации интервала ожидания)
                            if is_fresh_check:
                                fresh_signal_processed = True
                                _log(f"✅ Fresh signal processed (reversal) - will check for new signals immediately", symbol)
                            
                            # КРИТИЧЕСКИ ВАЖНО: Сохраняем сигнал SHORT в историю при реверсе
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                sig_signal_id = sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type=strategy_type,
                                    signal_id=sig_signal_id,
                                )
                                print(f"[live] 💾 Saved SHORT signal to history (reversal): {strategy_type.upper()} {sig.action.value} @ ${sig.price:.2f} ({sig.reason})")
                                
                                # КРИТИЧЕСКИ ВАЖНО: Сохраняем SHORT позицию в историю при реверсе
                                result = resp.get("result", {})
                                order_id = result.get("orderId", "") if result else ""
                                order_link_id_result = result.get("orderLinkId", unique_order_link_id) if result else unique_order_link_id
                                
                                # ВАЛИДАЦИЯ: Убеждаемся, что side соответствует sig.action
                                expected_side = "short" if sig.action == Action.SHORT else "long"
                                if expected_side != "short":
                                    print(f"[live] ⚠️ WARNING: sig.action={sig.action.value} but trying to save SHORT position (reversal)! Using expected_side={expected_side}")
                                
                                add_trade(
                                    entry_time=ts_log,
                                    exit_time=None,  # Позиция еще открыта
                                    side=expected_side,  # ВАЖНО: Используем валидированный side
                                    entry_price=sig.price,
                                    exit_price=0.0,
                                    size_usd=desired_usd,
                                    pnl=0.0,
                                    entry_reason=sig.reason,
                                    exit_reason="",
                                    strategy_type=strategy_type,
                                    symbol=symbol,
                                    order_id=order_id,
                                    order_link_id=order_link_id_result,
                                )
                                print(f"[live] 💾 Saved open {expected_side.upper()} trade to history (reversal): {strategy_type.upper()} @ ${sig.price:.2f} ({sig.reason})")
                            except Exception as e:
                                print(f"[live] ⚠️ Failed to save SHORT signal/trade to history (reversal): {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Устанавливаем TP/SL сразу после успешного открытия позиции
                            if take_profit and stop_loss:
                                try:
                                    import time as time_module
                                    time_module.sleep(0.5)
                                    
                                    tp_sl_resp = client.set_trading_stop(
                                        symbol=symbol,
                                        take_profit=take_profit,
                                        stop_loss=stop_loss,
                                    )
                                    if tp_sl_resp.get("retCode") == 0:
                                        print(f"[live] ✅ TP/SL set successfully: TP=${take_profit:.2f} ({((take_profit - sig.price) / sig.price * 100):.2f}%), SL=${stop_loss:.2f} ({((stop_loss - sig.price) / sig.price * 100):.2f}%)")
                                    else:
                                        print(f"[live] ⚠️ Failed to set TP/SL: {tp_sl_resp.get('retMsg', 'Unknown error')} (will retry via _ensure_tp_sl_set)")
                                except Exception as e:
                                    print(f"[live] ⚠️ Error setting TP/SL immediately: {e} (will retry via _ensure_tp_sl_set)")
                                    import traceback
                                    traceback.print_exc()
                            
                            processed_signals.add(signal_id)
                            _save_processed_signals(processed_signals, processed_signals_file)
                            last_handled_signal = (ts, sig.action.value)
                            
                            # Сохраняем состояние позиции (реверс)
                            result = resp.get("result", {})
                            order_id = result.get("orderId", "") if result else ""
                            order_link_id_result = result.get("orderLinkId", unique_order_link_id) if result else unique_order_link_id
                            
                            _update_and_save_position_state(
                                symbol=symbol,
                                position_strategy=position_strategy,
                                position_order_id=position_order_id,
                                position_order_link_id=position_order_link_id,
                                position_add_count=position_add_count,
                                position_entry_price=position_entry_price,
                                strategy_type=strategy_type,
                                order_id=order_id,
                                order_link_id=order_link_id_result,
                                add_count=0,
                                entry_price=sig.price
                            )
                            
                            # Сохраняем открытую сделку в историю (реверс LONG)
                            try:
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                
                                add_trade(
                                    entry_time=ts_log,
                                    exit_time=None,  # Позиция еще открыта
                                    side="long",
                                    entry_price=sig.price,
                                    exit_price=0.0,
                                    size_usd=desired_usd,
                                    pnl=0.0,
                                    entry_reason=sig.reason,
                                    exit_reason="",
                                    strategy_type=strategy_type,
                                    symbol=symbol,
                                    order_id=order_id,
                                    order_link_id=order_link_id_result,
                                )
                                print(f"[live] 💾 Saved open LONG trade to history (reversal): {strategy_type.upper()} @ ${sig.price:.2f} ({sig.reason})")
                            except Exception as e:
                                print(f"[live] ⚠️ Failed to save open LONG trade to history (reversal): {e}")
                                import traceback
                                traceback.print_exc()
                        elif resp.get("retCode") == 110072:
                            # Ошибка дубликата order_link_id - сигнал уже обработан
                            print(f"[live] ⚠️ OrderLinkID duplicate - signal already processed: {signal_id}")
                            processed_signals.add(signal_id)
                            _save_processed_signals(processed_signals, processed_signals_file)
                        else:
                            strategy_type = get_strategy_type_from_signal(sig.reason)
                            print(f"[live] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to open SHORT position: {resp.get('retMsg', 'Unknown error')} (ErrCode: {resp.get('retCode')})")
                    else:
                        print(f"[live] ⚠️ Failed to close LONG position: {resp.get('retMsg', 'Unknown error')}")
            
            elif sig.action == Action.ADD_LONG:
                if not position:
                    # Если позиции нет, открываем новую
                    balance = _get_balance(client)
                    if balance is None:
                        print(f"[live] ⚠️ Skipping ADD_LONG: failed to get balance")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                    qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_LONG: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📈 Opening LONG position (from ADD_LONG): {qty:.3f} @ ${sig.price:.2f}")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Buy",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] ✅ LONG position opened (from ADD_LONG)")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                        # Сохраняем стратегию, которая открыла позицию
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        position_strategy[symbol] = strategy_type
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        
                        # Сохраняем открытую сделку в историю
                        try:
                            result = resp.get("result", {})
                            order_id = result.get("orderId", "") if result else ""
                            order_link_id = result.get("orderLinkId", "") if result else ""
                            
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            add_trade(
                                entry_time=ts_log,
                                exit_time=None,  # Позиция еще открыта
                                side="long",
                                entry_price=sig.price,
                                exit_price=0.0,
                                size_usd=desired_usd,
                                pnl=0.0,
                                entry_reason=sig.reason,
                                exit_reason="",
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=order_id,
                                order_link_id=order_link_id,
                            )
                            print(f"[live] 💾 Saved open LONG trade to history: {strategy_type.upper()} @ ${sig.price:.2f} ({sig.reason})")
                        except Exception as e:
                            print(f"[live] ⚠️ Failed to save open LONG trade to history: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] [{symbol}] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to open LONG position: {resp.get('retMsg', 'Unknown error')}")
                elif current_position_bias == Bias.LONG:
                    # Проверяем smart add условие
                    if current_settings.risk.enable_smart_add:
                        max_price = position_max_price.get(symbol, sig.price)
                        pullback_pct = ((max_price - sig.price) / max_price) * 100 if max_price > 0 else 0
                        
                        if pullback_pct < current_settings.risk.smart_add_pullback_pct * 100:
                            print(f"[live] ⚠️ Skipping ADD_LONG: pullback too small ({pullback_pct:.2f}% < {current_settings.risk.smart_add_pullback_pct * 100:.2f}%)")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                    
                    # Добавляем к существующей LONG позиции
                    # Рассчитываем количество контрактов как половину от текущего размера позиции
                    current_size = position.get("size", 0)
                    if current_size <= 0:
                        print(f"[live] ⚠️ Skipping ADD_LONG: invalid position size ({current_size})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    qty = _calculate_add_position_qty(client, current_size, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_LONG: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📈 Adding to LONG position: {qty:.3f} (half of {current_size:.3f}) @ ${sig.price:.2f}")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Buy",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] ✅ Added to LONG position successfully")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                    else:
                        print(f"[live] ⚠️ Failed to add to LONG position: {resp.get('retMsg', 'Unknown error')}")
                else:
                    print(f"[live] ⚠️ Skipping ADD_LONG: position is SHORT, not LONG")
            
            elif sig.action == Action.ADD_SHORT:
                if not position:
                    # Если позиции нет, открываем новую
                    balance = _get_balance(client)
                    if balance is None:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: failed to get balance")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    desired_usd = balance * (current_settings.risk.balance_percent_per_trade / 100)
                    qty = _calculate_order_qty(client, sig.price, desired_usd, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📉 Opening SHORT position (from ADD_SHORT): {qty:.3f} @ ${sig.price:.2f}")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Sell",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] ✅ EXECUTED: {strategy_type.upper()} signal {sig.action.value} - SHORT position opened (from ADD_SHORT)")
                        print(f"[live]   Qty: {qty:.3f}, Price: ${sig.price:.2f}")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                        # Сохраняем стратегию, которая открыла позицию
                        position_strategy[symbol] = strategy_type
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        
                        # Сохраняем открытую сделку в историю
                        try:
                            result = resp.get("result", {})
                            order_id = result.get("orderId", "") if result else ""
                            order_link_id = result.get("orderLinkId", "") if result else ""
                            
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            
                            # ВАЛИДАЦИЯ: Убеждаемся, что side соответствует sig.action
                            expected_side = "short" if sig.action == Action.SHORT else "long"
                            if expected_side != "short":
                                print(f"[live] ⚠️ WARNING: sig.action={sig.action.value} but trying to save SHORT position! Using expected_side={expected_side}")
                            
                            add_trade(
                                entry_time=ts_log,
                                exit_time=None,  # Позиция еще открыта
                                side=expected_side,  # ВАЖНО: Используем валидированный side
                                entry_price=sig.price,
                                exit_price=0.0,
                                size_usd=desired_usd,
                                pnl=0.0,
                                entry_reason=sig.reason,
                                exit_reason="",
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=order_id,
                                order_link_id=order_link_id,
                            )
                            print(f"[live] 💾 Saved open {expected_side.upper()} trade to history: {strategy_type.upper()} @ ${sig.price:.2f} ({sig.reason})")
                        except Exception as e:
                            print(f"[live] ⚠️ Failed to save open SHORT trade to history: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] [{symbol}] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to open SHORT position: {resp.get('retMsg', 'Unknown error')}")
                elif current_position_bias == Bias.SHORT:
                    # Проверяем smart add условие
                    if current_settings.risk.enable_smart_add:
                        max_price = position_max_price.get(symbol, sig.price)
                        pullback_pct = ((sig.price - max_price) / max_price) * 100 if max_price > 0 else 0
                        
                        if pullback_pct < current_settings.risk.smart_add_pullback_pct * 100:
                            print(f"[live] ⚠️ Skipping ADD_SHORT: pullback too small ({pullback_pct:.2f}% < {current_settings.risk.smart_add_pullback_pct * 100:.2f}%)")
                            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                                break
                            continue
                    
                    # Добавляем к существующей SHORT позиции
                    # Рассчитываем количество контрактов как половину от текущего размера позиции
                    current_size = position.get("size", 0)
                    if current_size <= 0:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: invalid position size ({current_size})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    qty = _calculate_add_position_qty(client, current_size, current_settings)
                    
                    if qty <= 0:
                        print(f"[live] ⚠️ Skipping ADD_SHORT: invalid qty ({qty})")
                        if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                            break
                        continue
                    
                    print(f"[live] 📉 Adding to SHORT position: {qty:.3f} (half of {current_size:.3f}) @ ${sig.price:.2f}")
                    resp = client.place_order(
                        symbol=symbol,
                        side="Sell",
                        qty=qty,
                    )
                    
                    if resp.get("retCode") == 0:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] ✅ EXECUTED: {strategy_type.upper()} signal {sig.action.value} - SHORT position added successfully")
                        print(f"[live]   Added Qty: {qty:.3f}, Price: ${sig.price:.2f}")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                    else:
                        strategy_type = get_strategy_type_from_signal(sig.reason)
                        print(f"[live] ❌ FAILED: {strategy_type.upper()} signal {sig.action.value} - Failed to add to SHORT position: {resp.get('retMsg', 'Unknown error')}")
                else:
                    print(f"[live] ⚠️ Skipping ADD_SHORT: position is LONG, not SHORT")
            
            elif sig.action == Action.CLOSE:
                if not position:
                    print(f"[live] ⚠️ Skipping CLOSE: no position open")
                else:
                    side = "Sell" if current_position_bias == Bias.LONG else "Buy"
                    resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=position["size"],
                        reduce_only=True,
                    )
                    
                    if resp.get("retCode") == 0:
                        print(f"[live] ✅ Position closed: {sig.reason}")
                        processed_signals.add(signal_id)
                        _save_processed_signals(processed_signals, processed_signals_file)
                        last_handled_signal = (ts, sig.action.value)
                        position_max_profit.pop(symbol, None)
                        position_max_price.pop(symbol, None)
                        position_partial_closed.pop(symbol, None)
                        # Используем сохраненную стратегию, которая открыла позицию
                        strategy_type = position_strategy.pop(symbol, "unknown")
                        _clear_bot_state(symbol)
                        # Если стратегия unknown, пытаемся определить с fallback логикой
                        if strategy_type == "unknown":
                            strategy_type = _determine_strategy_with_fallback(
                                symbol,
                                position_strategy,
                                position,
                                entry_time=datetime.now(timezone.utc),  # Используем текущее время как приближение
                            )
                        try:
                            # Получаем orderId из ответа place_order (ID ордера закрытия)
                            close_order_id = None
                            result = resp.get("result", {})
                            if result:
                                close_order_id = result.get("orderId") or None
                            
                            # Получаем orderLinkId из сохраненных данных (ID открытия позиции)
                            order_link_id = position_order_link_id.pop(symbol, None)
                            
                            # Если orderId не получен из ответа, пытаемся получить из истории исполненных ордеров
                            if not close_order_id:
                                try:
                                    # Получаем историю исполненных ордеров за последние 5 минут
                                    exec_start_time = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)
                                    exec_end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                                    exec_resp = client.get_execution_list(
                                        symbol=symbol,
                                        start_time=exec_start_time,
                                        end_time=exec_end_time,
                                        limit=50,
                                    )
                                    if exec_resp.get("retCode") == 0:
                                        exec_result = exec_resp.get("result", {})
                                        exec_list = exec_result.get("list", [])
                                        # Ищем последний исполненный ордер для закрытия позиции (reduceOnly)
                                        for exec_order in exec_list:
                                            if exec_order.get("reduceOnly") and exec_order.get("execQty"):
                                                close_order_id = exec_order.get("orderId")
                                                if close_order_id:
                                                    break
                                except Exception as e:
                                    print(f"[live] ⚠️ Error getting orderId from execution history: {e}")
                            
                            add_trade(
                                entry_time=datetime.now(),
                                exit_time=datetime.now(),
                                side=current_position_bias.value,
                                entry_price=position.get("avg_price", current_price),
                                exit_price=current_price,
                                size_usd=position["size"] * current_price,
                                pnl=position.get("unrealised_pnl", 0),
                                entry_reason="unknown",
                                exit_reason=sig.reason,
                                strategy_type=strategy_type,
                                symbol=symbol,
                                order_id=close_order_id,
                                order_link_id=order_link_id,
                            )
                        except Exception as e:
                            print(f"[live] Warning: Failed to log trade: {e}")
                    else:
                        print(f"[live] ⚠️ Failed to close position: {resp.get('retMsg', 'Unknown error')}")
            
            # Обновляем статус обратно на "Running" после обработки сигнала
            if bot_state:
                bot_state["current_status"] = "Running"
                bot_state["last_action"] = "Signal processed, waiting..."
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            update_worker_status(symbol, current_status="Running", last_action="Signal processed, waiting...")
            
            # Пауза перед следующей итерацией
            # ВАЖНО: Если обработан свежий сигнал ИЛИ есть свежие сигналы - используем минимальный интервал для немедленной проверки
            # Это гарантирует, что новые сигналы обрабатываются сразу же, как только попадают в историю
            if fresh_signal_processed:
                # Минимальный интервал (1 секунда) для немедленной проверки новых сигналов после обработки свежего
                wait_interval = 1.0
                _log(f"⚡ Fresh signal was processed - using minimal interval ({wait_interval}s) to check for new signals immediately", symbol)
            elif fresh_signals_available:
                # Минимальный интервал (1 секунда) если есть свежие сигналы, но они еще не обработаны
                # Это позволяет обработать их в следующей итерации немедленно, без задержек
                wait_interval = 1.0
                _log(f"⚡ Fresh signals available - using minimal interval ({wait_interval}s) to process them immediately", symbol)
            else:
                # Обычный интервал, если нет свежих сигналов
                wait_interval = current_settings.live_poll_seconds
                _log(f"⏳ No fresh signals - using normal interval ({wait_interval}s)", symbol)
            
            if _wait_with_stop_check(stop_event, wait_interval, symbol):
                break
        
        except KeyboardInterrupt:
            print(f"[live] Bot stopped by user")
            if bot_state:
                bot_state["is_running"] = False
                bot_state["current_status"] = "Stopped"
            break
        except Exception as e:
            print(f"[live] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            if bot_state:
                bot_state["current_status"] = "Error"
                bot_state["last_error"] = str(e)
                bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
            if _wait_with_stop_check(stop_event, current_settings.live_poll_seconds, symbol):
                break
