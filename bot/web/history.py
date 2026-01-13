"""
Модуль для хранения истории сделок и сигналов для веб-админки.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional


HISTORY_FILE = Path(__file__).parent.parent.parent / "bot_history.json"
MAX_TRADES = 1000
MAX_SIGNALS = 5000


def check_recent_loss_trade(
    side: str,
    symbol: str,
    cooldown_minutes: int = 60,
    max_losses: int = 2,
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Проверяет, не было ли недавно убыточных сделок в том же направлении.
    
    Args:
        side: Направление позиции ("long" или "short")
        symbol: Торговая пара
        cooldown_minutes: Период "охлаждения" в минутах после убыточной сделки
        max_losses: Максимальное количество убыточных сделок подряд, после которого блокируем
    
    Returns:
        (should_block, last_loss_trade) - нужно ли блокировать открытие позиции и информация о последней убыточной сделке
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        history = _load_history()
        trades = history.get("trades", [])
        
        if not trades:
            return False, None
        
        # Фильтруем сделки по символу и направлению
        relevant_trades = [
            t for t in trades
            if t.get("symbol", "").upper() == symbol.upper() and t.get("side", "").lower() == side.lower()
        ]
        
        if not relevant_trades:
            return False, None
        
        # Сортируем по времени выхода (последние сначала)
        relevant_trades.sort(key=lambda x: x.get("exit_time", ""), reverse=True)
        
        # Проверяем последние сделки на убыточность
        now = datetime.now(timezone.utc)
        recent_losses = []
        
        for trade in relevant_trades:
            # Парсим время выхода
            exit_time_str = trade.get("exit_time", "")
            if not exit_time_str:
                continue
            
            try:
                if isinstance(exit_time_str, str):
                    # Пытаемся распарсить ISO формат
                    if 'T' in exit_time_str:
                        exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                    else:
                        exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                else:
                    continue
            except Exception:
                continue
            
            # Проверяем, не слишком ли старая сделка
            time_diff = now - exit_time
            if time_diff.total_seconds() > cooldown_minutes * 60:
                break  # Сделки дальше будут еще старше
            
            # Проверяем, была ли сделка убыточной
            pnl = trade.get("pnl", 0)
            exit_reason = trade.get("exit_reason", "").lower()
            
            # Убыточная сделка: отрицательный PnL или закрытие по SL
            is_loss = pnl < 0 or "stop" in exit_reason or "sl" in exit_reason or "loss" in exit_reason
            
            if is_loss:
                recent_losses.append(trade)
        
        # Если было слишком много убыточных сделок подряд - блокируем
        if len(recent_losses) >= max_losses:
            return True, recent_losses[0] if recent_losses else None
        
        # Если была хотя бы одна убыточная сделка в период cooldown - блокируем
        if recent_losses:
            return True, recent_losses[0]
        
        return False, None
    
    except Exception as e:
        print(f"[history] Error checking recent loss trade: {e}")
        return False, None


def _load_history() -> Dict[str, List]:
    """Загрузить историю из файла."""
    if not HISTORY_FILE.exists():
        return {"trades": [], "signals": []}
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            # Автоматически удаляем дубликаты при каждой загрузке для гарантии чистоты данных
            # Это гарантирует, что дубликаты будут удалены даже если они появились после первой очистки
            try:
                duplicates_removed = remove_duplicate_trades_internal(history)
                if duplicates_removed > 0:
                    _save_history(history)
                    print(f"[history] Auto-removed {duplicates_removed} duplicate trades on load")
            except Exception as e:
                print(f"[history] ⚠️ Error removing duplicates on load: {e}")
            return history
    except Exception:
        return {"trades": [], "signals": []}


def remove_duplicate_trades_internal(history: Optional[Dict[str, List]] = None) -> int:
    """Внутренняя функция для удаления дубликатов. Можно вызвать с уже загруженной историей."""
    if history is None:
        history = _load_history()
    
    trades = history.get("trades", [])
    
    if not trades:
        return 0
    
    # Создаем словарь для отслеживания уникальных сделок
    # Ключ: (exit_time_normalized, entry_price, exit_price, pnl, side, symbol)
    seen_trades = {}
    unique_trades = []
    duplicates_removed = 0
    
    for trade in trades:
        # Нормализуем параметры для сравнения
        exit_time_str = str(trade.get("exit_time", ""))
        entry_price = round(float(trade.get("entry_price", 0)), 2)
        exit_price = round(float(trade.get("exit_price", 0)), 2)
        pnl = round(float(trade.get("pnl", 0)), 2)
        side = str(trade.get("side", "")).lower()
        symbol = str(trade.get("symbol", "")).upper()
        
        # Нормализуем время до секунды для сравнения
        # Упрощенный подход: используем первые 19 символов после удаления микросекунд и таймзоны
        exit_time_normalized = ""
        if exit_time_str:
            try:
                time_str_clean = str(exit_time_str).strip()
                
                # Удаляем микросекунды (все после точки)
                if '.' in time_str_clean:
                    time_str_clean = time_str_clean.split('.')[0]
                
                # Удаляем таймзону (+HH:MM, -HH:MM, Z)
                if '+' in time_str_clean:
                    time_str_clean = time_str_clean.split('+')[0]
                elif 'Z' in time_str_clean:
                    time_str_clean = time_str_clean.split('Z')[0]
                elif '-' in time_str_clean[10:] and time_str_clean.count('-') > 2:
                    # Формат с таймзоной: 2026-01-09T20:29:27-05:00
                    # Ищем 'T' и берем 19 символов (YYYY-MM-DDTHH:MM:SS)
                    t_idx = time_str_clean.find('T')
                    if t_idx > 0 and t_idx == 10:
                        # Стандартный формат: берем 19 символов
                        time_str_clean = time_str_clean[:19]
                    elif t_idx > 0:
                        # Нестандартный формат, но есть 'T'
                        time_str_clean = time_str_clean[:19] if len(time_str_clean) >= 19 else time_str_clean
                    else:
                        # Нет 'T', пробуем обрезать до 19 символов
                        time_str_clean = time_str_clean[:19] if len(time_str_clean) >= 19 else time_str_clean
                
                # Финальная обрезка до 19 символов (YYYY-MM-DDTHH:MM:SS)
                exit_time_normalized = time_str_clean[:19] if len(time_str_clean) >= 19 else time_str_clean
            except Exception as e:
                # В случае ошибки, просто обрезаем до 19 символов
                exit_time_normalized = str(exit_time_str)[:19] if len(str(exit_time_str)) >= 19 else str(exit_time_str)
        
        # Создаем ключ для сравнения
        trade_key = (
            exit_time_normalized,
            entry_price,
            exit_price,
            pnl,
            side,
            symbol,
        )
        
        # Проверяем, видели ли мы эту сделку раньше
        if trade_key in seen_trades:
            # Дубликат найден - пропускаем, но обновляем strategy_type если нужно
            existing_trade = seen_trades[trade_key]
            existing_strategy = existing_trade.get("strategy_type", "unknown")
            new_strategy = trade.get("strategy_type", "unknown")
            
            # Если существующая сделка имеет unknown, а новая - нет, обновляем
            if existing_strategy == "unknown" and new_strategy != "unknown":
                existing_trade["strategy_type"] = new_strategy
                print(f"[history] Updated strategy_type for duplicate: {new_strategy} (time: {exit_time_normalized})")
            
            duplicates_removed += 1
            # Не добавляем дубликат в unique_trades
        else:
            # Уникальная сделка - добавляем
            seen_trades[trade_key] = trade
            unique_trades.append(trade)
    
    # Обновляем историю
    if duplicates_removed > 0:
        history["trades"] = unique_trades
    
    return duplicates_removed


def _save_history(history: Dict[str, List]):
    """Сохранить историю в файл."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[history] Error saving history: {e}")


def add_signal(action: str, reason: str, price: float, timestamp: Any = None, symbol: str = "", strategy_type: str = "unknown", signal_id: Optional[str] = None):
    """Добавить сигнал в историю с дедупликацией."""
    history = _load_history()
    
    # Нормализуем timestamp в UTC
    if timestamp is None:
        ts_str = datetime.now(timezone.utc).isoformat()
    elif hasattr(timestamp, 'isoformat'):
        # Если это pandas Timestamp или datetime
        try:
            import pandas as pd
            if isinstance(timestamp, pd.Timestamp):
                # pandas Timestamp - конвертируем в UTC
                if timestamp.tz is None:
                    ts_str = timestamp.tz_localize('UTC').isoformat()
                else:
                    ts_str = timestamp.tz_convert('UTC').isoformat()
            else:
                # datetime
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
                    ts_str = timestamp.replace(tzinfo=timezone.utc).isoformat()
                elif hasattr(timestamp, 'astimezone'):
                    ts_str = timestamp.astimezone(timezone.utc).isoformat()
                else:
                    ts_str = timestamp.isoformat()
        except ImportError:
            # Если pandas не доступен, используем стандартный метод
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
                ts_str = timestamp.replace(tzinfo=timezone.utc).isoformat()
            elif hasattr(timestamp, 'astimezone'):
                ts_str = timestamp.astimezone(timezone.utc).isoformat()
            else:
                ts_str = timestamp.isoformat()
    else:
        ts_str = str(timestamp)
    
    # Нормализуем price для сравнения (округление до 2 знаков)
    price_normalized = round(float(price), 2)
    
    # Проверяем на дубликаты: ищем сигнал с таким же timestamp, reason и symbol
    # Если найден такой сигнал, заменяем его на новый (latest сигнал с обновленной ценой)
    # Используем нормализованные значения для сравнения
    signal_index_to_replace = None
    for idx, existing_signal in enumerate(history["signals"]):
        existing_ts = existing_signal.get("timestamp", "")
        existing_reason = existing_signal.get("reason", "")
        existing_symbol = existing_signal.get("symbol", "")
        existing_strategy = existing_signal.get("strategy_type", "")
        
        # Сравниваем timestamp (может быть в разных форматах, поэтому сравниваем строки)
        # Сравниваем первые 16 символов (YYYY-MM-DD HH:MM), чтобы игнорировать секунды и микросекунды
        ts_match = (existing_ts == ts_str or 
                   (len(existing_ts) >= 16 and len(ts_str) >= 16 and existing_ts[:16] == ts_str[:16]) or
                   existing_ts.startswith(ts_str[:16]) or 
                   ts_str.startswith(existing_ts[:16]))
        
        # Сравниваем reason, symbol и strategy_type (не сравниваем цену, так как latest сигнал может иметь обновленную цену)
        if (ts_match and 
            existing_reason == reason and 
            existing_symbol == symbol and
            existing_strategy == strategy_type):
            # Найден сигнал с таким же timestamp, reason, symbol и strategy - это latest сигнал, заменяем его
            signal_index_to_replace = idx
            break
    
    # Если найден сигнал для замены, удаляем старый
    if signal_index_to_replace is not None:
        old_signal = history["signals"].pop(signal_index_to_replace)
        old_price = old_signal.get("price", 0.0)
        # Если цена не изменилась, это полный дубликат, не добавляем новый
        if round(old_price, 2) == price_normalized:
            return
    
    # Генерируем signal_id если он не передан
    if not signal_id:
        import hashlib
        ts_str_for_id = ts_str
        # Нормализуем timestamp для генерации ID (убираем микросекунды и таймзону для совместимости)
        if '.' in ts_str_for_id:
            ts_str_for_id = ts_str_for_id.split('.')[0]
        if '+' in ts_str_for_id:
            ts_str_for_id = ts_str_for_id.split('+')[0]
        elif 'Z' in ts_str_for_id:
            ts_str_for_id = ts_str_for_id.replace('Z', '')
        id_string = f"{ts_str_for_id}_{action}_{reason}_{price_normalized:.4f}"
        signal_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    signal = {
        "timestamp": ts_str,
        "action": action,
        "reason": reason,
        "price": price_normalized,
        "symbol": symbol,
        "strategy_type": strategy_type,
        "signal_id": signal_id,  # Сохраняем signal_id для восстановления стратегии
    }
    
    history["signals"].append(signal)
    
    # Ограничиваем размер истории
    if len(history["signals"]) > MAX_SIGNALS:
        history["signals"] = history["signals"][-MAX_SIGNALS:]
    
    _save_history(history)


def add_trade(
    entry_time: Any,
    exit_time: Any,
    side: str,
    entry_price: float,
    exit_price: float,
    size_usd: float,
    pnl: float,
    entry_reason: str = "",
    exit_reason: str = "",
    strategy_type: str = "unknown",  # "trend", "flat", "ml", "hybrid"
    symbol: str = "",  # Торговая пара
    order_id: Optional[str] = None,  # ID ордера от Bybit
    order_link_id: Optional[str] = None,  # Custom ID ордера
):
    """Добавить сделку в историю с проверкой на дубликаты."""
    history = _load_history()
    
    # Нормализуем время выхода для сравнения
    exit_time_str = exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time) if exit_time else None
    entry_time_str = entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time) if entry_time else None
    
    # Нормализуем цены для сравнения (округление до 2 знаков)
    entry_price_normalized = round(float(entry_price), 2)
    exit_price_normalized = round(float(exit_price), 2)
    pnl_normalized = round(float(pnl), 2)
    size_usd_normalized = round(float(size_usd), 2)
    
    # Проверяем на дубликаты: ищем существующую сделку с теми же параметрами
    for existing_trade in history.get("trades", []):
        existing_exit_time = existing_trade.get("exit_time", "")
        existing_entry_price = round(float(existing_trade.get("entry_price", 0)), 2)
        existing_exit_price = round(float(existing_trade.get("exit_price", 0)), 2)
        existing_pnl = round(float(existing_trade.get("pnl", 0)), 2)
        existing_side = existing_trade.get("side", "").lower()
        existing_symbol = existing_trade.get("symbol", "").upper()
        
        # Сравниваем ключевые параметры
        # Время выхода должно совпадать с точностью до секунды (первые 19 символов ISO формата)
        time_match = False
        if exit_time_str and existing_exit_time:
            # Сравниваем первые 19 символов (YYYY-MM-DDTHH:MM:SS) для точности до секунды
            exit_time_short = exit_time_str[:19] if len(exit_time_str) >= 19 else exit_time_str
            existing_exit_time_short = existing_exit_time[:19] if len(existing_exit_time) >= 19 else existing_exit_time
            time_match = exit_time_short == existing_exit_time_short
        
        # Проверяем совпадение всех ключевых параметров
        if (time_match and
            existing_entry_price == entry_price_normalized and
            existing_exit_price == exit_price_normalized and
            existing_pnl == pnl_normalized and
            existing_side.lower() == side.lower() and
            existing_symbol == symbol.upper()):
            # Дубликат найден, не добавляем, но обновляем strategy_type если он был unknown
            if existing_trade.get("strategy_type", "unknown") == "unknown" and strategy_type != "unknown":
                existing_trade["strategy_type"] = strategy_type
                _save_history(history)
                print(f"[history] Updated strategy_type for duplicate trade: {strategy_type}")
            return  # Дубликат, не добавляем
    
    # Создаем новую сделку
    trade = {
        "entry_time": entry_time_str,
        "exit_time": exit_time_str,
        "side": side,
        "entry_price": entry_price_normalized,
        "exit_price": exit_price_normalized,
        "size_usd": size_usd_normalized,
        "pnl": pnl_normalized,
        "entry_reason": entry_reason,
        "exit_reason": exit_reason,
        "strategy_type": strategy_type,  # Тип стратегии
        "symbol": symbol,  # Торговая пара
        "order_id": order_id,  # ID ордера от Bybit
        "order_link_id": order_link_id,  # Custom ID ордера
    }
    
    history["trades"].append(trade)
    
    # Ограничиваем размер истории
    if len(history["trades"]) > MAX_TRADES:
        history["trades"] = history["trades"][-MAX_TRADES:]
    
    _save_history(history)


def get_trades(limit: int = 50, strategy_filter: Optional[str] = None, symbol_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Получить последние сделки, отсортированные от новых к старым (по времени выхода).
    
    Args:
        limit: Максимальное количество сделок
        strategy_filter: Фильтр по типу стратегии (trend, flat, ml, hybrid, unknown)
        symbol_filter: Фильтр по символу (BTCUSDT, ETHUSDT, SOLUSDT)
    
    Returns:
        Список сделок, отсортированных от новых к старым
    """
    history = _load_history()
    trades = history.get("trades", [])
    
    # Фильтр по типу стратегии
    if strategy_filter:
        trades = [t for t in trades if t.get("strategy_type") == strategy_filter]
    
    # Фильтр по символу
    if symbol_filter:
        trades = [t for t in trades if t.get("symbol", "").upper() == symbol_filter.upper()]
    
    # Сортируем по времени выхода (exit_time) от новых к старым (по убыванию)
    def get_exit_time(trade):
        """Извлекает время выхода для сортировки."""
        exit_time_str = trade.get("exit_time", "")
        if not exit_time_str:
            return datetime.min.replace(tzinfo=timezone.utc)  # Сделки без времени выхода в конец
        
        try:
            # Парсим время в разных форматах
            if isinstance(exit_time_str, str):
                if 'T' in exit_time_str:
                    # ISO формат: 2026-01-09T20:29:27.916000+00:00
                    dt = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                else:
                    # Другой формат: 2026-01-09 20:29:27
                    dt = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                    dt = dt.replace(tzinfo=timezone.utc)
                
                # Убеждаемся, что время в UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                
                return dt
        except Exception as e:
            # Если не удалось распарсить, возвращаем минимальное время (в конец списка)
            return datetime.min.replace(tzinfo=timezone.utc)
        
        return datetime.min.replace(tzinfo=timezone.utc)
    
    # Сортируем по времени выхода по убыванию (от новых к старым)
    trades_sorted = sorted(trades, key=get_exit_time, reverse=True)
    
    # Ограничиваем количество
    if limit:
        trades_sorted = trades_sorted[:limit]
    
    return trades_sorted


def get_signals(limit: int = 100, symbol_filter: Optional[str] = None, include_smc: bool = True) -> List[Dict[str, Any]]:
    """
    Получить последние сигналы.
    
    Args:
        limit: Максимальное количество сигналов
        symbol_filter: Фильтр по символу (если None, возвращаются все сигналы)
        include_smc: Включать ли SMC сигналы из CSV (по умолчанию True)
    
    Returns:
        Список сигналов, отсортированных от новых к старым
    """
    history = _load_history()
    signals = history.get("signals", [])
    
    # Добавляем SMC сигналы из CSV файла
    if include_smc:
        try:
            smc_signals = get_smc_history(limit=500)  # Получаем больше для сортировки
            for smc_sig in smc_signals:
                # Конвертируем SMC сигнал в формат общей истории
                signals.append({
                    "timestamp": smc_sig.get("timestamp", ""),
                    "action": smc_sig.get("action", "").lower(),
                    "reason": smc_sig.get("reason", ""),
                    "price": float(smc_sig.get("price", 0)),
                    "symbol": smc_sig.get("symbol", ""),
                    "strategy_type": "smc",
                    "stop_loss": smc_sig.get("stop_loss"),
                    "take_profit": smc_sig.get("take_profit"),
                    "rr_ratio": smc_sig.get("rr_ratio"),
                })
        except Exception as e:
            print(f"[history] Error loading SMC signals: {e}")
    
    # Фильтруем по символу, если указан
    if symbol_filter:
        signals = [s for s in signals if s.get("symbol", "").upper() == symbol_filter.upper()]
    
    # Сортируем сигналы по timestamp (от новых к старым)
    def get_timestamp(signal):
        """Извлекает timestamp для сортировки."""
        ts_str = signal.get("timestamp", "")
        if not ts_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        try:
            if isinstance(ts_str, str):
                if 'T' in ts_str:
                    # ISO формат
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                else:
                    # Другой формат
                    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    dt = dt.replace(tzinfo=timezone.utc)
                
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                
                return dt
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        return datetime.min.replace(tzinfo=timezone.utc)
    
    # Удаляем дубликаты по timestamp + symbol + action
    seen = set()
    unique_signals = []
    for sig in signals:
        key = (sig.get("timestamp", ""), sig.get("symbol", ""), sig.get("action", ""))
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)
    
    # Сортируем по timestamp по убыванию (от новых к старым)
    signals_sorted = sorted(unique_signals, key=get_timestamp, reverse=True)
    
    # Возвращаем последние limit сигналов
    return signals_sorted[:limit] if signals_sorted else []


def clear_signals():
    """Очистить всю историю сигналов."""
    history = _load_history()
    history["signals"] = []
    _save_history(history)
    print(f"[history] All signals cleared")


def clear_trades():
    """Очистить всю историю сделок."""
    history = _load_history()
    history["trades"] = []
    _save_history(history)
    print(f"[history] All trades cleared")


def clear_all_history():
    """Очистить всю историю (сигналы и сделки)."""
    history = _load_history()
    history["signals"] = []
    history["trades"] = []
    _save_history(history)
    print(f"[history] All history cleared")


def remove_duplicate_trades() -> int:
    """Удалить дубликаты сделок из истории. Возвращает количество удаленных дубликатов."""
    history = _load_history()
    duplicates_removed = remove_duplicate_trades_internal(history)
    
    # Сохраняем очищенную историю
    if duplicates_removed > 0:
        _save_history(history)
        remaining = len(history.get('trades', []))
        print(f"[history] ✅ Removed {duplicates_removed} duplicate trades. Remaining: {remaining}")
    
    return duplicates_removed


def get_pnl_stats(symbol: Optional[str] = None) -> Dict[str, float]:
    """
    Получить статистику PnL: сегодня, за неделю, за месяц, общий по паре.
    
    Args:
        symbol: Торговая пара для фильтрации (если None, берет все пары)
    
    Returns:
        Словарь с ключами: pnl_today, pnl_week, pnl_month, pnl_total
    """
    history = _load_history()
    trades = history.get("trades", [])
    
    # Фильтруем по символу если указан
    if symbol:
        trades = [t for t in trades if t.get("symbol", "").upper() == symbol.upper()]
    
    if not trades:
        return {
            "pnl_today": 0.0,
            "pnl_week": 0.0,
            "pnl_month": 0.0,
            "pnl_total": 0.0,
        }
    
    # Текущее время в UTC
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)
    
    pnl_today = 0.0
    pnl_week = 0.0
    pnl_month = 0.0
    pnl_total = 0.0
    
    for trade in trades:
        exit_time_str = trade.get("exit_time", "")
        if not exit_time_str:
            # Если нет времени выхода, добавляем к общему
            pnl_total += float(trade.get("pnl", 0))
            continue
        
        try:
            # Парсим время выхода
            if isinstance(exit_time_str, str):
                if 'T' in exit_time_str:
                    exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                else:
                    try:
                        exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                    except ValueError:
                        # Попробуем другой формат
                        exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
            else:
                exit_time = datetime.fromisoformat(str(exit_time_str).replace('Z', '+00:00'))
            
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=timezone.utc)
            else:
                exit_time = exit_time.astimezone(timezone.utc)
            
            pnl = float(trade.get("pnl", 0))
            
            # Суммируем PnL по периодам
            pnl_total += pnl
            
            if exit_time >= month_start:
                pnl_month += pnl
                
                if exit_time >= week_start:
                    pnl_week += pnl
                    
                    if exit_time >= today_start:
                        pnl_today += pnl
        
        except Exception as e:
            # Если не удалось распарсить время, просто добавляем к общему
            pnl_total += float(trade.get("pnl", 0))
            continue
    
    return {
        "pnl_today": round(pnl_today, 2),
        "pnl_week": round(pnl_week, 2),
        "pnl_month": round(pnl_month, 2),
        "pnl_total": round(pnl_total, 2),
    }


def get_all_symbols_pnl_stats(active_symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Получить статистику PnL для всех активных символов.
    
    Args:
        active_symbols: Список активных символов
    
    Returns:
        Словарь {symbol: {pnl_today, pnl_week, pnl_month, pnl_total}}
    """
    all_stats = {}
    
    for symbol in active_symbols:
        stats = get_pnl_stats(symbol=symbol)
        all_stats[symbol] = stats
    
    return all_stats


def get_combined_pnl_stats(active_symbols: List[str]) -> Dict[str, float]:
    """
    Получить сводную статистику PnL по всем активным символам.
    
    Args:
        active_symbols: Список активных символов
    
    Returns:
        Словарь с суммарной статистикой: {pnl_today, pnl_week, pnl_month, pnl_total}
    """
    all_stats = get_all_symbols_pnl_stats(active_symbols)
    
    combined = {
        "pnl_today": 0.0,
        "pnl_week": 0.0,
        "pnl_month": 0.0,
        "pnl_total": 0.0,
    }
    
    for symbol, stats in all_stats.items():
        combined["pnl_today"] += stats.get("pnl_today", 0.0)
        combined["pnl_week"] += stats.get("pnl_week", 0.0)
        combined["pnl_month"] += stats.get("pnl_month", 0.0)
        combined["pnl_total"] += stats.get("pnl_total", 0.0)
    
    return combined


def get_strategy_stats(strategy_type: Optional[str] = None) -> Dict[str, Any]:
    """Получить статистику по стратегии(ям)."""
    history = _load_history()
    trades = history["trades"]
    
    # Фильтр по типу стратегии
    if strategy_type:
        trades = [t for t in trades if t.get("strategy_type") == strategy_type]
    
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    avg_pnl = total_pnl / len(trades) if trades else 0.0
    avg_win = sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": (len(winning_trades) / len(trades) * 100) if trades else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def get_smc_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Получить историю сигналов SMC из CSV файла.
    
    Args:
        limit: Максимальное количество записей
        
    Returns:
        Список словарей с данными сигналов
    """
    import csv
    file_path = Path(__file__).parent.parent.parent / "smc_trade_history.csv"
    
    if not file_path.exists():
        return []
        
    history = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
                
        # Сортируем от новых к старым (последние записи в конце файла)
        history.reverse()
        
        if limit:
            history = history[:limit]
            
        return history
    except Exception as e:
        print(f"[history] Error reading SMC history: {e}")
        return []

