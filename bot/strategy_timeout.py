"""
Модуль для управления таймаутами стратегий.
Если по одной и той же стратегии два раза подряд в течение 2 часов сделки приносили убыток,
то стратегия ставится в таймаут на 2 часа.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from bot.web.history import _load_history, _save_history, HISTORY_FILE

_history_lock = threading.Lock()


def check_strategy_timeout(
    strategy_type: str,
    symbol: str,
) -> tuple[bool, Optional[datetime]]:
    """
    Проверяет, находится ли стратегия в таймауте.
    
    Args:
        strategy_type: Тип стратегии ("ml", "trend", "flat", "smc", "ict", "zscore", "vbo", "amt_of")
        symbol: Торговая пара (например, "BTCUSDT")
    
    Returns:
        (is_timeout, timeout_until) - находится ли стратегия в таймауте и до какого времени
    """
    try:
        history = _load_history()
        timeouts = history.get("strategy_timeouts", {})
        
        symbol_timeouts = timeouts.get(symbol.upper(), {})
        timeout_until_str = symbol_timeouts.get(strategy_type.lower())
        
        if not timeout_until_str:
            return False, None
        
        # Парсим время таймаута
        try:
            if isinstance(timeout_until_str, str):
                if 'T' in timeout_until_str:
                    timeout_until = datetime.fromisoformat(timeout_until_str.replace('Z', '+00:00'))
                else:
                    timeout_until = datetime.strptime(timeout_until_str, '%Y-%m-%d %H:%M:%S')
                    timeout_until = timeout_until.replace(tzinfo=timezone.utc)
            else:
                return False, None
        except Exception:
            # Если не удалось распарсить, считаем таймаут истекшим
            return False, None
        
        # Проверяем, не истек ли таймаут
        now = datetime.now(timezone.utc)
        if now >= timeout_until:
            # Таймаут истек, удаляем его
            symbol_timeouts.pop(strategy_type.lower(), None)
            if not symbol_timeouts:
                timeouts.pop(symbol.upper(), None)
            history["strategy_timeouts"] = timeouts
            _save_history(history)
            return False, None
        
        # Таймаут активен
        remaining_minutes = (timeout_until - now).total_seconds() / 60
        print(f"[strategy_timeout] ⏸️ Strategy {strategy_type.upper()} for {symbol} is in timeout until {timeout_until.isoformat()} ({remaining_minutes:.1f} min remaining)")
        return True, timeout_until
    
    except Exception as e:
        print(f"[strategy_timeout] Error checking strategy timeout: {e}")
        return False, None


def set_strategy_timeout(
    strategy_type: str,
    symbol: str,
    timeout_hours: float = 2.0,
) -> None:
    """
    Устанавливает таймаут для стратегии.
    
    Args:
        strategy_type: Тип стратегии
        symbol: Торговая пара
        timeout_hours: Длительность таймаута в часах (по умолчанию 2 часа)
    """
    try:
        history = _load_history()
        if "strategy_timeouts" not in history:
            history["strategy_timeouts"] = {}
        
        timeouts = history["strategy_timeouts"]
        if symbol.upper() not in timeouts:
            timeouts[symbol.upper()] = {}
        
        timeout_until = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)
        timeouts[symbol.upper()][strategy_type.lower()] = timeout_until.isoformat()
        
        _save_history(history)
        print(f"[strategy_timeout] ⏸️ Strategy {strategy_type.upper()} for {symbol} set to timeout until {timeout_until.isoformat()} ({timeout_hours}h)")
    
    except Exception as e:
        print(f"[strategy_timeout] Error setting strategy timeout: {e}")


def check_consecutive_losses(
    strategy_type: str,
    symbol: str,
    lookback_hours: float = 2.0,
    max_losses: int = 2,
) -> bool:
    """
    Проверяет, было ли две убыточные сделки подряд в течение указанного периода.
    
    Args:
        strategy_type: Тип стратегии
        symbol: Торговая пара
        lookback_hours: Период для проверки в часах (по умолчанию 2 часа)
        max_losses: Количество убыточных сделок подряд для установки таймаута (по умолчанию 2)
    
    Returns:
        True, если было две убыточные сделки подряд в течение периода
    """
    try:
        history = _load_history()
        trades = history.get("trades", [])
        
        if not trades:
            return False
        
        # Фильтруем сделки по стратегии и символу
        relevant_trades = [
            t for t in trades
            if (t.get("strategy_type", "").lower() == strategy_type.lower() and
                t.get("symbol", "").upper() == symbol.upper() and
                t.get("exit_time"))  # Только закрытые сделки
        ]
        
        if len(relevant_trades) < max_losses:
            return False
        
        # Сортируем по времени выхода (последние сначала)
        relevant_trades.sort(key=lambda x: x.get("exit_time", ""), reverse=True)
        
        # Проверяем последние сделки на убыточность
        now = datetime.now(timezone.utc)
        lookback_start = now - timedelta(hours=lookback_hours)
        consecutive_losses = []
        
        for trade in relevant_trades:
            # Парсим время выхода
            exit_time_str = trade.get("exit_time", "")
            if not exit_time_str:
                continue
            
            try:
                if isinstance(exit_time_str, str):
                    if 'T' in exit_time_str:
                        exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                    else:
                        exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                else:
                    continue
            except Exception:
                continue
            
            # Проверяем, попадает ли сделка в период проверки
            if exit_time < lookback_start:
                break  # Сделки дальше будут еще старше
            
            # Проверяем, была ли сделка убыточной
            pnl = trade.get("pnl", 0)
            exit_reason = trade.get("exit_reason", "").lower()
            
            is_loss = pnl < 0 or "stop" in exit_reason or "sl" in exit_reason or "loss" in exit_reason
            
            if is_loss:
                consecutive_losses.append(trade)
            else:
                # Если встретили прибыльную сделку, прерываем последовательность убытков
                break
        
        # Если было две убыточные сделки подряд в течение периода
        if len(consecutive_losses) >= max_losses:
            print(f"[strategy_timeout] ⚠️ Found {len(consecutive_losses)} consecutive losses for {strategy_type.upper()} on {symbol} within {lookback_hours}h")
            return True
        
        return False
    
    except Exception as e:
        print(f"[strategy_timeout] Error checking consecutive losses: {e}")
        return False
