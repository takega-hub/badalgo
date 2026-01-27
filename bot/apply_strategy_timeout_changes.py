#!/usr/bin/env python3
"""
Скрипт для применения изменений таймаутов стратегий в рабочую папку.
Запустите этот скрипт из корня проекта crypto_bot.
"""
import os
import shutil
from pathlib import Path

# Определяем пути
# Если передан аргумент командной строки - используем его как рабочую папку
import sys
if len(sys.argv) > 1:
    WORK_DIR = Path(sys.argv[1])
else:
    WORK_DIR = Path(__file__).parent
BOT_DIR = WORK_DIR / "bot"

print(f"Рабочая директория: {WORK_DIR}")
print(f"Директория bot: {BOT_DIR}")

# 1. Создаем bot/strategy_timeout.py
strategy_timeout_content = '''"""
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
'''

strategy_timeout_file = BOT_DIR / "strategy_timeout.py"
print(f"\n1. Создаю {strategy_timeout_file}...")
strategy_timeout_file.write_text(strategy_timeout_content, encoding='utf-8')
print(f"   ✅ Создан {strategy_timeout_file}")

# 2. Обновляем bot/live.py - добавляем импорт
live_file = BOT_DIR / "live.py"
if live_file.exists():
    print(f"\n2. Обновляю {live_file}...")
    content = live_file.read_text(encoding='utf-8')
    
    # Добавляем импорт после строки с check_recent_loss_trade
    if "from bot.strategy_timeout import" not in content:
        content = content.replace(
            "from bot.web.history import add_signal, add_trade, check_recent_loss_trade",
            "from bot.web.history import add_signal, add_trade, check_recent_loss_trade\nfrom bot.strategy_timeout import check_strategy_timeout"
        )
        print("   ✅ Добавлен импорт check_strategy_timeout")
    else:
        print("   ⚠️ Импорт уже существует")
    
    # Добавляем функцию is_strategy_in_timeout перед генерацией сигналов
    if "def is_strategy_in_timeout" not in content:
        # Ищем место после symbol_strategy_settings = current_settings.get_strategy_settings_for_symbol(symbol)
        marker = "symbol_strategy_settings = current_settings.get_strategy_settings_for_symbol(symbol)"
        if marker in content:
            insert_pos = content.find(marker) + len(marker)
            # Ищем следующую строку с комментарием или кодом
            next_line = content.find("\n            # Trend", insert_pos)
            if next_line == -1:
                next_line = content.find("\n            # Flat", insert_pos)
            if next_line == -1:
                next_line = content.find("\n            if", insert_pos)
            
            if next_line > insert_pos:
                function_code = '''
            # Вспомогательная функция для проверки таймаута стратегии
            def is_strategy_in_timeout(strategy_type: str) -> bool:
                """Проверяет, находится ли стратегия в таймауте."""
                is_timeout, timeout_until = check_strategy_timeout(strategy_type, symbol)
                if is_timeout:
                    _log(f"⏸️ Strategy {strategy_type.upper()} is in timeout until {timeout_until.isoformat() if timeout_until else 'unknown'}", symbol)
                return is_timeout

'''
                content = content[:next_line] + function_code + content[next_line:]
                print("   ✅ Добавлена функция is_strategy_in_timeout")
            else:
                print("   ⚠️ Не найдено место для вставки функции")
        else:
            print("   ⚠️ Не найдена строка с symbol_strategy_settings")
    else:
        print("   ⚠️ Функция is_strategy_in_timeout уже существует")
    
    # Добавляем проверки таймаута для всех стратегий
    replacements = [
        ('if symbol_strategy_settings.enable_trend_strategy or symbol_strategy_settings.enable_momentum_strategy:', 
         'if (symbol_strategy_settings.enable_trend_strategy or symbol_strategy_settings.enable_momentum_strategy) and not is_strategy_in_timeout("trend"):'),
        ('if symbol_strategy_settings.enable_flat_strategy:', 
         'if symbol_strategy_settings.enable_flat_strategy and not is_strategy_in_timeout("flat"):'),
        ('if symbol_strategy_settings.enable_ml_strategy and current_settings.ml_model_path:', 
         'if symbol_strategy_settings.enable_ml_strategy and current_settings.ml_model_path and not is_strategy_in_timeout("ml"):'),
        ('if symbol_strategy_settings.enable_smc_strategy:', 
         'if symbol_strategy_settings.enable_smc_strategy and not is_strategy_in_timeout("smc"):'),
        ('if symbol_strategy_settings.enable_ict_strategy:', 
         'if symbol_strategy_settings.enable_ict_strategy and not is_strategy_in_timeout("ict"):'),
        ('if symbol_strategy_settings.enable_zscore_strategy:', 
         'if symbol_strategy_settings.enable_zscore_strategy and not is_strategy_in_timeout("zscore"):'),
        ('if symbol_strategy_settings.enable_vbo_strategy:', 
         'if symbol_strategy_settings.enable_vbo_strategy and not is_strategy_in_timeout("vbo"):'),
        ('if symbol_strategy_settings.enable_amt_of_strategy:', 
         'if symbol_strategy_settings.enable_amt_of_strategy and not is_strategy_in_timeout("amt_of"):'),
    ]
    
    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            print(f"   ✅ Добавлена проверка таймаута для стратегии")
        elif new in content:
            print(f"   ⚠️ Проверка таймаута уже добавлена")
    
    live_file.write_text(content, encoding='utf-8')
    print(f"   ✅ Обновлен {live_file}")
else:
    print(f"\n2. ⚠️ Файл {live_file} не найден")

# 3. Обновляем bot/web/history.py
history_file = BOT_DIR / "web" / "history.py"
if history_file.exists():
    print(f"\n3. Обновляю {history_file}...")
    content = history_file.read_text(encoding='utf-8')
    
    # Добавляем инициализацию strategy_timeouts
    if 'history.setdefault("strategy_timeouts", {})' not in content:
        content = content.replace(
            '    # Гарантируем наличие ключей\n    history.setdefault("trades", [])\n    history.setdefault("signals", [])\n    return history',
            '    # Гарантируем наличие ключей\n    history.setdefault("trades", [])\n    history.setdefault("signals", [])\n    history.setdefault("strategy_timeouts", {})\n    return history'
        )
        print("   ✅ Добавлена инициализация strategy_timeouts")
    else:
        print("   ⚠️ Инициализация strategy_timeouts уже существует")
    
    # Добавляем проверку убыточных сделок в add_trade
    if 'from bot.strategy_timeout import check_consecutive_losses, set_strategy_timeout' not in content:
        old_code = '''    # Ограничиваем размер истории
    if len(history["trades"]) > MAX_TRADES:
        history["trades"] = history["trades"][-MAX_TRADES:]
    
    # Сохраняем историю в файл (без лишнего логирования)
    _save_history(history)'''
        
        new_code = '''    # Ограничиваем размер истории
    if len(history["trades"]) > MAX_TRADES:
        history["trades"] = history["trades"][-MAX_TRADES:]
    
    # Проверяем, была ли сделка убыточной и нужно ли установить таймаут
    # Только для закрытых сделок (exit_time не None)
    if exit_time_str and strategy_type != "unknown":
        pnl = pnl_normalized
        exit_reason_lower = exit_reason.lower()
        is_loss = pnl < 0 or "stop" in exit_reason_lower or "sl" in exit_reason_lower or "loss" in exit_reason_lower
        
        if is_loss:
            # Импортируем функции таймаута
            from bot.strategy_timeout import check_consecutive_losses, set_strategy_timeout
            
            # Проверяем, было ли две убыточные сделки подряд в течение 2 часов
            if check_consecutive_losses(strategy_type, symbol, lookback_hours=2.0, max_losses=2):
                # Устанавливаем таймаут на 2 часа
                set_strategy_timeout(strategy_type, symbol, timeout_hours=2.0)
    
    # Сохраняем историю в файл (без лишнего логирования)
    _save_history(history)'''
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            print("   ✅ Добавлена проверка убыточных сделок в add_trade")
        else:
            print("   ⚠️ Не найдено место для вставки проверки убыточных сделок")
    else:
        print("   ⚠️ Проверка убыточных сделок уже добавлена")
    
    history_file.write_text(content, encoding='utf-8')
    print(f"   ✅ Обновлен {history_file}")
else:
    print(f"\n3. ⚠️ Файл {history_file} не найден")

print("\n✅ Все изменения применены!")
print("\nСледующие файлы были изменены/созданы:")
print(f"  - {strategy_timeout_file}")
print(f"  - {live_file}")
print(f"  - {history_file}")
