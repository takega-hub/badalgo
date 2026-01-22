#!/usr/bin/env python3
"""
Генератор сводного отчета по всем стратегиям
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Ensure stdout encoding supports utf-8 on Windows consoles to avoid UnicodeEncodeError
try:
    # Python 3.7+ supports reconfigure
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    try:
        # Fallback: wrap stdout buffer with TextIOWrapper set to utf-8
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If even that fails, continue without raising — prints may replace unsupported chars
        pass

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, generate_trend_signal, generate_range_signal, generate_momentum_breakout_signal, detect_market_phase, MarketPhase, Bias, Action, Signal, build_signals
from bot.smc_strategy import build_smc_signals
from bot.ict_strategy import build_ict_signals
try:
    from bot.ml.strategy_ml import build_ml_signals
except Exception:
    build_ml_signals = None
from bot.liquidation_hunter_strategy import build_liquidation_hunter_signals
from bot.zscore_strategy import build_zscore_signals
from bot.vbo_strategy import build_vbo_signals
from bot.simulation import Simulator

def simulate_trading(df: pd.DataFrame, signals: List[Signal], fee: float = 0.05) -> List[float]:
    if not signals:
        return []

    results = []
    active_pos = None
    cooldown_until_idx = -1
    
    # Индексируем сигналы по времени для быстрого поиска
    sig_map = {sig.timestamp: sig for sig in signals}
    
    # Итерируемся по каждой свече
    for idx, (ts, candle) in enumerate(df.iterrows()):
        # 1. Если есть открытая позиция - проверяем её
        if active_pos:
            is_closed = False
            # Проверяем LONG
            if active_pos['type'] == Action.LONG:
                if candle['low'] <= active_pos['sl']:
                    results.append(((active_pos['sl'] - active_pos['entry']) / active_pos['entry']) * 100 - fee)
                    is_closed = True
                elif candle['high'] >= active_pos['tp']:
                    results.append(((active_pos['tp'] - active_pos['entry']) / active_pos['entry']) * 100 - fee)
                    is_closed = True
            # Проверяем SHORT
            elif active_pos['type'] == Action.SHORT:
                if candle['high'] >= active_pos['sl']:
                    results.append(((active_pos['entry'] - active_pos['sl']) / active_pos['entry']) * 100 - fee)
                    is_closed = True
                elif candle['low'] <= active_pos['tp']:
                    results.append(((active_pos['entry'] - active_pos['tp']) / active_pos['entry']) * 100 - fee)
                    is_closed = True
            
            if is_closed:
                active_pos = None
                cooldown_until_idx = idx + 6 # Остывание 6 свечей после выхода
                continue

        # 2. Если позиции нет и нет остывания - ищем сигнал
        if not active_pos and idx > cooldown_until_idx and ts in sig_map:
            sig = sig_map[ts]
            # Важно: берем SL/TP из сигнала, если их нет - ставим дефолт 1%
            sl = sig.stop_loss if sig.stop_loss else sig.price * 0.99 if sig.action == Action.LONG else sig.price * 1.01
            tp = sig.take_profit if sig.take_profit else sig.price * 1.02 if sig.action == Action.LONG else sig.price * 0.98
            
            if sig.action in [Action.LONG, Action.SHORT]:
                active_pos = {
                    'entry': sig.price,
                    'sl': sl,
                    'tp': tp,
                    'type': sig.action
                }
    return results

# Debug flags: enable/disable per-strategy diagnostic prints
DEBUG_TREND = False
DEBUG_FLAT = False
DEBUG_MOMENTUM = False


def _check_bar_tp_sl(position_side, high, low, current_price, tp_price, sl_price):
    """
    Проверяет, были ли на свече исполнены TP или SL.

    Возвращает кортеж (event_type, price) где event_type в ('tp','sl','sl_gap') или None если ничего не сработало.
    Логика: для LONG проверяем сначала TP по high, затем SL по low, затем SL по gap (current_price <= sl).
           для SHORT проверяем сначала TP по low, затем SL по high, затем SL по gap (current_price >= sl).
    """
    try:
        if position_side.value == "long":
            if tp_price and high >= tp_price:
                return ("tp", tp_price)
            if sl_price and low <= sl_price:
                return ("sl", sl_price)
            if sl_price and current_price <= sl_price:
                return ("sl_gap", current_price)
        else:  # SHORT
            if tp_price and low <= tp_price:
                return ("tp", tp_price)
            if sl_price and high >= sl_price:
                return ("sl", sl_price)
            if sl_price and current_price >= sl_price:
                return ("sl_gap", current_price)
    except Exception:
        return None
    return None


@dataclass
class StrategyResult:
    """Результаты тестирования стратегии"""
    strategy: str
    symbol: str
    total_trades: int
    profitable: int
    losing: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    profit_factor: float
    signals_count: int
    error: Optional[str] = None


def test_strategy_silent(strategy_name: str, symbol: str, days_back: int = 30) -> Optional[StrategyResult]:
    """
    Тестирует стратегию и возвращает результаты без вывода в консоль
    """
    try:
        settings = load_settings()
        
        # Получаем данные
        client = BybitClient(api=settings.api)
        candles_needed = days_back * 96
        if candles_needed > 1000:
            candles_needed = 1000
        
        interval = str(settings.timeframe) if isinstance(settings.timeframe, int) else settings.timeframe
        df = client.get_kline_df(symbol=symbol, interval=interval, limit=candles_needed)
        if df is None or len(df) == 0:
            return StrategyResult(
                strategy=strategy_name,
                symbol=symbol,
                total_trades=0,
                profitable=0,
                losing=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                signals_count=0,
                error="Не удалось получить данные"
            )
        
        # Подготавливаем данные
        df = prepare_with_indicators(
            df,
            adx_length=settings.strategy.adx_length,
            di_length=settings.strategy.di_length,
            sma_length=settings.strategy.sma_length,
            rsi_length=settings.strategy.rsi_length,
            breakout_lookback=settings.strategy.breakout_lookback,
            bb_length=settings.strategy.bb_length,
            bb_std=settings.strategy.bb_std,
            atr_length=14,
            ema_fast_length=settings.strategy.ema_fast_length,
            ema_slow_length=settings.strategy.ema_slow_length,
            ema_timeframe=settings.strategy.momentum_ema_timeframe,
        )
        df = enrich_for_strategy(df, settings.strategy)
        
        # Генерируем сигналы в зависимости от стратегии
        signals = []
        position_bias = None
        state = {} # Состояние для Cooldown и других параметров
        
        if strategy_name == "flat":
            signals = []
            position_bias = None
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                # Pass explicit strategy hint so detect_market_phase returns the expected phase
                market_phase = detect_market_phase(row, 'FLAT')
                if market_phase == MarketPhase.FLAT:
                    # Используем generate_flat_signal через обертку или напрямую, если нужно
                    # Но в generate_report используется generate_range_signal (row-based)
                    # Для поддержки Cooldown в row-based версии, нам нужно передать state
                    sig = generate_range_signal(row, position_bias, settings.strategy)
                    
                    # Ручной Cooldown для row-based стратегий в отчете
                    last_idx = state.get('last_signal_idx', -100)
                    if idx - last_idx < 10:
                        sig = Signal(timestamp=row.name, action=Action.HOLD, reason="cooldown", price=row["close"])
                    # debug indicators_info
                    ind = getattr(sig, 'indicators_info', None) if sig is not None else None
                    if ind is None and isinstance(sig, dict):
                        ind = sig.get('indicators_info')
                    # Print debug even if indicators dict is empty (check is not truthy)
                    if ind is not None and DEBUG_FLAT:
                        print(f"[debug][FLAT] {timestamp} indicators: atr={ind.get('atr') if isinstance(ind, dict) else ind}, rsi={ind.get('rsi') if isinstance(ind, dict) else None}, vol_avg5={ind.get('vol_avg5') if isinstance(ind, dict) else None}, bb_width={ind.get('bb_width') if isinstance(ind, dict) else None}")
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="flat_not_in_flat_phase", price=row["close"])
                signals.append(sig)
                if sig.action in (Action.LONG, Action.SHORT):
                    state['last_signal_idx'] = idx
                    if sig.action == Action.LONG:
                        if position_bias is None:
                            position_bias = Bias.LONG
                        elif position_bias == Bias.SHORT:
                            position_bias = Bias.LONG
                    elif sig.action == Action.SHORT:
                        if position_bias is None:
                            position_bias = Bias.SHORT
                        elif position_bias == Bias.LONG:
                            position_bias = Bias.SHORT
        elif strategy_name == "trend":
            signals = []
            position_bias = None
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                # Pass explicit strategy hint so detect_market_phase returns the expected phase
                market_phase = detect_market_phase(row, 'TREND')
                if market_phase == MarketPhase.TREND:
                    # Diagnostic: print key row indicators that generate_trend_signal expects
                    if DEBUG_TREND:
                        try:
                            print(f"[debug][TREND] {timestamp} row indicators: sma={row.get('sma')}, sma_prev={row.get('sma_prev')}, atr={row.get('atr')}, close={row.get('close')}")
                        except Exception:
                            print(f"[debug][TREND] {timestamp} row indicators: <could not read row>")

                    # Ручной Cooldown для TREND
                    last_idx = state.get('last_signal_idx', -100)
                    if idx - last_idx < 10:
                        sig = {"signal": None, "reason": "cooldown"}
                    else:
                        sig = generate_trend_signal(row, position_bias, settings.strategy)
                    
                    # normalize legacy dict response to Signal object for downstream processing
                    try:
                        if isinstance(sig, dict):
                            res = sig
                            sig_action = res.get('signal')
                            if sig_action == 'LONG':
                                action_obj = Action.LONG
                            elif sig_action == 'SHORT':
                                action_obj = Action.SHORT
                            else:
                                action_obj = Action.HOLD

                            reason = res.get('reason', '')
                            price = float(row.get('close', 0.0))
                            indicators = res.get('indicators_info', {})
                            sig = Signal(
                                timestamp=row.name,
                                action=action_obj,
                                reason=reason,
                                price=price,
                                stop_loss=res.get('stop_loss'),
                                take_profit=res.get('take_profit'),
                                indicators_info=indicators
                            )
                    except Exception:
                        # if normalization fails, fall back to a HOLD Signal to avoid crashing
                        sig = Signal(timestamp=row.name, action=Action.HOLD, reason=f"trend_normalization_error", price=float(row.get('close', 0.0)))
                    ind = getattr(sig, 'indicators_info', None) if sig is not None else None
                    if ind is None and isinstance(sig, dict):
                        ind = sig.get('indicators_info')
                    # Additional diagnostic output to catch all cases where indicators are missing
                    if DEBUG_TREND:
                        try:
                            sig_type = type(sig).__name__
                            if isinstance(sig, dict):
                                reason = sig.get('reason')
                                has_ind = 'indicators_info' in sig
                            else:
                                reason = getattr(sig, 'reason', None)
                                has_ind = getattr(sig, 'indicators_info', None) is not None
                            print(f"[debug][TREND] {timestamp} sig_type={sig_type} reason={reason} has_indicators={has_ind} indicators={ind}")
                        except Exception as _:
                            print(f"[debug][TREND] {timestamp} sig={sig}")
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="trend_not_in_trend_phase", price=row["close"])
                signals.append(sig)
                if sig.action in (Action.LONG, Action.SHORT):
                    state['last_signal_idx'] = idx
                    if sig.action == Action.LONG:
                        if position_bias is None:
                            position_bias = Bias.LONG
                        elif position_bias == Bias.SHORT:
                            position_bias = Bias.LONG
                    elif sig.action == Action.SHORT:
                        if position_bias is None:
                            position_bias = Bias.SHORT
                        elif position_bias == Bias.LONG:
                            position_bias = Bias.SHORT
        elif strategy_name == "smc":
            signals = build_smc_signals(df, settings.strategy, symbol=symbol)
        elif strategy_name == "ict":
            signals = build_ict_signals(df, settings.strategy, symbol=symbol)
        elif strategy_name == "momentum":
            signals = []
            position_bias = None
            prev_ema_fast = None
            prev_ema_slow = None
            ema_timeframe = settings.strategy.momentum_ema_timeframe
            ema_fast_col = f"ema_fast_{ema_timeframe}"
            ema_slow_col = f"ema_slow_{ema_timeframe}"
            
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                ema_fast = row.get(ema_fast_col, None)
                ema_slow = row.get(ema_slow_col, None)
                
                if idx > 0 and (prev_ema_fast is not None and prev_ema_slow is not None and 
                    all(x is not None and not (isinstance(x, float) and (x != x)) for x in [ema_fast, ema_slow, prev_ema_fast, prev_ema_slow])):
                    ema_cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                    ema_already_bullish = ema_fast > ema_slow and (ema_fast - ema_slow) / ema_slow > 0.0005
                    ema_cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
                    ema_already_bearish = ema_fast < ema_slow and (ema_slow - ema_fast) / ema_slow > 0.0005
                    
                    if (ema_cross_up or (ema_already_bullish and position_bias is None)) or \
                       (ema_cross_down or (ema_already_bearish and position_bias is None)):
                        sig = generate_momentum_breakout_signal(row, position_bias, settings.strategy)
                        ind = getattr(sig, 'indicators_info', None) if sig is not None else None
                        if ind is None and isinstance(sig, dict):
                            ind = sig.get('indicators_info')
                        if ind and DEBUG_MOMENTUM:
                            print(f"[debug][MOMENTUM] {timestamp} indicators: ema20={ind.get('ema_short') or ind.get('ema20')}, ema50={ind.get('ema_long') or ind.get('ema50')}, rsi={ind.get('rsi')}, vol_current={ind.get('vol_current')}, vol_avg5={ind.get('vol_avg5')}")
                    else:
                        sig = Signal(timestamp=row.name, action=Action.HOLD, reason="momentum_no_ema_setup", price=row["close"])
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="momentum_no_data", price=row["close"])
                
                signals.append(sig)
                
                if sig.action == Action.LONG:
                    if position_bias is None:
                        position_bias = Bias.LONG
                    elif position_bias == Bias.SHORT:
                        position_bias = Bias.LONG
                elif sig.action == Action.SHORT:
                    if position_bias is None:
                        position_bias = Bias.SHORT
                    elif position_bias == Bias.LONG:
                        position_bias = Bias.SHORT
                
                prev_ema_fast = ema_fast if ema_fast is not None else prev_ema_fast
                prev_ema_slow = ema_slow if ema_slow is not None else prev_ema_slow
        elif strategy_name == "ml":
            # Находим модель для символа
            print(f"[test_strategy_silent] 🔍 ML: Searching for model for {symbol}...", flush=True)
            model_path = None
            model_dir = Path(__file__).parent / "ml_models"
            model_type_preference = getattr(settings, 'ml_model_type_for_all', None)
            all_model_files = list(model_dir.glob(f"*{symbol}*.pkl"))
            
            if all_model_files:
                if model_type_preference:
                    preferred_files = [f for f in all_model_files if f.name.startswith(f"{model_type_preference}_")]
                    if preferred_files:
                        model_path = str(preferred_files[0])
                    else:
                        model_path = str(all_model_files[0])
                else:
                    for model_type in ["ensemble", "rf", "xgb"]:
                        preferred_files = [f for f in all_model_files if f.name.startswith(f"{model_type}_")]
                        if preferred_files:
                            model_path = str(preferred_files[0])
                            break
                    if not model_path:
                        model_path = str(all_model_files[0])
            else:
                if settings.ml_model_path and Path(settings.ml_model_path).exists():
                    model_path = settings.ml_model_path
                else:
                    print(f"[test_strategy_silent] ❌ ML: Model not found for {symbol}", flush=True)
                    return StrategyResult(
                        strategy=strategy_name,
                        symbol=symbol,
                        total_trades=0,
                        profitable=0,
                        losing=0,
                        win_rate=0.0,
                        total_pnl=0.0,
                        avg_pnl=0.0,
                        avg_win=0.0,
                        avg_loss=0.0,
                        max_win=0.0,
                        max_loss=0.0,
                        profit_factor=0.0,
                        signals_count=0,
                        error="ML модель не найдена"
                    )
            
            print(f"[test_strategy_silent] 🤖 ML: Loading model from {model_path}...", flush=True)
            
            try:
                import threading
                import time
                
                # Функция для генерации сигналов с таймаутом
                signals_result = [None]
                signals_error = [None]
                signals_exception = [None]
                
                def generate_signals():
                    try:
                        print(f"[test_strategy_silent] 🔄 ML: Starting build_ml_signals for {symbol}...", flush=True)
                        start_time = time.time()
                        
                        signals_result[0] = build_ml_signals(
                            df,
                            model_path,
                            confidence_threshold=settings.ml_confidence_threshold,
                            min_signal_strength=settings.ml_min_signal_strength,
                            stability_filter=settings.ml_stability_filter,
                            leverage=settings.leverage,
                            target_profit_pct_margin=getattr(settings, 'ml_target_profit_pct_margin', 25.0),
                            max_loss_pct_margin=getattr(settings, 'ml_max_loss_pct_margin', 10.0),
                        )
                        
                        elapsed = time.time() - start_time
                        print(f"[test_strategy_silent] ⏱️ ML: build_ml_signals completed in {elapsed:.1f}s for {symbol}", flush=True)
                    except Exception as e:
                        signals_exception[0] = e
                        signals_error[0] = str(e)
                        import traceback
                        signals_error[0] += f"\n{traceback.format_exc()}"
                
                # Запускаем в отдельном потоке с таймаутом
                print(f"[test_strategy_silent] 🚀 ML: Starting thread for {symbol}...", flush=True)
                thread = threading.Thread(target=generate_signals, daemon=True)
                thread.start()
                
                # Ждем с таймаутом 5 минут (300 секунд)
                print(f"[test_strategy_silent] ⏳ ML: Waiting for completion (timeout: 300s)...", flush=True)
                thread.join(timeout=300)
                
                if thread.is_alive():
                    print(f"[test_strategy_silent] ⚠️ ML: Timeout (5 min) for {symbol}, thread still alive", flush=True)
                    return StrategyResult(
                        strategy=strategy_name,
                        symbol=symbol,
                        total_trades=0,
                        profitable=0,
                        losing=0,
                        win_rate=0.0,
                        total_pnl=0.0,
                        avg_pnl=0.0,
                        avg_win=0.0,
                        avg_loss=0.0,
                        max_win=0.0,
                        max_loss=0.0,
                        profit_factor=0.0,
                        signals_count=0,
                        error="ML стратегия превысила таймаут (5 минут) - возможно, модель слишком большая или зависла при загрузке"
                    )
                
                if signals_error[0]:
                    print(f"[test_strategy_silent] ❌ ML: Error generating signals: {signals_error[0]}", flush=True)
                    return StrategyResult(
                        strategy=strategy_name,
                        symbol=symbol,
                        total_trades=0,
                        profitable=0,
                        losing=0,
                        win_rate=0.0,
                        total_pnl=0.0,
                        avg_pnl=0.0,
                        avg_win=0.0,
                        avg_loss=0.0,
                        max_win=0.0,
                        max_loss=0.0,
                        profit_factor=0.0,
                        signals_count=0,
                        error=f"Ошибка ML стратегии: {signals_error[0][:200]}"  # Ограничиваем длину ошибки
                    )
                
                if signals_result[0] is None:
                    print(f"[test_strategy_silent] ❌ ML: signals_result is None for {symbol}", flush=True)
                    return StrategyResult(
                        strategy=strategy_name,
                        symbol=symbol,
                        total_trades=0,
                        profitable=0,
                        losing=0,
                        win_rate=0.0,
                        total_pnl=0.0,
                        avg_pnl=0.0,
                        avg_win=0.0,
                        avg_loss=0.0,
                        max_win=0.0,
                        max_loss=0.0,
                        profit_factor=0.0,
                        signals_count=0,
                        error="ML стратегия вернула None (неизвестная ошибка)"
                    )
                
                signals = signals_result[0]
                print(f"[test_strategy_silent] ✅ ML: Generated {len(signals)} signals for {symbol}", flush=True)
            except Exception as e:
                print(f"[test_strategy_silent] ❌ ML: Exception: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return StrategyResult(
                    strategy=strategy_name,
                    symbol=symbol,
                    total_trades=0,
                    profitable=0,
                    losing=0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    avg_pnl=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    max_win=0.0,
                    max_loss=0.0,
                    profit_factor=0.0,
                    signals_count=0,
                    error=f"Исключение при тестировании ML: {str(e)}"
                )
        elif strategy_name == "liquidity":
            # LIQUIDITY стратегия использует build_signals с use_liquidity=True
            signals = build_signals(df, settings.strategy, use_liquidity=True)
        elif strategy_name == "liquidation_hunter":
            signals = build_liquidation_hunter_signals(df, settings.strategy, symbol=symbol)
        elif strategy_name == "zscore":
            signals = build_zscore_signals(df, settings.strategy, symbol=symbol)
        elif strategy_name == "vbo":
            signals = build_vbo_signals(df, settings.strategy, symbol=symbol)
        else:
            return StrategyResult(
                strategy=strategy_name,
                symbol=symbol,
                total_trades=0,
                profitable=0,
                losing=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                signals_count=0,
                error=f"Неизвестная стратегия: {strategy_name}"
            )
        
        actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
        
        if len(actionable_signals) == 0:
            return StrategyResult(
                strategy=strategy_name,
                symbol=symbol,
                total_trades=0,
                profitable=0,
                losing=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                signals_count=0,
                error="Нет actionable сигналов"
            )
        
        # Подготавливаем TP/SL для сигналов, если они не заполнены
        for s in signals:
            if s.action in (Action.LONG, Action.SHORT):
                # Парсим TP/SL для ICT и ML из reason, если они там есть
                if "ict_" in s.reason:
                    import re
                    sl_match = re.search(r'sl_([\d.]+)', s.reason)
                    tp_match = re.search(r'tp_([\d.]+)', s.reason)
                    if sl_match: s.stop_loss = float(sl_match.group(1))
                    if tp_match: s.take_profit = float(tp_match.group(1))
                elif "ml_" in s.reason:
                    import re
                    tp_match = re.search(r'TP_([\d.]+)%', s.reason)
                    sl_match = re.search(r'SL_([\d.]+)%', s.reason)
                    if tp_match: s.take_profit = s.price * (1 + float(tp_match.group(1))/100.0) if s.action == Action.LONG else s.price * (1 - float(tp_match.group(1))/100.0)
                    if sl_match: s.stop_loss = s.price * (1 - float(sl_match.group(1))/100.0) if s.action == Action.LONG else s.price * (1 + float(sl_match.group(1))/100.0)

                # Если все еще None, используем дефолтные значения
                if s.stop_loss is None or s.take_profit is None:
                    if strategy_name == "flat":
                        sl_pct = settings.strategy.range_stop_loss_pct
                        tp_pct = settings.risk.take_profit_pct
                    elif strategy_name == "vbo":
                        tp_pct = 0.032
                        sl_pct = 0.011
                    elif strategy_name == "liquidation_hunter":
                        tp_pct = 0.025
                        sl_pct = 0.010
                    elif strategy_name == "zscore":
                        tp_pct = 0.030
                        sl_pct = 0.010
                    else:
                        sl_pct = settings.risk.stop_loss_pct
                        tp_pct = settings.risk.take_profit_pct

                    if s.action == Action.LONG:
                        if s.stop_loss is None: s.stop_loss = s.price * (1 - sl_pct)
                        if s.take_profit is None: s.take_profit = s.price * (1 + tp_pct)
                    else:
                        if s.stop_loss is None: s.stop_loss = s.price * (1 + sl_pct)
                        if s.take_profit is None: s.take_profit = s.price * (1 - tp_pct)

        # Симулируем торговлю с использованием новой функции
        fee = 0.05
        pnl_list = simulate_trading(df, signals, fee=fee)
        
        if not pnl_list:
            return StrategyResult(
                strategy=strategy_name,
                symbol=symbol,
                total_trades=0,
                profitable=0,
                losing=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                signals_count=len(actionable_signals),
                error="Нет сделок"
            )

        profitable_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p < 0]
        
        total_pnl = sum(pnl_list)
        avg_pnl = total_pnl / len(pnl_list)
        avg_win = sum(profitable_trades) / len(profitable_trades) if profitable_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        max_win = max(pnl_list)
        max_loss = min(pnl_list)
        win_rate = len(profitable_trades) / len(pnl_list) * 100
        
        total_wins = sum(profitable_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return StrategyResult(
            strategy=strategy_name,
            symbol=symbol,
            total_trades=len(pnl_list),
            profitable=len(profitable_trades),
            losing=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            profit_factor=profit_factor,
            signals_count=len(actionable_signals)
        )
    
    except Exception as e:
        return StrategyResult(
            strategy=strategy_name,
            symbol=symbol,
            total_trades=0,
            profitable=0,
            losing=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0,
            profit_factor=0.0,
            signals_count=0,
            error=str(e)
        )


def generate_report(strategies: List[str], symbols: List[str], days: int = 30, output_file: Optional[str] = None):
    """
    Генерирует сводный отчет по всем стратегиям
    """
    print("=" * 100)
    print("📊 ГЕНЕРАЦИЯ СВОДНОГО ОТЧЕТА ПО СТРАТЕГИЯМ")
    print("=" * 100)
    print(f"Стратегии: {', '.join(strategies)}")
    print(f"Символы: {', '.join(symbols)}")
    print(f"Период: {days} дней")
    print()
    
    results: List[StrategyResult] = []
    total_tests = len(strategies) * len(symbols)
    current_test = 0
    
    for strategy in strategies:
        for symbol in symbols:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Тестирование {strategy.upper()} на {symbol}...", end=" ", flush=True)
            result = test_strategy_silent(strategy, symbol, days)
            results.append(result)
            if result.error:
                print(f"❌ Ошибка: {result.error}")
            else:
                print(f"✅ {result.total_trades} сделок, PnL: {result.total_pnl:+.2f} USDT, WR: {result.win_rate:.1f}%")
    
    print("\n" + "=" * 100)
    print("📈 СВОДНЫЙ ОТЧЕТ")
    print("=" * 100)
    
    # Группируем результаты по стратегиям
    by_strategy: Dict[str, List[StrategyResult]] = {}
    for result in results:
        if result.strategy not in by_strategy:
            by_strategy[result.strategy] = []
        by_strategy[result.strategy].append(result)
    
    # Выводим таблицу по стратегиям
    print("\n📊 РЕЗУЛЬТАТЫ ПО СТРАТЕГИЯМ:")
    print("-" * 100)
    print(f"{'Стратегия':<15} {'Символ':<10} {'Сделок':<8} {'WR%':<7} {'PnL':<12} {'Avg PnL':<10} {'PF':<8} {'Сигналов':<10}")
    print("-" * 100)
    
    for strategy in strategies:
        strategy_results = by_strategy.get(strategy, [])
        for result in strategy_results:
            if result.error:
                print(f"{strategy:<15} {result.symbol:<10} {'ERROR':<8} {'-':<7} {'-':<12} {'-':<10} {'-':<8} {'-':<10}")
                print(f"  └─ {result.error}")
            else:
                pnl_str = f"{result.total_pnl:+.2f}"
                avg_pnl_str = f"{result.avg_pnl:+.2f}"
                pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "inf"
                print(f"{strategy:<15} {result.symbol:<10} {result.total_trades:<8} {result.win_rate:>6.1f}% {pnl_str:<12} {avg_pnl_str:<10} {pf_str:<8} {result.signals_count:<10}")
    
    # Итоговая статистика
    print("\n" + "=" * 100)
    print("📊 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 100)
    
    successful_results = [r for r in results if not r.error and r.total_trades > 0]
    
    if successful_results:
        total_pnl_all = sum(r.total_pnl for r in successful_results)
        total_trades_all = sum(r.total_trades for r in successful_results)
        total_profitable = sum(r.profitable for r in successful_results)
        avg_win_rate = sum(r.win_rate for r in successful_results) / len(successful_results) if successful_results else 0
        
        print(f"Всего успешных тестов: {len(successful_results)}")
        print(f"Общий PnL: {total_pnl_all:+.2f} USDT")
        print(f"Всего сделок: {total_trades_all}")
        print(f"Средний Win Rate: {avg_win_rate:.1f}%")
        
        # Лучшие стратегии
        print("\n🏆 ТОП-3 СТРАТЕГИИ ПО PnL:")
        sorted_by_pnl = sorted(successful_results, key=lambda x: x.total_pnl, reverse=True)
        for i, result in enumerate(sorted_by_pnl[:3], 1):
            print(f"  {i}. {result.strategy.upper()} на {result.symbol}: {result.total_pnl:+.2f} USDT (WR: {result.win_rate:.1f}%, {result.total_trades} сделок)")
        
        print("\n🏆 ТОП-3 СТРАТЕГИИ ПО WIN RATE:")
        sorted_by_wr = sorted(successful_results, key=lambda x: x.win_rate, reverse=True)
        for i, result in enumerate(sorted_by_wr[:3], 1):
            print(f"  {i}. {result.strategy.upper()} на {result.symbol}: {result.win_rate:.1f}% (PnL: {result.total_pnl:+.2f} USDT, {result.total_trades} сделок)")
    
    # Сохраняем в JSON, если указан файл
    if output_file:
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "strategies": strategies,
            "symbols": symbols,
            "results": [
                {
                    "strategy": r.strategy,
                    "symbol": r.symbol,
                    "total_trades": r.total_trades,
                    "profitable": r.profitable,
                    "losing": r.losing,
                    "win_rate": r.win_rate,
                    "total_pnl": r.total_pnl,
                    "avg_pnl": r.avg_pnl,
                    "avg_win": r.avg_win,
                    "avg_loss": r.avg_loss,
                    "max_win": r.max_win,
                    "max_loss": r.max_loss,
                    "profit_factor": r.profit_factor if r.profit_factor != float('inf') else "inf",
                    "signals_count": r.signals_count,
                    "error": r.error
                }
                for r in results
            ]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Отчет сохранен в {output_file}")
    
    print("\n" + "=" * 100)
    print("✅ ОТЧЕТ ЗАВЕРШЕН")
    print("=" * 100)


def optimize_strategies_auto(symbols: List[str] = None, days: int = 30, min_pnl: float = 0.0, min_win_rate: float = 0.0, progress_callback=None) -> Dict:
    """
    Автоматически оптимизирует стратегии: тестирует все стратегии для всех символов,
    определяет лучшие (прибыльные) стратегии и возвращает рекомендации по настройкам.
    
    Args:
        symbols: Список символов для тестирования (по умолчанию: все доступные)
        days: Количество дней для тестирования
        min_pnl: Минимальный PnL для включения стратегии (по умолчанию: 0.0 - только прибыльные)
        min_win_rate: Минимальный Win Rate для включения стратегии (по умолчанию: 0.0)
    
    Returns:
        Dict с рекомендациями по настройкам для каждого символа
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Все доступные стратегии (liquidity и ml отключены - ml может зависать при загрузке больших моделей)
    all_strategies = ["trend", "flat", "momentum", "smc", "ict", "liquidation_hunter", "zscore", "vbo"]
    
    print("=" * 100)
    print("🤖 АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ СТРАТЕГИЙ")
    print("=" * 100)
    print(f"Символы: {', '.join(symbols)}")
    print(f"Период: {days} дней")
    print(f"Минимальный PnL: {min_pnl:+.2f} USDT")
    print(f"Минимальный Win Rate: {min_win_rate:.1f}%")
    print()
    
    # Тестируем все стратегии для всех символов
    results: List[StrategyResult] = []
    total_tests = len(all_strategies) * len(symbols)
    current_test = 0
    
    for strategy in all_strategies:
        for symbol in symbols:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Тестирование {strategy.upper()} на {symbol}...", end=" ", flush=True)
            
            # Обновляем прогресс через callback, если он предоставлен
            if progress_callback:
                progress_callback(current_test, total_tests, f"{strategy.upper()} на {symbol}")
            
            # Для ML стратегии добавляем дополнительное логирование
            if strategy == "ml":
                print(f"\n[optimize] ⏳ ML стратегия может занять больше времени...", flush=True)
            
            result = test_strategy_silent(strategy, symbol, days)
            results.append(result)
            
            if result.error:
                print(f"❌ Ошибка: {result.error}", flush=True)
            else:
                status = "✅" if result.total_pnl > min_pnl and result.win_rate >= min_win_rate else "⚠️"
                print(f"{status} {result.total_trades} сделок, PnL: {result.total_pnl:+.2f} USDT, WR: {result.win_rate:.1f}%", flush=True)
            
            # Для ML стратегии добавляем сообщение о завершении
            if strategy == "ml":
                print(f"[optimize] ✅ ML стратегия завершена для {symbol}", flush=True)
    
    print("\n" + "=" * 100)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 100)
    
    # Группируем результаты по символам
    recommendations: Dict[str, Dict] = {}
    
    for symbol in symbols:
        symbol_results = [r for r in results if r.symbol == symbol and not r.error]
        
        # Фильтруем только прибыльные стратегии
        profitable_strategies = [
            r for r in symbol_results 
            if r.total_pnl > min_pnl and r.win_rate >= min_win_rate and r.total_trades > 0
        ]
        
        # Сортируем по PnL (лучшие первыми)
        profitable_strategies.sort(key=lambda x: x.total_pnl, reverse=True)
        
        # Определяем приоритетную стратегию (лучшая по PnL)
        priority_strategy = None
        if profitable_strategies:
            priority_strategy = profitable_strategies[0].strategy
        
        # Формируем настройки для символа
        symbol_settings = {
            "enable_trend_strategy": False,
            "enable_flat_strategy": False,
            "enable_ml_strategy": False,
            "enable_momentum_strategy": False,
            "enable_liquidity_sweep_strategy": False,
            "enable_smc_strategy": False,
            "enable_ict_strategy": False,
            "enable_liquidation_hunter_strategy": False,
            "enable_zscore_strategy": False,
            "enable_vbo_strategy": False,
            "strategy_priority": priority_strategy if priority_strategy else "hybrid"
        }
        
        # Включаем только прибыльные стратегии
        for result in profitable_strategies:
            strategy_key = f"enable_{result.strategy}_strategy"
            if strategy_key in symbol_settings:
                symbol_settings[strategy_key] = True
        
        # Если нет прибыльных стратегий, используем hybrid режим
        if not profitable_strategies:
            symbol_settings["strategy_priority"] = "hybrid"
        
        recommendations[symbol] = {
            "settings": symbol_settings,
            "profitable_strategies": [
                {
                    "strategy": r.strategy,
                    "pnl": r.total_pnl,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades
                }
                for r in profitable_strategies
            ],
            "all_results": [
                {
                    "strategy": r.strategy,
                    "pnl": r.total_pnl,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "error": r.error
                }
                for r in symbol_results
            ]
        }
        
        print(f"\n📈 {symbol}:")
        print(f"  Прибыльных стратегий: {len(profitable_strategies)}")
        if profitable_strategies:
            print(f"  Приоритетная стратегия: {priority_strategy.upper()}")
            print(f"  Включенные стратегии:")
            for r in profitable_strategies:
                print(f"    ✅ {r.strategy.upper()}: PnL {r.total_pnl:+.2f} USDT, WR {r.win_rate:.1f}%, {r.total_trades} сделок")
        else:
            print(f"  ⚠️ Нет прибыльных стратегий для {symbol}")
    
    print("\n" + "=" * 100)
    print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 100)
    
    # Сводка по всем символам
    total_profitable = sum(len(rec.get("profitable_strategies", [])) for rec in recommendations.values())
    print(f"\n📊 ИТОГОВАЯ СВОДКА:")
    print(f"  Всего символов проанализировано: {len(recommendations)}")
    print(f"  Всего прибыльных стратегий найдено: {total_profitable}")
    for symbol, rec in recommendations.items():
        profitable = rec.get("profitable_strategies", [])
        if profitable:
            best = profitable[0]
            print(f"  {symbol}: {len(profitable)} стратегий, лучшая - {best['strategy'].upper()} (PnL: {best['pnl']:+.2f} USDT, WR: {best['win_rate']:.1f}%)")
        else:
            print(f"  {symbol}: ⚠️ Нет прибыльных стратегий")
    
    print("\n" + "=" * 100)
    
    return {
        "recommendations": recommendations,
        "test_period_days": days,
        "min_pnl_threshold": min_pnl,
        "min_win_rate_threshold": min_win_rate,
        "generated_at": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Генерация сводного отчета по всем стратегиям")
    parser.add_argument("--strategies", type=str, nargs="+", 
                       default=["trend", "flat", "momentum", "smc", "ict", "liquidation_hunter", "zscore", "vbo"],  # liquidity и ml отключены
                       help="Список стратегий для тестирования")
    parser.add_argument("--symbols", type=str, nargs="+",
                       default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                       help="Список символов для тестирования")
    parser.add_argument("--days", type=int, default=7,
                       help="Количество дней для тестирования (по умолчанию: 7)")
    parser.add_argument("--output", type=str, default=None,
                       help="Путь к файлу для сохранения JSON отчета")
    parser.add_argument("--optimize", action="store_true",
                       help="Запустить автоматическую оптимизацию стратегий")
    parser.add_argument("--min-pnl", type=float, default=0.0,
                       help="Минимальный PnL для включения стратегии (по умолчанию: 0.0)")
    parser.add_argument("--min-win-rate", type=float, default=0.0,
                       help="Минимальный Win Rate для включения стратегии (по умолчанию: 0.0)")
    
    args = parser.parse_args()
    
    if args.optimize:
        result = optimize_strategies_auto(args.symbols, args.days, args.min_pnl, args.min_win_rate)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результаты сохранены в {args.output}")
    else:
        generate_report(args.strategies, args.symbols, args.days, args.output)


if __name__ == "__main__":
    main()
