#!/usr/bin/env python3
"""
Генератор сводного отчета по всем стратегиям
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, generate_trend_signal, generate_range_signal, generate_momentum_breakout_signal, detect_market_phase, MarketPhase, Bias, Action, Signal, build_signals
from bot.smc_strategy import build_smc_signals
from bot.ict_strategy import build_ict_signals
from bot.ml.strategy_ml import build_ml_signals
from bot.liquidation_hunter_strategy import build_liquidation_hunter_signals
from bot.zscore_strategy import build_zscore_signals
from bot.vbo_strategy import build_vbo_signals
from bot.simulation import Simulator


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


def test_strategy_silent(strategy_name: str, symbol: str, days_back: int = 10) -> Optional[StrategyResult]:
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
        
        if strategy_name == "flat":
            signals = []
            position_bias = None
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                market_phase = detect_market_phase(row, settings.strategy)
                if market_phase == MarketPhase.FLAT:
                    sig = generate_range_signal(row, position_bias, settings.strategy)
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="flat_not_in_flat_phase", price=row["close"])
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
        elif strategy_name == "trend":
            signals = []
            position_bias = None
            for idx, (timestamp, row) in enumerate(df.iterrows()):
                market_phase = detect_market_phase(row, settings.strategy)
                if market_phase == MarketPhase.TREND:
                    sig = generate_trend_signal(row, position_bias, settings.strategy)
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="trend_not_in_trend_phase", price=row["close"])
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
        
        # Симулируем торговлю (упрощенная версия без TP/SL из reason для отчета)
        simulator = Simulator(settings)
        signal_dict = {s.timestamp: s for s in signals}
        position_tp_sl = {}
        
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            # Определяем цены свечи в начале итерации (нужны для всех проверок)
            current_price = row['close']
            high = row.get('high', current_price)
            low = row.get('low', current_price)
            
            if simulator.position:
                # Сохраняем данные позиции перед проверками (на случай закрытия)
                position_side = simulator.position.side
                entry_price = simulator.position.avg_price
                entry_reason = simulator.position.entry_reason
                
                tp_price = None
                sl_price = None
                
                if (strategy_name == "ict" or strategy_name == "ml") and entry_reason in position_tp_sl:
                    tp_price = position_tp_sl[entry_reason]["tp"]
                    sl_price = position_tp_sl[entry_reason]["sl"]
                else:
                    if strategy_name == "flat":
                        sl_pct = settings.strategy.range_stop_loss_pct
                        tp_pct = settings.risk.take_profit_pct
                    elif strategy_name == "vbo":
                        # Для VBO используем сбалансированные TP/SL: TP=3.2%, SL=1.1% от цены
                        # Пробои волатильности требуют баланса: достаточно широкий TP, но не слишком
                        # чтобы не терять сделки по SL до достижения TP
                        tp_pct = 0.032  # 3.2% от цены (32% от маржи при 10x)
                        sl_pct = 0.011  # 1.1% от цены (11% от маржи при 10x) - RR ~2.9:1
                    elif strategy_name == "liquidation_hunter":
                        # Для Liquidation Hunter используем оптимизированные TP/SL: TP=2.5%, SL=1.0% от цены
                        # Увеличены уровни для лучшего RR и снижения убыточности
                        tp_pct = 0.025  # 2.5% от цены (25% от маржи при 10x)
                        sl_pct = 0.010  # 1.0% от цены (10% от маржи при 10x) - RR ~2.5:1
                    elif strategy_name == "zscore":
                        # Для ZSCORE используем оптимизированные TP/SL: TP=3.0%, SL=1.0% от цены
                        # ZSCORE - mean reversion стратегия, нужны быстрые тейки с хорошим RR
                        # Увеличены уровни для лучшей работы на волатильных парах (BTCUSDT, ETHUSDT)
                        # Проблема: при высоком WR (64.3%) отрицательный PnL означает плохой RR
                        # Увеличиваем TP для улучшения RR и снижения убыточности
                        tp_pct = 0.030  # 3.0% от цены (30% от маржи при 10x) - увеличено с 2.5%
                        sl_pct = 0.010  # 1.0% от цены (10% от маржи при 10x) - увеличено с 0.9%, RR ~3.0:1
                    else:
                        sl_pct = settings.risk.stop_loss_pct
                        tp_pct = settings.risk.take_profit_pct
                    
                    if position_side.value == "long":
                        tp_price = entry_price * (1 + tp_pct)
                        sl_price = entry_price * (1 - sl_pct)
                    else:
                        tp_price = entry_price * (1 - tp_pct)
                        sl_price = entry_price * (1 + sl_pct)
                
                # Проверяем, что позиция все еще открыта после расчетов
                if not simulator.position:
                    if timestamp in signal_dict:
                        sig = signal_dict[timestamp]
                        simulator.on_signal(sig)
                    continue
                
                if position_side.value == "long":
                    # TP: проверяем high свечи
                    if tp_price and high >= tp_price:
                        simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                        if timestamp in signal_dict:
                            continue
                        # Позиция закрыта, проверяем наличие перед продолжением
                        if not simulator.position:
                            if timestamp in signal_dict:
                                sig = signal_dict[timestamp]
                                simulator.on_signal(sig)
                            continue
                    # SL: проверяем low свечи и current_price (для учета gap)
                    if sl_price:
                        # Если low достиг SL, закрываем по SL
                        if low <= sl_price:
                            simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                            if timestamp in signal_dict:
                                continue
                            if not simulator.position:
                                if timestamp in signal_dict:
                                    sig = signal_dict[timestamp]
                                    simulator.on_signal(sig)
                                continue
                        # Если current_price уже за SL (gap), закрываем немедленно
                        elif current_price <= sl_price:
                            simulator._close(current_price, f"{strategy_name}_sl_hit_gap", timestamp)
                            if timestamp in signal_dict:
                                continue
                            if not simulator.position:
                                if timestamp in signal_dict:
                                    sig = signal_dict[timestamp]
                                    simulator.on_signal(sig)
                                continue
                    # Trailing Stop: если прибыль > 0.5%, подтягиваем SL к безубытку (только для не-ICT и не-ML стратегий)
                    # Momentum использует свой trailing stop на основе EMA50, поэтому пропускаем стандартный trailing stop
                    if strategy_name not in ("ict", "ml", "momentum") and simulator.position:
                        profit_pct = (current_price - entry_price) / entry_price
                        if profit_pct > 0.005:
                            breakeven_sl = entry_price * 1.001
                            if low <= breakeven_sl:
                                simulator._close(breakeven_sl, f"{strategy_name}_trailing_stop", timestamp)
                                if timestamp in signal_dict:
                                    continue
                                if not simulator.position:
                                    if timestamp in signal_dict:
                                        sig = signal_dict[timestamp]
                                        simulator.on_signal(sig)
                                    continue
                else:  # SHORT
                    # TP: проверяем low свечи
                    if tp_price and low <= tp_price:
                        simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                        if timestamp in signal_dict:
                            continue
                        if not simulator.position:
                            if timestamp in signal_dict:
                                sig = signal_dict[timestamp]
                                simulator.on_signal(sig)
                            continue
                    # SL: проверяем high свечи и current_price (для учета gap)
                    if sl_price:
                        # Если high достиг SL, закрываем по SL
                        if high >= sl_price:
                            simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                            if timestamp in signal_dict:
                                continue
                            if not simulator.position:
                                if timestamp in signal_dict:
                                    sig = signal_dict[timestamp]
                                    simulator.on_signal(sig)
                                continue
                        # Если current_price уже за SL (gap), закрываем немедленно
                        elif current_price >= sl_price:
                            simulator._close(current_price, f"{strategy_name}_sl_hit_gap", timestamp)
                            if timestamp in signal_dict:
                                continue
                            if not simulator.position:
                                if timestamp in signal_dict:
                                    sig = signal_dict[timestamp]
                                    simulator.on_signal(sig)
                                continue
                    # Trailing Stop: если прибыль > 0.5%, подтягиваем SL к безубытку (только для не-ICT и не-ML стратегий)
                    # Momentum использует свой trailing stop на основе EMA50, поэтому пропускаем стандартный trailing stop
                    if strategy_name not in ("ict", "ml", "momentum") and simulator.position:
                        profit_pct = (entry_price - current_price) / entry_price
                        if profit_pct > 0.005:
                            breakeven_sl = entry_price * 0.999
                            if high >= breakeven_sl:
                                simulator._close(breakeven_sl, f"{strategy_name}_trailing_stop", timestamp)
                                if timestamp in signal_dict:
                                    continue
                                if not simulator.position:
                                    if timestamp in signal_dict:
                                        sig = signal_dict[timestamp]
                                        simulator.on_signal(sig)
                                    continue
            
            if timestamp in signal_dict:
                sig = signal_dict[timestamp]
                # Обработка специальных сигналов HOLD для закрытия позиций
                if sig.reason == "range_sl_fixed" and simulator.position:
                    simulator._close(sig.price, f"{strategy_name}_sl_hit", timestamp)
                elif sig.reason in ("momentum_long_exit_trailing_stop", "momentum_short_exit_trailing_stop") and simulator.position:
                    # Momentum trailing stop: закрываем позицию по EMA50
                    simulator._close(sig.price, f"{strategy_name}_trailing_stop_ema", timestamp)
                elif sig.reason in ("momentum_long_exit_ema_reversal", "momentum_short_exit_ema_reversal") and simulator.position:
                    # Momentum exit signals: Action.SHORT для закрытия LONG, Action.LONG для закрытия SHORT
                    # Эти сигналы обрабатываются через simulator.on_signal, который должен закрыть позицию
                    was_position_open = simulator.position is not None
                    simulator.on_signal(sig)
                    # После обработки сигнала выхода, позиция должна быть закрыта
                    # Если позиция все еще открыта, закрываем её явно
                    if simulator.position and was_position_open:
                        simulator._close(sig.price, f"{strategy_name}_exit_ema_reversal", timestamp)
                elif sig.action != Action.HOLD or sig.reason != "range_sl_fixed":
                    was_position_open = simulator.position is not None
                    simulator.on_signal(sig)
                    
                    # Парсим TP/SL для ICT и ML
                    if sig.action in (Action.LONG, Action.SHORT) and sig.reason.startswith("ict_"):
                        import re
                        sl_match = re.search(r'sl_([\d.]+)', sig.reason)
                        tp_match = re.search(r'tp_([\d.]+)', sig.reason)
                        if sl_match and tp_match and simulator.position:
                            sl_price = float(sl_match.group(1))
                            tp_price = float(tp_match.group(1))
                            
                            # ВАЛИДАЦИЯ И ПЕРЕСЧЕТ TP/SL для ICT
                            # Проверяем, что SL соответствует требованиям (7-10% от маржи)
                            entry_price = simulator.position.avg_price
                            position_side = simulator.position.side
                            leverage = settings.leverage if hasattr(settings, 'leverage') else 10
                            
                            min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
                            max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
                            min_sl_pct_from_price = min_sl_pct_from_margin / leverage  # 0.7% от цены при 10x
                            max_sl_pct_from_price = max_sl_pct_from_margin / leverage  # 1.0% от цены при 10x
                            
                            # Проверяем SL
                            if position_side.value == "long":
                                sl_distance_pct = (entry_price - sl_price) / entry_price
                                if sl_distance_pct < min_sl_pct_from_price:
                                    # SL слишком близко - используем минимальный SL
                                    sl_price = entry_price * (1 - min_sl_pct_from_price)
                                    sl_distance_pct = min_sl_pct_from_price
                                elif sl_distance_pct > max_sl_pct_from_price:
                                    # SL слишком далеко - используем максимальный SL
                                    sl_price = entry_price * (1 - max_sl_pct_from_price)
                                    sl_distance_pct = max_sl_pct_from_price
                                
                                # Пересчитываем TP с R:R = 3.0 (улучшенное соотношение)
                                risk = entry_price - sl_price
                                rr_ratio = 3.0  # Улучшенный R:R для ICT
                                tp_price = entry_price + risk * rr_ratio
                            else:  # SHORT
                                sl_distance_pct = (sl_price - entry_price) / entry_price
                                if sl_distance_pct < min_sl_pct_from_price:
                                    # SL слишком близко - используем минимальный SL
                                    sl_price = entry_price * (1 + min_sl_pct_from_price)
                                    sl_distance_pct = min_sl_pct_from_price
                                elif sl_distance_pct > max_sl_pct_from_price:
                                    # SL слишком далеко - используем максимальный SL
                                    sl_price = entry_price * (1 + max_sl_pct_from_price)
                                    sl_distance_pct = max_sl_pct_from_price
                                
                                # Пересчитываем TP с R:R = 3.0 (улучшенное соотношение)
                                risk = sl_price - entry_price
                                rr_ratio = 3.0  # Улучшенный R:R для ICT
                                tp_price = entry_price - risk * rr_ratio
                            
                            position_tp_sl[simulator.position.entry_reason] = {"tp": tp_price, "sl": sl_price}
                    elif sig.action in (Action.LONG, Action.SHORT) and sig.reason.startswith("ml_"):
                        import re
                        tp_match = re.search(r'TP_([\d.]+)%', sig.reason)
                        sl_match = re.search(r'SL_([\d.]+)%', sig.reason)
                        if tp_match and sl_match and simulator.position:
                            tp_pct = float(tp_match.group(1)) / 100.0
                            sl_pct = float(sl_match.group(1)) / 100.0
                            # ВАЖНО: Используем реальную цену входа позиции, а не цену сигнала
                            # Сохраняем данные позиции перед использованием
                            if not simulator.position:
                                continue
                            entry_price = simulator.position.avg_price
                            position_side_ml = simulator.position.side
                            entry_reason_ml = simulator.position.entry_reason
                            # Вычисляем абсолютные цены TP/SL
                            if position_side_ml.value == "long":
                                tp_price = entry_price * (1 + tp_pct)
                                sl_price = entry_price * (1 - sl_pct)
                            else:  # SHORT
                                tp_price = entry_price * (1 - tp_pct)
                                sl_price = entry_price * (1 + sl_pct)
                            position_tp_sl[entry_reason_ml] = {"tp": tp_price, "sl": sl_price}
                    
                    # ВАЖНО: Если позиция была только что открыта на этой свече, проверяем SL сразу
                    if not was_position_open and simulator.position:
                        # Сохраняем данные позиции перед использованием
                        position_side_new = simulator.position.side
                        entry_price = simulator.position.avg_price
                        entry_reason = simulator.position.entry_reason
                        
                        # Для Momentum стратегии: не открываем позиции слишком близко к концу данных
                        # (минимум 10 свечей до конца, чтобы было время для выхода)
                        if strategy_name == "momentum":
                            # Находим индекс текущей свечи в df
                            try:
                                current_idx = df.index.get_loc(timestamp)
                                remaining_candles = len(df) - current_idx - 1
                                if remaining_candles < 10:
                                    # Закрываем позицию сразу, если осталось мало свечей
                                    simulator._close(current_price, f"{strategy_name}_too_close_to_end", timestamp)
                                    continue
                            except:
                                pass  # Если не удалось найти индекс, пропускаем проверку
                        
                        # Проверяем, что позиция все еще открыта
                        if not simulator.position:
                            continue
                        
                        # Определяем TP/SL для новой позиции
                        tp_price = None
                        sl_price = None
                        
                        if (strategy_name == "ict" or strategy_name == "ml") and entry_reason in position_tp_sl:
                            tp_price = position_tp_sl[entry_reason]["tp"]
                            sl_price = position_tp_sl[entry_reason]["sl"]
                        else:
                            if strategy_name == "flat":
                                sl_pct = settings.strategy.range_stop_loss_pct
                                tp_pct = settings.risk.take_profit_pct
                            elif strategy_name == "vbo":
                                # Для VBO используем сбалансированные TP/SL: TP=3.2%, SL=1.1% от цены
                                tp_pct = 0.032  # 3.2% от цены (32% от маржи при 10x)
                                sl_pct = 0.011  # 1.1% от цены (11% от маржи при 10x) - RR ~2.9:1
                            elif strategy_name == "liquidation_hunter":
                                # Для Liquidation Hunter используем специальные TP/SL: TP=1.8%, SL=0.7% от цены
                                tp_pct = 0.018  # 1.8% от цены
                                sl_pct = 0.007  # 0.7% от цены
                            elif strategy_name == "zscore":
                                # Для ZSCORE используем оптимизированные TP/SL: TP=2.0%, SL=0.8% от цены
                                # ZSCORE - mean reversion стратегия, нужны быстрые тейки с хорошим RR
                                tp_pct = 0.020  # 2.0% от цены (20% от маржи при 10x)
                                sl_pct = 0.008  # 0.8% от цены (8% от маржи при 10x) - RR ~2.5:1
                            else:
                                sl_pct = settings.risk.stop_loss_pct
                                tp_pct = settings.risk.take_profit_pct
                            
                            if position_side_new.value == "long":
                                tp_price = entry_price * (1 + tp_pct)
                                sl_price = entry_price * (1 - sl_pct)
                            else:
                                tp_price = entry_price * (1 - tp_pct)
                                sl_price = entry_price * (1 + sl_pct)
                        
                        # Проверяем SL/TP сразу на этой свече
                        # Проверяем, что позиция все еще открыта
                        if not simulator.position:
                            continue
                        if position_side_new.value == "long":
                            # Если цена входа уже за SL, закрываем сразу
                            if sl_price and entry_price <= sl_price:
                                simulator._close(entry_price, f"{strategy_name}_sl_hit_on_entry", timestamp)
                                if not simulator.position:
                                    continue
                            if tp_price and high >= tp_price:
                                simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                                if not simulator.position:
                                    continue
                            if sl_price:
                                if low <= sl_price:
                                    simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                                    if not simulator.position:
                                        continue
                                elif current_price <= sl_price:
                                    simulator._close(current_price, f"{strategy_name}_sl_hit_gap", timestamp)
                                    if not simulator.position:
                                        continue
                        else:  # SHORT
                            # Если цена входа уже за SL, закрываем сразу
                            if sl_price and entry_price >= sl_price:
                                simulator._close(entry_price, f"{strategy_name}_sl_hit_on_entry", timestamp)
                                if not simulator.position:
                                    continue
                            if tp_price and low <= tp_price:
                                simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                                if not simulator.position:
                                    continue
                            if sl_price:
                                if high >= sl_price:
                                    simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                                    if not simulator.position:
                                        continue
                                elif current_price >= sl_price:
                                    simulator._close(current_price, f"{strategy_name}_sl_hit_gap", timestamp)
                                    if not simulator.position:
                                        continue
        
        # Закрываем последнюю позицию, если она открыта
        if simulator.position:
            last_row = df.iloc[-1]
            last_price = last_row['close']
            last_high = last_row.get('high', last_price)
            last_low = last_row.get('low', last_price)
            last_timestamp = df.index[-1]
            # Сохраняем данные позиции перед использованием
            position_side_end = simulator.position.side
            entry_price = simulator.position.avg_price
            entry_reason = simulator.position.entry_reason
            
            # Определяем TP/SL для последней позиции
            tp_price = None
            sl_price = None
            
            if (strategy_name == "ict" or strategy_name == "ml") and entry_reason in position_tp_sl:
                tp_price = position_tp_sl[entry_reason]["tp"]
                sl_price = position_tp_sl[entry_reason]["sl"]
            else:
                if strategy_name == "flat":
                    sl_pct = settings.strategy.range_stop_loss_pct
                    tp_pct = settings.risk.take_profit_pct
                elif strategy_name == "vbo":
                    # Для VBO используем сбалансированные TP/SL: TP=3.2%, SL=1.1% от цены
                    tp_pct = 0.032  # 3.2% от цены (32% от маржи при 10x)
                    sl_pct = 0.011  # 1.1% от цены (11% от маржи при 10x) - RR ~2.9:1
                elif strategy_name == "liquidation_hunter":
                    # Для Liquidation Hunter используем специальные TP/SL: TP=1.8%, SL=0.7% от цены
                    tp_pct = 0.018  # 1.8% от цены
                    sl_pct = 0.007  # 0.7% от цены
                elif strategy_name == "zscore":
                    # Для ZSCORE используем оптимизированные TP/SL: TP=2.0%, SL=0.8% от цены
                    # ZSCORE - mean reversion стратегия, нужны быстрые тейки с хорошим RR
                    tp_pct = 0.020  # 2.0% от цены (20% от маржи при 10x)
                    sl_pct = 0.008  # 0.8% от цены (8% от маржи при 10x) - RR ~2.5:1
                else:
                    sl_pct = settings.risk.stop_loss_pct
                    tp_pct = settings.risk.take_profit_pct
                
                if position_side_end.value == "long":
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                else:
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
            
            # Проверяем TP/SL перед закрытием по end_of_data
            # Используем сохраненный position_side_end
            if position_side_end.value == "long":
                if tp_price and last_high >= tp_price:
                    simulator._close(tp_price, f"{strategy_name}_tp_hit", last_timestamp)
                elif sl_price and last_low <= sl_price:
                    simulator._close(sl_price, f"{strategy_name}_sl_hit", last_timestamp)
                elif sl_price and last_price <= sl_price:
                    simulator._close(last_price, f"{strategy_name}_sl_hit_gap", last_timestamp)
                else:
                    simulator._close(last_price, f"{strategy_name}_end_of_data", last_timestamp)
            else:  # SHORT
                if tp_price and last_low <= tp_price:
                    simulator._close(tp_price, f"{strategy_name}_tp_hit", last_timestamp)
                elif sl_price and last_high >= sl_price:
                    simulator._close(sl_price, f"{strategy_name}_sl_hit", last_timestamp)
                elif sl_price and last_price >= sl_price:
                    simulator._close(last_price, f"{strategy_name}_sl_hit_gap", last_timestamp)
                else:
                    simulator._close(last_price, f"{strategy_name}_end_of_data", last_timestamp)
        
        # Собираем статистику
        trades = simulator.trades
        if len(trades) == 0:
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
        
        profitable = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        avg_win = sum(t.pnl for t in profitable) / len(profitable) if profitable else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        max_win = max((t.pnl for t in trades), default=0)
        max_loss = min((t.pnl for t in trades), default=0)
        
        win_rate = len(profitable) / len(trades) * 100 if trades else 0
        
        total_wins = sum(t.pnl for t in profitable) if profitable else 0
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return StrategyResult(
            strategy=strategy_name,
            symbol=symbol,
            total_trades=len(trades),
            profitable=len(profitable),
            losing=len(losing),
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


def generate_report(strategies: List[str], symbols: List[str], days: int = 10, output_file: Optional[str] = None):
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


def optimize_strategies_auto(symbols: List[str] = None, days: int = 10, min_pnl: float = 0.0, min_win_rate: float = 0.0, progress_callback=None) -> Dict:
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
    parser.add_argument("--days", type=int, default=10,
                       help="Количество дней для тестирования (по умолчанию: 10)")
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
