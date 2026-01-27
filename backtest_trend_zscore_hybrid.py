"""
Бэктест гибридной стратегии TREND_ZSCORE_HYBRID с сравнением результатов.

Тестирует гибридную стратегию на исторических данных и сравнивает результаты
с оригинальными стратегиями TREND и ZSCORE для оценки эффективности объединения.
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, StrategyParams
from bot.trend_zscore_hybrid import build_trend_zscore_signals
from bot.strategy import build_signals, generate_trend_signal
from bot.zscore_strategy import build_zscore_signals
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal
from backtest_zscore_strategy import (
    ZScoreBacktestSimulator,
    Trade,
    BacktestMetrics,
    calculate_metrics as calculate_base_metrics,
)


@dataclass
class StrategyComparison:
    """Результаты сравнения стратегий."""
    strategy_name: str
    metrics: BacktestMetrics
    trades: List[Trade]
    signals_count: int


def run_hybrid_backtest(
    symbols: List[str] = ["SOLUSDT", "ETHUSDT"],
    timeframe: str = "15m",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    data_dir: str = "data",
    output_dir: str = "results",
    days_back: int = 30,
    relax_filters: bool = False,  # Новый параметр для ослабления фильтров
    ultra_relaxed: bool = False,  # Ультра-ослабленный режим для тестирования
    show_diagnostics: bool = True,  # Показывать диагностику фильтров
) -> Dict[str, Any]:
    """
    Запускает бэктест гибридной стратегии с сравнением результатов.
    
    Args:
        symbols: Список символов для тестирования
        timeframe: Таймфрейм данных
        initial_balance: Начальный баланс
        risk_per_trade: Риск на сделку (доля от баланса)
        data_dir: Директория с историческими данными
        output_dir: Директория для сохранения результатов
        days_back: Количество дней назад для загрузки данных
        
    Returns:
        Словарь с результатами сравнения стратегий
    """
    from bot.exchange.bybit_client import BybitClient
    from bot.config import load_settings
    
    settings = load_settings()
    client = BybitClient(settings.api)
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print(f"TREND_ZSCORE_HYBRID BACKTEST")
    print(f"{'='*80}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Risk per Trade: {risk_per_trade*100:.1f}%")
    print(f"Days Back: {days_back}")
    print(f"{'='*80}\n")
    
    all_comparisons: Dict[str, List[StrategyComparison]] = {}
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING {symbol}")
        print(f"{'='*80}")
        
        try:
            # Загружаем данные
            print(f"📥 Loading data for {symbol}...")
            df_raw = client.get_kline_df(
                symbol=symbol,
                interval=timeframe,
                limit=days_back * 24 * 4 if timeframe == "15m" else days_back * 24,
            )
            
            if df_raw is None or df_raw.empty:
                print(f"⚠️  No data for {symbol}, skipping...")
                continue
            
            print(f"✅ Loaded {len(df_raw)} candles")
            
            # Подготавливаем индикаторы
            print(f"📊 Preparing indicators...")
            df_ready = prepare_with_indicators(
                df_raw,
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
            
            # Для бэктеста увеличиваем лимиты возраста зон, чтобы использовать больше исторических данных
            # Сохраняем оригинальные значения для восстановления
            original_fvg_age = settings.strategy.smc_max_fvg_age_bars
            original_ob_age = settings.strategy.smc_max_ob_age_bars
            # Увеличиваем до 50% от длины данных (для 30 дней = ~1440 баров)
            adaptive_age_limit = max(500, len(df_ready) // 2)
            settings.strategy.smc_max_fvg_age_bars = adaptive_age_limit
            settings.strategy.smc_max_ob_age_bars = adaptive_age_limit + 100
            
            print(f"✅ Indicators prepared")
            print(f"   📊 Adjusted zone age limits for backtest: FVG={adaptive_age_limit}, OB={adaptive_age_limit + 100}")
            
            # Если включено ослабление фильтров, применяем настройки
            # Инициализируем переменные перед условием для правильной области видимости
            original_vol_multiplier = None
            original_adx_threshold = None
            relax_filters_state = {}
            
            # Сохраняем оригинальные значения для восстановления (используем существующие атрибуты)
            original_adx_threshold = getattr(settings.strategy, 'adx_threshold', 25.0)
            original_vol_multiplier = None  # Будет установлен в strategy_params
            
            # Переменные для хранения ослабленных значений (будут использованы в strategy_params)
            relaxed_adx_threshold = None
            relaxed_vol_multiplier = None
            
            if ultra_relaxed:
                print(f"   ⚠️  ULTRA-RELAXED MODE: Maximum filter relaxation for testing")
                # Сохраняем ослабленные значения для передачи через strategy_params
                relaxed_adx_threshold = 10.0  # Очень низкий порог ADX
                relaxed_vol_multiplier = 0.3  # Очень низкое требование к объему
                # Устанавливаем также в settings.strategy для совместимости
                settings.strategy.adx_threshold = 10.0
                # Флаги для отключения guards и ослабления условий входа
                relax_filters_state = {
                    'relax_di_guard': True,
                    'relax_rsi_guard': True,
                    'ultra_relaxed': True,  # Флаг для ослабления условий тренда/breakout
                }
            elif relax_filters:
                print(f"   ⚠️  RELAXED FILTERS MODE: Some guards will be disabled for testing")
                # Сохраняем ослабленные значения
                relaxed_adx_threshold = 20.0  # Снижаем порог ADX
                relaxed_vol_multiplier = 0.8  # Снижаем требование к объему
                # Устанавливаем также в settings.strategy для совместимости
                settings.strategy.adx_threshold = 20.0
                # Флаги для отключения guards передадим через state в generate_trend_signal
                relax_filters_state = {
                    'relax_di_guard': True,
                    'relax_rsi_guard': True,
                }
            else:
                relax_filters_state = {}
            
            comparisons = []
            
            # Счетчики для диагностики фильтров
            filter_stats = {
                'hybrid': {
                    'total_checked': 0,
                    'trend_no_signal': 0,
                    'zscore_not_confirmed': 0,
                    'signals_generated': 0,
                },
                'trend': {
                    'total_checked': 0,
                    'no_entry_conditions': 0,
                    'no_trend_detected': 0,
                    'di_guard_blocked': 0,
                    'rsi_guard_blocked': 0,
                    'volume_filter_blocked': 0,
                    'adx_filter_blocked': 0,
                    'signals_generated': 0,
                },
                'zscore': {
                    'total_checked': 0,
                    'market_not_allowed': 0,
                    'volatility_filter_blocked': 0,
                    'volume_filter_blocked': 0,
                    'zscore_threshold_not_met': 0,
                    'rsi_filter_blocked': 0,
                    'signals_generated': 0,
                },
            }
            
            # 1. Тестируем ГИБРИДНУЮ стратегию
            print(f"\n🔗 Testing HYBRID strategy...")
            try:
                # Гибридная стратегия должна итерироваться по всем свечам, как TREND
                hybrid_state = {'backtest_mode': True}
                hybrid_signals = []
                
                # Устанавливаем флаг бэктеста в параметрах стратегии
                settings.strategy.backtest_mode = True
                
                # Генерируем сигналы для каждой свечи (начиная с 200 для достаточной истории)
                for i in range(200, len(df_ready)):
                    filter_stats['hybrid']['total_checked'] += 1
                    # Получаем данные до текущей свечи
                    df_slice = df_ready.iloc[:i+1]
                    
                    # Генерируем гибридный сигнал для текущей свечи
                    hybrid_result = build_trend_zscore_signals(df_slice, settings.strategy, symbol)
                    
                    # Если есть сигнал, добавляем его
                    if hybrid_result and len(hybrid_result) > 0:
                        # Берем последний сигнал (если их несколько)
                        signal = hybrid_result[-1]
                        # Обновляем timestamp на текущую свечу
                        signal.timestamp = df_ready.index[i]
                        signal.price = float(df_ready.iloc[i]['close'])
                        hybrid_signals.append(signal)
                        filter_stats['hybrid']['signals_generated'] += 1
                    else:
                        # Отслеживаем причины отсутствия сигнала (упрощенно)
                        filter_stats['hybrid']['trend_no_signal'] += 1
                
                print(f"   Generated {len(hybrid_signals)} signals")
                if show_diagnostics:
                    print(f"   📊 HYBRID Diagnostics:")
                    print(f"      Total candles checked: {filter_stats['hybrid']['total_checked']}")
                    print(f"      Signals generated: {filter_stats['hybrid']['signals_generated']}")
                    print(f"      Blocked (no trend signal): {filter_stats['hybrid']['trend_no_signal']}")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, hybrid_signals, symbol)
                hybrid_trades = result["trades"]
                hybrid_metrics = calculate_base_metrics(
                    hybrid_trades, initial_balance, hybrid_signals, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="HYBRID",
                    metrics=hybrid_metrics,
                    trades=hybrid_trades,
                    signals_count=len(hybrid_signals),
                ))
                
                print(f"   ✅ HYBRID: {hybrid_metrics.total_trades} trades, "
                      f"WR: {hybrid_metrics.win_rate:.1f}%, "
                      f"PnL: ${hybrid_metrics.total_pnl:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error testing HYBRID: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Восстанавливаем флаг бэктеста и фильтры
                if hasattr(settings.strategy, 'backtest_mode'):
                    settings.strategy.backtest_mode = False
                # Восстанавливаем оригинальные значения фильтров
                if (relax_filters or ultra_relaxed) and original_adx_threshold is not None:
                    settings.strategy.adx_threshold = original_adx_threshold
            
            # 2. Тестируем TREND стратегию
            print(f"\n📈 Testing TREND strategy...")
            try:
                trend_state = {
                    'backtest_mode': True,
                    'enable_time_filter': False,  # Отключаем фильтр по времени в бэктесте
                }
                trend_signals_list = []
                
                # Используем build_signals как в рабочем тесте backtest_strategy_mtf.py
                # Подготавливаем параметры для стратегии
                # ВАЖНО: передаем adx_threshold и vol_multiplier через params, чтобы они использовались
                # Используем ослабленные значения, если они установлены, иначе стандартные
                if relaxed_adx_threshold is not None:
                    adx_threshold_value = relaxed_adx_threshold
                else:
                    adx_threshold_value = getattr(settings.strategy, 'adx_threshold', 25.0)
                
                if relaxed_vol_multiplier is not None:
                    vol_multiplier_value = relaxed_vol_multiplier
                else:
                    vol_multiplier_value = 1.3  # Стандартное значение
                
                if show_diagnostics:
                    print(f"   📊 Strategy params: ADX threshold={adx_threshold_value}, Vol multiplier={vol_multiplier_value}")
                
                strategy_params = {
                    'sma_period': getattr(settings.strategy, 'sma_length', 21),
                    'atr_period': getattr(settings.strategy, 'adx_length', 14),
                    'atr_multiplier': getattr(settings.strategy, 'trend_atr_multiplier', 3.0),
                    'max_pyramid': getattr(settings.strategy, 'trend_max_pyramid', 2),
                    'min_history': 100,
                    'adx_threshold': adx_threshold_value,
                    'vol_multiplier': vol_multiplier_value,
                }
                
                # Генерируем сигналы для каждой свечи
                min_history = strategy_params.get('min_history', 100)
                for i in range(min_history, len(df_ready)):
                    filter_stats['trend']['total_checked'] += 1
                    # Берем данные до текущего момента (включительно)
                    df_slice = df_ready.iloc[:i+1]
                    
                    # Объединяем состояние бэктеста с флагами ослабления фильтров
                    combined_state = {**trend_state, **relax_filters_state}
                    combined_state['last_signal_idx'] = -100  # Сбрасываем cooldown для бэктеста
                    
                    # Используем build_signals как в рабочем тесте
                    try:
                        candle_signals = build_signals(
                            df_slice,
                            settings.strategy,
                            use_momentum=False,
                            use_liquidity=False,
                            params=strategy_params,
                            state=combined_state,
                        )
                        
                        # Обрабатываем сигналы
                        for signal in candle_signals:
                            if signal.action in (Action.LONG, Action.SHORT):
                                # Обновляем timestamp на текущую свечу
                                signal.timestamp = df_ready.index[i]
                                signal.price = float(df_ready.iloc[i]['close'])
                                trend_signals_list.append(signal)
                                filter_stats['trend']['signals_generated'] += 1
                    except Exception as e:
                        # Пропускаем ошибки для отдельных свечей
                        continue
                    
                    # Для диагностики проверяем результат (если нет сигналов)
                    if not candle_signals or all(s.action == Action.HOLD for s in candle_signals):
                        # Используем generate_trend_signal для диагностики
                        # ВАЖНО: передаем те же параметры, что и в build_signals
                        trend_result = generate_trend_signal(
                            df_slice,
                            state=combined_state,
                            sma_period=strategy_params['sma_period'],
                            atr_period=strategy_params['atr_period'],
                            atr_multiplier=strategy_params['atr_multiplier'],
                            max_pyramid=strategy_params['max_pyramid'],
                            min_history=strategy_params['min_history'],
                            adx_threshold=strategy_params['adx_threshold'],
                            vol_multiplier=strategy_params['vol_multiplier'],
                        )
                        
                        # Отслеживаем причины отсутствия сигнала
                        reason = trend_result.get('reason', '') if trend_result else 'no_result'
                        indicators_info = trend_result.get('indicators_info', {}) if trend_result else {}
                        entry_conditions = indicators_info.get('entry_conditions', {})
                        
                        # Получаем данные индикаторов для точной диагностики
                        plus_di = indicators_info.get('plus_di')
                        minus_di = indicators_info.get('minus_di')
                        rsi = indicators_info.get('rsi')
                        
                        # Проверяем, был ли сигнал разрешен изначально (long_allowed или short_allowed)
                        was_allowed = 'long_allowed' in str(reason).lower() or 'short_allowed' in str(reason).lower() or \
                                     'breakout' in str(reason).lower() or 'pullback' in str(reason).lower()
                        
                        if was_allowed and plus_di is not None and minus_di is not None:
                            # Сигнал был разрешен, но заблокирован guard'ами
                            if not relax_filters_state.get('relax_di_guard', False):
                                if (plus_di <= minus_di) or (minus_di <= plus_di):
                                    filter_stats['trend']['di_guard_blocked'] += 1
                                elif rsi is not None and not relax_filters_state.get('relax_rsi_guard', False):
                                    if (rsi >= 70) or (rsi <= 30):
                                        filter_stats['trend']['rsi_guard_blocked'] += 1
                                    else:
                                        filter_stats['trend']['no_entry_conditions'] += 1
                                else:
                                    filter_stats['trend']['no_entry_conditions'] += 1
                            elif rsi is not None and not relax_filters_state.get('relax_rsi_guard', False):
                                if (rsi >= 70) or (rsi <= 30):
                                    filter_stats['trend']['rsi_guard_blocked'] += 1
                                else:
                                    filter_stats['trend']['no_entry_conditions'] += 1
                            else:
                                if 'volume' in reason.lower() or 'vol' in reason.lower():
                                    filter_stats['trend']['volume_filter_blocked'] += 1
                                elif 'adx' in reason.lower():
                                    filter_stats['trend']['adx_filter_blocked'] += 1
                                else:
                                    filter_stats['trend']['no_entry_conditions'] += 1
                        else:
                            # Базовые условия не выполнены
                            if entry_conditions:
                                if not entry_conditions.get('adx_ok', True):
                                    filter_stats['trend']['adx_filter_blocked'] += 1
                                elif not entry_conditions.get('vol_ok', True):
                                    filter_stats['trend']['volume_filter_blocked'] += 1
                                elif not entry_conditions.get('is_uptrend', False) and not entry_conditions.get('is_breakout_long', False) and \
                                     not entry_conditions.get('is_downtrend', False) and not entry_conditions.get('is_breakout_short', False):
                                    filter_stats['trend']['no_trend_detected'] += 1
                                else:
                                    filter_stats['trend']['no_entry_conditions'] += 1
                            else:
                                filter_stats['trend']['no_entry_conditions'] += 1
                
                print(f"   Generated {len(trend_signals_list)} signals")
                if show_diagnostics:
                    print(f"   📊 TREND Diagnostics:")
                    print(f"      Total candles checked: {filter_stats['trend']['total_checked']}")
                    print(f"      Signals generated: {filter_stats['trend']['signals_generated']}")
                    print(f"      Blocked - No trend/breakout detected: {filter_stats['trend']['no_trend_detected']}")
                    print(f"      Blocked - ADX filter: {filter_stats['trend']['adx_filter_blocked']}")
                    print(f"      Blocked - Volume filter: {filter_stats['trend']['volume_filter_blocked']}")
                    print(f"      Blocked - DI guard: {filter_stats['trend']['di_guard_blocked']}")
                    print(f"      Blocked - RSI guard: {filter_stats['trend']['rsi_guard_blocked']}")
                    print(f"      Blocked - Other: {filter_stats['trend']['no_entry_conditions']}")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, trend_signals_list, symbol)
                trend_trades = result["trades"]
                trend_metrics = calculate_base_metrics(
                    trend_trades, initial_balance, trend_signals_list, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="TREND",
                    metrics=trend_metrics,
                    trades=trend_trades,
                    signals_count=len(trend_signals_list),
                ))
                
                print(f"   ✅ TREND: {trend_metrics.total_trades} trades, "
                      f"WR: {trend_metrics.win_rate:.1f}%, "
                      f"PnL: ${trend_metrics.total_pnl:.2f}")
                if show_diagnostics and filter_stats['trend']['signals_generated'] == 0:
                    total_blocked = (filter_stats['trend']['di_guard_blocked'] + 
                                    filter_stats['trend']['rsi_guard_blocked'] + 
                                    filter_stats['trend']['volume_filter_blocked'] + 
                                    filter_stats['trend']['adx_filter_blocked'] + 
                                    filter_stats['trend']['no_entry_conditions'])
                    if relax_filters and filter_stats['trend']['di_guard_blocked'] > 0:
                        print(f"   💡 Note: {filter_stats['trend']['di_guard_blocked']} signals show DI guard blocking")
                        print(f"      even with --relax-filters. This may indicate DI strength check is still active.")
                
            except Exception as e:
                print(f"   ❌ Error testing TREND: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. Тестируем ZSCORE стратегию
            print(f"\n📊 Testing ZSCORE strategy...")
            try:
                # ZSCORE стратегия обрабатывает весь DataFrame сразу
                filter_stats['zscore']['total_checked'] = len(df_ready)
                zscore_signals = build_zscore_signals(df_ready, settings.strategy, symbol)
                filter_stats['zscore']['signals_generated'] = len(zscore_signals)
                print(f"   Generated {len(zscore_signals)} signals")
                if show_diagnostics:
                    print(f"   📊 ZSCORE Diagnostics:")
                    print(f"      Total candles checked: {filter_stats['zscore']['total_checked']}")
                    print(f"      Signals generated: {filter_stats['zscore']['signals_generated']}")
                    print(f"      (Detailed filter stats require v2 strategy implementation)")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, zscore_signals, symbol)
                zscore_trades = result["trades"]
                zscore_metrics = calculate_base_metrics(
                    zscore_trades, initial_balance, zscore_signals, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="ZSCORE",
                    metrics=zscore_metrics,
                    trades=zscore_trades,
                    signals_count=len(zscore_signals),
                ))
                
                print(f"   ✅ ZSCORE: {zscore_metrics.total_trades} trades, "
                      f"WR: {zscore_metrics.win_rate:.1f}%, "
                      f"PnL: ${zscore_metrics.total_pnl:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error testing ZSCORE: {e}")
                import traceback
                traceback.print_exc()
            
            all_comparisons[symbol] = comparisons
            
            # Выводим сравнение результатов
            print(f"\n{'='*80}")
            print(f"COMPARISON FOR {symbol}")
            print(f"{'='*80}")
            print(f"{'Strategy':<15} {'Trades':<8} {'WR %':<8} {'PnL $':<12} {'PF':<8} {'Signals':<8}")
            print(f"{'-'*80}")
            
            for comp in comparisons:
                print(f"{comp.strategy_name:<15} "
                      f"{comp.metrics.total_trades:<8} "
                      f"{comp.metrics.win_rate:<8.1f} "
                      f"${comp.metrics.total_pnl:<11.2f} "
                      f"{comp.metrics.profit_factor:<8.2f} "
                      f"{comp.signals_count:<8}")
            
            # Определяем лучшую стратегию
            if comparisons:
                best_strategy = max(comparisons, key=lambda c: c.metrics.total_pnl)
                print(f"\n🏆 Best Strategy: {best_strategy.strategy_name} "
                      f"(PnL: ${best_strategy.metrics.total_pnl:.2f})")
                
                # Сравниваем HYBRID с оригинальными
                hybrid_comp = next((c for c in comparisons if c.strategy_name == "HYBRID"), None)
                if hybrid_comp:
                    trend_comp = next((c for c in comparisons if c.strategy_name == "TREND"), None)
                    zscore_comp = next((c for c in comparisons if c.strategy_name == "ZSCORE"), None)
                    
                    print(f"\n📊 HYBRID vs Original Strategies:")
                    if trend_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - trend_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - trend_comp.metrics.win_rate
                        print(f"   vs TREND: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%")
                    
                    if zscore_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - zscore_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - zscore_comp.metrics.win_rate
                        print(f"   vs ZSCORE: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Восстанавливаем оригинальные значения возраста зон
            if 'original_fvg_age' in locals():
                settings.strategy.smc_max_fvg_age_bars = original_fvg_age
                settings.strategy.smc_max_ob_age_bars = original_ob_age
    
    # Сохраняем результаты сравнения
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    
    # CSV отчет с сравнением
    comparison_data = []
    for symbol, comparisons in all_comparisons.items():
        for comp in comparisons:
            comparison_data.append({
                "Symbol": symbol,
                "Strategy": comp.strategy_name,
                "Total Trades": comp.metrics.total_trades,
                "Winning Trades": comp.metrics.winning_trades,
                "Losing Trades": comp.metrics.losing_trades,
                "Win Rate %": f"{comp.metrics.win_rate:.2f}",
                "Total PnL": f"${comp.metrics.total_pnl:.2f}",
                "Total PnL %": f"{comp.metrics.total_pnl_pct:.2f}",
                "Profit Factor": f"{comp.metrics.profit_factor:.2f}",
                "Max Drawdown": f"${comp.metrics.max_drawdown:.2f}",
                "Max Drawdown %": f"{comp.metrics.max_drawdown_pct:.2f}",
                "Sharpe Ratio": f"{comp.metrics.sharpe_ratio:.2f}",
                "Avg Win": f"${comp.metrics.avg_win:.2f}",
                "Avg Loss": f"${comp.metrics.avg_loss:.2f}",
                "Total Signals": comp.signals_count,
            })
    
    if comparison_data:
        comparison_file = os.path.join(output_dir, f"hybrid_comparison_{timestamp}.csv")
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_file, index=False)
        print(f"✅ Comparison report saved: {comparison_file}")
    
    # Текстовый отчет с рекомендациями
    report_file = os.path.join(output_dir, f"hybrid_backtest_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TREND_ZSCORE_HYBRID BACKTEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        for symbol, comparisons in all_comparisons.items():
            f.write(f"\n{symbol}\n")
            f.write("-"*80 + "\n")
            
            hybrid_comp = next((c for c in comparisons if c.strategy_name == "HYBRID"), None)
            trend_comp = next((c for c in comparisons if c.strategy_name == "TREND"), None)
            zscore_comp = next((c for c in comparisons if c.strategy_name == "ZSCORE"), None)
            
            if hybrid_comp and trend_comp and zscore_comp:
                f.write(f"\nHYBRID Results:\n")
                f.write(f"  Trades: {hybrid_comp.metrics.total_trades}\n")
                f.write(f"  Win Rate: {hybrid_comp.metrics.win_rate:.1f}%\n")
                f.write(f"  PnL: ${hybrid_comp.metrics.total_pnl:.2f}\n")
                f.write(f"  Profit Factor: {hybrid_comp.metrics.profit_factor:.2f}\n")
                
                f.write(f"\nComparison:\n")
                pnl_vs_trend = hybrid_comp.metrics.total_pnl - trend_comp.metrics.total_pnl
                pnl_vs_zscore = hybrid_comp.metrics.total_pnl - zscore_comp.metrics.total_pnl
                wr_vs_trend = hybrid_comp.metrics.win_rate - trend_comp.metrics.win_rate
                wr_vs_zscore = hybrid_comp.metrics.win_rate - zscore_comp.metrics.win_rate
                
                f.write(f"  vs TREND: PnL {pnl_vs_trend:+.2f}, WR {wr_vs_trend:+.1f}%\n")
                f.write(f"  vs ZSCORE: PnL {pnl_vs_zscore:+.2f}, WR {wr_vs_zscore:+.1f}%\n")
                
                if pnl_vs_trend > 0 and pnl_vs_zscore > 0:
                    f.write(f"\n✅ HYBRID outperforms both strategies!\n")
                elif pnl_vs_trend > 0 or pnl_vs_zscore > 0:
                    f.write(f"\n⚠️ HYBRID outperforms one strategy\n")
                else:
                    f.write(f"\n❌ HYBRID underperforms both strategies\n")
    
    print(f"✅ Report saved: {report_file}")
    
    return {
        "comparisons": all_comparisons,
        "timestamp": timestamp,
    }


def main():
    """Главная функция для запуска бэктеста."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TREND_ZSCORE_HYBRID Backtest")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["SOLUSDT", "ETHUSDT"],
        help="Symbols to test (default: SOLUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe for backtest (default: 15m)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Initial balance (default: 1000.0)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.02,
        help="Risk per trade as fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days back for data (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--relax-filters",
        action="store_true",
        help="Relax filters for testing (disables DI guard, RSI guard, lowers volume/ADX requirements)",
    )
    parser.add_argument(
        "--ultra-relaxed",
        action="store_true",
        help="Ultra-relaxed mode: Maximum filter relaxation (volume 0.3x, ADX 10, relaxed trend/breakout conditions)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable filter diagnostics output",
    )
    
    args = parser.parse_args()
    
    run_hybrid_backtest(
        symbols=args.symbols,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        days_back=args.days,
        output_dir=args.output_dir,
        relax_filters=args.relax_filters,
        ultra_relaxed=args.ultra_relaxed,
        show_diagnostics=not args.no_diagnostics,
    )


if __name__ == "__main__":
    main()
