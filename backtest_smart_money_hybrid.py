"""
Бэктест гибридной стратегии SMART_MONEY_HYBRID с сравнением результатов.

Тестирует гибридную стратегию на исторических данных и сравнивает результаты
с оригинальными стратегиями ICT и SMC для оценки эффективности объединения.
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
from bot.smart_money_hybrid import build_smart_money_signals
from bot.ict_strategy import build_ict_signals
from bot.smc_strategy import build_smc_signals
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


def run_smart_money_backtest(
    symbols: List[str] = ["BTCUSDT"],
    timeframe: str = "15m",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    data_dir: str = "data",
    output_dir: str = "results",
    days_back: int = 30,
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
    print(f"SMART_MONEY_HYBRID BACKTEST")
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
            
            print(f"✅ Indicators prepared")
            
            comparisons = []
            
            # 1. Тестируем ГИБРИДНУЮ стратегию
            print(f"\n🔗 Testing HYBRID strategy...")
            try:
                # Диагностика: проверяем зоны отдельно
                from bot.ict_strategy import ICTStrategy
                from bot.smc_strategy import SMCStrategy
                ict_strategy = ICTStrategy(settings.strategy)
                smc_strategy = SMCStrategy(settings.strategy)
                
                # Проверяем ICT зоны
                try:
                    liquidity_sweeps = ict_strategy.find_liquidity_sweeps(df_ready)
                    if not liquidity_sweeps:
                        liquidity_sweeps = ict_strategy.find_liquidity_sweeps_alternative(df_ready, 50)
                    ict_fvg_zones = ict_strategy.find_fvg(df_ready, liquidity_sweeps)
                    print(f"   📊 ICT zones: {len(ict_fvg_zones)} FVG zones found")
                except Exception as e:
                    print(f"   ⚠️  ICT zones error: {e}")
                    ict_fvg_zones = []
                
                # Проверяем SMC зоны
                try:
                    highs = df_ready['high'].values
                    lows = df_ready['low'].values
                    opens = df_ready['open'].values
                    closes = df_ready['close'].values
                    if 'timestamp' in df_ready.columns:
                        times = df_ready['timestamp'].values
                    else:
                        times = df_ready.index.values
                    smc_fvg_zones = smc_strategy._find_fvg(df_ready, highs, lows, opens, closes, times)
                    smc_ob_zones = smc_strategy._find_ob(df_ready, highs, lows, opens, closes, times)
                    print(f"   📊 SMC zones: {len(smc_fvg_zones)} FVG, {len(smc_ob_zones)} OB zones found")
                except Exception as e:
                    print(f"   ⚠️  SMC zones error: {e}")
                    smc_fvg_zones = []
                    smc_ob_zones = []
                
                if not ict_fvg_zones:
                    print(f"   ⚠️  No ICT FVG zones found - HYBRID requires both ICT and SMC zones")
                if not smc_fvg_zones and not smc_ob_zones:
                    print(f"   ⚠️  No SMC zones found - HYBRID requires both ICT and SMC zones")
                
                # Дополнительная диагностика для активных зон
                if ict_fvg_zones:
                    active_ict_zones = [z for z in ict_fvg_zones if z.active]
                    print(f"   📊 Active ICT zones: {len(active_ict_zones)}/{len(ict_fvg_zones)}")
                    
                    # Проверяем возраст зон
                    current_idx = len(df_ready) - 1
                    max_age = getattr(settings.strategy, 'hybrid_max_zone_age_bars', 200)
                    recent_ict_zones = [z for z in active_ict_zones if (current_idx - z.bar_index) <= max_age]
                    print(f"   📊 Recent active ICT zones (age <= {max_age}): {len(recent_ict_zones)}")
                    
                    # Показываем распределение возрастов активных зон
                    if active_ict_zones:
                        ages = [current_idx - z.bar_index for z in active_ict_zones]
                        print(f"   📊 Active zone ages: min={min(ages)}, max={max(ages)}, avg={sum(ages)//len(ages)}")
                        
                        # Проверяем, сколько активных зон не митигированы по цене
                        last_row = df_ready.iloc[-1]
                        current_price = float(last_row['close'])
                        valid_price_zones = []
                        for zone in active_ict_zones:
                            if zone.direction == "bullish" and current_price >= zone.lower:
                                valid_price_zones.append(zone)
                            elif zone.direction == "bearish" and current_price <= zone.upper:
                                valid_price_zones.append(zone)
                        print(f"   📊 Active zones with valid price context: {len(valid_price_zones)}/{len(active_ict_zones)}")
                
                # Включаем debug логирование для диагностики
                import logging
                debug_logger = logging.getLogger('bot.smart_money_hybrid')
                smc_debug_logger = logging.getLogger('bot.smc_strategy')
                original_level_hybrid = debug_logger.level
                original_level_smc = smc_debug_logger.level
                debug_logger.setLevel(logging.DEBUG)
                smc_debug_logger.setLevel(logging.DEBUG)
                
                # Настраиваем handler для вывода в консоль
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
                console_handler.setFormatter(formatter)
                debug_logger.addHandler(console_handler)
                smc_debug_logger.addHandler(console_handler)
                
                hybrid_signals = build_smart_money_signals(df_ready, settings.strategy, symbol)
                
                # Восстанавливаем уровень логирования и удаляем handler
                debug_logger.removeHandler(console_handler)
                smc_debug_logger.removeHandler(console_handler)
                debug_logger.setLevel(original_level_hybrid)
                smc_debug_logger.setLevel(original_level_smc)
                
                print(f"   Generated {len(hybrid_signals)} signals")
                
                if len(hybrid_signals) == 0 and ict_fvg_zones and (smc_fvg_zones or smc_ob_zones):
                    print(f"   ⚠️  Zones found but no signals - possible reasons:")
                    print(f"      - Overlap tolerance: {getattr(settings.strategy, 'hybrid_fvg_overlap_tolerance', 0.3)}")
                    print(f"      - Require both trend filters: {getattr(settings.strategy, 'hybrid_require_both_trend_filters', True)}")
                    print(f"      - Require OB confirmation: {getattr(settings.strategy, 'hybrid_require_ob_confirmation', True)}")
                    print(f"      - Check debug logs for detailed filter stats")
                
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
            
            # 2. Тестируем ICT стратегию
            print(f"\n📈 Testing ICT strategy...")
            try:
                ict_signals = build_ict_signals(df_ready, settings.strategy, symbol)
                print(f"   Generated {len(ict_signals)} signals")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, ict_signals, symbol)
                ict_trades = result["trades"]
                ict_metrics = calculate_base_metrics(
                    ict_trades, initial_balance, ict_signals, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="ICT",
                    metrics=ict_metrics,
                    trades=ict_trades,
                    signals_count=len(ict_signals),
                ))
                
                print(f"   ✅ ICT: {ict_metrics.total_trades} trades, "
                      f"WR: {ict_metrics.win_rate:.1f}%, "
                      f"PnL: ${ict_metrics.total_pnl:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error testing ICT: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. Тестируем SMC стратегию
            print(f"\n📊 Testing SMC strategy...")
            try:
                # Диагностика SMC
                from bot.smc_strategy import SMCStrategy
                smc_strategy = SMCStrategy(settings.strategy)
                
                # Проверяем зоны
                highs = df_ready['high'].values
                lows = df_ready['low'].values
                opens = df_ready['open'].values
                closes = df_ready['close'].values
                if 'timestamp' in df_ready.columns:
                    times = df_ready['timestamp'].values
                else:
                    times = df_ready.index.values
                
                smc_fvg_zones = smc_strategy._find_fvg(df_ready, highs, lows, opens, closes, times)
                smc_ob_zones = smc_strategy._find_ob(df_ready, highs, lows, opens, closes, times)
                print(f"   📊 Found {len(smc_fvg_zones)} FVG zones, {len(smc_ob_zones)} OB zones")
                
                # Проверяем фильтр сессий
                session_filter_enabled = getattr(settings.strategy, 'smc_enable_session_filter', True)
                print(f"   📅 Session filter: {'ENABLED' if session_filter_enabled else 'DISABLED'}")
                
                if session_filter_enabled:
                    last_row = df_ready.iloc[-1]
                    last_ts = last_row.get('timestamp', last_row.name)
                    if not isinstance(last_ts, pd.Timestamp):
                        last_ts = pd.to_datetime(last_ts)
                    is_session = smc_strategy._is_trading_session(last_ts)
                    print(f"   📅 Current time in trading session: {is_session}")
                    if not is_session:
                        print(f"   ⚠️  Session filter blocking signals (current hour: {last_ts.hour} UTC)")
                        print(f"   💡 Disabling session filter for backtest...")
                        # Отключаем фильтр сессий для бэктеста
                        original_value = settings.strategy.smc_enable_session_filter
                        settings.strategy.smc_enable_session_filter = False
                
                # Включаем debug логирование для SMC диагностики
                import logging
                smc_debug_logger = logging.getLogger('bot.smc_strategy')
                original_level_smc = smc_debug_logger.level
                smc_debug_logger.setLevel(logging.DEBUG)
                
                # Настраиваем handler для вывода в консоль
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
                console_handler.setFormatter(formatter)
                smc_debug_logger.addHandler(console_handler)
                
                smc_signals = build_smc_signals(df_ready, settings.strategy, symbol)
                
                # Восстанавливаем уровень логирования и удаляем handler
                smc_debug_logger.removeHandler(console_handler)
                smc_debug_logger.setLevel(original_level_smc)
                
                # Восстанавливаем оригинальное значение фильтра сессий
                if session_filter_enabled and not is_session:
                    settings.strategy.smc_enable_session_filter = original_value
                
                print(f"   Generated {len(smc_signals)} signals")
                
                # Диагностика если нет сигналов
                if len(smc_signals) == 0 and (len(smc_fvg_zones) > 0 or len(smc_ob_zones) > 0):
                    print(f"   ⚠️  Zones found but no signals - checking filters:")
                    current_idx = len(df_ready) - 1
                    max_fvg_age = getattr(settings.strategy, 'smc_max_fvg_age_bars', 200)
                    max_ob_age = getattr(settings.strategy, 'smc_max_ob_age_bars', 300)
                    recent_fvg = [z for z in smc_fvg_zones if (current_idx - z.bar_index) <= max_fvg_age]
                    recent_ob = [z for z in smc_ob_zones if (current_idx - z.bar_index) <= max_ob_age]
                    print(f"      - Recent FVG zones (age <= {max_fvg_age}): {len(recent_fvg)}/{len(smc_fvg_zones)}")
                    print(f"      - Recent OB zones (age <= {max_ob_age}): {len(recent_ob)}/{len(smc_ob_zones)}")
                    
                    # Проверяем тренд фильтр
                    last_row = df_ready.iloc[-1]
                    close_price = last_row['close']
                    ema_200 = df_ready['close'].ewm(span=200, adjust=False).mean().iloc[-1]
                    is_bullish = close_price > ema_200
                    is_bearish = close_price < ema_200
                    print(f"      - Trend context: {'Bullish' if is_bullish else 'Bearish' if is_bearish else 'Neutral'} (Price: ${close_price:.2f}, EMA200: ${ema_200:.2f})")
                    
                    # Проверяем касание зон
                    if recent_fvg or recent_ob:
                        touch_tolerance = getattr(settings.strategy, 'smc_touch_tolerance_pct', 0.001)
                        zones_in_range = 0
                        for zone in recent_fvg + recent_ob:
                            if zone.direction == "bullish":
                                if last_row['low'] <= (zone.upper + zone.upper * touch_tolerance) and close_price > zone.lower:
                                    zones_in_range += 1
                            elif zone.direction == "bearish":
                                if last_row['high'] >= (zone.lower - zone.lower * touch_tolerance) and close_price < zone.upper:
                                    zones_in_range += 1
                        print(f"      - Zones in touch range: {zones_in_range}/{len(recent_fvg) + len(recent_ob)}")
                        print(f"      - Touch tolerance: {touch_tolerance*100:.3f}%")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, smc_signals, symbol)
                smc_trades = result["trades"]
                smc_metrics = calculate_base_metrics(
                    smc_trades, initial_balance, smc_signals, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="SMC",
                    metrics=smc_metrics,
                    trades=smc_trades,
                    signals_count=len(smc_signals),
                ))
                
                print(f"   ✅ SMC: {smc_metrics.total_trades} trades, "
                      f"WR: {smc_metrics.win_rate:.1f}%, "
                      f"PnL: ${smc_metrics.total_pnl:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error testing SMC: {e}")
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
                    ict_comp = next((c for c in comparisons if c.strategy_name == "ICT"), None)
                    smc_comp = next((c for c in comparisons if c.strategy_name == "SMC"), None)
                    
                    print(f"\n📊 HYBRID vs Original Strategies:")
                    if ict_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - ict_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - ict_comp.metrics.win_rate
                        trades_diff = hybrid_comp.metrics.total_trades - ict_comp.metrics.total_trades
                        print(f"   vs ICT: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%, "
                              f"Trades diff: {trades_diff:+d}")
                    
                    if smc_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - smc_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - smc_comp.metrics.win_rate
                        trades_diff = hybrid_comp.metrics.total_trades - smc_comp.metrics.total_trades
                        print(f"   vs SMC: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%, "
                              f"Trades diff: {trades_diff:+d}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
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
        comparison_file = os.path.join(output_dir, f"smart_money_comparison_{timestamp}.csv")
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_file, index=False)
        print(f"✅ Comparison report saved: {comparison_file}")
    
    # Текстовый отчет с рекомендациями
    report_file = os.path.join(output_dir, f"smart_money_backtest_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SMART_MONEY_HYBRID BACKTEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        for symbol, comparisons in all_comparisons.items():
            f.write(f"\n{symbol}\n")
            f.write("-"*80 + "\n")
            
            hybrid_comp = next((c for c in comparisons if c.strategy_name == "HYBRID"), None)
            ict_comp = next((c for c in comparisons if c.strategy_name == "ICT"), None)
            smc_comp = next((c for c in comparisons if c.strategy_name == "SMC"), None)
            
            if hybrid_comp and ict_comp and smc_comp:
                f.write(f"\nHYBRID Results:\n")
                f.write(f"  Trades: {hybrid_comp.metrics.total_trades}\n")
                f.write(f"  Win Rate: {hybrid_comp.metrics.win_rate:.1f}%\n")
                f.write(f"  PnL: ${hybrid_comp.metrics.total_pnl:.2f}\n")
                f.write(f"  Profit Factor: {hybrid_comp.metrics.profit_factor:.2f}\n")
                f.write(f"  Signals: {hybrid_comp.signals_count}\n")
                
                f.write(f"\nComparison:\n")
                pnl_vs_ict = hybrid_comp.metrics.total_pnl - ict_comp.metrics.total_pnl
                pnl_vs_smc = hybrid_comp.metrics.total_pnl - smc_comp.metrics.total_pnl
                wr_vs_ict = hybrid_comp.metrics.win_rate - ict_comp.metrics.win_rate
                wr_vs_smc = hybrid_comp.metrics.win_rate - smc_comp.metrics.win_rate
                
                f.write(f"  vs ICT: PnL {pnl_vs_ict:+.2f}, WR {wr_vs_ict:+.1f}%\n")
                f.write(f"  vs SMC: PnL {pnl_vs_smc:+.2f}, WR {wr_vs_smc:+.1f}%\n")
                
                # Рекомендации
                f.write(f"\nRecommendations:\n")
                if pnl_vs_ict > 0 and pnl_vs_smc > 0:
                    f.write(f"  ✅ HYBRID outperforms both strategies!\n")
                    f.write(f"  → Consider implementing in live trading\n")
                elif pnl_vs_ict > 0 or pnl_vs_smc > 0:
                    f.write(f"  ⚠️ HYBRID outperforms one strategy\n")
                    f.write(f"  → Consider optimizing parameters\n")
                else:
                    f.write(f"  ❌ HYBRID underperforms both strategies\n")
                    f.write(f"  → Review logic and parameters\n")
                
                # Анализ сигналов
                if hybrid_comp.signals_count < ict_comp.signals_count:
                    f.write(f"\n  Note: HYBRID generates fewer signals ({hybrid_comp.signals_count} vs {ict_comp.signals_count})\n")
                    f.write(f"  → This is expected due to double filtering\n")
                    f.write(f"  → Quality over quantity approach\n")
    
    print(f"✅ Report saved: {report_file}")
    
    return {
        "comparisons": all_comparisons,
        "timestamp": timestamp,
    }


def main():
    """Главная функция для запуска бэктеста."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SMART_MONEY_HYBRID Backtest")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTCUSDT"],
        help="Symbols to test (default: BTCUSDT)",
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
    
    args = parser.parse_args()
    
    run_smart_money_backtest(
        symbols=args.symbols,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        days_back=args.days,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
