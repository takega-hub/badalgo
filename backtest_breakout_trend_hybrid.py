"""
Бэктест гибридной стратегии BREAKOUT_TREND_HYBRID с сравнением результатов.

Тестирует гибридную стратегию на исторических данных и сравнивает результаты
с оригинальными стратегиями VBO и TREND для оценки эффективности объединения.
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
from bot.breakout_trend_hybrid import build_breakout_trend_signals
from bot.vbo_strategy import build_vbo_signals
from bot.strategy import generate_trend_signal
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


def run_breakout_trend_backtest(
    symbols: List[str] = ["SOLUSDT"],
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
    print(f"BREAKOUT_TREND_HYBRID BACKTEST")
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
                hybrid_signals = build_breakout_trend_signals(df_ready, settings.strategy, symbol)
                print(f"   Generated {len(hybrid_signals)} signals")
                
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
            
            # 2. Тестируем VBO стратегию
            print(f"\n📈 Testing VBO strategy...")
            try:
                vbo_signals = build_vbo_signals(df_ready, settings.strategy, symbol)
                print(f"   Generated {len(vbo_signals)} signals")
                
                simulator = ZScoreBacktestSimulator(
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                )
                
                result = simulator.run(df_ready, vbo_signals, symbol)
                vbo_trades = result["trades"]
                vbo_metrics = calculate_base_metrics(
                    vbo_trades, initial_balance, vbo_signals, symbol
                )
                
                comparisons.append(StrategyComparison(
                    strategy_name="VBO",
                    metrics=vbo_metrics,
                    trades=vbo_trades,
                    signals_count=len(vbo_signals),
                ))
                
                print(f"   ✅ VBO: {vbo_metrics.total_trades} trades, "
                      f"WR: {vbo_metrics.win_rate:.1f}%, "
                      f"PnL: ${vbo_metrics.total_pnl:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error testing VBO: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. Тестируем TREND стратегию
            print(f"\n📊 Testing TREND strategy...")
            try:
                trend_state = {'backtest_mode': True}
                trend_signals_list = []
                
                # Генерируем сигналы для каждой свечи
                for i in range(200, len(df_ready)):
                    trend_result = generate_trend_signal(
                        df_ready.iloc[:i+1],
                        state=trend_state,
                        sma_period=getattr(settings.strategy, 'trend_sma_period', 21),
                        atr_period=getattr(settings.strategy, 'trend_atr_period', 14),
                        atr_multiplier=getattr(settings.strategy, 'trend_atr_multiplier', 3.0),
                        max_pyramid=getattr(settings.strategy, 'trend_max_pyramid', 2),
                        min_history=100,
                    )
                    
                    if trend_result and trend_result.get('signal') in ('LONG', 'SHORT'):
                        action = Action.LONG if trend_result['signal'] == 'LONG' else Action.SHORT
                        price = float(df_ready.iloc[i]['close'])
                        timestamp = df_ready.index[i]
                        
                        signal = Signal(
                            timestamp=timestamp,
                            action=action,
                            reason=trend_result.get('reason', 'trend'),
                            price=price,
                            stop_loss=trend_result.get('stop_loss'),
                            take_profit=trend_result.get('take_profit'),
                        )
                        trend_signals_list.append(signal)
                
                print(f"   Generated {len(trend_signals_list)} signals")
                
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
                
            except Exception as e:
                print(f"   ❌ Error testing TREND: {e}")
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
                    vbo_comp = next((c for c in comparisons if c.strategy_name == "VBO"), None)
                    trend_comp = next((c for c in comparisons if c.strategy_name == "TREND"), None)
                    
                    print(f"\n📊 HYBRID vs Original Strategies:")
                    if vbo_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - vbo_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - vbo_comp.metrics.win_rate
                        trades_diff = hybrid_comp.metrics.total_trades - vbo_comp.metrics.total_trades
                        print(f"   vs VBO: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%, "
                              f"Trades diff: {trades_diff:+d}")
                    
                    if trend_comp:
                        pnl_diff = hybrid_comp.metrics.total_pnl - trend_comp.metrics.total_pnl
                        wr_diff = hybrid_comp.metrics.win_rate - trend_comp.metrics.win_rate
                        trades_diff = hybrid_comp.metrics.total_trades - trend_comp.metrics.total_trades
                        print(f"   vs TREND: PnL diff: ${pnl_diff:+.2f}, WR diff: {wr_diff:+.1f}%, "
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
        comparison_file = os.path.join(output_dir, f"breakout_trend_comparison_{timestamp}.csv")
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_file, index=False)
        print(f"✅ Comparison report saved: {comparison_file}")
    
    # Текстовый отчет с рекомендациями
    report_file = os.path.join(output_dir, f"breakout_trend_backtest_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BREAKOUT_TREND_HYBRID BACKTEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        for symbol, comparisons in all_comparisons.items():
            f.write(f"\n{symbol}\n")
            f.write("-"*80 + "\n")
            
            hybrid_comp = next((c for c in comparisons if c.strategy_name == "HYBRID"), None)
            vbo_comp = next((c for c in comparisons if c.strategy_name == "VBO"), None)
            trend_comp = next((c for c in comparisons if c.strategy_name == "TREND"), None)
            
            if hybrid_comp and vbo_comp and trend_comp:
                f.write(f"\nHYBRID Results:\n")
                f.write(f"  Trades: {hybrid_comp.metrics.total_trades}\n")
                f.write(f"  Win Rate: {hybrid_comp.metrics.win_rate:.1f}%\n")
                f.write(f"  PnL: ${hybrid_comp.metrics.total_pnl:.2f}\n")
                f.write(f"  Profit Factor: {hybrid_comp.metrics.profit_factor:.2f}\n")
                f.write(f"  Signals: {hybrid_comp.signals_count}\n")
                
                f.write(f"\nComparison:\n")
                pnl_vs_vbo = hybrid_comp.metrics.total_pnl - vbo_comp.metrics.total_pnl
                pnl_vs_trend = hybrid_comp.metrics.total_pnl - trend_comp.metrics.total_pnl
                wr_vs_vbo = hybrid_comp.metrics.win_rate - vbo_comp.metrics.win_rate
                wr_vs_trend = hybrid_comp.metrics.win_rate - trend_comp.metrics.win_rate
                
                f.write(f"  vs VBO: PnL {pnl_vs_vbo:+.2f}, WR {wr_vs_vbo:+.1f}%\n")
                f.write(f"  vs TREND: PnL {pnl_vs_trend:+.2f}, WR {wr_vs_trend:+.1f}%\n")
                
                # Рекомендации
                f.write(f"\nRecommendations:\n")
                if pnl_vs_vbo > 0 and pnl_vs_trend > 0:
                    f.write(f"  ✅ HYBRID outperforms both strategies!\n")
                    f.write(f"  → Consider implementing in live trading\n")
                elif pnl_vs_vbo > 0 or pnl_vs_trend > 0:
                    f.write(f"  ⚠️ HYBRID outperforms one strategy\n")
                    f.write(f"  → Consider optimizing parameters\n")
                else:
                    f.write(f"  ❌ HYBRID underperforms both strategies\n")
                    f.write(f"  → Review logic and parameters\n")
                
                # Анализ сигналов
                if hybrid_comp.signals_count < vbo_comp.signals_count:
                    f.write(f"\n  Note: HYBRID generates fewer signals ({hybrid_comp.signals_count} vs {vbo_comp.signals_count})\n")
                    f.write(f"  → This is expected due to TREND filtering\n")
                    f.write(f"  → Quality over quantity approach\n")
    
    print(f"✅ Report saved: {report_file}")
    
    return {
        "comparisons": all_comparisons,
        "timestamp": timestamp,
    }


def main():
    """Главная функция для запуска бэктеста."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BREAKOUT_TREND_HYBRID Backtest")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["SOLUSDT"],
        help="Symbols to test (default: SOLUSDT)",
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
    
    run_breakout_trend_backtest(
        symbols=args.symbols,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        days_back=args.days,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
