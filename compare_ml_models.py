"""
Скрипт для массового тестирования ВСЕХ ML моделей по каждому символу.

Запускает бэктест (через backtest_ml_strategy.run_ml_backtest) для всех моделей в
директории ml_models и формирует сводную таблицу с результатами.

Использование:
    python compare_ml_models.py

Опции:
    --days 30           # Сколько дней тестировать (по умолчанию 30)
    --symbols BTCUSDT,ETHUSDT,SOLUSDT  # Ограничить список символов
    --models-dir ml_models             # Путь к директории с моделями
    --output csv                        # Дополнительно сохранить таблицу в CSV
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd

from backtest_ml_strategy import run_ml_backtest, BacktestMetrics


def find_models_for_symbol(models_dir: Path, symbol: str) -> List[Path]:
    """
    Ищет все ML модели для указанного символа.
    
    Ожидаемый формат имени файла:
        {model_type}_{SYMBOL}_{INTERVAL}.pkl
        {model_type}_{SYMBOL}_{INTERVAL}_{mode_suffix}.pkl  # mtf / 15m
    
    Примеры:
        ensemble_BTCUSDT_15.pkl
        ensemble_BTCUSDT_15_mtf.pkl
        quad_ensemble_ETHUSDT_15_15m.pkl
    """
    if not models_dir.exists():
        return []
    
    patterns = [
        f"*_{symbol}_*.pkl",
    ]
    
    results: List[Path] = []
    for pattern in patterns:
        for f in models_dir.glob(pattern):
            if f.is_file():
                results.append(f)
    # Убираем дубликаты и сортируем по имени
    results = sorted(list({f.resolve() for f in results}))
    return results


def metrics_to_dict(m: BacktestMetrics) -> Dict[str, Any]:
    """Преобразует BacktestMetrics в словарь для удобного сохранения/анализа."""
    return {
        "symbol": m.symbol,
        "model_name": m.model_name,
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "win_rate_pct": m.win_rate,
        "total_pnl_usd": m.total_pnl,
        "total_pnl_pct": m.total_pnl_pct,
        "profit_factor": m.profit_factor,
        "max_drawdown_usd": m.max_drawdown,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "long_trades": m.long_signals,
        "short_trades": m.short_signals,
        "avg_trade_duration_hours": m.avg_trade_duration_hours,
        "avg_win_usd": m.avg_win,
        "avg_loss_usd": m.avg_loss,
        "best_trade_usd": m.best_trade_pnl,
        "worst_trade_usd": m.worst_trade_pnl,
        "largest_win_usd": m.largest_win,
        "largest_loss_usd": m.largest_loss,
        "consecutive_wins": m.consecutive_wins,
        "consecutive_losses": m.consecutive_losses,
        "avg_confidence": m.avg_confidence,
    }


def compare_models(
    symbols: List[str],
    models_dir: Path,
    days: int = 30,
    interval: str = "15m",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
) -> pd.DataFrame:
    """
    Запускает бэктест для всех моделей и возвращает DataFrame с результатами.
    """
    all_results: List[Dict[str, Any]] = []
    
    print("=" * 80)
    print("🚀 ML MODELS COMPARISON BACKTEST")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Models dir: {models_dir}")
    print(f"Days: {days}, Interval: {interval}, Initial balance: {initial_balance}, Risk per trade: {risk_per_trade*100:.1f}%, Leverage: {leverage}x")
    print("=" * 80)
    
    for symbol in symbols:
        print(f"\n\n🔍 SYMBOL: {symbol}")
        print("-" * 80)
        
        models = find_models_for_symbol(models_dir, symbol)
        if not models:
            print(f"❌ No models found for {symbol} in {models_dir}")
            continue
        
        print(f"📦 Found {len(models)} models for {symbol}:")
        for mpath in models:
            print(f"   - {mpath.name}")
        
        for model_path in models:
            try:
                metrics = run_ml_backtest(
                    model_path=str(model_path),
                    symbol=symbol,
                    days_back=days,
                    interval=interval,
                    initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade,
                    leverage=leverage,
                )
                if metrics is None:
                    print(f"⚠️  Backtest failed for model {model_path.name}, skipping.")
                    continue
                
                row = metrics_to_dict(metrics)
                # Добавляем техническую информацию о типе модели и MTF-суффиксе
                filename = Path(model_path).name
                name_no_ext = filename.replace(".pkl", "")
                parts = name_no_ext.split("_")
                model_type = parts[0] if parts else "unknown"
                mode_suffix = None
                if len(parts) >= 4:
                    mode_suffix = parts[-1]  # mtf / 15m / др.
                row["model_type"] = model_type
                row["mode_suffix"] = mode_suffix or ""
                
                all_results.append(row)
            except Exception as e:
                print(f"❌ Exception while backtesting {model_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not all_results:
        print("❌ No results collected.")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    
    # Сортировка: по символу, затем по total_pnl_pct (убывание)
    df_results.sort_values(
        by=["symbol", "total_pnl_pct", "win_rate_pct"],
        ascending=[True, False, False],
        inplace=True,
    )
    
    return df_results


def print_summary_table(df_results: pd.DataFrame) -> None:
    """Печатает компактную сводную таблицу по каждому символу."""
    if df_results.empty:
        print("❌ No results to display.")
        return
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY: BEST MODELS PER SYMBOL")
    print("=" * 80)
    
    for symbol, group in df_results.groupby("symbol"):
        print(f"\n📈 {symbol}:")
        # Берём top-5 по PnL%
        top = group.head(5).copy()
        cols = [
            "model_name",
            "model_type",
            "mode_suffix",
            "total_trades",
            "win_rate_pct",
            "total_pnl_usd",
            "total_pnl_pct",
            "profit_factor",
            "max_drawdown_pct",
        ]
        print(top[cols].to_string(index=False, formatters={
            "win_rate_pct": "{:.2f}".format,
            "total_pnl_usd": "{:.2f}".format,
            "total_pnl_pct": "{:+.2f}".format,
            "profit_factor": "{:.2f}".format,
            "max_drawdown_pct": "{:.2f}".format,
        }))


def main():
    parser = argparse.ArgumentParser(description="Compare all ML models via backtesting")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest (default: 30)")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="Comma-separated list of symbols (default: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="ml_models",
        help="Directory with ML models (default: ml_models)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Timeframe interval (default: 15m)",
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
        help="Risk per trade fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=10,
        help="Leverage (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["none", "csv"],
        default="csv",
        help="Save results to CSV (default: csv)",
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    models_dir = Path(args.models_dir)
    
    df_results = compare_models(
        symbols=symbols,
        models_dir=models_dir,
        days=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
    )
    
    if df_results.empty:
        return
    
    # Печатаем сводку
    print_summary_table(df_results)
    
    # Сохраняем CSV при необходимости
    if args.output == "csv":
        output_name = f"ml_models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(output_name, index=False)
        print(f"\n💾 Full comparison table saved to: {output_name}")


if __name__ == "__main__":
    main()

