"""
Быстрый тест одной модели для отладки.
"""
import sys
from pathlib import Path
from backtest_ml_strategy import run_ml_backtest

if __name__ == "__main__":
    # Тестируем одну модель для быстрой проверки
    symbol = "BTCUSDT"
    model_path = None
    
    # Ищем первую доступную модель для BTCUSDT
    models_dir = Path("ml_models")
    if models_dir.exists():
        models = list(models_dir.glob(f"*_{symbol}_*.pkl"))
        if models:
            model_path = str(models[0])
            print(f"🔍 Testing model: {model_path}")
        else:
            print(f"❌ No models found for {symbol}")
            sys.exit(1)
    else:
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)
    
    print(f"\n📊 Running backtest for {symbol}...")
    print(f"   Model: {Path(model_path).name}")
    print(f"   Days: 30")
    print(f"   Interval: 15m")
    print("=" * 80)
    
    metrics = run_ml_backtest(
        model_path=model_path,
        symbol=symbol,
        days_back=30,
        interval="15",
        initial_balance=1000.0,
        risk_per_trade=0.02,
        leverage=10,
    )
    
    if metrics:
        print("\n" + "=" * 80)
        print("📈 RESULTS:")
        print("=" * 80)
        print(f"Model: {metrics.model_name}")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Winning Trades: {metrics.winning_trades}")
        print(f"Losing Trades: {metrics.losing_trades}")
        print(f"Win Rate: {metrics.win_rate:.2f}%")
        print(f"Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Long Trades: {metrics.long_signals}")
        print(f"Short Trades: {metrics.short_signals}")
        print(f"Avg Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
        print(f"Avg Win: ${metrics.avg_win:.2f}")
        print(f"Avg Loss: ${metrics.avg_loss:.2f}")
        print(f"Avg Confidence: {metrics.avg_confidence:.4f}")
    else:
        print("❌ Backtest failed!")
