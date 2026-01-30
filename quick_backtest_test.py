#!/usr/bin/env python3
"""
Quick test to verify the backtest functionality with our fixes.
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_backtest():
    """Test running a quick backtest."""
    try:
        # Import the backtest function
        from backtest_ml_strategy import run_ml_backtest
        
        # Try to run a very short backtest with a small model
        model_files = list(Path("ml_models").glob("*.pkl"))
        
        if not model_files:
            print("❌ No model files found in ml_models directory")
            return False
            
        # Use the first model file
        model_path = model_files[0]
        print(f"Testing backtest with model: {model_path.name}")
        
        # Run a very short backtest (1 day, small data)
        metrics = run_ml_backtest(
            model_path=str(model_path),
            symbol="BTCUSDT",
            days_back=1,  # Very short test
            interval="15m",
            initial_balance=1000.0,
            risk_per_trade=0.01,
            leverage=5,
            output_plots=False,  # Disable plots for quick test
            validate_data=False  # Disable data validation for quick test
        )
        
        if metrics:
            print("✅ Backtest completed successfully!")
            print(f"   Total PnL: ${metrics.total_pnl:.2f}")
            print(f"   Win Rate: {metrics.win_rate:.2f}%")
            print(f"   Total Trades: {metrics.total_trades}")
            return True
        else:
            print("❌ Backtest failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during backtest test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Backtest Functionality...")
    print("=" * 50)
    
    success = test_backtest()
    
    print("=" * 50)
    if success:
        print("✅ Backtest test passed!")
        sys.exit(0)
    else:
        print("❌ Backtest test failed!")
        sys.exit(1)