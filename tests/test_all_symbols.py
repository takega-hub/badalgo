"""
Скрипт для тестирования всех стратегий на всех символах.
"""
import subprocess
import sys
import os
from pathlib import Path

# Список стратегий для тестирования
STRATEGIES = ["trend", "flat", "momentum", "liquidity", "smc", "ict", "ml"]

# Список символов для тестирования
SYMBOLS = ["ETHUSDT", "SOLUSDT"]

# Количество дней для тестирования
DAYS = 30

def test_strategy_on_symbol(strategy: str, symbol: str, days: int = 30):
    """Тестирует стратегию на символе."""
    print("\n" + "=" * 80)
    print(f"🧪 Тестирование {strategy.upper()} стратегии на {symbol}")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    script_path = script_dir / "test_all_strategies.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--strategy", strategy,
        "--symbol", symbol,
        "--days", str(days)
    ]
    
    try:
        # Используем os.system для более надежного запуска
        cmd_str = " ".join(f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd)
        result = os.system(cmd_str)
        return result == 0
    except Exception as e:
        print(f"❌ Ошибка при тестировании {strategy} на {symbol}: {e}", file=sys.stderr)
        return False

def main():
    """Главная функция."""
    print("=" * 80)
    print("🚀 ТЕСТИРОВАНИЕ ВСЕХ СТРАТЕГИЙ НА ETHUSDT И SOLUSDT")
    print("=" * 80)
    print(f"Стратегии: {', '.join(STRATEGIES)}")
    print(f"Символы: {', '.join(SYMBOLS)}")
    print(f"Период: {DAYS} дней")
    print("=" * 80)
    
    results = {}
    
    for symbol in SYMBOLS:
        results[symbol] = {}
        for strategy in STRATEGIES:
            success = test_strategy_on_symbol(strategy, symbol, DAYS)
            results[symbol][strategy] = success
    
    # Итоговая статистика
    print("\n" + "=" * 80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    
    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        for strategy in STRATEGIES:
            status = "✅" if results[symbol][strategy] else "❌"
            print(f"  {status} {strategy}")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)

if __name__ == "__main__":
    main()
