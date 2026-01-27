import pandas as pd
import numpy as np
import os
import sys
import pytz
from datetime import time

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings
from bot.ict_strategy import ICTStrategy, build_ict_signals

def analyze_ict(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df = df.set_index('datetime')
    
    settings = load_settings()
    params = settings.strategy
    strategy = ICTStrategy(params)
    
    output = []
    output.append(f"Analyzing ICT data from {csv_path}")
    output.append(f"Total Rows: {len(df)}")
    
    # 1. Проверка сессий
    def check_session(ts):
        if ts.tzinfo is None:
            ts_utc = ts.tz_localize('UTC')
        else:
            ts_utc = ts.tz_convert('UTC')
        ts_ny = ts_utc.astimezone(strategy.ny_tz)
        current_time_ny = ts_ny.time()
        for start, end in strategy.sb_windows_ny:
            if start <= current_time_ny <= end:
                return True
        return False

    df['is_session'] = [check_session(ts) for ts in df.index]
    output.append(f"Rows in ICT Sessions: {df['is_session'].sum()}")
    
    # 2. Проверка Аллигатора
    jaw, teeth, lips = strategy.calculate_williams_alligator(df)
    expanded_count = 0
    for i in range(200, len(df)):
        exp, _ = strategy.is_alligator_expanded(jaw, teeth, lips, i)
        if exp: expanded_count += 1
    output.append(f"Rows with Alligator Expanded: {expanded_count}")
    
    # 3. Ликвидность
    sweeps = strategy.find_liquidity_sweeps(df, lookback_days=5)
    output.append(f"Liquidity Sweeps found: {len(sweeps)}")
    
    # 4. FVG
    fvg = strategy.find_fvg(df, sweeps)
    output.append(f"FVG found (after liquidity): {len(fvg)}")
    
    # 5. Итоговые сигналы
    signals = build_ict_signals(df, params, symbol="BTCUSDT")
    output.append(f"Final Signals: {len(signals)}")
    
    with open("analyze_ict_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print("Results written to analyze_ict_results.txt")

if __name__ == "__main__":
    csv_file = "data/btcusdt_15m.csv"
    if not os.path.exists(csv_file): csv_file = "data/btc_15m.csv"
    if os.path.exists(csv_file):
        analyze_ict(csv_file)
    else:
        print("CSV not found.")
