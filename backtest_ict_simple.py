import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings
from bot.ict_strategy import ICTStrategy, build_ict_signals
from bot.strategy import Action

def run_ict_backtest(csv_path, initial_balance=100, pos_size_pct=0.2, tp_pct=0.02, sl_pct=0.01):
    print("--- ICT BACKTEST VERSION 2.0 ---")
    output_log = []
    def log(msg):
        print(msg)
        output_log.append(str(msg))

    log(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Подготовка данных
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df = df.set_index('datetime')
    
    # Загружаем настройки
    settings = load_settings()
    params = settings.strategy
    
    # ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ
    params.ict_fvg_max_age_bars = 100 
    params.ict_rr_ratio = 2.0 
    params.ict_mss_close_required = False 
    params.ict_fvg_search_window = 60 
    params.ict_mtf_bias_timeframe = None # Отключаем MTF фильтр для теста, чтобы увидеть больше сигналов
    
    commission = 0.0006 
    
    log("\n--- Анализ структуры ICT ---")
    strategy = ICTStrategy(params)
    
    # 1. Сессии
    def check_session(ts):
        if ts.tzinfo is None: ts_utc = ts.tz_localize('UTC')
        else: ts_utc = ts.tz_convert('UTC')
        ts_ny = ts_utc.astimezone(strategy.ny_tz)
        for start, end in strategy.sb_windows_ny:
            if start <= ts_ny.time() <= end: return True
        return False
    session_mask = [check_session(ts) for ts in df.index]
    log(f"Candles in Silver Bullet windows: {sum(session_mask)} / {len(df)}")

    # 2. Ликвидность
    sweeps = strategy.find_liquidity_sweeps(df, lookback_days=5)
    if not sweeps:
        log("No session sweeps found, using alternative method...")
        sweeps = strategy.find_liquidity_sweeps_alternative(df, lookback_bars=100)
    log(f"Liquidity Sweeps: {len(sweeps)}")
    
    # 3. FVG
    fvg = strategy.find_fvg(df, sweeps)
    log(f"FVG after sweeps: {len(fvg)}")
    if fvg:
        log("First 5 FVGs:")
        for f in fvg[:5]:
            log(f"  {f.timestamp}: {f.direction} at index {f.bar_index}")
    
    # Генерация сигналов
    log("\nGenerating signals...")
    signals = build_ict_signals(df, params, symbol="BTCUSDT")
    actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
    log(f"Actionable signals: {len(actionable_signals)}")
    
    # DEBUG: Посмотрим на первые несколько временных меток в сессии
    session_indices = [i for i, val in enumerate(session_mask) if val]
    if session_indices:
        log(f"First session candle: {df.index[session_indices[0]]}")
    
    with open("debug_ict_signals.txt", "w") as f:
        f.write(f"Total signals: {len(signals)}\n")
        f.write(f"Actionable signals: {len(actionable_signals)}\n")
        for s in actionable_signals[:10]:
            f.write(f"{s.timestamp}: {s.action} {s.price}\n")
    
    if actionable_signals:
        log(f"\nFirst 5 signals:")
        for s in actionable_signals[:5]:
            log(f"  {s.timestamp}: {s.action.value} @ {s.price:.2f} ({s.reason})")
    else:
        # Пытаемся понять почему 0
        log("\nDEBUG: Why 0 signals?")
        # Проверим Аллигатор
        jaw, teeth, lips = strategy.calculate_williams_alligator(df)
        alligator_count = 0
        for i in range(200, len(df)):
            exp, _ = strategy.is_alligator_expanded(jaw, teeth, lips, i)
            if exp: alligator_count += 1
        log(f"Candles with Alligator expanded: {alligator_count}")
        
        # Попробуем БЕЗ Аллигатора для теста
        log("Retrying WITHOUT Alligator filter...")
        strategy.is_alligator_expanded = lambda j, t, l, idx: (True, "bullish") 
        signals = strategy.get_signals(df, symbol="BTCUSDT_NO_ALLIGATOR")
        actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
        log(f"Signals without Alligator: {len(actionable_signals)}")
    
    balance = initial_balance
    position = None 
    entry_price = 0
    trades = []
    signals_map = {s.timestamp: s for s in actionable_signals}
    
    log(f"\nStarting backtest loop (Balance=${balance})...")
    
    for i in range(len(df)):
        curr_time = df.index[i]
        row = df.iloc[i]
        curr_price = row['close']
        
        if position == 'LONG':
            if row['low'] <= entry_price * (1 - sl_pct):
                exit_price = entry_price * (1 - sl_pct)
                pnl_factor = (exit_price - entry_price) / entry_price - (2 * commission)
                trade_pnl = (balance * pos_size_pct) * pnl_factor
                balance += trade_pnl
                trades.append({'type': 'LONG', 'pnl': trade_pnl, 'win': 0, 'time': curr_time, 'reason': 'SL'})
                position = None
            elif row['high'] >= entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
                pnl_factor = (exit_price - entry_price) / entry_price - (2 * commission)
                trade_pnl = (balance * pos_size_pct) * pnl_factor
                balance += trade_pnl
                trades.append({'type': 'LONG', 'pnl': trade_pnl, 'win': 1, 'time': curr_time, 'reason': 'TP'})
                position = None
        elif position == 'SHORT':
            if row['high'] >= entry_price * (1 + sl_pct):
                exit_price = entry_price * (1 + sl_pct)
                pnl_factor = (entry_price - exit_price) / entry_price - (2 * commission)
                trade_pnl = (balance * pos_size_pct) * pnl_factor
                balance += trade_pnl
                trades.append({'type': 'SHORT', 'pnl': trade_pnl, 'win': 0, 'time': curr_time, 'reason': 'SL'})
                position = None
            elif row['low'] <= entry_price * (1 - tp_pct):
                exit_price = entry_price * (1 - tp_pct)
                pnl_factor = (entry_price - exit_price) / entry_price - (2 * commission)
                trade_pnl = (balance * pos_size_pct) * pnl_factor
                balance += trade_pnl
                trades.append({'type': 'SHORT', 'pnl': trade_pnl, 'win': 1, 'time': curr_time, 'reason': 'TP'})
                position = None

        if position is None and curr_time in signals_map:
            sig = signals_map[curr_time]
            position = 'LONG' if sig.action == Action.LONG else 'SHORT'
            entry_price = curr_price
                    
    final_results = []
    final_results.append("\n" + "="*30)
    final_results.append("ICT SILVER BULLET BACKTEST RESULTS")
    final_results.append("="*30)
    
    if not trades:
        final_results.append("No trades executed.")
    else:
        tr_df = pd.DataFrame(trades)
        total_pnl = balance - initial_balance
        final_results.append(f"Total PnL: {total_pnl:.2f}$ ({(total_pnl/initial_balance)*100:.2f}%)")
        final_results.append(f"Final Balance: {balance:.2f}$")
        final_results.append(f"Win Rate: {tr_df['win'].mean()*100:.2f}%")
        final_results.append(f"Total Trades: {len(trades)}")
        final_results.append(f"Longs: {len(tr_df[tr_df['type']=='LONG'])}, Shorts: {len(tr_df[tr_df['type']=='SHORT'])}")
        final_results.append(f"TP hits: {len(tr_df[tr_df['reason']=='TP'])}, SL hits: {len(tr_df[tr_df['reason']=='SL'])}")
    
    log("\n".join(final_results))
    with open("backtest_ict_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_log))
    return trades

if __name__ == "__main__":
    csv_file = "data/btcusdt_15m.csv"
    if not os.path.exists(csv_file): csv_file = "data/btc_15m.csv"
    if os.path.exists(csv_file):
        run_ict_backtest(csv_file)
    else:
        print("CSV not found.")
