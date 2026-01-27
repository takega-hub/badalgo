import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone
import pytz

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings
from bot.ict_strategy import ICTStrategy, build_ict_signals
from bot.strategy import Action

def run_multi_symbol_backtest(symbols=["ETHUSDT", "SOLUSDT"], initial_balance=100):
    print(f"\n{'='*50}")
    print(f"   ICT SILVER BULLET MULTI-SYMBOL BACKTEST (V10)")
    print(f"{'='*50}\n")
    
    settings = load_settings()
    params = settings.strategy
    
    # Применяем параметры V9 Risk Master
    params.ict_mtf_bias_timeframe = "1h"
    params.ict_rr_ratio = 2.0 
    params.ict_breakeven_rr = 1.0 # БУ при 1.0R
    params.ict_mss_close_required = True
    
    commission = 0.0006
    pos_size_pct = 0.2
    
    all_trades = []
    total_balance = initial_balance
    
    # Ищем файлы для каждого символа
    symbol_files = {
        "BTCUSDT": ["data/btcusdt_15m.csv", "data/btc_15m.csv"],
        "ETHUSDT": ["data/ethusdt_15m.csv", "data/eth_15m.csv"],
        "SOLUSDT": ["data/solusdt_15m.csv", "data/sol_15m.csv"]
    }
    
    for symbol in symbols:
        csv_path = None
        # Пытаемся найти файл данных (более гибкий поиск)
        potential_paths = symbol_files.get(symbol, [])
        # Также добавим общие паттерны
        potential_paths.extend([
            f"data/{symbol.lower()}_15m.csv",
            f"data/{symbol[:3].lower()}_15m.csv",
            f"data/{symbol}_15m.csv"
        ])
        
        for path in potential_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            print(f"⚠️  Data for {symbol} not found. Skipping...")
            continue
            
        print(f"\n>>> Backtesting {symbol} using {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Автоматическое определение колонок времени
        time_col = None
        for col in ['datetime', 'timestamp', 'Time', 'time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            if df[time_col].dtype == object:
                df['datetime'] = pd.to_datetime(df[time_col])
            else:
                # Если это числа (мс), указываем unit='ms'
                u = 'ms' if df[time_col].iloc[0] > 1e12 else 's'
                df['datetime'] = pd.to_datetime(df[time_col], unit=u)
            df = df.set_index('datetime')
        else:
            print(f"❌ Could not find time column for {symbol}. Skipping...")
            continue
            
        # Ограничим данные последними 3 месяцами для скорости, если их много
        if len(df) > 10000:
            df = df.iloc[-10000:]
            
        strategy = ICTStrategy(params)
        signals = strategy.get_signals(df, symbol=symbol)
        actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
        print(f"    Found {len(actionable_signals)} actionable signals")
        
        signals_map = {s.timestamp: s for s in actionable_signals}
        
        # Симуляция торговли
        position = None
        opening_session_id = None # V10.2: Запоминаем сессию открытия
        entry_price = 0
        sl_price = 0
        tp_price = 0
        breakeven_active = False
        
        # Получаем данные о свипах и FVG для этого символа для точных стопов
        strategy = ICTStrategy(params)
        liquidity_sweeps = strategy.find_liquidity_sweeps(df)
        if not liquidity_sweeps:
            liquidity_sweeps = strategy.find_liquidity_sweeps_alternative(df, 50)
        liq_map = {liq.bar_index: liq for liq in liquidity_sweeps}
        
        for i in range(len(df)):
            curr_time = df.index[i]
            row = df.iloc[i]
            curr_price = row['close']
            
            if position:
                # 1. Проверка стоп-лосса и тейк-профита
                trade_closed = False
                if position == 'LONG':
                    if row['low'] <= sl_price:
                        pnl_pct = (sl_price - entry_price) / entry_price - (2 * commission)
                        trade_pnl = (total_balance * pos_size_pct) * pnl_pct
                        total_balance += trade_pnl
                        all_trades.append({'symbol': symbol, 'type': 'LONG', 'pnl': trade_pnl, 'win': 0, 'reason': 'SL', 'time': curr_time, 'entry': entry_price, 'exit': sl_price})
                        print(f"      [LOSS] LONG {symbol} at {curr_time} | Entry: {entry_price:.2f}, SL: {sl_price:.2f}, PnL: {trade_pnl:.2f}$")
                        
                        # V10.2: Записываем убыток в сессию открытия
                        if opening_session_id:
                            strategy.session_results[opening_session_id] = False
                            print(f"      [DEBUG] Recording LOSS for session {opening_session_id}")
                        
                        position = None
                        opening_session_id = None
                        trade_closed = True
                    elif row['high'] >= tp_price:
                        pnl_pct = (tp_price - entry_price) / entry_price - (2 * commission)
                        trade_pnl = (total_balance * pos_size_pct) * pnl_pct
                        total_balance += trade_pnl
                        all_trades.append({'symbol': symbol, 'type': 'LONG', 'pnl': trade_pnl, 'win': 1, 'reason': 'TP', 'time': curr_time, 'entry': entry_price, 'exit': tp_price})
                        print(f"      [WIN]  LONG {symbol} at {curr_time} | Entry: {entry_price:.2f}, TP: {tp_price:.2f}, PnL: {trade_pnl:.2f}$")
                        
                        # V10.2: Записываем прибыль
                        if opening_session_id:
                            strategy.session_results[opening_session_id] = True
                            print(f"      [DEBUG] Recording WIN for session {opening_session_id}")
                        
                        position = None
                        opening_session_id = None
                        trade_closed = True
                    # Безубыток: переносим в +0.15% при 1.5R (больше воздуха)
                    elif not breakeven_active and (curr_price - entry_price) >= (abs(entry_price - sl_price) * 1.0):
                        sl_price = entry_price * 1.0015 
                        breakeven_active = True
                        print(f"      [INFO] Breakeven (+0.15%) activated for LONG {symbol}")
                        
                elif position == 'SHORT':
                    if row['high'] >= sl_price:
                        pnl_pct = (entry_price - sl_price) / entry_price - (2 * commission)
                        trade_pnl = (total_balance * pos_size_pct) * pnl_pct
                        total_balance += trade_pnl
                        all_trades.append({'symbol': symbol, 'type': 'SHORT', 'pnl': trade_pnl, 'win': 0, 'reason': 'SL', 'time': curr_time, 'entry': entry_price, 'exit': sl_price})
                        print(f"      [LOSS] SHORT {symbol} at {curr_time} | Entry: {entry_price:.2f}, SL: {sl_price:.2f}, PnL: {trade_pnl:.2f}$")
                        
                        # V10.2: Записываем убыток
                        if opening_session_id:
                            strategy.session_results[opening_session_id] = False
                            print(f"      [DEBUG] Recording LOSS for session {opening_session_id}")
                        
                        position = None
                        opening_session_id = None
                        trade_closed = True
                    elif row['low'] <= tp_price:
                        pnl_pct = (entry_price - tp_price) / entry_price - (2 * commission)
                        trade_pnl = (total_balance * pos_size_pct) * pnl_pct
                        total_balance += trade_pnl
                        all_trades.append({'symbol': symbol, 'type': 'SHORT', 'pnl': trade_pnl, 'win': 1, 'reason': 'TP', 'time': curr_time, 'entry': entry_price, 'exit': tp_price})
                        print(f"      [WIN]  SHORT {symbol} at {curr_time} | Entry: {entry_price:.2f}, TP: {tp_price:.2f}, PnL: {trade_pnl:.2f}$")
                        
                        # V10.2: Записываем прибыль
                        if opening_session_id:
                            strategy.session_results[opening_session_id] = True
                            print(f"      [DEBUG] Recording WIN for session {opening_session_id}")
                        
                        position = None
                        opening_session_id = None
                        trade_closed = True
                    # Безубыток для SHORT при 1.5R
                    elif not breakeven_active and (entry_price - curr_price) >= (abs(entry_price - sl_price) * 1.0):
                        sl_price = entry_price * 0.9985
                        breakeven_active = True
                        print(f"      [INFO] Breakeven (+0.15%) activated for SHORT {symbol}")

            # 2. Вход в позицию
            if position is None and curr_time in signals_map:
                # V10: Проверка на тильт (cooldown)
                s_id = strategy.get_session_id(curr_time)
                prev_s_id = strategy.get_previous_session_id(s_id)
                
                # DEBUG: Лог проверки cooldown
                print(f"      [DEBUG] Signal at {curr_time}, Session: {s_id}, Prev: {prev_s_id}, Prev Result: {strategy.session_results.get(prev_s_id)}")
                
                if strategy.session_results.get(prev_s_id) == False:
                    # Пропускаем вход, так как прошлую сессию закрыли в минус
                    print(f"      [COOLDOWN] Skipping signal at {curr_time} due to previous loss in {prev_s_id}")
                    continue

                sig = signals_map[curr_time]
                position = 'LONG' if sig.action == Action.LONG else 'SHORT'
                opening_session_id = s_id # V10.2: Запоминаем ID сессии открытия
                entry_price = curr_price
                
                # Используем SL/TP напрямую из сигнала стратегии (V8 Professional)
                sl_price = sig.stop_loss
                tp_price = sig.take_profit
                
                if sl_price and tp_price:
                    breakeven_active = False
                else:
                    # Fallback если стратегия не вернула SL/TP
                    risk = entry_price * 0.01
                    sl_price = entry_price - risk if position == 'LONG' else entry_price + risk
                    tp_price = entry_price + (risk * params.ict_rr_ratio) if position == 'LONG' else entry_price - (risk * params.ict_rr_ratio)
                    breakeven_active = False

    # Итоговый отчет
    print(f"\n{'='*50}")
    print(f"        FINAL MULTI-SYMBOL REPORT")
    print(f"{'='*50}")
    
    if not all_trades:
        print("No trades executed across all symbols.")
    else:
        tr_df = pd.DataFrame(all_trades)
        total_pnl = total_balance - initial_balance
        print(f"Total PnL: {total_pnl:.2f}$ ({(total_pnl/initial_balance)*100:.2f}%)")
        print(f"Final Combined Balance: {total_balance:.2f}$")
        print(f"Overall Win Rate: {tr_df['win'].mean()*100:.2f}%")
        print(f"Total Trades: {len(all_trades)}")
        
        for symbol in symbols:
            s_trades = tr_df[tr_df['symbol'] == symbol]
            if not s_trades.empty:
                print(f"\n--- {symbol} Stats ---")
                print(f"  Trades: {len(s_trades)}")
                print(f"  Win Rate: {s_trades['win'].mean()*100:.2f}%")
                print(f"  PnL: {s_trades['pnl'].sum():.2f}$")

    # Сохранение результатов
    with open("backtest_ict_multi_results.txt", "w", encoding="utf-8") as f:
        f.write("ICT MULTI-SYMBOL BACKTEST RESULTS\n")
        f.write(f"Initial Balance: {initial_balance}$\n")
        f.write(f"Final Balance: {total_balance:.2f}$\n")
        if all_trades:
            f.write(f"Total Trades: {len(all_trades)}\n")
            f.write(f"Win Rate: {tr_df['win'].mean()*100:.2f}%\n")
            f.write("\nTrades log:\n")
            f.write(tr_df.to_string())

if __name__ == "__main__":
    run_multi_symbol_backtest()
