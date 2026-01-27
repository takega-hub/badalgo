# Создайте новый файл: amt_simple_backtest.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SimpleSignal:
    timestamp: datetime
    action: str  # "LONG" or "SHORT"
    price: float
    reason: str

def generate_simple_signals(df_candles: pd.DataFrame) -> List[Tuple[datetime, SimpleSignal]]:
    """Генерирует простые сигналы на основе прорыва скользящих средних"""
    signals = []
    
    # Рассчитываем индикаторы
    df_candles['sma20'] = df_candles['close'].rolling(window=20).mean()
    df_candles['sma50'] = df_candles['close'].rolling(window=50).mean()
    df_candles['volume_sma'] = df_candles['volume'].rolling(window=20).mean()
    
    for i in range(50, len(df_candles)):
        ts = df_candles.index[i]
        candle = df_candles.iloc[i]
        
        # Проверяем условия для LONG
        if (candle['close'] > candle['sma20'] and 
            candle['sma20'] > candle['sma50'] and
            candle['volume'] > candle['volume_sma'] * 1.2):
            
            signal = SimpleSignal(
                timestamp=ts,
                action="LONG",
                price=candle['close'],
                reason="MA_Breakout"
            )
            signals.append((ts, signal))
            print(f"[{ts}] LONG сигнал по {candle['close']:.2f}")
        
        # Проверяем условия для SHORT
        elif (candle['close'] < candle['sma20'] and 
              candle['sma20'] < candle['sma50'] and
              candle['volume'] > candle['volume_sma'] * 1.2):
            
            signal = SimpleSignal(
                timestamp=ts,
                action="SHORT",
                price=candle['close'],
                reason="MA_Breakout"
            )
            signals.append((ts, signal))
            print(f"[{ts}] SHORT сигнал по {candle['close']:.2f}")
    
    return signals

def run_quick_backtest():
    """Быстрый бэктест для проверки работы"""
    print("🚀 ЗАПУСК БЫСТРОГО БЭКТЕСТА")
    print("="*60)
    
    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='15min', tz='UTC')
    
    # Создаем трендовые данные
    trend = np.sin(np.linspace(0, 4*np.pi, 500)) * 100 + 2500
    noise = np.random.normal(0, 50, 500)
    
    df = pd.DataFrame({
        'open': trend + noise - np.random.uniform(10, 50, 500),
        'high': trend + noise + np.random.uniform(20, 100, 500),
        'low': trend + noise - np.random.uniform(20, 100, 500),
        'close': trend + noise,
        'volume': np.random.uniform(10000, 50000, 500)
    }, index=dates)
    
    print(f"Создано {len(df)} свечей")
    print(f"Диапазон цен: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Генерируем сигналы
    signals = generate_simple_signals(df)
    print(f"\n📈 Сгенерировано {len(signals)} сигналов")
    
    if signals:
        # Простая симуляция
        balance = 1000.0
        position = None
        trades = []
        
        for idx in range(len(df)):
            ts = df.index[idx]
            price = df.iloc[idx]['close']
            
            # Закрываем позицию если есть
            if position:
                entry_price, action, entry_time = position
                
                # Простой тейк-профит / стоп-лосс
                if action == "LONG":
                    if price >= entry_price * 1.02:  # +2%
                        pnl = (price - entry_price) / entry_price * 1000
                        balance += pnl
                        trades.append({
                            "entry": entry_time,
                            "exit": ts,
                            "action": action,
                            "pnl": pnl,
                            "return": (price - entry_price) / entry_price * 100
                        })
                        print(f"[{ts}] Закрытие LONG: {entry_price:.2f} -> {price:.2f} (+{pnl:.2f})")
                        position = None
                    elif price <= entry_price * 0.99:  # -1%
                        pnl = (price - entry_price) / entry_price * 1000
                        balance += pnl
                        trades.append({
                            "entry": entry_time,
                            "exit": ts,
                            "action": action,
                            "pnl": pnl,
                            "return": (price - entry_price) / entry_price * 100
                        })
                        print(f"[{ts}] Закрытие LONG: {entry_price:.2f} -> {price:.2f} ({pnl:.2f})")
                        position = None
                
                elif action == "SHORT":
                    if price <= entry_price * 0.98:  # -2%
                        pnl = (entry_price - price) / entry_price * 1000
                        balance += pnl
                        trades.append({
                            "entry": entry_time,
                            "exit": ts,
                            "action": action,
                            "pnl": pnl,
                            "return": (entry_price - price) / entry_price * 100
                        })
                        print(f"[{ts}] Закрытие SHORT: {entry_price:.2f} -> {price:.2f} (+{pnl:.2f})")
                        position = None
                    elif price >= entry_price * 1.01:  # +1%
                        pnl = (entry_price - price) / entry_price * 1000
                        balance += pnl
                        trades.append({
                            "entry": entry_time,
                            "exit": ts,
                            "action": action,
                            "pnl": pnl,
                            "return": (entry_price - price) / entry_price * 100
                        })
                        print(f"[{ts}] Закрытие SHORT: {entry_price:.2f} -> {price:.2f} ({pnl:.2f})")
                        position = None
            
            # Открываем новую позицию если есть сигнал
            if not position:
                for signal_ts, signal in signals:
                    if signal_ts == ts:
                        position = (signal.price, signal.action, ts)
                        print(f"[{ts}] Открытие {signal.action}: {signal.price:.2f}")
                        break
        
        # Выводим результаты
        print("\n" + "="*60)
        print("📊 ИТОГИ БЭКТЕСТА:")
        print(f"  Начальный баланс: $1000.00")
        print(f"  Финальный баланс: ${balance:.2f}")
        print(f"  Общий PnL: ${balance - 1000:.2f}")
        print(f"  Всего сделок: {len(trades)}")
        
        if trades:
            winning = [t for t in trades if t['pnl'] > 0]
            losing = [t for t in trades if t['pnl'] <= 0]
            
            print(f"  Выигрышных: {len(winning)}")
            print(f"  Проигрышных: {len(losing)}")
            
            if len(trades) > 0:
                win_rate = len(winning) / len(trades) * 100
                print(f"  Win Rate: {win_rate:.1f}%")
                
                total_profit = sum(t['pnl'] for t in winning)
                total_loss = sum(t['pnl'] for t in losing)
                
                if total_loss != 0:
                    profit_factor = abs(total_profit / total_loss)
                    print(f"  Profit Factor: {profit_factor:.2f}")
                
                avg_return = sum(t['return'] for t in trades) / len(trades)
                print(f"  Средняя доходность: {avg_return:.2f}%")
    else:
        print("⚠️ Нет сигналов для торговли")

if __name__ == "__main__":
    run_quick_backtest()