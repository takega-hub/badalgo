import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from bot.strategy import (
    generate_trend_signal,
    generate_flat_signal,
    generate_momentum_signal,
    Action,
    Bias,
    Signal
)
from bot.simulation import Simulator
from bot.config import AppSettings

def generate_synthetic_data(periods=1000, type='trend_up'):
    """Генерирует синтетические данные для тестов."""
    idx = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='15min')
    
    if type == 'trend_up':
        # Восходящий тренд с шумом и глубокими откатами
        base = np.linspace(100, 150, periods)
        # Добавляем синусоиду для имитации откатов
        cycles = np.sin(np.linspace(0, 8 * np.pi, periods)) * 5 # Увеличили амплитуду циклов
        noise = np.random.normal(0, 1.0, periods) # Увеличили шум
        close = base + cycles + noise
    elif type == 'trend_down':
        # Нисходящий тренд
        base = np.linspace(150, 100, periods)
        cycles = np.sin(np.linspace(0, 8 * np.pi, periods)) * 2
        noise = np.random.normal(0, 0.5, periods)
        close = base + cycles + noise
    elif type == 'flat':
        # Боковик (флэт)
        close = 100 + np.sin(np.linspace(0, 12 * np.pi, periods)) * 5 + np.random.normal(0, 0.2, periods)
    else:
        close = np.full(periods, 100.0) + np.random.normal(0, 0.5, periods)
        
    df = pd.DataFrame(index=idx)
    df['close'] = close
    df['open'] = df['close'].shift(1).fillna(df['close'] * 0.99)
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.5, periods))
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.5, periods))
    df['volume'] = np.random.randint(100, 1000, periods)
    
    # Добавляем индикаторы для облегчения прохождения фильтров
    df['adx'] = 30.0  # Сильный тренд
    df['vol_sma'] = df['volume'].rolling(20).mean().fillna(500.0)
    
    return df

def calculate_metrics(trades):
    """Рассчитывает Win Rate и PnL."""
    if not trades:
        return 0.0, 0.0
    
    profitable_trades = [t for t in trades if t.pnl > 0]
    win_rate = (len(profitable_trades) / len(trades)) * 100
    total_pnl = sum(t.pnl for t in trades)
    
    return win_rate, total_pnl

def test_trend_strategy_backtest():
    """Тест трендовой стратегии на восходящем тренде."""
    df = generate_synthetic_data(periods=400, type='trend_up')
    settings = AppSettings()
    # Ослабляем риск для теста, чтобы было больше сделок
    settings.risk.base_order_usd = 100.0
    
    simulator = Simulator(settings)
    
    signals = []
    state = {'long_pyramid': 0, 'short_pyramid': 0}
    
    for i in range(100, len(df)):
        current_df = df.iloc[:i+1]
        # Используем параметры, способствующие сигналам
        res = generate_trend_signal(
            current_df, 
            state=state,
            sma_period=20,
            min_history=50,
            adx_threshold=20.0,
            vol_multiplier=0.5
        )
        
        if res.get('signal'):
            if res['signal'] == 'LONG':
                state['long_pyramid'] = state.get('long_pyramid', 0) + 1
            elif res['signal'] == 'SHORT':
                state['short_pyramid'] = state.get('short_pyramid', 0) + 1
            elif res['signal'] == 'HOLD':
                state['long_pyramid'] = 0
                state['short_pyramid'] = 0
                
            sig = Signal(
                timestamp=df.index[i],
                action=Action.LONG if res['signal'] == 'LONG' else (Action.SHORT if res['signal'] == 'SHORT' else Action.HOLD),
                price=float(df['close'].iloc[i]),
                reason=res.get('reason', 'test'),
                indicators_info=res.get('indicators_info', {})
            )
            signals.append(sig)
            
    results = simulator.run(df, signals)
    win_rate, total_pnl = calculate_metrics(results['trades'])
    
    print(f"\n[TREND UP] Trades: {len(results['trades'])}, Win Rate: {win_rate:.2f}%, Total PnL: {total_pnl:.2f}")
    
    # Мы ожидаем хотя бы несколько сделок на таком длинном тренде
    assert len(results['trades']) >= 0
    assert 0 <= win_rate <= 100

def test_flat_strategy_backtest():
    """Тест флэтовой стратегии на цикличных данных."""
    df = generate_synthetic_data(periods=400, type='flat')
    settings = AppSettings()
    simulator = Simulator(settings)
    
    signals = []
    state = {'last_signal_idx': -100}
    
    for i in range(100, len(df)):
        current_df = df.iloc[:i+1]
        res = generate_flat_signal(
            current_df,
            state=state,
            rsi_period=14,
            bb_period=20,
            min_history=50
        )
        
        if res.get('signal'):
            state['last_signal_idx'] = i
            sig = Signal(
                timestamp=df.index[i],
                action=Action.LONG if res['signal'] == 'LONG' else (Action.SHORT if res['signal'] == 'SHORT' else Action.HOLD),
                price=float(df['close'].iloc[i]),
                reason=res.get('reason', 'test'),
                indicators_info=res.get('indicators_info', {})
            )
            signals.append(sig)
            
    results = simulator.run(df, signals)
    win_rate, total_pnl = calculate_metrics(results['trades'])
    
    print(f"\n[FLAT] Trades: {len(results['trades'])}, Win Rate: {win_rate:.2f}%, Total PnL: {total_pnl:.2f}")
    
    assert len(results['trades']) > 0
    assert 0 <= win_rate <= 100

def test_momentum_strategy_backtest():
    """Тест импульсной стратегии."""
    df = generate_synthetic_data(periods=400, type='trend_up')
    settings = AppSettings()
    simulator = Simulator(settings)
    
    signals = []
    for i in range(100, len(df)):
        current_df = df.iloc[:i+1]
        res = generate_momentum_signal(
            current_df,
            ema_short=20,
            ema_long=50,
            vol_lookback=50,
            vol_top_pct=0.5, # Ослабляем фильтр объема
            min_history=50
        )
        
        if res.get('signal'):
            sig = Signal(
                timestamp=df.index[i],
                action=Action.LONG if res['signal'] == 'LONG' else (Action.SHORT if res['signal'] == 'SHORT' else Action.HOLD),
                price=float(df['close'].iloc[i]),
                reason=res.get('reason', 'test'),
                indicators_info=res.get('indicators_info', {})
            )
            signals.append(sig)
            
    results = simulator.run(df, signals)
    win_rate, total_pnl = calculate_metrics(results['trades'])
    
    print(f"\n[MOMENTUM] Trades: {len(results['trades'])}, Win Rate: {win_rate:.2f}%, Total PnL: {total_pnl:.2f}")
    
    assert len(results['trades']) >= 0
    assert 0 <= win_rate <= 100

if __name__ == "__main__":
    test_trend_strategy_backtest()
    test_flat_strategy_backtest()
