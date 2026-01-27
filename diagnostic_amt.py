# Новый Python файл
# Начните писать свой код здесь
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import sys

# Добавляем путь для импорта
sys.path.append('.')  # или укажите правильный путь

from bot.exchange.bybit_client import BybitClient
from bot.config import ApiSettings
from bot.amt_orderflow_strategy import (
    get_signals_for_symbol, 
    calculate_atr, 
    AMT_CONFIG_REGISTRY,
    enhanced_signal_filter
)
from bot.strategy import Signal, Action
from bot.logger_config import log as bot_log

class DiagnosticTrailingStop:
    """Упрощенный трейлинг стоп для диагностики"""
    def __init__(self, activation_pct=0.003, trail_pct=0.002):
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.activated = False
        self.best_price = 0.0
        self.current_stop = 0.0
        self.entry_price = 0.0
        self.action = None

    def reset(self, entry_price: float, action: Action):
        self.entry_price = entry_price
        self.best_price = entry_price
        self.action = action
        self.activated = False
        if action == Action.LONG:
            self.current_stop = entry_price * (1 - self.trail_pct)
        else:
            self.current_stop = entry_price * (1 + self.trail_pct)

    def update(self, current_price: float):
        if self.action == Action.LONG:
            if current_price > self.best_price:
                self.best_price = current_price
                if not self.activated and current_price >= self.entry_price * (1 + self.activation_pct):
                    self.activated = True
                if self.activated:
                    self.current_stop = self.best_price * (1 - self.trail_pct)
        else:
            if current_price < self.best_price:
                self.best_price = current_price
                if not self.activated and current_price <= self.entry_price * (1 - self.activation_pct):
                    self.activated = True
                if self.activated:
                    self.current_stop = self.best_price * (1 + self.trail_pct)

    def should_close(self, current_price: float) -> bool:
        if not self.activated:
            return False
        if self.action == Action.LONG:
            return current_price <= self.current_stop
        else:
            return current_price >= self.current_stop

@dataclass
class DiagnosticPosition:
    entry_price: float
    entry_time: datetime
    action: Action
    tp_price: float
    sl_price: float
    qty: float
    trailing: DiagnosticTrailingStop

class DiagnosticSimulator:
    def __init__(self, initial_balance=100.0, risk_per_trade=0.15):
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.position = None
        self.history = []
        self.total_trades = 0
        self.winning_trades = 0

    def _check_tp_sl(self, current_price: float, current_time: datetime):
        if not self.position:
            return
        
        pos = self.position
        
        # Проверяем TP/SL
        if pos.action == Action.LONG:
            if current_price >= pos.tp_price:
                self._close_position(pos.tp_price, current_time, "TP_hit")
            elif current_price <= pos.sl_price:
                self._close_position(pos.sl_price, current_time, "SL_hit")
        else:
            if current_price <= pos.tp_price:
                self._close_position(pos.tp_price, current_time, "TP_hit")
            elif current_price >= pos.sl_price:
                self._close_position(pos.sl_price, current_time, "SL_hit")

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        if not self.position:
            return
            
        pos = self.position
        
        # Рассчитываем PnL
        if pos.action == Action.LONG:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        pnl_val = self.balance * self.risk_per_trade * pnl_pct
        self.balance += pnl_val
        
        self.total_trades += 1
        if pnl_pct > 0:
            self.winning_trades += 1
        
        self.history.append({
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "action": pos.action.value,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_val": pnl_val,
            "reason": reason
        })
        
        print(f"  [{exit_time}] Закрытие {pos.action.value}: {pos.entry_price:.2f} -> {exit_price:.2f} ({pnl_pct:.2%}), причина: {reason}")
        
        self.position = None

    def run(self, df_candles: pd.DataFrame, signals: List[Tuple[datetime, Signal]]):
        """
        signals: список кортежей (время свечи, сигнал)
        """
        # Сортируем сигналы по времени
        signals.sort(key=lambda x: x[0])
        
        for idx in range(len(df_candles)):
            ts = df_candles.index[idx]
            current_price = df_candles.iloc[idx]['close']
            
            # Проверяем существующую позицию
            if self.position:
                self._check_tp_sl(current_price, ts)
            
            # Ищем сигналы для этой свечи
            for signal_ts, sig in signals:
                # Если сигнал в пределах этой свечи
                if signal_ts == ts and not self.position:
                    print(f"\n[ПОПЫТКА ОТКРЫТИЯ] {ts}: {sig.action.value} по {sig.price:.2f}")
                    
                    # Рассчитываем ATR
                    atr = df_candles.iloc[idx].get('atr', sig.price * 0.002)
                    
                    # Упрощенные TP/SL
                    if sig.action == Action.LONG:
                        tp = sig.price * 1.015  # +1.5%
                        sl = sig.price * 0.985  # -1.5%
                    else:
                        tp = sig.price * 0.985  # -1.5%
                        sl = sig.price * 1.015  # +1.5%
                    
                    # Открываем позицию
                    trailing = DiagnosticTrailingStop()
                    trailing.reset(sig.price, sig.action)
                    
                    self.position = DiagnosticPosition(
                        entry_price=sig.price,
                        entry_time=ts,
                        action=sig.action,
                        tp_price=tp,
                        sl_price=sl,
                        qty=self.balance * self.risk_per_trade / sig.price,
                        trailing=trailing
                    )
                    
                    print(f"  Открыта {sig.action.value} позиция по {sig.price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}")
                    break

def create_test_trades_for_candle(candle, candle_time, num_trades=20):
    """Создает тестовые сделки с явным дисбалансом"""
    trades = []
    
    high = candle['high']
    low = candle['low']
    volume = candle['volume']
    
    # Создаем дисбаланс в сторону покупок (70% покупок, 30% продаж)
    for i in range(num_trades):
        trade_time = candle_time + timedelta(seconds=i*2)
        price = np.random.uniform(low, high)
        
        # 70% покупок, 30% продаж
        if np.random.random() < 0.7:
            side = "BUY"
        else:
            side = "SELL"
        
        # Объем пропорционален общему объему свечи
        qty = volume / num_trades * np.random.uniform(0.8, 1.2)
        
        trades.append({
            "time": trade_time,
            "price": price,
            "qty": qty,
            "side": side
        })
    
    df = pd.DataFrame(trades)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def test_signal_generation():
    """Тестируем генерацию сигналов напрямую"""
    print("=" * 80)
    print("🔍 ДИАГНОСТИКА ГЕНЕРАЦИИ СИГНАЛОВ")
    print("=" * 80)
    
    # Создаем тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=200, freq='15min', tz='UTC')
    df_candles = pd.DataFrame({
        'open': np.random.uniform(100, 200, 200),
        'high': np.random.uniform(105, 210, 200),
        'low': np.random.uniform(95, 195, 200),
        'close': np.random.uniform(100, 200, 200),
        'volume': np.random.uniform(1000, 5000, 200)
    }, index=dates)
    
    # Добавляем тренд
    df_candles['close'] = np.linspace(100, 200, 200) + np.random.normal(0, 5, 200)
    df_candles['high'] = df_candles['close'] + np.random.uniform(1, 5, 200)
    df_candles['low'] = df_candles['close'] - np.random.uniform(1, 5, 200)
    
    # Рассчитываем ATR
    df_candles['atr'] = calculate_atr(df_candles)
    
    # Тестируем для разных символов
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        print(f"\n📊 Тестирование {symbol}:")
        print("-" * 40)
        
        # Получаем настройки
        settings = AMT_CONFIG_REGISTRY.get(symbol)
        
        # Тестируем последние 10 свечей
        for i in range(190, 200):
            ts = df_candles.index[i]
            candle = df_candles.iloc[i]
            
            # Создаем окно данных
            df_window = df_candles.iloc[max(0, i-100):i+1]
            
            # Создаем сделки с дисбалансом
            trades_df = create_test_trades_for_candle(candle, ts, num_trades=50)
            
            # Рассчитываем CVD метрики вручную для проверки
            trades_df["signed_vol"] = np.where(trades_df["side"] == "BUY", 
                                              trades_df["qty"], 
                                              -trades_df["qty"])
            cvd = trades_df["signed_vol"].sum()
            total_volume = trades_df["qty"].sum()
            
            print(f"  [{ts}] Цена: {candle['close']:.2f}, CVD: {cvd:.0f}, Объем: {total_volume:.0f}")
            
            # Пробуем вызвать функцию генерации сигналов
            try:
                # Имитируем вызов get_signals_for_symbol
                class MockClient:
                    def get_recent_trades(self, symbol, limit=1000):
                        # Возвращаем наши тестовые сделки
                        return trades_df.to_dict('records')
                
                client = MockClient()
                
                # Вызываем generate_amt_signals напрямую
                from bot.amt_orderflow_strategy import generate_amt_signals
                
                signals = generate_amt_signals(
                    client=client,
                    symbol=symbol,
                    current_price=candle['close'],
                    df_ohlcv=df_window,
                    vp_config=settings.volume_profile if settings else None,
                    abs_config=settings.absorption if settings else None,
                    delta_aggr_mult=2.0,
                    trades_df=trades_df,
                    current_time=ts
                )
                
                if signals:
                    print(f"    ✅ Найдено {len(signals)} сигналов:")
                    for sig in signals:
                        print(f"      - {sig.action.value}: {sig.price:.2f}, причина: {sig.reason}")
                else:
                    print(f"    ❌ Сигналы не найдены")
                    
            except Exception as e:
                print(f"    ⚠️ Ошибка: {e}")
                import traceback
                traceback.print_exc()

def run_simple_backtest(symbol="ETHUSDT", days_back=3):
    """Упрощенный бэктест для диагностики"""
    print(f"\n{'='*80}")
    print(f"🚀 УПРОЩЕННЫЙ БЭКТЕСТ ДЛЯ {symbol}")
    print(f"{'='*80}")
    
    # Создаем тестовые данные
    num_candles = days_back * 24 * 4  # 4 свечи в час
    dates = pd.date_range(end=datetime.now(timezone.utc), 
                         periods=num_candles, 
                         freq='15min')
    
    # Создаем реалистичный тренд
    base_price = 2500 if symbol == "ETHUSDT" else 50000 if symbol == "BTCUSDT" else 100
    
    # Имитируем тренд вверх
    trend = np.linspace(0, 0.05, num_candles)  # +5% за период
    noise = np.random.normal(0, 0.005, num_candles)  # шум 0.5%
    
    prices = base_price * (1 + trend + noise)
    
    df_candles = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.002, 0.002, num_candles)),
        'high': prices * (1 + np.random.uniform(0.001, 0.01, num_candles)),
        'low': prices * (1 + np.random.uniform(-0.01, -0.001, num_candles)),
        'close': prices,
        'volume': np.random.uniform(10000, 50000, num_candles)
    }, index=dates)
    
    # Рассчитываем ATR
    df_candles['atr'] = calculate_atr(df_candles)
    
    print(f"Создано {len(df_candles)} свечей")
    print(f"Цена от {df_candles['close'].iloc[0]:.2f} до {df_candles['close'].iloc[-1]:.2f}")
    
    # Собираем сигналы
    all_signals = []
    settings = AMT_CONFIG_REGISTRY.get(symbol)
    
    print("\n🔍 Генерация сигналов...")
    
    for i in range(100, len(df_candles)):
        ts = df_candles.index[i]
        candle = df_candles.iloc[i]
        df_window = df_candles.iloc[max(0, i-100):i+1]
        
        # Создаем сделки с явным дисбалансом для тестирования
        trades = []
        total_volume = candle['volume']
        
        # Создаем сильный дисбаланс покупок (80% покупок, 20% продаж)
        for j in range(30):
            trade_time = ts - timedelta(seconds=(30-j)*30)
            price = np.random.uniform(candle['low'], candle['high'])
            
            if j < 24:  # 80% покупок
                side = "BUY"
            else:  # 20% продаж
                side = "SELL"
            
            qty = total_volume / 30 * np.random.uniform(0.8, 1.2)
            
            trades.append({
                "time": trade_time,
                "price": price,
                "qty": qty,
                "side": side
            })
        
        trades_df = pd.DataFrame(trades)
        trades_df["time"] = pd.to_datetime(trades_df["time"], utc=True)
        
        # Вручную проверяем, проходит ли сигнал
        total_buy = trades_df[trades_df["side"] == "BUY"]["qty"].sum()
        total_sell = trades_df[trades_df["side"] == "SELL"]["qty"].sum()
        total_volume = total_buy + total_sell
        cvd_delta = total_buy - total_sell
        ratio = total_buy / max(total_sell, 0.001)
        
        # Проверяем условия поглощения
        if settings:
            abs_config = settings.absorption
            
            # Проверяем LONG сигнал (сильные продажи)
            if total_volume >= abs_config.min_total_volume and -cvd_delta >= abs_config.min_cvd_delta:
                sell_ratio = total_sell / max(total_buy, 0.001)
                if sell_ratio >= abs_config.min_buy_sell_ratio:
                    sig = Signal(
                        timestamp=pd.Timestamp(ts),
                        action=Action.LONG,
                        price=candle['close'],
                        reason=f"TEST_abs_long_cvd_{int(cvd_delta)}"
                    )
                    all_signals.append((ts, sig))
                    print(f"[{ts}] Создан тестовый LONG сигнал: CVD={cvd_delta:.0f}, Ratio={sell_ratio:.2f}")
            
            # Проверяем SHORT сигнал (сильные покупки)
            if total_volume >= abs_config.min_total_volume and cvd_delta >= abs_config.min_cvd_delta:
                if ratio >= abs_config.min_buy_sell_ratio:
                    sig = Signal(
                        timestamp=pd.Timestamp(ts),
                        action=Action.SHORT,
                        price=candle['close'],
                        reason=f"TEST_abs_short_cvd_{int(cvd_delta)}"
                    )
                    all_signals.append((ts, sig))
                    print(f"[{ts}] Создан тестовый SHORT сигнал: CVD={cvd_delta:.0f}, Ratio={ratio:.2f}")
    
    print(f"\n📈 Сгенерировано {len(all_signals)} тестовых сигналов")
    
    # Запускаем симулятор
    if all_signals:
        sim = DiagnosticSimulator(initial_balance=1000.0, risk_per_trade=0.1)
        sim.run(df_candles, all_signals)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА {symbol}:")
        print(f"  Всего сделок: {sim.total_trades}")
        print(f"  Выигрышных: {sim.winning_trades}")
        
        if sim.total_trades > 0:
            win_rate = (sim.winning_trades / sim.total_trades) * 100
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Финальный баланс: ${sim.balance:.2f}")
            print(f"  Общий PnL: ${sim.balance - 1000:.2f}")
            
            if sim.history:
                print("\n  Последние сделки:")
                for trade in sim.history[-5:]:
                    print(f"    {trade['action']}: {trade['entry_price']:.2f}->{trade['exit_price']:.2f} "
                          f"({trade['pnl_pct']:.1f}%), {trade['reason']}")
    else:
        print("⚠️ Нет сигналов для бэктеста")

def check_configuration():
    """Проверяем конфигурацию стратегии"""
    print("\n🔧 ПРОВЕРКА КОНФИГУРАЦИИ:")
    print("-" * 40)
    
    for symbol, settings in AMT_CONFIG_REGISTRY.items():
        print(f"\n{symbol}:")
        print(f"  Absorption Config:")
        print(f"    lookback_seconds: {settings.absorption.lookback_seconds}")
        print(f"    min_total_volume: {settings.absorption.min_total_volume}")
        print(f"    min_cvd_delta: {settings.absorption.min_cvd_delta}")
        print(f"    min_buy_sell_ratio: {settings.absorption.min_buy_sell_ratio}")
        print(f"    max_price_drift_pct: {settings.absorption.max_price_drift_pct}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Диагностика AMT стратегии')
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["test", "backtest", "config", "all"],
                       help="Режим работы: test=тест сигналов, backtest=бэктест, config=проверка конфигурации")
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--days", type=int, default=3)
    
    args = parser.parse_args()
    
    if args.mode in ["test", "all"]:
        test_signal_generation()
    
    if args.mode in ["config", "all"]:
        check_configuration()
    
    if args.mode in ["backtest", "all"]:
        run_simple_backtest(symbol=args.symbol, days_back=args.days)
    
    print("\n" + "="*80)
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*80)