"""
Скрипт бэктестинга для стратегии V17 Optimized
Тестирует обученную модель на исторических данных
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_env_v17_optimized import CryptoTradingEnvV17_Optimized
from indicators import prepare_with_indicators


def load_historical_data(symbol: str = 'BTCUSDT', days: int = 30, timeframe: str = '15m'):
    """
    Загрузка исторических данных
    
    Args:
        symbol: Торговая пара (BTCUSDT, ETHUSDT, SOLUSDT)
        days: Количество дней истории
        timeframe: Таймфрейм (15m, 1h, 4h)
    
    Returns:
        DataFrame с историческими данными
    """
    print(f"\n{'='*60}")
    print(f"📥 ЗАГРУЗКА ДАННЫХ: {symbol} ({days} дней, {timeframe})")
    print(f"{'='*60}")
    
    # Проверяем наличие локального файла
    data_file = f'./data/{symbol.lower()}_{timeframe}.csv'
    
    if os.path.exists(data_file):
        print(f"✅ Найден локальный файл: {data_file}")
        df = pd.read_csv(data_file)
        
        # Конвертируем timestamp если нужно
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'int64':
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Фильтруем по количеству дней
        if 'datetime' in df.columns:
            cutoff_date = df['datetime'].max() - timedelta(days=days)
            df = df[df['datetime'] >= cutoff_date].copy()
            print(f"   Загружено {len(df)} свечей из файла")
            print(f"   Период: {df['datetime'].min()} - {df['datetime'].max()}")
        
        return df
    
    # Если файла нет, пытаемся загрузить с биржи
    try:
        import ccxt
        print(f"📡 Загрузка данных с Binance...")
        
        exchange = ccxt.binance()
        
        # Рассчитываем количество свечей
        if timeframe == '15m':
            candles_per_day = 96
        elif timeframe == '1h':
            candles_per_day = 24
        elif timeframe == '4h':
            candles_per_day = 6
        else:
            candles_per_day = 96
        
        target_candles = days * candles_per_day
        
        # Конвертируем символ для Binance
        binance_symbol = symbol.replace('USDT', '/USDT')
        
        # Загружаем данные
        ms_per_candle = {
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000
        }.get(timeframe, 15 * 60 * 1000)
        
        duration_ms = target_candles * ms_per_candle
        start_time = exchange.milliseconds() - duration_ms
        
        all_ohlcv = []
        current_since = start_time
        
        while len(all_ohlcv) < target_candles:
            try:
                new_ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, since=current_since, limit=1000)
                
                if not new_ohlcv:
                    break
                
                all_ohlcv.extend(new_ohlcv)
                current_since = new_ohlcv[-1][0] + 1
                
                print(f"   Загружено: {len(all_ohlcv)} / {target_candles} свечей...")
                
                if current_since > exchange.milliseconds():
                    break
                
                import time
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки: {e}")
                import time
                time.sleep(5)
                continue
        
        # Формируем DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Сохраняем для будущего использования
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_file, index=False)
        print(f"✅ Данные сохранены в {data_file}")
        
        return df
        
    except ImportError:
        print("❌ Модуль ccxt не установлен. Установите: pip install ccxt")
        return None
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None


def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка данных с индикаторами для V17 стратегии
    """
    print(f"\n{'='*60}")
    print("📊 ПОДГОТОВКА ДАННЫХ С ИНДИКАТОРАМИ")
    print(f"{'='*60}")
    
    # Используем функцию из indicators.py
    df_ind = prepare_with_indicators(
        df,
        adx_length=14,
        di_length=14,
        sma_length=200,
        rsi_length=14,
        breakout_lookback=20,
        bb_length=20,
        bb_std=2,
        atr_length=14,
        ema_fast_length=12,
        ema_slow_length=26,
        ema_timeframe='1h'
    )
    
    # Добавляем дополнительные индикаторы для V17
    # RSI normalized
    if 'rsi' in df_ind.columns:
        df_ind['rsi_norm'] = (df_ind['rsi'] - 50) / 50
    else:
        # Рассчитываем RSI если его нет
        delta = df_ind['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ind['rsi'] = 100 - (100 / (1 + rs))
        df_ind['rsi'] = df_ind['rsi'].fillna(50)
        df_ind['rsi_norm'] = (df_ind['rsi'] - 50) / 50
    
    # Trend bias (используем DI или SMA)
    if 'plus_di' in df_ind.columns and 'minus_di' in df_ind.columns:
        df_ind['trend_bias_1h'] = np.where(
            df_ind['plus_di'] > df_ind['minus_di'], 0.5, -0.5
        )
    elif 'DMP_14' in df_ind.columns and 'DMN_14' in df_ind.columns:
        df_ind['trend_bias_1h'] = np.where(
            df_ind['DMP_14'] > df_ind['DMN_14'], 0.5, -0.5
        )
    else:
        # Fallback на SMA
        if 'sma' in df_ind.columns:
            df_ind['trend_bias_1h'] = np.where(
                df_ind['close'] > df_ind['sma'], 0.5, -0.5
            )
        else:
            df_ind['trend_bias_1h'] = 0.0
    
    # Volatility ratio
    df_ind['returns'] = df_ind['close'].pct_change()
    df_ind['volatility_ratio'] = df_ind['returns'].rolling(20).std().fillna(1.5)
    
    # Volume ratio
    df_ind['volume_ratio'] = df_ind['volume'] / df_ind['volume'].rolling(20).mean().fillna(1.2)
    
    # Заполняем пропуски
    for col in df_ind.columns:
        if df_ind[col].isnull().any() and pd.api.types.is_numeric_dtype(df_ind[col]):
            df_ind[col] = df_ind[col].fillna(df_ind[col].mean() if not df_ind[col].isnull().all() else 0)
    
    print(f"✅ Подготовлено {len(df_ind)} строк с индикаторами")
    print(f"   Доступные колонки: {len(df_ind.columns)}")
    
    return df_ind


def load_model(model_path: str = None):
    """
    Загрузка обученной модели V17
    """
    if model_path is None:
        model_path = './models/v17_optimized/ppo_final.zip'
    
    if not os.path.exists(model_path):
        # Пробуем найти любую модель в директории
        model_dir = './models/v17_optimized/'
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if models:
                model_path = os.path.join(model_dir, models[0])
                print(f"⚠️ Используется модель: {model_path}")
            else:
                print(f"❌ Модели не найдены в {model_dir}")
                return None
        else:
            print(f"❌ Модель не найдена: {model_path}")
            return None
    
    print(f"\n{'='*60}")
    print(f"🤖 ЗАГРУЗКА МОДЕЛИ")
    print(f"{'='*60}")
    print(f"   Путь: {model_path}")
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Модель загружена успешно")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None


def run_backtest(model, df: pd.DataFrame, symbol: str = 'BTCUSDT', initial_balance: float = 10000.0):
    """
    Запуск бэктеста на исторических данных
    """
    print(f"\n{'='*60}")
    print(f"🚀 ЗАПУСК БЭКТЕСТА: {symbol}")
    print(f"{'='*60}")
    
    # Определяем колонки для наблюдений
    obs_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    additional_cols = ['rsi_norm', 'trend_bias_1h', 'volatility_ratio', 'volume_ratio']
    
    for col in additional_cols:
        if col in df.columns:
            obs_cols.append(col)
    
    print(f"   Используется {len(obs_cols)} признаков для наблюдений")
    
    # Создаем лог-файл
    log_file = f'./logs/v17_optimized/backtest_{symbol.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Создаем среду
    def make_env():
        return CryptoTradingEnvV17_Optimized(
            df=df.copy(),
            obs_cols=obs_cols,
            initial_balance=initial_balance,
            commission=0.001,
            slippage=0.0005,
            log_file=log_file,
            training_mode='optimized'
        )
    
    env = DummyVecEnv([make_env])
    
    # Запускаем бэктест
    # DummyVecEnv.reset() возвращает только наблюдения (массив), не кортеж
    obs = env.reset()
    done = False
    steps = 0
    max_steps = len(df)
    
    print(f"   Максимальное количество шагов: {max_steps}")
    print(f"   Начальный баланс: ${initial_balance:.2f}")
    print(f"\n   Запуск симуляции...")
    
    while not done and steps < max_steps:
        # Предсказываем действие
        action, _ = model.predict(obs, deterministic=True)
        
        # Выполняем шаг
        # DummyVecEnv.step() возвращает (obs, reward, done, info)
        obs, reward, done_array, info = env.step(action)
        
        # done может быть массивом или булевым значением
        if isinstance(done_array, (list, np.ndarray)):
            done = bool(done_array[0])
        else:
            done = bool(done_array)
        steps += 1
        
        # Прогресс каждые 10%
        if steps % (max_steps // 10) == 0:
            # Получаем информацию из среды напрямую
            try:
                env_info = env.get_attr('_get_info')[0]()
                net_worth = env_info.get('net_worth', initial_balance) if isinstance(env_info, dict) else initial_balance
            except:
                net_worth = initial_balance
            progress = (steps / max_steps) * 100
            print(f"   [{progress:.0f}%] Шаг {steps}/{max_steps}, Net Worth: ${net_worth:.2f}")
    
    env.close()
    
    print(f"\n✅ Бэктест завершен")
    print(f"   Выполнено шагов: {steps}")
    print(f"   Лог-файл: {log_file}")
    
    return log_file


def analyze_results(log_file: str, initial_balance: float = 10000.0, symbol: str = 'BTCUSDT'):
    """
    Анализ результатов бэктеста
    
    Args:
        log_file: Путь к лог-файлу с результатами сделок
        initial_balance: Начальный баланс
        symbol: Торговая пара (для отчета)
    """
    print(f"\n{'='*60}")
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    
    if not os.path.exists(log_file):
        print(f"❌ Лог-файл не найден: {log_file}")
        return None
    
    try:
        df_log = pd.read_csv(log_file)
        
        if len(df_log) <= 1:
            print(f"⚠️ В логе недостаточно данных для анализа")
            return None
        
        # Фильтруем только закрытые сделки
        closed_trades = df_log[df_log['exit_reason'].notna()].copy()
        
        if len(closed_trades) == 0:
            print(f"⚠️ Нет закрытых сделок для анализа")
            return None
        
        # Конвертируем pnl_percent в числовой формат, если это строка
        if 'pnl_percent' in closed_trades.columns:
            # Убираем знак % и конвертируем в float
            if closed_trades['pnl_percent'].dtype == 'object':
                # Обрабатываем строки с процентами или другими форматами
                closed_trades['pnl_percent'] = (
                    closed_trades['pnl_percent']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .str.strip()
                )
                closed_trades['pnl_percent'] = pd.to_numeric(closed_trades['pnl_percent'], errors='coerce')
            else:
                closed_trades['pnl_percent'] = pd.to_numeric(closed_trades['pnl_percent'], errors='coerce')
        
        # Удаляем строки с NaN в pnl_percent
        closed_trades = closed_trades[closed_trades['pnl_percent'].notna()].copy()
        
        if len(closed_trades) == 0:
            print(f"⚠️ Нет валидных сделок для анализа (все pnl_percent пустые)")
            return None
        
        # Базовые метрики
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl_percent'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl_percent'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL метрики
        total_pnl = closed_trades['pnl_percent'].sum()
        avg_pnl = closed_trades['pnl_percent'].mean()
        
        # Profit Factor
        gross_profit = closed_trades[closed_trades['pnl_percent'] > 0]['pnl_percent'].sum()
        gross_loss = abs(closed_trades[closed_trades['pnl_percent'] <= 0]['pnl_percent'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # RR метрики
        if 'rr_ratio' in closed_trades.columns:
            # Конвертируем rr_ratio в числовой формат, если нужно
            if closed_trades['rr_ratio'].dtype == 'object':
                closed_trades['rr_ratio'] = pd.to_numeric(closed_trades['rr_ratio'], errors='coerce')
            closed_trades_rr = closed_trades[closed_trades['rr_ratio'].notna()]
            if len(closed_trades_rr) > 0:
                avg_rr = closed_trades_rr['rr_ratio'].mean()
                min_rr = closed_trades_rr['rr_ratio'].min()
                max_rr = closed_trades_rr['rr_ratio'].max()
                rr_above_min = len(closed_trades_rr[closed_trades_rr['rr_ratio'] >= 1.5]) / total_trades * 100
            else:
                avg_rr = min_rr = max_rr = rr_above_min = 0
        else:
            avg_rr = min_rr = max_rr = rr_above_min = 0
        
        # Просадка
        if 'net_worth' in closed_trades.columns:
            equity_curve = closed_trades['net_worth'].values
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak * 100
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        # Финальный баланс
        final_balance = closed_trades['net_worth'].iloc[-1] if 'net_worth' in closed_trades.columns else initial_balance
        total_return = ((final_balance / initial_balance) - 1) * 100
        
        # Exit reasons
        exit_reasons = closed_trades['exit_reason'].value_counts()
        
        # Выводим результаты
        print(f"\n📈 ОСНОВНЫЕ МЕТРИКИ:")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Прибыльных: {winning_trades} ({win_rate:.1f}%)")
        print(f"   Убыточных: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"   Win Rate: {win_rate:.2f}%")
        
        print(f"\n💰 ДОХОДНОСТЬ:")
        print(f"   Начальный баланс: ${initial_balance:.2f}")
        print(f"   Финальный баланс: ${final_balance:.2f}")
        print(f"   Общий PnL: {total_pnl:.2f}%")
        print(f"   Общая доходность: {total_return:.2f}%")
        print(f"   Средний PnL на сделку: {avg_pnl:.2f}%")
        
        print(f"\n📊 КАЧЕСТВО СДЕЛОК:")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Средний RR: {avg_rr:.2f}")
        print(f"   Минимальный RR: {min_rr:.2f}")
        print(f"   Максимальный RR: {max_rr:.2f}")
        print(f"   Сделок с RR ≥ 1.5: {rr_above_min:.1f}%")
        
        print(f"\n📉 РИСКИ:")
        print(f"   Максимальная просадка: {max_drawdown:.2f}%")
        
        print(f"\n🚪 ПРИЧИНЫ ВЫХОДА:")
        for reason, count in exit_reasons.items():
            pct = (count / total_trades) * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
        
        # Сохраняем отчет
        report = {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'avg_rr': avg_rr,
            'min_rr': min_rr,
            'max_rr': max_rr,
            'rr_above_min_pct': rr_above_min,
            'max_drawdown': max_drawdown,
            'initial_balance': initial_balance,
            'final_balance': final_balance
        }
        
        return report
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Бэктестинг стратегии V17 Optimized')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Торговая пара (BTCUSDT, ETHUSDT, SOLUSDT)')
    parser.add_argument('--days', type=int, default=30, 
                       help='Количество дней истории')
    parser.add_argument('--timeframe', type=str, default='15m', 
                       help='Таймфрейм (15m, 1h, 4h)')
    parser.add_argument('--model', type=str, default=None, 
                       help='Путь к модели (по умолчанию: ./models/v17_optimized/ppo_final.zip)')
    parser.add_argument('--balance', type=float, default=10000.0, 
                       help='Начальный баланс')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 БЭКТЕСТИНГ СТРАТЕГИИ V17 OPTIMIZED")
    print("="*60)
    
    # 1. Загрузка данных
    df = load_historical_data(args.symbol, args.days, args.timeframe)
    if df is None or len(df) == 0:
        print("❌ Не удалось загрузить данные")
        return
    
    # 2. Подготовка данных с индикаторами
    df_prepared = prepare_data_with_indicators(df)
    if df_prepared is None or len(df_prepared) == 0:
        print("❌ Не удалось подготовить данные")
        return
    
    # 3. Загрузка модели
    model = load_model(args.model)
    if model is None:
        print("❌ Не удалось загрузить модель")
        return
    
    # 4. Запуск бэктеста
    log_file = run_backtest(model, df_prepared, args.symbol, args.balance)
    
    # 5. Анализ результатов
    report = analyze_results(log_file, args.balance, args.symbol)
    
    if report:
        print(f"\n{'='*60}")
        print("✅ БЭКТЕСТИНГ ЗАВЕРШЕН УСПЕШНО")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("⚠️ БЭКТЕСТИНГ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
