import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
"""
ВНИМАНИЕ по поводу импортов:
- При запуске как модуль:  `python -m bot.train_v17_optimized`
  модуль `bot` является пакетом, и нужно использовать `from bot.crypto_env_v17_optimized import ...`.
- При запуске как скрипт из корня: `python bot/train_v17_optimized.py`
  можно использовать простой импорт `from crypto_env_v17_optimized import ...` при наличии `bot` в PYTHONPATH.

Ниже делаем двойной импорт, чтобы покрыть оба сценария.
"""
try:
    from bot.crypto_env_v17_optimized import CryptoTradingEnvV17_Optimized
except ModuleNotFoundError:
    # fallback для запуска как скрипта из корня проекта
    from crypto_env_v17_optimized import CryptoTradingEnvV17_Optimized


class RRMonitoringCallback(BaseCallback):
    """Callback для мониторинга RR ratio и разнообразия действий"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rr_history = []
        self.trade_count = 0
        self.action_history = []
        
    def _on_step(self) -> bool:
        # Получаем информацию из среды
        try:
            if hasattr(self.locals, 'env'):
                env_info = self.locals['env'].get_attr('_get_info')[0]
                
                # Сохраняем действия для анализа разнообразия
                if hasattr(self.locals, 'actions'):
                    actions = self.locals.get('actions', [])
                    if len(actions) > 0:
                        self.action_history.append(int(actions[0]))
                        if len(self.action_history) > 1000:
                            self.action_history.pop(0)
                
                # Мониторим RR статистику
                if 'rr_stats' in env_info:
                    rr_stats = env_info['rr_stats']
                    
                    # Сохраняем историю
                    self.rr_history.append(rr_stats['avg'])
                    if len(self.rr_history) > 100:
                        self.rr_history.pop(0)
                    
                    # Логируем каждые 500 шагов
                    if self.num_timesteps % 500 == 0:
                        avg_rr = np.mean(self.rr_history) if self.rr_history else 0
                        
                        # Анализ разнообразия действий
                        action_diversity = ""
                        if len(self.action_history) >= 100:
                            action_counts = {}
                            for a in self.action_history[-100:]:
                                action_counts[a] = action_counts.get(a, 0) + 1
                            
                            action_names = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'OPEN_SHORT'}
                            action_strs = []
                            for a_id in sorted(action_counts.keys()):
                                count = action_counts[a_id]
                                pct = (count / len(self.action_history[-100:])) * 100
                                action_strs.append(f"{action_names.get(a_id, f'UNK_{a_id}')}: {pct:.1f}%")
                            action_diversity = ", ".join(action_strs)
                            
                            # Предупреждение если нет разнообразия
                            max_ratio = max(action_counts.values()) / len(self.action_history[-100:])
                            if max_ratio > 0.8:
                                print(f"⚠️ [DIVERSITY_WARNING] Модель выбирает одно действие {max_ratio*100:.1f}% времени!")
                        
                        print(f"[MONITOR] Step {self.num_timesteps}: Avg RR = {avg_rr:.2f}, "
                              f"Violations = {rr_stats['violations']}")
                        if action_diversity:
                            print(f"         Действия (последние 100): {action_diversity}")
                        
                        # Предупреждение если RR низкий
                        if avg_rr < 1.2:
                            print(f"⚠️ [RR_WARNING] Средний RR слишком низкий: {avg_rr:.2f}")
                
                # Мониторим сделки
                if 'total_trades' in env_info:
                    new_trades = env_info['total_trades']
                    if new_trades > self.trade_count:
                        trades_diff = new_trades - self.trade_count
                        self.trade_count = new_trades
                        
                        if trades_diff > 0 and self.num_timesteps % 100 == 0:
                            print(f"[TRADE_MONITOR] Новых сделок: {trades_diff}, Всего: {self.trade_count}")
                            
        except Exception as e:
            if self.num_timesteps % 500 == 0:
                print(f"[CALLBACK_ERROR] {e}")
        
        return True


def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        './logs/v17_optimized_v2',
        './models/v17_optimized_v2',
        './data'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Создана директория: {directory}")
        except Exception as e:
            print(f"⚠️ Ошибка создания {directory}: {e}")


def create_sample_data_with_indicators():
    """Создание тестовых данных с правильными индикаторами"""
    data_file = './data/btc_15m.csv'
    
    print("📊 Создаю тестовые данные с индикаторами...")
    
    np.random.seed(42)
    n_rows = 10000
    
    # Создаем реалистичные данные
    time = np.arange(n_rows)
    trend = np.sin(time * 0.001) * 0.5 + time * 0.00005
    noise = np.random.randn(n_rows) * 0.01
    
    close = 50000 * np.exp(trend + noise)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='15min'),
        'open': close * np.random.uniform(0.998, 1.002, n_rows),
        'high': close * np.random.uniform(1.002, 1.008, n_rows),
        'low': close * np.random.uniform(0.992, 0.998, n_rows),
        'close': close,
        'volume': np.random.lognormal(8, 1, n_rows)
    })
    
    # Добавляем ATR (делаем достаточно большим для прохождения фильтров)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean().fillna(500)
    
    # Добавляем необходимые индикаторы для фильтров
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    df['rsi_norm'] = (df['rsi'] - 50) / 50
    
    # Тренд (создаем сильный тренд для прохождения фильтров)
    df['trend_bias_1h'] = np.sin(time * 0.01) * 0.8
    
    # Волатильность
    df['returns'] = df['close'].pct_change()
    df['volatility_ratio'] = df['returns'].rolling(20).std().fillna(1.5)
    
    # Объем
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1.2)
    
    # Заполняем пропуски
    for col in df.columns:
        if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean() if not df[col].isnull().all() else 0)
    
    # Сохраняем
    df.to_csv(data_file, index=False)
    print(f"✅ Созданы тестовые данные с индикаторами: {data_file}")
    print(f"   Строк: {len(df)}, Колонок: {len(df.columns)}")
    
    return df


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    data_file = './data/btc_15m.csv'
    
    if not os.path.exists(data_file):
        return create_sample_data_with_indicators()
    
    try:
        print(f"\n📥 Загрузка данных из {data_file}...")
        df = pd.read_csv(data_file)
        print(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
        
        # Переименование колонок если нужно
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        }
        
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Проверяем обязательные колонки
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ Отсутствует колонка {col}, создаю...")
                if col == 'close':
                    df[col] = 50000
                else:
                    df[col] = df['close'] * np.random.uniform(0.99, 1.01)
        
        # Добавляем ATR если нет
        if 'atr' not in df.columns:
            print("⚠️ ATR не найден, создаю...")
            high_low = df['high'] - df['low']
            df['atr'] = high_low.rolling(window=14, min_periods=1).mean()
            df['atr'] = df['atr'].fillna(df['close'].iloc[0] * 0.02)
        
        # Добавляем необходимые индикаторы для фильтров
        # RSI
        if 'rsi_norm' not in df.columns:
            print("⚠️ RSI не найден, создаю...")
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)
            df['rsi_norm'] = (df['rsi'] - 50) / 50
        
        # ADX (Average Directional Index) - замена trend_bias_1h
        if 'adx' not in df.columns:
            print("⚠️ ADX не найден, создаю...")
            # Упрощенный расчет ADX
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0)
            plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=14).mean() / tr.rolling(window=14).mean()
            minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=14).mean() / tr.rolling(window=14).mean()
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['adx'] = dx.rolling(window=14).mean().fillna(25)
            df['plus_di'] = plus_di.fillna(25)
            df['minus_di'] = minus_di.fillna(25)
            print("   ✅ ADX, +DI, -DI созданы")
        
        # Fallback: trend_bias_1h (если нужен для совместимости, но не используется в фильтрах)
        if 'trend_bias_1h' not in df.columns:
            df['trend_bias_1h'] = np.sin(np.arange(len(df)) * 0.01) * 0.8
        
        # Волатильность
        if 'volatility_ratio' not in df.columns:
            print("⚠️ Волатильность не найдена, создаю...")
            df['returns'] = df['close'].pct_change()
            df['volatility_ratio'] = df['returns'].rolling(20).std().fillna(1.5)
        
        # Объем
        if 'volume_ratio' not in df.columns:
            print("⚠️ Объем не найден, создаю...")
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1.2)
        
        # ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (ФАЗА 1: Технические индикаторы)
        print("\n📈 Создание дополнительных признаков (Фаза 1)...")
        
        # Bollinger Bands
        if 'bb_position' not in df.columns:
            print("   Создаю Bollinger Bands...")
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['sma_20'] + (rolling_std * 2)
            df['bb_lower'] = df['sma_20'] - (rolling_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            df['bb_position'] = df['bb_position'].fillna(0.5)
        
        # Momentum
        if 'momentum' not in df.columns:
            print("   Создаю Momentum...")
            df['momentum'] = df['close'] - df['close'].shift(5)
            df['momentum'] = df['momentum'].fillna(0)
        
        # ADX (упрощенный) + DI индикаторы
        if 'adx' not in df.columns:
            print("   Создаю ADX, +DI, -DI...")
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            df['adx'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['adx'] = df['adx'].fillna(df['adx'].mean() if not df['adx'].isnull().all() else 25)
            # Сохраняем +DI и -DI для использования в наблюдениях
            df['plus_di'] = plus_di.fillna(25)
            df['minus_di'] = minus_di.fillna(25)
            print("   ✅ ADX, +DI, -DI созданы")
        
        # RSI (если еще не создан)
        if 'rsi' not in df.columns and 'rsi_norm' in df.columns:
            df['rsi'] = (df['rsi_norm'] * 50) + 50
        
        # ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (ФАЗА 2: TP-ориентированные)
        print("📈 Создание TP-ориентированных признаков (Фаза 2)...")
        
        base_atr = df['atr'].fillna(df['atr'].mean())
        current_price = df['close']
        
        # TP признаки для LONG
        if 'tp_up_atr_1' not in df.columns:
            print("   Создаю TP признаки для LONG...")
            for i, multiplier in enumerate([1.2, 1.8, 2.4], 1):
                tp_distance = base_atr * multiplier
                df[f'tp_up_atr_{i}'] = tp_distance / current_price
                df[f'tp_up_atr_{i}'] = df[f'tp_up_atr_{i}'].fillna(0.01)
                
                # Вероятность TP (упрощенная на основе RSI)
                if 'rsi' in df.columns:
                    rsi_factor = np.where(df['rsi'] < 40, 1.5, 
                                        np.where(df['rsi'] > 70, 0.7, 1.0))
                    df[f'tp_up_prob_{i}'] = 0.5 * rsi_factor
                else:
                    df[f'tp_up_prob_{i}'] = 0.5
        
        # TP признаки для SHORT
        if 'tp_down_atr_1' not in df.columns:
            print("   Создаю TP признаки для SHORT...")
            for i, multiplier in enumerate([1.2, 1.8, 2.4], 1):
                tp_distance = base_atr * multiplier
                df[f'tp_down_atr_{i}'] = -tp_distance / current_price  # Отрицательное
                df[f'tp_down_atr_{i}'] = df[f'tp_down_atr_{i}'].fillna(-0.01)
                
                # Вероятность TP для SHORT
                if 'rsi' in df.columns:
                    rsi_factor = np.where(df['rsi'] > 60, 1.5,
                                        np.where(df['rsi'] < 30, 0.7, 1.0))
                    df[f'tp_down_prob_{i}'] = 0.5 * rsi_factor
                else:
                    df[f'tp_down_prob_{i}'] = 0.5
        
        # SL признаки
        if 'sl_up_atr' not in df.columns:
            print("   Создаю SL признаки...")
            sl_multiplier = 1.5
            sl_distance_long = base_atr * sl_multiplier
            df['sl_up_atr'] = -sl_distance_long / current_price
            df['sl_up_atr'] = df['sl_up_atr'].fillna(-0.01)
            
            sl_distance_short = base_atr * sl_multiplier
            df['sl_down_atr'] = sl_distance_short / current_price
            df['sl_down_atr'] = df['sl_down_atr'].fillna(0.01)
        
        # Признаки прогресса к TP (динамика движения к целям)
        if 'progress_to_tp_up_1' not in df.columns:
            print("   Создаю признаки прогресса к TP...")
            tp_multipliers = [1.2, 1.8, 2.4]
            for i, multiplier in enumerate(tp_multipliers, 1):
                tp_distance = base_atr * multiplier
                # Прогресс к TP для LONG (положительное движение к TP)
                df[f'progress_to_tp_up_{i}'] = (df['close'] - df['close'].shift(1)) / (tp_distance + 1e-10)
                df[f'progress_to_tp_up_{i}'] = df[f'progress_to_tp_up_{i}'].fillna(0)
                
                # Прогресс к TP для SHORT (отрицательное движение к TP)
                df[f'progress_to_tp_down_{i}'] = (df['close'].shift(1) - df['close']) / (tp_distance + 1e-10)
                df[f'progress_to_tp_down_{i}'] = df[f'progress_to_tp_down_{i}'].fillna(0)
        
        # ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (ФАЗА 3: Базовые)
        print("📈 Создание базовых признаков (Фаза 3)...")
        
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['log_ret'] = df['log_ret'].fillna(0)
        
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df['returns'] = df['returns'].fillna(0)
        
        if 'high_low_ratio' not in df.columns:
            df['high_low_ratio'] = df['high'] / df['low']
            df['high_low_ratio'] = df['high_low_ratio'].fillna(1.0)
        
        if 'close_open_ratio' not in df.columns:
            df['close_open_ratio'] = df['close'] / df['open']
            df['close_open_ratio'] = df['close_open_ratio'].fillna(1.0)
        
        # Заполняем пропуски
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean() if not df[col].isnull().all() else 0)
        
        print(f"📊 Подготовлено данных: {len(df)} строк")
        print(f"📋 Пример данных (первые 5 строк):")
        print(df[['close', 'atr', 'rsi_norm', 'trend_bias_1h']].head())
        return df
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_data_with_indicators()


def load_optimized_config():
    """Загрузка оптимизированной конфигурации"""
    config_file = './models/v16_profit_focused_btc/optimized_config.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Загружена оптимизированная конфигурация из {config_file}")
        return config
    else:
        print(f"⚠️ Оптимизированная конфигурация не найдена, используются параметры по умолчанию")
        return {}


def train_optimized_model():
    """Обучение на оптимизированной среде"""
    print("\n" + "="*60)
    print("🚀 ОБУЧЕНИЕ НА ОПТИМИЗИРОВАННОЙ СРЕДЕ V17")
    print("="*60)
    
    # Создаем директории
    setup_directories()
    
    # Загружаем данные
    df = load_and_prepare_data()
    
    if df is None or len(df) < 100:
        print("❌ Недостаточно данных для обучения")
        return
    
    # ОПТИМИЗАЦИЯ ПРИЗНАКОВ НА ОСНОВЕ АНАЛИЗА КОРРЕЛЯЦИИ
    print("\n" + "="*60)
    print("🔬 ОПТИМИЗАЦИЯ ПРИЗНАКОВ ПО РЕЗУЛЬТАТАМ АНАЛИЗА")
    print("="*60)
    
    # Базовые признаки (всегда используем)
    obs_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    
    # ✅ ТОП-ПРИЗНАКИ ПО АНАЛИЗУ (добавляем в наблюдения)
    # 1) volatility_ratio, 2) volume/volume_ratio, 3) atr, 4) rsi_norm, + трендовые (adx/di)
    positive_features = [
        'volatility_ratio',
        'volume_ratio',
        'rsi_norm',
        'adx',
        'plus_di',
        'minus_di',
    ]
    print("\n📈 Признаки с ПОЛОЖИТЕЛЬНОЙ корреляцией:")
    for feat in positive_features:
        if feat in df.columns:
            obs_cols.append(feat)
            if feat == 'volatility_ratio':
                print(f"   ✅ {feat} (разница WR: 18.7%, Q1: 33.9% vs Q4: 52.6%)")
            elif feat == 'rsi_norm':
                print(f"   ✅ {feat} (корреляция: 0.126, разница WR: 29.8%)")
            elif feat == 'volume_ratio':
                print(f"   ✅ {feat} (корреляция: 0.025, разница WR: 16.2%)")
    
    # ❌ Признаки с отрицательной корреляцией (не добавляем в obs_cols)
    # trend_bias_1h используем только как фильтр (в env), но НЕ как наблюдение
    negative_features = {'trend_bias_1h': 'negative correlation - used only in filters'}
    print("\n📉 Признаки с ОТРИЦАТЕЛЬНОЙ корреляцией (НЕ добавляем в наблюдения):")
    for feat, reason in negative_features.items():
        if feat in df.columns:
            print(f"   ❌ {feat} - {reason}")
    
    # УБЕЖДАЕМСЯ, ЧТО trend_bias_1h НЕ В obs_cols (даже если был добавлен ранее)
    if 'trend_bias_1h' in obs_cols:
        obs_cols.remove('trend_bias_1h')
        print(f"   ⚠️  Удален trend_bias_1h из наблюдений (был добавлен ранее)")
    
    # 🆕 НОВЫЕ ПРИЗНАКИ ДЛЯ ТЕСТИРОВАНИЯ (добавляем постепенно)
    print("\n🆕 НОВЫЕ признаки для тестирования:")
    
    # ФАЗА 1: Технические индикаторы (проверяем их эффективность)
    phase1_features = [
        'bb_position',   # Позиция в Bollinger Bands (перекупленность/перепроданность)
        'momentum',      # Моментум цены (сила движения)
        'adx',           # Сила тренда (используется в фильтрах)
        'plus_di',       # Направление тренда вверх (используется в фильтрах для LONG)
        'minus_di',      # Направление тренда вниз (используется в фильтрах для SHORT)
        # rsi_norm добавлен в positive_features (положительная корреляция: 0.137)
    ]
    print("   ФАЗА 1: Технические индикаторы")
    for feat in phase1_features:
        if feat in df.columns:
            obs_cols.append(feat)
            print(f"   ✅ {feat}")
    
    # ФАЗА 2: TP-ориентированные признаки (критично важные для нашей стратегии!)
    phase2_features = [
        'tp_up_atr_1',      # Расстояние до TP1 для LONG
        'tp_up_prob_1',     # Вероятность TP1 для LONG
        'tp_up_atr_2',      # Расстояние до TP2 для LONG (дополнительный уровень)
        'tp_up_prob_2',     # Вероятность TP2 для LONG
        'tp_down_atr_1',    # Расстояние до TP1 для SHORT
        'tp_down_prob_1',   # Вероятность TP1 для SHORT
        'tp_down_atr_2',    # Расстояние до TP2 для SHORT (дополнительный уровень)
        'tp_down_prob_2',   # Вероятность TP2 для SHORT
        'sl_up_atr',        # Расстояние до SL для LONG
        'sl_down_atr',      # Расстояние до SL для SHORT
        'progress_to_tp_up_1',    # Прогресс к TP1 для LONG (динамика движения)
        'progress_to_tp_down_1',  # Прогресс к TP1 для SHORT (динамика движения)
    ]
    print("\n   ФАЗА 2: TP-ориентированные признаки (критично важные!)")
    for feat in phase2_features:
        if feat in df.columns:
            obs_cols.append(feat)
            print(f"   ✅ {feat}")
    
    # ФАЗА 3: Базовые дополнительные признаки
    phase3_features = [
        'log_ret',          # Логарифмическая доходность
        'returns',          # Простая доходность
        'high_low_ratio',   # Соотношение high/low
        'close_open_ratio', # Соотношение close/open
    ]
    print("\n   ФАЗА 3: Базовые дополнительные признаки")
    for feat in phase3_features:
        if feat in df.columns:
            obs_cols.append(feat)
            print(f"   ✅ {feat}")
    
    # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что trend_bias_1h точно не в obs_cols
    if 'trend_bias_1h' in obs_cols:
        obs_cols.remove('trend_bias_1h')
        print(f"\n   ⚠️  УДАЛЕН trend_bias_1h из наблюдений (отрицательная корреляция: -0.030)")
    
    # УБЕЖДАЕМСЯ, ЧТО volume_ratio В obs_cols (теперь положительная корреляция!)
    if 'volume_ratio' not in obs_cols and 'volume_ratio' in df.columns:
        obs_cols.append('volume_ratio')
        print(f"   ✅ Добавлен volume_ratio в наблюдения (положительная корреляция: 0.031, разница WR: 16.2%)")
    
    print(f"\n📊 ИТОГО: Используется {len(obs_cols)} признаков")
    print(f"   Базовые: 6")
    print(f"   С положительной корреляцией: {len([f for f in positive_features if f in obs_cols])}")
    print(f"   Новые для тестирования: {len([f for f in phase1_features + phase2_features + phase3_features if f in obs_cols])}")
    print(f"   Исключено с отрицательной корреляцией: {len(negative_features)}")
    print(f"\n💡 Примечание: Признаки с отрицательной корреляцией (trend_bias_1h)")
    print(f"   остаются для фильтров входа, но НЕ используются в наблюдениях модели.")
    print(f"   ✅ trend_bias_1h исключен из наблюдений (корреляция: -0.048)")
    print(f"   ✅ volume_ratio ДОБАВЛЕН в наблюдения (положительная корреляция: 0.031, разница WR: 16.2%)")
    print(f"   ✅ rsi_norm ДОБАВЛЕН в наблюдения (положительная корреляция: 0.127, разница WR: 30.1%)")
    
    # Разделение данных
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"   Обучение: {len(train_df):,} строк")
    print(f"   Тестирование: {len(test_df):,} строк")
    
    # Загружаем оптимизированную конфигурацию
    optimized_config = load_optimized_config()
    
    # Создаем среду с оптимизированными параметрами
    log_file = os.path.abspath('./logs/v17_optimized_v2/train_v17_log.csv')
    
    def make_train_env():
        # Базовые параметры
        env_params = {
            'df': train_df,
            'obs_cols': obs_cols,
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'log_file': log_file,
            'training_mode': 'optimized',
            # V2: больше сделок для обучения (быстрее учится балансировать LONG/SHORT)
            'max_daily_trades': 15,
            'trade_cooldown_steps': 5
        }
        
        # Добавляем оптимизированные параметры если они есть
        if optimized_config:
            # Основные параметры
            env_params.update({
                'rr_ratio': optimized_config.get('base_rr_ratio', 2.0),
                'atr_multiplier': optimized_config.get('atr_multiplier', 2.5),
            })
        
        # Добавляем параметры для интеграции сигналов стратегий
        env_params['use_strategy_signals'] = True  # Включаем сигналы стратегий
        env_params['strategy_signals_weight'] = 0.3  # Вес сигналов стратегий в reward (30%)
        
        env = CryptoTradingEnvV17_Optimized(**env_params)
        return env
    
    try:
        # Создаем среду
        train_env = DummyVecEnv([make_train_env])
        
        # УЛУЧШЕННАЯ КОНФИГУРАЦИЯ МОДЕЛИ
        # Более глубокая архитектура с лучшей способностью к обучению
        # 🔥 ИСПРАВЛЕНО: учитываем сигналы стратегий (14 признаков)
        n_strategy_signals = 14  # ZScore, SMC, ICT, Trend, Flat, ML, Momentum (по 2 сигнала каждый)
        n_features = len(obs_cols) + n_strategy_signals + 12  # market_data + strategy_signals + position_state
        
        # Отладочный вывод для проверки размеров
        print(f"\n📊 РАЗМЕРЫ OBSERVATION SPACE:")
        print(f"   obs_cols: {len(obs_cols)} признаков")
        print(f"   strategy_signals: {n_strategy_signals} признаков")
        print(f"   position_state: 12 признаков")
        print(f"   ИТОГО: {n_features} признаков")
        
        # Проверяем, что размер окружения совпадает
        env_obs_size = train_env.observation_space.shape[0]
        if env_obs_size != n_features:
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Размер окружения ({env_obs_size}) не совпадает с вычисленным ({n_features})")
        else:
            print(f"✅ Размеры совпадают: {env_obs_size}")
        
        # Увеличиваем размер скрытых слоев для лучшей способности к обучению
        hidden_size = min(512, max(256, n_features * 3))  # УВЕЛИЧЕНО с 256 до 512
        
        # УЛУЧШЕННАЯ АРХИТЕКТУРА: более глубокая сеть с residual-like структурами
        # Policy network: более глубокая для лучшего принятия решений
        # Value network: более широкая для лучшей оценки состояний
        policy_kwargs = dict(
            net_arch=[dict(
                pi=[hidden_size, hidden_size, hidden_size//2, hidden_size//4],  # 4 слоя вместо 3
                vf=[hidden_size, hidden_size//2, hidden_size//4]  # Value network остается 3 слоя
            )],
            activation_fn=nn.ReLU,  # Явно указываем активацию
            ortho_init=False  # Отключаем ортогональную инициализацию для более гибкого обучения
        )
        
        # Проверяем, есть ли существующая модель для продолжения обучения
        model_path = "./models/v17_optimized_v2/ppo_final"
        continue_training = False
        
        # Проверяем аргументы командной строки
        force_new = '--new' in sys.argv or '--fresh' in sys.argv
        
        if os.path.exists(model_path + ".zip") and not force_new:
            print(f"📂 Найдена существующая модель: {model_path}")
            response = input("Продолжить обучение с этой модели? (y/n, по умолчанию y): ").strip().lower()
            if response == '' or response == 'y':
                continue_training = True
                print("✅ Продолжаем обучение с существующей модели")
            else:
                print("🆕 Начинаем обучение с нуля")
        elif force_new:
            print("🆕 Запуск обучения с нуля (--new флаг)")
        else:
            print("🆕 Начинаем обучение с нуля (модель не найдена)")
        
        # Загружаем существующую модель или создаем новую
        if continue_training:
            print(f"📥 Загрузка модели из {model_path}...")
            try:
                # Проверяем совместимость размеров observation space
                # Получаем размер из окружения
                env_obs_size = train_env.observation_space.shape[0]
                
                # Пытаемся загрузить модель без env для проверки
                temp_model = PPO.load(model_path, env=None)
                model_obs_size = temp_model.observation_space.shape[0]
                
                if env_obs_size != model_obs_size:
                    print(f"⚠️ НЕСОВМЕСТИМОСТЬ РАЗМЕРОВ:")
                    print(f"   Окружение ожидает: {env_obs_size} признаков")
                    print(f"   Модель ожидает: {model_obs_size} признаков")
                    print(f"   🔥 Старая модель несовместима с новым окружением (добавлены сигналы стратегий)")
                    print(f"   🆕 Создаем новую модель...")
                    continue_training = False
                else:
                    # Размеры совпадают, загружаем с окружением
                    model = PPO.load(model_path, env=train_env)
                    print("✅ Модель успешно загружена!")
                    print(f"   Текущий шаг обучения: {model.num_timesteps:,}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("🆕 Создаем новую модель...")
                continue_training = False
        
        if not continue_training:
            model = PPO(
                "MlpPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=1.5e-4,  # Базовый learning rate (можно адаптировать по фазам)
                ent_coef=0.10,  # УВЕЛИЧЕНО до 0.10 для максимального exploration SHORT (по рекомендации анализа)
                n_steps=2048,  # Размер буфера для сбора опыта
                batch_size=128,  # Размер батча для обновления
                n_epochs=15,  # Количество эпох обновления на каждый буфер
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.15,
                vf_coef=0.6,
                max_grad_norm=0.5,
                tensorboard_log="./logs/v17_optimized_v2/tensorboard/"
            )
        
        # Callback для мониторинга
        rr_callback = RRMonitoringCallback()
        
        # Обучение
        print("\n🎯 ЗАПУСК ОБУЧЕНИЯ V17 (ОПТИМИЗИРОВАННОЕ)")
        print("="*40)
        
        # УВЕЛИЧЕННОЕ количество шагов для более долгого и качественного обучения
        # Увеличено до 400000 шагов для глубокого обучения и лучшей конвергенции
        total_steps = 400000  # УВЕЛИЧЕНО с 200000 до 400000 для более глубокого и качественного обучения
        
        # Расширенное поэтапное обучение с постепенным увеличением сложности
        # Каждая фаза фокусируется на разных аспектах обучения
        phases = [
            {'steps': 40000, 'name': 'phase_1_adaptation'},      # Адаптация к среде (базовые паттерны)
            {'steps': 50000, 'name': 'phase_2_exploration'},     # Исследование стратегий (разнообразие действий)
            {'steps': 60000, 'name': 'phase_3_consolidation'},   # Консолидация знаний (стабильность)
            {'steps': 70000, 'name': 'phase_4_refinement'},      # Уточнение стратегии (оптимизация)
            {'steps': 80000, 'name': 'phase_5_mastery'},         # Мастерство (финальная полировка)
            {'steps': 100000, 'name': 'phase_6_excellence'},     # Превосходство (дополнительное обучение)
        ]
        
        print(f"\n📊 План обучения: {len(phases)} фаз, всего {sum(p['steps'] for p in phases):,} шагов")
        if continue_training:
            print(f"   Начальный шаг: {model.num_timesteps:,}")
        print(f"   Конечный шаг: {model.num_timesteps + sum(p['steps'] for p in phases):,}")
        
        # Адаптивные learning rates для разных фаз (более агрессивное обучение в начале, более консервативное в конце)
        phase_learning_rates = {
            'phase_1_adaptation': 1.5e-4,      # Базовый rate для адаптации
            'phase_2_exploration': 1.2e-4,       # Немного снижаем для стабильности
            'phase_3_consolidation': 1.0e-4,     # Дальше снижаем для консолидации
            'phase_4_refinement': 8.0e-5,       # Еще ниже для уточнения
            'phase_5_mastery': 6.0e-5,          # Низкий rate для мастерства
            'phase_6_excellence': 5.0e-5,      # Минимальный rate для финальной полировки
        }
        
        for i, phase in enumerate(phases, 1):
            print(f"\n{'='*60}")
            print(f"📈 Фаза {i}/{len(phases)}: {phase['steps']:,} шагов ({phase['name']})")
            print(f"   Текущий шаг: {model.num_timesteps:,}")
            # Адаптивный learning rate для фазы
            phase_lr = phase_learning_rates.get(phase['name'], 1.5e-4)
            model.learning_rate = phase_lr
            print(f"   Learning Rate: {phase_lr:.2e}")
            print(f"{'='*60}")
            
            model.learn(
                total_timesteps=phase['steps'],
                callback=rr_callback,
                log_interval=20000,  # Логирование каждые 20000 шагов
                progress_bar=True,
                tb_log_name=phase['name'],
                reset_num_timesteps=False  # НЕ сбрасываем счетчик шагов при продолжении обучения
            )
            
            # Сохраняем промежуточную модель
            phase_model_path = f"./models/v17_optimized_v2/ppo_{phase['name']}"
            model.save(phase_model_path)
            print(f"💾 Сохранена модель фазы {i} (шаг {model.num_timesteps:,})")
            
            # Показываем прогресс
            total_completed = sum(p['steps'] for p in phases[:i])
            total_planned = sum(p['steps'] for p in phases)
            progress_pct = (total_completed / total_planned) * 100
            print(f"📊 Прогресс: {progress_pct:.1f}% ({total_completed:,} / {total_planned:,} шагов)")
            
            # Проверяем логи
            if os.path.exists(log_file):
                try:
                    log_df = pd.read_csv(log_file)
                    trades = len(log_df) - 1
                    if trades > 0:
                        print(f"📝 Сделок в логе: {trades}")
                        
                        # Анализ RR в логах
                        if 'rr_ratio' in log_df.columns:
                            # Парсим RR значения
                            def parse_rr(rr_val):
                                try:
                                    if isinstance(rr_val, str):
                                        return float(rr_val.replace('"', '').strip())
                                    return float(rr_val)
                                except:
                                    return 0.0
                            
                            rr_values = []
                            for idx in range(1, min(6, len(log_df))):  # Первые 5 сделок
                                rr_val = log_df.iloc[idx]['rr_ratio']
                                rr_values.append(parse_rr(rr_val))
                            
                            if rr_values:
                                avg_rr = np.mean(rr_values)
                                min_rr = min(rr_values)
                                print(f"📊 RR первых {len(rr_values)} сделок: Avg = {avg_rr:.2f}, Min = {min_rr:.2f}")
                                
                                if min_rr < 1.0:
                                    print(f"⚠️ Обнаружены сделки с RR < 1.0: {min_rr:.2f}")
                except Exception as e:
                    print(f"⚠️ Ошибка анализа лога: {e}")
        
        print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        
        # Сохранение финальной модели
        final_model_path = "./models/v17_optimized_v2/ppo_final"
        model.save(final_model_path)
        print(f"💾 Финальная модель сохранена: {final_model_path}")
        
        train_env.close()
        
        # Анализ результатов
        analyze_results(log_file)
        
        # Тестирование на новых данных
        test_model(model, test_df, obs_cols)
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()


def analyze_results(log_file):
    """Анализ результатов обучения"""
    print(f"\n{'='*60}")
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ V17")
    print("="*60)
    
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            
            if len(df) > 1:
                trades_df = df.iloc[1:].copy()
                
                print(f"Всего сделок: {len(trades_df)}")
                
                # Анализ PnL
                def parse_pnl(pnl_str):
                    try:
                        if isinstance(pnl_str, str):
                            return float(pnl_str.replace('%', '').strip())
                        return float(pnl_str)
                    except:
                        return 0.0
                
                trades_df['pnl_value'] = trades_df['pnl_percent'].apply(parse_pnl)
                
                profitable = (trades_df['pnl_value'] > 0).sum()
                losing = (trades_df['pnl_value'] < 0).sum()
                win_rate = profitable / len(trades_df) * 100 if len(trades_df) > 0 else 0
                avg_pnl = trades_df['pnl_value'].mean()
                total_pnl = trades_df['pnl_value'].sum()
                
                print(f"Прибыльных: {profitable} ({win_rate:.1f}%)")
                print(f"Убыточных: {losing}")
                print(f"Средний PnL: {avg_pnl:.2f}%")
                print(f"Общий PnL: {total_pnl:.2f}%")
                
                # Анализ RR
                if 'rr_ratio' in trades_df.columns:
                    # Конвертируем RR значения
                    def parse_rr(rr_str):
                        try:
                            if isinstance(rr_str, str):
                                return float(rr_str.replace('"', '').strip())
                            return float(rr_str)
                        except:
                            return 0.0
                    
                    trades_df['rr_value'] = trades_df['rr_ratio'].apply(parse_rr)
                    
                    avg_rr = trades_df['rr_value'].mean()
                    min_rr = trades_df['rr_value'].min()
                    max_rr = trades_df['rr_value'].max()
                    
                    print(f"\n📈 АНАЛИЗ RR RATIO:")
                    print(f"  Средний RR: {avg_rr:.2f}")
                    print(f"  Минимальный RR: {min_rr:.2f}")
                    print(f"  Максимальный RR: {max_rr:.2f}")
                    
                    # Сделки с плохим RR
                    bad_rr_trades = trades_df[trades_df['rr_value'] < 1.0]
                    if len(bad_rr_trades) > 0:
                        print(f"\n⚠️  Сделки с RR < 1.0: {len(bad_rr_trades)}")
                        print(f"   Их средний PnL: {bad_rr_trades['pnl_value'].mean():.2f}%")
                    
                    # Сделки с хорошим RR
                    good_rr_trades = trades_df[trades_df['rr_value'] >= 1.5]
                    if len(good_rr_trades) > 0:
                        print(f"\n✅  Сделки с RR ≥ 1.5: {len(good_rr_trades)}")
                        print(f"   Их средний PnL: {good_rr_trades['pnl_value'].mean():.2f}%")
                        win_rate_good = 100 * (good_rr_trades['pnl_value'] > 0).sum() / len(good_rr_trades)
                        print(f"   Win Rate: {win_rate_good:.1f}%")
                
                # Анализ по типам выходов
                if 'exit_reason' in trades_df.columns:
                    print(f"\n🔚 РАСПРЕДЕЛЕНИЕ ПО ПРИЧИНАМ ВЫХОДА:")
                    exit_stats = trades_df['exit_reason'].value_counts()
                    for reason, count in exit_stats.head(10).items():
                        reason_trades = trades_df[trades_df['exit_reason'] == reason]
                        avg_pnl_reason = reason_trades['pnl_value'].mean()
                        print(f"  {reason}: {count} сделок (Avg PnL: {avg_pnl_reason:.2f}%)")
                
            else:
                print("⚠️ В логе только заголовки, сделок нет")
                
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
    else:
        print(f"❌ Лог-файл не найден: {log_file}")


def test_model(model, test_df, obs_cols):
    """Тестирование модели на новых данных"""
    print(f"\n{'='*60}")
    print("🧪 ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ")
    print("="*60)
    
    test_log_file = os.path.abspath('./logs/v17_optimized_v2/test_results.csv')
    
    def make_test_env():
        # Используем больше тестовых данных для статистически значимых результатов
        env = CryptoTradingEnvV17_Optimized(
            df=test_df.copy(),  # Используем все тестовые данные вместо [:1000]
            obs_cols=obs_cols,
            initial_balance=10000,
            commission=0.001,
            slippage=0.0005,
            log_file=test_log_file,
            use_strategy_signals=True,  # Включаем сигналы стратегий для тестирования
            strategy_signals_weight=0.3,  # Вес сигналов стратегий в reward (30%)
            training_mode='optimized'
        )
        return env
    
    test_env = DummyVecEnv([make_test_env])
    obs = test_env.reset()
    
    steps = 0
    max_steps = min(len(test_df), 2000)  # Увеличено с 300 до 2000 или до размера тестовых данных
    print(f"Тестирование на {max_steps} шагах (из {len(test_df)} доступных)...")
    
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        steps += 1
        
        if done[0]:
            print(f"Среда завершена на шаге {steps}")
            break
        
        if steps % 50 == 0:
            if isinstance(info, list) and len(info) > 0:
                net_worth = info[0].get('net_worth', 0) if isinstance(info[0], dict) else 0
            else:
                net_worth = 0
            print(f"  [Шаг {steps}] Reward: {reward[0]:.3f}, Net Worth: ${net_worth:.2f}")
    
    test_env.close()
    
    # Анализ тестовых результатов
    if os.path.exists(test_log_file):
        try:
            test_df_log = pd.read_csv(test_log_file)
            if len(test_df_log) > 1:
                print(f"\n📊 ТЕСТОВЫЕ РЕЗУЛЬТАТЫ: {len(test_df_log) - 1} сделок")
                
                # Анализ PnL
                def parse_pnl(pnl_str):
                    try:
                        if isinstance(pnl_str, str):
                            return float(pnl_str.replace('%', '').strip())
                        return float(pnl_str)
                    except:
                        return 0.0
                
                test_trades = test_df_log.iloc[1:].copy()
                test_trades['pnl_value'] = test_trades['pnl_percent'].apply(parse_pnl)
                
                profitable = (test_trades['pnl_value'] > 0).sum()
                total = len(test_trades)
                win_rate = profitable / total * 100 if total > 0 else 0
                avg_pnl = test_trades['pnl_value'].mean()
                
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Средний PnL: {avg_pnl:.2f}%")
                
        except Exception as e:
            print(f"⚠️ Ошибка анализа тестовых результатов: {e}")


def main():
    print("🐍 Запуск оптимизированного обучения V17...")
    print(f"📁 Текущая директория: {os.getcwd()}")
    
    # Показываем справку по аргументам
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nИспользование:")
        print("  python train_v17_optimized.py          # Интерактивный режим")
        print("  python train_v17_optimized.py --new     # Запуск с нуля (без запроса)")
        print("  python train_v17_optimized.py --fresh  # То же что --new")
        return
    
    train_optimized_model()
    
    print(f"\n{'='*60}")
    print("🎉 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("📁 Результаты сохранены в:")
    print("   - Модели: ./models/v17_optimized_v2/")
    print("   - Логи: ./logs/v17_optimized_v2/")
    print("   - Tensorboard логи: ./logs/v17_optimized_v2/tensorboard/")


if __name__ == "__main__":
    main()