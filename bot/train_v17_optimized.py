import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from crypto_env_v17_optimized import CryptoTradingEnvV17_Optimized


class RRMonitoringCallback(BaseCallback):
    """Callback для мониторинга RR ratio"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rr_history = []
        self.trade_count = 0
        
    def _on_step(self) -> bool:
        # Получаем информацию из среды
        try:
            if hasattr(self.locals, 'env'):
                env_info = self.locals['env'].get_attr('_get_info')[0]
                
                # Мониторим RR статистику
                if 'rr_stats' in env_info:
                    rr_stats = env_info['rr_stats']
                    
                    # Сохраняем историю
                    self.rr_history.append(rr_stats['avg'])
                    if len(self.rr_history) > 100:
                        self.rr_history.pop(0)
                    
                    # Логируем каждые 100 шагов
                    if self.num_timesteps % 100 == 0:
                        avg_rr = np.mean(self.rr_history) if self.rr_history else 0
                        print(f"[RR_MONITOR] Step {self.num_timesteps}: Avg RR = {avg_rr:.2f}, "
                              f"Violations = {rr_stats['violations']}")
                        
                        # Предупреждение если RR низкий
                        if avg_rr < 1.2:
                            print(f"⚠️ [RR_WARNING] Средний RR слишком низкий: {avg_rr:.2f}")
                
                # Мониторим сделки
                if 'total_trades' in env_info:
                    new_trades = env_info['total_trades']
                    if new_trades > self.trade_count:
                        trades_diff = new_trades - self.trade_count
                        self.trade_count = new_trades
                        
                        if trades_diff > 0 and self.num_timesteps % 50 == 0:
                            print(f"[TRADE_MONITOR] Новых сделок: {trades_diff}, Всего: {self.trade_count}")
                            
        except Exception as e:
            if self.num_timesteps % 500 == 0:
                print(f"[CALLBACK_ERROR] {e}")
        
        return True


def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        './logs/v17_optimized',
        './models/v17_optimized',
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
        
        # Тренд
        if 'trend_bias_1h' not in df.columns:
            print("⚠️ Тренд не найден, создаю...")
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
    
    # Подготовка признаков
    obs_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    additional_cols = ['rsi_norm', 'trend_bias_1h', 'volatility_ratio', 'volume_ratio']
    for col in additional_cols:
        if col in df.columns:
            obs_cols.append(col)
    
    print(f"📊 Используется {len(obs_cols)} признаков")
    
    # Разделение данных
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"   Обучение: {len(train_df):,} строк")
    print(f"   Тестирование: {len(test_df):,} строк")
    
    # Загружаем оптимизированную конфигурацию
    optimized_config = load_optimized_config()
    
    # Создаем среду с оптимизированными параметрами
    log_file = os.path.abspath('./logs/v17_optimized/train_v17_log.csv')
    
    def make_train_env():
        # Базовые параметры
        env_params = {
            'df': train_df,
            'obs_cols': obs_cols,
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'log_file': log_file,
            'training_mode': 'optimized'
        }
        
        # Добавляем оптимизированные параметры если они есть
        if optimized_config:
            # Основные параметры
            env_params.update({
                'rr_ratio': optimized_config.get('base_rr_ratio', 2.0),
                'atr_multiplier': optimized_config.get('atr_multiplier', 2.5),
            })
        
        env = CryptoTradingEnvV17_Optimized(**env_params)
        return env
    
    try:
        # Создаем среду
        train_env = DummyVecEnv([make_train_env])
        
        # Конфигурация модели
        n_features = len(obs_cols) + 12
        hidden_size = min(256, max(128, n_features * 2))
        
        policy_kwargs = dict(
            net_arch=[dict(
                pi=[hidden_size, hidden_size//2, hidden_size//4],
                vf=[hidden_size, hidden_size//2, hidden_size//4]
            )]
        )
        
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1.5e-4,
            ent_coef=0.015,
            n_steps=2048,
            batch_size=128,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            vf_coef=0.6,
            max_grad_norm=0.7,
            tensorboard_log="./logs/v17_optimized/tensorboard/"
        )
        
        # Callback для мониторинга
        rr_callback = RRMonitoringCallback()
        
        # Обучение
        print("\n🎯 ЗАПУСК ОБУЧЕНИЯ V17 (ОПТИМИЗИРОВАННОЕ)")
        print("="*40)
        
        total_steps = 20000
        
        # Поэтапное обучение
        phases = [
            {'steps': 5000, 'name': 'phase_1_adaptation'},
            {'steps': 5000, 'name': 'phase_2_consolidation'},
            {'steps': 10000, 'name': 'phase_3_refinement'},
        ]
        
        for i, phase in enumerate(phases, 1):
            print(f"\n📈 Фаза {i}/{len(phases)}: {phase['steps']:,} шагов ({phase['name']})")
            
            model.learn(
                total_timesteps=phase['steps'],
                callback=rr_callback,
                log_interval=10,
                progress_bar=True,
                tb_log_name=phase['name']
            )
            
            # Сохраняем промежуточную модель
            phase_model_path = f"./models/v17_optimized/ppo_{phase['name']}"
            model.save(phase_model_path)
            print(f"💾 Сохранена модель фазы {i}")
            
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
        final_model_path = "./models/v17_optimized/ppo_final"
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
    
    test_log_file = os.path.abspath('./logs/v17_optimized/test_results.csv')
    
    def make_test_env():
        env = CryptoTradingEnvV17_Optimized(
            df=test_df.iloc[:1000].copy(),
            obs_cols=obs_cols,
            initial_balance=10000,
            commission=0.001,
            slippage=0.0005,
            log_file=test_log_file,
            training_mode='optimized'
        )
        return env
    
    test_env = DummyVecEnv([make_test_env])
    obs = test_env.reset()
    
    steps = 0
    max_steps = 300
    print(f"Тестирование на {max_steps} шагах...")
    
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
    
    train_optimized_model()
    
    print(f"\n{'='*60}")
    print("🎉 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("📁 Результаты сохранены в:")
    print("   - Модели: ./models/v17_optimized/")
    print("   - Логи: ./logs/v17_optimized/")
    print("   - Tensorboard логи: ./logs/v17_optimized/tensorboard/")


if __name__ == "__main__":
    main()