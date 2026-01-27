"""
Скрипт обучения модели с мультитаймфреймовым анализом (V18 MTF)
Основан на train_v17_2_optimized.py с добавлением поддержки MTF данных
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
# Добавляем путь к корню проекта для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.crypto_env_v18_mtf import CryptoTradingEnvV18_MTF
from bot.data_preprocessor_mtf import create_mtf_dataset

# Импортируем callback из родительского модуля
try:
    from bot.train_v17_2_optimized import RRMonitoringCallback
except ImportError:
    # Fallback: определяем callback здесь если импорт не работает
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np
    
    class RRMonitoringCallback(BaseCallback):
        """Callback для мониторинга RR ratio и разнообразия действий"""
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.rr_history = []
            self.trade_count = 0
            self.action_history = []
            self._last_open_step = None
            
        def _on_step(self) -> bool:
            try:
                infos = self.locals.get('infos', None)
                if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                    env_info = infos[0]
                    if 'rr_stats' in env_info:
                        rr_stats = env_info['rr_stats']
                        self.rr_history.append(rr_stats['avg'])
                        if len(self.rr_history) > 100:
                            self.rr_history.pop(0)
                        
                        if self.num_timesteps % 500 == 0:
                            avg_rr = np.mean(self.rr_history) if self.rr_history else 0
                            print(f"[MONITOR] Step {self.num_timesteps}: Avg RR = {avg_rr:.2f}")
            except:
                pass
            return True


def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        './logs/v18_mtf',
        './models/v18_mtf',
        './data/mtf'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Создана директория: {directory}")
        except Exception as e:
            print(f"⚠️ Ошибка создания {directory}: {e}")


def load_and_prepare_mtf_data(symbol: str = "BTCUSDT"):
    """
    Загружает и подготавливает MTF данные
    
    Args:
        symbol: Торговая пара
    
    Returns:
        Список датафреймов [df_15m, df_1h, df_4h]
    """
    print(f"\n{'='*60}")
    print(f"📊 ПОДГОТОВКА MTF ДАННЫХ ДЛЯ {symbol}")
    print(f"{'='*60}\n")
    
    base_path = './data'
    
    # Создаем MTF датасет
    df_list = create_mtf_dataset(base_path, symbol, output_path='./data/mtf')
    
    if len(df_list) == 0 or len(df_list[0]) == 0:
        print("❌ Не удалось загрузить данные для основного таймфрейма (15m)")
        return None
    
    df_15m = df_list[0]
    
    # Проверяем наличие данных для старших ТФ
    if len(df_list) > 1 and df_list[1] is not None and len(df_list[1]) > 0:
        print(f"✅ Данные 1h: {len(df_list[1])} строк")
    else:
        print("⚠️ Данные 1h отсутствуют, MTF будет работать в ограниченном режиме")
    
    if len(df_list) > 2 and df_list[2] is not None and len(df_list[2]) > 0:
        print(f"✅ Данные 4h: {len(df_list[2])} строк")
    else:
        print("⚠️ Данные 4h отсутствуют, MTF будет работать в ограниченном режиме")
    
    return df_list


def get_mtf_observation_columns(df_15m: pd.DataFrame) -> list:
    """
    Определяет список признаков для наблюдения с учетом MTF
    
    Args:
        df_15m: Основной датафрейм (15m)
    
    Returns:
        Список колонок для наблюдения
    """
    # Базовые признаки (как в V17_2)
    obs_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    
    # Признаки с положительной корреляцией
    positive_features = [
        'volatility_ratio',
        'rsi_norm',
        'volume_ratio',
    ]
    
    for feat in positive_features:
        if feat in df_15m.columns:
            obs_cols.append(feat)
    
    # Технические индикаторы
    phase1_features = [
        'bb_position',
        'momentum',
        'adx',
        'plus_di',
        'minus_di',
    ]
    
    for feat in phase1_features:
        if feat in df_15m.columns:
            obs_cols.append(feat)
    
    # TP-ориентированные признаки
    phase2_features = [
        'tp_up_atr_1', 'tp_up_prob_1', 'tp_up_atr_2', 'tp_up_prob_2',
        'tp_down_atr_1', 'tp_down_prob_1', 'tp_down_atr_2', 'tp_down_prob_2',
        'sl_up_atr', 'sl_down_atr',
        'progress_to_tp_up_1', 'progress_to_tp_down_1',
    ]
    
    for feat in phase2_features:
        if feat in df_15m.columns:
            obs_cols.append(feat)
    
    # Базовые дополнительные признаки
    phase3_features = [
        'log_ret', 'returns', 'high_low_ratio', 'close_open_ratio',
    ]
    
    for feat in phase3_features:
        if feat in df_15m.columns:
            obs_cols.append(feat)
    
    # Убираем trend_bias_1h если есть (отрицательная корреляция)
    if 'trend_bias_1h' in obs_cols:
        obs_cols.remove('trend_bias_1h')
    
    print(f"\n📊 Используется {len(obs_cols)} признаков для наблюдения")
    print(f"   MTF признаки будут добавлены автоматически средой")
    
    return obs_cols


def train_mtf_model():
    """Обучение модели с MTF анализом"""
    print("\n" + "="*60)
    print("🚀 ОБУЧЕНИЕ С МУЛЬТИТАЙМФРЕЙМОВЫМ АНАЛИЗОМ V18 MTF")
    print("="*60)
    
    # Создаем директории
    setup_directories()
    
    # Загружаем MTF данные
    df_list = load_and_prepare_mtf_data("BTCUSDT")
    
    if df_list is None or len(df_list) == 0 or len(df_list[0]) < 100:
        print("❌ Недостаточно данных для обучения")
        return
    
    df_15m = df_list[0]
    
    # Определяем признаки для наблюдения
    obs_cols = get_mtf_observation_columns(df_15m)
    
    # Разделение данных
    train_size = int(len(df_15m) * 0.7)
    
    # Разделяем все таймфреймы
    train_df_list = []
    test_df_list = []
    
    for df in df_list:
        if df is not None and len(df) > 0:
            # Для старших ТФ находим соответствующие индексы
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
            else:
                train_df_list.append(df.iloc[:train_size].copy())
                test_df_list.append(df.iloc[train_size:].copy())
                continue
            
            # Находим время разделения
            split_time = pd.to_datetime(df_15m.iloc[train_size]['timestamp'] if 'timestamp' in df_15m.columns else df_15m.index[train_size])
            
            # Разделяем по времени
            df[time_col] = pd.to_datetime(df[time_col])
            train_mask = df[time_col] <= split_time
            test_mask = df[time_col] > split_time
            
            train_df_list.append(df[train_mask].copy())
            test_df_list.append(df[test_mask].copy())
        else:
            train_df_list.append(pd.DataFrame())
            test_df_list.append(pd.DataFrame())
    
    print(f"\n📊 Разделение данных:")
    print(f"   Обучение: {len(train_df_list[0]):,} строк (15m)")
    print(f"   Тестирование: {len(test_df_list[0]):,} строк (15m)")
    
    # Создаем среду обучения
    log_file = os.path.abspath('./logs/v18_mtf/train_v18_mtf_log.csv')
    
    def make_train_env():
        env_params = {
            'df_list': train_df_list,
            'obs_cols': obs_cols,
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'log_file': log_file,
            'log_open_positions': True,
            'open_log_file': os.path.abspath('./logs/v18_mtf/opens_train_v18_mtf_log.csv'),
            'training_mode': 'mtf'
        }
        
        env = CryptoTradingEnvV18_MTF(**env_params)
        return env
    
    try:
        # Создаем среду
        train_env = DummyVecEnv([make_train_env])
        
        # Конфигурация модели
        # MTF добавляет дополнительные признаки, поэтому размер увеличивается
        n_features = len(obs_cols) + 12 + 20  # Базовые + состояние + MTF признаки
        hidden_size = min(256, max(128, n_features * 2))
        
        policy_kwargs = dict(
            net_arch=[dict(
                pi=[hidden_size, hidden_size//2, hidden_size//4],
                vf=[hidden_size, hidden_size//2, hidden_size//4]
            )]
        )
        
        # Проверяем существующую модель
        model_path = "./models/v18_mtf/ppo_final"
        continue_training = False
        
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
        
        # Загружаем или создаем модель
        if continue_training:
            print(f"📥 Загрузка модели из {model_path}...")
            try:
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
                learning_rate=1.5e-4,
                ent_coef=0.05,
                n_steps=2048,
                batch_size=128,
                n_epochs=15,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.15,
                vf_coef=0.6,
                max_grad_norm=0.5,
                tensorboard_log="./logs/v18_mtf/tensorboard/"
            )
        
        # Callback для мониторинга
        rr_callback = RRMonitoringCallback()
        
        # Обучение
        print("\n🎯 ЗАПУСК ОБУЧЕНИЯ V18 MTF")
        print("="*40)
        
        total_steps = 400000
        
        # Фазы обучения (аналогично V17_2)
        phases = [
            {'steps': 40000, 'name': 'phase_1_adaptation'},
            {'steps': 50000, 'name': 'phase_2_exploration'},
            {'steps': 60000, 'name': 'phase_3_consolidation'},
            {'steps': 70000, 'name': 'phase_4_refinement'},
            {'steps': 80000, 'name': 'phase_5_mastery'},
            {'steps': 100000, 'name': 'phase_6_excellence'},
        ]
        
        print(f"\n📊 План обучения: {len(phases)} фаз, всего {sum(p['steps'] for p in phases):,} шагов")
        if continue_training:
            print(f"   Начальный шаг: {model.num_timesteps:,}")
        print(f"   Конечный шаг: {model.num_timesteps + sum(p['steps'] for p in phases):,}")
        
        # Адаптивные learning rates
        phase_learning_rates = {
            'phase_1_adaptation': 1.5e-4,
            'phase_2_exploration': 1.2e-4,
            'phase_3_consolidation': 1.0e-4,
            'phase_4_refinement': 8.0e-5,
            'phase_5_mastery': 6.0e-5,
            'phase_6_excellence': 5.0e-5,
        }
        
        for i, phase in enumerate(phases, 1):
            print(f"\n{'='*60}")
            print(f"📈 Фаза {i}/{len(phases)}: {phase['steps']:,} шагов ({phase['name']})")
            print(f"   Текущий шаг: {model.num_timesteps:,}")
            phase_lr = phase_learning_rates.get(phase['name'], 1.5e-4)
            model.learning_rate = phase_lr
            print(f"   Learning Rate: {phase_lr:.2e}")
            print(f"{'='*60}")
            
            model.learn(
                total_timesteps=phase['steps'],
                callback=rr_callback,
                log_interval=20000,
                progress_bar=True,
                tb_log_name=phase['name'],
                reset_num_timesteps=False
            )
            
            # Сохраняем промежуточную модель
            phase_model_path = f"./models/v18_mtf/ppo_{phase['name']}"
            model.save(phase_model_path)
            print(f"💾 Сохранена модель фазы {i} (шаг {model.num_timesteps:,})")
            
            # Прогресс
            total_completed = sum(p['steps'] for p in phases[:i])
            total_planned = sum(p['steps'] for p in phases)
            progress_pct = (total_completed / total_planned) * 100
            print(f"📊 Прогресс: {progress_pct:.1f}% ({total_completed:,} / {total_planned:,} шагов)")
        
        print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        
        # Сохранение финальной модели
        final_model_path = "./models/v18_mtf/ppo_final"
        model.save(final_model_path)
        print(f"💾 Финальная модель сохранена: {final_model_path}")
        
        train_env.close()
        
        # Анализ результатов
        analyze_results(log_file)
        
        # Тестирование на новых данных
        test_model(model, test_df_list, obs_cols)
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()


def analyze_results(log_file):
    """Анализ результатов обучения"""
    print(f"\n{'='*60}")
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ V18 MTF")
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
                
                # Анализ по типам позиций
                if 'type' in trades_df.columns:
                    long_trades = trades_df[trades_df['type'].str.contains('LONG', na=False)]
                    short_trades = trades_df[trades_df['type'].str.contains('SHORT', na=False)]
                    
                    print(f"\n📊 Распределение по типам:")
                    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
                    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
    else:
        print(f"❌ Лог-файл не найден: {log_file}")


def test_model(model, test_df_list, obs_cols):
    """Тестирование модели на новых данных"""
    print(f"\n{'='*60}")
    print("🧪 ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ")
    print("="*60)
    
    test_log_file = os.path.abspath('./logs/v18_mtf/test_results.csv')
    
    def make_test_env():
        env = CryptoTradingEnvV18_MTF(
            df_list=test_df_list,
            obs_cols=obs_cols,
            initial_balance=10000,
            commission=0.001,
            slippage=0.0005,
            log_file=test_log_file,
            log_open_positions=True,
            open_log_file=os.path.abspath('./logs/v18_mtf/opens_test_results.csv'),
            training_mode='mtf'
        )
        return env
    
    test_env = DummyVecEnv([make_test_env])
    obs = test_env.reset()
    
    steps = 0
    max_steps = min(len(test_df_list[0]), 2000) if len(test_df_list) > 0 and len(test_df_list[0]) > 0 else 2000
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
    print("🐍 Запуск MTF обучения V18...")
    print(f"📁 Текущая директория: {os.getcwd()}")
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nИспользование:")
        print("  python train_v18_mtf.py          # Интерактивный режим")
        print("  python train_v18_mtf.py --new    # Запуск с нуля (без запроса)")
        print("  python train_v18_mtf.py --fresh   # То же что --new")
        return
    
    train_mtf_model()
    
    print(f"\n{'='*60}")
    print("🎉 MTF ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("📁 Результаты сохранены в:")
    print("   - Модели: ./models/v18_mtf/")
    print("   - Логи: ./logs/v18_mtf/")
    print("   - Tensorboard логи: ./logs/v18_mtf/tensorboard/")


if __name__ == "__main__":
    main()
