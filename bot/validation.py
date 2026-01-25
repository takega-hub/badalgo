import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from crypto_env import CryptoTradingEnv, make_env
from data_processor import DataProcessor
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def load_test_data():
    """Загрузка и подготовка тестовых данных"""
    print("📊 Загрузка тестовых данных...")
    
    data_path = "data/btc_15m.csv"
    processor = DataProcessor(data_path)
    processor.load_data()
    
    # Подготавливаем фичи
    df = processor.prepare_features()
    
    # Разделяем на train/test (90/10)
    train_df, test_df = processor.split_data(test_size=0.1)
    
    # Используем те же ключевые признаки, что и при обучении
    key_features = [
        'log_ret',           # Доходность
        'rsi_norm',          # RSI (нормализованный)
        'atr_norm',          # Волатильность (нормализованная)
        'trend_bias_1h',     # Тренд на старшем ТФ
        'volatility_ratio',  # Отношение волатильности
        'bb_width',          # Ширина Боллинджера
        'dist_to_local_high' # Расстояние до локального максимума
    ]
    
    # Фильтруем только существующие признаки
    obs_cols = [col for col in key_features if col in test_df.columns]
    
    print(f"✅ Загружено {len(test_df)} тестовых строк")
    print(f"📈 Используется {len(obs_cols)} признаков")
    
    return test_df, obs_cols

def run_backtest(model_path, test_df, obs_cols, num_episodes=1):
    """
    Запуск бэктеста модели на тестовых данных
    """
    print(f"\n🔍 Запуск бэктеста модели: {model_path}")
    
    # Создаем тестовую среду
    env = CryptoTradingEnv(
        df=test_df,
        obs_cols=obs_cols,
        initial_balance=1000,
        commission=0.001,
        slippage=0.0005,
        rr_ratio=3.5,
        atr_multiplier=3.5,
        log_file="v15_test_log.csv",
        training_mode="conservative"
    )
    
    # Загружаем модель
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"✅ Модель загружена: {model_path}")
    else:
        print(f"❌ Модель не найдена: {model_path}")
        return None
    
    # Сбрасываем логи
    env._init_log_file()
    
    results = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        print(f"\n📈 Эпизод {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action[0])
            episode_reward += reward
            
            # Сохраняем информацию о шаге
            step_info = info.copy()
            step_info['step'] = step_count
            step_info['reward'] = reward
            step_info['action'] = action[0]
            step_info['position'] = env.position
            step_info['net_worth'] = env.net_worth
            
            results.append(step_info)
            step_count += 1
            
            if step_count % 1000 == 0:
                print(f"   Шаг {step_count}, Net Worth: ${env.net_worth:.2f}, Reward: {episode_reward:.2f}")
        
        # Сохраняем метрики эпизода
        episode_metrics.append({
            'episode': episode,
            'total_steps': step_count,
            'total_reward': episode_reward,
            'final_net_worth': env.net_worth,
            'max_net_worth': env.max_net_worth,
            'total_trades': env.total_trades,
            'winning_trades': env.winning_trades,
            'losing_trades': env.losing_trades,
        })
        
        print(f"   Завершено: {step_count} шагов, Net Worth: ${env.net_worth:.2f}")
        print(f"   Сделок: {env.total_trades}, Win Rate: {(env.winning_trades/max(1, env.total_trades)*100):.1f}%")
    
    # Получаем финальные метрики
    final_metrics = env.get_performance_metrics()
    
    return env, results, episode_metrics, final_metrics

def analyze_trades(env):
    """
    Детальный анализ сделок по типам выхода
    """
    print("\n" + "="*70)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК")
    print("="*70)
    
    if not env.trade_history:
        print("❌ История сделок пуста")
        return None
    
    # Создаем DataFrame из истории сделок
    trades_df = pd.DataFrame(env.trade_history)
    
    # Анализ по типам выхода
    exit_types = trades_df['exit_type'].unique()
    
    print(f"\n=== АНАЛИЗ {len(trades_df)} СДЕЛОК ===")
    
    # Основные метрики
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_profit = winning_trades['pnl'].mean() * 100 if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) * 100 if len(losing_trades) > 0 else 0
    
    # Profit Factor
    total_profit = winning_trades['pnl'].sum() * 100 if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) * 100 if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Средняя прибыль: {avg_profit:.2f}%")
    print(f"Средний убыток: {avg_loss:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Анализ по типам выхода
    print(f"\n📈 ТИПЫ ВЫХОДА:")
    
    exit_stats = []
    for exit_type in exit_types:
        type_trades = trades_df[trades_df['exit_type'] == exit_type]
        type_pnl_mean = type_trades['pnl'].mean() * 100
        
        exit_stats.append({
            'exit_type': exit_type,
            'count': len(type_trades),
            'percentage': len(type_trades) / len(trades_df) * 100,
            'avg_pnl_percent': type_pnl_mean,
            'total_pnl': type_trades['pnl'].sum() * 100
        })
    
    # Сортируем по количеству сделок
    exit_stats.sort(key=lambda x: x['count'], reverse=True)
    
    for stat in exit_stats:
        print(f"  {stat['exit_type']}: {stat['count']} сделок ({stat['percentage']:.1f}%), средний PnL: {stat['avg_pnl_percent']:.2f}%")
    
    # Анализ длительности сделок
    print(f"\n⏱️  ДЛИТЕЛЬНОСТЬ СДЕЛОК:")
    print(f"  Средняя: {trades_df['duration'].mean():.1f} шагов")
    print(f"  Медиана: {trades_df['duration'].median():.1f} шагов")
    print(f"  Минимум: {trades_df['duration'].min()} шагов")
    print(f"  Максимум: {trades_df['duration'].max()} шагов")
    
    # Анализ по типам позиций
    print(f"\n🎯 ТИПЫ ПОЗИЦИЙ:")
    long_trades = trades_df[trades_df['type'] == 'LONG']
    short_trades = trades_df[trades_df['type'] == 'SHORT']
    
    if len(long_trades) > 0:
        long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100
        print(f"  LONG: {len(long_trades)} сделок, Win Rate: {long_win_rate:.1f}%")
    
    if len(short_trades) > 0:
        short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100
        print(f"  SHORT: {len(short_trades)} сделок, Win Rate: {short_win_rate:.1f}%")
    
    # Распределение PnL
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ PnL:")
    print(f"  Лучшая сделка: {trades_df['pnl'].max() * 100:.2f}%")
    print(f"  Худшая сделка: {trades_df['pnl'].min() * 100:.2f}%")
    print(f"  Стандартное отклонение: {trades_df['pnl'].std() * 100:.2f}%")
    
    # Максимальная серия
    print(f"\n📈 СЕРИИ:")
    print(f"  Максимальная прибыльная серия: {_max_consecutive_wins(trades_df)}")
    print(f"  Максимальная убыточная серия: {_max_consecutive_losses(trades_df)}")
    
    return {
        'trades_df': trades_df,
        'exit_stats': exit_stats,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df)
    }

def _max_consecutive_wins(trades_df):
    """Найти максимальную серию прибыльных сделок"""
    max_streak = 0
    current_streak = 0
    
    for pnl in trades_df['pnl']:
        if pnl > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def _max_consecutive_losses(trades_df):
    """Найти максимальную серию убыточных сделок"""
    max_streak = 0
    current_streak = 0
    
    for pnl in trades_df['pnl']:
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def plot_detailed_analysis(env, trades_analysis):
    """
    Построение детальных графиков анализа
    """
    print("\n📈 Построение графиков анализа...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Детальный анализ стратегии v15', fontsize=16, fontweight='bold')
    
    # 1. Динамика капитала
    axes[0, 0].plot(env.net_worth_history, 'b-', linewidth=1.5, alpha=0.8)
    axes[0, 0].axhline(y=env.initial_balance, color='r', linestyle='--', alpha=0.5, label='Начальный баланс')
    axes[0, 0].set_title('Динамика капитала')
    axes[0, 0].set_xlabel('Шаг')
    axes[0, 0].set_ylabel('Капитал ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Кумулятивный PnL
    if trades_analysis and 'trades_df' in trades_analysis:
        trades_df = trades_analysis['trades_df']
        cumulative_pnl = np.cumsum(trades_df['pnl'] * 100)
        axes[0, 1].plot(cumulative_pnl, 'g-', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Кумулятивный PnL (%)')
        axes[0, 1].set_xlabel('Сделка')
        axes[0, 1].set_ylabel('PnL (%)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Распределение PnL
    if trades_analysis and 'trades_df' in trades_analysis:
        pnls = trades_df['pnl'] * 100
        axes[0, 2].hist(pnls, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
        axes[0, 2].set_title('Распределение PnL сделок')
        axes[0, 2].set_xlabel('PnL (%)')
        axes[0, 2].set_ylabel('Количество')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Распределение по типам выхода
    if trades_analysis and 'exit_stats' in trades_analysis:
        exit_stats = trades_analysis['exit_stats']
        exit_types = [stat['exit_type'] for stat in exit_stats]
        exit_counts = [stat['count'] for stat in exit_stats]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(exit_types)))
        axes[1, 0].bar(exit_types, exit_counts, color=colors, edgecolor='black')
        axes[1, 0].set_title('Количество сделок по типам выхода')
        axes[1, 0].set_xlabel('Тип выхода')
        axes[1, 0].set_ylabel('Количество сделок')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Средний PnL по типам выхода
    if trades_analysis and 'exit_stats' in trades_analysis:
        exit_types = [stat['exit_type'] for stat in exit_stats]
        avg_pnls = [stat['avg_pnl_percent'] for stat in exit_stats]
        
        colors = ['green' if pnl > 0 else 'red' for pnl in avg_pnls]
        axes[1, 1].bar(exit_types, avg_pnls, color=colors, edgecolor='black', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Средний PnL по типам выхода (%)')
        axes[1, 1].set_xlabel('Тип выхода')
        axes[1, 1].set_ylabel('Средний PnL (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Распределение длительности сделок
    if trades_analysis and 'trades_df' in trades_analysis:
        durations = trades_df['duration']
        axes[1, 2].hist(durations, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 2].set_title('Распределение длительности сделок')
        axes[1, 2].set_xlabel('Длительность (шаги)')
        axes[1, 2].set_ylabel('Количество')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Длительность vs PnL
    if trades_analysis and 'trades_df' in trades_analysis:
        scatter = axes[2, 0].scatter(trades_df['duration'], trades_df['pnl'] * 100, 
                                     c=trades_df['pnl'] * 100, cmap='RdYlGn', 
                                     alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2, 0].set_title('Длительность vs PnL')
        axes[2, 0].set_xlabel('Длительность (шаги)')
        axes[2, 0].set_ylabel('PnL (%)')
        axes[2, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2, 0])
    
    # 8. Просадка (Drawdown)
    if len(env.net_worth_history) > 1:
        net_worth_array = np.array(env.net_worth_history)
        running_max = np.maximum.accumulate(net_worth_array)
        drawdown = (running_max - net_worth_array) / running_max * 100
        
        axes[2, 1].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        axes[2, 1].plot(drawdown, color='darkred', linewidth=1)
        axes[2, 1].set_title('Просадка капитала')
        axes[2, 1].set_xlabel('Шаг')
        axes[2, 1].set_ylabel('Просадка (%)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim(0, max(drawdown) * 1.1 if max(drawdown) > 0 else 10)
    
    # 9. История действий (Action Distribution)
    if hasattr(env, 'actions_history') and env.actions_history:
        actions = [step['action'] for step in env.actions_history[:1000]]  # Берем первые 1000 шагов для наглядности
        action_names = ['HOLD', 'LONG', 'SHORT']
        action_counts = [actions.count(i) for i in range(3)]
        
        colors = ['gray', 'green', 'red']
        axes[2, 2].bar(action_names, action_counts, color=colors, edgecolor='black', alpha=0.7)
        axes[2, 2].set_title('Распределение действий')
        axes[2, 2].set_xlabel('Действие')
        axes[2, 2].set_ylabel('Количество')
        axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('v15_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Графики сохранены в 'v15_detailed_analysis.png'")
    plt.show()

def generate_report(env, trades_analysis, final_metrics):
    """
    Генерация детального отчета в формате, аналогичном предоставленным данным
    """
    print("\n" + "="*70)
    print("📋 ДЕТАЛЬНЫЙ ОТЧЕТ О РЕЗУЛЬТАТАХ")
    print("="*70)
    
    if not trades_analysis:
        print("❌ Нет данных для отчета")
        return
    
    # Базовые метрики
    print(f"\n=== АНАЛИЗ {trades_analysis['total_trades']} СДЕЛОК ===")
    print(f"Win Rate: {trades_analysis['win_rate']:.1f}%")
    print(f"Средняя прибыль: {trades_analysis['avg_profit']:.2f}%")
    print(f"Средний убыток: {trades_analysis['avg_loss']:.2f}%")
    print(f"Profit Factor: {trades_analysis['profit_factor']:.2f}")
    
    print(f"\n📊 ТИПЫ ВЫХОДА:")
    for stat in trades_analysis['exit_stats']:
        print(f"  {stat['exit_type']}: {stat['count']} сделок, средний PnL: {stat['avg_pnl_percent']:.2f}%")
    
    # Дополнительные метрики из final_metrics
    if final_metrics:
        print(f"\n📈 ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")
        print(f"  Общая доходность: {final_metrics.get('total_return', 0):.2f}%")
        print(f"  Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Максимальная просадка: {final_metrics.get('max_drawdown', 0):.2f}%")
        print(f"  Средняя длительность сделки: {final_metrics.get('avg_trade_duration', 0):.1f} шагов")
    
    # Сравнение с целевыми метриками
    print(f"\n🎯 СРАВНЕНИЕ С ЦЕЛЕВЫМИ МЕТРИКАМИ:")
    
    target_metrics = {
        'win_rate': 40.0,
        'profit_factor': 1.1,
        'avg_profit/avg_loss_ratio': 1.2
    }
    
    current_metrics = {
        'win_rate': trades_analysis['win_rate'],
        'profit_factor': trades_analysis['profit_factor'],
        'avg_profit/avg_loss_ratio': trades_analysis['avg_profit'] / trades_analysis['avg_loss'] if trades_analysis['avg_loss'] > 0 else 0
    }
    
    for metric, target in target_metrics.items():
        current = current_metrics[metric]
        status = "✅ ВЫПОЛНЕНО" if current >= target else "❌ НЕ ВЫПОЛНЕНО"
        print(f"  {metric}: {current:.2f} (цель: {target:.1f}) {status}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if trades_analysis['profit_factor'] < 1.0:
        print("  ❗ Profit Factor < 1.0: Стратегия убыточна")
        print("  → Увеличить RR ratio, улучшить timing входа")
    elif trades_analysis['profit_factor'] < 1.2:
        print("  ⚠️  Profit Factor < 1.2: Есть потенциал для улучшения")
        print("  → Оптимизировать управление позицией, добавить фильтры")
    else:
        print("  ✅ Profit Factor > 1.2: Хороший результат!")
        print("  → Можно увеличить размер позиции")
    
    if trades_analysis['win_rate'] < 35:
        print("  ⚠️  Низкий Win Rate: Много стоп-лоссов")
        print("  → Проверить уровни SL, улучшить качество входов")
    
    # Сохраняем отчет в файл
    save_report_to_file(env, trades_analysis, final_metrics)

def save_report_to_file(env, trades_analysis, final_metrics):
    """Сохранение отчета в файл"""
    report_lines = []
    
    report_lines.append("="*70)
    report_lines.append("ОТЧЕТ О РЕЗУЛЬТАТАХ ТЕСТИРОВАНИЯ v15")
    report_lines.append("="*70)
    report_lines.append(f"\n=== АНАЛИЗ {trades_analysis['total_trades']} СДЕЛОК ===")
    report_lines.append(f"Win Rate: {trades_analysis['win_rate']:.1f}%")
    report_lines.append(f"Средняя прибыль: {trades_analysis['avg_profit']:.2f}%")
    report_lines.append(f"Средний убыток: {trades_analysis['avg_loss']:.2f}%")
    report_lines.append(f"Profit Factor: {trades_analysis['profit_factor']:.2f}")
    
    report_lines.append(f"\n📊 ТИПЫ ВЫХОДА:")
    for stat in trades_analysis['exit_stats']:
        report_lines.append(f"  {stat['exit_type']}: {stat['count']} сделок, средний PnL: {stat['avg_pnl_percent']:.2f}%")
    
    if final_metrics:
        report_lines.append(f"\n📈 ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")
        report_lines.append(f"  Общая доходность: {final_metrics.get('total_return', 0):.2f}%")
        report_lines.append(f"  Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
        report_lines.append(f"  Максимальная просадка: {final_metrics.get('max_drawdown', 0):.2f}%")
        report_lines.append(f"  Средняя длительность сделки: {final_metrics.get('avg_trade_duration', 0):.1f} шагов")
    
    report_lines.append(f"\n📊 ИСТОРИЯ СДЕЛОК:")
    if hasattr(env, 'trade_history') and env.trade_history:
        for i, trade in enumerate(env.trade_history[:10]):  # Первые 10 сделок
            report_lines.append(f"  {i+1}. {trade['type']}: вход ${trade['entry_price']:.2f}, выход ${trade['exit_price']:.2f}, "
                              f"PnL: {trade['pnl']*100:.2f}%, тип: {trade['exit_type']}")
    
    with open('v15_test_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ Отчет сохранен в 'v15_test_report.txt'")

def main():
    """
    Основная функция тестирования
    """
    print("\n" + "="*70)
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ v15")
    print("="*70)
    
    # 1. Загрузка тестовых данных
    test_df, obs_cols = load_test_data()
    
    # 2. Определение пути к модели
    model_paths = [
        "./models/v16_best/best_model.zip",  # Лучшая модель
        "ppo_crypto_bot_v16_final.zip",           # Финальная модель
        "./models/v16_checkpoints/ppo_bot_v16_300000_steps.zip"  # Последний чекпоинт
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Найдена модель: {path}")
            break
    
    if not model_path:
        print("❌ Модель не найдена!")
        print("Доступные пути:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    # 3. Запуск бэктеста
    env, results, episode_metrics, final_metrics = run_backtest(
        model_path=model_path,
        test_df=test_df,
        obs_cols=obs_cols,
        num_episodes=1
    )
    
    if not env:
        print("❌ Ошибка при запуске бэктеста")
        return
    
    # 4. Анализ сделок
    trades_analysis = analyze_trades(env)
    
    # 5. Построение графиков
    plot_detailed_analysis(env, trades_analysis)
    
    # 6. Генерация отчета
    generate_report(env, trades_analysis, final_metrics)
    
    # 7. Показываем итоговые метрики из среды
    print(f"\n🎯 ИТОГОВЫЕ МЕТРИКИ СРЕДЫ:")
    print(f"  Финансовые:")
    print(f"    Начальный баланс: ${env.initial_balance:.2f}")
    print(f"    Финальный баланс: ${env.net_worth:.2f}")
    print(f"    Общая доходность: {((env.net_worth / env.initial_balance - 1) * 100):.2f}%")
    
    print(f"\n  Торговые:")
    print(f"    Всего сделок: {env.total_trades}")
    print(f"    Прибыльных: {env.winning_trades} ({env.winning_trades/max(1, env.total_trades)*100:.1f}%)")
    print(f"    Убыточных: {env.losing_trades} ({env.losing_trades/max(1, env.total_trades)*100:.1f}%)")
    
    if hasattr(env, 'consecutive_wins'):
        print(f"    Текущая серия побед: {env.consecutive_wins}")
        print(f"    Текущая серия поражений: {env.consecutive_losses}")
    
    print(f"\n  Риски:")
    print(f"    Максимальная просадка: {(env.max_net_worth - min(env.net_worth_history)) / env.max_net_worth * 100:.2f}%")
    
    print(f"\n" + "="*70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    print("📁 Созданные файлы:")
    print("  - v15_detailed_analysis.png (графики)")
    print("  - v15_test_report.txt (отчет)")
    print("  - v15_test_log.csv (лог сделок)")

if __name__ == "__main__":
    main()