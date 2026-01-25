import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

def analyze_trading_results():
    """Анализ результатов торговли"""
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ТОРГОВЛИ")
    print("="*60)
    
    # Папка с логами
    log_dir = './logs/v16_profit_focused_btc'
    
    # Собираем все лог-файлы
    log_files = []
    for file in os.listdir(log_dir):
        if file.endswith('.csv') and 'log' in file.lower():
            log_files.append(os.path.join(log_dir, file))
    
    print(f"Найдено {len(log_files)} лог-файлов")
    
    all_results = []
    
    for log_file in log_files:
        print(f"\n📄 Анализ: {os.path.basename(log_file)}")
        
        try:
            df = pd.read_csv(log_file)
            
            if len(df) > 1:
                trades_df = df.iloc[1:].copy()  # Пропускаем заголовок
                
                # Анализ PnL
                def parse_pnl(pnl_str):
                    try:
                        if isinstance(pnl_str, str):
                            clean_str = str(pnl_str).replace('%', '').strip()
                            return float(clean_str)
                        return float(pnl_str)
                    except:
                        return 0.0
                
                trades_df['pnl_value'] = trades_df['pnl_percent'].apply(parse_pnl)
                
                # Базовая статистика
                profitable = (trades_df['pnl_value'] > 0).sum()
                losing = (trades_df['pnl_value'] < 0).sum()
                total = len(trades_df)
                win_rate = profitable / total * 100 if total > 0 else 0
                avg_pnl = trades_df['pnl_value'].mean()
                total_pnl = trades_df['pnl_value'].sum()
                
                print(f"   Сделок: {total}")
                print(f"   Прибыльных: {profitable} ({win_rate:.1f}%)")
                print(f"   Убыточных: {losing}")
                print(f"   Средний PnL: {avg_pnl:.2f}%")
                print(f"   Общий PnL: {total_pnl:.2f}%")
                
                # Сохраняем для общего анализа
                all_results.append({
                    'file': os.path.basename(log_file),
                    'trades': total,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': total_pnl,
                    'profitable': profitable,
                    'losing': losing
                })
                
                # Детальный анализ по типам сделок
                if 'type' in trades_df.columns:
                    print(f"\n   📈 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ:")
                    type_stats = {}
                    for trade_type in trades_df['type'].unique():
                        type_trades = trades_df[trades_df['type'] == trade_type]
                        type_profitable = (type_trades['pnl_value'] > 0).sum()
                        type_total = len(type_trades)
                        type_win_rate = type_profitable / type_total * 100 if type_total > 0 else 0
                        type_avg_pnl = type_trades['pnl_value'].mean()
                        
                        type_stats[trade_type] = {
                            'count': type_total,
                            'win_rate': type_win_rate,
                            'avg_pnl': type_avg_pnl
                        }
                        
                        print(f"     {trade_type}: {type_total} сделок, Win Rate: {type_win_rate:.1f}%, Avg PnL: {type_avg_pnl:.2f}%")
                
                # Анализ причин выхода
                if 'exit_reason' in trades_df.columns:
                    print(f"\n   🔚 ПРИЧИНЫ ВЫХОДА:")
                    exit_stats = trades_df['exit_reason'].value_counts()
                    for reason, count in exit_stats.head(10).items():
                        reason_trades = trades_df[trades_df['exit_reason'] == reason]
                        reason_pnl = reason_trades['pnl_value'].mean() if len(reason_trades) > 0 else 0
                        print(f"     {reason}: {count} (Avg PnL: {reason_pnl:.2f}%)")
                
                # Анализ по длительности
                if 'duration' in trades_df.columns:
                    print(f"\n   ⏱️  АНАЛИЗ ПО ДЛИТЕЛЬНОСТИ:")
                    
                    # Группируем по длительности
                    trades_df['duration_group'] = pd.cut(trades_df['duration'], 
                                                       bins=[0, 5, 10, 20, 50, 100, 200],
                                                       labels=['<5', '5-10', '10-20', '20-50', '50-100', '>100'])
                    
                    duration_stats = trades_df.groupby('duration_group')['pnl_value'].agg(['count', 'mean'])
                    for duration, stats in duration_stats.iterrows():
                        print(f"     {duration}: {int(stats['count'])} сделок, Avg PnL: {stats['mean']:.2f}%")
                
            else:
                print(f"   ⚠️ Только заголовок, сделок нет")
                
        except Exception as e:
            print(f"   ❌ Ошибка анализа: {e}")
    
    # Общий анализ
    if all_results:
        print(f"\n{'='*60}")
        print("📊 ОБЩИЙ АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        total_trades = sum(r['trades'] for r in all_results)
        avg_win_rate = sum(r['win_rate'] * r['trades'] for r in all_results) / total_trades if total_trades > 0 else 0
        avg_pnl = sum(r['avg_pnl'] * r['trades'] for r in all_results) / total_trades if total_trades > 0 else 0
        
        print(f"Всего сделок во всех логах: {total_trades}")
        print(f"Средний Win Rate: {avg_win_rate:.1f}%")
        print(f"Средний PnL: {avg_pnl:.2f}%")
        
        # Лучший файл
        best_file = max(all_results, key=lambda x: x['total_pnl'])
        print(f"\n🏆 Лучший результат: {best_file['file']}")
        print(f"   PnL: {best_file['total_pnl']:.2f}%, Win Rate: {best_file['win_rate']:.1f}%")
    
    return all_results


def identify_problems_and_solutions():
    """Выявление проблем и предложение решений"""
    print(f"\n{'='*60}")
    print("🔍 ВЫЯВЛЕНИЕ ПРОБЛЕМ И ПРЕДЛОЖЕНИЕ РЕШЕНИЙ")
    print("="*60)
    
    problems = [
        {
            'problem': 'Низкий RR ratio (0.76)',
            'cause': 'Слишком широкий SL или слишком близкий TP',
            'solution': 'Увеличить atr_multiplier или уменьшить tp_levels[0]',
            'action': 'Установить min_rr_ratio=1.5 и отбрасывать сделки с RR < 1.5'
        },
        {
            'problem': 'Много SL выходов',
            'cause': 'Фильтры входа недостаточно строгие или SL слишком близко',
            'solution': 'Ужесточить фильтры входа, увеличить min_trend_strength',
            'action': 'Увеличить min_trend_strength до 0.5, добавить фильтр по объему'
        },
        {
            'problem': 'Win Rate 56-65% но средний PnL низкий',
            'cause': 'Прибыльные сделки маленькие, убыточные большие',
            'solution': 'Увеличить TP уровни, улучшить trailing stop',
            'action': 'tp_levels = [2.0, 3.0, 4.0], trailing_activation_atr=0.2'
        },
        {
            'problem': 'Тестирование показывает 0 reward',
            'cause': 'Модель не адаптирована к новым данным',
            'solution': 'Увеличить разнообразие данных при обучении',
            'action': 'Использовать больше исторических данных, data augmentation'
        }
    ]
    
    for i, prob in enumerate(problems, 1):
        print(f"\n{i}. {prob['problem']}")
        print(f"   Причина: {prob['cause']}")
        print(f"   Решение: {prob['solution']}")
        print(f"   Действие: {prob['action']}")


def create_optimized_environment_config():
    """Создание конфигурации для оптимизированной среды"""
    print(f"\n{'='*60}")
    print("⚙️  КОНФИГУРАЦИЯ ОПТИМИЗИРОВАННОЙ СРЕДЫ")
    print("="*60)
    
    config = {
        # Жесткие фильтры входа
        'min_tp_percent': 0.006,  # 0.6% вместо 0.8%
        'min_sl_percent': 0.003,  # 0.3%
        'max_sl_percent': 0.008,  # 0.8%
        
        # Улучшенный RR ratio
        'base_rr_ratio': 2.0,  # Целевой RR 1:2
        'atr_multiplier': 2.5,  # Увеличили для большего SL
        
        # TP уровни для большего профита
        'tp_levels': [1.8, 2.5, 3.5],  # Увеличили
        'tp_close_percentages': [0.25, 0.35, 0.40],  # Последний TP больше
        
        # Улучшенный трейлинг-стоп
        'trailing_activation_atr': 0.25,  # Ранее активация
        'trailing_distance_atr': 0.35,    # Ближе трейлинг
        
        # Жесткие фильтры
        'min_trend_strength': 0.45,  # Усилили
        'max_volatility_ratio': 1.8,  # Ужесточили
        'min_rsi_threshold': 0.15,    # Ближе к 50
        'max_rsi_threshold': 0.7,     # Дальше от экстремумов
        
        # Награды и штрафы
        'tp_bonus_multiplier': 10.0,  # Больше награда за TP
        'sl_penalty_multiplier': 5.0,  # Больше штраф за SL
        'quality_bonus_threshold': 0.015,  # 1.5% для бонуса
        
        # Управление рисками
        'base_margin_percent': 0.07,  # Немного уменьшили
        'max_daily_trades': 3,        # Меньше сделок, выше качество
        
        # Время удержания
        'min_hold_steps': 8,          # Быстрее можно выйти
        'max_hold_steps': 60,         # Меньше максимальное время
    }
    
    print("Оптимизированные параметры:")
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Сохраняем конфигурацию
    config_file = './models/v16_profit_focused_btc/optimized_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Конфигурация сохранена: {config_file}")
    
    return config


def create_training_plan():
    """Создание плана дальнейшего обучения"""
    print(f"\n{'='*60}")
    print("📋 ПЛАН ДАЛЬНЕЙШЕГО ОБУЧЕНИЯ")
    print("="*60)
    
    plan = {
        'phase_1': {
            'name': 'Фиксация RR ratio',
            'steps': 'Внедрить минимальный RR=1.5 в _check_entry_filters_strict',
            'goal': 'Исключить сделки с плохим RR',
            'expected': 'Уменьшение количества убыточных сделок'
        },
        'phase_2': {
            'name': 'Ужесточение фильтров',
            'steps': 'Увеличить min_trend_strength, добавить фильтр объема',
            'goal': 'Улучшение качества входов',
            'expected': 'Увеличение Win Rate'
        },
        'phase_3': {
            'name': 'Оптимизация TP/SL',
            'steps': 'Настроить tp_levels и trailing stop',
            'goal': 'Увеличение средней прибыли',
            'expected': 'Рост среднего PnL'
        },
        'phase_4': {
            'name': 'Обучение на оптимизированной среде',
            'steps': 'Загрузить оптимизированную конфигурацию, обучить 10000 шагов',
            'goal': 'Получить стабильную прибыльную стратегию',
            'expected': 'Profit Factor > 1.5, Win Rate > 60%'
        }
    }
    
    for phase, details in plan.items():
        print(f"\n{phase.upper()}: {details['name']}")
        print(f"  Шаги: {details['steps']}")
        print(f"  Цель: {details['goal']}")
        print(f"  Ожидаемый результат: {details['expected']}")
    
    # Создаем план в файле
    plan_file = './models/v16_profit_focused_btc/training_plan.txt'
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write("ПЛАН ОПТИМИЗАЦИИ И ОБУЧЕНИЯ\n")
        f.write("="*50 + "\n\n")
        for phase, details in plan.items():
            f.write(f"{phase.upper()}: {details['name']}\n")
            f.write(f"  Шаги: {details['steps']}\n")
            f.write(f"  Цель: {details['goal']}\n")
            f.write(f"  Ожидаемый результат: {details['expected']}\n\n")
    
    print(f"\n✅ План сохранен: {plan_file}")


def main():
    """Основная функция"""
    print("🔍 АНАЛИЗ РЕЗУЛЬТАТОВ И ОПТИМИЗАЦИЯ")
    print("="*60)
    
    # 1. Анализ результатов
    results = analyze_trading_results()
    
    # 2. Выявление проблем
    identify_problems_and_solutions()
    
    # 3. Создание оптимизированной конфигурации
    config = create_optimized_environment_config()
    
    # 4. Создание плана обучения
    create_training_plan()
    
    print(f"\n{'='*60}")
    print("🎯 РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШИХ ДЕЙСТВИЙ")
    print("="*60)
    print("1. Сначала исправьте RR ratio - добавьте проверку в фильтры входа")
    print("2. Ужесточите фильтры для уменьшения SL выходов")
    print("3. Используйте оптимизированную конфигурацию для следующего обучения")
    print("4. Увеличьте количество шагов обучения до 20000-50000")
    print("5. Регулярно анализируйте логи и корректируйте параметры")


if __name__ == "__main__":
    main()