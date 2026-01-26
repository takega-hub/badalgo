"""
Скрипт для анализа результатов оптимизации параметров V17
Помогает визуализировать и интерпретировать результаты оптимизации
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_optimization_results(json_file: str) -> Dict[str, Any]:
    """Загрузка результатов оптимизации из JSON файла"""
    if not os.path.exists(json_file):
        print(f"❌ Файл не найден: {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def analyze_results(data: Dict[str, Any]):
    """Анализ результатов оптимизации"""
    print(f"\n{'='*60}")
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
    print(f"{'='*60}")
    
    results = data.get('results', [])
    if not results:
        print("❌ Нет результатов для анализа")
        return
    
    print(f"\n📈 Общая статистика:")
    print(f"   Всего тестов: {len(results)}")
    
    # Создаем DataFrame для анализа
    df = pd.DataFrame(results)
    
    # Базовые метрики
    print(f"\n📊 МЕТРИКИ:")
    print(f"   Средний Win Rate: {df['win_rate'].mean():.2f}%")
    print(f"   Средний Profit Factor: {df['profit_factor'].mean():.2f}")
    print(f"   Средний Total Return: {df['total_return'].mean():.2f}%")
    print(f"   Средний Avg RR: {df['avg_rr'].mean():.2f}")
    print(f"   Средний Max Drawdown: {df['max_drawdown'].mean():.2f}%")
    print(f"   Среднее количество сделок: {df['total_trades'].mean():.0f}")
    
    print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    
    # Лучший по Total Return
    best_return = df.loc[df['total_return'].idxmax()]
    print(f"\n   1. По Total Return:")
    print(f"      Win Rate: {best_return['win_rate']:.2f}%")
    print(f"      Profit Factor: {best_return['profit_factor']:.2f}")
    print(f"      Total Return: {best_return['total_return']:.2f}%")
    print(f"      Avg RR: {best_return['avg_rr']:.2f}")
    print(f"      Max Drawdown: {best_return['max_drawdown']:.2f}%")
    print(f"      Total Trades: {best_return['total_trades']}")
    print(f"      Параметры: {best_return['params']}")
    
    # Лучший по Win Rate
    best_winrate = df.loc[df['win_rate'].idxmax()]
    print(f"\n   2. По Win Rate:")
    print(f"      Win Rate: {best_winrate['win_rate']:.2f}%")
    print(f"      Profit Factor: {best_winrate['profit_factor']:.2f}")
    print(f"      Total Return: {best_winrate['total_return']:.2f}%")
    print(f"      Параметры: {best_winrate['params']}")
    
    # Лучший по Profit Factor
    # Фильтруем бесконечные значения
    df_pf = df[df['profit_factor'] != float('inf')]
    if len(df_pf) > 0:
        best_pf = df_pf.loc[df_pf['profit_factor'].idxmax()]
        print(f"\n   3. По Profit Factor:")
        print(f"      Win Rate: {best_pf['win_rate']:.2f}%")
        print(f"      Profit Factor: {best_pf['profit_factor']:.2f}")
        print(f"      Total Return: {best_pf['total_return']:.2f}%")
        print(f"      Параметры: {best_pf['params']}")
    
    # Анализ параметров
    print(f"\n{'='*60}")
    print("🔍 АНАЛИЗ ПАРАМЕТРОВ")
    print(f"{'='*60}")
    
    # Анализируем влияние каждого параметра
    param_analysis = {}
    
    for result in results:
        params = result.get('params', {})
        for param_name, param_value in params.items():
            if param_name not in param_analysis:
                param_analysis[param_name] = {}
            
            # Группируем по значениям параметра
            if param_value not in param_analysis[param_name]:
                param_analysis[param_name][param_value] = {
                    'count': 0,
                    'win_rates': [],
                    'profit_factors': [],
                    'returns': []
                }
            
            param_analysis[param_name][param_value]['count'] += 1
            param_analysis[param_name][param_value]['win_rates'].append(result['win_rate'])
            param_analysis[param_name][param_value]['profit_factors'].append(
                result['profit_factor'] if result['profit_factor'] != float('inf') else 0
            )
            param_analysis[param_name][param_value]['returns'].append(result['total_return'])
    
    # Выводим анализ параметров
    for param_name, param_values in param_analysis.items():
        print(f"\n   📌 {param_name}:")
        
        # Сортируем по среднему return
        sorted_values = sorted(
            param_values.items(),
            key=lambda x: np.mean(x[1]['returns']),
            reverse=True
        )
        
        for param_value, stats in sorted_values[:5]:  # Топ-5 значений
            avg_winrate = np.mean(stats['win_rates'])
            avg_pf = np.mean(stats['profit_factors'])
            avg_return = np.mean(stats['returns'])
            
            print(f"      {param_value}: "
                  f"WR={avg_winrate:.1f}%, "
                  f"PF={avg_pf:.2f}, "
                  f"Return={avg_return:.2f}% "
                  f"({stats['count']} тестов)")
    
    # Рекомендации
    print(f"\n{'='*60}")
    print("💡 РЕКОМЕНДАЦИИ")
    print(f"{'='*60}")
    
    # Находим параметры с лучшими результатами
    recommendations = {}
    
    for param_name, param_values in param_analysis.items():
        # Находим значение с лучшим средним return
        best_value = max(
            param_values.items(),
            key=lambda x: np.mean(x[1]['returns'])
        )[0]
        
        recommendations[param_name] = best_value
    
    print(f"\n   Рекомендуемые параметры (на основе средних результатов):")
    for param_name, param_value in recommendations.items():
        print(f"      {param_name}: {param_value}")
    
    # Проверяем, есть ли прибыльные стратегии
    profitable = df[df['total_return'] > 0]
    if len(profitable) > 0:
        print(f"\n   ✅ Найдено {len(profitable)} прибыльных стратегий")
        print(f"      Лучшая доходность: {profitable['total_return'].max():.2f}%")
    else:
        print(f"\n   ⚠️ Нет прибыльных стратегий в тестах")
        print(f"      Лучшая доходность: {df['total_return'].max():.2f}%")
    
    # Проверяем Win Rate
    good_winrate = df[df['win_rate'] > 50]
    if len(good_winrate) > 0:
        print(f"   ✅ Найдено {len(good_winrate)} стратегий с Win Rate > 50%")
        print(f"      Лучший Win Rate: {good_winrate['win_rate'].max():.2f}%")
    else:
        print(f"   ⚠️ Нет стратегий с Win Rate > 50%")
        print(f"      Лучший Win Rate: {df['win_rate'].max():.2f}%")
    
    return df, recommendations

def save_recommendations(recommendations: Dict[str, Any], output_file: str):
    """Сохранение рекомендаций в файл"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Рекомендации сохранены в: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Анализ результатов оптимизации V17')
    parser.add_argument('--file', type=str, 
                       default='./logs/v17_optimized/optimization_results_*.json',
                       help='Путь к JSON файлу с результатами')
    parser.add_argument('--output', type=str, 
                       default='./logs/v17_optimized/recommended_params.json',
                       help='Путь для сохранения рекомендаций')
    
    args = parser.parse_args()
    
    # Находим файл результатов
    if '*' in args.file:
        # Ищем последний файл
        log_dir = os.path.dirname(args.file)
        pattern = os.path.basename(args.file).replace('*', '')
        
        if os.path.exists(log_dir):
            files = [f for f in os.listdir(log_dir) if f.startswith('optimization_results_') and f.endswith('.json')]
            if files:
                files.sort(reverse=True)
                args.file = os.path.join(log_dir, files[0])
                print(f"📁 Используется файл: {args.file}")
            else:
                print(f"❌ Файлы результатов не найдены в {log_dir}")
                return
        else:
            print(f"❌ Директория не найдена: {log_dir}")
            return
    
    # Загружаем результаты
    data = load_optimization_results(args.file)
    if not data:
        return
    
    # Анализируем
    df, recommendations = analyze_results(data)
    
    # Сохраняем рекомендации
    if recommendations:
        save_recommendations(recommendations, args.output)

if __name__ == "__main__":
    main()
