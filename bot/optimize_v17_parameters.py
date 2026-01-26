"""
Скрипт для анализа и оптимизации параметров стратегии V17 Optimized
Тестирует различные комбинации параметров и находит оптимальные значения
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_env_v17_optimized import CryptoTradingEnvV17_Optimized
from backtest_v17_optimized import load_historical_data, prepare_data_with_indicators, analyze_results


class ParameterOptimizer:
    """Класс для оптимизации параметров стратегии"""
    
    def __init__(self, df: pd.DataFrame, model_path: str, initial_balance: float = 10000.0):
        self.df = df
        self.model_path = model_path
        self.initial_balance = initial_balance
        self.model = None
        self.results = []
        
    def load_model(self):
        """Загрузка модели"""
        print(f"🤖 Загрузка модели: {self.model_path}")
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ Модель загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def create_env_with_params(self, params: Dict[str, Any], log_file: str) -> CryptoTradingEnvV17_Optimized:
        """Создание среды с заданными параметрами"""
        obs_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
        additional_cols = ['rsi_norm', 'trend_bias_1h', 'volatility_ratio', 'volume_ratio']
        
        for col in additional_cols:
            if col in self.df.columns:
                obs_cols.append(col)
        
        # Базовые параметры среды
        env_params = {
            'df': self.df.copy(),
            'obs_cols': obs_cols,
            'initial_balance': self.initial_balance,
            'commission': 0.001,
            'slippage': 0.0005,
            'log_file': log_file,
            'training_mode': 'optimized'
        }
        
        # Создаем среду
        env = CryptoTradingEnvV17_Optimized(**env_params)
        
        # Применяем оптимизированные параметры через прямое изменение атрибутов
        if 'min_rr_ratio' in params:
            env.min_rr_ratio = params['min_rr_ratio']
        if 'tp_levels' in params:
            env.tp_levels = params['tp_levels']
        if 'tp_close_percentages' in params:
            env.tp_close_percentages = params['tp_close_percentages']
        if 'trailing_activation_atr' in params:
            env.trailing_activation_atr = params['trailing_activation_atr']
        if 'trailing_distance_atr' in params:
            env.trailing_distance_atr = params['trailing_distance_atr']
        if 'min_trend_strength' in params:
            env.min_trend_strength = params['min_trend_strength']
        if 'min_volume_ratio' in params:
            env.min_volume_ratio = params['min_volume_ratio']
        if 'max_volatility_ratio' in params:
            env.max_volatility_ratio = params['max_volatility_ratio']
        if 'min_rsi_threshold' in params:
            env.min_rsi_threshold = params['min_rsi_threshold']
        if 'max_rsi_threshold' in params:
            env.max_rsi_threshold = params['max_rsi_threshold']
        
        return env
    
    def run_backtest_with_params(self, params: Dict[str, Any], test_name: str) -> Dict[str, Any]:
        """Запуск бэктеста с заданными параметрами"""
        log_file = f'./logs/v17_optimized/optimize_{test_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            # Создаем среду с параметрами
            env = self.create_env_with_params(params, log_file)
            env_wrapped = DummyVecEnv([lambda: env])
            
            # Запускаем бэктест
            # DummyVecEnv.reset() возвращает только наблюдения, не кортеж
            obs = env_wrapped.reset()
            done = False
            steps = 0
            max_steps = len(self.df)
            
            while not done and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                # DummyVecEnv.step() возвращает (obs, reward, done, info)
                obs, reward, done_array, info = env_wrapped.step(action)
                # done может быть массивом или булевым значением
                if isinstance(done_array, (list, np.ndarray)):
                    done = bool(done_array[0])
                else:
                    done = bool(done_array)
                steps += 1
            
            env_wrapped.close()
            
            # Анализируем результаты
            report = analyze_results(log_file, self.initial_balance, 'BTCUSDT')
            
            if report:
                report['params'] = params
                report['test_name'] = test_name
                report['log_file'] = log_file
                return report
            else:
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при тестировании {test_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_parameters(self, param_grid: Dict[str, List[Any]], max_tests: int = None):
        """
        Оптимизация параметров методом Grid Search
        
        Args:
            param_grid: Словарь с параметрами и их возможными значениями
            max_tests: Максимальное количество тестов (None = все комбинации)
        """
        print(f"\n{'='*60}")
        print("🔍 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        print(f"{'='*60}")
        
        # Генерируем все комбинации параметров
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)
        
        if max_tests:
            all_combinations = all_combinations[:max_tests]
            total_combinations = len(all_combinations)
        
        print(f"   Всего комбинаций: {total_combinations}")
        print(f"   Параметры для оптимизации: {param_names}")
        
        results = []
        
        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))
            test_name = f"test_{i:03d}"
            
            print(f"\n[{i}/{total_combinations}] Тестирование комбинации {i}...")
            print(f"   Параметры: {params}")
            
            result = self.run_backtest_with_params(params, test_name)
            
            if result:
                results.append(result)
                print(f"   ✅ Win Rate: {result['win_rate']:.2f}%, "
                      f"Profit Factor: {result['profit_factor']:.2f}, "
                      f"Return: {result['total_return']:.2f}%")
            else:
                print(f"   ❌ Тест провален")
        
        self.results = results
        return results
    
    def find_best_parameters(self, metric: str = 'total_return', min_trades: int = 10):
        """
        Поиск лучших параметров по заданной метрике
        
        Args:
            metric: Метрика для сравнения ('total_return', 'win_rate', 'profit_factor', 'sharpe_ratio')
            min_trades: Минимальное количество сделок для валидности результата
        """
        if not self.results:
            print("❌ Нет результатов для анализа")
            return None
        
        # Фильтруем результаты с достаточным количеством сделок
        valid_results = [r for r in self.results if r.get('total_trades', 0) >= min_trades]
        
        if not valid_results:
            print(f"❌ Нет результатов с минимум {min_trades} сделками")
            return None
        
        # Сортируем по метрике
        if metric == 'sharpe_ratio':
            # Рассчитываем Sharpe Ratio если его нет
            for r in valid_results:
                if 'sharpe_ratio' not in r:
                    # Упрощенный расчет: return / max_drawdown
                    if r.get('max_drawdown', 0) > 0:
                        r['sharpe_ratio'] = r['total_return'] / r['max_drawdown']
                    else:
                        r['sharpe_ratio'] = 0
        
        valid_results.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        best = valid_results[0]
        
        print(f"\n{'='*60}")
        print(f"🏆 ЛУЧШИЕ ПАРАМЕТРЫ (по {metric})")
        print(f"{'='*60}")
        print(f"\n📊 Результаты:")
        print(f"   Win Rate: {best['win_rate']:.2f}%")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Total Return: {best['total_return']:.2f}%")
        print(f"   Avg RR: {best['avg_rr']:.2f}")
        print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")
        print(f"   Total Trades: {best['total_trades']}")
        
        print(f"\n⚙️ Параметры:")
        for key, value in best['params'].items():
            print(f"   {key}: {value}")
        
        return best
    
    def save_results(self, output_file: str = None):
        """Сохранение результатов оптимизации"""
        if not output_file:
            output_file = f'./logs/v17_optimized/optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Подготавливаем данные для сохранения
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'results': []
        }
        
        for result in self.results:
            result_copy = result.copy()
            # Конвертируем numpy типы в Python типы для JSON
            for key, value in result_copy.items():
                if isinstance(value, (np.integer, np.floating)):
                    result_copy[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result_copy[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                    result_copy[key] = [float(v) for v in value]
            
            output_data['results'].append(result_copy)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Результаты сохранены в: {output_file}")
        return output_file
    
    def generate_report(self, top_n: int = 10):
        """Генерация отчета с топ-N результатами"""
        if not self.results:
            print("❌ Нет результатов для отчета")
            return
        
        # Сортируем по total_return
        sorted_results = sorted(self.results, key=lambda x: x.get('total_return', 0), reverse=True)
        
        print(f"\n{'='*60}")
        print(f"📊 ТОП-{top_n} РЕЗУЛЬТАТОВ")
        print(f"{'='*60}")
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. Тест: {result.get('test_name', 'N/A')}")
            print(f"   Win Rate: {result['win_rate']:.2f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            print(f"   Avg RR: {result['avg_rr']:.2f}")
            print(f"   Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"   Total Trades: {result['total_trades']}")
            print(f"   Параметры: {result['params']}")


def create_parameter_grid(mode: str = 'full') -> Dict[str, List[Any]]:
    """
    Создание сетки параметров для оптимизации
    
    Args:
        mode: Режим оптимизации ('full', 'quick', 'focused')
    """
    if mode == 'quick':
        # Быстрая оптимизация - меньше комбинаций
        return {
            'min_rr_ratio': [1.5, 1.8, 2.0],
            'min_trend_strength': [0.25, 0.35, 0.45],
            'min_volume_ratio': [0.8, 1.0],
            'trailing_activation_atr': [0.25, 0.35]
        }
    elif mode == 'focused':
        # Фокус на ключевых параметрах
        return {
            'min_rr_ratio': [1.5, 1.8, 2.0, 2.2],
            'tp_levels': [
                [1.8, 2.5, 3.5],
                [2.0, 3.0, 4.0],
                [1.5, 2.0, 2.5]
            ],
            'min_trend_strength': [0.25, 0.35, 0.45],
            'trailing_activation_atr': [0.25, 0.35, 0.45]
        }
    else:  # full
        # Полная оптимизация - все комбинации
        return {
            'min_rr_ratio': [1.5, 1.8, 2.0, 2.2],
            'tp_levels': [
                [1.8, 2.5, 3.5],
                [2.0, 3.0, 4.0],
                [1.5, 2.0, 2.5]
            ],
            'min_trend_strength': [0.25, 0.35, 0.45],
            'min_volume_ratio': [0.8, 1.0, 1.2],
            'max_volatility_ratio': [1.8, 2.0, 2.2],
            'trailing_activation_atr': [0.25, 0.35, 0.45],
            'trailing_distance_atr': [0.35, 0.45, 0.55]
        }


def main():
    parser = argparse.ArgumentParser(description='Оптимизация параметров стратегии V17 Optimized')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Торговая пара')
    parser.add_argument('--days', type=int, default=30, 
                       help='Количество дней истории')
    parser.add_argument('--timeframe', type=str, default='15m', 
                       help='Таймфрейм')
    parser.add_argument('--model', type=str, default='./models/v17_optimized/ppo_final.zip', 
                       help='Путь к модели')
    parser.add_argument('--mode', type=str, default='focused', 
                       choices=['quick', 'focused', 'full'],
                       help='Режим оптимизации')
    parser.add_argument('--max-tests', type=int, default=None, 
                       help='Максимальное количество тестов')
    parser.add_argument('--balance', type=float, default=10000.0, 
                       help='Начальный баланс')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔍 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ V17 OPTIMIZED")
    print("="*60)
    
    # 1. Загрузка данных
    print(f"\n📥 Загрузка данных...")
    df = load_historical_data(args.symbol, args.days, args.timeframe)
    if df is None or len(df) == 0:
        print("❌ Не удалось загрузить данные")
        return
    
    # 2. Подготовка данных
    print(f"\n📊 Подготовка данных...")
    df_prepared = prepare_data_with_indicators(df)
    if df_prepared is None or len(df_prepared) == 0:
        print("❌ Не удалось подготовить данные")
        return
    
    # 3. Создание оптимизатора
    optimizer = ParameterOptimizer(df_prepared, args.model, args.balance)
    
    # 4. Загрузка модели
    if not optimizer.load_model():
        print("❌ Не удалось загрузить модель")
        return
    
    # 5. Создание сетки параметров
    param_grid = create_parameter_grid(args.mode)
    print(f"\n⚙️ Режим оптимизации: {args.mode}")
    print(f"   Параметры: {list(param_grid.keys())}")
    
    # 6. Запуск оптимизации
    results = optimizer.optimize_parameters(param_grid, args.max_tests)
    
    if not results:
        print("❌ Оптимизация не дала результатов")
        return
    
    # 7. Поиск лучших параметров
    best_params = optimizer.find_best_parameters(metric='total_return', min_trades=10)
    
    # 8. Генерация отчета
    optimizer.generate_report(top_n=10)
    
    # 9. Сохранение результатов
    output_file = optimizer.save_results()
    
    print(f"\n{'='*60}")
    print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"{'='*60}")
    print(f"\n📁 Результаты сохранены в: {output_file}")
    
    if best_params:
        print(f"\n💡 Рекомендуемые параметры:")
        for key, value in best_params['params'].items():
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
