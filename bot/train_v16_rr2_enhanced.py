import os
import numpy as np
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from crypto_env_v16_rr2_enhanced import CryptoTradingEnvV16_RR2_Enhanced
from data_processor_enhanced import DataProcessorEnhanced


class HistoricalTester:
    """Класс для тестирования модели на разных исторических периодах"""
    
    def __init__(self, model, df, obs_cols, symbol="BTC", initial_balance=10000):
        self.model = model
        self.df = df.copy()
        self.obs_cols = obs_cols
        self.symbol = symbol
        self.initial_balance = initial_balance
        
    def create_test_periods(self, n_periods=5, period_length=500, overlap=0.2):
        """Создание разных тестовых периодов из данных"""
        total_length = len(self.df)
        period_size = period_length
        overlap_size = int(period_size * overlap)
        
        test_periods = []
        
        # Если данных мало, просто делим на части
        if total_length < period_size * 2:
            n_chunks = min(n_periods, total_length // 200)  # Минимум 200 баров на период
            chunk_size = total_length // n_chunks
            
            for i in range(n_periods):
                if i >= n_chunks:
                    break
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < n_chunks - 1 else total_length
                
                period_data = self.df.iloc[start_idx:end_idx].copy()
                
                # Создаем serializable информацию о периоде (без DataFrame)
                period_info = {
                    'id': i + 1,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'length': int(len(period_data)),
                    'data_indices': list(range(int(start_idx), int(end_idx))),
                    'start_price': float(period_data['close'].iloc[0]),
                    'end_price': float(period_data['close'].iloc[-1]),
                    'price_change_pct': float((period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0] * 100),
                    'avg_price': float(period_data['close'].mean()),
                    'avg_volume': float(period_data['volume'].mean()) if 'volume' in period_data.columns else 0,
                    'volatility': float(period_data['close'].std() / period_data['close'].mean() * 100) if len(period_data) > 1 else 0
                }
                
                # Сохраняем отдельно данные для тестирования
                test_periods.append({
                    'info': period_info,
                    'data': period_data
                })
        else:
            # Создаем перекрывающиеся периоды
            step_size = period_size - overlap_size
            
            for i in range(n_periods):
                start_idx = i * step_size
                end_idx = start_idx + period_size
                
                # Если вышли за пределы данных
                if end_idx > total_length:
                    start_idx = total_length - period_size
                    end_idx = total_length
                
                period_data = self.df.iloc[start_idx:end_idx].copy()
                
                # Создаем serializable информацию о периоде
                period_info = {
                    'id': i + 1,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'length': int(len(period_data)),
                    'data_indices': list(range(int(start_idx), int(end_idx))),
                    'start_price': float(period_data['close'].iloc[0]),
                    'end_price': float(period_data['close'].iloc[-1]),
                    'price_change_pct': float((period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0] * 100),
                    'avg_price': float(period_data['close'].mean()),
                    'avg_volume': float(period_data['volume'].mean()) if 'volume' in period_data.columns else 0,
                    'volatility': float(period_data['close'].std() / period_data['close'].mean() * 100) if len(period_data) > 1 else 0
                }
                
                test_periods.append({
                    'info': period_info,
                    'data': period_data
                })
                
                # Если достигли конца данных
                if end_idx == total_length:
                    break
        
        return test_periods
    
    def create_market_regimes(self, n_periods=5, regime_length=400):
        """Создание периодов с разными рыночными режимами"""
        total_length = len(self.df)
        
        # Анализируем данные для определения режимов
        returns = self.df['close'].pct_change().fillna(0)
        volatility = returns.rolling(50).std().fillna(0)
        
        # Находим периоды с разной волатильностью
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)
        
        regimes = []
        regime_id = 0
        
        for i in range(0, total_length - regime_length, regime_length // 2):
            if regime_id >= n_periods:
                break
                
            end_idx = min(i + regime_length, total_length)
            period_data = self.df.iloc[i:end_idx].copy()
            
            # Определяем режим периода
            period_vol = volatility.iloc[i:end_idx].mean()
            period_return = returns.iloc[i:end_idx].sum() * 100
            
            if period_vol > high_vol_threshold:
                regime_type = "HIGH_VOLATILITY"
            elif period_vol < low_vol_threshold:
                regime_type = "LOW_VOLATILITY"
            else:
                regime_type = "NORMAL"
            
            # Определяем тренд
            if period_return > 5:
                trend = "STRONG_UP"
            elif period_return > 1:
                trend = "UP"
            elif period_return < -5:
                trend = "STRONG_DOWN"
            elif period_return < -1:
                trend = "DOWN"
            else:
                trend = "SIDEWAYS"
            
            # Создаем serializable информацию о периоде
            period_info = {
                'id': regime_id + 1,
                'start_idx': int(i),
                'end_idx': int(end_idx),
                'length': int(len(period_data)),
                'regime': regime_type,
                'trend': trend,
                'volatility': float(period_vol),
                'return_pct': float(period_return),
                'start_price': float(period_data['close'].iloc[0]),
                'end_price': float(period_data['close'].iloc[-1]),
                'price_change_pct': float((period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0] * 100),
                'avg_price': float(period_data['close'].mean()),
                'avg_volume': float(period_data['volume'].mean()) if 'volume' in period_data.columns else 0,
                'data_indices': list(range(int(i), int(end_idx)))
            }
            
            regimes.append({
                'info': period_info,
                'data': period_data
            })
            regime_id += 1
        
        return regimes
    
    def run_test_on_period(self, period_item, log_suffix=""):
        """Запуск теста на одном периоде"""
        period_info = period_item['info']
        period_data = period_item['data']
        
        log_dir = f"./logs/v16_historical_test/{self.symbol}/"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f"{log_dir}period_{period_info['id']}_{log_suffix}.csv"
        
        # Создаем лог
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("step,type,entry,sl_initial,sl_current,tp_levels,exit,pnl_percent,net_worth,exit_reason,duration,trailing,tp_closed,partial_closes\n")
        
        # Создаем среду для этого периода
        def make_test_env():
            env = CryptoTradingEnvV16_RR2_Enhanced(
                df=period_data,
                obs_cols=self.obs_cols,
                initial_balance=self.initial_balance,
                commission=0.001,
                slippage=0.0005,
                rr_ratio=2.0,
                atr_multiplier=1.4,
                log_file=log_file,
                training_mode="rr2_enhanced"
            )
            return env
        
        test_env = DummyVecEnv([make_test_env])
        
        # Запускаем тест
        obs = test_env.reset()
        episode_reward = 0
        done = False
        steps = 0
        trades_count = 0
        
        while not done and steps < 1000:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            episode_reward += float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
            steps += 1
            
            # Считаем сделки
            if info and isinstance(info, list) and len(info) > 0:
                if 'total_trades' in info[0]:
                    trades_count = info[0]['total_trades']
            
            if done[0]:
                break
        
        # Анализируем лог
        detailed_stats = self.analyze_period_log(log_file, period_info)
        
        test_env.close()
        
        return {
            'period_id': period_info['id'],
            'period_info': period_info,  # Только serializable информация
            'steps': steps,
            'reward': episode_reward,
            'trades': trades_count,
            'detailed_stats': detailed_stats,
            'log_file': log_file
        }
    
    def analyze_period_log(self, log_file, period_info):
        """Анализ лога периода"""
        if not os.path.exists(log_file):
            return {}
        
        try:
            df_log = pd.read_csv(log_file)
            
            if len(df_log) == 0:
                return {}
            
            stats = {
                'total_trades': len(df_log),
            }
            
            # Анализ PnL
            if 'pnl_percent' in df_log.columns:
                # Парсим PnL
                def parse_pnl(pnl_str):
                    try:
                        if isinstance(pnl_str, str):
                            clean = pnl_str.replace('%', '').replace('@', '0').strip()
                            return float(clean)
                        return float(pnl_str)
                    except:
                        return 0.0
                
                df_log['pnl_value'] = df_log['pnl_percent'].apply(parse_pnl)
                pnl_values = df_log['pnl_value']
                
                # Фильтруем выбросы
                pnl_filtered = pnl_values[(pnl_values >= -100) & (pnl_values <= 100)]
                
                if len(pnl_filtered) > 0:
                    stats['avg_pnl'] = float(pnl_filtered.mean())
                    stats['total_pnl'] = float(pnl_filtered.sum())
                    stats['profitable_trades'] = int((pnl_filtered > 0).sum())
                    stats['loss_trades'] = int((pnl_filtered < 0).sum())
                    stats['win_rate'] = float(stats['profitable_trades'] / len(pnl_filtered)) if len(pnl_filtered) > 0 else 0
                    
                    if stats['profitable_trades'] > 0:
                        stats['avg_win'] = float(pnl_filtered[pnl_filtered > 0].mean())
                    if stats['loss_trades'] > 0:
                        stats['avg_loss'] = float(pnl_filtered[pnl_filtered < 0].mean())
            
            # Анализ типов сделок
            if 'type' in df_log.columns:
                stats['long_trades'] = int((df_log['type'] == 'LONG').sum())
                stats['short_trades'] = int((df_log['type'] == 'SHORT').sum())
                stats['partial_trades'] = int(df_log['type'].str.contains('PARTIAL').sum())
            
            # Анализ причин выхода
            if 'exit_reason' in df_log.columns:
                exit_stats = df_log['exit_reason'].value_counts().to_dict()
                # Преобразуем в serializable формат
                stats['exit_stats'] = {str(k): int(v) for k, v in exit_stats.items()}
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Ошибка анализа лога: {e}")
            return {}
    
    def run_comprehensive_test(self, n_periods=8, test_type="sequential"):
        """Комплексное тестирование на разных периодах"""
        print(f"\n🧪 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
        print(f"   Символ: {self.symbol}")
        print(f"   Тип теста: {test_type}")
        print(f"   Периодов: {n_periods}")
        print("-" * 60)
        
        # Создаем тестовые периоды
        if test_type == "regimes":
            test_periods = self.create_market_regimes(n_periods=n_periods)
        else:
            test_periods = self.create_test_periods(n_periods=n_periods)
        
        print(f"📊 Создано тестовых периодов: {len(test_periods)}")
        
        # Запускаем тесты на каждом периоде
        all_results = []
        
        for i, period_item in enumerate(test_periods):
            period_info = period_item['info']
            print(f"\n🔍 Тестирование периода {i+1}/{len(test_periods)}...")
            
            # Информация о периоде
            if 'regime' in period_info:
                print(f"   Режим: {period_info['regime']}, Тренд: {period_info['trend']}")
                print(f"   Волатильность: {period_info['volatility']:.6f}, Доходность: {period_info['return_pct']:.2f}%")
            else:
                print(f"   Длина: {period_info['length']} баров")
                print(f"   Цена: ${period_info['avg_price']:.2f}, Изменение: {period_info['price_change_pct']:.2f}%")
            
            # Запуск теста
            result = self.run_test_on_period(period_item, f"period_{i+1}")
            all_results.append(result)
            
            print(f"   Результат: {result['trades']} сделок, Награда: {result['reward']:.2f}")
            
            # Детальная статистика
            if result['detailed_stats']:
                stats = result['detailed_stats']
                if 'avg_pnl' in stats:
                    print(f"   Средний PnL: {stats['avg_pnl']:.4f}%, Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
        
        # Анализ результатов по всем периодам
        print("\n" + "="*60)
        print("📊 КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        self.analyze_comprehensive_results(all_results, test_type)
        
        return all_results
    
    def analyze_comprehensive_results(self, all_results, test_type):
        """Анализ результатов по всем периодам"""
        if not all_results:
            print("⚠️  Нет результатов для анализа")
            return
        
        # Собираем сводную статистику
        total_trades = sum(r['trades'] for r in all_results)
        total_reward = sum(r['reward'] for r in all_results)
        total_steps = sum(r['steps'] for r in all_results)
        
        # Периоды со сделками
        periods_with_trades = [r for r in all_results if r['trades'] > 0]
        periods_without_trades = [r for r in all_results if r['trades'] == 0]
        
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Всего периодов: {len(all_results)}")
        print(f"   Периодов со сделками: {len(periods_with_trades)}")
        print(f"   Периодов без сделок: {len(periods_without_trades)}")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Всего шагов: {total_steps}")
        print(f"   Средняя частота: {(total_trades/total_steps*1000):.1f} сделок/1000 шагов")
        print(f"   Общая награда: {total_reward:.2f}")
        
        # Детальный анализ по периодам со сделками
        if periods_with_trades:
            print(f"\n📊 АНАЛИЗ ПО ПЕРИОДАМ СО СДЕЛКАМИ:")
            
            # Группировка по типам периодов если есть
            if test_type == "regimes":
                self.analyze_by_regime(periods_with_trades)
            
            # Статистика PnL по всем сделкам
            all_pnls = []
            all_win_rates = []
            
            for result in periods_with_trades:
                if result['detailed_stats']:
                    stats = result['detailed_stats']
                    if 'avg_pnl' in stats:
                        all_pnls.append(stats['avg_pnl'])
                    if 'win_rate' in stats:
                        all_win_rates.append(stats['win_rate'])
            
            if all_pnls:
                print(f"\n💰 СТАТИСТИКА PnL:")
                print(f"   Средний PnL по периодам: {np.mean(all_pnls):.4f}%")
                print(f"   Медианный PnL: {np.median(all_pnls):.4f}%")
                print(f"   Стандартное отклонение: {np.std(all_pnls):.4f}%")
                
                # Profit Factor (упрощенный)
                positive_pnls = [p for p in all_pnls if p > 0]
                negative_pnls = [p for p in all_pnls if p < 0]
                
                if positive_pnls and negative_pnls:
                    avg_win = np.mean(positive_pnls)
                    avg_loss = abs(np.mean(negative_pnls))
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                    print(f"   Profit Factor: {profit_factor:.2f}")
            
            if all_win_rates:
                print(f"\n🎯 СТАТИСТИКА WIN RATE:")
                print(f"   Средний Win Rate: {np.mean(all_win_rates)*100:.1f}%")
                print(f"   Медианный Win Rate: {np.median(all_win_rates)*100:.1f}%")
        
        # Анализ периодов без сделок
        if periods_without_trades:
            print(f"\n⚠️  ПЕРИОДЫ БЕЗ СДЕЛОК ({len(periods_without_trades)}):")
            
            for result in periods_without_trades[:3]:  # Показываем первые 3
                period_info = result['period_info']
                if 'regime' in period_info:
                    print(f"   Период {period_info['id']}: {period_info['regime']}, {period_info['trend']}")
                else:
                    print(f"   Период {period_info['id']}: шаги {result['steps']}, награда {result['reward']:.2f}")
            
            if len(periods_without_trades) > 3:
                print(f"   ... и еще {len(periods_without_trades) - 3} периодов")
        
        # Сохранение результатов
        self.save_test_results(all_results, test_type)
    
    def analyze_by_regime(self, results):
        """Анализ результатов по рыночным режимам"""
        regime_stats = {}
        
        for result in results:
            period_info = result['period_info']
            if 'regime' not in period_info:
                continue
            
            regime = period_info['regime']
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'count': 0,
                    'total_trades': 0,
                    'total_reward': 0,
                    'pnls': [],
                    'win_rates': []
                }
            
            stats = regime_stats[regime]
            stats['count'] += 1
            stats['total_trades'] += result['trades']
            stats['total_reward'] += result['reward']
            
            if result['detailed_stats']:
                det_stats = result['detailed_stats']
                if 'avg_pnl' in det_stats:
                    stats['pnls'].append(det_stats['avg_pnl'])
                if 'win_rate' in det_stats:
                    stats['win_rates'].append(det_stats['win_rate'])
        
        if regime_stats:
            print(f"\n📊 РЕЗУЛЬТАТЫ ПО РЫНОЧНЫМ РЕЖИМАМ:")
            
            for regime, stats in regime_stats.items():
                if stats['count'] > 0:
                    avg_trades = stats['total_trades'] / stats['count']
                    avg_reward = stats['total_reward'] / stats['count']
                    
                    print(f"\n   {regime}:")
                    print(f"      Периодов: {stats['count']}")
                    print(f"      Среднее сделок: {avg_trades:.1f}")
                    print(f"      Средняя награда: {avg_reward:.2f}")
                    
                    if stats['pnls']:
                        avg_pnl = np.mean(stats['pnls'])
                        print(f"      Средний PnL: {avg_pnl:.4f}%")
                    
                    if stats['win_rates']:
                        avg_win_rate = np.mean(stats['win_rates'])
                        print(f"      Средний Win Rate: {avg_win_rate*100:.1f}%")
    
    def save_test_results(self, all_results, test_type):
        """Сохранение результатов тестирования"""
        results_dir = f"./results/v16_historical/{self.symbol}/"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Собираем данные для сохранения (только serializable данные)
        summary_data = {
            'symbol': self.symbol,
            'test_type': test_type,
            'timestamp': timestamp,
            'initial_balance': self.initial_balance,
            'total_periods': len(all_results),
            'summary': {},
            'detailed_results': []
        }
        
        # Сводная статистика
        periods_with_trades = [r for r in all_results if r['trades'] > 0]
        
        if periods_with_trades:
            total_trades = sum(r['trades'] for r in periods_with_trades)
            total_reward = sum(r['reward'] for r in periods_with_trades)
            total_steps = sum(r['steps'] for r in periods_with_trades)
            
            summary_data['summary'] = {
                'periods_with_trades': len(periods_with_trades),
                'periods_without_trades': len(all_results) - len(periods_with_trades),
                'total_trades': int(total_trades),
                'total_reward': float(total_reward),
                'avg_trades_per_period': float(total_trades / len(periods_with_trades)),
                'avg_reward_per_period': float(total_reward / len(periods_with_trades)),
                'trades_per_1000_steps': float((total_trades / total_steps * 1000) if total_steps > 0 else 0)
            }
        
        # Детальные результаты (только serializable данные)
        for result in all_results:
            detailed = {
                'period_id': result['period_id'],
                'period_info': result['period_info'],  # Уже serializable
                'steps': int(result['steps']),
                'reward': float(result['reward']),
                'trades': int(result['trades']),
                'detailed_stats': result.get('detailed_stats', {}),
                'log_file': str(result.get('log_file', ''))
            }
            summary_data['detailed_results'].append(detailed)
        
        # Сохраняем в JSON
        results_file = f"{results_dir}historical_test_{test_type}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)  # Используем default=str для безопасности
        
        print(f"\n📝 Результаты сохранены в: {results_file}")
        
        # Также сохраняем краткий отчет в CSV
        self.save_csv_report(summary_data['detailed_results'], results_dir, timestamp, test_type)
    
    def save_csv_report(self, results, results_dir, timestamp, test_type):
        """Сохранение краткого отчета в CSV"""
        csv_file = f"{results_dir}summary_{test_type}_{timestamp}.csv"
        
        rows = []
        for result in results:
            row = {
                'period_id': result['period_id'],
                'steps': result['steps'],
                'reward': result['reward'],
                'trades': result['trades']
            }
            
            # Добавляем информацию о периоде
            period_info = result['period_info']
            if 'regime' in period_info:
                row['regime'] = period_info['regime']
                row['trend'] = period_info['trend']
                row['volatility'] = period_info.get('volatility', 0)
                row['return_pct'] = period_info.get('return_pct', 0)
            else:
                row['price_change_pct'] = period_info.get('price_change_pct', 0)
                row['volatility'] = period_info.get('volatility', 0)
            
            # Добавляем статистику PnL
            if result['detailed_stats']:
                stats = result['detailed_stats']
                row['avg_pnl'] = stats.get('avg_pnl', 0)
                row['win_rate'] = stats.get('win_rate', 0)
                row['total_pnl'] = stats.get('total_pnl', 0)
                row['profitable_trades'] = stats.get('profitable_trades', 0)
                row['loss_trades'] = stats.get('loss_trades', 0)
            
            rows.append(row)
        
        if rows:
            df_report = pd.DataFrame(rows)
            df_report.to_csv(csv_file, index=False)
            print(f"📊 CSV отчет сохранен: {csv_file}")


def train_and_test_historical():
    """Обучение и историческое тестирование"""
    print("\n" + "="*60)
    print("🚀 ОБУЧЕНИЕ С ИСТОРИЧЕСКИМ ТЕСТИРОВАНИЕМ")
    print("="*60)
    
    # Конфигурация
    DATA_PATH = "data/btc_15m.csv"
    SYMBOL = "BTC"
    MODEL_DIR = f"./models/v16_historical_{SYMBOL.lower()}/"
    LOG_DIR = f"./logs/v16_historical_{SYMBOL.lower()}/"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Загрузка данных
    print("\n📥 Загрузка данных...")
    
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            print(f"✅ Загружено {len(df)} строк данных")
            
            # Переименовываем колонки если нужно
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)
        else:
            print(f"⚠️  Файл {DATA_PATH} не найден, создаю тестовые данные")
            # Создание тестовых данных
            np.random.seed(42)
            n_rows = 5000
            
            # Создаем более длинные данные для исторического тестирования
            time = np.arange(n_rows)
            close = 50000 + np.sin(time * 0.01) * 5000 + np.random.randn(n_rows) * 1000
            
            df = pd.DataFrame({
                'open': close * np.random.uniform(0.995, 1.005, n_rows),
                'high': close * np.random.uniform(1.005, 1.015, n_rows),
                'low': close * np.random.uniform(0.985, 0.995, n_rows),
                'close': close,
                'volume': np.random.randint(1000, 10000, n_rows)
            })
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # 2. Подготовка признаков
    print("\n🎯 Подготовка признаков...")
    
    # Базовые признаки
    obs_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Добавляем ATR если нет
    if 'atr' not in df.columns:
        df['atr'] = (df['high'] - df['low']).rolling(14).mean().fillna(df['close'].iloc[0] * 0.01)
        obs_cols.append('atr')
    
    # Ищем дополнительные признаки
    possible_features = ['rsi', 'macd', 'bb_width', 'vwap', 'trend', 'momentum']
    for feature in possible_features:
        if feature in df.columns:
            obs_cols.append(feature)
    
    print(f"📊 Используется {len(obs_cols)} признаков")
    
    # 3. Разделение данных на обучение/тест
    print("\n📊 Разделение данных...")
    
    # Используем первые 70% для обучения, остальные 30% для исторического тестирования
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"   Обучение: {len(train_df):,} строк")
    print(f"   Тестирование: {len(test_df):,} строк")
    
    # 4. Обучение модели
    print("\n🧠 Обучение модели...")
    
    def make_train_env():
        log_file = f"{LOG_DIR}train_log.csv"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("step,type,entry,sl_initial,sl_current,tp_levels,exit,pnl_percent,net_worth,exit_reason,duration,trailing,tp_closed,partial_closes\n")
        
        env = CryptoTradingEnvV16_RR2_Enhanced(
            df=train_df,
            obs_cols=obs_cols,
            initial_balance=10000,
            commission=0.001,
            slippage=0.0005,
            rr_ratio=2.0,
            atr_multiplier=1.4,
            log_file=log_file,
            training_mode="rr2_enhanced"
        )
        return env
    
    try:
        train_env = DummyVecEnv([make_train_env])
        
        n_features = len(obs_cols) + 11
        hidden_size = min(256, max(128, n_features * 2))
        
        policy_kwargs = dict(
            net_arch=[dict(
                pi=[hidden_size, hidden_size//2],
                vf=[hidden_size, hidden_size//2]
            )]
        )
        
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=2e-4,
            ent_coef=0.03,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=f"{LOG_DIR}tensorboard/"
        )
        
        # Короткое обучение для демонстрации
        total_steps = 30000
        
        print(f"\n🎯 ОБУЧЕНИЕ ({total_steps:,} шагов)")
        print("="*40)
        
        model.learn(
            total_timesteps=total_steps,
            log_interval=10,
            progress_bar=True,
            tb_log_name=f"PPO_{SYMBOL}_historical"
        )
        
        print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        
        # Сохранение модели
        model.save(f"{MODEL_DIR}ppo_model")
        print(f"💾 Модель сохранена: {MODEL_DIR}")
        
        train_env.close()
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return
    
    # 5. Историческое тестирование
    print("\n" + "="*60)
    print("🧪 ИСТОРИЧЕСКОЕ ТЕСТИРОВАНИЕ НА РАЗНЫХ ПЕРИОДАХ")
    print("="*60)
    
    # Создаем тестер
    tester = HistoricalTester(
        model=model,
        df=test_df,
        obs_cols=obs_cols,
        symbol=SYMBOL,
        initial_balance=10000
    )
    
    # Тест 1: Последовательные периоды
    print("\n1. 📊 ТЕСТ НА ПОСЛЕДОВАТЕЛЬНЫХ ПЕРИОДАХ")
    print("-" * 40)
    
    results_sequential = tester.run_comprehensive_test(
        n_periods=6,
        test_type="sequential"
    )
    
    # Тест 2: Разные рыночные режимы
    print("\n2. 📊 ТЕСТ НА РАЗНЫХ РЫНОЧНЫХ РЕЖИМАХ")
    print("-" * 40)
    
    results_regimes = tester.run_comprehensive_test(
        n_periods=6,
        test_type="regimes"
    )
    
    print("\n" + "="*60)
    print("🎉 ИСТОРИЧЕСКОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*60)


if __name__ == "__main__":
    train_and_test_historical()