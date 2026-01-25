# monitor_progress_fixed.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob


class TrainingMonitor:
    def __init__(self, base_path="./"):
        self.base_path = base_path
        
    def find_log_files(self):
        """Поиск всех файлов логов"""
        log_files = []
        
        # Ищем в стандартных папках
        search_paths = [
            "./logs/",
            "./ppo_logs/",
            "./models/",
            "./",
            "logs/",
            "ppo_logs/"
        ]
        
        patterns = [
            "*trade*.csv",
            "*log*.csv",
            "v16*.csv",
            "train*.csv",
            "test*.csv"
        ]
        
        for path in search_paths:
            for pattern in patterns:
                full_pattern = os.path.join(path, pattern)
                try:
                    found_files = glob.glob(full_pattern)
                    for file in found_files:
                        if os.path.exists(file) and os.path.getsize(file) > 0:
                            log_files.append(file)
                except:
                    continue
        
        return list(set(log_files))  # Убираем дубликаты
    
    def analyze_all_logs(self):
        """Анализ всех найденных логов"""
        print("\n" + "="*60)
        print("🔍 ПОИСК И АНАЛИЗ ЛОГОВ ОБУЧЕНИЯ")
        print("="*60)
        
        # Находим все файлы логов
        log_files = self.find_log_files()
        
        if not log_files:
            print("❌ Не найдено ни одного файла логов!")
            print("\n📂 Поиск в папках:")
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith('.csv') and ('log' in file.lower() or 'trade' in file.lower()):
                        full_path = os.path.join(root, file)
                        print(f"   • {full_path}")
            return
        
        print(f"✅ Найдено файлов логов: {len(log_files)}")
        
        for i, log_file in enumerate(log_files, 1):
            print(f"\n{'='*40}")
            print(f"📊 ФАЙЛ {i}: {log_file}")
            print(f"{'='*40}")
            
            try:
                self._analyze_single_log(log_file)
            except Exception as e:
                print(f"⚠️  Ошибка анализа {log_file}: {e}")
    
    def _analyze_single_log(self, log_file):
        """Анализ одного файла логов"""
        try:
            # Читаем файл
            df = pd.read_csv(log_file)
            print(f"   Строк: {len(df):,}")
            print(f"   Колонок: {len(df.columns)}")
            print(f"   Колонки: {list(df.columns)}")
            
            # Анализ содержимого
            if len(df) > 0:
                print(f"\n   📈 СТАТИСТИКА:")
                print(f"      Первая запись: строка {df.iloc[0]['step'] if 'step' in df.columns else 'N/A'}")
                print(f"      Последняя запись: строка {df.iloc[-1]['step'] if 'step' in df.columns else 'N/A'}")
                
                # Анализ сделок
                if 'type' in df.columns:
                    trade_types = df['type'].value_counts()
                    print(f"      Типы сделок: {dict(trade_types)}")
                
                if 'exit_reason' in df.columns:
                    exit_stats = df['exit_reason'].value_counts()
                    print(f"      Причины выхода: {dict(exit_stats)}")
                    
                    # Расчет TP Rate
                    tp_count = sum(1 for x in df['exit_reason'] if 'TP' in str(x))
                    total_trades = len(df)
                    if total_trades > 0:
                        tp_rate = tp_count / total_trades * 100
                        print(f"      TP Rate: {tp_rate:.1f}% ({tp_count}/{total_trades})")
                
                if 'pnl_percent' in df.columns:
                    # Извлекаем числовые значения из процентов
                    pnl_values = []
                    for val in df['pnl_percent']:
                        try:
                            if isinstance(val, str):
                                num = float(val.replace('%', '').replace(' ', ''))
                            else:
                                num = float(val)
                            pnl_values.append(num)
                        except:
                            continue
                    
                    if pnl_values:
                        avg_pnl = np.mean(pnl_values)
                        win_rate = sum(1 for x in pnl_values if x > 0) / len(pnl_values) * 100
                        print(f"      Средний PnL: {avg_pnl:.2f}%")
                        print(f"      Win Rate: {win_rate:.1f}%")
                
                # Показываем последние 5 записей
                print(f"\n   📋 ПОСЛЕДНИЕ 5 ЗАПИСЕЙ:")
                print(df.tail(5).to_string())
                
        except Exception as e:
            print(f"   ❌ Ошибка чтения файла: {e}")
    
    def check_tensorboard_logs(self):
        """Проверка логов TensorBoard"""
        print(f"\n{'='*60}")
        print("📊 ПРОВЕРКА ЛОГОВ TENSORBOARD")
        print('='*60)
        
        tensorboard_paths = [
            "./ppo_logs/",
            "./ppo_logs/v16_rr2_enhanced_fixed/",
            "./tensorboard_logs/",
            "ppo_logs/",
            "tensorboard_logs/"
        ]
        
        for path in tensorboard_paths:
            if os.path.exists(path):
                print(f"✅ Найдена папка TensorBoard: {path}")
                
                # Считаем события
                event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
                if event_files:
                    print(f"   Найдено файлов событий: {len(event_files)}")
                    for file in event_files[:3]:  # Показываем первые 3
                        file_size = os.path.getsize(file) / 1024 / 1024  # MB
                        print(f"   • {os.path.basename(file)} ({file_size:.1f} MB)")
                else:
                    print("   ❌ Файлы событий не найдены")
                
                # Считаем подпапки
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    print(f"   Подпапки: {subdirs}")
    
    def check_model_files(self):
        """Проверка сохраненных моделей"""
        print(f"\n{'='*60}")
        print("🤖 ПРОВЕРКА СОХРАНЕННЫХ МОДЕЛЕЙ")
        print('='*60)
        
        model_paths = [
            "./models/",
            "./models/v16_rr2_enhanced_fixed/",
            "models/",
            "models/v16_rr2_enhanced_fixed/"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"✅ Найдена папка моделей: {path}")
                
                # Ищем файлы моделей
                model_files = []
                for ext in ['.zip', '.pkl', '.pt', '.model']:
                    model_files.extend(glob.glob(os.path.join(path, f"*{ext}")))
                
                if model_files:
                    print(f"   Найдено файлов моделей: {len(model_files)}")
                    for file in model_files[:5]:  # Показываем первые 5
                        file_size = os.path.getsize(file) / 1024 / 1024  # MB
                        file_name = os.path.basename(file)
                        print(f"   • {file_name} ({file_size:.1f} MB)")
                else:
                    print("   ❌ Файлы моделей не найдены")
                
                # Ищем JSON файлы с информацией
                json_files = glob.glob(os.path.join(path, "*.json"))
                if json_files:
                    print(f"   Найдено JSON файлов: {len(json_files)}")
                    for file in json_files:
                        try:
                            with open(file, 'r') as f:
                                data = json.load(f)
                            print(f"   • {os.path.basename(file)}: {len(data)} записей")
                        except:
                            print(f"   • {os.path.basename(file)} (ошибка чтения)")
    
    def create_sample_log(self):
        """Создание тестового файла логов для проверки"""
        print(f"\n{'='*60}")
        print("📝 СОЗДАНИЕ ТЕСТОВОГО ФАЙЛА ЛОГОВ")
        print('='*60)
        
        sample_data = {
            'step': list(range(100)),
            'type': ['LONG' if i % 2 == 0 else 'SHORT' for i in range(100)],
            'entry': [100 + i * 0.1 for i in range(100)],
            'exit': [101 + i * 0.1 for i in range(100)],
            'pnl_percent': [np.random.uniform(-2, 5) for _ in range(100)],
            'exit_reason': ['TP_FULL' if i % 3 == 0 else 'SL_INITIAL' if i % 3 == 1 else 'MANUAL' for i in range(100)],
            'duration': [np.random.randint(5, 50) for _ in range(100)]
        }
        
        df = pd.DataFrame(sample_data)
        test_file = "./logs/test_sample_log.csv"
        
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        df.to_csv(test_file, index=False)
        
        print(f"✅ Тестовый файл создан: {test_file}")
        print(f"   Строк: {len(df)}")
        print(f"   Колонки: {list(df.columns)}")
        
        return test_file


def main():
    """Главная функция"""
    print("\n" + "="*60)
    print("📊 МОНИТОРИНГ ПРОГРЕССА ОБУЧЕНИЯ v16_rr2_enhanced")
    print("="*60)
    
    monitor = TrainingMonitor()
    
    while True:
        print("\n" + "="*60)
        print("МЕНЮ МОНИТОРИНГА")
        print("="*60)
        print("1. 🔍 Найти и проанализировать все логи")
        print("2. 📊 Проверить логи TensorBoard")
        print("3. 🤖 Проверить сохраненные модели")
        print("4. 📝 Создать тестовый файл логов")
        print("5. 📂 Показать структуру папок")
        print("6. 🚪 Выход")
        print("="*60)
        
        choice = input("Выберите действие (1-6): ").strip()
        
        if choice == "1":
            monitor.analyze_all_logs()
        elif choice == "2":
            monitor.check_tensorboard_logs()
        elif choice == "3":
            monitor.check_model_files()
        elif choice == "4":
            monitor.create_sample_log()
        elif choice == "5":
            print("\n📂 СТРУКТУРА ПАПОК:")
            for root, dirs, files in os.walk("."):
                level = root.replace(".", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Показываем только первые 5 файлов
                    if file.endswith(('.csv', '.json', '.zip', '.log')):
                        print(f"{subindent}{file}")
        elif choice == "6":
            print("👋 Выход...")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    # Создаем папку logs если её нет
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./ppo_logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    main()