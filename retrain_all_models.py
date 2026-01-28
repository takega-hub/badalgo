"""
Модуль для автоматического переобучения всех ML моделей.
Обновляет все модели в директории ml_models с новыми параметрами (min_risk_reward_ratio=2.0).

Использование:
    python retrain_all_models.py                    # Переобучить все модели
    python retrain_all_models.py --symbol BTCUSDT    # Переобучить только для BTCUSDT
    python retrain_all_models.py --type quad_ensemble # Переобучить только QuadEnsemble
    python retrain_all_models.py --days 180          # Использовать 180 дней данных
"""
import warnings
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

# Настраиваем кодировку для Windows
if sys.platform == 'win32':
    try:
        # Пытаемся установить UTF-8 для консоли
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        # Если не получилось, используем безопасный вывод без эмодзи
        pass

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings

# Функция для безопасного вывода (заменяет эмодзи на текстовые метки для Windows)
def safe_print(*args, **kwargs):
    """Безопасный print, который заменяет эмодзи на текстовые метки."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Заменяем эмодзи на текстовые метки
        text = ' '.join(str(arg) for arg in args)
        text = text.replace('🚀', '[START]')
        text = text.replace('📊', '[INFO]')
        text = text.replace('✅', '[OK]')
        text = text.replace('❌', '[ERROR]')
        text = text.replace('⏳', '[WAIT]')
        text = text.replace('⏱️', '[TIME]')
        text = text.replace('🔄', '[RETRAIN]')
        print(text, **kwargs)


class ModelRetrainer:
    """Класс для автоматического переобучения всех моделей."""
    
    def __init__(self, models_dir: Path = None):
        if models_dir is None:
            models_dir = Path(__file__).parent / "ml_models"
        self.models_dir = models_dir
        self.settings = load_settings()
        
        # Маппинг типов моделей на скрипты обучения
        self.model_type_scripts = {
            "rf": "retrain_ml_optimized.py",
            "xgb": "retrain_ml_optimized.py",
            "ensemble": "retrain_ml_optimized.py",
            "triple_ensemble": "retrain_ml_optimized.py",
            "quad_ensemble": "train_quad_ensemble.py",
            "lstm": "train_lstm_model.py",
        }
        
        # Символы для обучения
        self.available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.default_interval = "15"
    
    def scan_models(self, symbol_filter: Optional[str] = None, 
                    type_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Сканирует директорию ml_models и возвращает список моделей для переобучения.
        
        Args:
            symbol_filter: Фильтр по символу (например, "BTCUSDT")
            type_filter: Фильтр по типу модели (например, "quad_ensemble")
        
        Returns:
            Список словарей с информацией о моделях
        """
        models = []
        
        if not self.models_dir.exists():
            safe_print(f"❌ Директория {self.models_dir} не существует!")
            return models
        
        # Сканируем все .pkl файлы
        for model_file in self.models_dir.glob("*.pkl"):
            filename = model_file.name
            
            # Парсим имя файла: {model_type}_{SYMBOL}_{INTERVAL}.pkl
            # Примеры: quad_ensemble_BTCUSDT_15.pkl, rf_ETHUSDT_15.pkl
            parts = filename.replace(".pkl", "").split("_")
            
            if len(parts) < 3:
                print(f"⚠️  Пропускаем файл с нестандартным форматом: {filename}")
                continue
            
            # Определяем тип модели
            if parts[0] in ["triple", "quad"]:
                model_type = f"{parts[0]}_{parts[1]}"  # triple_ensemble или quad_ensemble
                symbol = parts[2]
                interval = parts[3] if len(parts) > 3 else self.default_interval
            else:
                model_type = parts[0]  # rf, xgb, ensemble, lstm
                symbol = parts[1]
                interval = parts[2] if len(parts) > 2 else self.default_interval
            
            # Применяем фильтры
            if symbol_filter and symbol != symbol_filter:
                continue
            if type_filter and model_type != type_filter:
                continue
            
            models.append({
                "filename": filename,
                "path": str(model_file),
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
            })
        
        return models
    
    def get_training_command(self, model_info: Dict[str, str], days: int = 180) -> List[str]:
        """
        Возвращает команду для обучения модели.
        
        Args:
            model_info: Информация о модели
            days: Количество дней исторических данных
        
        Returns:
            Список аргументов для subprocess
        """
        model_type = model_info["model_type"]
        symbol = model_info["symbol"]
        interval = model_info["interval"]
        
        # Определяем скрипт обучения
        script_name = self.model_type_scripts.get(model_type)
        if not script_name:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Скрипт обучения не найден: {script_path}")
        
        # Формируем команду
        cmd = [sys.executable, str(script_path), "--symbol", symbol, "--days", str(days)]
        
        # Для разных типов моделей добавляем специфичные параметры
        if model_type == "quad_ensemble":
            cmd.extend(["--interval", f"{interval}m"])
        elif model_type == "lstm":
            cmd.extend(["--interval", f"{interval}m"])
        # Для retrain_ml_optimized.py интервал не передается через аргументы,
        # он жестко задан в скрипте как "15"
        
        return cmd
    
    def retrain_model(self, model_info: Dict[str, str], days: int = 180, 
                     dry_run: bool = False) -> Tuple[bool, str]:
        """
        Переобучает одну модель.
        
        Args:
            model_info: Информация о модели
            days: Количество дней исторических данных
            dry_run: Если True, только показывает команду без выполнения
        
        Returns:
            (success, message) - результат выполнения
        """
        model_type = model_info["model_type"]
        symbol = model_info["symbol"]
        filename = model_info["filename"]
        
        safe_print(f"\n{'='*80}")
        safe_print(f"🔄 Переобучение: {filename}")
        safe_print(f"   Тип: {model_type}")
        safe_print(f"   Символ: {symbol}")
        safe_print(f"   Интервал: {model_info['interval']}")
        safe_print(f"{'='*80}")
        
        try:
            cmd = self.get_training_command(model_info, days)
            
            if dry_run:
                safe_print(f"   [DRY RUN] Команда: {' '.join(cmd)}")
                return True, "Dry run completed"
            
            # Запускаем обучение
            start_time = time.time()
            
            # Устанавливаем UTF-8 кодировку для Windows
            env = os.environ.copy()
            if sys.platform == 'win32':
                env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Заменяем нечитаемые символы вместо ошибки
                cwd=Path(__file__).parent,
                env=env
            )
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                safe_print(f"✅ Успешно переобучена за {elapsed_time:.1f} сек")
                return True, f"Successfully retrained in {elapsed_time:.1f}s"
            else:
                error_msg = result.stderr or result.stdout
                safe_print(f"❌ Ошибка при переобучении:")
                safe_print(f"   {error_msg[:500]}")
                return False, f"Error: {error_msg[:200]}"
        
        except Exception as e:
            error_msg = str(e)
            safe_print(f"❌ Исключение при переобучении: {error_msg}")
            return False, f"Exception: {error_msg}"
    
    def retrain_all(self, symbol_filter: Optional[str] = None,
                   type_filter: Optional[str] = None,
                   days: int = 180,
                   dry_run: bool = False,
                   skip_existing: bool = False) -> Dict[str, any]:
        """
        Переобучает все модели.
        
        Args:
            symbol_filter: Фильтр по символу
            type_filter: Фильтр по типу модели
            days: Количество дней исторических данных
            dry_run: Если True, только показывает команды без выполнения
            skip_existing: Если True, пропускает модели, которые уже переобучены сегодня
        
        Returns:
            Словарь с результатами
        """
        safe_print("="*80)
        safe_print("🚀 АВТОМАТИЧЕСКОЕ ПЕРЕОБУЧЕНИЕ ВСЕХ ML МОДЕЛЕЙ")
        safe_print("="*80)
        print(f"Директория моделей: {self.models_dir}")
        print(f"Дни данных: {days}")
        if symbol_filter:
            print(f"Фильтр по символу: {symbol_filter}")
        if type_filter:
            print(f"Фильтр по типу: {type_filter}")
        if dry_run:
            print("⚠️  DRY RUN MODE - команды не будут выполнены")
        print("="*80)
        
        # Сканируем модели
        models = self.scan_models(symbol_filter=symbol_filter, type_filter=type_filter)
        
        if not models:
            safe_print("❌ Модели не найдены!")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "results": []
            }
        
        safe_print(f"\n📊 Найдено моделей для переобучения: {len(models)}")
        
        # Группируем по типу для лучшей организации
        models_by_type = {}
        for model in models:
            model_type = model["model_type"]
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model)
        
        print(f"\n📋 Распределение по типам:")
        for model_type, type_models in models_by_type.items():
            print(f"   {model_type}: {len(type_models)} моделей")
        
        # Переобучаем модели
        results = {
            "total": len(models),
            "success": 0,
            "failed": 0,
            "results": []
        }
        
        start_time = time.time()
        
        for i, model in enumerate(models, 1):
            safe_print(f"\n[{i}/{len(models)}] Обработка: {model['filename']}")
            
            # Проверяем, нужно ли пропустить (если skip_existing=True)
            if skip_existing:
                try:
                    from bot.ml.model_trainer import ModelTrainer
                    trainer = ModelTrainer()
                    metadata = trainer.load_model_metadata(model["path"])
                    if metadata and metadata.get("trained_at"):
                        trained_date = datetime.fromisoformat(metadata["trained_at"])
                        if (datetime.now() - trained_date).days == 0:
                            print(f"   ⏭️  Пропущена (уже переобучена сегодня)")
                            results["results"].append({
                                "model": model["filename"],
                                "success": True,
                                "skipped": True,
                                "message": "Already retrained today"
                            })
                            continue
                except:
                    pass  # Если не удалось проверить, продолжаем
        
            success, message = self.retrain_model(model, days=days, dry_run=dry_run)
            
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
            
            results["results"].append({
                "model": model["filename"],
                "success": success,
                "message": message
            })
        
        total_time = time.time() - start_time
        
        # Выводим итоги
        print("\n" + "="*80)
        safe_print("📊 ИТОГИ ПЕРЕОБУЧЕНИЯ")
        safe_print("="*80)
        safe_print(f"Всего моделей: {results['total']}")
        safe_print(f"✅ Успешно: {results['success']}")
        safe_print(f"❌ Ошибок: {results['failed']}")
        safe_print(f"⏱️  Время: {total_time/60:.1f} минут")
        print("="*80)
        
        # Выводим список неудачных
        if results['failed'] > 0:
            safe_print("\n❌ Модели с ошибками:")
            for result in results["results"]:
                if not result["success"]:
                    safe_print(f"   - {result['model']}: {result['message']}")
        
        return results


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description='Автоматическое переобучение всех ML моделей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python retrain_all_models.py                          # Переобучить все модели
  python retrain_all_models.py --symbol BTCUSDT         # Только BTCUSDT
  python retrain_all_models.py --type quad_ensemble     # Только QuadEnsemble
  python retrain_all_models.py --days 180               # Использовать 180 дней
  python retrain_all_models.py --dry-run                 # Показать команды без выполнения
  python retrain_all_models.py --skip-existing          # Пропустить модели, переобученные сегодня
        """
    )
    
    parser.add_argument('--symbol', type=str, default=None,
                       help='Фильтр по символу (BTCUSDT, ETHUSDT, SOLUSDT)')
    parser.add_argument('--type', type=str, default=None,
                       help='Фильтр по типу модели (rf, xgb, ensemble, triple_ensemble, quad_ensemble, lstm)')
    parser.add_argument('--days', type=int, default=180,
                       help='Количество дней исторических данных (по умолчанию: 180)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Показать команды без выполнения')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Пропустить модели, переобученные сегодня')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Путь к директории с моделями (по умолчанию: ml_models)')
    
    args = parser.parse_args()
    
    # Создаем retrainer
    models_dir = Path(args.models_dir) if args.models_dir else None
    retrainer = ModelRetrainer(models_dir=models_dir)
    
    # Запускаем переобучение
    results = retrainer.retrain_all(
        symbol_filter=args.symbol,
        type_filter=args.type,
        days=args.days,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing
    )
    
    # Возвращаем код выхода
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
