# test_best_models.py
"""
Тестирование лучших моделей на исторических данных.
"""
import subprocess
import time
from pathlib import Path
import pandas as pd

def test_model(model_path, symbol, days=7):
    """Запускает тестирование модели."""
    print(f"\n🧪 Тестирование {Path(model_path).name} для {symbol}...")
    
    cmd = [
        "python", "test_ml_strategy.py",
        "--symbol", symbol,
        "--model", model_path,
        "--days", str(days)
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Тест завершен за {elapsed_time:.1f} сек")
            # Парсим результаты из вывода
            output = result.stdout
            return True, output
        else:
            print(f"❌ Ошибка тестирования")
            return False, result.stderr
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False, str(e)

def main():
    print("=" * 80)
    print("🧪 ТЕСТИРОВАНИЕ ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 80)
    
    models_dir = Path("ml_models")
    
    # Выбираем модели для тестирования
    test_candidates = []
    
    # Ищем модели по приоритету:
    # 1. quad_ensemble (самая сложная)
    # 2. triple_ensemble 
    # 3. ensemble
    # 4. xgb
    # 5. rf
    
    priority_patterns = [
        "*quad_ensemble*_opt.pkl",
        "*quad_ensemble_*.pkl",
        "*triple_ensemble*_opt.pkl", 
        "*triple_ensemble_*.pkl",
        "*ensemble*_opt.pkl",
        "*ensemble_*.pkl",
        "*xgb*_opt.pkl",
        "*xgb_*.pkl",
        "*rf*_opt.pkl",
        "*rf_*.pkl"
    ]
    
    for pattern in priority_patterns:
        models = list(models_dir.glob(pattern))
        for model in models:
            # Извлекаем символ из имени файла
            parts = model.name.replace('.pkl', '').split('_')
            if len(parts) >= 2:
                symbol = parts[1] if parts[0] not in ['triple', 'quad'] else parts[2]
                if (model, symbol) not in test_candidates:
                    test_candidates.append((model, symbol))
    
    # Ограничиваем количество тестов
    max_tests = 5
    test_candidates = test_candidates[:max_tests]
    
    if not test_candidates:
        print("❌ Нет моделей для тестирования")
        return
    
    print(f"📊 Будут протестированы {len(test_candidates)} моделей:")
    for model, symbol in test_candidates:
        print(f"   • {model.name} ({symbol})")
    
    print(f"\n{'='*80}")
    print("НАЧАЛО ТЕСТИРОВАНИЯ")
    print(f"{'='*80}")
    
    results = []
    
    for i, (model_path, symbol) in enumerate(test_candidates, 1):
        print(f"\n[{i}/{len(test_candidates)}] Тестирование...")
        
        success, output = test_model(str(model_path), symbol, days=7)
        
        if success:
            # Пытаемся извлечь ключевые метрики из вывода
            metrics = extract_metrics(output)
            results.append({
                "model": model_path.name,
                "symbol": symbol,
                "status": "success",
                "metrics": metrics
            })
            
            # Краткий вывод результатов
            print(f"📊 Результаты для {model_path.name}:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
        else:
            results.append({
                "model": model_path.name,
                "symbol": symbol, 
                "status": "failed",
                "error": output[:200] if output else "Unknown error"
            })
    
    # Итоговый отчет
    print(f"\n{'='*80}")
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Успешно: {len(successful)}")
    print(f"❌ Ошибок: {len(failed)}")
    
    if successful:
        print(f"\n🏆 ЛУЧШИЕ МОДЕЛИ:")
        # Сортируем по предполагаемой прибыльности
        for result in sorted(successful, key=lambda x: x.get('metrics', {}).get('total_pnl_pct', 0), reverse=True)[:3]:
            print(f"   • {result['model']}")
            metrics = result.get('metrics', {})
            if 'total_pnl_pct' in metrics:
                print(f"     PnL: {metrics['total_pnl_pct']}%")
            if 'win_rate_pct' in metrics:
                print(f"     Win Rate: {metrics['win_rate_pct']}%")
    
    if failed:
        print(f"\n⚠️  МОДЕЛИ С ОШИБКАМИ:")
        for result in failed:
            print(f"   • {result['model']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if successful:
        print("   1. Используйте лучшие модели для демо-торговли")
        print("   2. Проведите дополнительное тестирование на 30+ днях")
        print("   3. Сравните с предыдущими результатами")
    else:
        print("   1. Проверьте скрипт test_ml_strategy.py")
        print("   2. Убедитесь что есть исторические данные")
        print("   3. Попробуйте тестировать по одной модели")

def extract_metrics(output):
    """Извлекает метрики из вывода тестирования."""
    metrics = {}
    
    # Простые паттерны для поиска метрик
    lines = output.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        if 'total pnl' in line_lower and '%' in line:
            try:
                # Ищем число с процентами
                import re
                match = re.search(r'([-+]?\d*\.?\d+)%', line)
                if match:
                    metrics['total_pnl_pct'] = float(match.group(1))
            except:
                pass
        
        elif 'win rate' in line_lower and '%' in line:
            try:
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    metrics['win_rate_pct'] = float(match.group(1))
            except:
                pass
        
        elif 'total trades' in line_lower:
            try:
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    metrics['total_trades'] = int(match.group(1))
            except:
                pass
    
    return metrics

if __name__ == "__main__":
    main()