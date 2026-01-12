"""
Скрипт для проверки метаданных обученной модели.
Показывает, с какими данными модель была обучена.
"""
import pickle
from pathlib import Path

def check_model(symbol="SOLUSDT"):
    model_path = Path(f"ml_models/ensemble_{symbol}_15.pkl")
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    print("=" * 80)
    print(f"🔍 ПРОВЕРКА МОДЕЛИ: {symbol}")
    print("=" * 80)
    
    # Загружаем модель
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Общая информация
    print(f"\n📦 Модель: {model_data.get('model_type', 'unknown')}")
    print(f"📅 Дата обучения: {model_data.get('timestamp', 'unknown')}")
    print(f"🔢 Количество фичей: {len(model_data.get('feature_names', []))}")
    
    # Информация о данных
    data_info = model_data.get('data_info', {})
    print(f"\n📊 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
    print(f"   Всего строк: {data_info.get('total_rows', 'unknown')}")
    
    # Распределение классов
    class_dist = data_info.get('class_distribution', {})
    if class_dist:
        total = sum(class_dist.values())
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ (в обучающих данных):")
        for cls, count in sorted(class_dist.items()):
            pct = count / total * 100 if total > 0 else 0
            cls_name = "HOLD" if cls == 0 else ("LONG" if cls == 1 else "SHORT")
            print(f"   {cls_name:6} ({cls:2}): {count:5} ({pct:5.1f}%)")
        
        # Проверяем, соответствует ли распределение ожиданиям
        hold_pct = class_dist.get(0, 0) / total * 100 if total > 0 else 0
        long_pct = class_dist.get(1, 0) / total * 100 if total > 0 else 0
        short_pct = class_dist.get(-1, 0) / total * 100 if total > 0 else 0
        
        print(f"\n💡 ОЦЕНКА РАСПРЕДЕЛЕНИЯ:")
        if hold_pct > 70:
            print(f"   ⚠️ HOLD слишком много ({hold_pct:.1f}%) - модель будет консервативной!")
            print(f"   ✅ Целевое значение: HOLD ~58%, LONG ~21%, SHORT ~21%")
        elif hold_pct < 50:
            print(f"   ⚠️ HOLD слишком мало ({hold_pct:.1f}%) - модель может быть слишком агрессивной!")
        else:
            print(f"   ✅ Распределение нормальное!")
        
        if abs(long_pct - short_pct) > 10:
            print(f"   ⚠️ LONG/SHORT несбалансированы ({long_pct:.1f}% vs {short_pct:.1f}%)")
        else:
            print(f"   ✅ LONG/SHORT сбалансированы!")
    else:
        print(f"   ⚠️ Информация о распределении классов отсутствует")
    
    # Метрики
    metrics = model_data.get('metrics', {})
    if metrics:
        print(f"\n📈 МЕТРИКИ МОДЕЛИ:")
        print(f"   CV Mean Accuracy: {metrics.get('cv_mean', 0):.4f}")
        print(f"   CV Std:           {metrics.get('cv_std', 0):.4f}")
        print(f"   F1-Score:         {metrics.get('f1_score', 0):.4f}")
    
    # Class weights (если есть)
    class_weights = model_data.get('class_weights', None)
    if class_weights:
        print(f"\n⚖️ CLASS WEIGHTS (использованные при обучении):")
        for cls, weight in sorted(class_weights.items()):
            cls_name = "HOLD" if cls == 0 else ("LONG" if cls == 1 else "SHORT")
            print(f"   {cls_name:6} ({cls:2}): {weight:.3f}")
    else:
        print(f"\n⚠️ Информация о class_weights отсутствует (возможно, использовалось 'balanced')")
    
    print("\n" + "=" * 80)
    
    # Дополнительные параметры обучения
    training_params = model_data.get('training_params', {})
    if training_params:
        print(f"\n🔧 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
        for key, value in training_params.items():
            print(f"   {key}: {value}")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    symbols = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    
    if len(sys.argv) > 1:
        symbols = [sys.argv[1]]
    
    for symbol in symbols:
        check_model(symbol)
        print("\n")
