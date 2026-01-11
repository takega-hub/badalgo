"""
Диагностический скрипт для проверки ML модели.
Помогает понять, почему модель не генерирует сигналы.
"""
import warnings
import os

# Подавляем предупреждения scikit-learn ДО импорта библиотек
# Устанавливаем переменную окружения ПЕРВОЙ
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['SKLEARN_WARNINGS'] = 'ignore'

# Фильтруем все предупреждения sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*propagate the scikit-learn configuration.*')
# Специфичное предупреждение из терминала
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

import pickle
import pathlib
import pandas as pd
import numpy as np
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy
from bot.config import load_settings


def diagnose_model(model_path: str, symbol: str = "ETHUSDT", interval: str = "15"):
    """
    Диагностирует ML модель и показывает статистику предсказаний.
    
    Args:
        model_path: Путь к модели
        symbol: Торговая пара
        interval: Интервал свечей
    """
    print("=" * 80)
    print(f"ML Model Diagnostics")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print()
    
    # Загружаем модель
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        metadata = model_data.get("metadata", {})
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Training date: {metadata.get('trained_at', 'unknown')}")
        print(f"   Training accuracy: {metadata.get('accuracy', 'unknown')}")
        print(f"   CV accuracy: {metadata.get('cv_mean', 'unknown')}")
        
        if hasattr(model, "classes_"):
            print(f"   Classes: {model.classes_}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Загружаем настройки
    settings = load_settings()
    
    # Создаем клиент
    client = BybitClient(settings.api)
    
    # Получаем данные
    print("📊 Fetching market data...")
    try:
        # Преобразуем интервал в формат Bybit (15 -> "15", "15m" -> "15")
        # Bybit использует числовые значения: "1", "3", "5", "15", "30", "60", "240" и т.д.
        if interval.endswith("m"):
            bybit_interval = interval[:-1]  # "15m" -> "15"
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            bybit_interval = str(hours * 60)  # "1h" -> "60"
        elif interval.endswith("d"):
            days = int(interval[:-1])
            bybit_interval = str(days * 1440)  # "1d" -> "1440"
        else:
            bybit_interval = str(interval)  # "15" -> "15"
        
        df = client.get_kline_df(symbol, bybit_interval, limit=500)
        if df.empty:
            print("❌ No klines data received")
            return
        
        # Убеждаемся, что timestamp в правильном формате
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index, unit="ms")
        
        print(f"✅ Loaded {len(df)} candles")
        print()
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Создаем стратегию
    confidence_threshold = settings.ml_confidence_threshold
    strategy = MLStrategy(model_path, confidence_threshold)
    
    print("🔍 Analyzing predictions...")
    print(f"   Confidence threshold: {confidence_threshold:.2%}")
    print()
    
    # Анализируем последние 50 баров
    predictions_stats = {"LONG": 0, "SHORT": 0, "HOLD": 0}
    confidence_stats = {"LONG": [], "SHORT": [], "HOLD": []}
    actionable_signals = 0
    
    for i in range(200, min(len(df), 250)):  # Нужно минимум 200 баров для индикаторов
        try:
            df_until_now = df.iloc[:i+1]
            prediction, confidence = strategy.predict(df_until_now)
            
            pred_name = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(prediction, f"UNKNOWN({prediction})")
            predictions_stats[pred_name] += 1
            confidence_stats[pred_name].append(confidence)
            
            # Проверяем, будет ли сигнал actionable
            if prediction != 0 and confidence >= confidence_threshold:
                actionable_signals += 1
        except Exception as e:
            print(f"⚠️ Error at bar {i}: {e}")
            continue
    
    # Выводим статистику
    print("=" * 80)
    print("📈 Prediction Statistics (last 50 bars):")
    print("=" * 80)
    total = sum(predictions_stats.values())
    for pred_name, count in predictions_stats.items():
        pct = (count / total * 100) if total > 0 else 0
        avg_conf = np.mean(confidence_stats[pred_name]) if confidence_stats[pred_name] else 0
        max_conf = np.max(confidence_stats[pred_name]) if confidence_stats[pred_name] else 0
        min_conf = np.min(confidence_stats[pred_name]) if confidence_stats[pred_name] else 0
        
        print(f"   {pred_name:6s}: {count:3d} ({pct:5.1f}%) | "
              f"Confidence: avg={avg_conf:.4f}, min={min_conf:.4f}, max={max_conf:.4f}")
    
    print()
    print(f"   Actionable signals (prediction != 0 AND confidence >= {confidence_threshold:.2%}): {actionable_signals}")
    print()
    
    # Анализируем последний бар детально
    print("=" * 80)
    print("🔬 Last Bar Detailed Analysis:")
    print("=" * 80)
    try:
        prediction, confidence = strategy.predict(df)
        pred_name = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(prediction, f"UNKNOWN({prediction})")
        
        print(f"   Prediction: {pred_name} ({prediction})")
        print(f"   Confidence: {confidence:.4f} ({confidence:.2%})")
        print(f"   Threshold: {confidence_threshold:.4f} ({confidence_threshold:.2%})")
        print(f"   Will generate signal: {'✅ YES' if (prediction != 0 and confidence >= confidence_threshold) else '❌ NO'}")
        
        if hasattr(model, "predict_proba"):
            # Получаем вероятности для всех классов
            X = strategy.prepare_features(df)
            X_last = X[-1:].reshape(1, -1)
            proba = model.predict_proba(X_last)[0]
            
            print()
            print("   Class Probabilities:")
            if len(proba) == 3:
                class_names = ["SHORT (-1)", "HOLD (0)", "LONG (1)"]
                for i, (name, prob) in enumerate(zip(class_names, proba)):
                    marker = "👉" if i == (prediction + 1) else "  "
                    print(f"   {marker} {name:12s}: {prob:.4f} ({prob:.2%})")
            else:
                for i, prob in enumerate(proba):
                    print(f"      Class {i}: {prob:.4f} ({prob:.2%})")
    except Exception as e:
        print(f"❌ Error analyzing last bar: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("💡 Recommendations:")
    print("=" * 80)
    
    if predictions_stats["HOLD"] > total * 0.8:
        print("   ⚠️ Model predicts HOLD too often (>80%)")
        print("      → Consider retraining with different target thresholds")
        print("      → Check if target variable creation is correct")
    
    if actionable_signals == 0:
        print("   ⚠️ No actionable signals generated")
        avg_long_conf = np.mean(confidence_stats["LONG"]) if confidence_stats["LONG"] else 0
        avg_short_conf = np.mean(confidence_stats["SHORT"]) if confidence_stats["SHORT"] else 0
        
        if avg_long_conf > 0 or avg_short_conf > 0:
            max_avg_conf = max(avg_long_conf, avg_short_conf)
            if max_avg_conf < confidence_threshold:
                print(f"      → Average confidence ({max_avg_conf:.2%}) < threshold ({confidence_threshold:.2%})")
                print(f"      → Consider lowering confidence threshold to {max_avg_conf * 0.9:.2%}")
        else:
            print("      → Model only predicts HOLD")
            print("      → Model may need retraining")
    
    if predictions_stats["LONG"] == 0 and predictions_stats["SHORT"] == 0:
        print("   ⚠️ Model never predicts LONG or SHORT")
        print("      → Model may be overfitted to HOLD class")
        print("      → Consider retraining with balanced classes")
    
    print()


if __name__ == "__main__":
    import sys
    
    # Определяем путь к модели
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Ищем модель по умолчанию
        project_root = pathlib.Path(__file__).parent.parent.parent
        model_dir = project_root / "ml_models"
        models = list(model_dir.glob("*.pkl"))
        if not models:
            print("❌ No models found. Please provide model path as argument.")
            sys.exit(1)
        model_path = str(models[0])
        print(f"Using model: {model_path}")
    
    # Определяем символ из имени файла
    model_name = pathlib.Path(model_path).stem
    if "ETH" in model_name:
        symbol = "ETHUSDT"
    elif "BTC" in model_name:
        symbol = "BTCUSDT"
    elif "SOL" in model_name:
        symbol = "SOLUSDT"
    else:
        symbol = "SOLUSDT"  # По умолчанию SOLUSDT, так как это текущая торговая пара
    
    diagnose_model(model_path, symbol)

