"""
Скрипт для быстрого переобучения ML-модели с улучшенными параметрами.
Использование: python -m bot.ml.retrain_model [SYMBOL] [INTERVAL]
Пример: python -m bot.ml.retrain_model SOLUSDT 15
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

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer


def main():
    """Основная функция для переобучения модели."""
    print("=" * 80)
    print("ML Model Retraining Script (Improved Parameters)")
    print("=" * 80)
    
    # Параметры из аргументов или по умолчанию
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SOLUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "15"
    
    print(f"\nSymbol: {symbol}")
    print(f"Interval: {interval}")
    print()
    
    # Загружаем настройки
    settings = load_settings()
    
    # === Шаг 1: Сбор данных ===
    print("[Step 1] Collecting historical data...")
    collector = DataCollector(settings.api)
    
    # Собираем больше данных для лучшего обучения (последние 6 месяцев)
    df_raw = collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=None,  # Автоматически 6 месяцев назад
        end_date=None,
        limit=500,  # Увеличиваем лимит для большего количества данных
    )
    
    if df_raw.empty:
        print("❌ No data collected. Exiting.")
        return
    
    print(f"✅ Collected {len(df_raw)} candles")
    
    # === Шаг 2: Feature Engineering ===
    print("\n[Step 2] Creating features...")
    feature_engineer = FeatureEngineer()
    
    # Создаем технические индикаторы
    df_features = feature_engineer.create_technical_indicators(df_raw)
    print(f"✅ Created {len(feature_engineer.get_feature_names())} features")
    
    # Создаем целевую переменную с улучшенными параметрами для большего количества сигналов
    print("\n[Step 3] Creating target variable with optimized parameters...")
    # Используем более низкий порог для более активной модели
    # Для криптовалют с высокой волатильностью (SOL, BTC) используем 0.3-0.4%
    # Это позволит получить больше LONG/SHORT сигналов в обучающих данных
    threshold_pct = 0.3 if symbol in ["SOLUSDT", "BTCUSDT"] else 0.4
    # Пробуем с динамическим порогом на основе ATR (но fallback на статический если ATR недоступен)
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=4,  # 4 * 15m = 1 час
        threshold_pct=threshold_pct,  # Сниженный порог для большего количества сигналов
        use_atr_threshold=True,  # Использовать динамический порог на основе ATR
    )
    print(f"  Using threshold: {threshold_pct}% (optimized for {symbol})")
    print(f"  Dynamic ATR threshold: enabled")
    
    print(f"✅ Created target variable")
    target_dist = df_with_target['target'].value_counts().to_dict()
    total = len(df_with_target)
    print(f"  Target distribution:")
    for target_val, count in sorted(target_dist.items()):
        pct = (count / total * 100) if total > 0 else 0
        target_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(target_val, f"UNKNOWN({target_val})")
        print(f"    {target_name:6s}: {count:5d} ({pct:5.1f}%)")
    
    # Проверяем баланс классов
    if target_dist.get(0, 0) > total * 0.8:
        print(f"\n⚠️  WARNING: HOLD class is >80% of data. Model may be biased.")
        print(f"   Consider adjusting threshold_pct or forward_periods.")
    
    if target_dist.get(1, 0) == 0 or target_dist.get(-1, 0) == 0:
        print(f"\n⚠️  WARNING: No LONG or SHORT signals found. Model cannot learn these classes.")
        print(f"   Consider lowering threshold_pct.")
        return
    
    # === Шаг 4: Подготовка данных для обучения ===
    print("\n[Step 4] Preparing data for training...")
    X, y = feature_engineer.prepare_features_for_ml(df_with_target)
    
    print(f"✅ Prepared data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # === Шаг 5: Обучение моделей ===
    print("\n[Step 5] Training models...")
    trainer = ModelTrainer()
    
    # Обучаем Random Forest
    print("\n--- Training Random Forest ---")
    rf_model, rf_metrics = trainer.train_random_forest_classifier(
        X, y,
        n_estimators=100,
        max_depth=10,
    )
    
    # Сохраняем Random Forest модель
    model_filename = f"rf_{symbol}_{interval}.pkl"
    trainer.save_model(
        rf_model,
        trainer.scaler,
        feature_engineer.get_feature_names(),
        rf_metrics,
        model_filename,
        symbol=symbol,
        interval=interval,
    )
    print(f"✅ Saved Random Forest model: {model_filename}")
    
    # Обучаем XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
        X, y,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    
    # Сохраняем XGBoost модель
    xgb_filename = f"xgb_{symbol}_{interval}.pkl"
    trainer.save_model(
        xgb_model,
        trainer.scaler,
        feature_engineer.get_feature_names(),
        xgb_metrics,
        xgb_filename,
        symbol=symbol,
        interval=interval,
    )
    print(f"✅ Saved XGBoost model: {xgb_filename}")
    
    # === Шаг 6: Сравнение моделей ===
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  CV Accuracy: {rf_metrics['cv_mean']:.4f} (+/- {rf_metrics['cv_std'] * 2:.4f})")
    
    print(f"\nXGBoost:")
    print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"  CV Accuracy: {xgb_metrics['cv_mean']:.4f} (+/- {xgb_metrics['cv_std'] * 2:.4f})")
    
    # Выбираем лучшую модель
    if xgb_metrics['cv_mean'] > rf_metrics['cv_mean']:
        best_model_name = "XGBoost"
        best_metrics = xgb_metrics
        best_filename = xgb_filename
    else:
        best_model_name = "Random Forest"
        best_metrics = rf_metrics
        best_filename = model_filename
    
    print(f"\n✅ Best model: {best_model_name}")
    print(f"   CV Accuracy: {best_metrics['cv_mean']:.4f}")
    print(f"   Model file: {best_filename}")
    
    print("\n" + "=" * 80)
    print("Retraining completed!")
    print("=" * 80)
    print(f"\n💡 Next steps:")
    print(f"   1. Test the model with: python -m bot.ml.diagnose_model {best_filename}")
    print(f"   2. Update ML_MODEL_PATH in .env to point to: ml_models/{best_filename}")
    print(f"   3. Restart the bot to use the new model")


if __name__ == "__main__":
    main()

