"""
Скрипт для обучения ML-модели на исторических данных.
Использование: python -m bot.ml.train_ml_model
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
    """Основная функция для обучения модели."""
    print("=" * 60)
    print("ML Model Training Script")
    print("=" * 60)
    
    # Загружаем настройки
    settings = load_settings()
    
    # === Шаг 1: Сбор данных ===
    print("\n[Step 1] Collecting historical data...")
    collector = DataCollector(settings.api)
    
    symbol = "ETHUSDT"  # Тестируем на ETH
    interval = "15"  # 15 минут
    
    # Собираем данные за последние 6 месяцев
    df_raw = collector.collect_klines(
        symbol=symbol,
        interval=interval,
        start_date=None,  # Автоматически 6 месяцев назад
        end_date=None,
        limit=200,
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
    
    # Создаем целевую переменную (предсказываем движение через 1 час = 4 периода)
    # Используем оптимизированный порог для более активной модели
    threshold_pct = 0.3 if symbol in ["SOLUSDT", "BTCUSDT"] else 0.4
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=4,  # 4 * 15m = 1 час
        threshold_pct=threshold_pct,  # Сниженный порог для большего количества сигналов
        use_atr_threshold=True,  # Использовать динамический порог на основе ATR
    )
    print(f"  Using threshold: {threshold_pct}% (optimized for {symbol})")
    
    print(f"✅ Created target variable")
    print(f"  Target distribution: {df_with_target['target'].value_counts().to_dict()}")
    
    # === Шаг 3: Подготовка данных для обучения ===
    print("\n[Step 3] Preparing data for training...")
    X, y = feature_engineer.prepare_features_for_ml(df_with_target)
    
    print(f"✅ Prepared data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # === Шаг 4: Обучение моделей ===
    print("\n[Step 4] Training models...")
    trainer = ModelTrainer()
    
    # Обучаем Random Forest
    print("\n--- Training Random Forest ---")
    rf_model, rf_metrics = trainer.train_random_forest_classifier(
        X, y,
        n_estimators=100,
        max_depth=10,
    )
    
    # Сохраняем Random Forest модель
    trainer.save_model(
        rf_model,
        trainer.scaler,
        feature_engineer.get_feature_names(),
        rf_metrics,
        f"rf_{symbol}_{interval}.pkl",
        symbol=symbol,
        interval=interval,
    )
    
    # Обучаем XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
        X, y,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    
    # Сохраняем XGBoost модель
    trainer.save_model(
        xgb_model,
        trainer.scaler,
        feature_engineer.get_feature_names(),
        xgb_metrics,
        f"xgb_{symbol}_{interval}.pkl",
        symbol=symbol,
        interval=interval,
    )
    
    # === Шаг 5: Сравнение моделей ===
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
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
    else:
        best_model_name = "Random Forest"
        best_metrics = rf_metrics
    
    print(f"\n✅ Best model: {best_model_name}")
    print(f"   CV Accuracy: {best_metrics['cv_mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

