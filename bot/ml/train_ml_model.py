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
    
    # Список символов для обучения
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    interval = "15"  # 15 минут
    
    # Обучаем модели для каждого символа
    for symbol in symbols:
        print("\n" + "=" * 60)
        print(f"Training models for {symbol}")
        print("=" * 60)
        
        # === Шаг 1: Сбор данных ===
        print(f"\n[Step 1] Collecting historical data for {symbol}...")
        collector = DataCollector(settings.api)
        
        # Собираем данные за последние 6 месяцев
        df_raw = collector.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=None,  # Автоматически 6 месяцев назад
            end_date=None,
            limit=200,
        )
        
        if df_raw.empty:
            print(f"❌ No data collected for {symbol}. Skipping.")
            continue
        
        print(f"✅ Collected {len(df_raw)} candles")
        
        # === Шаг 2: Feature Engineering ===
        print(f"\n[Step 2] Creating features for {symbol}...")
        feature_engineer = FeatureEngineer()
        
        # Создаем технические индикаторы
        df_features = feature_engineer.create_technical_indicators(df_raw)
        print(f"✅ Created {len(feature_engineer.get_feature_names())} features")
        
        # Создаем целевую переменную (предсказываем движение через 1 час = 4 периода)
        # Используем более мягкие пороги для более активной модели (больше LONG/SHORT сигналов)
        threshold_pct = 0.2  # Сниженный порог для большего количества сигналов (было 0.3-0.4)
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=4,  # 4 * 15m = 1 час
            threshold_pct=threshold_pct,  # Более мягкий порог для большего количества сигналов
            use_atr_threshold=True,  # Использовать динамический порог на основе ATR
            use_risk_adjusted=True,  # Использовать риск-скорректированную целевую переменную
            min_risk_reward_ratio=2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам TP=25%, SL=10%)
        )
        print(f"  Using threshold: {threshold_pct}% (optimized for {symbol})")
        
        print(f"✅ Created target variable")
        print(f"  Target distribution: {df_with_target['target'].value_counts().to_dict()}")
        
        # === Шаг 3: Подготовка данных для обучения ===
        print(f"\n[Step 3] Preparing data for training {symbol}...")
        X, y = feature_engineer.prepare_features_for_ml(df_with_target)
        
        print(f"✅ Prepared data:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        # === Шаг 4: Обучение моделей ===
        print(f"\n[Step 4] Training models for {symbol}...")
        trainer = ModelTrainer()
        
        # Обучаем Random Forest
        print(f"\n--- Training Random Forest for {symbol} ---")
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
        print(f"\n--- Training XGBoost for {symbol} ---")
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
        
        # Обучаем Ансамбль (взвешенное усреднение)
        print(f"\n--- Training Ensemble Model for {symbol} ---")
        ensemble_model, ensemble_metrics = trainer.train_ensemble(
            X, y,
            rf_n_estimators=100,
            rf_max_depth=10,
            xgb_n_estimators=100,
            xgb_max_depth=6,
            xgb_learning_rate=0.1,
            ensemble_method="weighted_average",  # Используем взвешенное усреднение
        )
        
        # Сохраняем ансамбль
        trainer.save_model(
            ensemble_model,
            trainer.scaler,
            feature_engineer.get_feature_names(),
            ensemble_metrics,
            f"ensemble_{symbol}_{interval}.pkl",
            symbol=symbol,
            interval=interval,
            model_type="ensemble_weighted",
        )
        
        # === Шаг 5: Сравнение моделей ===
        print("\n" + "=" * 60)
        print(f"Model Comparison for {symbol}")
        print("=" * 60)
        print(f"\nRandom Forest:")
        print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {rf_metrics['cv_mean']:.4f} (+/- {rf_metrics['cv_std'] * 2:.4f})")
        
        print(f"\nXGBoost:")
        print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {xgb_metrics['cv_mean']:.4f} (+/- {xgb_metrics['cv_std'] * 2:.4f})")
        
        print(f"\nEnsemble (Weighted Average):")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"  Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  Recall: {ensemble_metrics['recall']:.4f}")
        print(f"  F1-Score: {ensemble_metrics['f1_score']:.4f}")
        print(f"  CV Accuracy: {ensemble_metrics['cv_mean']:.4f} (+/- {ensemble_metrics['cv_std'] * 2:.4f})")
        print(f"  CV F1-Score: {ensemble_metrics['cv_f1_mean']:.4f}")
        
        # Выбираем лучшую модель
        models_comparison = [
            ("Random Forest", rf_metrics['cv_mean'], rf_metrics),
            ("XGBoost", xgb_metrics['cv_mean'], xgb_metrics),
            ("Ensemble", ensemble_metrics['cv_mean'], ensemble_metrics),
        ]
        models_comparison.sort(key=lambda x: x[1], reverse=True)
        best_model_name, best_cv_score, best_metrics = models_comparison[0]
        
        print(f"\n✅ Best model for {symbol}: {best_model_name}")
        print(f"   CV Accuracy: {best_cv_score:.4f}")
        if best_model_name == "Ensemble":
            print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
            print(f"   CV F1-Score: {best_metrics['cv_f1_mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed for all symbols!")
    print("=" * 60)


if __name__ == "__main__":
    main()

