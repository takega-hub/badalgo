"""
Улучшенный скрипт переобучения ML-модели с оптимизациями для большего количества сигналов.

Улучшения:
1. Более агрессивный таргет (movement > 1%)
2. Балансировка классов (class_weight)
3. Увеличенные данные (30 дней)
4. Оптимизированные гиперпараметры
"""
import warnings
import os

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer


def main():
    """Переобучение с оптимизированными параметрами."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Торговая пара для переобучения")
    args = parser.parse_known_args()[0]
    
    print("=" * 80)
    print("🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ML МОДЕЛИ")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Список символов для обучения
    symbols = [args.symbol] if args.symbol else ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    interval = "15"  # 15 минут
    
    # Обучаем модели для каждого символа
    for symbol in symbols:
        print("\n" + "=" * 80)
        print(f"📊 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ {symbol}")
        print("=" * 80)
        
        # === Шаг 1: Сбор данных (30 дней) ===
        print(f"\n[1/5] 📥 Сбор исторических данных для {symbol}...")
        collector = DataCollector(settings.api)
        
        # Увеличиваем количество данных
        # 30 дней * 24 часа * 4 (15-минутки) = ~2880 свечей
        df_raw = collector.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=None,
            end_date=None,
            limit=3000,  # Увеличено с 200 до 3000 для большей глубины
        )
        
        if df_raw.empty:
            print(f"❌ Нет данных для {symbol}. Пропускаем.")
            continue
        
        print(f"✅ Собрано {len(df_raw)} свечей (~{len(df_raw)/96:.1f} дней)")
        
        # === Шаг 2: Feature Engineering ===
        print(f"\n[2/5] 🔧 Создание признаков для {symbol}...")
        feature_engineer = FeatureEngineer()
        
        # Создаем технические индикаторы
        df_features = feature_engineer.create_technical_indicators(df_raw)
        feature_names = feature_engineer.get_feature_names()
        print(f"✅ Создано {len(feature_names)} признаков")
        
        # === Шаг 3: Создание таргета (оптимизированный) ===
        print(f"\n[3/5] 🎯 Создание целевой переменной (оптимизированный таргет)...")
        print("   Параметры:")
        print("   • Forward periods: 5 (75 минут)")
        print("   • Threshold: 1.0% (вместо 0.2%)")
        print("   • Risk/Reward: 1.5:1")
        print("   • Use ATR threshold: True")
        
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=5,  # 5 * 15m = 75 минут (вместо 4 = 60 минут)
            threshold_pct=1.0,  # Увеличен с 0.2% до 1.0% для более сильных сигналов
            use_atr_threshold=True,
            use_risk_adjusted=True,
            min_risk_reward_ratio=1.5,  # Меньше требования для большего количества сигналов
        )
        
        # Анализ распределения классов
        target_dist = df_with_target['target'].value_counts()
        print(f"\n✅ Целевая переменная создана")
        print(f"   Распределение классов:")
        for label, count in target_dist.items():
            pct = count / len(df_with_target) * 100
            emoji = "🟢" if label == 1 else ("🔴" if label == -1 else "⚪")
            label_name = "LONG" if label == 1 else ("SHORT" if label == -1 else "HOLD")
            print(f"   {emoji} {label_name:5s}: {count:5d} ({pct:5.1f}%)")
        
        # === Шаг 4: Подготовка данных ===
        print(f"\n[4/5] 📦 Подготовка данных для обучения...")
        X, y = feature_engineer.prepare_features_for_ml(df_with_target)
        
        print(f"✅ Данные подготовлены:")
        print(f"   Features: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"   Target: {y.shape[0]} labels")
        
        # === Шаг 5: Обучение с балансировкой классов ===
        print(f"\n[5/5] 🤖 Обучение моделей с балансировкой классов...")
        trainer = ModelTrainer()
        
        # Вычисляем веса классов для балансировки
        # LONG и SHORT получают больший вес, HOLD - меньший
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y)
        base_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Усиливаем веса для LONG/SHORT, ослабляем для HOLD
        class_weight_dict = {}
        for i, cls in enumerate(classes):
            if cls == 0:  # HOLD
                class_weight_dict[cls] = base_weights[i] * 0.3  # Уменьшаем вес HOLD
            else:  # LONG or SHORT
                class_weight_dict[cls] = base_weights[i] * 2.0  # Увеличиваем вес LONG/SHORT
        
        print(f"\n   📊 Веса классов:")
        for cls, weight in class_weight_dict.items():
            label_name = "LONG" if cls == 1 else ("SHORT" if cls == -1 else "HOLD")
            print(f"      {label_name}: {weight:.2f}")
        
        # Обучаем Random Forest с балансировкой
        print(f"\n   🌲 Обучение Random Forest...")
        rf_model, rf_metrics = trainer.train_random_forest_classifier(
            X, y,
            n_estimators=150,  # Увеличено с 100 до 150
            max_depth=12,      # Увеличено с 10 до 12
            class_weight=class_weight_dict,  # Балансировка классов!
        )
        
        # Сохраняем модель с полными метаданными
        trainer.save_model(
            rf_model,
            trainer.scaler,
            feature_names,
            rf_metrics,
            f"rf_{symbol}_{interval}.pkl",
            symbol=symbol,
            interval=interval,
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "n_estimators": 150,
                "max_depth": 12,
                "forward_periods": 5,
                "threshold_pct": 1.0,
                "min_risk_reward_ratio": 1.5,
            },
        )
        print(f"      ✅ Accuracy: {rf_metrics['accuracy']:.4f}")
        print(f"      ✅ CV Accuracy: {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']*2:.4f}")
        
        # Обучаем XGBoost с балансировкой
        print(f"\n   ⚡ Обучение XGBoost...")
        
        # Преобразуем class_weight в scale_pos_weight для XGBoost
        # XGBoost использует scale_pos_weight для балансировки
        xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
            X, y,
            n_estimators=150,    # Увеличено с 100 до 150
            max_depth=8,         # Увеличено с 6 до 8
            learning_rate=0.05,  # Уменьшено с 0.1 до 0.05 для лучшей генерализации
            class_weight=class_weight_dict,  # Балансировка классов!
        )
        
        # Сохраняем модель с полными метаданными
        trainer.save_model(
            xgb_model,
            trainer.scaler,
            feature_names,
            xgb_metrics,
            f"xgb_{symbol}_{interval}.pkl",
            symbol=symbol,
            interval=interval,
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.05,
                "forward_periods": 5,
                "threshold_pct": 1.0,
                "min_risk_reward_ratio": 1.5,
            },
        )
        print(f"      ✅ Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"      ✅ CV Accuracy: {xgb_metrics['cv_mean']:.4f} ± {xgb_metrics['cv_std']*2:.4f}")
        
        # Обучаем Ensemble (RF + XGBoost)
        print(f"\n   🎯 Обучение Ensemble (RF + XGBoost)...")
        ensemble_model, ensemble_metrics = trainer.train_ensemble(
            X, y,
            rf_n_estimators=150,
            rf_max_depth=12,
            xgb_n_estimators=150,
            xgb_max_depth=8,
            xgb_learning_rate=0.05,
            ensemble_method="weighted_average",
            class_weight=class_weight_dict,  # Балансировка классов!
        )
        
        # Сохраняем модель с полными метаданными
        trainer.save_model(
            ensemble_model,
            trainer.scaler,
            feature_names,
            ensemble_metrics,
            f"ensemble_{symbol}_{interval}.pkl",
            symbol=symbol,
            interval=interval,
            model_type="ensemble_weighted",
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "xgb_n_estimators": 150,
                "xgb_max_depth": 8,
                "xgb_learning_rate": 0.05,
                "ensemble_method": "weighted_average",
                "forward_periods": 5,
                "threshold_pct": 1.0,
                "min_risk_reward_ratio": 1.5,
            },
        )
        print(f"      ✅ Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"      ✅ CV Accuracy: {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
        
        # Обучаем TripleEnsemble (RF + XGBoost + LightGBM)
        from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
        if LIGHTGBM_AVAILABLE:
            print(f"\n   🎯 Обучение TripleEnsemble (RF + XGBoost + LightGBM)...")
            triple_ensemble_model, triple_ensemble_metrics = trainer.train_ensemble(
                X, y,
                rf_n_estimators=150,
                rf_max_depth=12,
                xgb_n_estimators=150,
                xgb_max_depth=8,
                xgb_learning_rate=0.05,
                lgb_n_estimators=150,
                lgb_max_depth=8,
                lgb_learning_rate=0.05,
                ensemble_method="triple",
                include_lightgbm=True,
                class_weight=class_weight_dict,  # Балансировка классов!
            )
            
            # Сохраняем модель с полными метаданными
            trainer.save_model(
                triple_ensemble_model,
                trainer.scaler,
                feature_names,
                triple_ensemble_metrics,
                f"triple_ensemble_{symbol}_{interval}.pkl",
                symbol=symbol,
                interval=interval,
                model_type="triple_ensemble",
                class_weights=class_weight_dict,
                class_distribution=target_dist.to_dict(),
                training_params={
                    "rf_n_estimators": 150,
                    "rf_max_depth": 12,
                    "xgb_n_estimators": 150,
                    "xgb_max_depth": 8,
                    "xgb_learning_rate": 0.05,
                    "lgb_n_estimators": 150,
                    "lgb_max_depth": 8,
                    "lgb_learning_rate": 0.05,
                    "ensemble_method": "triple",
                    "forward_periods": 5,
                    "threshold_pct": 1.0,
                    "min_risk_reward_ratio": 1.5,
                },
            )
            print(f"      ✅ Accuracy: {triple_ensemble_metrics['accuracy']:.4f}")
            print(f"      ✅ CV Accuracy: {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")
            print(f"      ✅ Weights: RF={triple_ensemble_metrics['rf_weight']:.3f}, "
                  f"XGB={triple_ensemble_metrics['xgb_weight']:.3f}, "
                  f"LGB={triple_ensemble_metrics['lgb_weight']:.3f}")
        else:
            print(f"\n   ⚠️  LightGBM не установлен, пропускаем TripleEnsemble")
            triple_ensemble_metrics = None
        
        # Итоговые метрики
        print(f"\n" + "-" * 80)
        print(f"📊 ИТОГОВЫЕ МЕТРИКИ ДЛЯ {symbol}")
        print("-" * 80)
        print(f"\n🌲 Random Forest:")
        print(f"   Accuracy:     {rf_metrics['accuracy']:.4f}")
        print(f"   CV Accuracy:  {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']*2:.4f}")
        
        print(f"\n⚡ XGBoost:")
        print(f"   Accuracy:     {xgb_metrics['accuracy']:.4f}")
        print(f"   CV Accuracy:  {xgb_metrics['cv_mean']:.4f} ± {xgb_metrics['cv_std']*2:.4f}")
        
        print(f"\n🎯 Ensemble (RF+XGB):")
        print(f"   Accuracy:     {ensemble_metrics['accuracy']:.4f}")
        print(f"   Precision:    {ensemble_metrics['precision']:.4f}")
        print(f"   Recall:       {ensemble_metrics['recall']:.4f}")
        print(f"   F1-Score:     {ensemble_metrics['f1_score']:.4f}")
        print(f"   CV Accuracy:  {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
        print(f"   CV F1-Score:  {ensemble_metrics['cv_f1_mean']:.4f}")
        
        if triple_ensemble_metrics:
            print(f"\n🎯 TripleEnsemble (RF+XGB+LGB):")
            print(f"   Accuracy:     {triple_ensemble_metrics['accuracy']:.4f}")
            print(f"   Precision:    {triple_ensemble_metrics['precision']:.4f}")
            print(f"   Recall:       {triple_ensemble_metrics['recall']:.4f}")
            print(f"   F1-Score:     {triple_ensemble_metrics['f1_score']:.4f}")
            print(f"   CV Accuracy:  {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")
            print(f"   CV F1-Score:  {triple_ensemble_metrics['cv_f1_mean']:.4f}")
            print(f"   Weights:      RF={triple_ensemble_metrics['rf_weight']:.3f}, "
                  f"XGB={triple_ensemble_metrics['xgb_weight']:.3f}, "
                  f"LGB={triple_ensemble_metrics['lgb_weight']:.3f}")
        
        # Выбор лучшей модели
        models = [
            ("Random Forest", rf_metrics['cv_mean']),
            ("XGBoost", xgb_metrics['cv_mean']),
            ("Ensemble", ensemble_metrics['cv_mean']),
        ]
        if triple_ensemble_metrics:
            models.append(("TripleEnsemble", triple_ensemble_metrics['cv_mean']))
        models.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score = models[0]
        
        print(f"\n✅ Лучшая модель для {symbol}: {best_model}")
        print(f"   Cross-Validation Accuracy: {best_score:.4f}")
    
    # Финальное сообщение
    print("\n" + "=" * 80)
    print("🎉 ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print("\n📦 Созданные модели:")
    print("   • ml_models/rf_*_15.pkl (Random Forest)")
    print("   • ml_models/xgb_*_15.pkl (XGBoost)")
    print("   • ml_models/ensemble_*_15.pkl (RF + XGBoost)")
    from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
    if LIGHTGBM_AVAILABLE:
        print("   • ml_models/triple_ensemble_*_15.pkl (RF + XGBoost + LightGBM)")
    print("\n🚀 Следующие шаги:")
    print("   1. Протестируйте новые модели:")
    print("      python test_ml_strategy.py --symbol SOLUSDT --days 7")
    print("   2. Если результаты хорошие, задеплойте на сервер:")
    print("      scp ml_models/*.pkl user@server:/opt/crypto_bot/ml_models/")
    print("      ssh user@server 'sudo systemctl restart crypto-bot'")
    print("\n💡 Ожидаемое улучшение: 50-100 сигналов за 7 дней (вместо 19)")
    print("=" * 80)


if __name__ == "__main__":
    main()
