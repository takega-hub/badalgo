"""
Экстремально агрессивное переобучение ML модели.
Параметры настроены на МАКСИМАЛЬНОЕ количество сигналов.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer
from bot.config import load_settings
import warnings
warnings.filterwarnings('ignore')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Торговая пара для переобучения")
    args = parser.parse_known_args()[0]
    
    print("=" * 80)
    print("🔥 ЭКСТРЕМАЛЬНО АГРЕССИВНОЕ ПЕРЕОБУЧЕНИЕ ML")
    print("=" * 80)
    
    settings = load_settings()
    # Если символ передан через аргументы, используем его, иначе все три
    symbols = [args.symbol] if args.symbol else ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    interval = "15"
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"🎯 ПЕРЕОБУЧЕНИЕ: {symbol}")
        print(f"{'='*80}")
        
        # === Шаг 1: Сбор данных ===
        print(f"\n[1/5] 📊 Сбор исторических данных...")
        collector = DataCollector(settings.api)
        
        # Собираем данные за 6 месяцев (по умолчанию start_date = 180 дней назад)
        df_raw = collector.collect_klines(
            symbol=symbol,
            interval=interval,
            limit=200,
            save_to_file=False,  # Не сохраняем в файл
        )
        
        print(f"✅ Собрано {len(df_raw)} свечей")
        
        # === Шаг 2: Создание признаков ===
        print(f"\n[2/5] 🔧 Создание технических признаков...")
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_technical_indicators(df_raw)
        feature_names = feature_engineer.get_feature_names()
        
        print(f"✅ Создано {len(feature_names)} признаков")
        
        # === Шаг 3: Создание АГРЕССИВНОГО таргета ===
        print(f"\n[3/5] 🔥 Создание ЭКСТРЕМАЛЬНО агрессивной целевой переменной...")
        print("   🔥 НОВЫЕ параметры (экстремально агрессивные):")
        print("   • Forward periods: 3 (45 минут, было 75)")
        print("   • Threshold: 0.6% (было 1.0%)")
        print("   • Risk/Reward: 1.2:1 (было 1.5:1)")
        print("   • Цель: Ловить даже слабые движения!")
        
        df_with_target = feature_engineer.create_target_variable(
            df_features,
            forward_periods=3,  # 3 * 15m = 45 минут (короче!)
            threshold_pct=0.6,  # 0.6% (мягче!)
            use_atr_threshold=True,
            use_risk_adjusted=True,
            min_risk_reward_ratio=2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам TP=25%, SL=10%)
            max_hold_periods=48,  # Максимум 48 * 15m = 12 часов для качественных сделок (смягчено: было 32)
            min_profit_pct=1.0,  # Минимальная прибыль 1.0% для классификации как LONG/SHORT (смягчено: было 1.5%)
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
        
        # === Шаг 5: Обучение с ЭКСТРЕМАЛЬНОЙ балансировкой ===
        print(f"\n[5/5] 🔥 Обучение с ЭКСТРЕМАЛЬНОЙ балансировкой классов...")
        trainer = ModelTrainer()
        
        # ЭКСТРЕМАЛЬНЫЕ веса классов
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y)
        base_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # МАКСИМАЛЬНО усиливаем LONG/SHORT, МИНИМИЗИРУЕМ HOLD
        class_weight_dict = {}
        for i, cls in enumerate(classes):
            if cls == 0:  # HOLD
                class_weight_dict[cls] = base_weights[i] * 0.05  # 🔥 Экстремально низкий вес HOLD
            else:  # LONG or SHORT
                class_weight_dict[cls] = base_weights[i] * 4.0  # 🔥 Очень высокий вес (увеличено с 3.0 до 4.0)
        
        print(f"\n   🔥 ЭКСТРЕМАЛЬНЫЕ веса классов:")
        for cls, weight in class_weight_dict.items():
            label_name = "LONG" if cls == 1 else ("SHORT" if cls == -1 else "HOLD")
            multiplier = weight / base_weights[list(classes).index(cls)]
            print(f"      {label_name}: {weight:.3f} (x{multiplier:.1f})")
        
        # Обучаем Random Forest с экстремальной балансировкой
        print(f"\n   🔥 Обучение Random Forest...")
        rf_model, rf_metrics = trainer.train_random_forest_classifier(
            X, y,
            n_estimators=150,
            max_depth=12,
            class_weight=class_weight_dict,  # 🔥 ЭКСТРЕМАЛЬНАЯ балансировка!
        )
        
        # Определяем суффикс режима (MTF или 15m-only) по флагу окружения
        ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "1")
        ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")
        mode_suffix = "mtf" if ml_mtf_enabled else "15m"

        # Сохраняем модель с метаданными
        trainer.save_model(
            rf_model,
            trainer.scaler,
            feature_names,
            rf_metrics,
            f"rf_{symbol}_{interval}_{mode_suffix}.pkl",
            symbol=symbol,
            interval=interval,
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "n_estimators": 150,
                "max_depth": 12,
                "forward_periods": 3,  # 🔥 Короче!
                "threshold_pct": 0.6,  # 🔥 Мягче!
                "min_risk_reward_ratio": 2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам)
                "hold_weight_multiplier": 0.05,  # 🔥 Экстремально низкий!
                "long_short_weight_multiplier": 3.0,  # 🔥 Высокий!
            },
        )
        print(f"      ✅ Accuracy: {rf_metrics['accuracy']:.4f}")
        print(f"      ✅ CV Accuracy: {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']*2:.4f}")
        
        # Обучаем XGBoost с экстремальной балансировкой
        print(f"\n   🔥 Обучение XGBoost...")
        xgb_model, xgb_metrics = trainer.train_xgboost_classifier(
            X, y,
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            class_weight=class_weight_dict,  # 🔥 ЭКСТРЕМАЛЬНАЯ балансировка!
        )
        
        # Сохраняем модель с метаданными
        trainer.save_model(
            xgb_model,
            trainer.scaler,
            feature_names,
            xgb_metrics,
            f"xgb_{symbol}_{interval}_{mode_suffix}.pkl",
            symbol=symbol,
            interval=interval,
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.05,
                "forward_periods": 3,  # 🔥 Короче!
                "threshold_pct": 0.6,  # 🔥 Мягче!
                "min_risk_reward_ratio": 2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам)
                "hold_weight_multiplier": 0.05,  # 🔥 Экстремально низкий!
                "long_short_weight_multiplier": 3.0,  # 🔥 Высокий!
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
            class_weight=class_weight_dict,  # 🔥 ЭКСТРЕМАЛЬНАЯ балансировка!
        )
        
        # Сохраняем модель с метаданными
        trainer.save_model(
            ensemble_model,
            trainer.scaler,
            feature_names,
            ensemble_metrics,
            f"ensemble_{symbol}_{interval}_{mode_suffix}.pkl",
            symbol=symbol,
            interval=interval,
            model_type="ensemble_ultra_aggressive",
            class_weights=class_weight_dict,
            class_distribution=target_dist.to_dict(),
            training_params={
                "rf_n_estimators": 150,
                "rf_max_depth": 12,
                "xgb_n_estimators": 150,
                "xgb_max_depth": 8,
                "xgb_learning_rate": 0.05,
                "ensemble_method": "weighted_average",
                "forward_periods": 3,  # 🔥 Короче!
                "threshold_pct": 0.6,  # 🔥 Мягче!
                "min_risk_reward_ratio": 2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам)
                "hold_weight_multiplier": 0.05,  # 🔥 Экстремально низкий!
                "long_short_weight_multiplier": 3.0,  # 🔥 Высокий!
            },
        )
        
        print(f"\n   ✅ Метрики Ensemble:")
        print(f"      CV Accuracy:  {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']*2:.4f}")
        print(f"      F1-Score:     {ensemble_metrics['f1_score']:.4f}")
        
        # Обучаем TripleEnsemble (RF + XGBoost + LightGBM)
        from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
        if LIGHTGBM_AVAILABLE:
            print(f"\n   🔥 Обучение TripleEnsemble (RF + XGBoost + LightGBM)...")
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
                class_weight=class_weight_dict,  # 🔥 ЭКСТРЕМАЛЬНАЯ балансировка!
            )
            
            # Сохраняем модель с метаданными
            trainer.save_model(
                triple_ensemble_model,
                trainer.scaler,
                feature_names,
                triple_ensemble_metrics,
                f"triple_ensemble_{symbol}_{interval}_{mode_suffix}.pkl",
                symbol=symbol,
                interval=interval,
                model_type="triple_ensemble_ultra_aggressive",
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
                    "forward_periods": 3,  # 🔥 Короче!
                    "threshold_pct": 0.6,  # 🔥 Мягче!
                    "min_risk_reward_ratio": 2.0,  # Соотношение риск/прибыль 2:1 (соответствует торговым параметрам)
                    "hold_weight_multiplier": 0.05,  # 🔥 Экстремально низкий!
                    "long_short_weight_multiplier": 3.0,  # 🔥 Высокий!
                },
            )
            
            print(f"\n   ✅ Метрики TripleEnsemble:")
            print(f"      CV Accuracy:  {triple_ensemble_metrics['cv_mean']:.4f} ± {triple_ensemble_metrics['cv_std']*2:.4f}")
            print(f"      F1-Score:     {triple_ensemble_metrics['f1_score']:.4f}")
            print(f"      Weights:      RF={triple_ensemble_metrics['rf_weight']:.3f}, "
                  f"XGB={triple_ensemble_metrics['xgb_weight']:.3f}, "
                  f"LGB={triple_ensemble_metrics['lgb_weight']:.3f}")
        else:
            print(f"\n   ⚠️  LightGBM не установлен, пропускаем TripleEnsemble")
    
    # Финальное сообщение
    print("\n" + "=" * 80)
    print("🎉 ЭКСТРЕМАЛЬНО АГРЕССИВНОЕ ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print("\n📦 Обновлены модели:")
    print("   • ml_models/rf_*_15.pkl (Random Forest)")
    print("   • ml_models/xgb_*_15.pkl (XGBoost)")
    print("   • ml_models/ensemble_*_15.pkl (RF + XGBoost)")
    from bot.ml.model_trainer import LIGHTGBM_AVAILABLE
    if LIGHTGBM_AVAILABLE:
        print("   • ml_models/triple_ensemble_*_15.pkl (RF + XGBoost + LightGBM)")
    print("\n🔥 ОЖИДАЕМОЕ УЛУЧШЕНИЕ:")
    print("   • Сигналов: 15 → 100-200 (в 10+ раз больше!)")
    print("   • LONG:  4 → 50-100")
    print("   • SHORT: 11 → 50-100")
    print("\n🧪 СЛЕДУЮЩИЙ ШАГ:")
    print("   python test_ml_strategy.py --symbol SOLUSDT --days 14 --confidence 0.4 --strength слабое --no-stability")
    print("\n⚠️  Win Rate может снизиться (это нормально для агрессивной модели)")
    print("=" * 80)

if __name__ == "__main__":
    main()
