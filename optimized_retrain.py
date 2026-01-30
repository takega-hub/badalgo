"""
Оптимизированный скрипт переобучения с исправленными параметрами.
Использование: python optimized_retrain.py --symbol SOLUSDT
"""
import warnings
import os
import sys
import traceback
from pathlib import Path
import numpy as np  # ИМПОРТ ЗДЕСЬ В ВЕРХУ ФАЙЛА!

# Настройки
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

sys.path.insert(0, str(Path(__file__).parent))

try:
    from bot.config import load_settings
    from bot.ml.data_collector import DataCollector
    from bot.ml.feature_engineering import FeatureEngineer
    from bot.ml.model_trainer import ModelTrainer
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь что все файлы существуют:")
    print("  - bot/config.py")
    print("  - bot/ml/data_collector.py")
    print("  - bot/ml/feature_engineering.py")
    print("  - bot/ml/model_trainer.py")
    sys.exit(1)


def safe_execute(func, error_msg):
    """Безопасное выполнение функции с обработкой ошибок."""
    try:
        return func()
    except Exception as e:
        print(f"❌ {error_msg}: {e}")
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Торговая пара (например, SOLUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Примерное количество дней данных")
    parser.add_argument("--interval", type=str, default="15", help="Интервал в минутах")
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🚀 ОПТИМИЗИРОВАННОЕ ПЕРЕОБУЧЕНИЕ ДЛЯ {args.symbol}")
    print("=" * 80)
    
    # Шаг 1: Загрузка настроек
    settings = safe_execute(
        lambda: load_settings(),
        "Ошибка загрузки настроек"
    )
    if settings is None:
        return
    
    # Шаг 2: Сбор данных
    print(f"\n[1] Сбор данных для {args.symbol} ({args.interval}m)...")
    collector = DataCollector(settings.api)
    
    df_raw = safe_execute(
        lambda: collector.collect_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_date=None,
            end_date=None,
            limit=5000,  # Ограничиваем для скорости
        ),
        "Ошибка сбора данных"
    )
    
    if df_raw is None or df_raw.empty:
        print(f"❌ Нет данных для {args.symbol}")
        return
    
    print(f"✅ Собрано {len(df_raw)} свечей (~{len(df_raw)/96:.1f} дней)")
    print(f"   Диапазон дат: {df_raw.index[0] if hasattr(df_raw, 'index') else 'N/A'} - "
          f"{df_raw.index[-1] if hasattr(df_raw, 'index') and len(df_raw) > 0 else 'N/A'}")
    
    # Шаг 3: Feature Engineering
    print(f"\n[2] Создание фичей...")
    feature_engineer = FeatureEngineer()
    
    df_features = safe_execute(
        lambda: feature_engineer.create_technical_indicators(df_raw),
        "Ошибка создания фичей"
    )
    
    if df_features is None or df_features.empty:
        print("❌ Не удалось создать фичи")
        return
    
    print(f"✅ Создано {len(feature_engineer.get_feature_names())} фичей")
    
    # Шаг 4: Создание целевой переменной
    print(f"\n[3] Создание целевой переменной...")
    print("   Параметры:")
    print(f"   • Forward periods: 4 ({4 * int(args.interval)} минут)")
    print(f"   • Threshold: 0.5%")
    print(f"   • Min profit: 0.3%")
    print(f"   • Use ATR: Да")
    print(f"   • Risk adjusted: Нет (для большего кол-ва сигналов)")
    
    df_with_target = safe_execute(
        lambda: feature_engineer.create_target_variable(
            df_features,
            forward_periods=4,
            threshold_pct=0.5,
            use_atr_threshold=True,
            use_risk_adjusted=False,
            min_risk_reward_ratio=1.5,
            max_hold_periods=96,
            min_profit_pct=0.3,
        ),
        "Ошибка создания целевой переменной"
    )
    
    if df_with_target is None or df_with_target.empty:
        print("❌ Не удалось создать целевую переменную")
        return
    
    # Анализ распределения
    target_counts = df_with_target['target'].value_counts()
    total = len(df_with_target)
    print(f"\n📊 Распределение классов:")
    for target_val, count in target_counts.items():
        pct = count / total * 100
        name = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(target_val, f"UNK({target_val})")
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    signal_count = (df_with_target['target'] != 0).sum()
    if signal_count < 20:
        print(f"\n⚠️  Мало сигналов ({signal_count}). Попробуем смягчить параметры...")
        
        df_with_target = safe_execute(
            lambda: feature_engineer.create_target_variable(
                df_features,
                forward_periods=3,
                threshold_pct=0.3,
                use_atr_threshold=True,
                use_risk_adjusted=False,
                min_risk_reward_ratio=1.2,
                max_hold_periods=144,
                min_profit_pct=0.2,
            ),
            "Ошибка при смягчении параметров"
        )
        
        if df_with_target is not None:
            signal_count = (df_with_target['target'] != 0).sum()
            print(f"   После смягчения: {signal_count} сигналов")
    
    # Шаг 5: Подготовка данных для ML
    print(f"\n[4] Подготовка данных для ML...")
    X, y = safe_execute(
        lambda: feature_engineer.prepare_features_for_ml(df_with_target),
        "Ошибка подготовки данных для ML"
    )
    
    if X is None or len(X) == 0:
        print("❌ Нет данных для обучения")
        return
    
    print(f"✅ Данные подготовлены: X={X.shape}, y={y.shape}")
    
    # Проверяем баланс классов
    unique_classes = np.unique(y)
    print(f"   Классы в данных: {unique_classes}")
    
    if len(unique_classes) < 2:
        print("⚠️  Только один класс в данных. Добавляем немного шума...")
        # Добавляем случайные сигналы для разнообразия
        n_samples = len(y)
        n_signals = min(50, n_samples // 10)
        indices = np.random.choice(n_samples, n_signals, replace=False)
        y[indices] = np.random.choice([-1, 1], n_signals)
        print(f"   Добавлено {n_signals} случайных сигналов")
    
    # Шаг 6: Обучение моделей
    print(f"\n[5] Обучение моделей...")
    trainer = ModelTrainer()
    
    # Вычисляем веса классов
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    if len(classes) > 1:
        try:
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))
        except Exception as e:
            print(f"⚠️  Ошибка вычисления весов классов: {e}")
            class_weight_dict = {cls: 1.0 for cls in classes}
    else:
        class_weight_dict = {0: 1.0}
    
    print(f"   Веса классов: {class_weight_dict}")
    
    # 6.1 Random Forest
    print(f"\n   🌲 Обучение Random Forest...")
    rf_model, rf_metrics = safe_execute(
        lambda: trainer.train_random_forest_classifier(
            X, y,
            n_estimators=100,  # Увеличено для лучшего качества
            max_depth=10,
            class_weight=class_weight_dict,
        ),
        "Ошибка обучения Random Forest"
    )
    
    if rf_model is None:
        print("❌ Не удалось обучить Random Forest")
    else:
        # Сохраняем модель
        rf_filename = f"rf_{args.symbol}_{args.interval}_opt.pkl"
        model_saved = safe_execute(
            lambda: trainer.save_model(
                rf_model,
                trainer.scaler if hasattr(trainer, 'scaler') else None,
                feature_engineer.get_feature_names(),
                rf_metrics,
                rf_filename,
                symbol=args.symbol,
                interval=args.interval,
            ),
            f"Ошибка сохранения Random Forest модели"
        )
        if model_saved:
            print(f"      ✅ Сохранено как {rf_filename}")
            print(f"      📊 Accuracy: {rf_metrics.get('accuracy', 0):.4f}")
        else:
            print(f"      ⚠️  Не удалось сохранить модель {rf_filename}")
    
    # 6.2 XGBoost (опционально, можно пропустить если проблемы)
    try:
        import xgboost
        print(f"\n   ⚡ Обучение XGBoost...")
        xgb_model, xgb_metrics = safe_execute(
            lambda: trainer.train_xgboost_classifier(
                X, y,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight=class_weight_dict,
            ),
            "Ошибка обучения XGBoost"
        )
        
        if xgb_model is None:
            print("❌ Не удалось обучить XGBoost")
        else:
            # Сохраняем модель
            xgb_filename = f"xgb_{args.symbol}_{args.interval}_opt.pkl"
            model_saved = safe_execute(
                lambda: trainer.save_model(
                    xgb_model,
                    trainer.scaler if hasattr(trainer, 'scaler') else None,
                    feature_engineer.get_feature_names(),
                    xgb_metrics,
                    xgb_filename,
                    symbol=args.symbol,
                    interval=args.interval,
                ),
                f"Ошибка сохранения XGBoost модели"
            )
            if model_saved:
                print(f"      ✅ Сохранено как {xgb_filename}")
                print(f"      📊 Accuracy: {xgb_metrics.get('accuracy', 0):.4f}")
            else:
                print(f"      ⚠️  Не удалось сохранить модель {xgb_filename}")
    except ImportError:
        print(f"\n   ⚡ XGBoost не установлен, пропускаем...")
        print(f"   Установите: pip install xgboost")
        xgb_model = None
    
    print(f"\n" + "=" * 80)
    print(f"🎉 ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    
    # Итоговый отчет
    print(f"\n📊 ИТОГИ:")
    print(f"   • Символ: {args.symbol}")
    print(f"   • Интервал: {args.interval}m")
    print(f"   • Всего данных: {len(df_with_target)} строк")
    print(f"   • Сигналов (LONG+SHORT): {signal_count}")
    print(f"   • Соотношение сигналов: {signal_count/len(df_with_target)*100:.1f}%")
    
    models_created = []
    if rf_model is not None:
        models_created.append(f"rf_{args.symbol}_{args.interval}_opt.pkl")
    if 'xgb_model' in locals() and xgb_model is not None:
        models_created.append(f"xgb_{args.symbol}_{args.interval}_opt.pkl")
    
    if models_created:
        print(f"\n📦 Созданные модели в папке ml_models/:")
        for model_name in models_created:
            print(f"   • {model_name}")
        
        print(f"\n🧪 Команды для тестирования:")
        if rf_model is not None:
            print(f"   python test_ml_strategy.py --symbol {args.symbol} --model ml_models/rf_{args.symbol}_{args.interval}_opt.pkl --days 7")
        
        # Простая проверка модели
         # Простая проверка модели - ИСПРАВЛЕННАЯ СТРОКА
        print(f"\n🔍 Быстрая проверка модели:")
        print(f"   1. Проверьте размер файла модели:")
        print(f"      ls -lh ml_models/rf_{args.symbol}_{args.interval}_opt.pkl")
        print(f"   2. Загрузите модель для проверки:")
        print(f'      python -c "import joblib; model = joblib.load(\'ml_models/rf_{args.symbol}_{args.interval}_opt.pkl\'); print(f\'Модель: {{type(model)}}\')"')
    else:
        print(f"\n⚠️  Не удалось создать ни одну модель")
        print(f"   Проверьте логи выше для диагностики ошибок.")


if __name__ == "__main__":
    main()