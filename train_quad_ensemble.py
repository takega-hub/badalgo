"""
Скрипт для обучения QuadEnsemble модели (RF + XGBoost + LightGBM + LSTM).
Использование: python train_quad_ensemble.py --symbol BTCUSDT --days 180
"""
import warnings
import os
import argparse
import sys
from pathlib import Path

# Подавляем предупреждения
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ ERROR: PyTorch is not installed!")
    print("   Install with: pip install torch>=2.0.0")

from bot.config import load_settings
from bot.ml.data_collector import DataCollector
from bot.ml.feature_engineering import FeatureEngineer
from bot.ml.model_trainer import ModelTrainer, LIGHTGBM_AVAILABLE, LSTM_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description='Train QuadEnsemble ML model (RF+XGB+LGB+LSTM)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=180,
                       help='Number of days of historical data (default: 180)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Timeframe interval (default: 15m)')
    
    # RandomForest параметры
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                       help='Number of RF estimators (default: 100)')
    parser.add_argument('--rf_max_depth', type=int, default=10,
                       help='RF max depth (default: 10)')
    
    # XGBoost параметры
    parser.add_argument('--xgb_n_estimators', type=int, default=100,
                       help='Number of XGB estimators (default: 100)')
    parser.add_argument('--xgb_max_depth', type=int, default=6,
                       help='XGB max depth (default: 6)')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1,
                       help='XGB learning rate (default: 0.1)')
    
    # LightGBM параметры
    parser.add_argument('--lgb_n_estimators', type=int, default=100,
                       help='Number of LGB estimators (default: 100)')
    parser.add_argument('--lgb_max_depth', type=int, default=6,
                       help='LGB max depth (default: 6)')
    parser.add_argument('--lgb_learning_rate', type=float, default=0.1,
                       help='LGB learning rate (default: 0.1)')
    
    # LSTM параметры
    parser.add_argument('--lstm_sequence_length', type=int, default=60,
                       help='LSTM sequence length (default: 60)')
    parser.add_argument('--lstm_hidden_size', type=int, default=64,
                       help='LSTM hidden size (default: 64)')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='LSTM number of layers (default: 2)')
    parser.add_argument('--lstm_epochs', type=int, default=50,
                       help='LSTM training epochs (default: 50)')
    parser.add_argument('--lstm_batch_size', type=int, default=32,
                       help='LSTM batch size (default: 32)')
    parser.add_argument('--lstm_learning_rate', type=float, default=0.001,
                       help='LSTM learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Проверяем зависимости
    if not TORCH_AVAILABLE:
        return
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ ERROR: LightGBM is not installed!")
        print("   Install with: pip install lightgbm>=4.0.0")
        return
    
    if not LSTM_AVAILABLE:
        print("❌ ERROR: LSTM module is not available!")
        print("   Check that bot.ml.lstm_model can be imported")
        return
    
    print("=" * 80)
    print("🚀 QuadEnsemble ML Model Training (RF + XGBoost + LightGBM + LSTM)")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print(f"\n📊 Model Parameters:")
    print(f"  RandomForest: {args.rf_n_estimators} trees, max_depth={args.rf_max_depth}")
    print(f"  XGBoost: {args.xgb_n_estimators} trees, max_depth={args.xgb_max_depth}, lr={args.xgb_learning_rate}")
    print(f"  LightGBM: {args.lgb_n_estimators} trees, max_depth={args.lgb_max_depth}, lr={args.lgb_learning_rate}")
    print(f"  LSTM: seq_len={args.lstm_sequence_length}, hidden={args.lstm_hidden_size}, layers={args.lstm_num_layers}, epochs={args.lstm_epochs}")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # === Шаг 1: Сбор данных ===
    print(f"\n[Step 1] Collecting historical data for {args.symbol}...")
    collector = DataCollector(settings.api)
    
    # Собираем данные
    df_raw = collector.collect_klines(
        symbol=args.symbol,
        interval=args.interval.replace('m', ''),
        start_date=None,
        end_date=None,
        limit=200,
    )
    
    if df_raw.empty:
        print(f"❌ No data collected for {args.symbol}. Skipping.")
        return
    
    print(f"✅ Collected {len(df_raw)} candles")
    
    # === Шаг 2: Feature Engineering ===
    print(f"\n[Step 2] Creating features...")
    feature_engineer = FeatureEngineer()
    
    # Создаем технические индикаторы
    df_features = feature_engineer.create_technical_indicators(df_raw)
    print(f"✅ Created {len(feature_engineer.get_feature_names())} features")
    
    # Создаем целевую переменную
    print(f"\n[Step 3] Creating target variable...")
    df_with_target = feature_engineer.create_target_variable(
        df_features,
        forward_periods=5,  # 5 * 15m = 75 минут
        threshold_pct=1.0,  # 1.0% порог
        use_atr_threshold=True,
        use_risk_adjusted=True,
        min_risk_reward_ratio=1.5,
    )
    
    target_dist = df_with_target['target'].value_counts().to_dict()
    print(f"✅ Target distribution:")
    for target_val, count in sorted(target_dist.items()):
        pct = (count / len(df_with_target)) * 100
        target_name = {-1: "SHORT", 0: "HOLD", 1: "LONG"}.get(
            target_val, f"UNKNOWN({target_val})")
        print(f"    {target_name:6s}: {count:5d} ({pct:5.1f}%)")
    
    # === Шаг 4: Подготовка данных для ML ===
    print(f"\n[Step 4] Preparing data for ML...")
    X, y = feature_engineer.prepare_features_for_ml(df_with_target)
    print(f"✅ Prepared data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Проверяем, что данных достаточно для LSTM
    if len(df_with_target) < args.lstm_sequence_length + 100:
        print(f"⚠️  WARNING: Not enough data for LSTM (need at least {args.lstm_sequence_length + 100} rows, got {len(df_with_target)})")
        print(f"   Consider reducing --lstm_sequence_length or collecting more data")
    
    # === Шаг 5: Обучение QuadEnsemble ===
    print(f"\n[Step 5] Training QuadEnsemble...")
    print(f"   This will train 4 models sequentially:")
    print(f"   1. RandomForest")
    print(f"   2. XGBoost")
    print(f"   3. LightGBM")
    print(f"   4. LSTM (this may take longer)")
    print()
    
    trainer = ModelTrainer()
    
    try:
        model, metrics = trainer.train_quad_ensemble(
            X=X,
            y=y,
            df=df_with_target,  # Полный DataFrame для LSTM
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            lgb_n_estimators=args.lgb_n_estimators,
            lgb_max_depth=args.lgb_max_depth,
            lgb_learning_rate=args.lgb_learning_rate,
            lstm_sequence_length=args.lstm_sequence_length,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_epochs=args.lstm_epochs,
        )
        
        print(f"\n📊 QuadEnsemble Results:")
        print(f"  RandomForest CV Accuracy: {metrics['rf_metrics']['cv_mean']:.4f} (+/- {metrics['rf_metrics']['cv_std'] * 2:.4f})")
        print(f"  XGBoost CV Accuracy: {metrics['xgb_metrics']['cv_mean']:.4f} (+/- {metrics['xgb_metrics']['cv_std'] * 2:.4f})")
        print(f"  LightGBM CV Accuracy: {metrics['lgb_metrics']['cv_mean']:.4f} (+/- {metrics['lgb_metrics']['cv_std'] * 2:.4f})")
        print(f"  LSTM Accuracy: {metrics['lstm_metrics'].get('accuracy', 0):.4f}")
        print(f"\n  Ensemble Weights:")
        print(f"    RF:   {metrics['rf_weight']:.3f}")
        print(f"    XGB:  {metrics['xgb_weight']:.3f}")
        print(f"    LGB:  {metrics['lgb_weight']:.3f}")
        print(f"    LSTM: {metrics['lstm_weight']:.3f}")
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === Шаг 6: Сохранение модели ===
    print(f"\n[Step 6] Saving model...")
    model_filename = f"quad_ensemble_{args.symbol}_{args.interval.replace('m', '')}.pkl"
    
    try:
        trainer.save_model(
            model,
            trainer.scaler,
            feature_engineer.get_feature_names(),
            metrics,
            model_filename,
            symbol=args.symbol,
            interval=args.interval.replace('m', ''),
            model_type="quad_ensemble",
        )
        
        print(f"✅ Model saved: {model_filename}")
        print(f"\n🎉 Training completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Test the model: python -m bot.ml.diagnose_model ml_models/{model_filename}")
        print(f"   2. Backtest: python backtest_ml_strategy.py --model ml_models/{model_filename} --symbol {args.symbol} --days 30")
        print(f"   3. Use in live trading: Enable ML strategy in config")
        print(f"   4. Compare with other models: Check backtest results")
        
    except Exception as e:
        print(f"❌ ERROR saving model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
