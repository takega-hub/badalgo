"""
Анализ результатов обучения LSTM модели и сравнение с другими ML моделями.
"""
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Загружает метаданные модели."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            return data.get('metadata', {}) or data.get('metrics', {})
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None

def analyze_lstm_model(model_path: str = "ml_models/lstm_BTCUSDT_15.pkl"):
    """Анализирует LSTM модель."""
    print("=" * 70)
    print("📊 LSTM MODEL ANALYSIS")
    print("=" * 70)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        metrics = data.get('metrics', {})
        config = data.get('model_config', {})
        trainer_config = data.get('trainer_config', {})
        metadata = data.get('metadata', {})
        
        print(f"\n📦 Model: {model_path}")
        print(f"   Symbol: {metadata.get('symbol', 'N/A')}")
        print(f"   Interval: {metadata.get('interval', 'N/A')}")
        print(f"   Trained: {metadata.get('trained_at', 'N/A')}")
        
        print(f"\n🏗️ Architecture:")
        print(f"   Input Size: {config.get('input_size', 'N/A')}")
        print(f"   Hidden Size: {config.get('hidden_size', 'N/A')}")
        print(f"   Layers: {config.get('num_layers', 'N/A')}")
        print(f"   Dropout: {config.get('dropout', 'N/A')}")
        
        print(f"\n⚙️ Training Config:")
        print(f"   Sequence Length: {trainer_config.get('sequence_length', 'N/A')}")
        print(f"   Batch Size: {trainer_config.get('batch_size', 'N/A')}")
        print(f"   Learning Rate: {trainer_config.get('learning_rate', 'N/A')}")
        print(f"   Epochs Trained: {metrics.get('num_epochs_trained', 'N/A')}")
        
        print(f"\n📈 Performance:")
        print(f"   Best Val Accuracy: {metrics.get('best_val_acc', 0):.4f} ({metrics.get('best_val_acc', 0)*100:.2f}%)")
        print(f"   Final Val Accuracy: {metrics.get('final_val_acc', 0):.4f} ({metrics.get('final_val_acc', 0)*100:.2f}%)")
        
        # Анализ обучения
        train_losses = metrics.get('train_losses', [])
        val_losses = metrics.get('val_losses', [])
        val_accuracies = metrics.get('val_accuracies', [])
        
        if train_losses and val_losses:
            print(f"\n📉 Training Progress:")
            print(f"   Initial Train Loss: {train_losses[0]:.4f}")
            print(f"   Final Train Loss: {train_losses[-1]:.4f}")
            print(f"   Initial Val Loss: {val_losses[0]:.4f}")
            print(f"   Final Val Loss: {val_losses[-1]:.4f}")
            print(f"   Initial Val Acc: {val_accuracies[0]:.4f}")
            print(f"   Final Val Acc: {val_accuracies[-1]:.4f}")
            
            # Проверка на переобучение
            if len(train_losses) > 1:
                train_trend = train_losses[-1] < train_losses[0]
                val_trend = val_losses[-1] < val_losses[0]
                if train_trend and not val_trend:
                    print(f"\n⚠️  WARNING: Possible overfitting detected!")
                    print(f"   Train loss decreased but val loss increased")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return None

def compare_with_other_models(symbol: str = "BTCUSDT", interval: str = "15"):
    """Сравнивает LSTM с другими моделями."""
    print("\n" + "=" * 70)
    print("🔄 MODEL COMPARISON")
    print("=" * 70)
    
    models_to_check = [
        ("LSTM", f"ml_models/lstm_{symbol}_{interval}.pkl"),
        ("Ensemble", f"ml_models/ensemble_{symbol}_{interval}.pkl"),
        ("Random Forest", f"ml_models/rf_{symbol}_{interval}.pkl"),
        ("XGBoost", f"ml_models/xgb_{symbol}_{interval}.pkl"),
    ]
    
    results = []
    
    for model_name, model_path in models_to_check:
        if not Path(model_path).exists():
            continue
        
        metadata = load_model_metadata(model_path)
        if not metadata:
            continue
        
        # Извлекаем метрики
        if model_name == "LSTM":
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                metrics = data.get('metrics', {})
                accuracy = metrics.get('best_val_acc', 0)
                cv_mean = None
                cv_std = None
        else:
            accuracy = metadata.get('accuracy', 0)
            cv_mean = metadata.get('cv_mean', None)
            cv_std = metadata.get('cv_std', None)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'CV Mean': cv_mean,
            'CV Std': cv_std,
            'Path': model_path,
        })
    
    if not results:
        print("❌ No models found for comparison")
        return
    
    # Создаем DataFrame для красивого вывода
    df = pd.DataFrame(results)
    
    print(f"\n📊 Comparison for {symbol} ({interval}m):")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"\n{row['Model']}:")
        if row['Model'] == 'LSTM':
            print(f"   Validation Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
        else:
            if pd.notna(row['CV Mean']):
                print(f"   Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
                print(f"   CV Accuracy: {row['CV Mean']:.4f} ({row['CV Mean']*100:.2f}%)")
                if pd.notna(row['CV Std']):
                    print(f"   CV Std: ±{row['CV Std']:.4f}")
            else:
                print(f"   Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
    
    # Определяем лучшую модель
    if len(results) > 1:
        print("\n" + "-" * 70)
        best_lstm = df[df['Model'] == 'LSTM']['Accuracy'].values[0]
        best_other = df[df['Model'] != 'LSTM']['CV Mean'].fillna(df[df['Model'] != 'LSTM']['Accuracy']).max()
        
        if best_other and best_lstm < best_other:
            diff = (best_other - best_lstm) * 100
            print(f"\n⚠️  LSTM accuracy is {diff:.2f}% lower than best model")
            print(f"   Best model accuracy: {best_other:.4f} ({best_other*100:.2f}%)")
            print(f"   LSTM accuracy: {best_lstm:.4f} ({best_lstm*100:.2f}%)")
        elif best_other and best_lstm >= best_other:
            print(f"\n✅ LSTM performs similarly or better!")
            print(f"   LSTM accuracy: {best_lstm:.4f} ({best_lstm*100:.2f}%)")
            print(f"   Best other: {best_other:.4f} ({best_other*100:.2f}%)")

def suggest_improvements(metrics: Dict[str, Any]):
    """Предлагает улучшения для LSTM модели."""
    print("\n" + "=" * 70)
    print("💡 SUGGESTIONS FOR IMPROVEMENT")
    print("=" * 70)
    
    if not metrics:
        print("❌ No metrics available")
        return
    
    val_acc = metrics.get('best_val_acc', 0)
    epochs = metrics.get('num_epochs_trained', 0)
    
    suggestions = []
    
    # Проверка точности
    if val_acc < 0.65:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'Low validation accuracy ({val_acc:.2%})',
            'suggestions': [
                'Add class weights to handle imbalanced data',
                'Normalize/scaling features before training',
                'Increase sequence length (try 90-120 candles)',
                'Add more features (use all available features)',
                'Try different architecture (bidirectional LSTM)',
                'Increase hidden size (try 128 or 256)',
            ]
        })
    
    # Проверка количества эпох
    if epochs < 20:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'Training stopped early (only {epochs} epochs)',
            'suggestions': [
                'Increase max_patience for early stopping',
                'Try different learning rate schedule',
                'Use more data for training',
            ]
        })
    
    # Общие рекомендации
    suggestions.append({
        'priority': 'MEDIUM',
        'issue': 'General LSTM improvements',
        'suggestions': [
            'Add attention mechanism for better feature importance',
            'Use bidirectional LSTM to capture both past and future context',
            'Try ensemble of multiple LSTM models',
            'Add regularization (L2, dropout tuning)',
            'Use data augmentation (time warping, noise injection)',
            'Hyperparameter tuning (grid search or Bayesian optimization)',
        ]
    })
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. [{suggestion['priority']}] {suggestion['issue']}:")
        for j, sug in enumerate(suggestion['suggestions'], 1):
            print(f"   {j}. {sug}")

def main():
    """Основная функция."""
    symbol = "BTCUSDT"
    interval = "15"
    
    # Анализ LSTM
    metrics = analyze_lstm_model(f"ml_models/lstm_{symbol}_{interval}.pkl")
    
    # Сравнение с другими моделями
    compare_with_other_models(symbol, interval)
    
    # Предложения по улучшению
    if metrics:
        suggest_improvements(metrics)
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
