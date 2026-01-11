"""
Модуль для обучения ML-моделей на исторических данных.
"""
import warnings
import os
import sys

# Подавляем предупреждения scikit-learn ДО импорта библиотек
# Устанавливаем переменную окружения ПЕРВОЙ
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['SKLEARN_WARNINGS'] = 'ignore'
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # Для joblib

# Агрессивная фильтрация всех UserWarning
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*should be used with.*')
warnings.filterwarnings('ignore', message='.*propagate the scikit-learn configuration.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*joblib.*')

# Перехватываем предупреждения на уровне stderr для подавления sklearn warnings
class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.skip_patterns = [
            'sklearn.utils.parallel',
            'delayed',
            'joblib',
            'should be used with',
            'propagate the scikit-learn configuration'
        ]
    
    def write(self, message):
        # Пропускаем сообщения, содержащие паттерны предупреждений sklearn
        if any(pattern in message for pattern in self.skip_patterns):
            return
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # Проксируем все остальные атрибуты к оригинальному stderr
        return getattr(self.original_stderr, name)

# Сохраняем оригинальный stderr и устанавливаем фильтр
_original_stderr = sys.stderr
sys.stderr = WarningFilter(_original_stderr)

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from bot.ml.feature_engineering import FeatureEngineer


class ModelTrainer:
    """
    Обучает ML-модели для предсказания движения цены.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "ml_models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
    
    def train_random_forest_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Обучает Random Forest классификатор.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (1 = LONG, -1 = SHORT, 0 = HOLD)
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            min_samples_split: Минимальное количество образцов для разделения
            random_state: Seed для воспроизводимости
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training Random Forest Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y + 1)}")  # +1 чтобы индексы были 0,1,2
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Вычисляем веса классов для более агрессивного обучения на LONG/SHORT
        # Увеличиваем вес для LONG и SHORT классов относительно HOLD
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        class_weights = {}
        
        for cls, count in zip(unique_classes, class_counts):
            if count > 0:
                # Используем обратную частоту, но с дополнительным весом для LONG/SHORT
                base_weight = total_samples / (len(unique_classes) * count)
                if cls != 0:  # LONG (1) или SHORT (-1) получают дополнительный вес
                    class_weights[int(cls)] = base_weight * 1.5  # +50% вес для торговых сигналов
                else:  # HOLD (0) - базовый вес
                    class_weights[int(cls)] = base_weight * 0.8  # -20% вес для HOLD
        
        # Создаем и обучаем модель
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Используем все ядра
            class_weight=class_weights if class_weights else "balanced",  # Используем кастомные веса
        )
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        # Подавляем предупреждения при cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy", n_jobs=1)  # n_jobs=1 чтобы избежать предупреждений joblib
        
        # Обучаем на всех данных
        model.fit(X_scaled, y)
        
        # Предсказания для оценки
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "feature_importance": dict(zip(
                self.feature_engineer.get_feature_names(),
                model.feature_importances_
            )),
        }
        
        print(f"[model_trainer] Training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, metrics
    
    def train_xgboost_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """
        Обучает XGBoost классификатор.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            learning_rate: Скорость обучения
            random_state: Seed для воспроизводимости
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training XGBoost Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        # XGBoost может работать с ненормализованными данными, но нормализуем для консистентности
        X_scaled = self.scaler.fit_transform(X)
        
        # Преобразуем y для XGBoost (нужны индексы 0,1,2 вместо -1,0,1)
        y_xgb = y + 1  # -1,0,1 -> 0,1,2
        
        # Вычисляем веса классов для XGBoost (для классов 0,1,2)
        unique_classes, class_counts = np.unique(y_xgb, return_counts=True)
        total_samples = len(y_xgb)
        sample_weights = np.zeros(len(y_xgb))
        
        for cls, count in zip(unique_classes, class_counts):
            if count > 0:
                base_weight = total_samples / (len(unique_classes) * count)
                # Класс 1 (HOLD) - индекс 1 в XGBoost формате
                if cls == 1:  # HOLD - уменьшаем вес
                    weight = base_weight * 0.8
                else:  # LONG (2) или SHORT (0) - увеличиваем вес
                    weight = base_weight * 1.5
                sample_weights[y_xgb == cls] = weight
        
        # Создаем и обучаем модель
        # Примечание: scale_pos_weight работает только для бинарной классификации,
        # поэтому не используем его для мультиклассовой задачи (3 класса)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            # Балансировка классов выполняется через sample_weight в fit()
        )
        
        # Обучаем с весами образцов
        model.fit(X_scaled, y_xgb, sample_weight=sample_weights)
        
        # Time-series cross-validation (без весов, так как cross_val_score не поддерживает sample_weight напрямую)
        tscv = TimeSeriesSplit(n_splits=5)
        # Подавляем предупреждения при cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cv_scores = cross_val_score(model, X_scaled, y_xgb, cv=tscv, scoring="accuracy", n_jobs=1)  # n_jobs=1 чтобы избежать предупреждений joblib
        
        # Модель уже обучена выше с весами образцов
        
        # Предсказания
        y_pred_xgb = model.predict(X_scaled)
        y_pred = y_pred_xgb - 1  # Обратно в -1,0,1
        accuracy = accuracy_score(y, y_pred)
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "feature_importance": dict(zip(
                self.feature_engineer.get_feature_names(),
                model.feature_importances_
            )),
        }
        
        print(f"[model_trainer] Training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, metrics
    
    def save_model(
        self, 
        model: Any, 
        scaler: StandardScaler, 
        feature_names: list, 
        metrics: Dict[str, Any], 
        filename: str,
        symbol: str = "ETHUSDT",
        interval: str = "15",
        model_type: Optional[str] = None,
    ):
        """Сохраняет модель, scaler, метрики и метаданные в файл."""
        filepath = self.model_dir / filename
        
        # Определяем model_type из имени файла, если не передан
        if model_type is None:
            filename_base = filename.replace('.pkl', '')
            parts = filename_base.split('_')
            if len(parts) >= 1:
                model_type = parts[0].lower()  # rf, xgb и т.д.
            else:
                model_type = "unknown"
        
        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metrics": metrics,
            "metadata": {
                "symbol": symbol,
                "interval": interval,
                "model_type": model_type,
                "trained_at": datetime.now().isoformat(),
                "accuracy": metrics.get("accuracy", 0.0),
                "cv_mean": metrics.get("cv_mean", 0.0),
                "cv_std": metrics.get("cv_std", 0.0),
            }
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"[model_trainer] Model saved to {filepath}")
        return filepath
    
    def load_model_metadata(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Загружает только метаданные модели без самой модели.
        """
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                return model_data.get("metadata", {})
        except Exception as e:
            print(f"[model_trainer] Error loading model metadata: {e}")
            return None
    
    def load_model(self, filename: str) -> Dict[str, Any]:
        """Загружает модель из файла."""
        filepath = self.model_dir / filename
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"[model_trainer] Model loaded from {filepath}")
        return model_data

