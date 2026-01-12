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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from collections import Counter

from bot.ml.feature_engineering import FeatureEngineer


class PreTrainedVotingEnsemble:
    """Ансамбль с предобученными моделями для voting метода."""
    def __init__(self, rf_model, xgb_model, rf_weight, xgb_weight):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X):
        # Получаем вероятности от обеих моделей
        rf_proba = self.rf_model.predict_proba(X)
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        # XGBoost возвращает классы 0,1,2, нужно преобразовать в -1,0,1
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Взвешенное усреднение
        ensemble_proba = (self.rf_weight * rf_proba + 
                         self.xgb_weight * xgb_proba_reordered)
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class WeightedEnsemble:
    """Взвешенный ансамбль из RandomForest и XGBoost."""
    def __init__(self, rf_model, xgb_model, rf_weight=0.5, xgb_weight=0.5):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.classes_ = np.array([-1, 0, 1])  # SHORT, HOLD, LONG
    
    def predict_proba(self, X):
        # Получаем вероятности от обеих моделей
        rf_proba = self.rf_model.predict_proba(X)
        # Для XGBoost нужно преобразовать классы обратно
        xgb_proba = self.xgb_model.predict_proba(X)
        # XGBoost возвращает классы 0,1,2, нужно преобразовать в -1,0,1
        # Переупорядочиваем: [0,1,2] -> [-1,0,1]
        xgb_proba_reordered = np.zeros_like(rf_proba)
        xgb_proba_reordered[:, 0] = xgb_proba[:, 0]  # SHORT (0 -> -1)
        xgb_proba_reordered[:, 1] = xgb_proba[:, 1]  # HOLD (1 -> 0)
        xgb_proba_reordered[:, 2] = xgb_proba[:, 2]  # LONG (2 -> 1)
        
        # Взвешенное усреднение
        ensemble_proba = (self.rf_weight * rf_proba + 
                         self.xgb_weight * xgb_proba_reordered)
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


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
        class_weight: Optional[Dict[int, float]] = None,
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
            class_weight: Кастомные веса классов (если None, используется автоматическая балансировка)
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training Random Forest Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y + 1)}")  # +1 чтобы индексы были 0,1,2
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Определяем веса классов
        if class_weight is not None:
            # Используем переданные кастомные веса
            class_weights = class_weight
            print(f"  Using custom class weights: {class_weights}")
        else:
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
            print(f"  Using auto-balanced class weights: {class_weights}")
        
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
        class_weight: Optional[Dict[int, float]] = None,
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
            class_weight: Кастомные веса классов (если None, используется автоматическая балансировка)
        
        Returns:
            (model, metrics) - обученная модель и метрики
        """
        print(f"[model_trainer] Training XGBoost Classifier...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        # XGBoost может работать с ненормализованными данными, но нормализуем для консистентности
        X_scaled = self.scaler.fit_transform(X)
        
        # Преобразуем y для XGBoost (нужны индексы 0,1,2 вместо -1,0,1)
        y_xgb = y + 1  # -1,0,1 -> 0,1,2
        
        # Вычисляем веса образцов для XGBoost
        sample_weights = np.zeros(len(y_xgb))
        
        if class_weight is not None:
            # Используем переданные кастомные веса
            # Конвертируем класс-веса в веса образцов
            for orig_cls, weight in class_weight.items():
                xgb_cls = orig_cls + 1  # Преобразуем -1,0,1 -> 0,1,2
                sample_weights[y_xgb == xgb_cls] = weight
            print(f"  Using custom class weights (converted to sample_weights)")
        else:
            # Вычисляем веса классов для XGBoost (для классов 0,1,2)
            unique_classes, class_counts = np.unique(y_xgb, return_counts=True)
            total_samples = len(y_xgb)
            
            for cls, count in zip(unique_classes, class_counts):
                if count > 0:
                    base_weight = total_samples / (len(unique_classes) * count)
                    # Класс 1 (HOLD) - индекс 1 в XGBoost формате
                    if cls == 1:  # HOLD - уменьшаем вес
                        weight = base_weight * 0.8
                    else:  # LONG (2) или SHORT (0) - увеличиваем вес
                        weight = base_weight * 1.5
                    sample_weights[y_xgb == cls] = weight
            print(f"  Using auto-balanced sample weights")
        
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
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "cv_f1_mean": metrics.get("cv_f1_mean", 0.0),
                "rf_weight": metrics.get("rf_weight", None),  # Веса ансамбля (только для ансамблей)
                "xgb_weight": metrics.get("xgb_weight", None),
                "ensemble_method": metrics.get("ensemble_method", None),
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
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rf_n_estimators: int = 100,
        rf_max_depth: Optional[int] = 10,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        ensemble_method: str = "voting",  # "voting" или "weighted_average"
        random_state: int = 42,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Обучает ансамбль из RandomForest и XGBoost.
        
        Args:
            X: Матрица фичей
            y: Целевая переменная (1 = LONG, -1 = SHORT, 0 = HOLD)
            rf_n_estimators: Количество деревьев для RandomForest
            rf_max_depth: Максимальная глубина для RandomForest
            xgb_n_estimators: Количество деревьев для XGBoost
            xgb_max_depth: Максимальная глубина для XGBoost
            xgb_learning_rate: Скорость обучения для XGBoost
            ensemble_method: Метод ансамбля ("voting" или "weighted_average")
            random_state: Seed для воспроизводимости
            class_weight: Кастомные веса классов (передаются в обе модели)
        
        Returns:
            (ensemble_model, metrics) - обученный ансамбль и метрики
        """
        print(f"[model_trainer] Training Ensemble Model ({ensemble_method})...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Class distribution: {Counter(y)}")
        
        # Нормализуем фичи
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем отдельные модели
        print(f"\n  [1/2] Training RandomForest...")
        rf_model, rf_metrics = self.train_random_forest_classifier(
            X, y,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            class_weight=class_weight,  # Передаем веса классов
        )
        
        print(f"\n  [2/2] Training XGBoost...")
        xgb_model, xgb_metrics = self.train_xgboost_classifier(
            X, y,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=random_state,
            class_weight=class_weight,  # Передаем веса классов
        )
        
        # Вычисляем веса на основе CV метрик для обоих методов
        rf_cv_score = rf_metrics.get("cv_mean", 0.5)
        xgb_cv_score = xgb_metrics.get("cv_mean", 0.5)
        total_score = rf_cv_score + xgb_cv_score
        if total_score > 0:
            rf_weight = rf_cv_score / total_score
            xgb_weight = xgb_cv_score / total_score
        else:
            rf_weight = xgb_weight = 0.5
        
        # Создаем ансамбль
        if ensemble_method == "voting":
            # Используем класс, определенный на уровне модуля
            ensemble = PreTrainedVotingEnsemble(rf_model, xgb_model, rf_weight, xgb_weight)
            
        elif ensemble_method == "weighted_average":
            # Используем класс, определенный на уровне модуля
            ensemble = WeightedEnsemble(rf_model, xgb_model, rf_weight, xgb_weight)
        
        else:
            raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
        
        print(f"  Ensemble weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")
        
        # Улучшенная валидация: Walk-Forward Validation
        print(f"\n  Performing Walk-Forward Validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Обучаем модели на fold
                # Обучаем RF на fold
                rf_fold = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                rf_fold.fit(X_train_fold, y_train_fold)
                
                # Обучаем XGBoost на fold
                y_train_xgb = y_train_fold + 1
                xgb_fold = xgb.XGBClassifier(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                )
                xgb_fold.fit(X_train_fold, y_train_xgb)
                
                # Создаем ансамбль для fold
                if ensemble_method == "voting":
                    ensemble_fold = PreTrainedVotingEnsemble(rf_fold, xgb_fold, rf_weight, xgb_weight)
                else:
                    ensemble_fold = WeightedEnsemble(rf_fold, xgb_fold, rf_weight, xgb_weight)
                
                y_pred_fold = ensemble_fold.predict(X_val_fold)
                
                # Метрики для fold
                fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val_fold, y_pred_fold, average='weighted', zero_division=0
                )
                
                cv_scores.append(fold_accuracy)
                cv_precision.append(precision)
                cv_recall.append(recall)
                cv_f1.append(f1)
                
                print(f"    Fold {fold + 1}: Accuracy={fold_accuracy:.4f}, F1={f1:.4f}")
        
        # Предсказания на всех данных
        y_pred = ensemble.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        
        # Метрики
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_precision_mean": np.mean(cv_precision),
            "cv_recall_mean": np.mean(cv_recall),
            "cv_f1_mean": np.mean(cv_f1),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "rf_metrics": rf_metrics,
            "xgb_metrics": xgb_metrics,
            "ensemble_method": ensemble_method,
            "rf_weight": rf_weight,  # Веса ансамбля
            "xgb_weight": xgb_weight,
        }
        
        print(f"\n[model_trainer] Ensemble training completed:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
        print(f"  CV F1-Score: {metrics['cv_f1_mean']:.4f}")
        
        return ensemble, metrics

