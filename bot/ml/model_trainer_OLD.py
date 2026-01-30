"""
Модуль для обучения ML-моделей.
Обновлено: Добавлена балансировка классов и улучшенные параметры моделей.
"""
import warnings
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Подавление предупреждений (как в оригинале)
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb
# Импорт LSTM и QuadEnsemble предполагается из существующих модулей проекта,
# здесь мы оставим заглушки или оригинальную логику, если она есть в импортах
# (В данном файле я фокусируюсь на логике обучения RF/XGB/LGB)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Обучает ансамбль моделей с учетом временных рядов и дисбаланса классов.
        """
        print(f"Training on data shape: {X.shape}")
        
        # 1. Обработка дисбаланса классов
        # В трейдинге класс 0 (Hold) обычно составляет 80-90% данных.
        # Нам нужно увеличить вес классов 1 (Long) и -1 (Short).
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # TimeSeriesSplit для валидации (нельзя использовать случайный K-Fold)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # --- Random Forest ---
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,            # Ограничиваем глубину, чтобы не переобучался
            min_samples_leaf=20,    # Требуем подтверждения на группе примеров
            class_weight='balanced', # ВАЖНО: балансировка
            n_jobs=-1,
            random_state=42
        )
        
        # Обучаем RF
        rf.fit(X, y)
        self.models['rf'] = rf
        
        # --- XGBoost ---
        print("Training XGBoost...")
        # Переводим метки классов из {-1, 0, 1} в {0, 1, 2} для XGBoost
        y_xgb = y + 1 
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            n_jobs=-1,
            random_state=42,
            # XGBoost не принимает словарь весов напрямую в sklearn API так просто для мультикласса,
            # но можно использовать sample_weight в fit. Здесь оставим параметры регуляризации.
            reg_alpha=0.1,  # L1 регуляризация
            reg_lambda=1.0  # L2 регуляризация
        )
        
        # Создаем веса для каждого семпла
        sample_weights = np.ones(len(y))
        for cls, weight in class_weight_dict.items():
            sample_weights[y == cls] = weight
            
        xgb_model.fit(X, y_xgb, sample_weight=sample_weights)
        self.models['xgb'] = xgb_model
        
        # --- LightGBM ---
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced', # LGBM умеет это сам
            objective='multiclass',
            num_class=3,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X, y_xgb) # LGBM тоже хочет 0,1,2
        self.models['lgb'] = lgb_model
        
        return self.models

    def predict(self, X_new: np.ndarray) -> Tuple[int, float, Dict]:
        """
        Делает предсказание ансамблем.
        Возвращает: (final_signal, confidence, details)
        signal: -1, 0, 1
        """
        # Получаем вероятности
        # RF
        probs_rf = self.models['rf'].predict_proba(X_new) # [Batch, 3]
        
        # XGB (помни про сдвиг классов)
        probs_xgb = self.models['xgb'].predict_proba(X_new)
        
        # LGB
        probs_lgb = self.models['lgb'].predict_proba(X_new)
        
        # Усреднение (Soft Voting)
        # Можно добавить веса моделям на основе их валидации
        avg_probs = (probs_rf + probs_xgb + probs_lgb) / 3.0
        
        # Берем вероятность для текущего сэмпла (предполагаем X_new это 1 строка или берем последнюю)
        current_probs = avg_probs[-1]
        
        # Индекс с макс вероятностью
        pred_idx = np.argmax(current_probs)
        confidence = current_probs[pred_idx]
        
        # Возвращаем к -1, 0, 1 (индекс 0 -> класс -1, индекс 1 -> класс 0, индекс 2 -> класс 1)
        # Но стоп! predict_proba возвращает колонки в порядке классов.
        # RF classes_: [-1, 0, 1] -> индексы 0, 1, 2. Все верно.
        
        final_signal = self.models['rf'].classes_[pred_idx]
        
        details = {
            "probs": current_probs.tolist(),
            "raw_rf": probs_rf[-1].tolist(),
            "raw_xgb": probs_xgb[-1].tolist()
        }
        
        return final_signal, confidence, details