"""
ML-стратегия для торгового бота.
Использует обученную ML-модель для генерации торговых сигналов.
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
# Подавляем предупреждения XGBoost про pickle и версии
warnings.filterwarnings('ignore', message='.*loading a serialized model.*')
warnings.filterwarnings('ignore', message='.*XGBoost.*')
os.environ['XGB_SILENT'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from bot.strategy import Action, Bias, Signal
from bot.ml.feature_engineering import FeatureEngineer
from bot.config import StrategyParams
# Импортируем классы ансамбля для корректной десериализации pickle
from bot.ml.model_trainer import PreTrainedVotingEnsemble, WeightedEnsemble, TripleEnsemble


class MLStrategy:
    """
    ML-стратегия, использующая обученную модель для предсказания движения цены.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, min_signal_strength: str = "слабое", stability_filter: bool = True, use_dynamic_threshold: bool = True, min_signals_per_day: int = 1, max_signals_per_day: int = 10):
        """
        Инициализирует ML-стратегию.
        
        Args:
            model_path: Путь к сохраненной модели (.pkl файл)
            confidence_threshold: Минимальная уверенность модели для открытия позиции (0-1)
            min_signal_strength: Минимальная сила сигнала ("слабое", "умеренное", "среднее", "сильное", "очень_сильное")
            stability_filter: Фильтр стабильности - требовать более высокую уверенность для смены направления
            use_dynamic_threshold: Использовать динамические пороги на основе рыночных условий
            min_signals_per_day: Минимальное количество сигналов в день (гарантирует хотя бы 1 сигнал)
            max_signals_per_day: Максимальное количество сигналов в день (ограничивает избыточную торговлю)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_signal_strength = min_signal_strength
        self.stability_filter = stability_filter
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Определяем минимальный порог уверенности на основе силы сигнала
        strength_thresholds = {
            "слабое": 0.0,
            "умеренное": 0.6,
            "среднее": 0.7,
            "сильное": 0.8,
            "очень_сильное": 0.9
        }
        self.min_strength_threshold = strength_thresholds.get(min_signal_strength, 0.6)
        
        # История уверенности для адаптивных порогов
        self.confidence_history = []
        self.max_history_size = 100
        
        # История последних сигналов для предотвращения противоречивых сигналов
        # Хранит последние N сигналов: [(timestamp, action, confidence), ...]
        self.signal_history = []
        self.max_signal_history = 20  # Храним последние 20 сигналов
        self.min_bars_between_opposite_signals = 4  # Минимум баров между противоположными сигналами
        self.min_confidence_difference = 0.15  # Минимальная разница уверенности между LONG и SHORT (15%)
        
        # Отслеживание сигналов в день для ограничения количества
        # Хранит количество сигналов по датам: {date_str: count}
        self.daily_signals_count = {}
        self.min_signals_per_day = min_signals_per_day
        self.max_signals_per_day = max_signals_per_day
        
        # Загружаем модель
        self.model_data = self._load_model()
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_names = self.model_data["feature_names"]
        self.is_ensemble = self.model_data.get("metadata", {}).get("model_type", "").startswith("ensemble")
        
        # Если это QuadEnsemble, восстанавливаем feature_names в lstm_trainer
        if hasattr(self.model, 'lstm_trainer') and self.model.lstm_trainer is not None:
            # Если feature_names не установлены в lstm_trainer, пытаемся восстановить
            if not hasattr(self.model.lstm_trainer, 'feature_names') or self.model.lstm_trainer.feature_names is None:
                # Пытаемся определить из scaler (количество фичей)
                if hasattr(self.model.lstm_trainer, 'scaler') and self.model.lstm_trainer.scaler is not None:
                    expected_features = self.model.lstm_trainer.scaler.n_features_in_ if hasattr(self.model.lstm_trainer.scaler, 'n_features_in_') else None
                    if expected_features and self.feature_names:
                        # Используем первые expected_features фичей (как при обучении LSTM)
                        # LSTM обычно использует первые N фичей (например, 50)
                        self.model.lstm_trainer.feature_names = self.feature_names[:expected_features]
                        if not hasattr(self, '_lstm_feature_names_restored'):
                            print(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features")
                            self._lstm_feature_names_restored = True
                    elif self.feature_names:
                        # Если не можем определить из scaler, используем все feature_names
                        self.model.lstm_trainer.feature_names = self.feature_names
                        if not hasattr(self, '_lstm_feature_names_restored'):
                            print(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features (from all features)")
                            self._lstm_feature_names_restored = True
                elif self.feature_names:
                    # Если scaler недоступен, используем все feature_names
                    self.model.lstm_trainer.feature_names = self.feature_names
                    if not hasattr(self, '_lstm_feature_names_restored'):
                        print(f"[ml_strategy] Restored LSTM feature_names: {len(self.model.lstm_trainer.feature_names)} features (scaler unavailable)")
                        self._lstm_feature_names_restored = True
        
        # Инициализируем feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Извлекаем символ из пути к модели для логирования
        model_filename = Path(model_path).name
        symbol_from_model = "UNKNOWN"
        if "_" in model_filename:
            parts = model_filename.replace(".pkl", "").split("_")
            # Форматы:
            # - rf_ETHUSDT_15_15m.pkl -> ["rf","ETHUSDT","15","15m"]
            # - ensemble_BTCUSDT_15_mtf.pkl -> ["ensemble","BTCUSDT","15","mtf"]
            # - triple_ensemble_BTCUSDT_15_15m.pkl -> ["triple","ensemble","BTCUSDT","15","15m"]
            # - quad_ensemble_BTCUSDT_15_mtf.pkl -> ["quad","ensemble","BTCUSDT","15","mtf"]
            if len(parts) >= 3 and parts[0] in ("triple", "quad") and parts[1] == "ensemble":
                symbol_from_model = parts[2]
            elif len(parts) >= 2:
                symbol_from_model = parts[1]
        
        # Получаем метаданные модели
        model_metadata = self.model_data.get("metadata", {})
        model_type_str = model_metadata.get("model_type", "unknown")
        if "ensemble" in model_type_str.lower():
            self.is_ensemble = True
        
        # Компактный лог загрузки модели (только при первой загрузке)
        if not hasattr(self, '_model_loaded_logged'):
            model_type = '🎯 ENSEMBLE' if self.is_ensemble else 'Single'
            cv_acc = self.model_data.get("metrics", {}).get('cv_mean', 0) if self.is_ensemble else 0
            print(f"[ml] {symbol_from_model}: {model_type} (CV:{cv_acc:.3f}, conf:{confidence_threshold}, stab:{stability_filter})")
            self._model_loaded_logged = True
    
    def _load_model(self) -> Dict[str, Any]:
        """Загружает модель из файла."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        return model_data
    
    def prepare_features(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> np.ndarray:
        """
        Подготавливает фичи из DataFrame для предсказания модели.
        
        Args:
            df: DataFrame с OHLCV данными и индикаторами (может уже содержать фичи)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            Массив фичей для модели
        """
        # Если фичи уже созданы (skip_feature_creation=True), используем их напрямую
        if skip_feature_creation:
            df_with_features = df.copy()
        else:
            # Создаем фичи заново (для обратной совместимости)
            # Проверяем, есть ли timestamp как колонка (нужно для feature_engineer)
            df_work = df.copy()
            if "timestamp" in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                df_work = df_work.set_index("timestamp")
            elif "timestamp" not in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                # Если нет timestamp, создаем его из индекса
                if isinstance(df_work.index, pd.DatetimeIndex):
                    pass  # Уже DatetimeIndex
                else:
                    # Пытаемся создать временной индекс
                    df_work.index = pd.to_datetime(df_work.index, errors='coerce')
            
            # Создаем все необходимые фичи через FeatureEngineer
            print(f"[ml_strategy] Preparing features: input DataFrame has {len(df_work)} rows")
            try:
                df_with_features = self.feature_engineer.create_technical_indicators(df_work)
                print(f"[ml_strategy] After create_technical_indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
            except TypeError as e:
                if "'>' not supported" in str(e) or "NoneType" in str(e):
                    print(f"[ml_strategy] ❌ ERROR: Comparison with None detected in create_technical_indicators")
                    print(f"[ml_strategy]   Error: {e}")
                    print(f"[ml_strategy]   Checking for None values in DataFrame...")
                    # Проверяем наличие None в ключевых колонках
                    for col in ["open", "high", "low", "close", "volume", "atr", "atr_pct", "rsi"]:
                        if col in df_work.columns:
                            none_count = df_work[col].isna().sum() + (df_work[col] == None).sum()
                            if none_count > 0:
                                print(f"[ml_strategy]   Column '{col}' has {none_count} None/NaN values")
                    raise
                raise
        
        # Проверяем, что есть хотя бы основные данные (OHLCV)
        key_columns = ["open", "high", "low", "close", "volume"]
        if all(col in df_with_features.columns for col in key_columns):
            # Сохраняем только строки, где хотя бы основные колонки присутствуют
            rows_before = len(df_with_features)
            df_with_features = df_with_features[df_with_features[key_columns].notna().any(axis=1)]
            rows_after = len(df_with_features)
            # Логируем только если количество строк изменилось И это не skip_feature_creation (чтобы не засорять логи)
            if not skip_feature_creation and rows_before != rows_after:
                print(f"[ml_strategy] After filtering key columns: {rows_before} -> {rows_after} rows")
        else:
            # Логируем предупреждение только если это не skip_feature_creation
            if not skip_feature_creation:
                missing_key_cols = [col for col in key_columns if col not in df_with_features.columns]
                print(f"[ml_strategy] ⚠️ WARNING: Missing key columns: {missing_key_cols}")
        
        # Проверяем, что есть данные после фильтрации основных колонок
        if len(df_with_features) == 0:
            print(f"[ml_strategy] ❌ ERROR: No rows after filtering key columns")
            print(f"[ml_strategy]   Input DataFrame shape: {df_work.shape}")
            print(f"[ml_strategy]   After create_technical_indicators shape: {df_with_features.shape if 'df_with_features' in locals() else 'N/A'}")
            raise ValueError("No data available after creating features (all rows contain NaN in key columns)")
        
        # ВАЖНО: Заполняем NaN в фичах нулями ПЕРЕД любыми другими операциями
        # Это позволяет сохранить все строки, даже если некоторые индикаторы не вычислились
        # Сначала заполняем NaN в индикаторах (но не в основных колонках)
        feature_columns = [col for col in df_with_features.columns if col not in key_columns]
        if feature_columns:
            df_with_features[feature_columns] = df_with_features[feature_columns].fillna(0)
        
        # Удаляем только строки, где ВСЕ значения (включая основные колонки) NaN
        df_with_features = df_with_features.dropna(how='all')
        
        # Финальная проверка
        if len(df_with_features) == 0:
            raise ValueError("No data available after creating features (all rows contain NaN)")
        
        # Проверяем наличие всех необходимых фичей
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            # Выводим только один раз при первом обнаружении
            if not hasattr(self, "_missing_features_warned"):
                print(
                    f"[ml_strategy] ⚠️ WARNING: Missing {len(missing_features)} features: "
                    f"{missing_features[:10]}..."
                )
                print(
                    f"[ml_strategy]   Expected {len(self.feature_names)} features, "
                    f"got {len(df_with_features.columns)}"
                )
                self._missing_features_warned = True
            
            # Заполняем отсутствующие фичи нулями одним батчем, чтобы избежать фрагментации DataFrame
            zeros_df = pd.DataFrame(
                0.0,
                index=df_with_features.index,
                columns=missing_features,
            )
            df_with_features = pd.concat([df_with_features, zeros_df], axis=1)
        
        # Проверяем лишние фичи (которые есть в данных, но не ожидаются моделью)
        extra_features = [f for f in df_with_features.columns if f not in self.feature_names and f not in key_columns]
        # Убираем логи о лишних фичах - это нормальная ситуация (они просто игнорируются)
        if extra_features:
            self._extra_features_warned = True  # Устанавливаем флаг, но не логируем
        
        # Выбираем только нужные фичи в правильном порядке
        X = df_with_features[self.feature_names].values
        
        # Проверяем, что есть данные для нормализации
        if len(X) == 0:
            raise ValueError("No data available after feature selection")
        
        # Проверяем соответствие количества фичей с моделью
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature count mismatch: X has {X.shape[1]} features, but model expects {len(self.feature_names)}")
        
        # Нормализуем
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            if "features" in str(e).lower() or "n_features" in str(e).lower():
                # Пробуем исправить несоответствие количества фичей
                scaler_expected = getattr(self.scaler, 'n_features_in_', None)
                if scaler_expected is None:
                    # Старая версия sklearn - пробуем получить из shape
                    try:
                        scaler_expected = self.scaler.mean_.shape[0] if hasattr(self.scaler, 'mean_') else None
                    except:
                        pass
                
                if scaler_expected and X.shape[1] != scaler_expected:
                    # Автоматически исправляем несоответствие без логирования (это нормальная ситуация)
                    if not hasattr(self, '_feature_mismatch_warned'):
                        self._feature_mismatch_warned = True
                    
                    # Если scaler ожидает больше фичей, добавляем недостающие нулями
                    if X.shape[1] < scaler_expected:
                        missing_count = scaler_expected - X.shape[1]
                        if not hasattr(self, '_feature_adjustment_logged'):
                            self._feature_adjustment_logged = True
                        # Добавляем нулевые колонки
                        zeros = np.zeros((X.shape[0], missing_count))
                        X = np.hstack([X, zeros])
                    # Если scaler ожидает меньше фичей, обрезаем
                    elif X.shape[1] > scaler_expected:
                        X = X[:, :scaler_expected]
                
                # Пробуем снова после исправления
                try:
                    X_scaled = self.scaler.transform(X)
                except ValueError as e2:
                    print(f"[ml_strategy] ❌ ERROR: Still cannot transform after adjustment")
                    print(f"[ml_strategy]   Scaler expects: {scaler_expected} features")
                    print(f"[ml_strategy]   X has: {X.shape[1]} features")
                    raise ValueError(f"Feature count mismatch: Scaler expects {scaler_expected} features, but got {X.shape[1]}. "
                                   f"Please retrain the model with the current feature set.") from e2
            else:
                raise
        
        return X_scaled
    
    def prepare_features_with_df(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Подготавливает фичи из DataFrame и возвращает как массив, так и DataFrame с фичами.
        
        Args:
            df: DataFrame с OHLCV данными и индикаторами (может уже содержать фичи)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            (X_scaled, df_with_features) где:
            - X_scaled: Нормализованный массив фичей для модели
            - df_with_features: DataFrame со всеми фичами (для передачи в QuadEnsemble)
        """
        # Если фичи уже созданы (skip_feature_creation=True), используем их напрямую
        if skip_feature_creation:
            df_with_features = df.copy()
        else:
            # Создаем фичи заново (для обратной совместимости)
            # Проверяем, есть ли timestamp как колонка (нужно для feature_engineer)
            df_work = df.copy()
            if "timestamp" in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                df_work = df_work.set_index("timestamp")
            elif "timestamp" not in df_work.columns and not isinstance(df_work.index, pd.DatetimeIndex):
                # Если нет timestamp, создаем его из индекса
                if isinstance(df_work.index, pd.DatetimeIndex):
                    pass  # Уже DatetimeIndex
                else:
                    # Пытаемся создать временной индекс
                    df_work.index = pd.to_datetime(df_work.index, errors='coerce')
            
            # Создаем все необходимые фичи через FeatureEngineer
            if not skip_feature_creation:
                print(f"[ml_strategy] Preparing features: input DataFrame has {len(df_work)} rows")
            try:
                df_with_features = self.feature_engineer.create_technical_indicators(df_work)
                if not skip_feature_creation:
                    print(f"[ml_strategy] After create_technical_indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
            except TypeError as e:
                if "'>' not supported" in str(e) or "NoneType" in str(e):
                    print(f"[ml_strategy] ❌ ERROR: Comparison with None detected in create_technical_indicators")
                    print(f"[ml_strategy]   Error: {e}")
                    raise
                raise
        
        # Проверяем, что есть хотя бы основные данные (OHLCV)
        key_columns = ["open", "high", "low", "close", "volume"]
        if all(col in df_with_features.columns for col in key_columns):
            rows_before = len(df_with_features)
            df_with_features = df_with_features[df_with_features[key_columns].notna().any(axis=1)]
            rows_after = len(df_with_features)
        else:
            missing_key_cols = [col for col in key_columns if col not in df_with_features.columns]
            raise ValueError(f"Missing key columns: {missing_key_cols}")
        
        if len(df_with_features) == 0:
            raise ValueError("No data available after filtering key columns")
        
        # Заполняем NaN в фичах
        feature_columns = [col for col in df_with_features.columns if col not in key_columns]
        if feature_columns:
            df_with_features[feature_columns] = df_with_features[feature_columns].ffill().bfill().fillna(0.0)
        
        # Проверяем наличие всех необходимых фичей
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            # Заполняем отсутствующие фичи нулями
            zeros_df = pd.DataFrame(
                0.0,
                index=df_with_features.index,
                columns=missing_features,
            )
            df_with_features = pd.concat([df_with_features, zeros_df], axis=1)
        
        # Выбираем только нужные фичи в правильном порядке
        X = df_with_features[self.feature_names].values
        
        if len(X) == 0:
            raise ValueError("No data available after feature selection")
        
        # Нормализуем
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            if "features" in str(e).lower() or "n_features" in str(e).lower():
                scaler_expected = getattr(self.scaler, 'n_features_in_', None)
                if scaler_expected is None:
                    try:
                        scaler_expected = self.scaler.mean_.shape[0] if hasattr(self.scaler, 'mean_') else None
                    except:
                        pass
                
                if scaler_expected and X.shape[1] != scaler_expected:
                    if X.shape[1] < scaler_expected:
                        missing_count = scaler_expected - X.shape[1]
                        zeros = np.zeros((X.shape[0], missing_count))
                        X = np.hstack([X, zeros])
                    elif X.shape[1] > scaler_expected:
                        X = X[:, :scaler_expected]
                    
                    X_scaled = self.scaler.transform(X)
                else:
                    raise
            else:
                raise
        
        return X_scaled, df_with_features
    
    def predict(self, df: pd.DataFrame, skip_feature_creation: bool = False) -> tuple[int, float]:
        """
        Делает предсказание на основе последнего бара.
        
        Args:
            df: DataFrame с данными (OHLCV, фичи будут созданы автоматически или уже присутствуют)
            skip_feature_creation: Если True, пропускает создание фичей (предполагается, что они уже созданы)
        
        Returns:
            (prediction, confidence) где:
            - prediction: 1 (LONG), -1 (SHORT), 0 (HOLD)
            - confidence: уверенность модели (0-1)
        """
        # Берем последний бар
        if len(df) == 0:
            return 0, 0.0
        
        try:
            # Подготавливаем фичи (создаст все необходимые индикаторы или использует уже созданные)
            # Нужно получить и X (массив фичей) и df_with_features (DataFrame с фичами) для QuadEnsemble
            X, df_with_features = self.prepare_features_with_df(df, skip_feature_creation=skip_feature_creation)
            
            # Берем последний образец
            X_last = X[-1:].reshape(1, -1)
        except Exception as e:
            print(f"[ml_strategy] Error preparing features: {e}")
            return 0, 0.0
        
        # Предсказание
        if hasattr(self.model, "predict_proba"):
            # Для классификаторов с вероятностями (включая ансамбль)
            # Проверяем, является ли это QuadEnsemble (требует историю для LSTM)
            if hasattr(self.model, 'lstm_trainer') and hasattr(self.model, 'sequence_length'):
                # QuadEnsemble: передаем историю данных для LSTM
                # Используем df_with_features, который уже содержит все фичи
                proba = self.model.predict_proba(X_last, df_history=df_with_features)[0]
            else:
                # Обычные модели и ансамбли (TripleEnsemble, etc.)
                proba = self.model.predict_proba(X_last)[0]
            
            # Проверяем proba на NaN
            if np.any(np.isnan(proba)) or not np.all(np.isfinite(proba)):
                # Если proba содержит NaN, используем равномерное распределение
                proba = np.array([0.33, 0.34, 0.33])  # SHORT, HOLD, LONG
                print(f"[ml_strategy] Warning: proba contains NaN, using uniform distribution")
            
            # Для ансамбля proba уже в правильном формате [-1, 0, 1]
            # Для XGBoost нужно преобразовать из [0, 1, 2]
            if self.is_ensemble:
                # Ансамбль уже возвращает вероятности в формате [-1, 0, 1]
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
                # Проверяем на NaN
                if np.isnan(long_prob) or not np.isfinite(long_prob):
                    long_prob = 0.0
                if np.isnan(short_prob) or not np.isfinite(short_prob):
                    short_prob = 0.0
                if np.isnan(hold_prob) or not np.isfinite(hold_prob):
                    hold_prob = 0.0
                
                # УЛУЧШЕННАЯ ЛОГИКА ДЛЯ АНСАМБЛЕЙ: Требуем более высокую уверенность и разницу между LONG/SHORT
                # Повышаем минимальный порог для ансамблей (было 0.1%, теперь 0.3%)
                ensemble_absolute_min = 0.003  # Минимальная абсолютная уверенность 0.3% (повышено для фильтрации слабых сигналов)
                
                # Вычисляем разницу между LONG и SHORT
                prob_diff = abs(long_prob - short_prob)
                
                # Определяем предсказание: выбираем LONG или SHORT только если:
                # 1. Вероятность выше минимума
                # 2. Разница между LONG и SHORT достаточна (минимум min_confidence_difference)
                # 3. Вероятность выше противоположной
                if long_prob >= ensemble_absolute_min and long_prob > short_prob and prob_diff >= self.min_confidence_difference:
                    # LONG выше SHORT, выше минимума и разница достаточна - принимаем LONG
                    prediction = 1  # LONG
                    # Используем реальную вероятность LONG, но учитываем разницу
                    # Чем больше разница, тем выше уверенность (но не превышаем long_prob)
                    confidence = min(long_prob * (1 + prob_diff * 0.3), long_prob)
                    # Проверяем результат на NaN
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = long_prob
                elif short_prob >= ensemble_absolute_min and short_prob > long_prob and prob_diff >= self.min_confidence_difference:
                    # SHORT выше LONG, выше минимума и разница достаточна - принимаем SHORT
                    prediction = -1  # SHORT
                    # Используем реальную вероятность SHORT, но учитываем разницу
                    confidence = min(short_prob * (1 + prob_diff * 0.3), short_prob)
                    # Проверяем результат на NaN
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = short_prob
                else:
                    # HOLD - либо LONG и SHORT ниже минимума, либо разница недостаточна
                    prediction = 0
                    confidence = hold_prob
                
                # Fallback: если логика не сработала, используем стандартную
                # НО только если prediction действительно 0 (HOLD)
                # Если prediction уже установлен (LONG или SHORT), не переопределяем его
                if prediction == 0:
                    prediction_idx = np.argmax(proba)
                    prediction = prediction_idx - 1  # 0->-1, 1->0, 2->1
                    confidence = proba[prediction_idx]
                    # Проверяем на NaN
                    if np.isnan(confidence) or not np.isfinite(confidence):
                        confidence = hold_prob if np.isfinite(hold_prob) else 0.0
                    # Убеждаемся, что confidence не превышает реальную вероятность
                    if prediction == 1:  # LONG
                        confidence = min(confidence, long_prob)
                    elif prediction == -1:  # SHORT
                        confidence = min(confidence, short_prob)
                    else:  # HOLD
                        confidence = min(confidence, hold_prob)
                
                # Обновляем историю уверенности
                if len(self.confidence_history) >= self.max_history_size:
                    self.confidence_history.pop(0)
                self.confidence_history.append(confidence)
            elif len(proba) == 3:
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1  # 0->-1, 1->0, 2->1
                confidence = proba[prediction_idx]
                
                # Проверяем confidence на NaN
                if np.isnan(confidence) or not np.isfinite(confidence):
                    confidence = 0.0
                
                # УЛУЧШЕНИЕ: Если модель предсказывает HOLD, но вероятность LONG или SHORT достаточно высока,
                # используем эту вероятность для генерации сигнала
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
                # Проверяем на NaN
                if np.isnan(long_prob) or not np.isfinite(long_prob):
                    long_prob = 0.0
                if np.isnan(short_prob) or not np.isfinite(short_prob):
                    short_prob = 0.0
                if np.isnan(hold_prob) or not np.isfinite(hold_prob):
                    hold_prob = 0.0
                
                # Динамический порог на основе истории уверенности
                if self.use_dynamic_threshold and len(self.confidence_history) > 10:
                    # Вычисляем адаптивный порог на основе медианы последних уверенностей
                    recent_confidence_median = np.median(self.confidence_history[-20:])
                    # Если текущая уверенность выше медианы, используем более мягкий порог
                    adaptive_threshold = max(self.min_strength_threshold, recent_confidence_median * 0.9)
                else:
                    adaptive_threshold = self.min_strength_threshold
                
                # Если HOLD имеет максимальную вероятность, но LONG или SHORT имеют достаточно высокую вероятность,
                # используем их для генерации сигнала (если они превышают адаптивный порог)
                if prediction == 0:  # HOLD
                    # Используем адаптивный порог для переопределения HOLD
                    # Только если вероятность LONG или SHORT >= adaptive_threshold, переопределяем HOLD
                    if long_prob >= adaptive_threshold and long_prob > short_prob:
                        prediction = 1  # LONG
                        confidence = long_prob
                    elif short_prob >= adaptive_threshold and short_prob > long_prob:
                        prediction = -1  # SHORT
                        confidence = short_prob
                    # Иначе остаемся на HOLD
                
                # Обновляем историю уверенности
                if len(self.confidence_history) >= self.max_history_size:
                    self.confidence_history.pop(0)
                self.confidence_history.append(confidence)
            else:
                # Для других форматов
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1 if len(proba) == 3 else prediction_idx
                confidence = proba[prediction_idx]
                
                # Проверяем на NaN
                if np.isnan(prediction) or not np.isfinite(prediction):
                    prediction = 0
                if np.isnan(confidence) or not np.isfinite(confidence):
                    confidence = 0.0
        else:
            # Для моделей без predict_proba
            prediction_raw = self.model.predict(X_last)[0]
            # Проверяем на NaN перед преобразованием
            if np.isnan(prediction_raw) or not np.isfinite(prediction_raw):
                prediction = 0  # HOLD если prediction_raw NaN
            else:
                # Преобразуем в формат -1, 0, 1 если нужно
                if hasattr(self.model, 'classes_'):
                    # Если есть classes_, преобразуем индекс в значение
                    classes = self.model.classes_
                    if len(classes) == 3:
                        prediction = int(prediction_raw) - 1  # 0->-1, 1->0, 2->1
                    else:
                        prediction = int(prediction_raw)
                else:
                    prediction = int(prediction_raw)
            confidence = 1.0  # Нет информации об уверенности
        
        # Проверяем на NaN перед возвратом
        if np.isnan(prediction) or not np.isfinite(prediction):
            prediction = 0  # HOLD если prediction NaN
        if np.isnan(confidence) or not np.isfinite(confidence):
            confidence = 0.0  # Нулевая уверенность если confidence NaN
        
        return int(prediction), float(confidence)
    
    def generate_signal(
        self,
        row: pd.Series,
        df: pd.DataFrame,
        has_position: Optional[Bias],
        current_price: float,
        leverage: int = 10,
        target_profit_pct_margin: float = 25.0,
        max_loss_pct_margin: float = 10.0,
    ) -> Signal:
        """
        Генерирует торговый сигнал на основе ML-предсказания.
        
        Args:
            row: Текущий бар (pd.Series)
            df: DataFrame со всеми данными (для создания фичей)
            has_position: Текущая позиция (None, Bias.LONG, Bias.SHORT)
            current_price: Текущая цена
            leverage: Плечо для расчета TP/SL
            target_profit_pct_margin: Целевая прибыль от маржи в % (20-30%)
            max_loss_pct_margin: Максимальный убыток от маржи в %
        
        Returns:
            Signal объект
        """
        try:
            # Определяем символ из model_path для адаптивных фильтров
            symbol = getattr(self, '_symbol', None)
            if symbol is None:
                # Извлекаем символ из пути к модели
                model_filename = Path(self.model_path).name
                if "_" in model_filename:
                    parts = model_filename.replace(".pkl", "").split("_")
                    if len(parts) >= 3 and parts[0] in ("triple", "quad") and parts[1] == "ensemble":
                        symbol = parts[2].upper()
                        self._symbol = symbol
                    elif len(parts) >= 2:
                        symbol = parts[1].upper()  # Например, rf_ETHUSDT_15.pkl -> ETHUSDT
                        self._symbol = symbol
                    else:
                        symbol = "UNKNOWN"
                else:
                    symbol = "UNKNOWN"
            
            # Адаптивные пороги для разных символов
            is_volatile_symbol = symbol in ("ETHUSDT", "SOLUSDT")
            # Делаем предсказание (пропускаем создание фичей, так как они уже созданы в build_ml_signals)
            prediction, confidence = self.predict(df, skip_feature_creation=True)
            
            # Рассчитываем TP/SL в процентах от цены для достижения целевой прибыли от маржи
            # Если прибыль от маржи = 25%, а плечо = 10x, то TP = 25% / 10 = 2.5%
            tp_pct = target_profit_pct_margin / leverage
            sl_pct = max_loss_pct_margin / leverage
            
            # Определяем силу предсказания
            if confidence >= 0.9:
                strength = "очень_сильное"
            elif confidence >= 0.8:
                strength = "сильное"
            elif confidence >= 0.7:
                strength = "среднее"
            elif confidence >= 0.6:
                strength = "умеренное"
            else:
                strength = "слабое"
            
            # Формируем понятную причину
            # Проверяем на NaN перед преобразованием
            if np.isnan(confidence) or not np.isfinite(confidence):
                confidence = 0.0
            confidence_pct = int(confidence * 100) if np.isfinite(confidence) else 0
            profit_pct = int(target_profit_pct_margin)
            
            # Проверяем количество сигналов за сегодня
            from datetime import datetime, timezone
            current_date = datetime.now(timezone.utc).date()
            date_str = current_date.isoformat()
            
            # Получаем количество сигналов за сегодня (для статистики и оценки работы стратегии)
            signals_today = self.daily_signals_count.get(date_str, 0)
            
            # ПРИМЕЧАНИЕ: Не блокируем сигналы жестким лимитом
            # Цель: естественным образом получать 1-10 качественных сигналов в день через правильные пороги
            # Максимум сигналов используется только как защита от ошибок (например, 100+ сигналов)
            if prediction != 0 and signals_today >= 100:  # Только защита от ошибок (100+ сигналов - явная ошибка)
                return Signal(row.name, Action.HOLD, f"ml_защита_от_ошибок_слишком_много_сигналов_{signals_today}", current_price)
            
            # Проверяем минимальную силу сигнала (только для LONG/SHORT, не для HOLD)
            # Пороги настроены так, чтобы естественным образом получать 1-10 качественных сигналов в день
            if self.is_ensemble:
                # Для ансамблей используем сниженные пороги для получения достаточного количества сигналов
                # Цель: 1-10 сигналов в день естественным образом
                if is_volatile_symbol:
                    min_strength = 0.003  # 0.3% для волатильных символов (снижено: было 0.5%)
                else:
                    min_strength = 0.004  # 0.4% для стабильных символов (снижено: было 0.7%)
            else:
                # Для одиночных моделей используем стандартные пороги
                if is_volatile_symbol:
                    min_strength = self.min_strength_threshold * 0.3
                else:
                    min_strength = self.min_strength_threshold
            
            if prediction != 0 and confidence < min_strength:
                # Сигнал не проходит минимальный порог силы - возвращаем HOLD
                return Signal(row.name, Action.HOLD, f"ml_сила_слишком_слабая_{strength}_{confidence_pct}%_мин_{int(min_strength*100)}%", current_price)
            
            # НОВЫЙ ФИЛЬТР: Проверяем историю сигналов для предотвращения противоречивых сигналов
            if prediction != 0:
                # Проверяем, был ли недавно противоположный сигнал
                opposite_action = Action.SHORT if prediction == 1 else Action.LONG
                # Проверяем последние N сигналов с конца списка
                recent_opposite_count = 0
                for i in range(min(self.min_bars_between_opposite_signals, len(self.signal_history))):
                    idx = len(self.signal_history) - 1 - i
                    if idx >= 0:
                        sig = self.signal_history[idx]
                        if sig[1] == opposite_action:
                            recent_opposite_count += 1
                
                if recent_opposite_count > 0:
                    # Был недавно противоположный сигнал - требуем более высокую уверенность для смены направления
                    # Увеличиваем требуемую уверенность на 30-50% для смены направления
                    stability_multiplier = 1.3 if is_volatile_symbol else 1.5
                    required_confidence = min_strength * stability_multiplier
                    
                    if confidence < required_confidence:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_противоречивый_сигнал_{strength}_{confidence_pct}%_требуется_{int(required_confidence*100)}%_после_{opposite_action.value}", 
                            current_price
                        )
                
                # Проверяем, был ли недавно такой же сигнал (избегаем дублирования)
                same_action = Action.LONG if prediction == 1 else Action.SHORT
                recent_same_count = 0
                for i in range(min(2, len(self.signal_history))):
                    idx = len(self.signal_history) - 1 - i
                    if idx >= 0:
                        sig = self.signal_history[idx]
                        if sig[1] == same_action:
                            recent_same_count += 1
                
                if recent_same_count > 0:
                    # Был недавно такой же сигнал - требуем более высокую уверенность для повторного входа
                    # Увеличиваем требуемую уверенность на 20% для повторного входа
                    repeat_multiplier = 1.2
                    required_confidence = min_strength * repeat_multiplier
                    
                    if confidence < required_confidence:
                        return Signal(
                            row.name, 
                            Action.HOLD, 
                            f"ml_дублирующий_сигнал_{strength}_{confidence_pct}%_требуется_{int(required_confidence*100)}%", 
                            current_price
                        )
            
            # === Подготовка данных для дополнительной фильтрации ===
            
            # Проверяем объем для подтверждения направления движения цены (упрощенная проверка)
            volume = row.get("volume", np.nan)
            # Пытаемся получить vol_sma из разных возможных источников
            # В данных после prepare_with_indicators есть vol_sma
            # В данных после FeatureEngineer есть volume_sma_20
            vol_sma = row.get("vol_sma", np.nan)
            if not np.isfinite(vol_sma):
                # Fallback: используем volume_sma_20 из фичей FeatureEngineer
                vol_sma = row.get("volume_sma_20", np.nan)
            if not np.isfinite(vol_sma):
                # Если vol_sma все еще нет, вычисляем простую SMA за 20 периодов из df
                try:
                    if len(df) >= 20:
                        vol_sma = df["volume"].rolling(window=20).mean().iloc[-1]
                except:
                    pass
            # Упрощенная проверка объема: если vol_sma недоступен, считаем объем OK
            # Если vol_sma доступен, требуем только 50% от среднего (вместо 80%)
            if not np.isfinite(vol_sma):
                volume_ok = True  # Если нет данных о среднем объеме, не блокируем сигнал
            else:
                volume_ok = np.isfinite(volume) and volume > vol_sma * 0.5  # Объем должен быть выше 50% от среднего (упрощено)
            
            # === НОВЫЕ ФИЛЬТРЫ: Улучшения на основе лучших практик ML-трейдинга ===
            
            # 1. Фильтр по тренду (MA): проверяем, что цена находится в правильном направлении относительно важной MA
            # Пытаемся найти доступную MA (приоритет: SMA50 > EMA50 > SMA20 > EMA20)
            sma_20 = row.get("sma_20", np.nan)
            sma_50 = row.get("sma_50", np.nan)
            sma = row.get("sma", np.nan)  # SMA20 из индикаторов
            ema_20 = row.get("ema_20", np.nan)
            ema_50 = row.get("ema_50", np.nan)
            
            # Вычисляем SMA50/EMA50 динамически, если не доступны
            if not np.isfinite(sma_50):
                try:
                    if len(df) >= 50:
                        sma_50 = df["close"].rolling(window=50).mean().iloc[-1]
                except:
                    pass
            
            if not np.isfinite(ema_50):
                try:
                    if len(df) >= 50:
                        ema_50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                except:
                    pass
            
            # Выбираем лучшую доступную MA для фильтра тренда
            trend_ma = None
            ma_type = None  # Тип MA: "sma50", "ema50", "sma20", "ema20"
            if np.isfinite(sma_50):
                trend_ma = sma_50
                ma_type = "sma50"
            elif np.isfinite(ema_50):
                trend_ma = ema_50
                ma_type = "ema50"
            elif np.isfinite(sma) or np.isfinite(sma_20):
                trend_ma = sma if np.isfinite(sma) else sma_20
                ma_type = "sma20"
            elif np.isfinite(ema_20):
                trend_ma = ema_20
                ma_type = "ema20"
            
            trend_filter_ok = True  # По умолчанию пропускаем
            if prediction != 0 and trend_ma is not None and np.isfinite(trend_ma):
                price = row.get("close", current_price)
                # Для LONG: цена должна быть выше MA (или близко к ней, допуск зависит от типа MA)
                # Для SMA используем более строгий допуск (0.5%), для EMA - более мягкий (0.3%)
                ma_tolerance = 0.003 if ma_type in ("ema50", "ema20") else 0.005
                
                if prediction == 1:  # LONG сигнал
                    if price < trend_ma * (1 - ma_tolerance):  # Если цена ниже MA более чем на допуск
                        # Требуем более высокую уверенность для входа против тренда
                        # Для ETHUSDT и SOLUSDT используем более мягкий порог (они более волатильны)
                        if is_volatile_symbol:
                            threshold_multiplier = 1.05 if confidence < 0.5 else 1.08
                        else:
                            threshold_multiplier = 1.12 if confidence < 0.5 else 1.15
                        if confidence < self.confidence_threshold * threshold_multiplier:
                            trend_filter_ok = False
                elif prediction == -1:  # SHORT сигнал
                    if price > trend_ma * (1 + ma_tolerance):  # Если цена выше MA более чем на допуск
                        if is_volatile_symbol:
                            threshold_multiplier = 1.05 if confidence < 0.5 else 1.08
                        else:
                            threshold_multiplier = 1.12 if confidence < 0.5 else 1.15
                        if confidence < self.confidence_threshold * threshold_multiplier:
                            trend_filter_ok = False
            
            # 2. Фильтр по волатильности: не входить при слишком низкой волатильности
            atr = row.get("atr", np.nan)
            atr_pct = row.get("atr_pct", np.nan)
            if not np.isfinite(atr_pct):
                # Вычисляем ATR% из ATR и цены
                if np.isfinite(atr) and current_price > 0:
                    atr_pct = (atr / current_price) * 100
            
            volatility_ok = True  # По умолчанию пропускаем
            if np.isfinite(atr_pct):
                # Адаптивный порог волатильности в зависимости от символа
                # ETHUSDT и SOLUSDT обычно более волатильны, чем BTCUSDT
                # Используем более мягкий порог для них
                if is_volatile_symbol:
                    volatility_threshold = 0.20  # Еще более мягкий порог для волатильных символов
                    threshold_multiplier = 1.05  # Очень мягкий порог уверенности
                else:
                    volatility_threshold = 0.25  # Минимальная волатильность (0.25% вместо 0.3%)
                    threshold_multiplier = 1.08  # Более мягкий порог уверенности (+8% вместо +10%)
                
                # Если волатильность очень низкая, требуем более высокую уверенность
                if atr_pct < volatility_threshold and confidence < self.confidence_threshold * threshold_multiplier:
                    volatility_ok = False
            
            # 3. Фильтр по структуре рынка: проверяем Higher Highs / Higher Lows для LONG, Lower Highs / Lower Lows для SHORT
            structure_ok = True  # По умолчанию пропускаем
            try:
                if len(df) >= 20:
                    # Используем более короткое окно для более быстрой реакции (особенно для ETHUSDT и SOLUSDT)
                    window_size = 8  # Было 10, теперь 8 для более быстрой реакции
                    lookback = 4  # Было 5, теперь 4
                    
                    recent_highs = df["high"].rolling(window=window_size).max().iloc[-lookback:].values
                    recent_lows = df["low"].rolling(window=window_size).min().iloc[-lookback:].values
                    
                    if prediction == 1:  # LONG
                        # Проверяем, что последние максимумы растут (Higher Highs)
                        # Используем более мягкий допуск для ETHUSDT и SOLUSDT
                        tolerance = 0.0010 if is_volatile_symbol else 0.0015
                        if len(recent_highs) >= 2:
                            if recent_highs[-1] < recent_highs[-2] * (1 - tolerance):
                                # Требуем более высокую уверенность, но более мягкий порог для волатильных символов
                                if is_volatile_symbol:
                                    threshold_multiplier = 1.05 if confidence < 0.5 else 1.08
                                else:
                                    threshold_multiplier = 1.08 if confidence < 0.5 else 1.1
                                if confidence < self.confidence_threshold * threshold_multiplier:
                                    structure_ok = False
                    elif prediction == -1:  # SHORT
                        # Проверяем, что последние минимумы падают (Lower Lows)
                        tolerance = 0.0010 if is_volatile_symbol else 0.0015
                        if len(recent_lows) >= 2:
                            if recent_lows[-1] > recent_lows[-2] * (1 + tolerance):
                                if is_volatile_symbol:
                                    threshold_multiplier = 1.05 if confidence < 0.5 else 1.08
                                else:
                                    threshold_multiplier = 1.08 if confidence < 0.5 else 1.1
                                if confidence < self.confidence_threshold * threshold_multiplier:
                                    structure_ok = False
            except:
                pass  # Если не удалось проверить структуру, пропускаем фильтр
            
            # 4. Фильтр по силе тренда (ADX): для ETHUSDT и SOLUSDT используем более мягкий порог
            # Это помогает избежать входов в слабые тренды, но не блокирует полностью
            adx = row.get("adx", np.nan)
            adx_filter_ok = True  # По умолчанию пропускаем
            if np.isfinite(adx) and prediction != 0:
                # Для слабых сигналов (< 0.5 уверенности) требуем минимальный ADX
                # Для ETHUSDT и SOLUSDT используем более мягкий порог
                if is_volatile_symbol:
                    min_adx = 18 if confidence < 0.5 else 15
                    adx_threshold_multiplier = 1.02
                else:
                    min_adx = 20 if confidence < 0.5 else 18
                    adx_threshold_multiplier = 1.05
                if adx < min_adx and confidence < self.confidence_threshold * adx_threshold_multiplier:
                    # Только для очень слабых сигналов блокируем при слабом тренде
                    adx_filter_ok = False
            
            # Проверяем согласованность индикаторов
            rsi = row.get("rsi", np.nan)
            macd = row.get("macd", np.nan)
            macd_signal = row.get("macd_signal", np.nan)
            
            # Проверяем согласованность сигнала с индикаторами
            indicators_agree = True
            if prediction == 1:  # LONG сигнал
                # Для LONG: RSI не должен быть экстремально перекуплен, MACD должен быть выше сигнала (смягчено)
                if np.isfinite(rsi) and rsi > 85:  # Был 80
                    indicators_agree = False
                if np.isfinite(macd) and np.isfinite(macd_signal) and macd < macd_signal * 0.90:  # Был 0.95
                    indicators_agree = False
            elif prediction == -1:  # SHORT сигнал
                # Для SHORT: RSI не должен быть экстремально перепродан, MACD должен быть ниже сигнала (смягчено)
                if np.isfinite(rsi) and rsi < 15:  # Был 20
                    indicators_agree = False
                if np.isfinite(macd) and np.isfinite(macd_signal) and macd > macd_signal * 1.10:  # Был 1.05
                    indicators_agree = False
            
            # Проверяем объемное подтверждение (смягчено для агрессивной модели)
            volume_confirmation = True
            if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0:
                volume_ratio = volume / vol_sma
                # Адаптивный порог объема: для ETHUSDT и SOLUSDT используем более мягкий порог
                # Только для ОЧЕНЬ сильных сигналов (>85%) проверяем объем
                if is_volatile_symbol:
                    min_volume_ratio = 0.5 if confidence < 0.5 else 0.6
                    volume_check_threshold = 0.90  # Проверяем объем только для очень сильных сигналов
                else:
                    min_volume_ratio = 0.6 if confidence < 0.5 else 0.7
                    volume_check_threshold = 0.85
                if confidence > volume_check_threshold and volume_ratio < min_volume_ratio:
                    volume_confirmation = False
            
            # === Дополнительная фильтрация на основе контекста рынка ===
            
            # Динамический порог на основе рыночных условий
            # Для ETHUSDT и SOLUSDT используем очень мягкий порог
            if is_volatile_symbol:
                dynamic_threshold = self.confidence_threshold * 0.75  # Снижаем порог на 25% для волатильных символов
            else:
                dynamic_threshold = self.confidence_threshold
            
            # МЯГКИЕ ФИЛЬТРЫ: Применяем ТОЛЬКО для сигналов ниже основного порога
            # Если сигнал выше dynamic_threshold, мы доверяем модели!
            # Для ансамблей почти полностью отключаем фильтры
            if prediction != 0 and confidence < dynamic_threshold:
                if self.is_ensemble:
                    # Для ансамблей отключаем почти все фильтры - доверяем модели
                    # Только экстремальные случаи (RSI > 95 или < 5)
                    if np.isfinite(rsi):
                        extreme_rsi = (prediction == 1 and rsi > 95) or (prediction == -1 and rsi < 5)
                        if extreme_rsi:
                            rsi_int = int(rsi) if np.isfinite(rsi) else 0
                            return Signal(row.name, Action.HOLD, f"ml_экстремальный_RSI_{rsi_int}_{strength}_{confidence_pct}%", current_price)
                    # Все остальные фильтры отключены для ансамблей
                elif is_volatile_symbol:
                    # Для волатильных символов применяем ТОЛЬКО экстремальные проверки
                    # Только если RSI в экстремальной зоне (>90 или <10) и уверенность очень низкая
                    if np.isfinite(rsi):
                        extreme_rsi = (prediction == 1 and rsi > 90) or (prediction == -1 and rsi < 10)
                        if extreme_rsi and confidence < dynamic_threshold * 0.5:
                            rsi_int = int(rsi) if np.isfinite(rsi) else 0
                            return Signal(row.name, Action.HOLD, f"ml_экстремальный_RSI_{rsi_int}_{strength}_{confidence_pct}%", current_price)
                    # Все остальные фильтры отключены для волатильных символов
                else:
                    # Для BTCUSDT применяем все фильтры
                    if not indicators_agree:
                        return Signal(row.name, Action.HOLD, f"ml_индикаторы_не_согласны_{strength}_{confidence_pct}%", current_price)
                    if not volume_confirmation:
                        return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                    if not trend_filter_ok:
                        return Signal(row.name, Action.HOLD, f"ml_тренд_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                    if not volatility_ok:
                        return Signal(row.name, Action.HOLD, f"ml_низкая_волатильность_{strength}_{confidence_pct}%", current_price)
                    if not structure_ok:
                        return Signal(row.name, Action.HOLD, f"ml_структура_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                    if not adx_filter_ok:
                        adx_int = int(adx) if np.isfinite(adx) else 0
                        return Signal(row.name, Action.HOLD, f"ml_слабый_тренд_ADX_{adx_int}_{strength}_{confidence_pct}%", current_price)

            
            # УБРАНО: Фильтр по силе тренда (ADX) - ML стратегия должна работать на всех стадиях рынка
            # if prediction != 0 and confidence < 0.75 and not adx_strong:
            #     return Signal(row.name, Action.HOLD, f"ml_слабый_тренд_{strength}_{confidence_pct}%", current_price)
            
            # Дополнительная проверка: если цена находится в экстремальных зонах (RSI > 85 или < 15),
            # требуем более высокую уверенность (только для BTCUSDT, не для ансамблей)
            if prediction != 0 and np.isfinite(rsi) and not is_volatile_symbol and not self.is_ensemble:
                if (prediction == 1 and rsi > 85) or (prediction == -1 and rsi < 15):
                    # В экстремальных зонах требуем уверенность на 5% выше (было 10%)
                    extreme_threshold = dynamic_threshold * 1.05
                    if confidence < extreme_threshold:
                        rsi_int = int(rsi) if np.isfinite(rsi) else 0
                        return Signal(row.name, Action.HOLD, f"ml_индикаторы_не_согласны_RSI_{rsi_int}_{strength}_{confidence_pct}%", current_price)
            
            # Генерируем сигналы на основе предсказания
            # Возвращаем только LONG, SHORT или HOLD
            # Уже проверили min_strength_threshold выше, теперь проверяем confidence_threshold
            if prediction == 1:  # LONG
                # Настраиваем пороги для ансамблей для получения 1-10 качественных сигналов в день
                if self.is_ensemble:
                    # Для ансамблей используем сниженные пороги (15-20% от стандартного) для достаточного количества сигналов
                    # Цель: естественным образом получать 1-10 сигналов в день
                    threshold_mult = 0.15 if is_volatile_symbol else 0.20  # 15-20% от стандартного (снижено: было 25-35%)
                    # Для ансамблей также снижаем dynamic_threshold
                    dynamic_threshold = self.confidence_threshold * 0.20  # 20% от стандартного (снижено: было 30%)
                else:
                    # Для одиночных моделей используем стандартные пороги
                    threshold_mult = 0.70 if is_volatile_symbol else 0.85
                
                effective_threshold = max(dynamic_threshold * threshold_mult, min_strength)
                # Для ансамблей effective_threshold теперь выше, что фильтрует слабые сигналы
                if confidence < effective_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_не_проходит_порог_уверенности_{strength}_{confidence_pct}%_мин_{int(effective_threshold*100)}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                # Повышаем пороги для ансамблей для предотвращения частой смены направления
                if self.stability_filter and has_position == Bias.SHORT:
                    if self.is_ensemble:
                        stability_threshold = max(self.confidence_threshold * 0.40, 0.25)  # Повышено с 0.1% до 25-40%
                    elif is_volatile_symbol:
                        stability_threshold = max(self.confidence_threshold * 0.70, 0.35)  # Очень мягкий порог
                    else:
                        stability_threshold = max(self.confidence_threshold * 0.85, 0.45)
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%", current_price)
                
                # Проверяем объем (смягчено: если уверенность высокая, объем менее важен)
                # Для волатильных символов проверка объема полностью отключена
                if not is_volatile_symbol:
                    volume_threshold_mult = 1.2
                    if not volume_ok and confidence < dynamic_threshold * volume_threshold_mult:
                        return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                # Сигнал LONG
                reason = f"ml_LONG_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%"
                
                # Обновляем историю сигналов
                signal_action = Action.LONG
                self.signal_history.append((row.name, signal_action, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                # Обновляем счетчик сигналов за день
                self.daily_signals_count[date_str] = signals_today + 1
                # Очищаем старые даты (старше 7 дней) для экономии памяти
                from datetime import timedelta
                cutoff_date = (current_date - timedelta(days=7)).isoformat()
                self.daily_signals_count = {k: v for k, v in self.daily_signals_count.items() if k >= cutoff_date}
                
                # Собираем информацию о показателях для ML
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "LONG",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "tp_pct": round(tp_pct, 2),
                    "sl_pct": round(sl_pct, 2),
                    "target_profit_margin_pct": profit_pct,
                    "leverage": leverage,
                    "volume": round(volume, 0) if np.isfinite(volume) else None,
                    "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                    "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                    "volume_ok": volume_ok,
                    "has_position": has_position.value if has_position else None,
                    "indicators": f"ML Confidence={confidence_pct}% ({strength}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else f"ML Confidence={confidence_pct}% ({strength})"
                }
                return Signal(row.name, Action.LONG, reason, current_price, indicators_info=indicators_info)
            
            elif prediction == -1:  # SHORT
                # Настраиваем пороги для ансамблей для получения 1-10 качественных сигналов в день
                if self.is_ensemble:
                    # Для ансамблей используем сниженные пороги (15-20% от стандартного) для достаточного количества сигналов
                    # Цель: естественным образом получать 1-10 сигналов в день
                    threshold_mult = 0.15 if is_volatile_symbol else 0.20  # 15-20% от стандартного (снижено: было 25-35%)
                    # Для ансамблей также снижаем dynamic_threshold
                    dynamic_threshold = self.confidence_threshold * 0.20  # 20% от стандартного (снижено: было 30%)
                else:
                    # Для одиночных моделей используем стандартные пороги
                    threshold_mult = 0.70 if is_volatile_symbol else 0.85
                
                effective_threshold = max(dynamic_threshold * threshold_mult, min_strength)
                # Для ансамблей effective_threshold теперь выше, что фильтрует слабые сигналы
                if confidence < effective_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_не_проходит_порог_уверенности_{strength}_{confidence_pct}%_мин_{int(effective_threshold*100)}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                # Повышаем пороги для ансамблей для предотвращения частой смены направления
                if self.stability_filter and has_position == Bias.LONG:
                    if self.is_ensemble:
                        stability_threshold = max(self.confidence_threshold * 0.40, 0.25)  # Повышено с 0.1% до 25-40%
                    elif is_volatile_symbol:
                        stability_threshold = max(self.confidence_threshold * 0.70, 0.35)  # Очень мягкий порог
                    else:
                        stability_threshold = max(self.confidence_threshold * 0.85, 0.45)
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%", current_price)
                
                # Проверяем объем (смягчено: если уверенность высокая, объем менее важен)
                # Для ансамблей и волатильных символов проверка объема полностью отключена
                if not is_volatile_symbol and not self.is_ensemble:
                    volume_threshold_mult = 1.2
                    if not volume_ok and confidence < dynamic_threshold * volume_threshold_mult:
                        return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                # Сигнал SHORT
                reason = f"ml_SHORT_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%"
                
                # Обновляем историю сигналов
                signal_action = Action.SHORT
                self.signal_history.append((row.name, signal_action, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                # Обновляем счетчик сигналов за день
                self.daily_signals_count[date_str] = signals_today + 1
                # Очищаем старые даты (старше 7 дней) для экономии памяти
                from datetime import timedelta
                cutoff_date = (current_date - timedelta(days=7)).isoformat()
                self.daily_signals_count = {k: v for k, v in self.daily_signals_count.items() if k >= cutoff_date}
                
                # Собираем информацию о показателях для ML
                indicators_info = {
                    "strategy": "ML",
                    "prediction": "SHORT",
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "strength": strength,
                    "tp_pct": round(tp_pct, 2),
                    "sl_pct": round(sl_pct, 2),
                    "target_profit_margin_pct": profit_pct,
                    "leverage": leverage,
                    "volume": round(volume, 0) if np.isfinite(volume) else None,
                    "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                    "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                    "volume_ok": volume_ok,
                    "has_position": has_position.value if has_position else None,
                    "indicators": f"ML Confidence={confidence_pct}% ({strength}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else f"ML Confidence={confidence_pct}% ({strength})"
                }
                return Signal(row.name, Action.SHORT, reason, current_price, indicators_info=indicators_info)
            
            else:  # prediction == 0 (HOLD)
                # Модель предсказывает нейтральное движение
                # Обновляем историю сигналов (HOLD тоже записываем для отслеживания)
                self.signal_history.append((row.name, Action.HOLD, confidence))
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                reason = f"ml_нейтрально_сила_{strength}_{confidence_pct}%_ожидание"
                return Signal(row.name, Action.HOLD, reason, current_price)
        
        except Exception as e:
            print(f"[ml_strategy] Error generating signal: {e}")
            return Signal(row.name, Action.HOLD, f"ml_ошибка_{str(e)[:20]}", current_price)


def build_ml_signals(
    df: pd.DataFrame,
    model_path: str,
    confidence_threshold: float = 0.5,
    min_signal_strength: str = "слабое",
    stability_filter: bool = True,
    leverage: int = 10,
    target_profit_pct_margin: float = 25.0,
    max_loss_pct_margin: float = 10.0,
    min_signals_per_day: int = 1,
    max_signals_per_day: int = 10,
) -> list[Signal]:
    """
    Строит сигналы на основе ML-модели для всего DataFrame.
    
    Args:
        df: DataFrame с данными (должен содержать OHLCV и индикаторы)
        model_path: Путь к обученной модели
        confidence_threshold: Минимальная уверенность для открытия позиции
        min_signal_strength: Минимальная сила сигнала ("слабое", "умеренное", "среднее", "сильное", "очень_сильное")
        stability_filter: Фильтр стабильности - требовать более высокую уверенность для смены направления
    
    Returns:
        Список Signal объектов
    """
    strategy = MLStrategy(model_path, confidence_threshold, min_signal_strength, stability_filter, min_signals_per_day=min_signals_per_day, max_signals_per_day=max_signals_per_day)
    signals: list[Signal] = []
    position_bias: Optional[Bias] = None
    
    # Убеждаемся, что DataFrame имеет правильную структуру
    df_work = df.copy()
    
    # Если timestamp в колонках, используем его как индекс
    if "timestamp" in df_work.columns:
        df_work = df_work.set_index("timestamp")
    elif not isinstance(df_work.index, pd.DatetimeIndex):
        # Пытаемся преобразовать индекс в DatetimeIndex
        try:
            df_work.index = pd.to_datetime(df_work.index)
        except:
            pass
    
    # Убеждаемся, что есть необходимые колонки OHLCV
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df_work.columns for col in required_cols):
        print(f"[ml_strategy] Warning: Missing required columns. Available: {df_work.columns.tolist()}")
        # Возвращаем пустые сигналы
        return [Signal(df_work.index[i] if len(df_work) > 0 else pd.Timestamp.now(), Action.HOLD, "ml_missing_data", 0.0) 
                for i in range(len(df_work))]
    
    # ОПТИМИЗАЦИЯ: Вычисляем фичи один раз для всего DataFrame вместо пересчета для каждого бара
    # Это значительно ускоряет работу, так как создание индикаторов - самая затратная операция
    # Подготовка фичей (без verbose логирования)
    try:
        # Определяем, включен ли MTF-режим для ML (по окружению, синхронно с train_* скриптами)
        import os
        # ВАЖНО: по умолчанию MTF выключен (иначе 15m-модели получают чужие фичи)
        ml_mtf_enabled_env = os.getenv("ML_MTF_ENABLED", "0")
        ml_mtf_enabled = ml_mtf_enabled_env not in ("0", "false", "False", "no")

        # Базовые технические индикаторы на 15m
        df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)

        # Если включен MTF-режим, добавляем фичи 1h/4h по той же схеме, что и при обучении
        if ml_mtf_enabled:
            try:
                # Строим агрегированные OHLCV для 1h и 4h из 15m данных
                ohlcv_agg = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
                df_1h = df_work.resample("60min").agg(ohlcv_agg).dropna()
                df_4h = df_work.resample("240min").agg(ohlcv_agg).dropna()

                higher_timeframes = {}
                if df_1h is not None and not df_1h.empty:
                    higher_timeframes["60"] = df_1h
                if df_4h is not None and not df_4h.empty:
                    higher_timeframes["240"] = df_4h

                if higher_timeframes:
                    df_with_features = strategy.feature_engineer.add_mtf_features(
                        df_with_features,
                        higher_timeframes,
                    )
                    print(f"[ml_strategy] MTF features enabled for ML signals (1h/4h). Columns: {len(df_with_features.columns)}")
                else:
                    print("[ml_strategy] MTF enabled but failed to build 1h/4h data – using 15m-only features")
            except Exception as mtf_err:
                print(f"[ml_strategy] Warning: failed to add MTF features in build_ml_signals: {mtf_err}")
    except Exception as e:
        print(f"[ml_strategy] Error preparing features: {e}")
        # Возвращаем пустые сигналы при ошибке
        return [Signal(df_work.index[i] if len(df_work) > 0 else pd.Timestamp.now(), Action.HOLD, f"ml_error_{str(e)[:20]}", 0.0) 
                for i in range(len(df_work))]
    
    for idx, row in df_with_features.iterrows():
        try:
            # Получаем данные до текущего момента (уже с вычисленными фичами)
            df_until_now = df_with_features.loc[:idx]
            
            # Нужно минимум 200 баров для расчета всех индикаторов (SMA200, и т.д.)
            if len(df_until_now) < 200:
                signals.append(Signal(idx, Action.HOLD, "ml_insufficient_data", row["close"]))
                continue
            
            # Используем уже вычисленные фичи - не пересчитываем их
            signal = strategy.generate_signal(
                row=row,
                df=df_until_now,  # Уже содержит все фичи
                has_position=position_bias,
                current_price=row["close"],
                leverage=leverage,
                target_profit_pct_margin=target_profit_pct_margin,
                max_loss_pct_margin=max_loss_pct_margin,
            )
            signals.append(signal)
            # ВАЖНО: build_ml_signals не должен эмулировать позицию по сигналам.
            # Реальная позиция известна на уровне live/backtest-движка и должна передаваться в generate_signal,
            # иначе stability_filter начинает "залипать" в одном направлении (например, только SHORT).
        except Exception as e:
            print(f"[ml_strategy] Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            signals.append(Signal(idx, Action.HOLD, f"ml_error_{str(e)[:20]}", row.get("close", 0.0)))
    
    return signals

