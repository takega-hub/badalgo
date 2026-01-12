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

import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from bot.strategy import Action, Bias, Signal
from bot.ml.feature_engineering import FeatureEngineer
from bot.config import StrategyParams
# Импортируем классы ансамбля для корректной десериализации pickle
from bot.ml.model_trainer import PreTrainedVotingEnsemble, WeightedEnsemble


class MLStrategy:
    """
    ML-стратегия, использующая обученную модель для предсказания движения цены.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, min_signal_strength: str = "слабое", stability_filter: bool = True, use_dynamic_threshold: bool = True):
        """
        Инициализирует ML-стратегию.
        
        Args:
            model_path: Путь к сохраненной модели (.pkl файл)
            confidence_threshold: Минимальная уверенность модели для открытия позиции (0-1)
            min_signal_strength: Минимальная сила сигнала ("слабое", "умеренное", "среднее", "сильное", "очень_сильное")
            stability_filter: Фильтр стабильности - требовать более высокую уверенность для смены направления
            use_dynamic_threshold: Использовать динамические пороги на основе рыночных условий
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
        
        # Загружаем модель
        self.model_data = self._load_model()
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_names = self.model_data["feature_names"]
        self.is_ensemble = self.model_data.get("metadata", {}).get("model_type", "").startswith("ensemble")
        
        # Инициализируем feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Извлекаем символ из пути к модели для логирования
        model_filename = Path(model_path).name
        symbol_from_model = "UNKNOWN"
        if "_" in model_filename:
            parts = model_filename.split("_")
            if len(parts) >= 2:
                symbol_from_model = parts[1]  # Например, rf_ETHUSDT_15.pkl -> ETHUSDT
        
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
            if not hasattr(self, '_missing_features_warned'):
                print(f"[ml_strategy] ⚠️ WARNING: Missing {len(missing_features)} features: {missing_features[:10]}...")
                print(f"[ml_strategy]   Expected {len(self.feature_names)} features, got {len(df_with_features.columns)}")
                self._missing_features_warned = True
            
            # Заполняем отсутствующие фичи нулями
            for missing_feat in missing_features:
                df_with_features[missing_feat] = 0.0
        
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
            X = self.prepare_features(df, skip_feature_creation=skip_feature_creation)
            
            # Берем последний образец
            X_last = X[-1:].reshape(1, -1)
        except Exception as e:
            print(f"[ml_strategy] Error preparing features: {e}")
            return 0, 0.0
        
        # Предсказание
        if hasattr(self.model, "predict_proba"):
            # Для классификаторов с вероятностями (включая ансамбль)
            proba = self.model.predict_proba(X_last)[0]
            
            # Для ансамбля proba уже в правильном формате [-1, 0, 1]
            # Для XGBoost нужно преобразовать из [0, 1, 2]
            if self.is_ensemble:
                # Ансамбль уже возвращает вероятности в формате [-1, 0, 1]
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1  # 0->-1, 1->0, 2->1
                confidence = proba[prediction_idx]
                
                # УЛУЧШЕНИЕ: Если модель предсказывает HOLD, но вероятность LONG или SHORT достаточно высока,
                # используем эту вероятность для генерации сигнала
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
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
            elif len(proba) == 3:
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1  # 0->-1, 1->0, 2->1
                confidence = proba[prediction_idx]
                
                # УЛУЧШЕНИЕ: Если модель предсказывает HOLD, но вероятность LONG или SHORT достаточно высока,
                # используем эту вероятность для генерации сигнала
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
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
        else:
            # Для моделей без predict_proba
            prediction_raw = self.model.predict(X_last)[0]
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
            confidence_pct = int(confidence * 100)
            profit_pct = int(target_profit_pct_margin)
            
            # Проверяем минимальную силу сигнала (только для LONG/SHORT, не для HOLD)
            # Используем строгий порог min_strength_threshold для фильтрации слабых сигналов
            if prediction != 0 and confidence < self.min_strength_threshold:
                # Сигнал не проходит минимальный порог силы - возвращаем HOLD
                return Signal(row.name, Action.HOLD, f"ml_сила_слишком_слабая_{strength}_{confidence_pct}%_мин_{int(self.min_strength_threshold*100)}%", current_price)
            
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
            
            # УБРАНО: Проверка силы тренда (ADX) - ML стратегия должна работать на всех стадиях рынка
            # adx = row.get("adx", np.nan)
            # adx_strong = np.isfinite(adx) and adx > 25
            
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
                # Только для ОЧЕНЬ сильных сигналов (>85%) проверяем объем (был 80%)
                if confidence > 0.85 and volume_ratio < 0.7: # Был 0.8
                    volume_confirmation = False
            
            # === Дополнительная фильтрация на основе контекста рынка ===
            
            # Динамический порог на основе рыночных условий
            dynamic_threshold = self.confidence_threshold
            
            # МЯГКИЕ ФИЛЬТРЫ: Применяем ТОЛЬКО для сигналов ниже основного порога
            # Если сигнал выше dynamic_threshold, мы доверяем модели!
            if prediction != 0 and confidence < dynamic_threshold:
                if not indicators_agree:
                    return Signal(row.name, Action.HOLD, f"ml_индикаторы_не_согласны_{strength}_{confidence_pct}%", current_price)
                if not volume_confirmation:
                    return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)

            
            # УБРАНО: Фильтр по силе тренда (ADX) - ML стратегия должна работать на всех стадиях рынка
            # if prediction != 0 and confidence < 0.75 and not adx_strong:
            #     return Signal(row.name, Action.HOLD, f"ml_слабый_тренд_{strength}_{confidence_pct}%", current_price)
            
            # Дополнительная проверка: если цена находится в экстремальных зонах (RSI > 85 или < 15),
            # требуем более высокую уверенность (смягчено с 80/20)
            if prediction != 0 and np.isfinite(rsi):
                if (prediction == 1 and rsi > 85) or (prediction == -1 and rsi < 15):
                    # В экстремальных зонах требуем уверенность на 5% выше (было 10%)
                    extreme_threshold = dynamic_threshold * 1.05
                    if confidence < extreme_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_индикаторы_не_согласны_RSI_{int(rsi)}_{strength}_{confidence_pct}%", current_price)
            
            # Генерируем сигналы на основе предсказания
            # Возвращаем только LONG, SHORT или HOLD
            # Уже проверили min_strength_threshold выше, теперь проверяем confidence_threshold
            if prediction == 1:  # LONG
                # Смягченная проверка: если уверенность близка к порогу (в пределах 15%), все равно пропускаем
                effective_threshold = max(dynamic_threshold * 0.85, self.min_strength_threshold)  # Минимум 85% от порога
                if confidence < effective_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_не_проходит_порог_уверенности_{strength}_{confidence_pct}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                if self.stability_filter and has_position == Bias.SHORT:
                    stability_threshold = max(self.confidence_threshold * 0.85, 0.45)
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%", current_price)
                
                # Проверяем объем (смягчено: если уверенность высокая, объем менее важен)
                if not volume_ok and confidence < dynamic_threshold * 1.2:
                    return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                # Сигнал LONG
                reason = f"ml_LONG_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%"
                
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
                # Смягченная проверка: если уверенность близка к порогу (в пределах 15%), все равно пропускаем
                effective_threshold = max(dynamic_threshold * 0.85, self.min_strength_threshold)  # Минимум 85% от порога
                if confidence < effective_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_не_проходит_порог_уверенности_{strength}_{confidence_pct}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                if self.stability_filter and has_position == Bias.LONG:
                    stability_threshold = max(self.confidence_threshold * 0.85, 0.45)
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%", current_price)
                
                # Проверяем объем (смягчено: если уверенность высокая, объем менее важен)
                if not volume_ok and confidence < dynamic_threshold * 1.2:
                    return Signal(row.name, Action.HOLD, f"ml_объем_не_подтверждает_{strength}_{confidence_pct}%", current_price)
                # Сигнал SHORT
                reason = f"ml_SHORT_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%"
                
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
    strategy = MLStrategy(model_path, confidence_threshold, min_signal_strength, stability_filter)
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
        df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)
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
            
            # Обновляем позицию
            # Обновляем состояние позиции на основе сигнала
            if signal.action == Action.LONG:
                if position_bias is None or position_bias == Bias.SHORT:
                    position_bias = Bias.LONG
                # Если уже LONG - остаемся LONG
            elif signal.action == Action.SHORT:
                if position_bias is None or position_bias == Bias.LONG:
                    position_bias = Bias.SHORT
                # Если уже SHORT - остаемся SHORT
            # HOLD - позиция остается как есть
        except Exception as e:
            print(f"[ml_strategy] Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            signals.append(Signal(idx, Action.HOLD, f"ml_error_{str(e)[:20]}", row.get("close", 0.0)))
    
    return signals

