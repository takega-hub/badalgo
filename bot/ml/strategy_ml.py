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


class MLStrategy:
    """
    ML-стратегия, использующая обученную модель для предсказания движения цены.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7, min_signal_strength: str = "умеренное", stability_filter: bool = True):
        """
        Инициализирует ML-стратегию.
        
        Args:
            model_path: Путь к сохраненной модели (.pkl файл)
            confidence_threshold: Минимальная уверенность модели для открытия позиции (0-1)
            min_signal_strength: Минимальная сила сигнала ("слабое", "умеренное", "среднее", "сильное", "очень_сильное")
            stability_filter: Фильтр стабильности - требовать более высокую уверенность для смены направления
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_signal_strength = min_signal_strength
        self.stability_filter = stability_filter
        
        # Определяем минимальный порог уверенности на основе силы сигнала
        strength_thresholds = {
            "слабое": 0.0,
            "умеренное": 0.6,
            "среднее": 0.7,
            "сильное": 0.8,
            "очень_сильное": 0.9
        }
        self.min_strength_threshold = strength_thresholds.get(min_signal_strength, 0.6)
        
        # Загружаем модель
        self.model_data = self._load_model()
        self.model = self.model_data["model"]
        self.scaler = self.model_data["scaler"]
        self.feature_names = self.model_data["feature_names"]
        
        # Инициализируем feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Извлекаем символ из пути к модели для логирования
        model_filename = Path(model_path).name
        symbol_from_model = "UNKNOWN"
        if "_" in model_filename:
            parts = model_filename.split("_")
            if len(parts) >= 2:
                symbol_from_model = parts[1]  # Например, rf_ETHUSDT_15.pkl -> ETHUSDT
        
        print(f"[ml_strategy] ML model loaded from {model_path}")
        print(f"[ml_strategy] Model symbol: {symbol_from_model}")
        print(f"[ml_strategy] Confidence threshold: {confidence_threshold}")
        print(f"[ml_strategy] Min signal strength: {min_signal_strength} (threshold: {self.min_strength_threshold:.0%})")
        print(f"[ml_strategy] Stability filter: {stability_filter}")
        print(f"[ml_strategy] Features: {len(self.feature_names)}")
    
    def _load_model(self) -> Dict[str, Any]:
        """Загружает модель из файла."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        return model_data
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Подготавливает фичи из DataFrame для предсказания модели.
        
        Args:
            df: DataFrame с OHLCV данными и индикаторами
        
        Returns:
            Массив фичей для модели
        """
        # Всегда создаем фичи заново, чтобы убедиться, что все необходимые фичи присутствуют
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
        df_with_features = self.feature_engineer.create_technical_indicators(df_work)
        
        # Удаляем строки с NaN (после создания индикаторов первые строки могут быть NaN)
        df_with_features = df_with_features.dropna()
        
        # Проверяем, что есть данные после dropna
        if len(df_with_features) == 0:
            raise ValueError("No data available after creating features (all rows contain NaN)")
        
        # Проверяем наличие всех необходимых фичей
        missing_features = [f for f in self.feature_names if f not in df_with_features.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features[:10]}... (total {len(missing_features)})")
        
        # Выбираем только нужные фичи в правильном порядке
        X = df_with_features[self.feature_names].values
        
        # Проверяем, что есть данные для нормализации
        if len(X) == 0:
            raise ValueError("No data available after feature selection")
        
        # Нормализуем
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Делает предсказание на основе последнего бара.
        
        Args:
            df: DataFrame с данными (OHLCV, фичи будут созданы автоматически)
        
        Returns:
            (prediction, confidence) где:
            - prediction: 1 (LONG), -1 (SHORT), 0 (HOLD)
            - confidence: уверенность модели (0-1)
        """
        # Берем последний бар
        if len(df) == 0:
            return 0, 0.0
        
        try:
            # Подготавливаем фичи (создаст все необходимые индикаторы)
            X = self.prepare_features(df)
            
            # Берем последний образец
            X_last = X[-1:].reshape(1, -1)
        except Exception as e:
            print(f"[ml_strategy] Error preparing features: {e}")
            return 0, 0.0
        
        # Предсказание
        if hasattr(self.model, "predict_proba"):
            # Для классификаторов с вероятностями
            proba = self.model.predict_proba(X_last)[0]
            
            # Если модель была обучена на классах 0,1,2 (XGBoost), преобразуем обратно
            if len(proba) == 3:
                # proba[0] = SHORT (-1), proba[1] = HOLD (0), proba[2] = LONG (1)
                prediction_idx = np.argmax(proba)
                prediction = prediction_idx - 1  # 0->-1, 1->0, 2->1
                confidence = proba[prediction_idx]
                
                # УЛУЧШЕНИЕ: Если модель предсказывает HOLD, но вероятность LONG или SHORT достаточно высока,
                # используем эту вероятность для генерации сигнала
                long_prob = proba[2] if len(proba) > 2 else 0.0
                short_prob = proba[0] if len(proba) > 0 else 0.0
                hold_prob = proba[1] if len(proba) > 1 else 0.0
                
                # Если HOLD имеет максимальную вероятность, но LONG или SHORT имеют достаточно высокую вероятность,
                # используем их для генерации сигнала (если они превышают строгий порог min_strength_threshold)
                if prediction == 0:  # HOLD
                    # Используем строгий порог min_strength_threshold для переопределения HOLD
                    # Только если вероятность LONG или SHORT >= min_strength_threshold, переопределяем HOLD
                    if long_prob >= self.min_strength_threshold and long_prob > short_prob:
                        prediction = 1  # LONG
                        confidence = long_prob
                    elif short_prob >= self.min_strength_threshold and short_prob > long_prob:
                        prediction = -1  # SHORT
                        confidence = short_prob
                    # Иначе остаемся на HOLD
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
            # Делаем предсказание
            prediction, confidence = self.predict(df)
            
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
            
            # Генерируем сигналы на основе предсказания
            # Возвращаем только LONG, SHORT или HOLD
            # Уже проверили min_strength_threshold выше, теперь проверяем confidence_threshold
            if prediction == 1:  # LONG
                # Дополнительная проверка: уверенность должна быть >= min_strength_threshold
                # Это гарантирует, что слабые сигналы не проходят
                if confidence < self.min_strength_threshold:
                    return Signal(row.name, Action.HOLD, f"ml_сила_слишком_слабая_{strength}_{confidence_pct}%_мин_{int(self.min_strength_threshold*100)}%", current_price)
                
                # Используем confidence_threshold для дополнительной проверки уверенности модели
                if confidence < self.confidence_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_ожидание_сила_{strength}_{confidence_pct}%_порог_{int(self.confidence_threshold*100)}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                # Но используем более мягкий порог (60% вместо 70%)
                if self.stability_filter and has_position == Bias.SHORT:
                    stability_threshold = max(self.confidence_threshold * 0.9, 0.5)  # 90% от основного порога или минимум 50%
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%_для_смены_SHORT_на_LONG_текущая_{confidence_pct}%", current_price)
                
                # Проверяем, что объем подтверждает движение вверх
                if not volume_ok:
                    return Signal(row.name, Action.HOLD, f"ml_LONG_объем_не_подтверждает_{confidence_pct}%", current_price)
                # Сигнал LONG
                reason = f"ml_LONG_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%_прибыль_{profit_pct}%"
                
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
                # Дополнительная проверка: уверенность должна быть >= min_strength_threshold
                # Это гарантирует, что слабые сигналы не проходят
                if confidence < self.min_strength_threshold:
                    return Signal(row.name, Action.HOLD, f"ml_сила_слишком_слабая_{strength}_{confidence_pct}%_мин_{int(self.min_strength_threshold*100)}%", current_price)
                
                # Используем confidence_threshold для дополнительной проверки уверенности модели
                if confidence < self.confidence_threshold:
                    # Модель не уверена - HOLD
                    return Signal(row.name, Action.HOLD, f"ml_ожидание_сила_{strength}_{confidence_pct}%_порог_{int(self.confidence_threshold*100)}%", current_price)
                
                # Фильтр стабильности: если есть позиция в противоположном направлении, требуем более высокую уверенность
                # Но используем более мягкий порог (60% вместо 70%)
                if self.stability_filter and has_position == Bias.LONG:
                    stability_threshold = max(self.confidence_threshold * 0.9, 0.5)  # 90% от основного порога или минимум 50%
                    if confidence < stability_threshold:
                        return Signal(row.name, Action.HOLD, f"ml_стабильность_требует_{int(stability_threshold*100)}%_для_смены_LONG_на_SHORT_текущая_{confidence_pct}%", current_price)
                
                # Проверяем, что объем подтверждает движение вниз
                if not volume_ok:
                    return Signal(row.name, Action.HOLD, f"ml_SHORT_объем_не_подтверждает_{confidence_pct}%", current_price)
                # Сигнал SHORT
                reason = f"ml_SHORT_сила_{strength}_{confidence_pct}%_TP_{tp_pct:.2f}%_SL_{sl_pct:.2f}%_прибыль_{profit_pct}%"
                
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
    confidence_threshold: float = 0.7,
    min_signal_strength: str = "умеренное",
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
    
    for idx, row in df_work.iterrows():
        try:
            # Получаем данные до текущего момента (для корректного предсказания)
            df_until_now = df_work.loc[:idx]
            
            # Нужно минимум 200 баров для расчета всех индикаторов (SMA200, и т.д.)
            if len(df_until_now) < 200:
                signals.append(Signal(idx, Action.HOLD, "ml_insufficient_data", row["close"]))
                continue
            
            signal = strategy.generate_signal(
                row=row,
                df=df_until_now,
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

