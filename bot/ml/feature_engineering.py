"""
Модуль для создания фичей (признаков) из исторических данных для ML-моделей.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


class FeatureEngineer:
    """
    Создает технические индикаторы и другие фичи из OHLCV данных.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает технические индикаторы из OHLCV данных.
        
        Args:
            df: DataFrame с колонками timestamp, open, high, low, close, volume
        
        Returns:
            DataFrame с добавленными колонками индикаторов
        """
        df = df.copy()
        
        # Проверяем наличие необходимых колонок
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Устанавливаем timestamp как индекс если он есть
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        # === Трендовые индикаторы ===
        
        # Moving Averages
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)
        df["ema_12"] = ta.ema(df["close"], length=12)
        df["ema_26"] = ta.ema(df["close"], length=26)
        
        # ADX (Average Directional Index) - сила тренда
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_result[f"ADX_14"]
        df["plus_di"] = adx_result[f"DMP_14"]
        df["minus_di"] = adx_result[f"DMN_14"]
        
        # MACD
        macd_result = ta.macd(df["close"])
        df["macd"] = macd_result["MACD_12_26_9"]
        df["macd_signal"] = macd_result["MACDs_12_26_9"]
        df["macd_hist"] = macd_result["MACDh_12_26_9"]
        
        # === Осцилляторы ===
        
        # RSI (Relative Strength Index)
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_21"] = ta.rsi(df["close"], length=21)
        
        # Stochastic Oscillator
        stoch_result = ta.stoch(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch_result["STOCHk_14_3_3"]
        df["stoch_d"] = stoch_result["STOCHd_14_3_3"]
        
        # CCI (Commodity Channel Index)
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        
        # === Волатильность ===
        
        # Bollinger Bands
        bb_result = ta.bbands(df["close"], length=20, std=2.0)
        df["bb_upper"] = bb_result[f"BBU_20_2.0_2.0"]
        df["bb_middle"] = bb_result[f"BBM_20_2.0_2.0"]
        df["bb_lower"] = bb_result[f"BBL_20_2.0_2.0"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR (Average True Range)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100  # ATR в процентах
        
        # === Объемные индикаторы ===
        
        # Volume SMA
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        
        # OBV (On-Balance Volume)
        df["obv"] = ta.obv(df["close"], df["volume"])
        
        # === Ценовые паттерны ===
        
        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()
        
        # High-Low range
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["oc_range"] = (df["open"] - df["close"]) / df["close"]
        
        # === Лаговые фичи (цены за предыдущие периоды) ===
        
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            df[f"price_change_lag_{lag}"] = df["price_change"].shift(lag)
        
        # === Скользящие статистики ===
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"close_std_{window}"] = df["close"].rolling(window=window).std()
            df[f"close_mean_{window}"] = df["close"].rolling(window=window).mean()
            df[f"volume_mean_{window}"] = df["volume"].rolling(window=window).mean()
        
        # === Временные фичи ===
        
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["day_of_month"] = df.index.day
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        
        # === Дополнительные фичи ===
        
        # Momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        
        # Rate of Change
        df["roc_5"] = ta.roc(df["close"], length=5)
        df["roc_10"] = ta.roc(df["close"], length=10)
        
        # Удаляем строки с NaN (после создания индикаторов)
        df = df.dropna()
        
        # Сохраняем список фичей (исключаем исходные колонки)
        original_cols = ["open", "high", "low", "close", "volume", "turnover"]
        self.feature_names = [col for col in df.columns if col not in original_cols]
        
        return df
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_periods: int = 4,  # 4 периода вперед (например, 4 * 15m = 1 час)
        threshold_pct: float = 0.5,  # 0.5% изменение для сигнала (более реалистично для криптовалют)
        use_atr_threshold: bool = True,  # Использовать динамический порог на основе ATR
    ) -> pd.DataFrame:
        """
        Создает целевую переменную для обучения модели.
        
        Args:
            df: DataFrame с данными
            forward_periods: Сколько периодов вперед смотреть
            threshold_pct: Порог изменения цены для классификации (базовый или минимальный)
            use_atr_threshold: Если True, использует динамический порог на основе ATR для адаптации к волатильности
        
        Returns:
            DataFrame с добавленной колонкой 'target'
            target: 1 = LONG (цена вырастет), -1 = SHORT (цена упадет), 0 = HOLD (нейтрально)
        """
        df = df.copy()
        
        # Вычисляем будущую цену
        future_price = df["close"].shift(-forward_periods)
        current_price = df["close"]
        
        # Вычисляем процентное изменение
        price_change_pct = ((future_price - current_price) / current_price) * 100
        
        # Классифицируем:
        # 1 = LONG (цена вырастет больше чем threshold_pct %)
        # -1 = SHORT (цена упадет больше чем threshold_pct %)
        # 0 = HOLD (изменение в пределах threshold_pct %)
        # price_change_pct уже в процентах, поэтому сравниваем напрямую
        
        # Классифицируем: используем более низкий порог для получения большего количества сигналов
        df["target"] = 0
        
        if use_atr_threshold and "atr_pct" in df.columns and not df["atr_pct"].isna().all():
            # Используем динамический порог на основе ATR: минимум из базового порога и 0.6 * ATR
            # Это позволит адаптироваться к волатильности - при низкой волатильности используем базовый порог,
            # при высокой - уменьшаем порог до 0.6 * ATR для большего количества сигналов
            dynamic_threshold = df["atr_pct"] * 0.6  # 60% от ATR
            # Используем минимум из базового порога и динамического (более агрессивный порог = больше сигналов)
            # Заполняем NaN в dynamic_threshold базовым порогом
            dynamic_threshold = dynamic_threshold.fillna(threshold_pct)
            # Берем минимум между базовым порогом и динамическим
            effective_threshold = np.minimum(threshold_pct, dynamic_threshold)
            
            # Классифицируем с динамическим порогом для каждой строки
            mask_long = price_change_pct > effective_threshold
            mask_short = price_change_pct < -effective_threshold
            df.loc[mask_long, "target"] = 1  # LONG
            df.loc[mask_short, "target"] = -1  # SHORT
        else:
            # Статический порог (оригинальная логика)
            df.loc[price_change_pct > threshold_pct, "target"] = 1  # LONG
            df.loc[price_change_pct < -threshold_pct, "target"] = -1  # SHORT
        
        # Также сохраняем непрерывное значение для регрессии
        df["target_regression"] = price_change_pct
        
        # Удаляем строки где target = NaN (последние forward_periods строк)
        df = df.dropna(subset=["target"])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список названий всех созданных фичей."""
        return self.feature_names.copy()
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Подготавливает данные для обучения ML-модели.
        
        Args:
            df: DataFrame с фичами и целевой переменной
        
        Returns:
            (X, y) где X - матрица фичей, y - целевая переменная
        """
        # Выбираем только фичи (исключаем исходные колонки и target)
        exclude_cols = ["open", "high", "low", "close", "volume", "turnover", "target", "target_regression"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        # Целевая переменная (если есть)
        y = None
        if "target" in df.columns:
            y = df["target"].values
        
        return X, y

