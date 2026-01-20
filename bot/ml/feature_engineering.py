"""
Модуль для создания фичей (признаков) из исторических данных для ML-моделей.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
# Подавляем предупреждения о TA-Lib от pandas_ta
warnings.filterwarnings('ignore', message='.*Requires TA-Lib.*')
warnings.filterwarnings('ignore', message='.*pip install TA-Lib.*')
# Подавляем предупреждения о фрагментации DataFrame и устаревших методах
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
warnings.filterwarnings('ignore', message='.*Series.fillna with.*method.*is deprecated.*')
warnings.filterwarnings('ignore', message='.*fillna.*method.*is deprecated.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*fillna.*method.*')

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
        
            # ГЛОБАЛЬНАЯ ЗАЩИТА: Заменяем все None на NaN во всем DataFrame перед обработкой
            # Это предотвращает ошибки сравнения "'>' not supported between instances of 'float' and 'NoneType'"
            for col in df.columns:
                try:
                    # Заменяем None на NaN для всех колонок
                    # Проверяем наличие None значений
                    if df[col].dtype == 'object':
                        # Для object колонок проверяем наличие None
                        # Избегаем прямого сравнения с None, используем isnull()
                        mask_none = df[col].isnull()
                        # Дополнительная проверка через apply для безопасности
                        try:
                            none_mask = df[col].apply(lambda x: x is None if pd.notna(x) else True)
                            mask_none = mask_none | none_mask
                        except:
                            pass
                        if mask_none.any():
                            df[col] = df[col].where(~mask_none, np.nan)
                    else:
                        # Для числовых колонок также заменяем None
                        if df[col].isnull().any():
                            df[col] = df[col].replace([None], np.nan)
                    
                    # Преобразуем в числовой тип, если возможно
                    if col in required_cols or col in ['atr', 'atr_pct', 'rsi', 'macd', 'macd_signal', 'adx', 'plus_di', 'minus_di', 'bb_upper', 'bb_lower', 'bb_middle', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'volume_ratio', 'obv']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    # Если ошибка при обработке колонки, заполняем NaN
                    print(f"[feature_engineering] Warning: Error processing column '{col}': {e}")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Заполняем NaN в основных колонках безопасными значениями (forward fill, затем backward fill)
            for col in required_cols:
                if col in df.columns:
                    try:
                        # Используем новые методы вместо устаревшего fillna(method=...)
                        df[col] = df[col].ffill().bfill()
                        # Если все еще есть NaN, заполняем средним значением
                        if df[col].isna().any():
                            mean_val = df[col].mean()
                            if pd.notna(mean_val):
                                df[col] = df[col].fillna(mean_val)
                            else:
                                df[col] = df[col].fillna(1.0)  # Fallback значение
                        # Финальная проверка: заменяем оставшиеся None на безопасное значение
                        df[col] = df[col].replace([None], np.nan).fillna(1.0)
                    except Exception as e:
                        # Если ошибка, используем безопасное значение
                        print(f"[feature_engineering] Warning: Error filling NaN in column '{col}': {e}")
                        df[col] = df[col].replace([None], np.nan).fillna(1.0)
        
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
        # Защита от деления на ноль и None значений
        volume_safe = pd.to_numeric(df.get("volume", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        volume_sma_safe = pd.to_numeric(df.get("volume_sma_20", pd.Series([1]*len(df), index=df.index)), errors='coerce').replace(0, 1).fillna(1)
        df["volume_ratio"] = volume_safe / volume_sma_safe
        df["volume_ratio"] = df["volume_ratio"].fillna(1)  # Если все еще есть NaN, используем 1
        
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
        # Собираем все lag колонки сразу для оптимизации (избегаем фрагментации DataFrame)
        lag_columns = {}
        for lag in [1, 2, 3, 5, 10]:
            lag_columns[f"close_lag_{lag}"] = df["close"].shift(lag)
            lag_columns[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            lag_columns[f"price_change_lag_{lag}"] = df["price_change"].shift(lag)
        if lag_columns:
            df = pd.concat([df, pd.DataFrame(lag_columns, index=df.index)], axis=1)
        
        # === Скользящие статистики ===
        # Собираем все скользящие статистики сразу для оптимизации
        rolling_columns = {}
        for window in [5, 10, 20]:
            rolling_columns[f"close_std_{window}"] = df["close"].rolling(window=window).std()
            rolling_columns[f"close_mean_{window}"] = df["close"].rolling(window=window).mean()
            rolling_columns[f"volume_mean_{window}"] = df["volume"].rolling(window=window).mean()
        if rolling_columns:
            df = pd.concat([df, pd.DataFrame(rolling_columns, index=df.index)], axis=1)
        
        # === Временные фичи ===
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["day_of_month"] = df.index.day
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
            
            # Циклические фичи для лучшего обучения модели (sin/cos преобразования)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
            df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31.0)
            df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31.0)
            
            # Торговые сессии (азиатская, европейская, американская)
            df["is_asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
            df["is_european_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
            df["is_american_session"] = ((df["hour"] >= 16) | (df["hour"] < 24)).astype(int)
            
            # Дополнительные временные циклы (дублируем на случай пропуска индексов)
            df["hour_sin_cycle"] = np.sin(2 * np.pi * df.index.hour / 24.0)
            df["hour_cos_cycle"] = np.cos(2 * np.pi * df.index.hour / 24.0)
            df["dow_sin_cycle"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
            df["dow_cos_cycle"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
        
        # === Дополнительные фичи ===
        
        # Momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        
        # Rate of Change
        df["roc_5"] = ta.roc(df["close"], length=5)
        df["roc_10"] = ta.roc(df["close"], length=10)
        
        # === Advanced statistical & SMC/ICT features ===
        # 1) Z-Score по цене (20‑барное окно)
        window_z = 20
        close_rolling_mean = df["close"].rolling(window=window_z).mean()
        close_rolling_std = df["close"].rolling(window=window_z).std()
        df["z_score"] = (df["close"] - close_rolling_mean) / close_rolling_std
        df["z_score"] = df["z_score"].replace([np.inf, -np.inf], np.nan)
        
        # 2) Волатильность (стд. отклонение закрытия)
        df["volatility_10"] = df["close"].rolling(window=10).std()
        df["volatility_20"] = df["close"].rolling(window=20).std()
        
        # 3) Distance to EMA200 (в %)
        try:
            df["ema_200"] = ta.ema(df["close"], length=200)
            ema200_safe = pd.to_numeric(df.get("ema_200"), errors="coerce")
            df["dist_to_ema200_pct"] = (df["close"] - ema200_safe) / ema200_safe * 100.0
            df["dist_to_ema200_pct"] = df["dist_to_ema200_pct"].replace([np.inf, -np.inf], np.nan)
        except Exception:
            df["dist_to_ema200_pct"] = 0.0
        
        # 4) FVG (Fair Value Gap) — простая бинарная детекция
        # Bullish FVG: low[i] > high[i-2]
        df["fvg_up"] = ((df["low"] > df["high"].shift(2))).astype(int)
        # Bearish FVG: high[i] < low[i-2]
        df["fvg_down"] = ((df["high"] < df["low"].shift(2))).astype(int)
        # Сводный признак
        df["fvg_trend"] = df["fvg_up"] - df["fvg_down"]
        
        # 5) Wick-to-Body ratio (снятие ликвидности / SMC контекст)
        body = (df["close"] - df["open"]).abs()
        upper_wick = df[["close", "open"]].max(axis=1) - df["high"]
        lower_wick = df["low"] - df[["close", "open"]].min(axis=1)
        df["upper_wick_body_ratio"] = np.where(body > 0, (upper_wick.abs() / body), 0.0)
        df["lower_wick_body_ratio"] = np.where(body > 0, (lower_wick.abs() / body), 0.0)
        
        # 6) Market Structure — упрощённый флаг последнего пробоя (BOS)
        ms_lookback = 20
        prev_high_max = df["high"].shift(1).rolling(window=ms_lookback).max()
        prev_low_min = df["low"].shift(1).rolling(window=ms_lookback).min()
        df["bos_up"] = (df["high"] > prev_high_max).astype(int)
        df["bos_down"] = (df["low"] < prev_low_min).astype(int)
        df["market_structure_trend"] = df["bos_up"] - df["bos_down"]
        
        # === Свечные паттерны (Candlestick Patterns) ===
        try:
            important_patterns = ["doji", "hammer", "engulfing", "morningstar", "eveningstar",
                                  "shootingstar", "hangingman", "invertedhammer"]
            for pattern_name in important_patterns:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*Requires TA-Lib.*')
                        warnings.filterwarnings('ignore', message='.*pip install TA-Lib.*')
                        pattern_result = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name=pattern_name)
                    pattern_col = f"cdl_{pattern_name}"
                    if pattern_result is None or pattern_result.empty:
                        df[pattern_col] = 0
                    elif isinstance(pattern_result, pd.DataFrame):
                        df[pattern_col] = pattern_result.iloc[:, 0] if len(pattern_result.columns) > 0 else 0
                    else:
                        df[pattern_col] = pattern_result
                except Exception:
                    df[f"cdl_{pattern_name}"] = 0
        except Exception:
            pass
        
        # === Взаимодействия между индикаторами ===
        macd_clean = pd.to_numeric(df["macd"], errors='coerce').fillna(0)
        macd_signal_clean = pd.to_numeric(df["macd_signal"], errors='coerce').fillna(0)
        macd_prev = pd.to_numeric(df["macd"].shift(1), errors='coerce').fillna(0)
        macd_signal_prev = pd.to_numeric(df["macd_signal"].shift(1), errors='coerce').fillna(0)
        
        df["macd_cross_above"] = ((macd_clean > macd_signal_clean) &
                                  (macd_prev <= macd_signal_prev)).fillna(False).astype(int)
        df["macd_cross_below"] = ((macd_clean < macd_signal_clean) &
                                  (macd_prev >= macd_signal_prev)).fillna(False).astype(int)
        
        # EMA Crossovers
        ema12_safe = pd.to_numeric(df.get("ema_12", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        ema26_safe = pd.to_numeric(df.get("ema_26", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        ema12_prev = ema12_safe.shift(1).fillna(0)
        ema26_prev = ema26_safe.shift(1).fillna(0)
        ema_cross_features = {
            "ema_cross_above": ((ema12_safe > ema26_safe) & (ema12_prev <= ema26_prev)).fillna(False).astype(int),
            "ema_cross_below": ((ema12_safe < ema26_safe) & (ema12_prev >= ema26_prev)).fillna(False).astype(int),
        }
        ema_cross_df = pd.DataFrame(ema_cross_features, index=df.index)
        df = pd.concat([df, ema_cross_df], axis=1)
        
        # Price vs Moving Averages
        # Собираем все фичи в словарь для оптимизации (избегаем фрагментации)
        combined_indicators_features = {}
        
        close_safe = pd.to_numeric(df.get("close", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        sma20_safe = pd.to_numeric(df.get("sma_20", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        sma50_safe = pd.to_numeric(df.get("sma_50", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        sma200_safe = pd.to_numeric(df.get("sma_200", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        combined_indicators_features["price_above_sma20"] = (close_safe > sma20_safe).fillna(False).astype(int)
        combined_indicators_features["price_above_sma50"] = (close_safe > sma50_safe).fillna(False).astype(int)
        combined_indicators_features["price_above_sma200"] = (close_safe > sma200_safe).fillna(False).astype(int)
        combined_indicators_features["sma20_above_sma50"] = (sma20_safe > sma50_safe).fillna(False).astype(int)
        combined_indicators_features["sma50_above_sma200"] = (sma50_safe > sma200_safe).fillna(False).astype(int)
        
        # RSI Divergence indicators (упрощенная версия)
        rsi_safe = pd.to_numeric(df.get("rsi", pd.Series([50]*len(df), index=df.index)), errors='coerce').fillna(50)
        combined_indicators_features["rsi_oversold"] = (rsi_safe < 30).fillna(False).astype(int)
        combined_indicators_features["rsi_overbought"] = (rsi_safe > 70).fillna(False).astype(int)
        combined_indicators_features["rsi_neutral"] = ((rsi_safe >= 30) & (rsi_safe <= 70)).fillna(False).astype(int)
        
        # Bollinger Bands interactions
        high_safe = pd.to_numeric(df.get("high", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        low_safe = pd.to_numeric(df.get("low", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        bb_upper_safe = pd.to_numeric(df.get("bb_upper", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        bb_lower_safe = pd.to_numeric(df.get("bb_lower", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        combined_indicators_features["price_touch_bb_upper"] = (high_safe >= bb_upper_safe).fillna(False).astype(int)
        combined_indicators_features["price_touch_bb_lower"] = (low_safe <= bb_lower_safe).fillna(False).astype(int)
        combined_indicators_features["price_in_bb_middle"] = (
            (close_safe > bb_lower_safe) & (close_safe < bb_upper_safe)
        ).fillna(False).astype(int)
        
        # Добавляем все фичи одним pd.concat
        if combined_indicators_features:
            df = pd.concat([df, pd.DataFrame(combined_indicators_features, index=df.index)], axis=1)
        
        # === ADX Strength indicators и нормализованные метрики ===
        # Собираем все эти фичи в словарь для оптимизации
        adx_normalized_features = {}
        
        # ADX Strength indicators
        adx_safe = pd.to_numeric(df.get("adx", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        adx_normalized_features["adx_strong_trend"] = (adx_safe > 25).fillna(False).astype(int)
        adx_normalized_features["adx_weak_trend"] = ((adx_safe > 20) & (adx_safe <= 25)).fillna(False).astype(int)
        adx_normalized_features["adx_no_trend"] = (adx_safe <= 20).fillna(False).astype(int)
        
        # Нормализованные расстояния до индикаторов
        # Защита от None/NaN перед вычислениями
        bb_upper_safe = pd.to_numeric(df.get("bb_upper", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        bb_lower_safe = pd.to_numeric(df.get("bb_lower", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
        close_safe = (
            pd.to_numeric(df.get("close", pd.Series([1]*len(df), index=df.index)), errors='coerce')
            .replace(0, 1)
            .fillna(1)
        )
        sma20_safe = (
            pd.to_numeric(df.get("sma_20", pd.Series([1]*len(df), index=df.index)), errors='coerce')
            .replace(0, 1)
            .fillna(1)
        )
        sma50_safe = (
            pd.to_numeric(df.get("sma_50", pd.Series([1]*len(df), index=df.index)), errors='coerce')
            .replace(0, 1)
            .fillna(1)
        )
        rsi_safe = pd.to_numeric(df.get("rsi", pd.Series([50]*len(df), index=df.index)), errors='coerce').fillna(50)
        
        if bb_upper_safe.notna().any() and bb_lower_safe.notna().any():
            bb_range = bb_upper_safe - bb_lower_safe
            adx_normalized_features["price_norm_bb"] = np.where(
                bb_range > 0,
                (close_safe - bb_lower_safe) / bb_range,
                0,
            )
        else:
            adx_normalized_features["price_norm_bb"] = 0
            
            # Нормализованное расстояние до SMA
            adx_normalized_features["price_norm_sma20"] = (close_safe - sma20_safe) / sma20_safe
            adx_normalized_features["price_norm_sma50"] = (close_safe - sma50_safe) / sma50_safe
            
            # Нормализованный RSI (0-1)
            adx_normalized_features["rsi_norm"] = rsi_safe / 100.0
            
            # === Комбинированные индикаторы ===
            
            # Trend Strength Score (комбинация ADX, MACD, EMA)
            # Защита от None/NaN перед вычислениями
            macd_hist_safe = pd.to_numeric(df.get("macd_hist", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            ema12_safe = pd.to_numeric(df.get("ema_12", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            ema26_safe = pd.to_numeric(df.get("ema_26", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            
            adx_normalized_features["trend_strength"] = (
                (adx_safe / 100.0) * 0.4 +  # ADX вклад 40%
                (macd_hist_safe.abs() / (close_safe * 0.01)) * 0.3 +  # MACD вклад 30%
                ((ema12_safe - ema26_safe).abs() / close_safe) * 0.3  # EMA вклад 30%
            ).fillna(0)
            
            # Добавляем все ADX и нормализованные фичи сразу через pd.concat
            if adx_normalized_features:
                df = pd.concat([df, pd.DataFrame(adx_normalized_features, index=df.index)], axis=1)
            
            # === Расширенные интеракции индикаторов ===
            # Собираем все интеракции в словарь для оптимизации
            interaction_features = {}
            
            # Volume Confirmation Score
            volume_ratio_safe = pd.to_numeric(df.get("volume_ratio", pd.Series([1]*len(df), index=df.index)), errors='coerce').fillna(1)
            obv_safe = pd.to_numeric(df.get("obv", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            interaction_features["volume_confirmation"] = (
                (volume_ratio_safe > 1.0).astype(int) * 0.5 +  # Объем выше среднего
                (obv_safe.pct_change() > 0).astype(int) * 0.5  # OBV растет
            ).fillna(0)
            
            # RSI + MACD комбинация (сильные сигналы когда оба согласны)
            # Заменяем None на NaN и заполняем безопасными значениями перед сравнением
            rsi_clean = pd.to_numeric(df.get("rsi", pd.Series([50]*len(df), index=df.index)), errors='coerce').fillna(50)
            macd_clean = pd.to_numeric(df.get("macd", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            macd_signal_clean = pd.to_numeric(df.get("macd_signal", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            interaction_features["rsi_macd_bullish"] = ((rsi_clean < 70) & (macd_clean > macd_signal_clean)).fillna(False).astype(int)
            interaction_features["rsi_macd_bearish"] = ((rsi_clean > 30) & (macd_clean < macd_signal_clean)).fillna(False).astype(int)
            
            # ADX + Price Position (сильный тренд + цена в правильной позиции)
            adx_clean = pd.to_numeric(df.get("adx", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            plus_di_clean = pd.to_numeric(df.get("plus_di", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            minus_di_clean = pd.to_numeric(df.get("minus_di", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            close_clean = pd.to_numeric(df.get("close", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            sma20_clean = pd.to_numeric(df.get("sma_20", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            interaction_features["adx_price_bullish"] = ((adx_clean > 25) & (plus_di_clean > minus_di_clean) & 
                                   (close_clean > sma20_clean)).fillna(False).astype(int)
            interaction_features["adx_price_bearish"] = ((adx_clean > 25) & (minus_di_clean > plus_di_clean) & 
                                   (close_clean < sma20_clean)).fillna(False).astype(int)
            
            # Bollinger Bands + RSI (перекупленность/перепроданность + экстремальные цены)
            bb_lower_clean = pd.to_numeric(df.get("bb_lower", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            bb_upper_clean = pd.to_numeric(df.get("bb_upper", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            interaction_features["bb_rsi_oversold"] = ((rsi_clean < 30) & (close_clean < bb_lower_clean)).fillna(False).astype(int)
            interaction_features["bb_rsi_overbought"] = ((rsi_clean > 70) & (close_clean > bb_upper_clean)).fillna(False).astype(int)
            
            # Volume + Price Momentum (объем подтверждает движение)
            # Защита от None/NaN перед сравнением
            volume_ratio_safe = pd.to_numeric(df.get("volume_ratio", pd.Series([1]*len(df), index=df.index)), errors='coerce').fillna(1)
            price_change_safe = pd.to_numeric(df.get("price_change", pd.Series([0]*len(df), index=df.index)), errors='coerce').fillna(0)
            interaction_features["volume_price_momentum"] = (
                (volume_ratio_safe > 1.2).astype(int) * 
                np.sign(price_change_safe).fillna(0)
            ).fillna(0)
            
            # Добавляем все интеракции сразу через pd.concat
            if interaction_features:
                df = pd.concat([df, pd.DataFrame(interaction_features, index=df.index)], axis=1)
            
            # === Волатильность-скорректированные метрики ===
            # Собираем все волатильность-скорректированные метрики в словарь
            volatility_features = {}
            
            # Нормализованные изменения цены относительно ATR
            if "atr" in df.columns and df["atr"].notna().any():
                volatility_features["price_change_atr_ratio"] = (df["price_change_abs"] / (df["atr"] / df["close"]).fillna(1)).fillna(0)
            
            # Волатильность-скорректированное расстояние до SMA
            if "atr" in df.columns and "sma_20" in df.columns:
                atr_normalized = (df["atr"] / df["close"]).fillna(0.01)
                volatility_features["sma20_distance_atr"] = ((df["close"] - df["sma_20"]) / (df["sma_20"] * atr_normalized)).fillna(0)
            
            # Добавляем все волатильность-скорректированные метрики сразу через pd.concat
            if volatility_features:
                df = pd.concat([df, pd.DataFrame(volatility_features, index=df.index)], axis=1)
            
            # === Фичи на основе паттернов движения ===
            
            # Собираем все паттерны движения в словарь
            pattern_features = {}
            
            # Последовательность движений (тренд вверх/вниз)
            pattern_features["price_sequence_up"] = ((df["close"] > df["close"].shift(1)) & 
                                   (df["close"].shift(1) > df["close"].shift(2))).astype(int)
            pattern_features["price_sequence_down"] = ((df["close"] < df["close"].shift(1)) & 
                                     (df["close"].shift(1) < df["close"].shift(2))).astype(int)
            
            # Сила свечи (размер тела относительно диапазона)
            pattern_features["candle_body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, 1)
            pattern_features["candle_body_ratio"] = pattern_features["candle_body_ratio"].fillna(0)
            
            # Направление свечи
            pattern_features["candle_bullish"] = (df["close"] > df["open"]).astype(int)
            pattern_features["candle_bearish"] = (df["close"] < df["open"]).astype(int)
            
            # Добавляем все паттерны сразу через pd.concat
            if pattern_features:
                df = pd.concat([df, pd.DataFrame(pattern_features, index=df.index)], axis=1)
            
            # === Фичи на основе объема и цены ===
            # Собираем все фичи объема и цены в словарь
            volume_price_features = {}
            
            # Цена-объемная корреляция (скользящее окно)
            # Вычисляем процентные изменения, затем корреляцию между ними
            price_change = df["close"].pct_change()
            volume_change = df["volume"].pct_change()
            for window in [10, 20]:
                # Вычисляем корреляцию между изменениями цены и объема в скользящем окне
                volume_price_features[f"price_volume_corr_{window}"] = price_change.rolling(window=window).corr(volume_change).fillna(0)
            
            # Объем-взвешенная цена (VWAP отклонение)
            if "volume" in df.columns:
                volume_price_features["vwap_deviation"] = ((df["close"] - df["sma_20"]) / df["sma_20"]).fillna(0)  # Упрощенная версия VWAP
            
            # Добавляем все фичи объема и цены сразу через pd.concat
            if volume_price_features:
                df = pd.concat([df, pd.DataFrame(volume_price_features, index=df.index)], axis=1)
            
            # === Фичи на основе рыночной структуры ===
            
            # Собираем все фичи рыночной структуры в словарь
            market_structure_features = {}
            
            # Высокие/низкие за период (поддержка/сопротивление)
            for window in [20, 50]:
                # Защита от деления на ноль и None значений
                close_safe = df["close"].replace(0, np.nan)
                close_safe = close_safe.bfill().ffill()  # Заполняем NaN вперед и назад
                if close_safe.isna().any():
                    close_safe = close_safe.fillna(df["close"].mean() if not df["close"].empty else 1.0)
                
                high_max = df["high"].rolling(window=window).max()
                low_min = df["low"].rolling(window=window).min()
                
                # Защита от деления на ноль
                price_diff_high = (df["close"] - high_max).abs()
                price_diff_low = (df["close"] - low_min).abs()
                market_structure_features[f"price_near_high_{window}"] = (price_diff_high / close_safe) < 0.01
                market_structure_features[f"price_near_low_{window}"] = (price_diff_low / close_safe) < 0.01
                market_structure_features[f"price_near_high_{window}"] = market_structure_features[f"price_near_high_{window}"].fillna(False).astype(int)
                market_structure_features[f"price_near_low_{window}"] = market_structure_features[f"price_near_low_{window}"].fillna(False).astype(int)
            
            # Дивергенция RSI (упрощенная версия)
            # Заменяем None на NaN перед вычислениями
            rsi_series = df["rsi"].replace([None], np.nan) if "rsi" in df.columns else pd.Series([50.0]*len(df), index=df.index)
            close_series = df["close"].replace([None], np.nan)
            
            rsi_high = rsi_series.rolling(window=14).max()
            price_high = close_series.rolling(window=14).max()
            rsi_low = rsi_series.rolling(window=14).min()
            price_low = close_series.rolling(window=14).min()
            
            # Заполняем NaN перед сравнением безопасными значениями
            rsi_first = rsi_series.iloc[0] if len(rsi_series) > 0 and pd.notna(rsi_series.iloc[0]) else 50.0
            price_first = close_series.iloc[0] if len(close_series) > 0 and pd.notna(close_series.iloc[0]) else 1.0
            
            rsi_high_shifted = rsi_high.shift(1).replace([None], np.nan).fillna(rsi_first)
            price_high_shifted = price_high.shift(1).replace([None], np.nan).fillna(price_first)
            rsi_low_shifted = rsi_low.shift(1).replace([None], np.nan).fillna(rsi_first)
            price_low_shifted = price_low.shift(1).replace([None], np.nan).fillna(price_first)
            
            # Убеждаемся, что все значения числовые перед сравнением
            rsi_vals = pd.to_numeric(rsi_series, errors='coerce').fillna(rsi_first)
            close_vals = pd.to_numeric(close_series, errors='coerce').fillna(price_first)
            rsi_high_shifted = pd.to_numeric(rsi_high_shifted, errors='coerce').fillna(rsi_first)
            price_high_shifted = pd.to_numeric(price_high_shifted, errors='coerce').fillna(price_first)
            rsi_low_shifted = pd.to_numeric(rsi_low_shifted, errors='coerce').fillna(rsi_first)
            price_low_shifted = pd.to_numeric(price_low_shifted, errors='coerce').fillna(price_first)
            
            market_structure_features["rsi_divergence_bearish"] = ((rsi_vals < rsi_high_shifted) & 
                                        (close_vals > price_high_shifted)).fillna(False).astype(int)
            market_structure_features["rsi_divergence_bullish"] = ((rsi_vals > rsi_low_shifted) & 
                                        (close_vals < price_low_shifted)).fillna(False).astype(int)
            
            # Добавляем все фичи рыночной структуры сразу через pd.concat
            if market_structure_features:
                df = pd.concat([df, pd.DataFrame(market_structure_features, index=df.index)], axis=1)
            
            # === Скользящие окна для временных паттернов ===
            
            # Скользящие средние изменений - собираем все колонки сразу для оптимизации
            new_columns = {}
            for window in [3, 5, 10]:
                new_columns[f"price_change_ma_{window}"] = df["price_change"].rolling(window=window).mean()
                new_columns[f"volume_change_ma_{window}"] = df["volume"].pct_change().rolling(window=window).mean()
            
            # Добавляем все колонки сразу через pd.concat для избежания фрагментации
            if new_columns:
                df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
            
            # Мягкая обработка NaN: сохраняем строки, где хотя бы основные данные (OHLCV) присутствуют
            # Сохраняем исходный DataFrame перед dropna для fallback
            df_before_dropna = df.copy()
            
            # Сохраняем строки, где хотя бы основные колонки не все NaN
            key_columns = ["open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in key_columns):
                df = df[df[key_columns].notna().any(axis=1)]
            
            # Удаляем только строки, где все значения NaN
            df = df.dropna(how='all')
            
            # Если после обработки DataFrame пуст, возвращаем данные до dropna (с NaN, но с данными)
            if len(df) == 0:
                print(f"[feature_engineering] ⚠️ Warning: All rows removed after dropna, returning data before dropna")
                # Возвращаем данные до dropna, но удаляем только строки где все OHLCV NaN
                if all(col in df_before_dropna.columns for col in key_columns):
                    df_before_dropna = df_before_dropna[df_before_dropna[key_columns].notna().any(axis=1)]
                # Заполняем NaN в фичах нулями, чтобы сохранить строки
                feature_cols = [col for col in df_before_dropna.columns if col not in key_columns]
                if feature_cols:
                    # Используем pd.concat для избежания фрагментации
                    feature_data = df_before_dropna[feature_cols].fillna(0)
                    df_before_dropna = pd.concat([df_before_dropna[key_columns], feature_data], axis=1)
                df = df_before_dropna
            
            # Дефрагментируем DataFrame в конце для оптимизации производительности
            df = df.copy()
        
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
        use_risk_adjusted: bool = True,  # Использовать риск-скорректированную целевую переменную
        min_risk_reward_ratio: float = 2.0,  # Минимальное соотношение риск/прибыль
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
        
        # Защита от деления на ноль и None значений
        # Сначала заменяем None на NaN
        current_price = current_price.replace([None], np.nan)
        future_price = future_price.replace([None], np.nan)
        
        # Заменяем нули на NaN для безопасного деления
        current_price_safe = current_price.replace(0, np.nan)
        current_price_safe = current_price_safe.bfill().ffill()  # Заполняем NaN вперед и назад
        if current_price_safe.isna().any():
            mean_val = current_price.mean() if not current_price.empty and current_price.notna().any() else 1.0
            current_price_safe = current_price_safe.fillna(mean_val)
        
        # Убеждаемся, что нет None значений
        current_price_safe = current_price_safe.replace([None], np.nan).fillna(1.0)
        future_price = future_price.replace([None], np.nan).fillna(current_price_safe)
        
        # Вычисляем процентное изменение
        price_change_pct = ((future_price - current_price) / current_price_safe) * 100
        # Заполняем NaN и None нулями
        price_change_pct = price_change_pct.replace([None, np.nan, np.inf, -np.inf], 0).fillna(0)
        
        # Классифицируем:
        # 1 = LONG (цена вырастет больше чем threshold_pct %)
        # -1 = SHORT (цена упадет больше чем threshold_pct %)
        # 0 = HOLD (изменение в пределах threshold_pct %)
        # price_change_pct уже в процентах, поэтому сравниваем напрямую
        
        # Классифицируем: используем более низкий порог для получения большего количества сигналов
        df["target"] = 0
        
        # Вычисляем риск (максимальная просадка до достижения цели)
        if use_risk_adjusted and "atr" in df.columns:
            # Вычисляем максимальную просадку (drawdown) на пути к цели
            # Для LONG: ищем минимальную цену между текущей и будущей
            # Для SHORT: ищем максимальную цену между текущей и будущей
            max_drawdown_long = np.zeros(len(df))
            max_drawdown_short = np.zeros(len(df))
            
            for i in range(len(df) - forward_periods):
                future_idx = min(i + forward_periods, len(df) - 1)
                # Для LONG: ищем минимальную цену на пути
                if i < len(df) - forward_periods:
                    current_price_val = current_price.iloc[i]
                    if pd.notna(current_price_val) and current_price_val > 0:
                        prices_window = df["low"].iloc[i:future_idx+1]
                        if len(prices_window) > 0:
                            min_price = prices_window.min()
                            if pd.notna(min_price):
                                max_drawdown_long[i] = ((current_price_val - min_price) / current_price_val) * 100
                    
                    # Для SHORT: ищем максимальную цену на пути
                    if pd.notna(current_price_val) and current_price_val > 0:
                        prices_window = df["high"].iloc[i:future_idx+1]
                        if len(prices_window) > 0:
                            max_price = prices_window.max()
                            if pd.notna(max_price):
                                max_drawdown_short[i] = ((max_price - current_price_val) / current_price_val) * 100
            
            df["max_drawdown_long"] = max_drawdown_long
            df["max_drawdown_short"] = max_drawdown_short
            
            # Заполняем NaN в max_drawdown нулями (если нет данных о просадке, считаем риск = 0)
            df["max_drawdown_long"] = df["max_drawdown_long"].fillna(0)
            df["max_drawdown_short"] = df["max_drawdown_short"].fillna(0)
            
            # Риск-скорректированная классификация
            # Для LONG: прибыль должна быть >= min_risk_reward_ratio * риск
            # Для SHORT: прибыль должна быть >= min_risk_reward_ratio * риск
            # Заполняем NaN в price_change_pct нулями перед сравнением
            price_change_pct_safe = price_change_pct.fillna(0)
            # Убеждаемся, что max_drawdown не содержит None/NaN перед сравнением
            max_drawdown_long_safe = df["max_drawdown_long"].fillna(0)
            max_drawdown_short_safe = df["max_drawdown_short"].fillna(0)
            # Используем pd.Series для безопасного сравнения
            mask_long_risk = (price_change_pct_safe > 0) & (price_change_pct_safe >= max_drawdown_long_safe * min_risk_reward_ratio)
            mask_short_risk = (price_change_pct_safe < 0) & (price_change_pct_safe.abs() >= max_drawdown_short_safe * min_risk_reward_ratio)
            # Заполняем NaN в масках False
            mask_long_risk = mask_long_risk.fillna(False)
            mask_short_risk = mask_short_risk.fillna(False)
        else:
            mask_long_risk = pd.Series([False] * len(df), index=df.index)
            mask_short_risk = pd.Series([False] * len(df), index=df.index)
        
        if use_atr_threshold and "atr_pct" in df.columns and not df["atr_pct"].isna().all():
            # Используем динамический порог на основе ATR: минимум из базового порога и 0.6 * ATR
            # Это позволит адаптироваться к волатильности - при низкой волатильности используем базовый порог,
            # при высокой - уменьшаем порог до 0.6 * ATR для большего количества сигналов
            dynamic_threshold = df["atr_pct"] * 0.6  # 60% от ATR
            # Используем минимум из базового порога и динамического (более агрессивный порог = больше сигналов)
            # Заполняем NaN в dynamic_threshold базовым порогом
            dynamic_threshold = dynamic_threshold.fillna(threshold_pct)
            # Убеждаемся, что нет None значений перед np.minimum
            dynamic_threshold = dynamic_threshold.replace([None, np.inf, -np.inf], threshold_pct)
            # Берем минимум между базовым порогом и динамическим
            # Используем pandas min вместо np.minimum для безопасности
            # Заменяем None и inf на безопасные значения
            dynamic_threshold_clean = dynamic_threshold.replace([None, np.inf, -np.inf], threshold_pct)
            dynamic_threshold_clean = dynamic_threshold_clean.fillna(threshold_pct)
            # Используем np.minimum, но с защитой
            effective_threshold = pd.Series(
                np.minimum(threshold_pct, dynamic_threshold_clean.values),
                index=dynamic_threshold.index
            )
            # Заполняем оставшиеся NaN
            effective_threshold = effective_threshold.fillna(threshold_pct)
            # Заменяем None на threshold_pct
            effective_threshold = effective_threshold.replace([None], threshold_pct)
            
            # Классифицируем с динамическим порогом для каждой строки
            # Заполняем NaN перед сравнением
            price_change_pct_clean = price_change_pct.fillna(0)
            effective_threshold_clean = effective_threshold.fillna(threshold_pct)
            mask_long = price_change_pct_clean > effective_threshold_clean
            mask_short = price_change_pct_clean < -effective_threshold_clean
            # Заполняем NaN в масках False
            mask_long = mask_long.fillna(False)
            mask_short = mask_short.fillna(False)
            
            # Если используется риск-скорректированная классификация, применяем дополнительный фильтр
            if use_risk_adjusted:
                mask_long = mask_long & mask_long_risk
                mask_short = mask_short & mask_short_risk
            
            df.loc[mask_long, "target"] = 1  # LONG
            df.loc[mask_short, "target"] = -1  # SHORT
        else:
            # Статический порог (оригинальная логика)
            # Заполняем NaN перед сравнением
            price_change_pct_clean = price_change_pct.fillna(0)
            mask_long = price_change_pct_clean > threshold_pct
            mask_short = price_change_pct_clean < -threshold_pct
            # Заполняем NaN в масках False
            mask_long = mask_long.fillna(False)
            mask_short = mask_short.fillna(False)
            
            # Если используется риск-скорректированная классификация, применяем дополнительный фильтр
            if use_risk_adjusted:
                mask_long = mask_long & mask_long_risk
                mask_short = mask_short & mask_short_risk
            
            df.loc[mask_long, "target"] = 1  # LONG
            df.loc[mask_short, "target"] = -1  # SHORT
        
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

