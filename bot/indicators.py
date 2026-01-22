import os
# Исправление проблемы с Numba кэшированием в pandas_ta
# Если NUMBA_CACHE_DIR не установлен, используем /tmp/numba_cache или домашнюю директорию
if "NUMBA_CACHE_DIR" not in os.environ:
    # Пробуем сначала /tmp/numba_cache
    cache_dir = "/tmp/numba_cache"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
    except (PermissionError, OSError):
        # Если нет прав на /tmp, используем домашнюю директорию
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".numba_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir

import pandas as pd
import numpy as np
import pandas_ta as ta


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    # Пробуем сначала как миллисекунды, если не получилось — пусть pandas сам парсит строки.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    except (ValueError, TypeError):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def compute_4h_context(df_15m: pd.DataFrame, adx_length: int = 14) -> pd.DataFrame:
    """
    Resample 15m candles to 4H and compute ADX (trend filter only).
    Forward-fills back to 15m index.
    """
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_4h = df_15m.resample("4h").agg(ohlcv).dropna()
    adx = ta.adx(high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], length=adx_length)
    df_4h = df_4h.join(adx)

    # Map back to 15m index
    mapped = df_4h.reindex(df_15m.index, method="ffill")
    df_15m = df_15m.copy()
    df_15m["adx"] = mapped[f"ADX_{adx_length}"]
    return df_15m


def compute_atr_higher_timeframes(df_15m: pd.DataFrame, atr_length: int = 14) -> pd.DataFrame:
    """
    Вычисляет ATR на 1H и 4H таймфреймах для анализа среднесрочной волатильности.
    Используется вместо 15-минутного ATR для фильтрации точек входа.
    """
    df_15m = df_15m.copy()
    
    # Resample на 1H и вычисляем ATR
    ohlcv_1h = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_1h = df_15m.resample("1h").agg(ohlcv_1h).dropna()
    
    if len(df_1h) >= atr_length:
        atr_1h = ta.atr(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], length=atr_length)
        if isinstance(atr_1h, pd.Series):
            df_1h["atr_1h"] = atr_1h
        elif isinstance(atr_1h, pd.DataFrame) and len(atr_1h.columns) > 0:
            df_1h["atr_1h"] = atr_1h.iloc[:, 0]
        else:
            df_1h["atr_1h"] = pd.Series(index=df_1h.index, dtype=float)
    else:
        df_1h["atr_1h"] = pd.Series(index=df_1h.index, dtype=float)
    
    # Resample на 4H и вычисляем ATR
    ohlcv_4h = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_4h = df_15m.resample("4h").agg(ohlcv_4h).dropna()
    
    if len(df_4h) >= atr_length:
        atr_4h = ta.atr(high=df_4h["high"], low=df_4h["low"], close=df_4h["close"], length=atr_length)
        if isinstance(atr_4h, pd.Series):
            df_4h["atr_4h"] = atr_4h
        elif isinstance(atr_4h, pd.DataFrame) and len(atr_4h.columns) > 0:
            df_4h["atr_4h"] = atr_4h.iloc[:, 0]
        else:
            df_4h["atr_4h"] = pd.Series(index=df_4h.index, dtype=float)
    else:
        df_4h["atr_4h"] = pd.Series(index=df_4h.index, dtype=float)
    
    # Map обратно на 15m индекс, используя forward fill
    mapped_1h = df_1h.reindex(df_15m.index, method="ffill")
    mapped_4h = df_4h.reindex(df_15m.index, method="ffill")
    
    # Используем среднее значение ATR с 1H и 4H (или максимальное, если нужно более консервативно)
    # Для среднесрочного трейдинга используем среднее значение
    df_15m["atr_1h"] = mapped_1h["atr_1h"]
    df_15m["atr_4h"] = mapped_4h["atr_4h"]
    
    # Среднее значение ATR для анализа среднесрочной волатильности
    df_15m["atr_avg"] = (df_15m["atr_1h"] + df_15m["atr_4h"]) / 2
    
    # Также оставляем максимальное значение для более консервативного подхода (опционально)
    df_15m["atr_max"] = df_15m[["atr_1h", "atr_4h"]].max(axis=1)
    
    return df_15m


def compute_1h_context(df_15m: pd.DataFrame, di_length: int = 14) -> pd.DataFrame:
    """
    Resample 15m candles to 1H and compute PlusDI/MinusDI (direction).
    Forward-fills back to 15m index.
    """
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_1h = df_15m.resample("1h").agg(ohlcv).dropna()
    adx_result = ta.adx(high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], length=di_length)
    df_1h = df_1h.join(adx_result)

    # Map back to 15m index
    mapped = df_1h.reindex(df_15m.index, method="ffill")
    df_15m = df_15m.copy()
    df_15m["plus_di"] = mapped[f"DMP_{di_length}"]
    df_15m["minus_di"] = mapped[f"DMN_{di_length}"]
    return df_15m


def compute_15m_features(
    df_15m: pd.DataFrame,
    sma_length: int = 20,
    rsi_length: int = 14,
    breakout_lookback: int = 20,
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
) -> pd.DataFrame:
    """
    Adds 15m-level indicators required for entries and scaling.
    """
    df = df_15m.copy()
    # SMA: use rolling with min_periods=1 to avoid NaNs on short series
    df["sma"] = df["close"].rolling(window=sma_length, min_periods=1).mean()
    # Previous SMA value (sma_prev) is required by row-based trend signal generator
    # Shift by 1 to provide previous-period SMA for each row (will be NaN for first row)
    df["sma_prev"] = df["sma"].shift(1)

    # Also compute a trend EMA (faster) for trend strategy
    try:
        df["ema_trend"] = ta.ema(df["close"], length=sma_length)
    except Exception:
        # Fallback to simple EMA implementation
        df["ema_trend"] = df["close"].ewm(span=sma_length, adjust=False).mean()
    df["ema_prev"] = df["ema_trend"].shift(1)
    # Compute short/long EMA pair for double-EMA crossover if configured
    try:
        df["ema_short"] = ta.ema(df["close"], length=kwargs.get('ema_short', df.attrs.get('ema_short', 9)))
        df["ema_long"] = ta.ema(df["close"], length=kwargs.get('ema_long', df.attrs.get('ema_long', 21)))
    except Exception:
        df["ema_short"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=21, adjust=False).mean()
    # RSI: try pandas_ta, but ensure no NaNs remain (fallback to conservative 50)
    try:
        df["rsi"] = ta.rsi(df["close"], length=rsi_length)
    except Exception:
        df["rsi"] = pd.Series(index=df.index, dtype=float)
    df["rsi"] = df["rsi"].fillna(method='bfill').fillna(50.0)

    # Volume moving averages: ensure vol_sma and vol_avg5 always present (min_periods=1)
    df["vol_sma"] = df["volume"].rolling(window=breakout_lookback, min_periods=1).mean()
    df["vol_avg5"] = df["volume"].rolling(window=5, min_periods=1).mean()
    df["recent_high"] = df["high"].rolling(window=breakout_lookback).max().shift(1)
    df["recent_low"] = df["low"].rolling(window=breakout_lookback).min().shift(1)
    
    # ATR для определения волатильности и точек выхода
    atr = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_length)
    # ta.atr возвращает Series, поэтому просто присваиваем его напрямую
    if isinstance(atr, pd.Series):
        df["atr"] = atr
    elif isinstance(atr, pd.DataFrame):
        # Если вернулся DataFrame, берем первую колонку или колонку с именем ATR
        atr_col = None
        for col in atr.columns:
            if "ATR" in str(col).upper():
                atr_col = col
                break
        df["atr"] = atr[atr_col] if atr_col else atr.iloc[:, 0]
    else:
        # Fallback: пытаемся преобразовать в Series
        df["atr"] = pd.Series(atr, index=df.index) if hasattr(atr, '__iter__') else None
    
    # Bollinger Bands для флэтовой стратегии
    # pandas-ta использует lower_std и upper_std отдельно, формат имени: BBU_{length}_{lower_std}_{upper_std}
    # Bollinger Bands: try pandas_ta, but fallback to manual calculation if needed
    # Bollinger Bands: try pandas_ta, but fallback to manual calculation if needed
    bb = None
    try:
        bb = ta.bbands(df["close"], length=bb_length, lower_std=bb_std, upper_std=bb_std)
    except Exception:
        bb = None

    if isinstance(bb, pd.DataFrame) and len(bb.columns) >= 3:
        # Try to pick standard pandas_ta column names, otherwise take first three
        try:
            bb_col_suffix = f"_{bb_length}_{bb_std}_{bb_std}"
            upper_col = f"BBU{bb_col_suffix}"
            middle_col = f"BBM{bb_col_suffix}"
            lower_col = f"BBL{bb_col_suffix}"
            if upper_col in bb.columns and middle_col in bb.columns and lower_col in bb.columns:
                df["bb_upper"] = bb[upper_col]
                df["bb_middle"] = bb[middle_col]
                df["bb_lower"] = bb[lower_col]
            else:
                # Fallback: take first three columns
                df["bb_upper"] = bb.iloc[:, 0]
                df["bb_middle"] = bb.iloc[:, 1]
                df["bb_lower"] = bb.iloc[:, 2]
        except Exception:
            # As ultimate fallback, compute manually below
            bb = None

    if bb is None:
        mid = df["close"].rolling(window=bb_length, min_periods=1).mean()
        std = df["close"].rolling(window=bb_length, min_periods=1).std()
        df["bb_middle"] = mid
        df["bb_upper"] = mid + bb_std * std
        df["bb_lower"] = mid - bb_std * std

    # Ensure BB columns have no NaN by backfilling and using mid as fallback
    df["bb_middle"] = df["bb_middle"].bfill().fillna(df["close"].rolling(window=bb_length, min_periods=1).mean())
    df["bb_upper"] = df["bb_upper"].bfill().fillna(df["bb_middle"])
    df["bb_lower"] = df["bb_lower"].bfill().fillna(df["bb_middle"])

    # Bollinger Band width (relative) — used by flat/range strategies
    # Protect against division by zero by replacing zeros in middle with NaN
    mid_abs = df["bb_middle"].replace(0, np.nan).abs()
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / (mid_abs)).fillna(0.0)
    # alias used in some places
    df["bbw"] = df["bb_width"]
    
    # MACD для анализа тренда и моментума
    macd_result = None
    try:
        macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    except Exception:
        macd_result = None

    if isinstance(macd_result, pd.DataFrame):
        df["macd"] = macd_result.get("MACD_12_26_9", macd_result.iloc[:, 0] if len(macd_result.columns) > 0 else pd.Series(index=df.index))
        df["macd_signal"] = macd_result.get("MACDs_12_26_9", macd_result.iloc[:, 1] if len(macd_result.columns) > 1 else pd.Series(index=df.index))
        df["macd_hist"] = macd_result.get("MACDh_12_26_9", macd_result.iloc[:, 2] if len(macd_result.columns) > 2 else pd.Series(index=df.index))
    elif isinstance(macd_result, pd.Series):
        df["macd"] = macd_result
        df["macd_signal"] = pd.Series(index=df.index, dtype=float)
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)
    else:
        # fallback: create empty numeric columns
        df["macd"] = pd.Series(index=df.index, dtype=float)
        df["macd_signal"] = pd.Series(index=df.index, dtype=float)
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)
    
    return df


def compute_vwap(df: pd.DataFrame, anchor: str = "D") -> pd.DataFrame:
    """
    Вычисляет VWAP (Volume Weighted Average Price) с ежедневным сбросом.
    
    Args:
        df: DataFrame с OHLCV данными и DatetimeIndex
        anchor: Период сброса VWAP ("D" = ежедневно, "W" = еженедельно, "M" = ежемесячно)
    
    Returns:
        DataFrame с добавленной колонкой vwap
    """
    df = df.copy()
    
    # Проверяем, что индекс - DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for VWAP calculation")
    
    # Вычисляем VWAP с помощью pandas_ta
    # anchor="D" означает ежедневный сброс
    vwap_result = ta.vwap(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        anchor=anchor
    )
    
    if isinstance(vwap_result, pd.Series):
        df["vwap"] = vwap_result
    elif isinstance(vwap_result, pd.DataFrame):
        # Если вернулся DataFrame, берем первую колонку (обычно это VWAP_D)
        df["vwap"] = vwap_result.iloc[:, 0]
    else:
        df["vwap"] = pd.Series(index=df.index, dtype=float)
    
    return df


def compute_donchian_channels(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """
    Вычисляет Donchian Channels (каналы Дончиана).
    Показывают максимумы и минимумы за период.
    
    Args:
        df: DataFrame с OHLCV данными
        length: Период для вычисления каналов (по умолчанию 20)
    
    Returns:
        DataFrame с добавленными колонками donchian_upper, donchian_lower, donchian_middle
    """
    df = df.copy()
    
    # Donchian Channels: верхняя граница = максимум high за период, нижняя = минимум low за период
    df["donchian_upper"] = df["high"].rolling(window=length).max()
    df["donchian_lower"] = df["low"].rolling(window=length).min()
    df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2
    
    return df


def compute_ema_indicators(df: pd.DataFrame, ema_fast_length: int = 20, ema_slow_length: int = 50) -> pd.DataFrame:
    """
    Вычисляет EMA индикаторы для стратегии импульсного пробоя.
    EMA вычисляются на текущем таймфрейме (15m), но для трендовой стратегии
    рекомендуется использовать данные с более высокого таймфрейма (1h или 4h).
    
    Args:
        df: DataFrame с данными свечей
        ema_fast_length: Период быстрой EMA (по умолчанию 20)
        ema_slow_length: Период медленной EMA (по умолчанию 50)
    
    Returns:
        DataFrame с добавленными колонками ema_fast и ema_slow
    """
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=ema_fast_length)
    df["ema_slow"] = ta.ema(df["close"], length=ema_slow_length)
    return df


def compute_higher_timeframe_ema(
    df_15m: pd.DataFrame,
    timeframe: str = "1h",
    ema_fast_length: int = 20,
    ema_slow_length: int = 50,
) -> pd.DataFrame:
    """
    Вычисляет EMA на более высоком таймфрейме (1h или 4h) для стратегии импульсного пробоя.
    Затем маппит значения обратно на 15m индекс.
    
    Args:
        df_15m: DataFrame с 15m свечами
        timeframe: Таймфрейм для вычисления EMA ("1h" или "4h")
        ema_fast_length: Период быстрой EMA
        ema_slow_length: Период медленной EMA
    
    Returns:
        DataFrame с добавленными колонками ema_fast_htf и ema_slow_htf (higher timeframe)
    """
    df_15m = df_15m.copy()
    
    # Resample на более высокий таймфрейм
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_htf = df_15m.resample(timeframe).agg(ohlcv).dropna()
    
    # Вычисляем EMA на высоком таймфрейме
    df_htf["ema_fast"] = ta.ema(df_htf["close"], length=ema_fast_length)
    df_htf["ema_slow"] = ta.ema(df_htf["close"], length=ema_slow_length)
    
    # Map обратно на 15m индекс, используя forward fill
    mapped = df_htf.reindex(df_15m.index, method="ffill")
    df_15m[f"ema_fast_{timeframe}"] = mapped["ema_fast"]
    df_15m[f"ema_slow_{timeframe}"] = mapped["ema_slow"]
    
    return df_15m


def compute_support_resistance_levels(df: pd.DataFrame, lookback: int = 20, min_touches: int = 2) -> pd.DataFrame:
    """
    Вычисляет уровни поддержки и сопротивления на основе локальных максимумов и минимумов.
    Использует упрощенный подход: ищет локальные экстремумы и группирует близкие уровни.
    
    Args:
        df: DataFrame с OHLCV данными
        lookback: Период для поиска локальных экстремумов (по умолчанию 20)
        min_touches: Минимальное количество касаний для подтверждения уровня (по умолчанию 2)
    
    Returns:
        DataFrame с добавленными колонками:
        - nearest_support: Ближайший уровень поддержки снизу
        - nearest_resistance: Ближайший уровень сопротивления сверху
    """
    df = df.copy()
    
    # Упрощенный подход: используем recent_high и recent_low как базовые уровни
    # и дополняем их Donchian Channels и Bollinger Bands
    
    # Инициализируем колонки
    df["nearest_resistance"] = None
    df["nearest_support"] = None
    
    # Используем recent_high и recent_low как базовые уровни сопротивления/поддержки
    if "recent_high" in df.columns:
        df["nearest_resistance"] = df["recent_high"]
    if "recent_low" in df.columns:
        df["nearest_support"] = df["recent_low"]
    
    # Дополняем уровнями из Donchian Channels (более надежные)
    if "donchian_upper" in df.columns:
        # Используем Donchian верх как сопротивление, если он выше recent_high
        df["nearest_resistance"] = df[["nearest_resistance", "donchian_upper"]].max(axis=1, skipna=True)
        df["donchian_resistance"] = df["donchian_upper"]
    if "donchian_lower" in df.columns:
        # Используем Donchian низ как поддержку, если он ниже recent_low
        df["nearest_support"] = df[["nearest_support", "donchian_lower"]].min(axis=1, skipna=True)
        df["donchian_support"] = df["donchian_lower"]
    
    # Дополняем уровнями из Bollinger Bands (для флэта)
    if "bb_upper" in df.columns:
        df["bb_resistance"] = df["bb_upper"]
    if "bb_lower" in df.columns:
        df["bb_support"] = df["bb_lower"]
    
    return df


def prepare_with_indicators(
    df_raw: pd.DataFrame,
    adx_length: int,
    di_length: int,
    sma_length: int,
    rsi_length: int,
    breakout_lookback: int,
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
    ema_fast_length: int = 20,
    ema_slow_length: int = 50,
    ema_timeframe: str = "1h",
) -> pd.DataFrame:
    df = add_time_index(df_raw)
    df = compute_4h_context(df, adx_length=adx_length)  # ADX на 4H для фильтра тренда
    df = compute_1h_context(df, di_length=di_length)  # DI на 1H для направления
    df = compute_15m_features(
        df,
        sma_length=sma_length,
        rsi_length=rsi_length,
        breakout_lookback=breakout_lookback,
        bb_length=bb_length,
        bb_std=bb_std,
        atr_length=atr_length,
    )
    # Вычисляем ATR на 1H и 4H таймфреймах для анализа среднесрочной волатильности
    df = compute_atr_higher_timeframes(df, atr_length=atr_length)
    # Вычисляем EMA на более высоком таймфрейме для стратегии импульсного пробоя
    df = compute_higher_timeframe_ema(df, timeframe=ema_timeframe, ema_fast_length=ema_fast_length, ema_slow_length=ema_slow_length)
    # Вычисляем VWAP для стратегии институционального тренда (ежедневный сброс)
    df = compute_vwap(df, anchor="D")
    # Вычисляем Donchian Channels для стратегии снятия ликвидности
    df = compute_donchian_channels(df, length=20)
    # Вычисляем уровни поддержки и сопротивления
    df = compute_support_resistance_levels(df, lookback=breakout_lookback, min_touches=2)
    
    # Удаляем только строки, где все ключевые колонки NaN (более мягкая обработка)
    # Сохраняем строки, где хотя бы основные данные (OHLCV) присутствуют
    key_columns = ["open", "high", "low", "close", "volume"]
    if all(col in df.columns for col in key_columns):
        # Проверяем, что основные колонки не все NaN
        df = df[df[key_columns].notna().any(axis=1)]
    
    # Удаляем только строки, где все значения NaN (не удаляем строки с частичными NaN)
    # Это важно, так как индикаторы (rolling, shift) создают NaN в начале, но основные данные (OHLCV) есть
    df = df.dropna(how='all')
    
    # Если после обработки DataFrame пуст, это критическая ошибка
    if len(df) == 0:
        print(f"[indicators] ⚠️ Warning: All rows removed after processing, but this should not happen")
        # Попробуем вернуть исходные данные с минимальной обработкой
        df_fallback = add_time_index(df_raw)
        if all(col in df_fallback.columns for col in key_columns):
            df_fallback = df_fallback[df_fallback[key_columns].notna().any(axis=1)]
        if len(df_fallback) > 0:
            print(f"[indicators] ⚠️ Returning fallback data with {len(df_fallback)} rows")
            return df_fallback
    
    return df
