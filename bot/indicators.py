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
    df["sma"] = ta.sma(df["close"], length=sma_length)
    df["rsi"] = ta.rsi(df["close"], length=rsi_length)
    df["vol_sma"] = ta.sma(df["volume"], length=breakout_lookback)
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
    bb = ta.bbands(df["close"], length=bb_length, lower_std=bb_std, upper_std=bb_std)
    # Формируем правильное имя колонки
    bb_col_suffix = f"_{bb_length}_{bb_std}_{bb_std}"
    df["bb_upper"] = bb[f"BBU{bb_col_suffix}"]
    df["bb_middle"] = bb[f"BBM{bb_col_suffix}"]
    df["bb_lower"] = bb[f"BBL{bb_col_suffix}"]
    
    # MACD для анализа тренда и моментума
    macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if isinstance(macd_result, pd.DataFrame):
        df["macd"] = macd_result.get("MACD_12_26_9", macd_result.iloc[:, 0] if len(macd_result.columns) > 0 else pd.Series(index=df.index))
        df["macd_signal"] = macd_result.get("MACDs_12_26_9", macd_result.iloc[:, 1] if len(macd_result.columns) > 1 else pd.Series(index=df.index))
        df["macd_hist"] = macd_result.get("MACDh_12_26_9", macd_result.iloc[:, 2] if len(macd_result.columns) > 2 else pd.Series(index=df.index))
    elif isinstance(macd_result, pd.Series):
        # Если вернулась только одна серия (MACD line)
        df["macd"] = macd_result
        df["macd_signal"] = pd.Series(index=df.index, dtype=float)
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)
    
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
    df.dropna(inplace=True)
    return df

