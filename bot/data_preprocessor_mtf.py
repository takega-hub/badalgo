"""
Утилита для подготовки мультитаймфреймовых данных
Загружает и синхронизирует данные с разных таймфреймов
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
import os


def load_mtf_data(base_path: str, symbol: str = "BTCUSDT") -> List[pd.DataFrame]:
    """
    Загружает данные с разных таймфреймов и синхронизирует их
    
    Args:
        base_path: Базовый путь к папке с данными
        symbol: Торговая пара (например, "BTCUSDT")
    
    Returns:
        Список датафреймов [df_15m, df_1h, df_4h]
    """
    dataframes = []
    timeframes = ['15m', '1h', '4h']
    
    for tf in timeframes:
        # Пробуем разные варианты имен файлов
        possible_paths = [
            os.path.join(base_path, f"{symbol.lower()}_{tf}.csv"),
            os.path.join(base_path, f"{symbol}_{tf}.csv"),
            os.path.join(base_path, f"btc_{tf}.csv"),  # Fallback для BTC
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"✅ Загружен {tf}: {len(df)} строк из {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки {path}: {e}")
        
        if df is None:
            print(f"⚠️ Не найден файл для {tf}, создаю пустой датафрейм")
            df = pd.DataFrame()
        
        dataframes.append(df)
    
    # Синхронизируем данные
    if len(dataframes) > 0 and len(dataframes[0]) > 0:
        dataframes = synchronize_mtf_data(dataframes)
    
    return dataframes


def synchronize_mtf_data(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Синхронизирует датафреймы по временным меткам
    
    Args:
        df_list: Список датафреймов [15m, 1h, 4h]
    
    Returns:
        Синхронизированные датафреймы
    """
    if len(df_list) == 0:
        return df_list
    
    df_15m = df_list[0].copy()
    
    # Определяем колонку с временем для основного датафрейма
    if 'timestamp' in df_15m.columns:
        time_col_15m = 'timestamp'
    elif isinstance(df_15m.index, pd.DatetimeIndex):
        df_15m['timestamp'] = df_15m.index
        time_col_15m = 'timestamp'
    else:
        print("⚠️ Не удалось определить временную колонку для 15m")
        return df_list
    
        # Преобразуем в datetime
        # Проверяем формат: если это число (Unix timestamp в мс), используем unit='ms'
        if df_15m[time_col_15m].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Проверяем, это миллисекунды или секунды
            first_val = df_15m[time_col_15m].iloc[0]
            if first_val > 1e12:  # Если больше 1e12, это миллисекунды
                df_15m[time_col_15m] = pd.to_datetime(df_15m[time_col_15m], unit='ms')
            else:  # Иначе секунды
                df_15m[time_col_15m] = pd.to_datetime(df_15m[time_col_15m], unit='s')
        else:
            df_15m[time_col_15m] = pd.to_datetime(df_15m[time_col_15m])
    
    # Синхронизируем остальные таймфреймы
    synchronized = [df_15m]
    
    for i, df_tf in enumerate(df_list[1:], 1):
        if df_tf is None or len(df_tf) == 0:
            synchronized.append(pd.DataFrame())
            continue
        
        df_tf = df_tf.copy()
        
        # Определяем колонку с временем
        if 'timestamp' in df_tf.columns:
            time_col_tf = 'timestamp'
        elif isinstance(df_tf.index, pd.DatetimeIndex):
            df_tf['timestamp'] = df_tf.index
            time_col_tf = 'timestamp'
        else:
            print(f"⚠️ Не удалось определить временную колонку для ТФ {i}")
            synchronized.append(pd.DataFrame())
            continue
        
        # Преобразуем в datetime
        # Проверяем формат: если это число (Unix timestamp в мс), используем unit='ms'
        if df_tf[time_col_tf].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Проверяем, это миллисекунды или секунды
            first_val = df_tf[time_col_tf].iloc[0] if len(df_tf) > 0 else 0
            if first_val > 1e12:  # Если больше 1e12, это миллисекунды
                df_tf[time_col_tf] = pd.to_datetime(df_tf[time_col_tf], unit='ms')
            else:  # Иначе секунды
                df_tf[time_col_tf] = pd.to_datetime(df_tf[time_col_tf], unit='s')
        else:
            df_tf[time_col_tf] = pd.to_datetime(df_tf[time_col_tf])
        
        # Оставляем только данные в диапазоне основного датафрейма
        start_time = df_15m[time_col_15m].iloc[0]
        end_time = df_15m[time_col_15m].iloc[-1]
        
        mask = (df_tf[time_col_tf] >= start_time) & (df_tf[time_col_tf] <= end_time)
        df_tf_filtered = df_tf[mask].copy()
        
        if len(df_tf_filtered) > 0:
            print(f"✅ Синхронизирован ТФ {i}: {len(df_tf_filtered)} строк (из {len(df_tf)} исходных)")
            synchronized.append(df_tf_filtered)
        else:
            print(f"⚠️ Нет перекрывающихся данных для ТФ {i}")
            synchronized.append(pd.DataFrame())
    
    return synchronized


def resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Ресемплирует данные в целевой таймфрейм
    
    Args:
        df: Исходный датафрейм
        target_tf: Целевой таймфрейм ('1h', '4h', '1d')
    
    Returns:
        Ресемплированный датафрейм
    """
    if len(df) == 0:
        return df
    
    df_resampled = df.copy()
    
    # Определяем временную колонку
    if 'timestamp' in df_resampled.columns:
        time_col = 'timestamp'
        # Проверяем формат timestamp
        if df_resampled[time_col].dtype in ['int64', 'float64', 'int32', 'float32']:
            first_val = df_resampled[time_col].iloc[0] if len(df_resampled) > 0 else 0
            if first_val > 1e12:  # Миллисекунды
                df_resampled[time_col] = pd.to_datetime(df_resampled[time_col], unit='ms')
            else:  # Секунды
                df_resampled[time_col] = pd.to_datetime(df_resampled[time_col], unit='s')
        else:
            df_resampled[time_col] = pd.to_datetime(df_resampled[time_col])
        df_resampled = df_resampled.set_index(time_col)
    elif isinstance(df_resampled.index, pd.DatetimeIndex):
        pass  # Уже DatetimeIndex
    else:
        print("⚠️ Не удалось определить временную колонку для ресемплинга")
        return df
    
    # Ресемплируем
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Добавляем другие колонки если есть
    for col in df_resampled.columns:
        if col not in ohlc_dict:
            ohlc_dict[col] = 'last'  # Берем последнее значение
    
    df_resampled = df_resampled.resample(target_tf).agg(ohlc_dict).dropna()
    
    # Сбрасываем индекс в колонку timestamp
    df_resampled = df_resampled.reset_index()
    df_resampled.rename(columns={df_resampled.columns[0]: 'timestamp'}, inplace=True)
    
    return df_resampled


def calculate_mtf_indicators(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Рассчитывает индикаторы для всех таймфреймов
    
    Args:
        df_list: Список датафреймов [15m, 1h, 4h]
    
    Returns:
        Список датафреймов с рассчитанными индикаторами
    """
    from bot.indicators import prepare_with_indicators
    
    result = []
    
    for i, df in enumerate(df_list):
        if df is None or len(df) == 0:
            result.append(pd.DataFrame())
            continue
        
        try:
            # Рассчитываем индикаторы
            df_with_indicators = prepare_with_indicators(df.copy())
            
            # Добавляем суффикс к именам колонок для различения ТФ (кроме основного)
            if i > 0:
                suffix = f"_{['1h', '4h', '1d'][i-1]}"
                # Переименовываем только индикаторы, не базовые колонки
                indicator_cols = ['adx', 'plus_di', 'minus_di', 'rsi', 'rsi_norm', 'atr', 
                                 'volatility_ratio', 'volume_ratio', 'trend_bias_1h']
                for col in indicator_cols:
                    if col in df_with_indicators.columns:
                        df_with_indicators.rename(columns={col: col + suffix}, inplace=True)
            
            result.append(df_with_indicators)
            print(f"✅ Рассчитаны индикаторы для ТФ {i}")
            
        except Exception as e:
            print(f"⚠️ Ошибка расчета индикаторов для ТФ {i}: {e}")
            result.append(df)
    
    return result


def create_mtf_dataset(base_path: str, symbol: str = "BTCUSDT", 
                       output_path: Optional[str] = None) -> List[pd.DataFrame]:
    """
    Создает полный MTF датасет с синхронизированными данными и индикаторами
    
    Args:
        base_path: Путь к папке с данными
        symbol: Торговая пара
        output_path: Опциональный путь для сохранения результата
    
    Returns:
        Список готовых датафреймов [15m, 1h, 4h]
    """
    print(f"\n{'='*60}")
    print(f"📊 ПОДГОТОВКА MTF ДАННЫХ ДЛЯ {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Загружаем данные
    print("📥 Загрузка данных...")
    df_list = load_mtf_data(base_path, symbol)
    
    # 2. Если нет данных для старших ТФ, ресемплируем из 15m
    if len(df_list) > 0 and len(df_list[0]) > 0:
        df_15m = df_list[0]
        
        if len(df_list) > 1 and (df_list[1] is None or len(df_list[1]) == 0):
            print("📊 Ресемплирование 15m → 1h...")
            df_list[1] = resample_to_timeframe(df_15m, '1h')
        
        if len(df_list) > 2 and (df_list[2] is None or len(df_list[2]) == 0):
            print("📊 Ресемплирование 15m → 4h...")
            df_list[2] = resample_to_timeframe(df_15m, '4h')
    
    # 3. Рассчитываем индикаторы
    print("\n📈 Расчет индикаторов...")
    df_list = calculate_mtf_indicators(df_list)
    
    # 4. Сохраняем результат (опционально)
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        for i, df in enumerate(df_list):
            if df is not None and len(df) > 0:
                tf_name = ['15m', '1h', '4h'][i]
                output_file = os.path.join(output_path, f"{symbol.lower()}_{tf_name}_mtf.csv")
                df.to_csv(output_file, index=False)
                print(f"💾 Сохранен {tf_name}: {output_file}")
    
    print(f"\n✅ MTF датасет готов!")
    print(f"   15m: {len(df_list[0])} строк" if len(df_list) > 0 else "   15m: нет данных")
    print(f"   1h:  {len(df_list[1])} строк" if len(df_list) > 1 and df_list[1] is not None else "   1h: нет данных")
    print(f"   4h:  {len(df_list[2])} строк" if len(df_list) > 2 and df_list[2] is not None else "   4h: нет данных")
    
    return df_list


if __name__ == "__main__":
    # Пример использования
    base_path = "./data"
    symbol = "BTCUSDT"
    
    df_list = create_mtf_dataset(base_path, symbol, output_path="./data/mtf")
    
    print(f"\n📋 Пример данных 15m (первые 5 строк):")
    if len(df_list) > 0 and len(df_list[0]) > 0:
        print(df_list[0][['timestamp', 'close', 'atr', 'adx', 'rsi']].head())
