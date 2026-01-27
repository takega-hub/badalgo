"""
Скрипт для загрузки исторических данных с биржи Binance
Поддерживает несколько символов и таймфреймов для MTF анализа
"""

import pandas as pd
import ccxt
import os
import time
from typing import List, Optional, Dict
from datetime import datetime


def timeframe_to_ms(timeframe: str) -> int:
    """
    Преобразует таймфрейм в миллисекунды на одну свечу
    
    Args:
        timeframe: Таймфрейм ('15m', '1h', '4h', '1d')
    
    Returns:
        Миллисекунды на одну свечу
    """
    timeframe_map = {
        '1m': 1 * 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    
    return timeframe_map.get(timeframe, 15 * 60 * 1000)


def get_target_candles_for_timeframe(timeframe: str, days: int = 365) -> int:
    """
    Рассчитывает целевое количество свечей для таймфрейма
    
    Args:
        timeframe: Таймфрейм
        days: Количество дней истории
    
    Returns:
        Целевое количество свечей
    """
    candles_per_day = {
        '15m': 96,   # 24 часа * 4 свечи в час
        '1h': 24,    # 24 свечи в день
        '4h': 6,     # 6 свечей в день
        '1d': 1,     # 1 свеча в день
    }
    
    candles_per_day_value = candles_per_day.get(timeframe, 96)
    return candles_per_day_value * days


def download_history(
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    target_candles: Optional[int] = None,
    days: int = 365,
    output_dir: str = 'data'
) -> pd.DataFrame:
    """
    Загружает исторические данные с биржи Binance
    
    Args:
        symbol: Торговая пара (например, 'BTC/USDT', 'ETH/USDT', 'SOL/USDT')
        timeframe: Таймфрейм ('15m', '1h', '4h', '1d')
        target_candles: Целевое количество свечей (если None, рассчитывается из days)
        days: Количество дней истории (используется если target_candles не указан)
        output_dir: Директория для сохранения файлов
    
    Returns:
        DataFrame с историческими данными
    """
    exchange = ccxt.binance()
    
    # Определяем целевое количество свечей
    if target_candles is None:
        target_candles = get_target_candles_for_timeframe(timeframe, days)
    
    print(f"\n{'='*60}")
    print(f"📥 Загрузка данных: {symbol} | {timeframe}")
    print(f"   Целевое количество свечей: {target_candles:,}")
    print(f"{'='*60}")
    
    # Рассчитываем миллисекунды на одну свечу
    ms_per_candle = timeframe_to_ms(timeframe)
    duration_ms = target_candles * ms_per_candle
    
    # Точка старта = Текущее время минус необходимая длительность
    start_time = exchange.milliseconds() - duration_ms
    
    all_ohlcv = []
    current_since = start_time
    max_retries = 3
    retry_count = 0

    while len(all_ohlcv) < target_candles:
        try:
            # Запрашиваем данные начиная с current_since
            new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not new_ohlcv:
                print("⚠️ Данные закончились или достигнут предел биржи.")
                break
            
            all_ohlcv.extend(new_ohlcv)
            
            # Обновляем точку старта для следующего шага (время последней свечи + 1мс)
            current_since = new_ohlcv[-1][0] + 1
            
            progress = (len(all_ohlcv) / target_candles) * 100
            current_date = pd.to_datetime(current_since, unit='ms')
            print(f"   Загружено: {len(all_ohlcv):,} / {target_candles:,} свечей ({progress:.1f}%) | До: {current_date}")
            
            # Пауза, чтобы не получить бан от API
            time.sleep(exchange.rateLimit / 1000)
            
            # Если мы загрузили свечи, которые уже в "будущем" относительно запроса, выходим
            if current_since > exchange.milliseconds():
                print("   ✅ Достигнуто текущее время")
                break
            
            retry_count = 0  # Сбрасываем счетчик повторов при успехе
            
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"❌ Превышено количество попыток. Ошибка: {e}")
                break
            print(f"⚠️ Ошибка сети (попытка {retry_count}/{max_retries}): {e}")
            time.sleep(10)  # Ждем дольше при ошибке сети
            continue
            
        except ccxt.ExchangeError as e:
            print(f"❌ Ошибка биржи: {e}")
            break
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"❌ Неожиданная ошибка: {e}")
                break
            print(f"⚠️ Ошибка (попытка {retry_count}/{max_retries}): {e}")
            time.sleep(5)
            continue

    if not all_ohlcv:
        print(f"❌ Не удалось загрузить данные для {symbol} {timeframe}")
        return pd.DataFrame()

    # Формируем DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Сохраняем файл
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем имя файла: btcusdt_15m.csv, ethusdt_1h.csv и т.д.
    symbol_clean = symbol.replace('/', '').lower()
    filename = f"{symbol_clean}_{timeframe}.csv"
    file_path = os.path.join(output_dir, filename)
    
    df.to_csv(file_path, index=False)
    
    print(f"\n✅ Загрузка завершена!")
    print(f"   Файл: {file_path}")
    print(f"   Свечей: {len(df):,}")
    if len(df) > 0:
        print(f"   Диапазон: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    
    return df


def download_mtf_data(
    symbols: List[str] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    timeframes: List[str] = ['15m', '1h', '4h'],
    days: int = 365,
    output_dir: str = 'data'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Загружает данные для всех комбинаций символов и таймфреймов
    
    Args:
        symbols: Список торговых пар
        timeframes: Список таймфреймов
        days: Количество дней истории
        output_dir: Директория для сохранения файлов
    
    Returns:
        Словарь вида {symbol: {timeframe: DataFrame}}
    """
    print(f"\n{'='*60}")
    print(f"🚀 МАССОВАЯ ЗАГРУЗКА MTF ДАННЫХ")
    print(f"{'='*60}")
    print(f"Символы: {', '.join(symbols)}")
    print(f"Таймфреймы: {', '.join(timeframes)}")
    print(f"Дней истории: {days}")
    print(f"{'='*60}\n")
    
    results = {}
    total_tasks = len(symbols) * len(timeframes)
    current_task = 0
    
    for symbol in symbols:
        results[symbol] = {}
        
        for timeframe in timeframes:
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] Загрузка {symbol} {timeframe}...")
            
            try:
                df = download_history(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    output_dir=output_dir
                )
                results[symbol][timeframe] = df
                
                # Пауза между загрузками для избежания rate limit
                if current_task < total_tasks:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Ошибка при загрузке {symbol} {timeframe}: {e}")
                results[symbol][timeframe] = pd.DataFrame()
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        for timeframe in timeframes:
            df = results[symbol].get(timeframe, pd.DataFrame())
            if len(df) > 0:
                print(f"  ✅ {timeframe}: {len(df):,} свечей")
            else:
                print(f"  ❌ {timeframe}: не загружено")
    
    return results


def download_single_symbol_mtf(
    symbol: str = 'BTC/USDT',
    days: int = 365,
    output_dir: str = 'data'
) -> Dict[str, pd.DataFrame]:
    """
    Загружает все необходимые таймфреймы для одного символа
    
    Args:
        symbol: Торговая пара
        days: Количество дней истории
        output_dir: Директория для сохранения файлов
    
    Returns:
        Словарь {timeframe: DataFrame}
    """
    timeframes = ['15m', '1h', '4h']
    results = {}
    
    print(f"\n📥 Загрузка MTF данных для {symbol}")
    
    for timeframe in timeframes:
        df = download_history(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            output_dir=output_dir
        )
        results[timeframe] = df
        
        # Пауза между загрузками
        if timeframe != timeframes[-1]:
            time.sleep(1)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка исторических данных с Binance')
    parser.add_argument('--symbol', type=str, default=None, 
                       help='Торговая пара (например, BTC/USDT). Если не указан, загружаются все.')
    parser.add_argument('--timeframe', type=str, default=None,
                       choices=['15m', '1h', '4h', '1d'],
                       help='Таймфрейм. Если не указан, загружаются все для MTF.')
    parser.add_argument('--days', type=int, default=365,
                       help='Количество дней истории (по умолчанию 365)')
    parser.add_argument('--all', action='store_true',
                       help='Загрузить все символы и таймфреймы (BTC, ETH, SOL)')
    parser.add_argument('--output', type=str, default='data',
                       help='Директория для сохранения файлов')
    
    args = parser.parse_args()
    
    if args.all:
        # Загружаем все символы и таймфреймы
        download_mtf_data(
            symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            timeframes=['15m', '1h', '4h'],
            days=args.days,
            output_dir=args.output
        )
    elif args.symbol:
        if args.timeframe:
            # Загружаем конкретную комбинацию
            download_history(
                symbol=args.symbol,
                timeframe=args.timeframe,
                days=args.days,
                output_dir=args.output
            )
        else:
            # Загружаем все таймфреймы для символа
            download_single_symbol_mtf(
                symbol=args.symbol,
                days=args.days,
                output_dir=args.output
            )
    else:
        # По умолчанию загружаем BTC/USDT со всеми таймфреймами
        print("📥 Загрузка данных по умолчанию: BTC/USDT (15m, 1h, 4h)")
        download_single_symbol_mtf(
            symbol='BTC/USDT',
            days=args.days,
            output_dir=args.output
        )
