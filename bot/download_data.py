import pandas as pd
import ccxt
import os
import time

def download_history(symbol='ETH/USDT', timeframe='15m', target_candles=50000):
    exchange = ccxt.binance()
    print(f"--- Запуск глубокой загрузки истории для {symbol} ---")
    
    # 1. Рассчитываем, сколько миллисекунд назад нам нужно отступить
    # 15 минут * 60 сек * 1000 мс * количество свечей
    ms_per_candle = 15 * 60 * 1000
    duration_ms = target_candles * ms_per_candle
    
    # Точка старта = Текущее время минус необходимая длительность
    start_time = exchange.milliseconds() - duration_ms
    
    all_ohlcv = []
    current_since = start_time

    while len(all_ohlcv) < target_candles:
        try:
            # Запрашиваем данные начиная с current_since
            new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not new_ohlcv:
                print("Данные закончились или достигнут предел биржи.")
                break
                
            all_ohlcv.extend(new_ohlcv)
            
            # Обновляем точку старта для следующего шага (время последней свечи + 1мс)
            current_since = new_ohlcv[-1][0] + 1
            
            print(f"Загружено: {len(all_ohlcv)} / {target_candles} свечей... (Дошли до: {pd.to_datetime(current_since, unit='ms')})")
            
            # Пауза, чтобы не получить бан от API
            time.sleep(exchange.rateLimit / 1000)
            
            # Если мы загрузили свечи, которые уже в "будущем" относительно запроса, выходим
            if current_since > exchange.milliseconds():
                break

        except Exception as e:
            print(f"Ошибка API: {e}")
            time.sleep(5) # Ждем дольше при ошибке
            continue

    # Формируем и сохраняем данные
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    os.makedirs('data', exist_ok=True)
    file_path = 'data/eth_15m.csv'
    df.to_csv(file_path, index=False)
    
    print(f"--- Загрузка завершена! ---")
    print(f"Итого сохранено: {len(df)} свечей")
    print(f"Диапазон: {df['datetime'].iloc[0]} >>> {df['datetime'].iloc[-1]}")

if __name__ == "__main__":
    download_history(target_candles=50000)