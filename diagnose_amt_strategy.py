"""
Диагностический скрипт для проверки работы стратегии AMT & Order Flow Scalper.
Проверяет получение данных, расчет показателей и генерацию сигналов.
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import AppSettings, load_settings
from bot.exchange.bybit_client import BybitClient
from bot.amt_orderflow_strategy import (
    generate_amt_signals,
    _parse_trades,
    _compute_cvd_metrics,
    build_volume_profile_from_ohlcv,
    VolumeProfileConfig,
    AbsorptionConfig,
    _resolve_symbol_settings,
    resolve_final_amt_configs,
)


def diagnose_amt_strategy(symbol: str = "BTCUSDT"):
    """Полная диагностика стратегии AMT для указанного символа."""
    print(f"\n{'='*80}")
    print(f"🔍 ДИАГНОСТИКА AMT & ORDER FLOW STRATEGY для {symbol}")
    print(f"{'='*80}\n")

    # Загружаем настройки
    try:
        settings = load_settings()
        print(f"✅ Настройки загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки настроек: {e}")
        return

    # Инициализируем клиент
    try:
        client = BybitClient(settings.api)
        print(f"✅ Bybit клиент инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации клиента: {e}")
        return

    # 1. Проверка получения OHLCV данных
    print(f"\n{'─'*80}")
    print("1️⃣ ПРОВЕРКА OHLCV ДАННЫХ")
    print(f"{'─'*80}")
    try:
        df_ohlcv = client.get_kline_df(symbol=symbol, interval="15", limit=200)
        print(f"✅ Получено {len(df_ohlcv)} свечей")
        print(f"   Временной диапазон: {pd.to_datetime(df_ohlcv['timestamp'].min(), unit='ms')} - {pd.to_datetime(df_ohlcv['timestamp'].max(), unit='ms')}")
        print(f"   Последняя цена: {df_ohlcv['close'].iloc[-1]:.4f}")
        print(f"   Объем последней свечи: {df_ohlcv['volume'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ Ошибка получения OHLCV: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Проверка получения тиков (trades)
    print(f"\n{'─'*80}")
    print("2️⃣ ПРОВЕРКА ПОЛУЧЕНИЯ ТИКОВ (TRADES)")
    print(f"{'─'*80}")
    try:
        raw_trades = client.get_recent_trades(symbol=symbol, limit=2000)  # Увеличено для покрытия 300-секундного окна
        print(f"✅ Получено {len(raw_trades)} сырых тиков от API")
        
        if raw_trades:
            print(f"   Пример первого тика: {raw_trades[0]}")
            if len(raw_trades) > 1:
                print(f"   Пример последнего тика: {raw_trades[-1]}")
        
        # Парсинг тиков
        trades_df = _parse_trades(raw_trades)
        print(f"✅ После парсинга: {len(trades_df)} валидных тиков")
        
        if not trades_df.empty:
            print(f"   Временной диапазон: {trades_df['time'].min()} - {trades_df['time'].max()}")
            print(f"   Возраст самого старого тика: {(datetime.now(timezone.utc) - trades_df['time'].min()).total_seconds():.1f} сек")
            print(f"   Возраст самого нового тика: {(datetime.now(timezone.utc) - trades_df['time'].max()).total_seconds():.1f} сек")
            
            # Статистика по сторонам
            buy_count = len(trades_df[trades_df['side'].str.upper() == 'BUY'])
            sell_count = len(trades_df[trades_df['side'].str.upper() == 'SELL'])
            buy_volume = trades_df[trades_df['side'].str.upper() == 'BUY']['qty'].sum()
            sell_volume = trades_df[trades_df['side'].str.upper() == 'SELL']['qty'].sum()
            print(f"   Покупки: {buy_count} тиков, объем: {buy_volume:.2f}")
            print(f"   Продажи: {sell_count} тиков, объем: {sell_volume:.2f}")
            print(f"   CVD (Buy - Sell): {buy_volume - sell_volume:.2f}")
        else:
            print(f"⚠️ Нет валидных тиков после парсинга!")
            
    except Exception as e:
        print(f"❌ Ошибка получения тиков: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Проверка Volume Profile
    print(f"\n{'─'*80}")
    print("3️⃣ ПРОВЕРКА VOLUME PROFILE")
    print(f"{'─'*80}")
    try:
        symbol_settings = _resolve_symbol_settings(symbol)
        vp_config = symbol_settings.volume_profile
        print(f"✅ Конфигурация Volume Profile:")
        print(f"   price_step: {vp_config.price_step}")
        print(f"   value_area_pct: {vp_config.value_area_pct}")
        
        # Подготовка данных для Volume Profile
        df_vp = df_ohlcv.copy()
        if "timestamp" in df_vp.columns:
            df_vp["timestamp"] = pd.to_datetime(df_vp["timestamp"], unit="ms", utc=True)
            df_vp = df_vp.set_index("timestamp")
        
        vp = build_volume_profile_from_ohlcv(df_vp, vp_config)
        if vp:
            print(f"✅ Volume Profile построен:")
            print(f"   POC: {vp['poc']:.4f}")
            print(f"   VAH: {vp['vah']:.4f}")
            print(f"   VAL: {vp['val']:.4f}")
            print(f"   Total volume: {vp['total_volume']:.2f}")
            print(f"   Price range: {vp['prices'].min():.4f} - {vp['prices'].max():.4f}")
        else:
            print(f"⚠️ Volume Profile не построен (нет данных)")
            
    except Exception as e:
        print(f"❌ Ошибка построения Volume Profile: {e}")
        import traceback
        traceback.print_exc()

    # 4. Проверка CVD метрик
    print(f"\n{'─'*80}")
    print("4️⃣ ПРОВЕРКА CVD МЕТРИК")
    print(f"{'─'*80}")
    try:
        symbol_settings = _resolve_symbol_settings(symbol)
        abs_config = symbol_settings.absorption
        lookback_seconds = abs_config.lookback_seconds
        
        print(f"✅ Конфигурация Absorption:")
        print(f"   lookback_seconds: {lookback_seconds}")
        print(f"   min_total_volume: {abs_config.min_total_volume:.2f}")
        print(f"   min_cvd_delta: {abs_config.min_cvd_delta:.2f}")
        print(f"   min_buy_sell_ratio: {abs_config.min_buy_sell_ratio:.2f}")
        print(f"   max_price_drift_pct: {abs_config.max_price_drift_pct:.2f}%")
        
        if not trades_df.empty:
            cvd_metrics = _compute_cvd_metrics(trades_df, lookback_seconds=lookback_seconds)
            if cvd_metrics:
                print(f"✅ CVD метрики вычислены:")
                print(f"   cvd_now: {cvd_metrics['cvd_now']:.2f}")
                print(f"   delta_velocity: {cvd_metrics['delta_velocity']:.2f}")
                print(f"   avg_abs_delta: {cvd_metrics['avg_abs_delta']:.2f}")
            else:
                print(f"⚠️ CVD метрики не вычислены")
        else:
            print(f"⚠️ Нет данных тиков для вычисления CVD")
            
    except Exception as e:
        print(f"❌ Ошибка вычисления CVD метрик: {e}")
        import traceback
        traceback.print_exc()

    # 5. Генерация сигналов
    print(f"\n{'─'*80}")
    print("5️⃣ ГЕНЕРАЦИЯ СИГНАЛОВ")
    print(f"{'─'*80}")
    try:
        current_price = float(df_ohlcv["close"].iloc[-1])
        print(f"Текущая цена: {current_price:.4f}")
        
        # Используем интеллектуальное разрешение конфигов: для символов из реестра
        # используются их индивидуальные настройки (min_vol, min_cvd), а не глобальные
        # Применяем адаптивные пороги объема в зависимости от времени суток
        current_time_utc = datetime.now(timezone.utc)
        vp_cfg, abs_cfg = resolve_final_amt_configs(symbol, settings.strategy, current_time_utc=current_time_utc, use_adaptive_volume=True)
        
        print(f"\n📊 Финальные настройки:")
        print(f"   Absorption: lookback={abs_cfg.lookback_seconds}s, min_vol={abs_cfg.min_total_volume:.0f}, min_cvd={abs_cfg.min_cvd_delta:.0f}, min_ratio={abs_cfg.min_buy_sell_ratio:.2f}")
        print(f"   Volume Profile: value_area_pct={vp_cfg.value_area_pct:.2f}, price_step={vp_cfg.price_step:.2f}")
        print(f"   Delta aggr mult: {settings.strategy.amt_of_delta_aggr_mult}")
        
        signals = generate_amt_signals(
            client=client,
            symbol=symbol,
            current_price=current_price,
            df_ohlcv=df_ohlcv,
            vp_config=vp_cfg,
            abs_config=abs_cfg,
            delta_aggr_mult=settings.strategy.amt_of_delta_aggr_mult,
        )
        
        print(f"\n{'─'*80}")
        if signals:
            print(f"✅ СГЕНЕРИРОВАНО {len(signals)} СИГНАЛОВ:")
            for i, signal in enumerate(signals, 1):
                print(f"\n   Сигнал #{i}:")
                print(f"      Действие: {signal.action.value}")
                print(f"      Цена: {signal.price:.4f}")
                print(f"      Причина: {signal.reason}")
                print(f"      Время: {signal.timestamp}")
        else:
            print(f"⚠️ СИГНАЛОВ НЕ СГЕНЕРИРОВАНО")
            print(f"\n   Возможные причины:")
            print(f"   - Условия для сигналов не выполнены")
            print(f"   - Недостаточно данных (тики, объем)")
            print(f"   - Пороги слишком высокие")
            print(f"   - Проверьте логи выше для деталей")
        print(f"{'─'*80}\n")
        
    except Exception as e:
        print(f"❌ Ошибка генерации сигналов: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'='*80}")
    print(f"✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    diagnose_amt_strategy(symbol)
