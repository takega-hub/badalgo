"""
Скрипт для проверки корректности технических индикаторов и условий стратегий.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy
from bot.live import _timeframe_to_bybit_interval


def verify_indicators():
    """Проверка корректности расчета индикаторов"""
    print("=" * 80)
    print("🔍 ПРОВЕРКА ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
    print("=" * 80)
    
    settings = load_settings()
    client = BybitClient(api=settings.api)
    
    print(f"\n📊 Загрузка данных для {settings.symbol}...")
    interval = _timeframe_to_bybit_interval(settings.timeframe)
    df_raw = client.get_kline_df(symbol=settings.symbol, interval=interval, limit=settings.kline_limit)
    
    print(f"✅ Получено {len(df_raw)} свечей")
    
    # Подготовка индикаторов
    print(f"\n📈 Расчет индикаторов...")
    df_ready = prepare_with_indicators(
        df_raw,
        adx_length=settings.strategy.adx_length,
        di_length=settings.strategy.di_length,
        sma_length=settings.strategy.sma_length,
        rsi_length=settings.strategy.rsi_length,
        breakout_lookback=settings.strategy.breakout_lookback,
        bb_length=settings.strategy.bb_length,
        bb_std=settings.strategy.bb_std,
        ema_fast_length=settings.strategy.ema_fast_length,
        ema_slow_length=settings.strategy.ema_slow_length,
    )
    df_ready = enrich_for_strategy(df_ready, settings.strategy)
    
    print(f"✅ Подготовлено {len(df_ready)} строк")
    
    # ПРОВЕРКА ИНДИКАТОРОВ НА ПОСЛЕДНИХ 10 СВЕЧАХ
    print(f"\n{'=' * 80}")
    print("📊 ИНДИКАТОРЫ НА ПОСЛЕДНИХ 10 СВЕЧАХ")
    print("=" * 80)
    
    for i in range(max(0, len(df_ready) - 10), len(df_ready)):
        row = df_ready.iloc[i]
        time_str = df_ready.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(df_ready.index[i], 'strftime') else str(df_ready.index[i])
        
        print(f"\n[{i - len(df_ready) + 10 + 1}/10] {time_str}")
        print(f"  💰 Price: ${row['close']:.2f} (Open: ${row['open']:.2f}, High: ${row['high']:.2f}, Low: ${row['low']:.2f})")
        print(f"  📊 Volume: {row['volume']:.0f}")
        
        # ADX и DI
        adx = row.get('adx', np.nan)
        plus_di = row.get('plus_di', np.nan)
        minus_di = row.get('minus_di', np.nan)
        print(f"  🔵 ADX: {adx:.2f} {'✅ (тренд)' if adx > 25 else '❌ (флэт)'} (+DI: {plus_di:.2f}, -DI: {minus_di:.2f})")
        
        # RSI
        rsi = row.get('rsi', np.nan)
        rsi_status = "перепроданность" if rsi < 30 else "перекупленность" if rsi > 70 else "норма"
        print(f"  📈 RSI: {rsi:.2f} ({rsi_status})")
        
        # Bollinger Bands
        bb_upper = row.get('bb_upper', np.nan)
        bb_lower = row.get('bb_lower', np.nan)
        bb_mid = row.get('bb_mid', np.nan)
        price = row['close']
        if pd.notna([bb_upper, bb_lower, bb_mid]).all():
            bb_position = "выше верхней" if price > bb_upper else "ниже нижней" if price < bb_lower else "в диапазоне"
            print(f"  📏 BB: ${bb_lower:.2f} - ${bb_mid:.2f} - ${bb_upper:.2f} (цена {bb_position})")
        
        # EMA
        ema_timeframe = settings.strategy.momentum_ema_timeframe
        ema_fast = row.get(f'ema_fast_{ema_timeframe}', np.nan)
        ema_slow = row.get(f'ema_slow_{ema_timeframe}', np.nan)
        if pd.notna([ema_fast, ema_slow]).all():
            ema_trend = "бычий" if ema_fast > ema_slow else "медвежий"
            ema_spread = abs(ema_fast - ema_slow) / ema_slow * 100 if ema_slow > 0 else 0
            print(f"  🎯 EMA ({ema_timeframe}): Fast ${ema_fast:.2f}, Slow ${ema_slow:.2f} ({ema_trend}, spread: {ema_spread:.2f}%)")
        
        # Volume
        vol_sma = row.get('vol_sma', np.nan)
        if pd.notna(vol_sma) and vol_sma > 0:
            vol_ratio = row['volume'] / vol_sma
            vol_status = "spike ✅" if vol_ratio > 1.5 else "норма"
            print(f"  📊 Volume Ratio: {vol_ratio:.2f}x ({vol_status})")
        
        # ATR
        atr = row.get('atr', np.nan)
        atr_1h = row.get('atr_1h', np.nan)
        atr_4h = row.get('atr_4h', np.nan)
        if pd.notna(atr):
            print(f"  📉 ATR: 15m=${atr:.2f}, 1h=${atr_1h:.2f}, 4h=${atr_4h:.2f}")
    
    # АНАЛИЗ УСЛОВИЙ ДЛЯ КАЖДОЙ СТРАТЕГИИ
    print(f"\n{'=' * 80}")
    print("🎯 АНАЛИЗ УСЛОВИЙ СТРАТЕГИЙ (последняя свеча)")
    print("=" * 80)
    
    last_row = df_ready.iloc[-1]
    
    # MOMENTUM СТРАТЕГИЯ
    print(f"\n⚡ MOMENTUM СТРАТЕГИЯ:")
    print(f"  Требования для LONG сигнала:")
    
    adx = last_row.get('adx', np.nan)
    adx_ok = adx > settings.strategy.momentum_adx_threshold
    print(f"    1. ADX > {settings.strategy.momentum_adx_threshold}: {adx:.2f} {'✅' if adx_ok else '❌'}")
    
    ema_timeframe = settings.strategy.momentum_ema_timeframe
    ema_fast = last_row.get(f'ema_fast_{ema_timeframe}', np.nan)
    ema_slow = last_row.get(f'ema_slow_{ema_timeframe}', np.nan)
    ema_bullish = ema_fast > ema_slow
    print(f"    2. EMA Fast > EMA Slow: ${ema_fast:.2f} > ${ema_slow:.2f} {'✅' if ema_bullish else '❌'}")
    
    volume = last_row.get('volume', np.nan)
    vol_sma = last_row.get('vol_sma', np.nan)
    vol_ok = (volume >= vol_sma * settings.strategy.momentum_volume_spike_min and 
              volume <= vol_sma * settings.strategy.momentum_volume_spike_max)
    vol_ratio = volume / vol_sma if vol_sma > 0 else 0
    print(f"    3. Volume spike {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x: {vol_ratio:.2f}x {'✅' if vol_ok else '❌'}")
    
    ema_cross_up = last_row.get('ema_cross_up', False)
    print(f"    4. EMA пересечение вверх: {'✅' if ema_cross_up else '❌'}")
    
    all_ok = adx_ok and ema_bullish and vol_ok and ema_cross_up
    print(f"  \n  РЕЗУЛЬТАТ: {'✅ Все условия выполнены!' if all_ok else '❌ Условия не выполнены'}")
    
    # FLAT СТРАТЕГИЯ
    print(f"\n📊 FLAT (RANGE) СТРАТЕГИЯ:")
    print(f"  Требования для LONG сигнала:")
    
    adx = last_row.get('adx', np.nan)
    adx_flat = adx <= settings.strategy.adx_threshold
    print(f"    1. ADX <= {settings.strategy.adx_threshold}: {adx:.2f} {'✅' if adx_flat else '❌'}")
    
    rsi = last_row.get('rsi', np.nan)
    rsi_oversold = rsi <= settings.strategy.range_rsi_oversold
    print(f"    2. RSI <= {settings.strategy.range_rsi_oversold}: {rsi:.2f} {'✅' if rsi_oversold else '❌'}")
    
    bb_lower = last_row.get('bb_lower', np.nan)
    price = last_row['close']
    touch_lower = price <= bb_lower
    print(f"    3. Price <= BB Lower: ${price:.2f} <= ${bb_lower:.2f} {'✅' if touch_lower else '❌'}")
    
    volume = last_row.get('volume', np.nan)
    vol_sma = last_row.get('vol_sma', np.nan)
    vol_low = volume < vol_sma * settings.strategy.range_volume_mult
    vol_ratio = volume / vol_sma if vol_sma > 0 else 0
    print(f"    4. Volume < {settings.strategy.range_volume_mult}x: {vol_ratio:.2f}x {'✅' if vol_low else '❌'}")
    
    all_ok = adx_flat and rsi_oversold and touch_lower and vol_low
    print(f"  \n  РЕЗУЛЬТАТ: {'✅ Все условия выполнены!' if all_ok else '❌ Условия не выполнены'}")
    
    # ПРОВЕРКА КОРРЕКТНОСТИ РАСЧЕТОВ
    print(f"\n{'=' * 80}")
    print("🔬 ПРОВЕРКА КОРРЕКТНОСТИ РАСЧЕТОВ")
    print("=" * 80)
    
    issues = []
    
    # Проверка ADX
    adx_values = df_ready['adx'].dropna()
    if len(adx_values) > 0:
        if adx_values.min() < 0 or adx_values.max() > 100:
            issues.append(f"⚠️ ADX имеет недопустимые значения: min={adx_values.min():.2f}, max={adx_values.max():.2f}")
        else:
            print(f"✅ ADX: диапазон {adx_values.min():.2f} - {adx_values.max():.2f} (норма)")
    
    # Проверка RSI
    rsi_values = df_ready['rsi'].dropna()
    if len(rsi_values) > 0:
        if rsi_values.min() < 0 or rsi_values.max() > 100:
            issues.append(f"⚠️ RSI имеет недопустимые значения: min={rsi_values.min():.2f}, max={rsi_values.max():.2f}")
        else:
            print(f"✅ RSI: диапазон {rsi_values.min():.2f} - {rsi_values.max():.2f} (норма)")
    
    # Проверка EMA
    ema_fast_col = f"ema_fast_{settings.strategy.momentum_ema_timeframe}"
    ema_slow_col = f"ema_slow_{settings.strategy.momentum_ema_timeframe}"
    if ema_fast_col in df_ready.columns and ema_slow_col in df_ready.columns:
        ema_fast_values = df_ready[ema_fast_col].dropna()
        ema_slow_values = df_ready[ema_slow_col].dropna()
        if len(ema_fast_values) > 0 and len(ema_slow_values) > 0:
            print(f"✅ EMA Fast: диапазон ${ema_fast_values.min():.2f} - ${ema_fast_values.max():.2f}")
            print(f"✅ EMA Slow: диапазон ${ema_slow_values.min():.2f} - ${ema_slow_values.max():.2f}")
    
    # Проверка Volume
    vol_values = df_ready['volume'].dropna()
    if len(vol_values) > 0:
        if vol_values.min() < 0:
            issues.append(f"⚠️ Volume имеет отрицательные значения: min={vol_values.min():.0f}")
        else:
            print(f"✅ Volume: диапазон {vol_values.min():.0f} - {vol_values.max():.0f}")
    
    # Итоги
    print(f"\n{'=' * 80}")
    if issues:
        print("❌ НАЙДЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ ВСЕ ИНДИКАТОРЫ РАССЧИТАНЫ КОРРЕКТНО")
    
    print("=" * 80)
    
    # РЕКОМЕНДАЦИИ
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    last_adx = df_ready.iloc[-1].get('adx', 0)
    if last_adx < 25:
        print(f"  • ADX слишком низкий ({last_adx:.2f} < 25) - рынок во флэте")
        print(f"    → Momentum стратегия не будет генерировать сигналы")
        print(f"    → Рекомендация: Включите FLAT стратегию или понизьте ADX порог до 20")
    
    last_rsi = df_ready.iloc[-1].get('rsi', 50)
    if last_rsi > 70:
        print(f"  • RSI перекупленность ({last_rsi:.2f} > 70)")
        print(f"    → FLAT стратегия может генерировать SHORT сигналы")
    elif last_rsi < 30:
        print(f"  • RSI перепроданность ({last_rsi:.2f} < 30)")
        print(f"    → FLAT стратегия может генерировать LONG сигналы")
    
    if not settings.enable_trend_strategy:
        print(f"  • TREND стратегия отключена")
        print(f"    → Рекомендация: Включите TREND стратегию для большего покрытия рыночных условий")


if __name__ == "__main__":
    verify_indicators()
