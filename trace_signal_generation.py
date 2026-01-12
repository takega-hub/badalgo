"""
Трассировка генерации сигналов - показывает пошагово, почему сигналы генерируются или нет.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, generate_signal, Action, Bias
from bot.live import _timeframe_to_bybit_interval


def trace_last_candles(num_candles: int = 20):
    """Трассировка генерации сигналов для последних N свечей"""
    print("=" * 80)
    print(f"🔍 ТРАССИРОВКА ГЕНЕРАЦИИ СИГНАЛОВ (последние {num_candles} свечей)")
    print("=" * 80)
    
    settings = load_settings()
    client = BybitClient(api=settings.api)
    
    print(f"\n📊 Загрузка данных для {settings.symbol}...")
    interval = _timeframe_to_bybit_interval(settings.timeframe)
    df_raw = client.get_kline_df(symbol=settings.symbol, interval=interval, limit=settings.kline_limit)
    
    # Подготовка индикаторов
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
    
    print(f"✅ Данные готовы: {len(df_ready)} свечей")
    
    # Трассировка для MOMENTUM стратегии
    if settings.enable_momentum_strategy:
        print(f"\n{'=' * 80}")
        print("⚡ ТРАССИРОВКА MOMENTUM СТРАТЕГИИ")
        print("=" * 80)
        
        ema_timeframe = settings.strategy.momentum_ema_timeframe
        ema_fast_col = f"ema_fast_{ema_timeframe}"
        ema_slow_col = f"ema_slow_{ema_timeframe}"
        
        print(f"\nПараметры:")
        print(f"  • ADX порог: {settings.strategy.momentum_adx_threshold}")
        print(f"  • Volume spike: {settings.strategy.momentum_volume_spike_min}x - {settings.strategy.momentum_volume_spike_max}x")
        print(f"  • EMA таймфрейм: {ema_timeframe}")
        
        crossovers = []
        position_bias = None
        
        # Анализируем последние N свечей
        start_idx = max(1, len(df_ready) - num_candles)
        
        for i in range(start_idx, len(df_ready)):
            row = df_ready.iloc[i]
            prev_row = df_ready.iloc[i-1] if i > 0 else None
            
            time_str = df_ready.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(df_ready.index[i], 'strftime') else str(df_ready.index[i])
            
            # Проверяем пересечение EMA
            if prev_row is not None:
                ema_fast = row.get(ema_fast_col, np.nan)
                ema_slow = row.get(ema_slow_col, np.nan)
                prev_ema_fast = prev_row.get(ema_fast_col, np.nan)
                prev_ema_slow = prev_row.get(ema_slow_col, np.nan)
                
                if pd.notna([ema_fast, ema_slow, prev_ema_fast, prev_ema_slow]).all():
                    # Пересечение вверх
                    if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
                        adx = row.get('adx', np.nan)
                        volume = row.get('volume', np.nan)
                        vol_sma = row.get('vol_sma', np.nan)
                        
                        adx_ok = pd.notna(adx) and adx > settings.strategy.momentum_adx_threshold
                        vol_ok = (pd.notna([volume, vol_sma]).all() and 
                                 volume >= vol_sma * settings.strategy.momentum_volume_spike_min and
                                 volume <= vol_sma * settings.strategy.momentum_volume_spike_max)
                        
                        vol_ratio = volume / vol_sma if vol_sma > 0 else 0
                        
                        print(f"\n🔼 [{time_str}] ПЕРЕСЕЧЕНИЕ EMA ВВЕРХ:")
                        print(f"   Price: ${row['close']:.2f}")
                        print(f"   ADX: {adx:.2f} {'✅' if adx_ok else f'❌ (нужно > {settings.strategy.momentum_adx_threshold})'}")
                        print(f"   Volume: {vol_ratio:.2f}x {'✅' if vol_ok else f'❌ (нужно {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x)'}")
                        
                        if adx_ok and vol_ok:
                            print(f"   ✅ СИГНАЛ LONG должен быть сгенерирован!")
                            crossovers.append({
                                'time': time_str,
                                'type': 'LONG',
                                'generated': True
                            })
                        else:
                            print(f"   ❌ Сигнал НЕ сгенерирован (условия не выполнены)")
                            crossovers.append({
                                'time': time_str,
                                'type': 'LONG',
                                'generated': False,
                                'reason': f"ADX={adx:.2f}, Vol={vol_ratio:.2f}x"
                            })
                    
                    # Пересечение вниз
                    elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
                        adx = row.get('adx', np.nan)
                        volume = row.get('volume', np.nan)
                        vol_sma = row.get('vol_sma', np.nan)
                        
                        adx_ok = pd.notna(adx) and adx > settings.strategy.momentum_adx_threshold
                        vol_ok = (pd.notna([volume, vol_sma]).all() and 
                                 volume >= vol_sma * settings.strategy.momentum_volume_spike_min and
                                 volume <= vol_sma * settings.strategy.momentum_volume_spike_max)
                        
                        vol_ratio = volume / vol_sma if vol_sma > 0 else 0
                        
                        print(f"\n🔽 [{time_str}] ПЕРЕСЕЧЕНИЕ EMA ВНИЗ:")
                        print(f"   Price: ${row['close']:.2f}")
                        print(f"   ADX: {adx:.2f} {'✅' if adx_ok else f'❌ (нужно > {settings.strategy.momentum_adx_threshold})'}")
                        print(f"   Volume: {vol_ratio:.2f}x {'✅' if vol_ok else f'❌ (нужно {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x)'}")
                        
                        if adx_ok and vol_ok:
                            print(f"   ✅ СИГНАЛ SHORT должен быть сгенерирован!")
                            crossovers.append({
                                'time': time_str,
                                'type': 'SHORT',
                                'generated': True
                            })
                        else:
                            print(f"   ❌ Сигнал НЕ сгенерирован (условия не выполнены)")
                            crossovers.append({
                                'time': time_str,
                                'type': 'SHORT',
                                'generated': False,
                                'reason': f"ADX={adx:.2f}, Vol={vol_ratio:.2f}x"
                            })
        
        print(f"\n{'=' * 80}")
        print("📊 ИТОГИ MOMENTUM:")
        if crossovers:
            print(f"  Найдено {len(crossovers)} пересечений EMA за последние {num_candles} свечей:")
            generated_count = sum(1 for c in crossovers if c['generated'])
            blocked_count = len(crossovers) - generated_count
            print(f"  ✅ Сгенерировано сигналов: {generated_count}")
            print(f"  ❌ Заблокировано (условия не выполнены): {blocked_count}")
        else:
            print(f"  ⚠️ Нет пересечений EMA за последние {num_candles} свечей")
            print(f"  💡 Для генерации сигналов нужно пересечение EMA {settings.strategy.ema_fast_length}/{settings.strategy.ema_slow_length}")
    
    # Трассировка для FLAT стратегии
    if settings.enable_flat_strategy:
        print(f"\n{'=' * 80}")
        print("📊 ТРАССИРОВКА FLAT СТРАТЕГИИ")
        print("=" * 80)
        
        print(f"\nПараметры:")
        print(f"  • ADX порог: <= {settings.strategy.adx_threshold}")
        print(f"  • RSI перепроданность: <= {settings.strategy.range_rsi_oversold}")
        print(f"  • RSI перекупленность: >= {settings.strategy.range_rsi_overbought}")
        print(f"  • Volume множитель: < {settings.strategy.range_volume_mult}x")
        
        flat_opportunities = []
        
        # Анализируем последние N свечей
        start_idx = max(0, len(df_ready) - num_candles)
        
        for i in range(start_idx, len(df_ready)):
            row = df_ready.iloc[i]
            time_str = df_ready.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(df_ready.index[i], 'strftime') else str(df_ready.index[i])
            
            adx = row.get('adx', np.nan)
            rsi = row.get('rsi', np.nan)
            bb_upper = row.get('bb_upper', np.nan)
            bb_lower = row.get('bb_lower', np.nan)
            price = row['close']
            volume = row.get('volume', np.nan)
            vol_sma = row.get('vol_sma', np.nan)
            
            if pd.notna([adx, rsi, bb_upper, bb_lower, volume, vol_sma]).all():
                adx_flat = adx <= settings.strategy.adx_threshold
                rsi_oversold = rsi <= settings.strategy.range_rsi_oversold
                rsi_overbought = rsi >= settings.strategy.range_rsi_overbought
                touch_lower = price <= bb_lower
                touch_upper = price >= bb_upper
                vol_low = volume < vol_sma * settings.strategy.range_volume_mult
                vol_ratio = volume / vol_sma
                
                # LONG сигнал
                if adx_flat and rsi_oversold and touch_lower and vol_low:
                    print(f"\n🟢 [{time_str}] FLAT LONG СИГНАЛ:")
                    print(f"   ✅ ADX: {adx:.2f} <= {settings.strategy.adx_threshold}")
                    print(f"   ✅ RSI: {rsi:.2f} <= {settings.strategy.range_rsi_oversold}")
                    print(f"   ✅ Price: ${price:.2f} <= ${bb_lower:.2f} (касание нижней BB)")
                    print(f"   ✅ Volume: {vol_ratio:.2f}x < {settings.strategy.range_volume_mult}x")
                    flat_opportunities.append({'time': time_str, 'type': 'LONG'})
                
                # SHORT сигнал
                elif adx_flat and rsi_overbought and touch_upper and vol_low:
                    print(f"\n🔴 [{time_str}] FLAT SHORT СИГНАЛ:")
                    print(f"   ✅ ADX: {adx:.2f} <= {settings.strategy.adx_threshold}")
                    print(f"   ✅ RSI: {rsi:.2f} >= {settings.strategy.range_rsi_overbought}")
                    print(f"   ✅ Price: ${price:.2f} >= ${bb_upper:.2f} (касание верхней BB)")
                    print(f"   ✅ Volume: {vol_ratio:.2f}x < {settings.strategy.range_volume_mult}x")
                    flat_opportunities.append({'time': time_str, 'type': 'SHORT'})
        
        print(f"\n{'=' * 80}")
        print("📊 ИТОГИ FLAT:")
        if flat_opportunities:
            print(f"  ✅ Найдено {len(flat_opportunities)} возможностей для входа за последние {num_candles} свечей")
        else:
            print(f"  ⚠️ Нет возможностей для FLAT стратегии за последние {num_candles} свечей")
            
            # Диагностика почему нет сигналов
            last_row = df_ready.iloc[-1]
            adx = last_row.get('adx', np.nan)
            rsi = last_row.get('rsi', np.nan)
            price = last_row['close']
            bb_upper = last_row.get('bb_upper', np.nan)
            bb_lower = last_row.get('bb_lower', np.nan)
            volume = last_row.get('volume', np.nan)
            vol_sma = last_row.get('vol_sma', np.nan)
            
            print(f"\n  Текущие условия (последняя свеча):")
            adx_ok = adx <= settings.strategy.adx_threshold
            print(f"    ADX <= {settings.strategy.adx_threshold}: {adx:.2f} {'✅' if adx_ok else '❌ (рынок в тренде, не во флэте)'}")
            
            rsi_oversold = rsi <= settings.strategy.range_rsi_oversold
            rsi_overbought = rsi >= settings.strategy.range_rsi_overbought
            print(f"    RSI перепроданность (<= {settings.strategy.range_rsi_oversold}): {rsi:.2f} {'✅' if rsi_oversold else '❌'}")
            print(f"    RSI перекупленность (>= {settings.strategy.range_rsi_overbought}): {rsi:.2f} {'✅' if rsi_overbought else '❌'}")
            
            touch_lower = price <= bb_lower
            touch_upper = price >= bb_upper
            print(f"    Price касание BB Lower: ${price:.2f} <= ${bb_lower:.2f} {'✅' if touch_lower else '❌'}")
            print(f"    Price касание BB Upper: ${price:.2f} >= ${bb_upper:.2f} {'✅' if touch_upper else '❌'}")
            
            vol_ratio = volume / vol_sma if vol_sma > 0 else 0
            vol_ok = volume < vol_sma * settings.strategy.range_volume_mult
            print(f"    Volume < {settings.strategy.range_volume_mult}x: {vol_ratio:.2f}x {'✅' if vol_ok else '❌'}")
            
            print(f"\n  💡 Причина отсутствия сигналов:")
            reasons = []
            if not adx_ok:
                reasons.append(f"ADX слишком высокий ({adx:.2f} > {settings.strategy.adx_threshold}) - рынок в тренде")
            if not (rsi_oversold or rsi_overbought):
                reasons.append(f"RSI в нормальной зоне ({rsi:.2f}), нет экстремумов")
            if not (touch_lower or touch_upper):
                reasons.append(f"Цена не касается границ BB (в середине диапазона)")
            if not vol_ok:
                reasons.append(f"Volume слишком высокий ({vol_ratio:.2f}x)")
            
            for reason in reasons:
                print(f"    • {reason}")


if __name__ == "__main__":
    trace_last_candles(num_candles=20)
