"""
Диагностический скрипт для определения причины отсутствия сигналов.
Проверяет:
1. Генерацию сигналов от всех стратегий
2. Фильтры "свежести" сигналов
3. Блокировку по Loss Cooldown
4. Блокировку по ATR Entry Filter
5. Наличие открытых позиций
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import build_signals, Action, enrich_for_strategy
from bot.ml.strategy_ml import build_ml_signals
from bot.smc_strategy import build_smc_signals
from bot.live import _timeframe_to_bybit_interval
from bot.web.history import check_recent_loss_trade, get_trades


def diagnose_signals(symbol: str = None):
    """Полная диагностика системы генерации сигналов"""
    print("=" * 80)
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ ГЕНЕРАЦИИ СИГНАЛОВ")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    if symbol:
        settings.symbol = symbol
        settings.primary_symbol = symbol
    
    print(f"\n📊 Тестируем символ: {settings.symbol}")
    print(f"⏱️ Таймфрейм: {settings.timeframe}")
    print(f"📈 Лимит свечей: {settings.kline_limit}")
    
    # Инициализируем клиент
    client = BybitClient(api=settings.api)
    
    # 1. ПРОВЕРКА ПОЛУЧЕНИЯ ДАННЫХ
    print(f"\n{'=' * 80}")
    print("1️⃣ ПРОВЕРКА ПОЛУЧЕНИЯ ДАННЫХ")
    print("=" * 80)
    
    try:
        interval = _timeframe_to_bybit_interval(settings.timeframe)
        df_raw = client.get_kline_df(
            symbol=settings.symbol, 
            interval=interval, 
            limit=settings.kline_limit
        )
        print(f"✅ Получено {len(df_raw)} свечей")
        print(f"   Период: {df_raw.index[0]} - {df_raw.index[-1]}")
        print(f"   Последняя цена: ${df_raw.iloc[-1]['close']:.2f}")
    except Exception as e:
        print(f"❌ ОШИБКА получения данных: {e}")
        return
    
    # 2. ПОДГОТОВКА ИНДИКАТОРОВ
    print(f"\n{'=' * 80}")
    print("2️⃣ ПОДГОТОВКА ИНДИКАТОРОВ")
    print("=" * 80)
    
    try:
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
        print(f"✅ Подготовлено {len(df_ready)} строк с индикаторами")
        
        if not df_ready.empty:
            last_row = df_ready.iloc[-1]
            print(f"\n📊 Последние индикаторы:")
            print(f"   ADX: {last_row.get('adx', 'N/A'):.2f}")
            print(f"   RSI: {last_row.get('rsi', 'N/A'):.2f}")
            print(f"   +DI: {last_row.get('plus_di', 'N/A'):.2f}")
            print(f"   -DI: {last_row.get('minus_di', 'N/A'):.2f}")
            print(f"   Volume: {last_row.get('volume', 'N/A'):.0f}")
            print(f"   Vol SMA: {last_row.get('vol_sma', 'N/A'):.0f}")
            print(f"   BB Upper: ${last_row.get('bb_upper', 'N/A'):.2f}")
            print(f"   BB Lower: ${last_row.get('bb_lower', 'N/A'):.2f}")
    except Exception as e:
        print(f"❌ ОШИБКА подготовки индикаторов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. ГЕНЕРАЦИЯ СИГНАЛОВ ОТ ВСЕХ СТРАТЕГИЙ
    print(f"\n{'=' * 80}")
    print("3️⃣ ГЕНЕРАЦИЯ СИГНАЛОВ ОТ ВСЕХ СТРАТЕГИЙ")
    print("=" * 80)
    
    all_signals_by_strategy = {}
    
    # TREND стратегия
    if settings.enable_trend_strategy:
        print(f"\n📈 TREND стратегия:")
        try:
            trend_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=False)
            trend_actionable = [s for s in trend_signals if s.reason.startswith("trend_") and s.action in (Action.LONG, Action.SHORT)]
            print(f"   Всего сигналов: {len(trend_signals)}")
            print(f"   Действующих (LONG/SHORT): {len(trend_actionable)}")
            
            if trend_actionable:
                for i, sig in enumerate(trend_actionable[:3]):
                    print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{sig.timestamp}]")
            else:
                print(f"   ⚠️ Нет действующих TREND сигналов")
            
            all_signals_by_strategy['trend'] = trend_actionable
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print(f"\n📈 TREND стратегия: ❌ ОТКЛЮЧЕНА")
    
    # FLAT стратегия
    if settings.enable_flat_strategy:
        print(f"\n📊 FLAT стратегия:")
        try:
            flat_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=False)
            flat_actionable = [s for s in flat_signals if s.reason.startswith("range_") and s.action in (Action.LONG, Action.SHORT)]
            print(f"   Всего сигналов: {len(flat_signals)}")
            print(f"   Действующих (LONG/SHORT): {len(flat_actionable)}")
            
            if flat_actionable:
                for i, sig in enumerate(flat_actionable[:3]):
                    print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{sig.timestamp}]")
            else:
                print(f"   ⚠️ Нет действующих FLAT сигналов")
            
            all_signals_by_strategy['flat'] = flat_actionable
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print(f"\n📊 FLAT стратегия: ❌ ОТКЛЮЧЕНА")
    
    # MOMENTUM стратегия
    if settings.enable_momentum_strategy:
        print(f"\n⚡ MOMENTUM стратегия:")
        try:
            momentum_signals = build_signals(df_ready, settings.strategy, use_momentum=True, use_liquidity=False)
            momentum_actionable = [s for s in momentum_signals if s.reason.startswith("momentum_") and s.action in (Action.LONG, Action.SHORT)]
            print(f"   Всего сигналов: {len(momentum_signals)}")
            print(f"   Действующих (LONG/SHORT): {len(momentum_actionable)}")
            
            if momentum_actionable:
                for i, sig in enumerate(momentum_actionable[:3]):
                    print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{sig.timestamp}]")
            else:
                print(f"   ⚠️ Нет действующих MOMENTUM сигналов")
                
                # РАСШИРЕННАЯ ДИАГНОСТИКА: Проверяем условия для последних 10 свечей
                print(f"\n   📊 Анализ последних 10 свечей:")
                ema_timeframe = settings.strategy.momentum_ema_timeframe
                ema_fast_col = f"ema_fast_{ema_timeframe}"
                ema_slow_col = f"ema_slow_{ema_timeframe}"
                
                # Проверяем пересечения EMA
                crossovers_up = []
                crossovers_down = []
                
                for i in range(max(0, len(df_ready)-10), len(df_ready)):
                    if i > 0:
                        row = df_ready.iloc[i]
                        prev_row = df_ready.iloc[i-1]
                        
                        ema_fast = row.get(ema_fast_col, np.nan)
                        ema_slow = row.get(ema_slow_col, np.nan)
                        prev_ema_fast = prev_row.get(ema_fast_col, np.nan)
                        prev_ema_slow = prev_row.get(ema_slow_col, np.nan)
                        
                        if pd.notna([ema_fast, ema_slow, prev_ema_fast, prev_ema_slow]).all():
                            # Пересечение вверх
                            if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
                                adx = row.get("adx", np.nan)
                                volume = row.get("volume", np.nan)
                                vol_sma = row.get("vol_sma", np.nan)
                                
                                adx_ok = pd.notna(adx) and adx > settings.strategy.momentum_adx_threshold
                                vol_ok = (pd.notna([volume, vol_sma]).all() and 
                                         volume >= vol_sma * settings.strategy.momentum_volume_spike_min and
                                         volume <= vol_sma * settings.strategy.momentum_volume_spike_max)
                                
                                crossovers_up.append({
                                    'time': df_ready.index[i],
                                    'price': row['close'],
                                    'adx': adx,
                                    'adx_ok': adx_ok,
                                    'volume': volume,
                                    'vol_sma': vol_sma,
                                    'vol_ok': vol_ok
                                })
                            
                            # Пересечение вниз
                            elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
                                adx = row.get("adx", np.nan)
                                volume = row.get("volume", np.nan)
                                vol_sma = row.get("vol_sma", np.nan)
                                
                                adx_ok = pd.notna(adx) and adx > settings.strategy.momentum_adx_threshold
                                vol_ok = (pd.notna([volume, vol_sma]).all() and 
                                         volume >= vol_sma * settings.strategy.momentum_volume_spike_min and
                                         volume <= vol_sma * settings.strategy.momentum_volume_spike_max)
                                
                                crossovers_down.append({
                                    'time': df_ready.index[i],
                                    'price': row['close'],
                                    'adx': adx,
                                    'adx_ok': adx_ok,
                                    'volume': volume,
                                    'vol_sma': vol_sma,
                                    'vol_ok': vol_ok
                                })
                
                if crossovers_up:
                    print(f"\n   ✅ Найдено {len(crossovers_up)} пересечений EMA ВВЕРХ:")
                    for co in crossovers_up[-3:]:
                        print(f"      • {co['time']}: ${co['price']:.2f}")
                        adx_status = '✅' if co['adx_ok'] else f"❌ (нужно > {settings.strategy.momentum_adx_threshold})"
                        print(f"        ADX: {co['adx']:.2f} {adx_status}")
                        vol_ratio = co['volume'] / co['vol_sma'] if co['vol_sma'] > 0 else 0
                        vol_status = '✅' if co['vol_ok'] else f"❌ (нужно {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x)"
                        print(f"        Volume: {co['volume']:.0f} / {co['vol_sma']:.0f} ({vol_ratio:.2f}x) {vol_status}")
                
                if crossovers_down:
                    print(f"\n   ✅ Найдено {len(crossovers_down)} пересечений EMA ВНИЗ:")
                    for co in crossovers_down[-3:]:
                        print(f"      • {co['time']}: ${co['price']:.2f}")
                        adx_status = '✅' if co['adx_ok'] else f"❌ (нужно > {settings.strategy.momentum_adx_threshold})"
                        print(f"        ADX: {co['adx']:.2f} {adx_status}")
                        vol_ratio = co['volume'] / co['vol_sma'] if co['vol_sma'] > 0 else 0
                        vol_status = '✅' if co['vol_ok'] else f"❌ (нужно {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x)"
                        print(f"        Volume: {co['volume']:.0f} / {co['vol_sma']:.0f} ({vol_ratio:.2f}x) {vol_status}")
                
                if not crossovers_up and not crossovers_down:
                    print(f"   💡 Нет пересечений EMA за последние 10 свечей")
                    print(f"   💡 Для генерации сигналов нужно:")
                    print(f"      1. Пересечение EMA {settings.strategy.ema_fast_length}/{settings.strategy.ema_slow_length}")
                    print(f"      2. ADX > {settings.strategy.momentum_adx_threshold}")
                    print(f"      3. Volume spike {settings.strategy.momentum_volume_spike_min}-{settings.strategy.momentum_volume_spike_max}x")
            
            all_signals_by_strategy['momentum'] = momentum_actionable
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚡ MOMENTUM стратегия: ❌ ОТКЛЮЧЕНА")
    
    # LIQUIDITY стратегия
    if settings.enable_liquidity_sweep_strategy:
        print(f"\n💧 LIQUIDITY стратегия:")
        try:
            liquidity_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=True)
            liquidity_actionable = [s for s in liquidity_signals if s.reason.startswith("liquidity_") and s.action in (Action.LONG, Action.SHORT)]
            print(f"   Всего сигналов: {len(liquidity_signals)}")
            print(f"   Действующих (LONG/SHORT): {len(liquidity_actionable)}")
            
            if liquidity_actionable:
                for i, sig in enumerate(liquidity_actionable[:3]):
                    print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{sig.timestamp}]")
            else:
                print(f"   ⚠️ Нет действующих LIQUIDITY сигналов")
            
            all_signals_by_strategy['liquidity'] = liquidity_actionable
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print(f"\n💧 LIQUIDITY стратегия: ❌ ОТКЛЮЧЕНА")
    
    # SMC стратегия
    if settings.enable_smc_strategy:
        print(f"\n🟣 SMC стратегия:")
        try:
            smc_signals = build_smc_signals(df_ready, settings.strategy, symbol=settings.symbol)
            smc_actionable = [s for s in smc_signals if s.action in (Action.LONG, Action.SHORT)]
            print(f"   Всего сигналов: {len(smc_signals)}")
            print(f"   Действующих (LONG/SHORT): {len(smc_actionable)}")
            
            if smc_actionable:
                for i, sig in enumerate(smc_actionable[:3]):
                    sl_str = f"SL: ${sig.stop_loss:.2f}" if hasattr(sig, 'stop_loss') and sig.stop_loss else ""
                    tp_str = f"TP: ${sig.take_profit:.2f}" if hasattr(sig, 'take_profit') and sig.take_profit else ""
                    print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} {sl_str} {tp_str} - {sig.reason} [{sig.timestamp}]")
            else:
                print(f"   ⚠️ Нет действующих SMC сигналов")
                if len(df_ready) < 1000:
                    print(f"   💡 SMC требует минимум 1000 свечей. Текущее: {len(df_ready)}")
            
            all_signals_by_strategy['smc'] = smc_actionable
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n🟣 SMC стратегия: ❌ ОТКЛЮЧЕНА")
    
    # ML стратегия
    if settings.enable_ml_strategy:
        print(f"\n🤖 ML стратегия:")
        try:
            if settings.ml_model_path:
                ml_signals = build_ml_signals(
                    df_ready,
                    model_path=settings.ml_model_path,
                    confidence_threshold=settings.ml_confidence_threshold,
                    min_signal_strength=settings.ml_min_signal_strength,
                    stability_filter=settings.ml_stability_filter,
                    leverage=settings.leverage,
                    target_profit_pct_margin=settings.ml_target_profit_pct_margin,
                    max_loss_pct_margin=settings.ml_max_loss_pct_margin,
                )
                ml_actionable = [s for s in ml_signals if s.reason.startswith("ml_") and s.action in (Action.LONG, Action.SHORT)]
                print(f"   Модель: {settings.ml_model_path}")
                print(f"   Всего сигналов: {len(ml_signals)}")
                print(f"   Действующих (LONG/SHORT): {len(ml_actionable)}")
                
                if ml_actionable:
                    for i, sig in enumerate(ml_actionable[:3]):
                        print(f"   [{i+1}] {sig.action.value} @ ${sig.price:.2f} - {sig.reason} [{sig.timestamp}]")
                else:
                    print(f"   ⚠️ Нет действующих ML сигналов")
                
                all_signals_by_strategy['ml'] = ml_actionable
            else:
                print(f"   ⚠️ ML модель не найдена")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n🤖 ML стратегия: ❌ ОТКЛЮЧЕНА")
    
    # 4. ПРОВЕРКА ФИЛЬТРА "СВЕЖЕСТИ"
    print(f"\n{'=' * 80}")
    print("4️⃣ ПРОВЕРКА ФИЛЬТРА 'СВЕЖЕСТИ' СИГНАЛОВ")
    print("=" * 80)
    
    def is_signal_fresh_local(sig, df_ready):
        """Локальная копия функции проверки свежести"""
        try:
            if df_ready.empty:
                return True
            
            ts = sig.timestamp
            if isinstance(ts, pd.Timestamp):
                signal_ts = ts
                if signal_ts.tzinfo is None:
                    signal_ts = signal_ts.tz_localize('UTC')
                else:
                    signal_ts = signal_ts.tz_convert('UTC')
                
                # Проверяем последние 10 свечей
                num_candles_to_check = min(10, len(df_ready))
                last_timestamps = df_ready.index[-num_candles_to_check:].tolist()
                
                for last_ts in last_timestamps:
                    if isinstance(last_ts, pd.Timestamp):
                        last_ts_utc = last_ts
                        if last_ts_utc.tzinfo is None:
                            last_ts_utc = last_ts_utc.tz_localize('UTC')
                        else:
                            last_ts_utc = last_ts_utc.tz_convert('UTC')
                        
                        time_diff_seconds = abs((signal_ts - last_ts_utc).total_seconds())
                        if time_diff_seconds < 300:  # 5 минут
                            return True
            return False
        except Exception:
            return True
    
    total_fresh = 0
    total_old = 0
    
    for strategy_name, signals in all_signals_by_strategy.items():
        if signals:
            fresh = [s for s in signals if is_signal_fresh_local(s, df_ready)]
            old = [s for s in signals if not is_signal_fresh_local(s, df_ready)]
            total_fresh += len(fresh)
            total_old += len(old)
            
            print(f"\n{strategy_name.upper()}: {len(signals)} сигналов")
            print(f"   ✅ Свежих: {len(fresh)}")
            print(f"   ⏰ Старых: {len(old)}")
            
            if old:
                print(f"   Примеры старых сигналов:")
                for i, sig in enumerate(old[:3]):
                    sig_time = sig.timestamp if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                    last_candle_time = df_ready.index[-1] if not df_ready.empty else "N/A"
                    print(f"     [{i+1}] Сигнал: {sig_time}, Последняя свеча: {last_candle_time}")
    
    print(f"\n📊 Итого: {total_fresh} свежих, {total_old} старых")
    
    if total_old > 0 and total_fresh == 0:
        print(f"\n⚠️ ПРОБЛЕМА: Все сигналы помечены как 'старые' и будут отфильтрованы!")
        print(f"   Возможная причина: Таймстемпы сигналов не совпадают с таймстемпами последних свечей")
    
    # 5. ПРОВЕРКА БЛОКИРОВКИ ПО LOSS COOLDOWN
    print(f"\n{'=' * 80}")
    print("5️⃣ ПРОВЕРКА БЛОКИРОВКИ ПО LOSS COOLDOWN")
    print("=" * 80)
    
    if settings.risk.enable_loss_cooldown:
        print(f"✅ Loss Cooldown фильтр: ВКЛЮЧЕН")
        print(f"   Период: {settings.risk.loss_cooldown_minutes} минут")
        print(f"   Макс. убытков подряд: {settings.risk.max_consecutive_losses}")
        
        # Проверяем для LONG
        should_block_long, last_loss_long = check_recent_loss_trade(
            side="long",
            symbol=settings.symbol,
            cooldown_minutes=settings.risk.loss_cooldown_minutes,
            max_losses=settings.risk.max_consecutive_losses,
        )
        
        if should_block_long:
            print(f"\n   🚫 LONG сигналы БЛОКИРОВАНЫ!")
            if last_loss_long:
                print(f"      Последняя убыточная сделка:")
                print(f"      PnL: {last_loss_long.get('pnl', 0):.2f} USDT")
                print(f"      Причина: {last_loss_long.get('exit_reason', 'unknown')}")
                print(f"      Время: {last_loss_long.get('exit_time', 'N/A')}")
        else:
            print(f"   ✅ LONG сигналы разрешены")
        
        # Проверяем для SHORT
        should_block_short, last_loss_short = check_recent_loss_trade(
            side="short",
            symbol=settings.symbol,
            cooldown_minutes=settings.risk.loss_cooldown_minutes,
            max_losses=settings.risk.max_consecutive_losses,
        )
        
        if should_block_short:
            print(f"\n   🚫 SHORT сигналы БЛОКИРОВАНЫ!")
            if last_loss_short:
                print(f"      Последняя убыточная сделка:")
                print(f"      PnL: {last_loss_short.get('pnl', 0):.2f} USDT")
                print(f"      Причина: {last_loss_short.get('exit_reason', 'unknown')}")
                print(f"      Время: {last_loss_short.get('exit_time', 'N/A')}")
        else:
            print(f"   ✅ SHORT сигналы разрешены")
    else:
        print(f"⚠️ Loss Cooldown фильтр: ВЫКЛЮЧЕН")
    
    # 6. ПРОВЕРКА БЛОКИРОВКИ ПО ATR ENTRY FILTER
    print(f"\n{'=' * 80}")
    print("6️⃣ ПРОВЕРКА БЛОКИРОВКИ ПО ATR ENTRY FILTER")
    print("=" * 80)
    
    if settings.risk.enable_atr_entry_filter:
        print(f"✅ ATR Entry фильтр: ВКЛЮЧЕН")
        print(f"   Макс. прогресс ATR: {settings.risk.max_atr_progress_pct * 100:.1f}%")
        
        if not df_ready.empty and len(df_ready) >= 2:
            last_row = df_ready.iloc[-1]
            prev_row = df_ready.iloc[-2]
            
            atr_value = last_row.get("atr_avg", None)
            if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
                atr_value = last_row.get("atr", None)
            
            current_price = last_row['close']
            prev_close = prev_row.get("close", current_price)
            
            if atr_value and pd.notna(atr_value) and atr_value > 0:
                price_move = current_price - prev_close
                atr_progress = abs(price_move) / atr_value if atr_value > 0 else 0
                
                print(f"\n   Текущая цена: ${current_price:.2f}")
                print(f"   Предыдущая цена: ${prev_close:.2f}")
                print(f"   Движение: ${price_move:.2f} ({atr_progress*100:.1f}% ATR)")
                print(f"   ATR: ${atr_value:.2f}")
                
                if price_move > 0 and atr_progress > settings.risk.max_atr_progress_pct:
                    print(f"\n   🚫 LONG сигналы будут БЛОКИРОВАНЫ (цена прошла {atr_progress*100:.1f}% ATR вверх)")
                elif price_move < 0 and atr_progress > settings.risk.max_atr_progress_pct:
                    print(f"\n   🚫 SHORT сигналы будут БЛОКИРОВАНЫ (цена прошла {atr_progress*100:.1f}% ATR вниз)")
                else:
                    print(f"\n   ✅ Сигналы разрешены (прогресс ATR: {atr_progress*100:.1f}%)")
    else:
        print(f"⚠️ ATR Entry фильтр: ВЫКЛЮЧЕН")
    
    # 7. ПРОВЕРКА ОТКРЫТЫХ ПОЗИЦИЙ
    print(f"\n{'=' * 80}")
    print("7️⃣ ПРОВЕРКА ОТКРЫТЫХ ПОЗИЦИЙ")
    print("=" * 80)
    
    try:
        positions = client.get_positions(symbol=settings.symbol)
        if positions:
            for pos in positions:
                size = float(pos.get('size', 0))
                if size > 0:
                    print(f"📍 Открыта позиция:")
                    print(f"   Сторона: {pos.get('side', 'N/A')}")
                    print(f"   Размер: {size}")
                    print(f"   Цена входа: ${float(pos.get('avg_price', 0)):.2f}")
                    print(f"   PnL: ${float(pos.get('unrealised_pnl', 0)):.2f}")
        else:
            print(f"✅ Нет открытых позиций")
    except Exception as e:
        print(f"❌ Ошибка проверки позиций: {e}")
    
    # 8. ОБЩИЙ ВЫВОД
    print(f"\n{'=' * 80}")
    print("8️⃣ ОБЩИЙ ВЫВОД")
    print("=" * 80)
    
    total_signals = sum(len(signals) for signals in all_signals_by_strategy.values())
    
    if total_signals == 0:
        print(f"\n❌ ПРОБЛЕМА: НИ ОДНА СТРАТЕГИЯ НЕ ГЕНЕРИРУЕТ СИГНАЛЫ!")
        print(f"\n Возможные причины:")
        print(f"   1. Все стратегии отключены в настройках")
        print(f"   2. Рыночные условия не подходят ни для одной стратегии")
        print(f"   3. Ошибки в логике генерации сигналов")
        print(f"\n Рекомендации:")
        print(f"   - Проверьте настройки стратегий в .env (ENABLE_*_STRATEGY)")
        print(f"   - Проанализируйте последние индикаторы (раздел 2)")
        print(f"   - Попробуйте другой символ или таймфрейм")
    elif total_fresh == 0 and total_old > 0:
        print(f"\n⚠️ ПРОБЛЕМА: СИГНАЛЫ ГЕНЕРИРУЮТСЯ, НО ВСЕ ПОМЕЧЕНЫ КАК 'СТАРЫЕ'!")
        print(f"\n Возможная причина:")
        print(f"   Таймстемпы сигналов не совпадают с таймстемпами свечей")
        print(f"\n Рекомендация:")
        print(f"   Проверьте логику генерации timestamp в стратегиях")
    elif total_fresh > 0:
        print(f"\n✅ Найдено {total_fresh} свежих сигналов!")
        
        # Проверяем блокировки
        if settings.risk.enable_loss_cooldown:
            should_block_long, _ = check_recent_loss_trade("long", settings.symbol, settings.risk.loss_cooldown_minutes, settings.risk.max_consecutive_losses)
            should_block_short, _ = check_recent_loss_trade("short", settings.symbol, settings.risk.loss_cooldown_minutes, settings.risk.max_consecutive_losses)
            
            if should_block_long or should_block_short:
                print(f"\n⚠️ ВНИМАНИЕ: Loss Cooldown блокирует сигналы!")
                print(f"   Проверьте раздел 5 для деталей")
    
    print(f"\n{'=' * 80}")
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else None
    diagnose_signals(symbol)
