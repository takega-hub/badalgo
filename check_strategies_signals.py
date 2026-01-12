"""
Диагностический скрипт для проверки генерации сигналов по всем стратегиям и парам.
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import build_signals, Action, enrich_for_strategy
from bot.ml.strategy_ml import build_ml_signals
from bot.smc_strategy import build_smc_signals
from bot.live import _timeframe_to_bybit_interval
import pandas as pd
import numpy as np


def check_strategies_for_symbol(symbol: str, settings):
    """Проверяет генерацию сигналов для одного символа."""
    print(f"\n{'='*80}")
    print(f"🔍 Проверка стратегий для {symbol}")
    print(f"{'='*80}")
    
    try:
        # Создаем клиент
        client = BybitClient(settings.api)
        
        # Получаем данные
        interval = _timeframe_to_bybit_interval(settings.timeframe)
        print(f"\n📊 Получение данных для {symbol} ({interval})...")
        print(f"  ⚠️ ВАЖНО: Скрипт проверяет только последние 500 свечей!")
        print(f"  ⚠️ Сигналы в админке могут быть старше и не попасть в этот диапазон!")
        
        klines = client.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=500
        )
        
        if klines.get("retCode") != 0:
            print(f"❌ Ошибка получения данных: {klines.get('retMsg')}")
            return
        
        list_data = klines.get("result", {}).get("list", [])
        if not list_data:
            print(f"❌ Нет данных для {symbol}")
            return
        
        # Преобразуем в DataFrame
        df_raw = pd.DataFrame(list_data)
        df_raw = df_raw.iloc[::-1].reset_index(drop=True)  # Переворачиваем (старые -> новые)
        df_raw.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df_raw = df_raw.astype({
            "open": float, "high": float, "low": float, "close": float,
            "volume": float, "turnover": float
        })
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"].astype(int), unit="ms")
        
        print(f"✅ Получено {len(df_raw)} свечей")
        
        # Подготавливаем данные с индикаторами
        print(f"\n📈 Подготовка индикаторов...")
        df_ready = prepare_with_indicators(
            df_raw,
            adx_length=settings.strategy.adx_length,
            di_length=settings.strategy.di_length,
            sma_length=settings.strategy.sma_length,
            rsi_length=settings.strategy.rsi_length,
            breakout_lookback=settings.strategy.breakout_lookback,
            bb_length=settings.strategy.bb_length,
            bb_std=settings.strategy.bb_std,
            atr_length=14,
            ema_fast_length=settings.strategy.ema_fast_length,
            ema_slow_length=settings.strategy.ema_slow_length,
            ema_timeframe="1h",
        )
        
        if df_ready.empty:
            print(f"❌ DataFrame пуст после подготовки индикаторов")
            return
        
        print(f"✅ Подготовлено {len(df_ready)} строк с индикаторами")
        
        # Обогащаем данные для стратегий (добавляем bias, consolidation и т.д.)
        df_ready = enrich_for_strategy(df_ready, settings.strategy)
        
        # Берем последнюю строку для анализа
        last_row = df_ready.iloc[-1]
        
        # Показываем информацию о последних свечах
        print(f"\n📅 Информация о данных:")
        if not df_ready.empty:
            last_timestamp = df_ready.iloc[-1].get('timestamp', None)
            if last_timestamp is not None:
                if isinstance(last_timestamp, (int, float)):
                    last_timestamp = pd.to_datetime(last_timestamp, unit='ms')
                elif not isinstance(last_timestamp, pd.Timestamp):
                    last_timestamp = pd.to_datetime(last_timestamp)
                print(f"  • Последняя свеча: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  • Всего свечей для анализа: {len(df_ready)}")
        
        # Показываем текущие индикаторы
        print(f"\n📊 Текущие индикаторы для {symbol}:")
        print(f"  • Цена: ${last_row['close']:.2f}")
        print(f"  • ADX: {last_row.get('adx', np.nan):.2f}" if pd.notna(last_row.get('adx')) else "  • ADX: N/A")
        print(f"  • +DI: {last_row.get('plus_di', np.nan):.2f}" if pd.notna(last_row.get('plus_di')) else "  • +DI: N/A")
        print(f"  • -DI: {last_row.get('minus_di', np.nan):.2f}" if pd.notna(last_row.get('minus_di')) else "  • -DI: N/A")
        print(f"  • RSI: {last_row.get('rsi', np.nan):.2f}" if pd.notna(last_row.get('rsi')) else "  • RSI: N/A")
        print(f"  • Volume: {last_row.get('volume', 0):.0f}")
        print(f"  • Volume SMA: {last_row.get('vol_sma', np.nan):.0f}" if pd.notna(last_row.get('vol_sma')) else "  • Volume SMA: N/A")
        
        # Проверяем каждую стратегию
        print(f"\n🎯 Проверка стратегий:")
        print(f"  • Trend: {'✅ ВКЛ' if settings.enable_trend_strategy else '❌ ВЫКЛ'}")
        print(f"  • Flat: {'✅ ВКЛ' if settings.enable_flat_strategy else '❌ ВЫКЛ'}")
        print(f"  • ML: {'✅ ВКЛ' if settings.enable_ml_strategy else '❌ ВЫКЛ'}")
        print(f"  • Momentum: {'✅ ВКЛ' if settings.enable_momentum_strategy else '❌ ВЫКЛ'}")
        print(f"  • Liquidity: {'✅ ВКЛ' if settings.enable_liquidity_sweep_strategy else '❌ ВЫКЛ'}")
        print(f"  • SMC: {'✅ ВКЛ' if settings.enable_smc_strategy else '❌ ВЫКЛ'}")
        
        # 1. TREND стратегия
        if settings.enable_trend_strategy:
            print(f"\n📈 TREND стратегия:")
            trend_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=False)
            trend_actionable = [s for s in trend_signals if s.reason.startswith("trend_") and s.action in (Action.LONG, Action.SHORT)]
            
            if trend_actionable:
                print(f"  ✅ Найдено {len(trend_actionable)} сигналов:")
                for sig in trend_actionable[-5:]:  # Последние 5
                    print(f"    • {sig.action.value} @ ${sig.price:.2f} - {sig.reason}")
            else:
                print(f"  ⚠️ Нет сигналов LONG/SHORT")
                # Диагностика
                adx = last_row.get('adx', np.nan)
                if pd.notna(adx):
                    if adx <= settings.strategy.adx_threshold:
                        print(f"    💡 ADX ({adx:.2f}) <= порога ({settings.strategy.adx_threshold}) - рынок не в тренде")
                    else:
                        print(f"    💡 ADX ({adx:.2f}) > порога ({settings.strategy.adx_threshold}) - рынок в тренде, но нет условий для входа")
                        # Проверяем условия
                        plus_di = last_row.get('plus_di', np.nan)
                        minus_di = last_row.get('minus_di', np.nan)
                        recent_high = last_row.get('recent_high', np.nan)
                        recent_low = last_row.get('recent_low', np.nan)
                        price = last_row['close']
                        volume = last_row.get('volume', 0)
                        vol_sma = last_row.get('vol_sma', np.nan)
                        vol_ok = pd.notna(vol_sma) and volume > vol_sma * settings.strategy.breakout_volume_mult
                        
                        print(f"      - Price vs Recent High: ${price:.2f} vs ${recent_high:.2f} (breakout: {price > recent_high})")
                        print(f"      - Price vs Recent Low: ${price:.2f} vs ${recent_low:.2f} (breakout: {price < recent_low})")
                        print(f"      - Volume OK: {vol_ok} (Volume: {volume:.0f}, Vol SMA: {vol_sma:.0f}, Mult: {settings.strategy.breakout_volume_mult})")
                        print(f"      - +DI: {plus_di:.2f}, -DI: {minus_di:.2f}")
        
        # 2. FLAT стратегия
        if settings.enable_flat_strategy:
            print(f"\n📊 FLAT стратегия:")
            flat_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=False)
            flat_actionable = [s for s in flat_signals if s.reason.startswith("range_") and s.action in (Action.LONG, Action.SHORT)]
            
            if flat_actionable:
                print(f"  ✅ Найдено {len(flat_actionable)} сигналов:")
                for sig in flat_actionable[-5:]:  # Последние 5
                    print(f"    • {sig.action.value} @ ${sig.price:.2f} - {sig.reason}")
            else:
                print(f"  ⚠️ Нет сигналов LONG/SHORT")
                # Диагностика
                adx = last_row.get('adx', np.nan)
                rsi = last_row.get('rsi', np.nan)
                bb_upper = last_row.get('bb_upper', np.nan)
                bb_lower = last_row.get('bb_lower', np.nan)
                price = last_row['close']
                volume = last_row.get('volume', 0)
                vol_sma = last_row.get('vol_sma', np.nan)
                
                if pd.notna(adx) and adx > settings.strategy.adx_threshold:
                    print(f"    💡 ADX ({adx:.2f}) > порога ({settings.strategy.adx_threshold}) - рынок в тренде, FLAT не работает")
                else:
                    print(f"    💡 Рынок во флэте, но нет условий для входа:")
                    if pd.notna([rsi, bb_upper, bb_lower, price, volume, vol_sma]).all():
                        touch_lower = price <= bb_lower
                        touch_upper = price >= bb_upper
                        rsi_oversold = rsi <= settings.strategy.range_rsi_oversold
                        rsi_overbought = rsi >= settings.strategy.range_rsi_overbought
                        volume_ok = volume < vol_sma * settings.strategy.range_volume_mult
                        volume_confirms = volume > vol_sma * 0.8
                        
                        print(f"      - Touch BB Lower: {touch_lower} (Price: ${price:.2f}, BB Lower: ${bb_lower:.2f})")
                        print(f"      - Touch BB Upper: {touch_upper} (Price: ${price:.2f}, BB Upper: ${bb_upper:.2f})")
                        print(f"      - RSI Oversold: {rsi_oversold} (RSI: {rsi:.2f}, Threshold: {settings.strategy.range_rsi_oversold})")
                        print(f"      - RSI Overbought: {rsi_overbought} (RSI: {rsi:.2f}, Threshold: {settings.strategy.range_rsi_overbought})")
                        print(f"      - Volume OK: {volume_ok} (Volume: {volume:.0f}, Vol SMA: {vol_sma:.0f}, Mult: {settings.strategy.range_volume_mult})")
                        print(f"      - Volume Confirms: {volume_confirms}")
                        
                        # Проверяем условия для LONG
                        long_ready = touch_lower and rsi_oversold and volume_ok and volume_confirms
                        short_ready = touch_upper and rsi_overbought and volume_ok and volume_confirms
                        print(f"      - LONG ready: {long_ready}")
                        print(f"      - SHORT ready: {short_ready}")
        
        # 3. MOMENTUM стратегия
        if settings.enable_momentum_strategy:
            print(f"\n⚡ MOMENTUM стратегия:")
            momentum_signals = build_signals(df_ready, settings.strategy, use_momentum=True, use_liquidity=False)
            momentum_actionable = [s for s in momentum_signals if s.reason.startswith("momentum_") and s.action in (Action.LONG, Action.SHORT)]
            
            if momentum_actionable:
                print(f"  ✅ Найдено {len(momentum_actionable)} сигналов:")
                # Показываем все сигналы, отсортированные по времени
                for sig in sorted(momentum_actionable, key=lambda x: x.timestamp, reverse=True)[:10]:  # Последние 10
                    ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                    print(f"    • {ts_str} - {sig.action.value.upper()} @ ${sig.price:.2f} - {sig.reason}")
                
                # Показываем, когда был последний сигнал
                if momentum_actionable:
                    last_signal = max(momentum_actionable, key=lambda x: x.timestamp)
                    last_signal_ts = last_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_signal.timestamp, 'strftime') else str(last_signal.timestamp)
                    print(f"  📌 Последний MOMENTUM сигнал: {last_signal_ts}")
            else:
                print(f"  ⚠️ Нет сигналов LONG/SHORT")
                # Диагностика EMA
                ema_fast_1h = last_row.get('ema_fast_1h', np.nan)
                ema_slow_1h = last_row.get('ema_slow_1h', np.nan)
                price = last_row['close']
                
                if pd.notna([ema_fast_1h, ema_slow_1h]).all():
                    print(f"    💡 EMA Fast (1h): ${ema_fast_1h:.2f}, EMA Slow (1h): ${ema_slow_1h:.2f}, Price: ${price:.2f}")
                    print(f"      - EMA Fast > EMA Slow: {ema_fast_1h > ema_slow_1h} (бычий тренд)")
                    print(f"      - EMA Fast < EMA Slow: {ema_fast_1h < ema_slow_1h} (медвежий тренд)")
                    print(f"      - Price > EMA Fast: {price > ema_fast_1h}")
                    print(f"      - Price < EMA Fast: {price < ema_fast_1h}")
        
        # 4. LIQUIDITY стратегия
        if settings.enable_liquidity_sweep_strategy:
            print(f"\n💧 LIQUIDITY стратегия:")
            liquidity_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=True)
            liquidity_actionable = [s for s in liquidity_signals if s.reason.startswith("liquidity_") and s.action in (Action.LONG, Action.SHORT)]
            
            if liquidity_actionable:
                print(f"  ✅ Найдено {len(liquidity_actionable)} сигналов:")
                for sig in liquidity_actionable[-5:]:  # Последние 5
                    print(f"    • {sig.action.value} @ ${sig.price:.2f} - {sig.reason}")
            else:
                print(f"  ⚠️ Нет сигналов LONG/SHORT")
                # Диагностика Donchian
                donchian_upper = last_row.get('donchian_upper', np.nan)
                donchian_lower = last_row.get('donchian_lower', np.nan)
                price = last_row['close']
                
                if pd.notna([donchian_upper, donchian_lower]).all():
                    print(f"    💡 Donchian Upper: ${donchian_upper:.2f}, Donchian Lower: ${donchian_lower:.2f}, Price: ${price:.2f}")
                    print(f"      - Price > Donchian Upper: {price > donchian_upper} (пробой вверх)")
                    print(f"      - Price < Donchian Lower: {price < donchian_lower} (пробой вниз)")
        
        # 5. ML стратегия
        if settings.enable_ml_strategy and settings.ml_model_path:
            print(f"\n🤖 ML стратегия:")
            print(f"  • Model: {settings.ml_model_path}")
            try:
                ml_signals = build_ml_signals(
                    df_ready,
                    settings.ml_model_path,
                    settings.ml_confidence_threshold,
                    settings.ml_min_signal_strength,
                    settings.ml_stability_filter,
                )
                ml_actionable = [s for s in ml_signals if s.action in (Action.LONG, Action.SHORT)]
                
                # Показываем статистику по всем сигналам
                long_signals = [s for s in ml_actionable if s.action == Action.LONG]
                short_signals = [s for s in ml_actionable if s.action == Action.SHORT]
                hold_signals = [s for s in ml_signals if s.action == Action.HOLD]
                
                print(f"  📊 Статистика ML сигналов:")
                print(f"    • Всего предсказаний: {len(ml_signals)}")
                print(f"    • LONG: {len(long_signals)}")
                print(f"    • SHORT: {len(short_signals)}")
                print(f"    • HOLD: {len(hold_signals)}")
                print(f"    • Настройки: confidence_threshold={settings.ml_confidence_threshold}, min_signal_strength={settings.ml_min_signal_strength}, stability_filter={settings.ml_stability_filter}")
                
                if ml_actionable:
                    print(f"\n  ✅ Найдено {len(ml_actionable)} сигналов LONG/SHORT:")
                    # Показываем все сигналы, отсортированные по времени
                    for sig in sorted(ml_actionable, key=lambda x: x.timestamp, reverse=True)[:10]:  # Последние 10
                        ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                        print(f"    • {ts_str} - {sig.action.value.upper()} @ ${sig.price:.2f} - {sig.reason}")
                    
                    # Показываем, когда был последний сигнал
                    if ml_actionable:
                        last_signal = max(ml_actionable, key=lambda x: x.timestamp)
                        last_signal_ts = last_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_signal.timestamp, 'strftime') else str(last_signal.timestamp)
                        print(f"  📌 Последний ML сигнал: {last_signal_ts}")
                else:
                    print(f"\n  ⚠️ Нет сигналов LONG/SHORT после фильтрации")
                    # Показываем примеры HOLD сигналов с причинами
                    if hold_signals:
                        print(f"    💡 Примеры HOLD сигналов (первые 5):")
                        for sig in hold_signals[:5]:
                            ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                            print(f"      - {ts_str} - {sig.reason}")
                    
                    # ДЕТАЛЬНАЯ ДИАГНОСТИКА: Проверяем последние 10 свечей
                    print(f"\n  🔍 Детальная диагностика последних 10 свечей:")
                    try:
                        from bot.ml.strategy_ml import MLStrategy
                        ml_strategy = MLStrategy(
                            settings.ml_model_path,
                            settings.ml_confidence_threshold,
                            settings.ml_min_signal_strength,
                            settings.ml_stability_filter
                        )
                        
                        # Проверяем последние 10 свечей
                        last_10_rows = df_ready.tail(10)
                        for idx, (i, row) in enumerate(last_10_rows.iterrows()):
                            # Берем данные до этой свечи включительно
                            df_until_row = df_ready.loc[:i]
                            
                            try:
                                prediction, confidence = ml_strategy.predict(df_until_row, skip_feature_creation=False)
                                
                                # Преобразуем prediction в Action
                                if prediction == 1:
                                    action_str = "LONG"
                                elif prediction == -1:
                                    action_str = "SHORT"
                                else:
                                    action_str = "HOLD"
                                
                                # Проверяем, почему сигнал не прошел фильтры
                                timestamp = row.get('timestamp', pd.Timestamp.now())
                                if isinstance(timestamp, (int, float)):
                                    timestamp = pd.to_datetime(timestamp, unit='ms')
                                
                                price = row.get('close', 0)
                                
                                # Проверяем условия фильтрации
                                strength_thresholds = {
                                    "слабое": 0.0,
                                    "умеренное": 0.6,
                                    "среднее": 0.7,
                                    "сильное": 0.8,
                                    "очень_сильное": 0.9
                                }
                                min_strength = strength_thresholds.get(settings.ml_min_signal_strength, 0.6)
                                
                                reasons = []
                                if action_str == "HOLD":
                                    reasons.append("Модель предсказывает HOLD")
                                if confidence < settings.ml_confidence_threshold:
                                    reasons.append(f"Уверенность {confidence:.1%} < порога {settings.ml_confidence_threshold:.1%}")
                                if confidence < min_strength:
                                    reasons.append(f"Уверенность {confidence:.1%} < силы сигнала {min_strength:.1%} ({settings.ml_min_signal_strength})")
                                
                                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                                reason_str = ", ".join(reasons) if reasons else "Прошел все фильтры"
                                
                                print(f"    [{idx+1}] {ts_str} - {action_str} @ ${price:.2f} (confidence: {confidence:.1%})")
                                if reasons:
                                    print(f"         ❌ Отфильтрован: {reason_str}")
                                else:
                                    print(f"         ✅ Прошел фильтры, но не попал в список (возможно, stability_filter)")
                                
                            except Exception as e:
                                print(f"    [{idx+1}] Ошибка при проверке свечи: {e}")
                        
                    except Exception as e:
                        print(f"    ❌ Ошибка при детальной диагностике: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"  ❌ Ошибка ML стратегии: {e}")
                import traceback
                traceback.print_exc()
        elif settings.enable_ml_strategy:
            print(f"\n🤖 ML стратегия:")
            print(f"  ❌ ML стратегия включена, но модель не найдена")
            print(f"    💡 ml_model_path: {settings.ml_model_path}")
        
        # 6. SMC стратегия
        if settings.enable_smc_strategy:
            print(f"\n🟣 SMC стратегия (Smart Money Concepts):")
            try:
                # SMC требует много истории (минимум 500-1000 свечей для хороших результатов)
                if len(df_ready) >= 100:
                    # Включаем диагностику для поиска FVG
                    from bot.smc_strategy import find_fair_value_gaps
                    use_atr_filter = getattr(settings.strategy, 'smc_fvg_use_atr_filter', True)
                    atr_multiplier = getattr(settings.strategy, 'smc_fvg_atr_multiplier', 1.5)
                    
                    # Сначала ищем FVG с диагностикой
                    print(f"  🔍 Поиск FVG зон...")
                    fvg_zones_debug = find_fair_value_gaps(
                        df_ready,
                        min_gap_pct=settings.strategy.smc_fvg_min_gap_pct,
                        use_atr_filter=use_atr_filter,
                        atr_multiplier=atr_multiplier,
                        debug=True  # Включаем диагностику
                    )
                    
                    smc_signals = build_smc_signals(df_ready, settings.strategy, symbol=settings.symbol)
                    smc_actionable = [s for s in smc_signals if s.action in (Action.LONG, Action.SHORT)]
                    
                    # Показываем статистику
                    long_signals = [s for s in smc_actionable if s.action == Action.LONG]
                    short_signals = [s for s in smc_actionable if s.action == Action.SHORT]
                    
                    print(f"  📊 Статистика SMC сигналов:")
                    print(f"    • Всего сигналов: {len(smc_signals)}")
                    print(f"    • LONG: {len(long_signals)}")
                    print(f"    • SHORT: {len(short_signals)}")
                    print(f"    • Параметры:")
                    print(f"      - FVG min gap: {settings.strategy.smc_fvg_min_gap_pct*100:.3f}%")
                    use_atr_filter = getattr(settings.strategy, 'smc_fvg_use_atr_filter', True)
                    atr_multiplier = getattr(settings.strategy, 'smc_fvg_atr_multiplier', 1.5)
                    print(f"      - FVG ATR filter: {'ВКЛ' if use_atr_filter else 'ВЫКЛ'}")
                    if use_atr_filter:
                        print(f"      - FVG ATR multiplier: {atr_multiplier}x")
                    print(f"      - OB lookback: {settings.strategy.smc_ob_lookback}")
                    print(f"      - OB min move: {settings.strategy.smc_ob_min_move_pct*100:.3f}%")
                    print(f"      - Touch tolerance: {settings.strategy.smc_touch_tolerance_pct*100:.3f}%")
                    print(f"      - Max FVG age: {settings.strategy.smc_max_fvg_age_bars} bars")
                    print(f"      - Max OB age: {settings.strategy.smc_max_ob_age_bars} bars")
                    
                    if smc_actionable:
                        print(f"\n  ✅ Найдено {len(smc_actionable)} сигналов LONG/SHORT:")
                        # Показываем все сигналы, отсортированные по времени
                        for sig in sorted(smc_actionable, key=lambda x: x.timestamp, reverse=True)[:10]:  # Последние 10
                            ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                            print(f"    • {ts_str} - {sig.action.value.upper()} @ ${sig.price:.2f} - {sig.reason}")
                        
                        # Показываем, когда был последний сигнал
                        if smc_actionable:
                            last_signal = max(smc_actionable, key=lambda x: x.timestamp)
                            last_signal_ts = last_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_signal.timestamp, 'strftime') else str(last_signal.timestamp)
                            print(f"  📌 Последний SMC сигнал: {last_signal_ts}")
                    else:
                        print(f"\n  ⚠️ Нет сигналов LONG/SHORT")
                        print(f"    💡 SMC требует много истории (минимум 500-1000 свечей для хороших результатов)")
                        print(f"    💡 Текущее количество свечей: {len(df_ready)}")
                        print(f"    💡 Возможные причины:")
                        print(f"      - Недостаточно истории для поиска FVG/OB")
                        print(f"      - Нет активных FVG/OB зон, которые касается текущая цена")
                        print(f"      - Все найденные зоны уже закрыты (filled) или слишком старые")
                        
                        # Показываем примеры всех сигналов (включая те, что не прошли фильтры)
                        if smc_signals:
                            print(f"\n    📋 Всего найдено {len(smc_signals)} сигналов (включая неактивные):")
                            for sig in smc_signals[:5]:  # Первые 5
                                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                                print(f"      - {ts_str} - {sig.action.value.upper()} @ ${sig.price:.2f} - {sig.reason}")
                else:
                    print(f"  ⚠️ Недостаточно данных для SMC стратегии")
                    print(f"    💡 Требуется минимум 100 свечей, получено: {len(df_ready)}")
                    print(f"    💡 Для хороших результатов рекомендуется 500-1000 свечей")
            except Exception as e:
                print(f"  ❌ Ошибка SMC стратегии: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Ошибка при проверке {symbol}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция."""
    print("="*80)
    print("🔍 ДИАГНОСТИКА ГЕНЕРАЦИИ СИГНАЛОВ ПО ВСЕМ СТРАТЕГИЯМ")
    print("="*80)
    
    # Загружаем настройки
    settings = load_settings()
    
    if not settings:
        print("❌ Не удалось загрузить настройки")
        return
    
    # Получаем активные символы
    active_symbols = getattr(settings, 'active_symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    if not active_symbols:
        active_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print(f"\n📋 Активные символы: {', '.join(active_symbols)}")
    print(f"📋 Приоритет стратегий: {settings.strategy_priority}")
    
    # Проверяем каждый символ
    for symbol in active_symbols:
        # Временно переопределяем symbol в settings для этого символа
        original_symbol = settings.symbol
        settings.symbol = symbol
        settings.primary_symbol = symbol
        
        # Для ML стратегии нужно найти модель для этого символа
        if settings.enable_ml_strategy:
            from pathlib import Path
            models_dir = Path(__file__).parent / "ml_models"
            if models_dir.exists():
                # Ищем модель для символа
                model_type_preference = getattr(settings, 'ml_model_type_for_all', None)
                found_model = None
                
                if model_type_preference:
                    pattern = f"{model_type_preference}_{symbol}_*.pkl"
                    for model_file in sorted(models_dir.glob(pattern), reverse=True):
                        if model_file.is_file():
                            found_model = str(model_file)
                            break
                else:
                    # Авто-выбор: ensemble > rf > xgb
                    for model_type in ["ensemble", "rf", "xgb"]:
                        pattern = f"{model_type}_{symbol}_*.pkl"
                        for model_file in sorted(models_dir.glob(pattern), reverse=True):
                            if model_file.is_file():
                                found_model = str(model_file)
                                break
                        if found_model:
                            break
                
                if found_model:
                    settings.ml_model_path = found_model
                else:
                    print(f"  ⚠️ ML модель не найдена для {symbol}")
                    settings.ml_model_path = None
        
        check_strategies_for_symbol(symbol, settings)
        
        # Восстанавливаем оригинальный symbol
        settings.symbol = original_symbol
        settings.primary_symbol = original_symbol
    
    print(f"\n{'='*80}")
    print("✅ Диагностика завершена")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
