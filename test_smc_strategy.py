"""
Простой тест для проверки SMC стратегии.
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, Action
from bot.smc_strategy import build_smc_signals, SMCStrategy
from bot.live import _timeframe_to_bybit_interval
import pandas as pd

def test_smc_strategy():
    """Тестирует SMC стратегию на реальных данных."""
    print("="*80)
    print("🟣 ТЕСТ SMC СТРАТЕГИИ")
    print("="*80)
    
    # Загружаем настройки
    settings = load_settings()
    if not settings:
        print("❌ Не удалось загрузить настройки")
        return
    
    # Проверяем, включена ли SMC стратегия
    if not settings.enable_smc_strategy:
        print("⚠️ SMC стратегия выключена в настройках")
        print("   Включите её в админке или в .env файле (ENABLE_SMC_STRATEGY=true)")
        return
    
    # Выбираем символ для теста
    symbol = settings.symbol or "BTCUSDT"
    print(f"\n📊 Тестируем на символе: {symbol}")
    
    try:
        # Создаем клиент
        client = BybitClient(settings.api)
        
        # Получаем данные (больше свечей для SMC)
        interval = _timeframe_to_bybit_interval(settings.timeframe)
        print(f"\n📈 Получение данных ({interval})...")
        print(f"   ⚠️ SMC требует много истории (минимум 500-1000 свечей)")
        
        klines = client.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=1000  # Берем больше для SMC
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
        df_raw = df_raw.iloc[::-1].reset_index(drop=True)
        df_raw.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df_raw = df_raw.astype({
            "open": float, "high": float, "low": float, "close": float,
            "volume": float, "turnover": float
        })
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"].astype(int), unit="ms")
        
        print(f"✅ Получено {len(df_raw)} свечей")
        
        # Показываем диапазон данных
        if not df_raw.empty:
            first_ts = df_raw.iloc[0]['timestamp']
            last_ts = df_raw.iloc[-1]['timestamp']
            print(f"   📅 Период: {first_ts.strftime('%Y-%m-%d %H:%M')} - {last_ts.strftime('%Y-%m-%d %H:%M')}")
        
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
        
        # Обогащаем данные
        df_ready = enrich_for_strategy(df_ready, settings.strategy)
        
        print(f"✅ Подготовлено {len(df_ready)} строк")
        
        # Текущая цена
        last_row = df_ready.iloc[-1]
        current_price = last_row['close']
        print(f"\n💰 Текущая цена: ${current_price:.2f}")
        
        # Параметры SMC
        print(f"\n⚙️ Параметры SMC стратегии:")
        print(f"   • FVG min gap: {settings.strategy.smc_fvg_min_gap_pct*100:.3f}%")
        print(f"   • OB lookback: {settings.strategy.smc_ob_lookback}")
        print(f"   • OB min move: {settings.strategy.smc_ob_min_move_pct*100:.3f}%")
        print(f"   • Touch tolerance: {settings.strategy.smc_touch_tolerance_pct*100:.3f}%")
        print(f"   • Max FVG age: {settings.strategy.smc_max_fvg_age_bars} bars")
        print(f"   • Max OB age: {settings.strategy.smc_max_ob_age_bars} bars")
        
        # 1. Ищем FVG зоны
        print(f"\n🔍 Поиск Fair Value Gaps (FVG)...")
        strategy = SMCStrategy(settings.strategy)
        
        # Подготовка данных для NumPy как в самом классе
        highs = df_ready['high'].values
        lows = df_ready['low'].values
        closes = df_ready['close'].values
        opens = df_ready['open'].values
        if 'timestamp' in df_ready.columns:
            times = df_ready['timestamp'].values
        else:
            times = df_ready.index.values

        fvg_zones = strategy._find_fvg(df_ready, highs, lows, opens, closes, times)
        print(f"   ✅ Найдено {len(fvg_zones)} FVG зон")
        
        bullish_fvg = [fvg for fvg in fvg_zones if fvg.direction == "bullish"]
        bearish_fvg = [fvg for fvg in fvg_zones if fvg.direction == "bearish"]
        print(f"      • Бычьих: {len(bullish_fvg)}")
        print(f"      • Медвежьих: {len(bearish_fvg)}")
        
        # Показываем последние 5 FVG
        if fvg_zones:
            print(f"\n   📋 Последние 5 FVG зон:")
            for fvg in sorted(fvg_zones, key=lambda x: x.timestamp, reverse=True)[:5]:
                ts_str = fvg.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(fvg.timestamp, 'strftime') else str(fvg.timestamp)
                print(f"      • {ts_str} - {fvg.direction.upper()}: ${fvg.lower:.2f} - ${fvg.upper:.2f}")
        
        # 2. Ищем Order Blocks
        print(f"\n🔍 Поиск Order Blocks (OB)...")
        order_blocks = strategy._find_ob(df_ready, highs, lows, opens, closes, times)
        print(f"   ✅ Найдено {len(order_blocks)} Order Blocks")
        
        bullish_ob = [ob for ob in order_blocks if ob.direction == "bullish"]
        bearish_ob = [ob for ob in order_blocks if ob.direction == "bearish"]
        print(f"      • Бычьих: {len(bullish_ob)}")
        print(f"      • Медвежьих: {len(bearish_ob)}")
        
        # Показываем последние 5 OB
        if order_blocks:
            print(f"\n   📋 Последние 5 Order Blocks:")
            for ob in sorted(order_blocks, key=lambda x: x.timestamp, reverse=True)[:5]:
                ts_str = ob.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(ob.timestamp, 'strftime') else str(ob.timestamp)
                print(f"      • {ts_str} - {ob.direction.upper()}: ${ob.lower:.2f} - ${ob.upper:.2f}")
        
        # 3. Генерируем сигналы
        print(f"\n🎯 Генерация сигналов...")
        signals = build_smc_signals(df_ready, settings.strategy, symbol=settings.symbol)
        
        long_signals = [s for s in signals if s.action == Action.LONG]
        short_signals = [s for s in signals if s.action == Action.SHORT]
        
        print(f"   ✅ Сгенерировано {len(signals)} сигналов")
        print(f"      • LONG: {len(long_signals)}")
        print(f"      • SHORT: {len(short_signals)}")
        
        if signals:
            print(f"\n   📋 Последние 10 сигналов:")
            for sig in sorted(signals, key=lambda x: x.timestamp, reverse=True)[:10]:
                ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                print(f"      • {ts_str} - {sig.action.value.upper()} @ ${sig.price:.2f}")
                print(f"        Причина: {sig.reason}")
                if hasattr(sig, 'stop_loss') and sig.stop_loss:
                    print(f"        SL: ${sig.stop_loss:.2f} | TP: ${sig.take_profit:.2f} (RR: {settings.strategy.smc_rr_ratio})")
            
            # Последний сигнал
            last_signal = max(signals, key=lambda x: x.timestamp)
            last_ts = last_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_signal.timestamp, 'strftime') else str(last_signal.timestamp)
            print(f"\n   📌 Последний сигнал: {last_ts} - {last_signal.action.value.upper()} @ ${last_signal.price:.2f}")
        else:
            print(f"\n   ⚠️ Сигналов не найдено")
            print(f"      💡 Возможные причины:")
            print(f"         - Нет активных FVG/OB зон, которые касается текущая цена")
            print(f"         - Все найденные зоны уже закрыты или слишком старые")
            print(f"         - Текущая цена не попадает в зоны с учетом tolerance")
            print(f"\n      💡 Текущая цена: ${current_price:.2f}")
            if bullish_fvg:
                print(f"      💡 Ближайший бычий FVG: ${bullish_fvg[0].lower:.2f} - ${bullish_fvg[0].upper:.2f}")
            if bearish_fvg:
                print(f"      💡 Ближайший медвежий FVG: ${bearish_fvg[0].lower:.2f} - ${bearish_fvg[0].upper:.2f}")
        
        print(f"\n{'='*80}")
        print("✅ Тест завершен")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smc_strategy()
