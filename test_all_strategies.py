"""
Универсальный скрипт для тестирования всех стратегий на исторических данных.
Показывает статистику: количество сигналов, винрейт, общий PnL и другие метрики.
"""
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, build_signals, Action, generate_range_signal, generate_trend_signal, generate_momentum_breakout_signal, detect_market_phase, MarketPhase, Signal, Bias
from bot.smc_strategy import build_smc_signals
from bot.ict_strategy import build_ict_signals
from bot.ml.strategy_ml import build_ml_signals
from bot.simulation import Simulator, Trade


def test_strategy(
    strategy_name: str,
    symbol: str,
    days_back: int = 30,
    settings=None
):
    """
    Тестирует стратегию на исторических данных.
    
    Args:
        strategy_name: Название стратегии (trend, flat, momentum, liquidity, smc, ict, ml)
        symbol: Торговая пара (например, BTCUSDT)
        days_back: Количество дней для тестирования
        settings: Настройки бота
    """
    print("=" * 80)
    print(f"[TEST] Тестирование {strategy_name.upper()} стратегии для {symbol}")
    print("=" * 80)
    
    if settings is None:
        settings = load_settings()
    
    # Переопределяем символ для теста
    settings.symbol = symbol
    settings.primary_symbol = symbol
    
    # Настраиваем стратегию
    if strategy_name == "trend":
        settings.enable_trend_strategy = True
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = False
        use_momentum = False
        use_liquidity = False
    elif strategy_name == "flat":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = True
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = False
        use_momentum = False
        use_liquidity = False
    elif strategy_name == "momentum":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = True
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = False
        use_momentum = True
        use_liquidity = False
    elif strategy_name == "liquidity":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = True
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = False
        use_momentum = False
        use_liquidity = True
    elif strategy_name == "smc":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = True
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = False
        use_momentum = False
        use_liquidity = False
    elif strategy_name == "ict":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = True
        settings.enable_ml_strategy = False
        use_momentum = False
        use_liquidity = False
    elif strategy_name == "ml":
        settings.enable_trend_strategy = False
        settings.enable_flat_strategy = False
        settings.enable_momentum_strategy = False
        settings.enable_liquidity_sweep_strategy = False
        settings.enable_smc_strategy = False
        settings.enable_ict_strategy = False
        settings.enable_ml_strategy = True
        use_momentum = False
        use_liquidity = False
    else:
        print(f"[ERROR] Неизвестная стратегия: {strategy_name}")
        return
    
    print(f"\n⚙️  Параметры:")
    print(f"   - Символ: {symbol}")
    print(f"   - Период: {days_back} дней")
    print(f"   - Таймфрейм: {settings.timeframe}")
    
    # Получаем данные
    print(f"\n[DATA] Собираем данные за последние {days_back} дней...")
    client = BybitClient(api=settings.api)
    
    # Вычисляем количество свечей (15 минут = 96 свечей в день)
    candles_needed = days_back * 96
    if candles_needed > 1000:
        candles_needed = 1000  # Bybit ограничение
    
    # Преобразуем timeframe в формат Bybit (15 -> "15", "1h" -> "60")
    interval = str(settings.timeframe) if isinstance(settings.timeframe, int) else settings.timeframe
    df = client.get_kline_df(symbol=symbol, interval=interval, limit=candles_needed)
    if df is None or len(df) == 0:
        print("[ERROR] Не удалось получить данные")
        return
    
    print(f"[OK] Получено {len(df)} свечей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")
    
    # Подготавливаем данные
    print(f"\n🔧 Вычисляем индикаторы...")
    df = prepare_with_indicators(
        df,
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
        ema_timeframe=settings.strategy.momentum_ema_timeframe
    )
    df = enrich_for_strategy(df, settings.strategy)
    print(f"[OK] Данные подготовлены: {len(df)} свечей")
    
    # Генерируем сигналы
    print(f"\n🤖 Генерируем {strategy_name.upper()} сигналы...")
    print("-" * 80)
    
    if strategy_name == "flat":
        # Для Flat Strategy принудительно используем generate_range_signal
        # независимо от фазы рынка (чтобы протестировать стратегию в любых условиях)
        signals = []
        position_bias = None
        entry_price = None
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            sig = generate_range_signal(row, position_bias, settings.strategy, entry_price)
            signals.append(sig)
            # Обновляем состояние позиции
            # ВАЖНО: сигналы закрытия (range_sl_fixed) не должны открывать новые позиции
            # Они только закрывают существующие позиции
            if sig.reason == "range_sl_fixed":
                # Это сигнал закрытия - закрываем позицию и сбрасываем состояние
                if position_bias is not None:
                    position_bias = None
                    entry_price = None
            elif sig.action == Action.LONG:
                if position_bias is None:
                    position_bias = Bias.LONG
                    entry_price = sig.price
                elif position_bias == Bias.SHORT:
                    position_bias = Bias.LONG
                    entry_price = sig.price
            elif sig.action == Action.SHORT:
                if position_bias is None:
                    position_bias = Bias.SHORT
                    entry_price = sig.price
                elif position_bias == Bias.LONG:
                    position_bias = Bias.SHORT
                    entry_price = sig.price
    elif strategy_name == "trend":
        # Для Trend Strategy принудительно используем generate_trend_signal
        # только когда рынок в тренде (ADX > threshold)
        signals = []
        position_bias = None
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            market_phase = detect_market_phase(row, settings.strategy)
            if market_phase == MarketPhase.TREND:
                sig = generate_trend_signal(row, position_bias, settings.strategy)
            else:
                sig = Signal(timestamp=row.name, action=Action.HOLD, reason="trend_not_in_trend_phase", price=row["close"])
            signals.append(sig)
            # Обновляем состояние позиции
            if sig.action == Action.LONG:
                if position_bias is None:
                    position_bias = Bias.LONG
                elif position_bias == Bias.SHORT:
                    position_bias = Bias.LONG
            elif sig.action == Action.SHORT:
                if position_bias is None:
                    position_bias = Bias.SHORT
                elif position_bias == Bias.LONG:
                    position_bias = Bias.SHORT
    elif strategy_name == "smc":
        signals = build_smc_signals(df, settings.strategy, symbol=symbol)
    elif strategy_name == "ict":
        signals = build_ict_signals(df, settings.strategy, symbol=symbol)
    elif strategy_name == "momentum":
        # Для Momentum Strategy принудительно используем generate_momentum_breakout_signal
        # независимо от фазы рынка (чтобы протестировать стратегию в любых условиях)
        signals = []
        position_bias = None
        prev_ema_fast = None
        prev_ema_slow = None
        ema_timeframe = settings.strategy.momentum_ema_timeframe
        ema_fast_col = f"ema_fast_{ema_timeframe}"
        ema_slow_col = f"ema_slow_{ema_timeframe}"
        
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            ema_fast = row.get(ema_fast_col, np.nan)
            ema_slow = row.get(ema_slow_col, np.nan)
            
            # Проверяем пересечение EMA или разошедшиеся EMA
            if idx > 0 and (prev_ema_fast is not None and prev_ema_slow is not None and 
                np.isfinite([ema_fast, ema_slow, prev_ema_fast, prev_ema_slow]).all()):
                # Пересечение или разошедшиеся EMA
                ema_cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                ema_already_bullish = ema_fast > ema_slow and (ema_fast - ema_slow) / ema_slow > 0.001
                ema_cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
                ema_already_bearish = ema_fast < ema_slow and (ema_slow - ema_fast) / ema_slow > 0.001
                
                if (ema_cross_up or (ema_already_bullish and position_bias is None)) or \
                   (ema_cross_down or (ema_already_bearish and position_bias is None)):
                    sig = generate_momentum_breakout_signal(row, position_bias, settings.strategy)
                else:
                    sig = Signal(timestamp=row.name, action=Action.HOLD, reason="momentum_no_ema_setup", price=row["close"])
            else:
                sig = Signal(timestamp=row.name, action=Action.HOLD, reason="momentum_no_data", price=row["close"])
            
            signals.append(sig)
            
            # Обновляем состояние позиции
            if sig.action == Action.LONG:
                if position_bias is None:
                    position_bias = Bias.LONG
                elif position_bias == Bias.SHORT:
                    position_bias = Bias.LONG
            elif sig.action == Action.SHORT:
                if position_bias is None:
                    position_bias = Bias.SHORT
                elif position_bias == Bias.LONG:
                    position_bias = Bias.SHORT
            
            # Сохраняем EMA для следующей итерации
            prev_ema_fast = ema_fast if np.isfinite(ema_fast) else prev_ema_fast
            prev_ema_slow = ema_slow if np.isfinite(ema_slow) else prev_ema_slow
    elif strategy_name == "ml":
        # Находим модель для символа
        model_path = settings.ml_model_path
        if not model_path or not os.path.exists(model_path):
            # Пытаемся найти модель автоматически
            model_dir = Path(__file__).parent / "ml_models"
            model_files = list(model_dir.glob(f"*{symbol}*.pkl"))
            if model_files:
                model_path = str(model_files[0])
            else:
                print(f"[ERROR] Не найдена ML модель для {symbol}")
                return
        
        signals = build_ml_signals(
            df,
            model_path,
            confidence_threshold=settings.ml_confidence_threshold,
            min_signal_strength=settings.ml_min_signal_strength,
            stability_filter=settings.ml_stability_filter,
            leverage=settings.leverage,
            target_profit_pct_margin=getattr(settings, 'ml_target_profit_pct_margin', 25.0),
            max_loss_pct_margin=getattr(settings, 'ml_max_loss_pct_margin', 10.0),
        )
    else:
        signals = build_signals(df, settings.strategy, use_momentum=use_momentum, use_liquidity=use_liquidity)
    
    # Фильтруем только actionable сигналы
    actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
    
    print(f"[OK] Сгенерировано {len(signals)} сигналов")
    print(f"   Actionable (LONG/SHORT): {len(actionable_signals)}")
    print(f"   HOLD: {len(signals) - len(actionable_signals)}")
    
    if len(actionable_signals) == 0:
        print("\n[WARNING] Нет actionable сигналов для тестирования")
        return
    
    # Показываем первые 10 сигналов
    print("\n" + "=" * 80)
    print("[SIGNALS] ПЕРВЫЕ 10 СИГНАЛОВ:")
    print("=" * 80)
    for i, sig in enumerate(actionable_signals[:10], 1):
        action_mark = "[LONG]" if sig.action == Action.LONG else "[SHORT]"
        print(f" {i}. {action_mark} [{sig.timestamp}] {sig.action.value.upper():5s} @ ${sig.price:,.2f} - {sig.reason}")
    
    if len(actionable_signals) > 10:
        print(f"\n... и еще {len(actionable_signals) - 10} сигналов")
    
    # Симулируем торговлю
    print("\n" + "=" * 80)
    print("[SIMULATION] СИМУЛЯЦИЯ ТОРГОВЛИ")
    print("=" * 80)
    
    simulator = Simulator(settings)
    
    # Проходим по всем свечам и применяем сигналы
    signal_dict = {s.timestamp: s for s in signals}

    for idx, (timestamp, row) in enumerate(df.iterrows()):
        # Проверяем, есть ли сигнал на этой свече
        if timestamp in signal_dict:
            sig = signal_dict[timestamp]
            # Обрабатываем сигналы закрытия отдельно
            if sig.reason == "range_sl_fixed" and simulator.position:
                # Закрываем позицию по SL из сигнала
                simulator._close(sig.price, f"{strategy_name}_sl_hit", timestamp)
            elif sig.action != Action.HOLD or sig.reason != "range_sl_fixed":
                # Обрабатываем только входные сигналы (не HOLD и не range_sl_fixed)
                simulator.on_signal(sig)
        
        # Закрываем позицию по TP/SL (если есть)
        if simulator.position:
            current_price = row['close']
            high = row.get('high', current_price)
            low = row.get('low', current_price)
            entry_price = simulator.position.avg_price
            
            # Улучшенный TP/SL с проверкой high/low свечи
            if simulator.position.side.value == "long":
                # TP: проверяем high свечи
                tp_price = entry_price * (1 + settings.risk.take_profit_pct)
                if high >= tp_price:
                    simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                    continue
                # SL: проверяем low свечи
                sl_price = entry_price * (1 - settings.risk.stop_loss_pct)
                if low <= sl_price:
                    simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                    continue
                # Trailing Stop: если прибыль > 0.5%, подтягиваем SL к безубытку
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct > 0.005:  # 0.5% прибыли
                    breakeven_sl = entry_price * 1.001  # SL на 0.1% выше входа
                    if low <= breakeven_sl:
                        simulator._close(breakeven_sl, f"{strategy_name}_trailing_stop", timestamp)
                        continue
            else:  # SHORT
                # TP: проверяем low свечи
                tp_price = entry_price * (1 - settings.risk.take_profit_pct)
                if low <= tp_price:
                    simulator._close(tp_price, f"{strategy_name}_tp_hit", timestamp)
                    continue
                # SL: проверяем high свечи
                sl_price = entry_price * (1 + settings.risk.stop_loss_pct)
                if high >= sl_price:
                    simulator._close(sl_price, f"{strategy_name}_sl_hit", timestamp)
                    continue
                # Trailing Stop: если прибыль > 0.5%, подтягиваем SL к безубытку
                profit_pct = (entry_price - current_price) / entry_price
                if profit_pct > 0.005:  # 0.5% прибыли
                    breakeven_sl = entry_price * 0.999  # SL на 0.1% ниже входа
                    if high >= breakeven_sl:
                        simulator._close(breakeven_sl, f"{strategy_name}_trailing_stop", timestamp)
                        continue
    
    # Закрываем последнюю позицию, если она открыта
    if simulator.position:
        last_price = df.iloc[-1]['close']
        simulator._close(last_price, f"{strategy_name}_end_of_data", df.index[-1])
    
    # Статистика
    trades = simulator.trades
    if len(trades) == 0:
        print("\n⚠️  Нет сделок для анализа")
        return
    
    profitable = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl < 0]
    
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / len(trades) if trades else 0
    avg_win = sum(t.pnl for t in profitable) / len(profitable) if profitable else 0
    avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
    max_win = max((t.pnl for t in trades), default=0)
    max_loss = min((t.pnl for t in trades), default=0)
    
    win_rate = len(profitable) / len(trades) * 100 if trades else 0
    
    # Profit Factor
    total_wins = sum(t.pnl for t in profitable) if profitable else 0
    total_losses = abs(sum(t.pnl for t in losing)) if losing else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    print(f"\n[STATS] СТАТИСТИКА:")
    print(f"   Всего сделок: {len(trades)}")
    print(f"   Прибыльных: {len(profitable)} ({win_rate:.1f}%)")
    print(f"   Убыточных: {len(losing)}")
    
    print(f"\n[PNL] PnL:")
    print(f"   Общий PnL: {total_pnl:+.2f} USDT")
    print(f"   Средний PnL на сделку: {avg_pnl:+.2f} USDT")
    print(f"   Средний выигрыш: {avg_win:+.2f} USDT")
    print(f"   Средний проигрыш: {avg_loss:+.2f} USDT")
    print(f"   Максимальный выигрыш: {max_win:+.2f} USDT")
    print(f"   Максимальный проигрыш: {max_loss:+.2f} USDT")
    print(f"   Profit Factor: {profit_factor:.2f}")
    
    # Последние 10 сделок
    print("\n" + "=" * 80)
    print("📋 ПОСЛЕДНИЕ 10 СДЕЛОК:")
    print("=" * 80)
    for i, trade in enumerate(trades[-10:], 1):
        emoji = "[+]" if trade.pnl > 0 else "[-]"
        side_str = trade.side.value.upper()
        print(f" {i}. {emoji} {side_str:5s} | Entry: ${trade.entry_price:,.2f} @ {trade.entry_time} | "
              f"Exit: ${trade.exit_price:,.2f} @ {trade.exit_time} | "
              f"PnL: {trade.pnl:+.2f} USDT | Reason: {trade.exit_reason}")
    
    print("\n" + "=" * 80)
    print("[DONE] ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Тестирование стратегий на исторических данных")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["trend", "flat", "momentum", "liquidity", "smc", "ict", "ml"],
                       help="Название стратегии для тестирования")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Торговая пара (по умолчанию: BTCUSDT)")
    parser.add_argument("--days", type=int, default=30,
                       help="Количество дней для тестирования (по умолчанию: 30)")
    
    args = parser.parse_args()
    
    test_strategy(args.strategy, args.symbol, args.days)


if __name__ == "__main__":
    main()
