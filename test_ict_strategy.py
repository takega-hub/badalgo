"""
Скрипт для тестирования ICT Silver Bullet стратегии на исторических данных.
Показывает статистику: количество сигналов, винрейт, общий PnL и другие метрики.
"""
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy, Action, Bias
from bot.ict_strategy import build_ict_signals
from bot.simulation import Simulator, Trade


def extract_tp_sl_from_reason(reason: str) -> tuple:
    """
    Извлекает TP и SL из reason сигнала.
    Формат: ict_silver_bullet_long_fvg_reteest_sl_92000.00_tp_92500.00
    
    Returns:
        Tuple (sl_price, tp_price) или (None, None) если не найдено
    """
    sl_match = re.search(r'sl_([\d.]+)', reason)
    tp_match = re.search(r'tp_([\d.]+)', reason)
    
    sl_price = float(sl_match.group(1)) if sl_match else None
    tp_price = float(tp_match.group(1)) if tp_match else None
    
    return sl_price, tp_price


class ICTSimulator(Simulator):
    """
    Расширенный симулятор для ICT стратегии с поддержкой TP/SL из сигналов.
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self.open_trades: list = []  # Список открытых сделок с TP/SL
    
    def on_signal_with_tp_sl(self, sig, df: pd.DataFrame):
        """
        Обрабатывает сигнал с учетом TP/SL из reason.
        
        Args:
            sig: Signal объект
            df: DataFrame с данными для проверки TP/SL
        """
        if sig.action == Action.HOLD:
            return
        
        # Извлекаем TP/SL из reason
        sl_price, tp_price = extract_tp_sl_from_reason(sig.reason)
        
        if not sl_price or not tp_price:
            # Если TP/SL не найдены, используем стандартную логику
            self.on_signal(sig)
            return
        
        # Определяем направление
        side = Bias.LONG if sig.action == Action.LONG else Bias.SHORT
        
        # Если есть открытая позиция в противоположном направлении - закрываем
        if self.position and self.position.side != side:
            self._close(sig.price, f"flip_for_{sig.reason}", sig.timestamp)
        
        # Открываем новую позицию
        if not self.position:
            self._open(side, sig.price, self.settings.risk.base_order_usd, sig.timestamp, sig.reason)
            
            # Сохраняем информацию о TP/SL
            self.open_trades.append({
                'entry_price': sig.price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'side': side,
                'entry_time': sig.timestamp,
                'entry_reason': sig.reason
            })
        elif self.position.side == side:
            # Добавляем к существующей позиции
            self._open(side, sig.price, self.settings.risk.add_order_usd, sig.timestamp, sig.reason)
            # Обновляем TP/SL для последней сделки
            if self.open_trades:
                self.open_trades[-1]['sl_price'] = sl_price
                self.open_trades[-1]['tp_price'] = tp_price
    
    def check_tp_sl(self, df: pd.DataFrame) -> int:
        """
        Проверяет достижение TP/SL для открытых позиций.
        
        Args:
            df: DataFrame с данными (последняя свеча используется для проверки)
            
        Returns:
            Количество закрытых позиций
        """
        if not self.position or not self.open_trades:
            return 0
        
        closed_count = 0
        current_idx = len(df) - 1
        current_candle = df.iloc[current_idx]
        high = current_candle['high']
        low = current_candle['low']
        close = current_candle['close']
        timestamp = df.index[current_idx]
        
        # Получаем последнюю открытую сделку
        open_trade = self.open_trades[-1]
        sl_price = open_trade['sl_price']
        tp_price = open_trade['tp_price']
        side = self.position.side
        
        # Проверяем достижение TP/SL
        if side == Bias.LONG:
            # LONG: SL ниже, TP выше
            if low <= sl_price:
                # Сработал SL
                self._close(sl_price, f"ict_sl_hit_{sl_price:.2f}", timestamp)
                self.open_trades.pop()
                closed_count += 1
            elif high >= tp_price:
                # Сработал TP
                self._close(tp_price, f"ict_tp_hit_{tp_price:.2f}", timestamp)
                self.open_trades.pop()
                closed_count += 1
        else:  # SHORT
            # SHORT: SL выше, TP ниже
            if high >= sl_price:
                # Сработал SL
                self._close(sl_price, f"ict_sl_hit_{sl_price:.2f}", timestamp)
                self.open_trades.pop()
                closed_count += 1
            elif low <= tp_price:
                # Сработал TP
                self._close(tp_price, f"ict_tp_hit_{tp_price:.2f}", timestamp)
                self.open_trades.pop()
                closed_count += 1
        
        return closed_count
    
    def run_with_tp_sl(self, candles: pd.DataFrame, signals: list) -> dict:
        """
        Запускает симуляцию с проверкой TP/SL на каждой свече.
        
        Args:
            candles: DataFrame с данными
            signals: Список сигналов
            
        Returns:
            Словарь с результатами
        """
        # Сортируем сигналы по времени
        signals_sorted = sorted(signals, key=lambda s: s.timestamp)
        
        # Индекс текущего сигнала
        signal_idx = 0
        
        # Проходим по всем свечам
        for i in range(200, len(candles)):  # Начинаем с 200 для индикаторов
            current_candle = candles.iloc[i]
            current_time = candles.index[i]
            
            # Сначала проверяем TP/SL для открытой позиции (важно делать это до обработки новых сигналов)
            if self.position:
                self.check_tp_sl(candles.iloc[:i+1])
            
            # Обрабатываем все сигналы до текущего времени
            while signal_idx < len(signals_sorted):
                sig = signals_sorted[signal_idx]
                # Преобразуем timestamp сигнала для сравнения
                sig_time = sig.timestamp
                if not isinstance(sig_time, pd.Timestamp):
                    sig_time = pd.to_datetime(sig_time)
                
                # Конвертируем в тот же timezone что и индекс candles
                if candles.index.tzinfo is not None:
                    if sig_time.tzinfo is None:
                        sig_time = sig_time.tz_localize('UTC')
                    else:
                        sig_time = sig_time.tz_convert(candles.index.tz)
                
                if sig_time <= current_time:
                    self.on_signal_with_tp_sl(sig, candles.iloc[:i+1])
                    signal_idx += 1
                else:
                    break
        
        # Обрабатываем оставшиеся сигналы
        while signal_idx < len(signals_sorted):
            sig = signals_sorted[signal_idx]
            self.on_signal_with_tp_sl(sig, candles)
            signal_idx += 1
        
        # Проверяем TP/SL для оставшейся позиции на последней свече
        if self.position:
            self.check_tp_sl(candles)
        
        # Закрываем оставшуюся позицию в конце
        if self.position:
            last_idx = candles.index[-1]
            last_price = candles["close"].iloc[-1]
            self._close(last_price, "end_of_backtest", last_idx)
        
        # Подсчитываем статистику
        return self._calculate_stats()
    
    def _calculate_stats(self) -> dict:
        """Подсчитывает статистику по сделкам."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "profit_factor": 0.0,
                "trades": []
            }
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = total_pnl / len(self.trades) if self.trades else 0.0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        max_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        max_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0.0
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "profit_factor": profit_factor,
            "trades": self.trades
        }


def test_ict_strategy(
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    enable_london: bool = True,
    enable_ny: bool = True,
):
    """
    Тестирует ICT Silver Bullet стратегию на исторических данных.
    
    Args:
        symbol: Торговая пара
        days_back: Количество дней назад для тестирования
        enable_london: Включить Лондонскую сессию
        enable_ny: Включить Нью-Йоркскую сессию
    """
    print("=" * 80)
    print(f"🧪 Тестирование ICT Silver Bullet стратегии для {symbol}")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Включаем ICT стратегию
    settings.enable_ict_strategy = True
    settings.strategy.ict_enable_london_session = enable_london
    settings.strategy.ict_enable_ny_session = enable_ny
    
    print(f"⚙️  Параметры:")
    print(f"   - Лондонская сессия: {'Включена' if enable_london else 'Выключена'}")
    print(f"   - Нью-Йоркская сессия: {'Включена' if enable_ny else 'Выключена'}")
    print(f"   - Alligator: Jaw={settings.strategy.ict_alligator_jaw_period}, "
          f"Teeth={settings.strategy.ict_alligator_teeth_period}, "
          f"Lips={settings.strategy.ict_alligator_lips_period}")
    print(f"   - FVG max age: {settings.strategy.ict_fvg_max_age_bars} баров")
    print(f"   - R:R ratio: {settings.strategy.ict_rr_ratio}")
    print()
    
    # Получаем исторические данные
    print(f"📊 Собираем данные за последние {days_back} дней...")
    client = BybitClient(settings.api)
    
    # Вычисляем количество свечей (15 минут)
    # 1 день = 24 часа * 4 свечи в час = 96 свечей
    limit = days_back * 96 + 200  # +200 для индикаторов
    
    df_raw = client.get_kline_df(symbol=symbol, interval="15", limit=limit)
    
    if df_raw.empty:
        print(f"❌ Не удалось получить данные для {symbol}")
        return
    
    print(f"✅ Получено {len(df_raw)} свечей")
    print(f"   Период: {df_raw.index[0]} - {df_raw.index[-1]}")
    print()
    
    # Подготавливаем данные с индикаторами
    print("🔧 Вычисляем индикаторы...")
    df_ind = prepare_with_indicators(
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
        ema_timeframe=settings.strategy.momentum_ema_timeframe,
    )
    
    df_ready = enrich_for_strategy(df_ind, settings.strategy)
    
    print(f"✅ Данные подготовлены: {len(df_ready)} свечей")
    print()
    
    # Генерируем сигналы
    print("🤖 Генерируем ICT сигналы...")
    print("-" * 80)
    
    try:
        signals = build_ict_signals(df_ready, settings.strategy, symbol=symbol)
        
        actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
        
        print(f"✅ Сгенерировано {len(signals)} сигналов")
        print(f"   Actionable (LONG/SHORT): {len(actionable_signals)}")
        print(f"   HOLD: {len(signals) - len(actionable_signals)}")
        print()
        
        if not actionable_signals:
            print("⚠️  НЕ СГЕНЕРИРОВАНО НИ ОДНОГО ACTIONABLE СИГНАЛА!")
            print()
            print("💡 Возможные причины:")
            print("   - Стратегия работает только в определенные сессии (Лондон/Нью-Йорк)")
            print("   - Аллигатор не раскрыт (нет тренда)")
            print("   - Нет снятий ликвидности или FVG")
            print()
            print("🔍 Попробуйте:")
            print(f"   - Увеличить период тестирования: --days {days_back * 2}")
            print("   - Проверить, что данные содержат активные торговые сессии")
            return
        
        # Показываем первые 10 сигналов
        print("=" * 80)
        print("📋 ПЕРВЫЕ 10 СИГНАЛОВ:")
        print("=" * 80)
        for i, sig in enumerate(actionable_signals[:10], 1):
            ts_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
            sl_price, tp_price = extract_tp_sl_from_reason(sig.reason)
            sl_str = f"SL={sl_price:.2f}" if sl_price else "SL=N/A"
            tp_str = f"TP={tp_price:.2f}" if tp_price else "TP=N/A"
            print(f"{i:2d}. [{ts_str}] {sig.action.value.upper():5s} @ ${sig.price:,.2f} - {sl_str}, {tp_str}")
        
        if len(actionable_signals) > 10:
            print(f"\n... и еще {len(actionable_signals) - 10} сигналов")
        print()
        
        # Запускаем симуляцию
        print("=" * 80)
        print("💰 СИМУЛЯЦИЯ ТОРГОВЛИ")
        print("=" * 80)
        
        sim = ICTSimulator(settings)
        result = sim.run_with_tp_sl(df_ready, actionable_signals)
        
        # Выводим статистику
        print(f"📊 СТАТИСТИКА:")
        print(f"   Всего сделок: {result['total_trades']}")
        print(f"   Прибыльных: {result['winning_trades']} ({result['win_rate']:.1f}%)")
        print(f"   Убыточных: {result['losing_trades']}")
        print()
        print(f"💰 PnL:")
        print(f"   Общий PnL: {result['total_pnl']:+.2f} USDT")
        print(f"   Средний PnL на сделку: {result['avg_pnl']:+.2f} USDT")
        print(f"   Средний выигрыш: {result['avg_win']:+.2f} USDT")
        print(f"   Средний проигрыш: {result['avg_loss']:+.2f} USDT")
        print(f"   Максимальный выигрыш: {result['max_win']:+.2f} USDT")
        print(f"   Максимальный проигрыш: {result['max_loss']:+.2f} USDT")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print()
        
        # Показываем последние 10 сделок
        if result['trades']:
            print("=" * 80)
            print("📋 ПОСЛЕДНИЕ 10 СДЕЛОК:")
            print("=" * 80)
            for i, trade in enumerate(result['trades'][-10:], 1):
                entry_ts = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S') if trade.entry_time else "N/A"
                exit_ts = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else "N/A"
                pnl_str = f"{trade.pnl:+.2f}" if trade.pnl != 0 else "0.00"
                pnl_color = "✅" if trade.pnl > 0 else "❌" if trade.pnl < 0 else "⚪"
                print(f"{i:2d}. {pnl_color} {trade.side.value.upper():5s} | "
                      f"Entry: ${trade.entry_price:,.2f} @ {entry_ts} | "
                      f"Exit: ${trade.exit_price:,.2f} @ {exit_ts} | "
                      f"PnL: {pnl_str} USDT | "
                      f"Reason: {trade.exit_reason}")
        
        print()
        print("=" * 80)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ ОШИБКА при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование ICT Silver Bullet стратегии на исторических данных")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Торговая пара (по умолчанию: BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Количество дней назад для тестирования (по умолчанию: 30)")
    parser.add_argument("--no-london", action="store_true", help="Отключить Лондонскую сессию")
    parser.add_argument("--no-ny", action="store_true", help="Отключить Нью-Йоркскую сессию")
    
    args = parser.parse_args()
    
    test_ict_strategy(
        symbol=args.symbol,
        days_back=args.days,
        enable_london=not args.no_london,
        enable_ny=not args.no_ny,
    )
