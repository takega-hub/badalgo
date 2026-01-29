"""
Бэктест ML стратегии на исторических данных.
Тестирует ML модели (Ensemble, Triple Ensemble, LightGBM, LSTM) с расчетом PnL и Win Rate.

Использование:
    python backtest_ml_strategy.py --symbol BTCUSDT --days 30 --model ml_models/triple_ensemble_BTCUSDT_15.pkl
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, ApiSettings
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal, Bias


@dataclass
class Trade:
    """Сделка в бэктесте."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    action: Action
    size_usd: float
    pnl: float
    pnl_pct: float
    entry_reason: str
    exit_reason: str
    symbol: str
    confidence: float
    stop_loss: float
    take_profit: float


@dataclass
class BacktestMetrics:
    """Метрики бэктеста."""
    symbol: str
    model_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_signals: int
    long_signals: int
    short_signals: int
    avg_trade_duration_hours: float
    best_trade_pnl: float
    worst_trade_pnl: float
    consecutive_wins: int
    consecutive_losses: int
    largest_win: float
    largest_loss: float
    avg_confidence: float


class MLBacktestSimulator:
    """Симулятор для бэктеста ML стратегии."""
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.02,  # 2% риска на сделку
        commission: float = 0.0006,  # 0.06% комиссия Bybit
        max_position_size_pct: float = 0.1,  # Максимальный размер позиции 10%
        leverage: int = 10,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.max_position_size_pct = max_position_size_pct
        self.leverage = leverage
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.equity_curve: List[float] = [initial_balance]
        self.max_equity = initial_balance
        
    def open_position(
        self,
        signal: Signal,
        current_time: datetime,
        symbol: str,
    ) -> bool:
        """Открывает позицию на основе сигнала."""
        if self.current_position is not None:
            return False  # Уже есть открытая позиция
        
        if signal.action == Action.HOLD:
            return False
        
        def _as_float_or_none(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                fv = float(v)
                if not np.isfinite(fv):
                    return None
                return fv
            except Exception:
                return None

        def _pct_to_frac(pct: Optional[float]) -> Optional[float]:
            """
            В indicators_info ML tp_pct/sl_pct хранятся в ПРОЦЕНТАХ (например 1.80 означает 1.80%).
            В некоторых местах могут оказаться доли (0.018). Поддерживаем оба формата:
            - если значение >= 1.0 -> считаем, что это проценты и делим на 100
            - если 0 < значение < 1.0 -> считаем, что это уже доля
            """
            if pct is None:
                return None
            if pct <= 0:
                return None
            # ИСПРАВЛЕНО: >= 1.0 вместо > 1.0, чтобы 1.0% правильно обрабатывался
            return (pct / 100.0) if pct >= 1.0 else pct

        # Рассчитываем размер позиции на основе риска
        stop_loss = _as_float_or_none(signal.stop_loss) or (_as_float_or_none(signal.indicators_info.get('stop_loss')) if signal.indicators_info else None)
        take_profit = _as_float_or_none(signal.take_profit) or (_as_float_or_none(signal.indicators_info.get('take_profit')) if signal.indicators_info else None)
        
        if stop_loss is None or take_profit is None:
            # Если SL/TP не указаны, используем значения по умолчанию из indicators_info
            if signal.indicators_info:
                tp_pct_raw = _as_float_or_none(signal.indicators_info.get('tp_pct'))
                sl_pct_raw = _as_float_or_none(signal.indicators_info.get('sl_pct'))
                tp_frac = _pct_to_frac(tp_pct_raw) if tp_pct_raw is not None else 0.025  # 2.5% по умолчанию
                sl_frac = _pct_to_frac(sl_pct_raw) if sl_pct_raw is not None else 0.01   # 1% по умолчанию

                # Sanity: ограничим экстремальные значения (защита от кривых данных)
                tp_frac = float(np.clip(tp_frac, 0.001, 0.10))  # 0.1% .. 10%
                sl_frac = float(np.clip(sl_frac, 0.001, 0.10))  # 0.1% .. 10%

                stop_loss = signal.price * (1 - sl_frac) if signal.action == Action.LONG else signal.price * (1 + sl_frac)
                take_profit = signal.price * (1 + tp_frac) if signal.action == Action.LONG else signal.price * (1 - tp_frac)
                
                # Отладочный вывод для первых нескольких сигналов
                if not hasattr(self, '_debug_signals_count'):
                    self._debug_signals_count = 0
                if self._debug_signals_count < 3:
                    print(f"[DEBUG] Signal #{self._debug_signals_count + 1}: {signal.action.value} @ ${signal.price:.2f}")
                    print(f"  tp_pct_raw={tp_pct_raw}, sl_pct_raw={sl_pct_raw}")
                    print(f"  tp_frac={tp_frac:.4f} ({tp_frac*100:.2f}%), sl_frac={sl_frac:.4f} ({sl_frac*100:.2f}%)")
                    print(f"  TP=${take_profit:.2f} ({abs(take_profit-signal.price)/signal.price*100:.2f}% from price)")
                    print(f"  SL=${stop_loss:.2f} ({abs(stop_loss-signal.price)/signal.price*100:.2f}% from price)")
                    self._debug_signals_count += 1
            else:
                # Используем фиксированные значения
                stop_loss = signal.price * 0.99 if signal.action == Action.LONG else signal.price * 1.01
                take_profit = signal.price * 1.02 if signal.action == Action.LONG else signal.price * 0.98

        # Финальная sanity-проверка, что SL/TP по правильную сторону цены
        stop_loss = _as_float_or_none(stop_loss)
        take_profit = _as_float_or_none(take_profit)
        if stop_loss is None or take_profit is None:
            return False
        if signal.action == Action.LONG:
            if not (stop_loss < signal.price and take_profit > signal.price):
                # fallback на консервативные значения
                stop_loss = signal.price * 0.99
                take_profit = signal.price * 1.02
        else:
            if not (stop_loss > signal.price and take_profit < signal.price):
                stop_loss = signal.price * 1.01
                take_profit = signal.price * 0.98
        
        # Рассчитываем риск на сделку
        if signal.action == Action.LONG:
            risk_per_unit = abs(signal.price - stop_loss)
        else:
            risk_per_unit = abs(stop_loss - signal.price)
        
        if risk_per_unit == 0:
            return False
        
        # Размер позиции в USD с учетом риска
        risk_amount = self.balance * self.risk_per_trade
        position_size_usd = (risk_amount / risk_per_unit) * signal.price
        
        # Ограничиваем размер позиции
        max_position_size = self.balance * self.max_position_size_pct
        position_size_usd = min(position_size_usd, max_position_size)
        
        # Получаем уверенность из сигнала
        confidence = signal.indicators_info.get('confidence', 0.5) if signal.indicators_info else 0.5
        
        # Создаем позицию
        self.current_position = Trade(
            entry_time=current_time,
            exit_time=None,
            entry_price=signal.price,
            exit_price=None,
            action=signal.action,
            size_usd=position_size_usd,
            pnl=0.0,
            pnl_pct=0.0,
            entry_reason=signal.reason,
            exit_reason="",
            symbol=symbol,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        return True
    
    def check_exit(
        self,
        current_time: datetime,
        current_price: float,
        high: float,
        low: float,
        opposite_signal: Optional[Action] = None,
        max_position_hours: float = 168.0,  # 7 дней максимум
    ) -> bool:
        """Проверяет условия выхода из позиции."""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        
        # Проверяем максимальное время удержания позиции (7 дней)
        position_duration_hours = (current_time - pos.entry_time).total_seconds() / 3600
        if position_duration_hours >= max_position_hours:
            # Закрываем по времени
            exit_price = current_price
            exit_reason = "TIME_LIMIT"
            self.close_position(current_time, exit_price, exit_reason)
            return True
        
        # Проверяем противоположный сигнал (если есть)
        if opposite_signal is not None:
            if (pos.action == Action.LONG and opposite_signal == Action.SHORT) or \
               (pos.action == Action.SHORT and opposite_signal == Action.LONG):
                # Закрываем по противоположному сигналу
                exit_price = current_price
                exit_reason = "OPPOSITE_SIGNAL"
                self.close_position(current_time, exit_price, exit_reason)
                return True
        
        # Проверяем Stop Loss и Take Profit
        if pos.action == Action.LONG:
            # LONG позиция
            if low <= pos.stop_loss:
                # Stop Loss сработал
                exit_price = pos.stop_loss
                exit_reason = "SL"
            elif high >= pos.take_profit:
                # Take Profit сработал
                exit_price = pos.take_profit
                exit_reason = "TP"
            else:
                return False
        else:
            # SHORT позиция
            if high >= pos.stop_loss:
                # Stop Loss сработал
                exit_price = pos.stop_loss
                exit_reason = "SL"
            elif low <= pos.take_profit:
                # Take Profit сработал
                exit_price = pos.take_profit
                exit_reason = "TP"
            else:
                return False
        
        # Закрываем позицию
        self.close_position(current_time, exit_price, exit_reason)
        return True
    
    def close_position(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
    ):
        """Закрывает текущую позицию."""
        if self.current_position is None:
            return
        
        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        
        # Рассчитываем PnL
        if pos.action == Action.LONG:
            price_change = exit_price - pos.entry_price
        else:
            price_change = pos.entry_price - exit_price
        
        # PnL с учетом комиссии и плеча
        pnl_pct = (price_change / pos.entry_price) * self.leverage
        pnl_usd = pos.size_usd * pnl_pct
        
        # Вычитаем комиссию (вход + выход)
        # Комиссия считается от NOTIONAL, а не от маржи. В бэктесте pos.size_usd трактуем как маржу,
        # поэтому умножаем на leverage для приближённого notional.
        commission_cost = (pos.size_usd * self.leverage) * self.commission * 2
        pnl_usd -= commission_cost
        
        pos.pnl = pnl_usd
        pos.pnl_pct = pnl_pct * 100
        pos.exit_reason = exit_reason
        
        # Обновляем баланс
        self.balance += pnl_usd
        self.equity_curve.append(self.balance)
        
        # Обновляем максимальный equity для расчета просадки
        if self.balance > self.max_equity:
            self.max_equity = self.balance
        
        # Сохраняем сделку
        self.trades.append(pos)
        self.current_position = None
    
    def close_all_positions(self, final_time: datetime, final_price: float):
        """Закрывает все открытые позиции в конце бэктеста."""
        if self.current_position is not None:
            self.close_position(final_time, final_price, "END_OF_BACKTEST")
    
    def calculate_metrics(self, symbol: str, model_name: str) -> BacktestMetrics:
        """Рассчитывает метрики бэктеста."""
        if not self.trades:
            return BacktestMetrics(
                symbol=symbol,
                model_name=model_name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                total_signals=0,
                long_signals=0,
                short_signals=0,
                avg_trade_duration_hours=0.0,
                best_trade_pnl=0.0,
                worst_trade_pnl=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_confidence=0.0,
            )
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0.0
        
        total_pnl = self.balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Максимальная просадка
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak = self.initial_balance
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Sharpe Ratio (упрощенный, с защитой от почти-нулевой дисперсии)
        sharpe_ratio = 0.0
        if len(self.trades) > 1:
            returns = np.array([t.pnl_pct / 100 for t in self.trades], dtype=float)
            std = float(np.std(returns))
            if std >= 1e-9:
                sharpe_ratio = float(np.mean(returns) / std * np.sqrt(252))
        
        # Статистика по сигналам
        long_signals = len([t for t in self.trades if t.action == Action.LONG])
        short_signals = len([t for t in self.trades if t.action == Action.SHORT])
        
        # Средняя длительность сделки
        durations = []
        for t in self.trades:
            if t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration)
        avg_trade_duration_hours = np.mean(durations) if durations else 0.0
        
        # Лучшая и худшая сделки
        best_trade_pnl = max([t.pnl for t in self.trades]) if self.trades else 0.0
        worst_trade_pnl = min([t.pnl for t in self.trades]) if self.trades else 0.0
        
        # Серии побед/поражений
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        is_winning = None
        
        for t in self.trades:
            is_win = t.pnl > 0
            if is_winning is None:
                is_winning = is_win
                current_streak = 1
            elif is_win == is_winning:
                current_streak += 1
            else:
                if is_winning:
                    consecutive_wins = max(consecutive_wins, current_streak)
                else:
                    consecutive_losses = max(consecutive_losses, current_streak)
                is_winning = is_win
                current_streak = 1
        
        if is_winning is not None:
            if is_winning:
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                consecutive_losses = max(consecutive_losses, current_streak)
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Средняя уверенность
        avg_confidence = np.mean([t.confidence for t in self.trades]) if self.trades else 0.0
        
        return BacktestMetrics(
            symbol=symbol,
            model_name=model_name,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_signals=len(self.trades),
            long_signals=long_signals,
            short_signals=short_signals,
            avg_trade_duration_hours=avg_trade_duration_hours,
            best_trade_pnl=best_trade_pnl,
            worst_trade_pnl=worst_trade_pnl,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_confidence=avg_confidence,
        )


def run_ml_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    interval: str = "15",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
) -> BacktestMetrics:
    """
    Запускает бэктест ML стратегии.
    
    Args:
        model_path: Путь к ML модели
        symbol: Торговая пара
        days_back: Количество дней для бэктеста
        interval: Интервал свечей
        initial_balance: Начальный баланс
        risk_per_trade: Риск на сделку (доля от баланса)
        leverage: Плечо
    
    Returns:
        BacktestMetrics с результатами
    """
    print("=" * 80)
    print(f"🚀 ML Strategy Backtest")
    print("=" * 80)
    print(f"Model: {Path(model_path).name}")
    print(f"Symbol: {symbol}")
    print(f"Days: {days_back}")
    print(f"Interval: {interval}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Risk per Trade: {risk_per_trade*100:.1f}%")
    print(f"Leverage: {leverage}x")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Создаем клиент
    client = BybitClient(settings.api)
    
    # Получаем исторические данные
    print(f"\n📊 Loading historical data...")
    try:
        # Преобразуем интервал
        if interval.endswith("m"):
            bybit_interval = interval[:-1]
        else:
            bybit_interval = interval
        
        # Рассчитываем количество свечей
        candles_per_day = (24 * 60) // int(bybit_interval)
        total_candles = days_back * candles_per_day
        
        df = client.get_kline_df(symbol, bybit_interval, limit=total_candles)
        
        if df.empty:
            print(f"❌ No data received for {symbol}")
            return None
        
        print(f"✅ Loaded {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Подготавливаем индикаторы
    print(f"\n🔧 Preparing indicators...")
    try:
        df_with_indicators = prepare_with_indicators(df.copy())
        print(f"✅ Indicators prepared")
    except Exception as e:
        print(f"❌ Error preparing indicators: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Готовим MLStrategy и фичи (важно: has_position берём из симулятора, а не из прошлых сигналов)
    print(f"\n🤖 Preparing ML strategy & features...")
    try:
        strategy = MLStrategy(
            model_path=model_path,
            confidence_threshold=settings.ml_confidence_threshold,
            min_signal_strength=settings.ml_min_signal_strength,
            stability_filter=settings.ml_stability_filter,
            min_signals_per_day=settings.ml_min_signals_per_day,
            max_signals_per_day=settings.ml_max_signals_per_day,
        )

        # ВАЖНО: Для MTF-моделей добавляем MTF-фичи независимо от глобального ML_MTF_ENABLED,
        # чтобы сравнение разных моделей было корректным.
        filename = Path(model_path).name.lower()
        is_mtf_model = filename.endswith("_mtf.pkl")

        df_work = df_with_indicators.copy()
        if "timestamp" in df_work.columns:
            df_work = df_work.set_index("timestamp")
        if not isinstance(df_work.index, pd.DatetimeIndex):
            try:
                df_work.index = pd.to_datetime(df_work.index)
            except Exception:
                pass

        # Базовые фичи 15m
        df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)

        # MTF-фичи (1h/4h) только для *_mtf.pkl
        if is_mtf_model:
            try:
                ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                df_1h = df_work.resample("60min").agg(ohlcv_agg).dropna()
                df_4h = df_work.resample("240min").agg(ohlcv_agg).dropna()
                higher_timeframes = {}
                if not df_1h.empty:
                    higher_timeframes["60"] = df_1h
                if not df_4h.empty:
                    higher_timeframes["240"] = df_4h
                if higher_timeframes:
                    df_with_features = strategy.feature_engineer.add_mtf_features(df_with_features, higher_timeframes)
                    print(f"✅ MTF features enabled for this model ({Path(model_path).name})")
            except Exception as mtf_err:
                print(f"⚠️  Failed to add MTF features for {Path(model_path).name}: {mtf_err}")
        else:
            print(f"ℹ️  15m-only features for this model ({Path(model_path).name})")
    except Exception as e:
        print(f"❌ Error preparing ML strategy/features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Создаем симулятор
    simulator = MLBacktestSimulator(
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
    )
    
    # Запускаем бэктест
    print(f"\n📈 Running backtest...")

    total_signals = 0
    long_signals = 0
    short_signals = 0

    # Проходим по всем свечам (используем df_with_features, чтобы generate_signal работал с готовыми фичами)
    for idx, row in df_with_features.iterrows():
        current_time = idx
        current_price = row['close']
        high = row['high']
        low = row['low']

        # Проверяем выход из позиции (TP/SL/таймлимит) — всегда
        if simulator.current_position is not None:
            # Генерируем сигнал, чтобы учитывать выход по противоположному сигналу
            has_position = Bias.LONG if simulator.current_position.action == Action.LONG else Bias.SHORT
            df_until_now = df_with_features.loc[:idx]
            if len(df_until_now) >= 200:
                signal = strategy.generate_signal(
                    row=row,
                    df=df_until_now,
                    has_position=has_position,
                    current_price=current_price,
                    leverage=leverage,
                    target_profit_pct_margin=settings.ml_target_profit_pct_margin,
                    max_loss_pct_margin=settings.ml_max_loss_pct_margin,
                )
                if signal.action in (Action.LONG, Action.SHORT):
                    total_signals += 1
                    if signal.action == Action.LONG:
                        long_signals += 1
                    else:
                        short_signals += 1
                    opposite_signal = signal.action
                else:
                    opposite_signal = None
            else:
                opposite_signal = None

            simulator.check_exit(
                current_time, 
                current_price, 
                high, 
                low,
                opposite_signal=opposite_signal,
                max_position_hours=168.0,  # 7 дней максимум
            )

        # Проверяем вход в позицию
        if simulator.current_position is None:
            df_until_now = df_with_features.loc[:idx]
            if len(df_until_now) >= 200:
                signal = strategy.generate_signal(
                    row=row,
                    df=df_until_now,
                    has_position=None,
                    current_price=current_price,
                    leverage=leverage,
                    target_profit_pct_margin=settings.ml_target_profit_pct_margin,
                    max_loss_pct_margin=settings.ml_max_loss_pct_margin,
                )
                if signal.action in (Action.LONG, Action.SHORT):
                    total_signals += 1
                    if signal.action == Action.LONG:
                        long_signals += 1
                    else:
                        short_signals += 1
                    simulator.open_position(signal, current_time, symbol)

    print(f"📊 Generated actionable signals during simulation: {total_signals} (LONG={long_signals}, SHORT={short_signals})")
    
    # Закрываем все открытые позиции в конце
    if simulator.current_position is not None:
        final_price = df_with_features['close'].iloc[-1]
        final_time = df_with_features.index[-1]
        simulator.close_all_positions(final_time, final_price)
    
    # Рассчитываем метрики
    print(f"\n📊 Calculating metrics...")
    model_name = Path(model_path).stem
    metrics = simulator.calculate_metrics(symbol, model_name)
    
    # Выводим результаты
    print("\n" + "=" * 80)
    print("📈 BACKTEST RESULTS")
    print("=" * 80)
    print(f"Symbol: {metrics.symbol}")
    print(f"Model: {metrics.model_name}")
    print(f"\n💰 Financial Metrics:")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Final Balance: ${initial_balance + metrics.total_pnl:.2f}")
    print(f"   Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)")
    print(f"   Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
    
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Winning Trades: {metrics.winning_trades}")
    print(f"   Losing Trades: {metrics.losing_trades}")
    print(f"   Win Rate: {metrics.win_rate:.2f}%")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    print(f"\n📈 Trade Details:")
    print(f"   Average Win: ${metrics.avg_win:.2f}")
    print(f"   Average Loss: ${metrics.avg_loss:.2f}")
    print(f"   Best Trade: ${metrics.best_trade_pnl:.2f}")
    print(f"   Worst Trade: ${metrics.worst_trade_pnl:.2f}")
    print(f"   Largest Win: ${metrics.largest_win:.2f}")
    print(f"   Largest Loss: ${metrics.largest_loss:.2f}")
    
    print(f"\n⏱️  Timing:")
    print(f"   Average Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
    print(f"   Consecutive Wins: {metrics.consecutive_wins}")
    print(f"   Consecutive Losses: {metrics.consecutive_losses}")
    
    print(f"\n🎯 Signal Distribution:")
    print(f"   Total Signals: {metrics.total_signals}")
    print(f"   LONG Signals: {metrics.long_signals}")
    print(f"   SHORT Signals: {metrics.short_signals}")
    print(f"   Average Confidence: {metrics.avg_confidence:.2%}")
    
    print("\n" + "=" * 80)
    
    # Сохраняем результаты в файл
    results_file = f"ml_backtest_{symbol}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ML STRATEGY BACKTEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {metrics.model_name}\n")
        f.write(f"Symbol: {metrics.symbol}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Days Back: {days_back}\n\n")
        
        f.write("FINANCIAL METRICS:\n")
        f.write(f"  Initial Balance: ${initial_balance:.2f}\n")
        f.write(f"  Final Balance: ${initial_balance + metrics.total_pnl:.2f}\n")
        f.write(f"  Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)\n")
        f.write(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)\n\n")
        
        f.write("TRADE STATISTICS:\n")
        f.write(f"  Total Trades: {metrics.total_trades}\n")
        f.write(f"  Winning Trades: {metrics.winning_trades}\n")
        f.write(f"  Losing Trades: {metrics.losing_trades}\n")
        f.write(f"  Win Rate: {metrics.win_rate:.2f}%\n")
        f.write(f"  Profit Factor: {metrics.profit_factor:.2f}\n")
        f.write(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n\n")
        
        f.write("TRADE DETAILS:\n")
        f.write(f"  Average Win: ${metrics.avg_win:.2f}\n")
        f.write(f"  Average Loss: ${metrics.avg_loss:.2f}\n")
        f.write(f"  Best Trade: ${metrics.best_trade_pnl:.2f}\n")
        f.write(f"  Worst Trade: ${metrics.worst_trade_pnl:.2f}\n\n")
        
        f.write("DETAILED TRADES:\n")
        for i, trade in enumerate(simulator.trades, 1):
            f.write(f"\n  Trade #{i}:\n")
            f.write(f"    Entry: {trade.entry_time} @ ${trade.entry_price:.2f} ({trade.action.value})\n")
            f.write(f"    Exit: {trade.exit_time} @ ${trade.exit_price:.2f} ({trade.exit_reason})\n")
            f.write(f"    PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)\n")
            f.write(f"    Confidence: {trade.confidence:.2%}\n")
            f.write(f"    Reason: {trade.entry_reason}\n")
    
    print(f"✅ Results saved to: {results_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest ML strategy on historical data')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ML model file (e.g., ml_models/triple_ensemble_BTCUSDT_15.pkl)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to backtest (default: 30)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Timeframe interval (default: 15m)')
    parser.add_argument('--balance', type=float, default=1000.0,
                       help='Initial balance (default: 1000.0)')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Risk per trade as fraction (default: 0.02 = 2%%)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Leverage (default: 10)')
    
    args = parser.parse_args()
    
    # Проверяем существование модели
    model_path = Path(args.model)
    if not model_path.exists():
        # Пробуем найти в ml_models
        model_path_alt = Path("ml_models") / args.model
        if model_path_alt.exists():
            model_path = model_path_alt
        else:
            print(f"❌ Model file not found: {args.model}")
            print(f"   Tried: {model_path}")
            print(f"   Tried: {model_path_alt}")
            return
    
    # Запускаем бэктест
    metrics = run_ml_backtest(
        model_path=str(model_path),
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
    )
    
    if metrics:
        print(f"\n✅ Backtest completed successfully!")
    else:
        print(f"\n❌ Backtest failed!")


if __name__ == "__main__":
    main()
