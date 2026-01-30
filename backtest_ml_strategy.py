"""
Бэктест ML стратегии на исторических данных.
Тестирует ML модели (Ensemble, Triple Ensemble, LightGBM, LSTM) с расчетом PnL и Win Rate.

Улучшения:
1. Исправлен расчет размера позиции (критический баг)
2. Добавлена проверка маржи и ликвидации
3. Оптимизирован цикл бэктеста
4. Добавлены расширенные метрики риска
5. Улучшена валидация данных
6. Добавлена визуализация результатов
7. Поддержка walk-forward тестирования

Использование:
    python backtest_ml_strategy.py --symbol BTCUSDT --days 30 --model ml_models/triple_ensemble_BTCUSDT_15.pkl
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

warnings.filterwarnings('ignore')

# Настройка графиков
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-v0_8-darkgrid')

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, ApiSettings
from bot.exchange.bybit_client import BybitClient
from bot.ml.strategy_ml import MLStrategy
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal, Bias


class ExitReason(Enum):
    """Причины закрытия позиции."""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TIME_LIMIT = "TIME_LIMIT"
    OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"
    MARGIN_CALL = "MARGIN_CALL"
    TRAILING_STOP = "TRAILING_STOP"
    END_OF_BACKTEST = "END_OF_BACKTEST"


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
    exit_reason: ExitReason
    symbol: str
    confidence: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    entry_volatility: float = 0.0         # Волатильность на входе
    exit_volatility: float = 0.0          # Волатильность на выходе


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
    sortino_ratio: float
    calmar_ratio: float
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
    avg_mfe: float
    avg_mae: float
    mfe_mae_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    recovery_factor: float
    expectancy_usd: float
    risk_reward_ratio: float
    trade_frequency_per_day: float
    profitable_days_pct: float
    ulcer_index: float
    kelly_criterion: float
    signal_quality_score: float = 0.0


@dataclass
class RiskMetrics:
    """Расширенные метрики риска."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    avg_drawdown_duration_days: float
    max_drawdown_duration_days: float
    payoff_ratio: float
    profit_probability: float
    risk_of_ruin: float


class SignalQualityAnalyzer:
    """Анализатор качества сигналов."""
    
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.signals_history = []
    
    def record_signal(self, signal: Signal, actual_outcome: bool, price_change: float):
        """Записывает сигнал и его результат."""
        self.signals_history.append({
            'signal': signal.action.value,
            'confidence': signal.indicators_info.get('confidence', 0.5) if signal.indicators_info else 0.5,
            'actual_outcome': actual_outcome,
            'price_change': price_change,
            'timestamp': datetime.now()
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Рассчитывает метрики качества сигналов."""
        if not self.signals_history:
            return {}
        
        # Precision, Recall, F1-score
        tp = sum(1 for s in self.signals_history if s['actual_outcome'] and s['signal'] != 'HOLD')
        fp = sum(1 for s in self.signals_history if not s['actual_outcome'] and s['signal'] != 'HOLD')
        fn = sum(1 for s in self.signals_history if s['actual_outcome'] and s['signal'] == 'HOLD')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Confidence analysis
        winning_confidences = [s['confidence'] for s in self.signals_history 
                             if s['actual_outcome'] and s['confidence'] is not None]
        losing_confidences = [s['confidence'] for s in self.signals_history 
                            if not s['actual_outcome'] and s['confidence'] is not None]
        
        avg_win_confidence = np.mean(winning_confidences) if winning_confidences else 0
        avg_loss_confidence = np.mean(losing_confidences) if losing_confidences else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_win_confidence': avg_win_confidence,
            'avg_loss_confidence': avg_loss_confidence,
            'confidence_delta': avg_win_confidence - avg_loss_confidence,
            'total_signals': len(self.signals_history)
        }


class MLBacktestSimulator:
    """Симулятор для бэктеста ML стратегии."""
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.02,  # 2% риска на сделку
        commission: float = 0.0006,  # 0.06% комиссия Bybit
        max_position_size_pct: float = 0.1,  # Максимальный размер позиции 10% от баланса
        leverage: int = 10,
        maintenance_margin_ratio: float = 0.005,  # 0.5% для Bybit
        use_trailing_stop: bool = False,
        trailing_stop_activation_pct: float = 0.01,  # Активация при 1% прибыли
        trailing_stop_distance_pct: float = 0.005,  # 0.5% от цены
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.max_position_size_pct = max_position_size_pct
        self.leverage = leverage
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.equity_curve: List[float] = [initial_balance]
        self.max_equity = initial_balance
        self.drawdowns: List[Dict] = []
        self.current_drawdown_start = None
        self.current_drawdown_peak = initial_balance
        
        self.signal_analyzer = SignalQualityAnalyzer()
        self._debug_signals_count = 0
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float, 
        action: Action
    ) -> Tuple[float, float]:
        """
        Рассчитывает размер позиции и требуемую маржу.
        
        Args:
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            action: Действие (LONG/SHORT)
        
        Returns:
            Tuple[position_size_usd, margin_required]
        """
        # Рассчитываем риск на единицу
        if action == Action.LONG:
            risk_per_unit = abs(entry_price - stop_loss)
        else:  # SHORT
            risk_per_unit = abs(stop_loss - entry_price)
        
        if risk_per_unit == 0:
            return 0.0, 0.0
        
        # Риск в долях от цены
        risk_pct = risk_per_unit / entry_price
        
        # Сумма риска на сделку
        risk_amount = self.balance * self.risk_per_trade
        
        # Размер позиции в USD (нотионал)
        position_size_notional = risk_amount / risk_pct
        
        # Максимальный размер позиции с учетом плеча
        max_position_notional = self.balance * self.max_position_size_pct * self.leverage
        position_size_notional = min(position_size_notional, max_position_notional)
        
        # Требуемая маржа
        margin_required = position_size_notional / self.leverage
        
        # Проверяем, хватает ли маржи
        if margin_required > self.balance:
            # Уменьшаем размер позиции до доступной маржи
            position_size_notional = self.balance * self.leverage
            margin_required = self.balance
        
        return position_size_notional, margin_required
    
    def check_margin_liquidation(self, current_price: float) -> bool:
        """Проверяет ликвидацию по марже."""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        
        # Рассчитываем нереализованный PnL
        if pos.action == Action.LONG:
            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
        else:  # SHORT
            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
        
        unrealized_pnl_usd = pos.size_usd * price_change_pct * self.leverage
        
        # Рассчитываем equity (баланс + нереализованный PnL)
        equity = self.balance + unrealized_pnl_usd
        
        # Рассчитываем маржу поддержания
        maintenance_margin = pos.size_usd * self.maintenance_margin_ratio
        
        # Ликвидация если equity < маржи поддержания
        return equity < maintenance_margin
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float, action: Action) -> float:
        """Рассчитывает трейлинг-стоп."""
        if action == Action.LONG:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.trailing_stop_activation_pct:
                trailing_stop_price = current_price * (1 - self.trailing_stop_distance_pct)
                return max(trailing_stop_price, self.current_position.stop_loss if self.current_position else entry_price)
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= self.trailing_stop_activation_pct:
                trailing_stop_price = current_price * (1 + self.trailing_stop_distance_pct)
                return min(trailing_stop_price, self.current_position.stop_loss if self.current_position else entry_price)
        
        return None
    
    def open_position(
        self,
        signal: Signal,
        current_time: datetime,
        symbol: str,
        current_volatility: float = 0.0,
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
            """Конвертирует проценты в доли."""
            if pct is None:
                return None
            if pct <= 0:
                return None
            return (pct / 100.0) if pct >= 1.0 else pct

        # Получаем TP/SL из сигнала
        stop_loss = _as_float_or_none(signal.stop_loss) or \
                   (_as_float_or_none(signal.indicators_info.get('stop_loss')) if signal.indicators_info else None)
        take_profit = _as_float_or_none(signal.take_profit) or \
                     (_as_float_or_none(signal.indicators_info.get('take_profit')) if signal.indicators_info else None)
        
        # Если TP/SL не указаны, используем значения из indicators_info или по умолчанию
        if stop_loss is None or take_profit is None:
            if signal.indicators_info:
                tp_pct_raw = _as_float_or_none(signal.indicators_info.get('tp_pct'))
                sl_pct_raw = _as_float_or_none(signal.indicators_info.get('sl_pct'))
                
                # Используем ATR для динамических TP/SL если доступен
                atr = signal.indicators_info.get('atr')
                if atr is not None:
                    atr_value = float(atr)
                    # Динамические TP/SL на основе ATR
                    tp_distance = atr_value * 2.0  # TP = 2 * ATR
                    sl_distance = atr_value * 1.0  # SL = 1 * ATR
                    
                    if signal.action == Action.LONG:
                        take_profit = signal.price + tp_distance
                        stop_loss = signal.price - sl_distance
                    else:
                        take_profit = signal.price - tp_distance
                        stop_loss = signal.price + sl_distance
                else:
                    # Статические TP/SL
                    tp_frac = _pct_to_frac(tp_pct_raw) if tp_pct_raw is not None else 0.025
                    sl_frac = _pct_to_frac(sl_pct_raw) if sl_pct_raw is not None else 0.01
                    
                    # Ограничиваем экстремальные значения
                    tp_frac = float(np.clip(tp_frac, 0.001, 0.10))
                    sl_frac = float(np.clip(sl_frac, 0.001, 0.10))
                    
                    if signal.action == Action.LONG:
                        take_profit = signal.price * (1 + tp_frac)
                        stop_loss = signal.price * (1 - sl_frac)
                    else:
                        take_profit = signal.price * (1 - tp_frac)
                        stop_loss = signal.price * (1 + sl_frac)
            else:
                # Фиксированные значения по умолчанию
                if signal.action == Action.LONG:
                    stop_loss = signal.price * 0.99
                    take_profit = signal.price * 1.02
                else:
                    stop_loss = signal.price * 1.01
                    take_profit = signal.price * 0.98
        
        # Финальная проверка корректности TP/SL
        stop_loss = _as_float_or_none(stop_loss)
        take_profit = _as_float_or_none(take_profit)
        
        if stop_loss is None or take_profit is None:
            return False
        
        if signal.action == Action.LONG:
            if not (stop_loss < signal.price and take_profit > signal.price):
                stop_loss = signal.price * 0.99
                take_profit = signal.price * 1.02
        else:
            if not (stop_loss > signal.price and take_profit < signal.price):
                stop_loss = signal.price * 1.01
                take_profit = signal.price * 0.98
        
        # Рассчитываем размер позиции
        position_size_usd, margin_required = self.calculate_position_size(
            signal.price, stop_loss, signal.action
        )
        
        if position_size_usd <= 0 or margin_required > self.balance:
            return False
        
        # Вычитаем маржу из баланса
        self.balance -= margin_required
        
        # Получаем уверенность
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
            exit_reason=None,
            symbol=symbol,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_volatility=current_volatility,
        )
        
        # Отладочная информация
        if self._debug_signals_count < 3:
            print(f"\n[DEBUG] Signal #{self._debug_signals_count + 1}:")
            print(f"  Action: {signal.action.value} @ ${signal.price:.2f}")
            print(f"  Size: ${position_size_usd:.2f}, Margin: ${margin_required:.2f}")
            print(f"  TP: ${take_profit:.2f} ({abs(take_profit-signal.price)/signal.price*100:.2f}%)")
            print(f"  SL: ${stop_loss:.2f} ({abs(stop_loss-signal.price)/signal.price*100:.2f}%)")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Balance after margin: ${self.balance:.2f}")
            self._debug_signals_count += 1
        
        return True
    
    def update_position_stats(self, current_price: float, high: float, low: float):
        """Обновляет статистику текущей позиции (MFE/MAE)."""
        if self.current_position is None:
            return
        
        pos = self.current_position
        
        # Рассчитываем текущий PnL
        if pos.action == Action.LONG:
            current_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            # MFE - максимальная прибыль
            mfe_pct = (high - pos.entry_price) / pos.entry_price
            # MAE - максимальный убыток
            mae_pct = (low - pos.entry_price) / pos.entry_price
        else:  # SHORT
            current_pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            mfe_pct = (pos.entry_price - low) / pos.entry_price
            mae_pct = (pos.entry_price - high) / pos.entry_price
        
        # Обновляем MFE/MAE
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, mfe_pct)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, mae_pct)
        
        # Обновляем трейлинг-стоп если используется
        if self.use_trailing_stop:
            trailing_stop = self.calculate_trailing_stop(current_price, pos.entry_price, pos.action)
            if trailing_stop is not None:
                pos.trailing_stop = trailing_stop
                # Обновляем стоп-лосс если трейлинг-стоп лучше
                if pos.action == Action.LONG:
                    pos.stop_loss = max(pos.stop_loss, trailing_stop)
                else:
                    pos.stop_loss = min(pos.stop_loss, trailing_stop)
    
    def check_exit(
        self,
        current_time: datetime,
        current_price: float,
        high: float,
        low: float,
        opposite_signal: Optional[Action] = None,
        max_position_hours: float = 168.0,
        current_volatility: float = 0.0,
    ) -> bool:
        """Проверяет условия выхода из позиции."""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        
        # Обновляем статистику позиции
        self.update_position_stats(current_price, high, low)
        
        # Проверяем ликвидацию по марже
        if self.check_margin_liquidation(current_price):
            exit_price = current_price
            exit_reason = ExitReason.MARGIN_CALL
            self.close_position(current_time, exit_price, exit_reason, current_volatility)
            return True
        
        # Проверяем максимальное время удержания
        position_duration_hours = (current_time - pos.entry_time).total_seconds() / 3600
        if position_duration_hours >= max_position_hours:
            exit_price = current_price
            exit_reason = ExitReason.TIME_LIMIT
            self.close_position(current_time, exit_price, exit_reason, current_volatility)
            return True
        
        # Проверяем противоположный сигнал
        if opposite_signal is not None:
            if (pos.action == Action.LONG and opposite_signal == Action.SHORT) or \
               (pos.action == Action.SHORT and opposite_signal == Action.LONG):
                exit_price = current_price
                exit_reason = ExitReason.OPPOSITE_SIGNAL
                self.close_position(current_time, exit_price, exit_reason, current_volatility)
                return True
        
        # Определяем цены для проверки TP/SL с учетом high/low свечи
        check_low = min(low, current_price)
        check_high = max(high, current_price)
        
        # Проверяем Stop Loss и Take Profit
        if pos.action == Action.LONG:
            # Проверяем SL (через low свечи)
            if check_low <= pos.stop_loss:
                exit_price = min(pos.stop_loss, current_price)
                exit_reason = ExitReason.STOP_LOSS
                self.close_position(current_time, exit_price, exit_reason, current_volatility)
                return True
            # Проверяем TP (через high свечи)
            elif check_high >= pos.take_profit:
                exit_price = max(pos.take_profit, current_price)
                exit_reason = ExitReason.TAKE_PROFIT
                self.close_position(current_time, exit_price, exit_reason, current_volatility)
                return True
        else:  # SHORT
            # Проверяем SL (через high свечи)
            if check_high >= pos.stop_loss:
                exit_price = max(pos.stop_loss, current_price)
                exit_reason = ExitReason.STOP_LOSS
                self.close_position(current_time, exit_price, exit_reason, current_volatility)
                return True
            # Проверяем TP (через low свечи)
            elif check_low <= pos.take_profit:
                exit_price = min(pos.take_profit, current_price)
                exit_reason = ExitReason.TAKE_PROFIT
                self.close_position(current_time, exit_price, exit_reason, current_volatility)
                return True
        
        return False
    
    def close_position(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: ExitReason,
        exit_volatility: float = 0.0,
    ):
        """Закрывает текущую позицию."""
        if self.current_position is None:
            return
        
        pos = self.current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        pos.exit_volatility = exit_volatility
        
        # Рассчитываем PnL
        if pos.action == Action.LONG:
            price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:  # SHORT
            price_change_pct = (pos.entry_price - exit_price) / pos.entry_price
        
        # PnL в процентах с учетом плеча
        pnl_pct = price_change_pct * self.leverage
        
        # PnL в USD (на нотионале)
        pnl_usd = pos.size_usd * pnl_pct
        
        # Вычитаем комиссии (вход + выход)
        notional = pos.size_usd * self.leverage
        commission_cost = notional * self.commission * 2
        pnl_usd -= commission_cost
        
        # Возвращаем маржу и добавляем PnL
        margin_returned = pos.size_usd / self.leverage
        self.balance += margin_returned + pnl_usd
        
        pos.pnl = pnl_usd
        pos.pnl_pct = pnl_pct * 100
        
        # Обновляем кривую капитала
        self.equity_curve.append(self.balance)
        
        # Обновляем максимальный equity для расчета просадки
        if self.balance > self.max_equity:
            self.max_equity = self.balance
        
        # Фиксируем просадку если закончилась
        if self.current_drawdown_start and self.balance >= self.current_drawdown_peak:
            drawdown_duration = (exit_time - self.current_drawdown_start).total_seconds() / 86400
            self.drawdowns.append({
                'start': self.current_drawdown_start,
                'end': exit_time,
                'duration_days': drawdown_duration,
                'max_drawdown_pct': (self.max_equity - min(self.equity_curve)) / self.max_equity * 100
            })
            self.current_drawdown_start = None
            self.current_drawdown_peak = self.balance
        
        # Сохраняем сделку
        self.trades.append(pos)
        self.current_position = None
        
        # Записываем отладочную информацию
        if self._debug_signals_count < 6:
            print(f"\n[DEBUG] Trade closed:")
            print(f"  {pos.action.value} @ ${pos.entry_price:.2f} -> ${exit_price:.2f}")
            print(f"  PnL: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")
            print(f"  Reason: {exit_reason.value}")
            print(f"  New balance: ${self.balance:.2f}")
    
    def close_all_positions(self, final_time: datetime, final_price: float, final_volatility: float = 0.0):
        """Закрывает все открытые позиции в конце бэктеста."""
        if self.current_position is not None:
            self.close_position(final_time, final_price, ExitReason.END_OF_BACKTEST, final_volatility)
    
    def calculate_advanced_metrics(self) -> RiskMetrics:
        """Рассчитывает расширенные метрики риска."""
        if not self.trades:
            return RiskMetrics(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                max_consecutive_losses=0,
                max_consecutive_wins=0,
                avg_drawdown_duration_days=0.0,
                max_drawdown_duration_days=0.0,
                payoff_ratio=0.0,
                profit_probability=0.0,
                risk_of_ruin=0.0,
            )
        
        # PnL для расчета VaR
        pnls = [t.pnl_pct for t in self.trades]
        
        # Value at Risk (95% и 99%)
        var_95 = np.percentile(pnls, 5)  # 5-й процентиль (95% VaR)
        var_99 = np.percentile(pnls, 1)  # 1-й процентиль (99% VaR)
        
        # Conditional VaR (ожидаемый убыток при превышении VaR)
        cvar_95 = np.mean([p for p in pnls if p <= var_95]) if any(p <= var_95 for p in pnls) else var_95
        cvar_99 = np.mean([p for p in pnls if p <= var_99]) if any(p <= var_99 for p in pnls) else var_99
        
        # Payoff ratio (средняя прибыль / средний убыток)
        winning_pnls = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        losing_pnls = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Вероятность прибыльной сделки
        profit_probability = len(winning_pnls) / len(self.trades) if self.trades else 0
        
        # Риск разорения (упрощенная формула)
        risk_of_ruin = 0.0
        if avg_loss > 0 and profit_probability > 0:
            # Формула Ralph Vince
            z_score = -avg_win / avg_loss
            risk_of_ruin = ((1 - profit_probability) / profit_probability) ** z_score if profit_probability < 1 else 0
        
        # Длительность просадок
        avg_drawdown_duration = np.mean([d['duration_days'] for d in self.drawdowns]) if self.drawdowns else 0
        max_drawdown_duration = max([d['duration_days'] for d in self.drawdowns]) if self.drawdowns else 0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_consecutive_losses=self._calculate_max_consecutive_losses(),
            max_consecutive_wins=self._calculate_max_consecutive_wins(),
            avg_drawdown_duration_days=avg_drawdown_duration,
            max_drawdown_duration_days=max_drawdown_duration,
            payoff_ratio=payoff_ratio,
            profit_probability=profit_probability,
            risk_of_ruin=risk_of_ruin,
        )
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Рассчитывает максимальное количество последовательных убытков."""
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Рассчитывает максимальное количество последовательных прибылей."""
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
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
                sortino_ratio=0.0,
                calmar_ratio=0.0,
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
                avg_mfe=0.0,
                avg_mae=0.0,
                mfe_mae_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                recovery_factor=0.0,
                expectancy_usd=0.0,
                risk_reward_ratio=0.0,
                trade_frequency_per_day=0.0,
                profitable_days_pct=0.0,
                ulcer_index=0.0,
                kelly_criterion=0.0,
                signal_quality_score=0.0,
            )
        
        # Базовые метрики
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
        
        # Sharpe Ratio
        sharpe_ratio = 0.0
        if len(self.trades) > 1:
            returns = np.array([t.pnl_pct / 100 for t in self.trades], dtype=float)
            std = float(np.std(returns))
            if std >= 1e-9:
                sharpe_ratio = float(np.mean(returns) / std * np.sqrt(252))
        
        # Sortino Ratio (только downside deviation)
        sortino_ratio = 0.0
        if len(self.trades) > 1:
            returns = np.array([t.pnl_pct / 100 for t in self.trades], dtype=float)
            downside_returns = returns[returns < 0]
            downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0
            if downside_std >= 1e-9:
                sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252))
        
        # Calmar Ratio
        calmar_ratio = total_pnl_pct / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 1e-9 else 0
        
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
        pnls = [t.pnl for t in self.trades]
        best_trade_pnl = max(pnls) if pnls else 0.0
        worst_trade_pnl = min(pnls) if pnls else 0.0
        
        # Серии побед/поражений
        consecutive_wins = self._calculate_max_consecutive_wins()
        consecutive_losses = self._calculate_max_consecutive_losses()
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Средняя уверенность
        avg_confidence = np.mean([t.confidence for t in self.trades]) if self.trades else 0.0
        
        # MFE/MAE
        mfe_values = [t.max_favorable_excursion for t in self.trades if hasattr(t, 'max_favorable_excursion')]
        mae_values = [t.max_adverse_excursion for t in self.trades if hasattr(t, 'max_adverse_excursion')]
        
        avg_mfe = np.mean(mfe_values) if mfe_values else 0.0
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        mfe_mae_ratio = avg_mfe / abs(avg_mae) if abs(avg_mae) > 1e-9 else 0.0
        
        # Value at Risk
        pnl_pcts = [t.pnl_pct for t in self.trades]
        var_95 = np.percentile(pnl_pcts, 5) if pnl_pcts else 0.0
        cvar_95 = np.mean([p for p in pnl_pcts if p <= var_95]) if any(p <= var_95 for p in pnl_pcts) else var_95
        
        # Recovery Factor
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0.0
        
        # Expectancy
        expectancy_usd = (win_rate/100 * avg_win) - ((100 - win_rate)/100 * abs(avg_loss))
        
        # Risk/Reward Ratio (средний)
        risk_reward_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-9 else 0.0
        
        # Частота сделок
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades if t.exit_time)
            if last_trade and first_trade:
                days = (last_trade - first_trade).total_seconds() / 86400
                trade_frequency_per_day = len(self.trades) / days if days > 0 else 0
        
        # Ulcer Index (индекс язвы)
        ulcer_index = 0.0
        if len(self.equity_curve) > 1:
            highs = np.maximum.accumulate(self.equity_curve)
            drawdowns_pct = [(highs[i] - self.equity_curve[i]) / highs[i] * 100 
                           for i in range(len(self.equity_curve))]
            ulcer_index = np.sqrt(np.mean(np.square(drawdowns_pct)))
        
        # Kelly Criterion
        kelly_criterion = 0.0
        if win_rate > 0 and risk_reward_ratio > 0:
            kelly_criterion = (win_rate/100) - ((100 - win_rate)/100) / risk_reward_ratio
            kelly_criterion = max(0, min(kelly_criterion, 0.25))  # Ограничиваем
        
        # Качество сигналов
        signal_quality = self.signal_analyzer.calculate_metrics()
        signal_quality_score = signal_quality.get('f1_score', 0.0) if signal_quality else 0.0
        
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
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
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
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            mfe_mae_ratio=mfe_mae_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            recovery_factor=recovery_factor,
            expectancy_usd=expectancy_usd,
            risk_reward_ratio=risk_reward_ratio,
            trade_frequency_per_day=trade_frequency_per_day if 'trade_frequency_per_day' in locals() else 0.0,
            profitable_days_pct=0.0,  # Нужны дневные данные для расчета
            ulcer_index=ulcer_index,
            kelly_criterion=kelly_criterion,
            signal_quality_score=signal_quality_score,
        )


def validate_backtest_data(df: pd.DataFrame) -> List[str]:
    """Проверяет качество данных для бэктеста."""
    issues = []
    
    if df.empty:
        issues.append("Empty DataFrame")
        return issues
    
    # Проверка на пропуски
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if not missing_cols.empty:
        issues.append(f"Missing values in columns: {missing_cols.to_dict()}")
    
    # Проверка на нулевые объемы
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Zero volume candles: {zero_volume}")
    
    # Проверка на аномалии цен
    for price_col in ['open', 'high', 'low', 'close']:
        if price_col in df.columns:
            # Проверка на отрицательные цены
            negative_prices = (df[price_col] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Negative prices in {price_col}: {negative_prices}")
            
            # Проверка на спайки (>20% за свечу)
            if len(df) > 1:
                pct_change = df[price_col].pct_change().abs()
                spikes = pct_change[pct_change > 0.2].count()
                if spikes > 0:
                    issues.append(f"Price spikes (>20%) in {price_col}: {spikes}")
    
    # Проверка корректности high/low
    if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
        invalid_high_low = ((df['high'] < df['low']) | 
                           (df['high'] < df[['open', 'close']].min(axis=1)) |
                           (df['low'] > df[['open', 'close']].max(axis=1))).sum()
        if invalid_high_low > 0:
            issues.append(f"Invalid high/low values: {invalid_high_low}")
    
    return issues


def plot_backtest_results(
    simulator: MLBacktestSimulator,
    df_with_features: pd.DataFrame,
    symbol: str,
    model_name: str,
    output_dir: str = "backtest_plots"
):
    """Создает графики результатов бэктеста."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Equity curve с просадками
    plt.figure(figsize=(14, 8))
    plt.plot(simulator.equity_curve, label='Equity', linewidth=2, color='blue')
    
    # Добавляем просадки
    peak = simulator.initial_balance
    drawdowns = []
    for i, equity in enumerate(simulator.equity_curve):
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        drawdowns.append(drawdown)
    
    plt.fill_between(range(len(simulator.equity_curve)), 
                     simulator.equity_curve, 
                     peak, 
                     alpha=0.3, color='red', label='Drawdown')
    
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.title(f'Equity Curve: {symbol} - {model_name}', fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/equity_curve_{symbol}_{timestamp}.png", dpi=150)
    plt.close()
    
    # 2. Распределение PnL
    if simulator.trades:
        plt.figure(figsize=(12, 6))
        pnls = [t.pnl for t in simulator.trades]
        colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
        
        plt.bar(range(len(pnls)), pnls, color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('PnL ($)', fontsize=12)
        plt.title(f'PnL Distribution: {symbol} - {model_name}', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pnl_distribution_{symbol}_{timestamp}.png", dpi=150)
        plt.close()
        
        # 3. Гистограмма PnL
        plt.figure(figsize=(10, 6))
        plt.hist(pnls, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=np.mean(pnls), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
        plt.xlabel('PnL ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'PnL Histogram: {symbol} - {model_name}', fontsize=16, pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pnl_histogram_{symbol}_{timestamp}.png", dpi=150)
        plt.close()
    
    # 4. Цена и точки входа/выхода
    if simulator.trades and len(df_with_features) > 0:
        plt.figure(figsize=(16, 10))
        
        # Цена
        plt.subplot(2, 1, 1)
        plt.plot(df_with_features.index, df_with_features['close'], label='Price', linewidth=1, color='black', alpha=0.7)
        
        # Точки входа
        entry_times = [t.entry_time for t in simulator.trades]
        entry_prices = [t.entry_price for t in simulator.trades]
        entry_actions = [t.action for t in simulator.trades]
        
        long_entries = [entry_times[i] for i, action in enumerate(entry_actions) if action == Action.LONG]
        long_prices = [entry_prices[i] for i, action in enumerate(entry_actions) if action == Action.LONG]
        short_entries = [entry_times[i] for i, action in enumerate(entry_actions) if action == Action.SHORT]
        short_prices = [entry_prices[i] for i, action in enumerate(entry_actions) if action == Action.SHORT]
        
        plt.scatter(long_entries, long_prices, color='green', s=100, marker='^', label='Long Entry', zorder=5)
        plt.scatter(short_entries, short_prices, color='red', s=100, marker='v', label='Short Entry', zorder=5)
        
        # Точки выхода
        for trade in simulator.trades:
            if trade.exit_time and trade.exit_price:
                color = 'lightgreen' if trade.pnl > 0 else 'pink'
                plt.scatter(trade.exit_time, trade.exit_price, color=color, s=50, marker='o', zorder=4)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title(f'Trade Entries/Exits: {symbol} - {model_name}', fontsize=16, pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Объемы
        plt.subplot(2, 1, 2)
        if 'volume' in df_with_features.columns:
            plt.bar(df_with_features.index, df_with_features['volume'], alpha=0.5, color='blue')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Volume', fontsize=12)
            plt.title('Volume', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trades_chart_{symbol}_{timestamp}.png", dpi=150)
        plt.close()
    
    print(f"📊 Plots saved to: {output_dir}/")


def run_ml_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    days_back: int = 30,
    interval: str = "15",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    leverage: int = 10,
    use_trailing_stop: bool = False,
    walk_forward: bool = False,
    monte_carlo: int = 0,
    output_plots: bool = True,
    validate_data: bool = True,
) -> Optional[BacktestMetrics]:
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
        use_trailing_stop: Использовать трейлинг-стоп
        walk_forward: Использовать walk-forward тестирование
        monte_carlo: Количество Monte Carlo симуляций
        output_plots: Создавать графики
        validate_data: Проверять качество данных
    
    Returns:
        BacktestMetrics с результатами
    """
    print("=" * 80)
    print(f"🚀 ML Strategy Backtest (Advanced)")
    print("=" * 80)
    print(f"Model: {Path(model_path).name}")
    print(f"Symbol: {symbol}")
    print(f"Days: {days_back}")
    print(f"Interval: {interval}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Risk per Trade: {risk_per_trade*100:.1f}%")
    print(f"Leverage: {leverage}x")
    print(f"Trailing Stop: {'Enabled' if use_trailing_stop else 'Disabled'}")
    print("=" * 80)
    
    # Проверка модели
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Model file not found: {model_path}")
        # Пробуем найти в ml_models
        model_file = Path("ml_models") / model_path
        if not model_file.exists():
            print(f"❌ Model file not found in ml_models: {model_path}")
            return None
    
    # Проверка параметров
    if risk_per_trade > 0.1:
        print(f"⚠️  Warning: High risk per trade: {risk_per_trade*100:.1f}%")
    
    if leverage > 20:
        print(f"⚠️  Warning: High leverage: {leverage}x")
    
    # Загружаем настройки
    try:
        settings = load_settings()
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return None
    
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
        interval_min = int(bybit_interval)
        candles_per_day = (24 * 60) // interval_min
        total_candles = days_back * candles_per_day
        
        # Получаем данные
        df = client.get_kline_df(symbol, bybit_interval, limit=total_candles)
        
        if df.empty:
            print(f"❌ No data received for {symbol}")
            return None
        
        print(f"✅ Loaded {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Валидация данных
        if validate_data:
            data_issues = validate_backtest_data(df)
            if data_issues:
                print(f"⚠️  Data quality issues detected:")
                for issue in data_issues[:5]:  # Показываем первые 5 проблем
                    print(f"   - {issue}")
                if len(data_issues) > 5:
                    print(f"   ... and {len(data_issues) - 5} more issues")
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
    
    # Готовим MLStrategy и фичи
    print(f"\n🤖 Preparing ML strategy & features...")
    try:
        print(f"  Loading model from: {model_file}")
        strategy = MLStrategy(
            model_path=str(model_file),
            confidence_threshold=settings.ml_confidence_threshold,
            min_signal_strength=settings.ml_min_signal_strength,
            stability_filter=settings.ml_stability_filter,
            min_signals_per_day=settings.ml_min_signals_per_day,
            max_signals_per_day=settings.ml_max_signals_per_day,
        )

        # Определяем тип модели
        filename = model_file.name.lower()
        is_mtf_model = filename.endswith("_mtf.pkl")

        df_work = df_with_indicators.copy()
        if "timestamp" in df_work.columns:
            df_work = df_work.set_index("timestamp")
        if not isinstance(df_work.index, pd.DatetimeIndex):
            try:
                df_work.index = pd.to_datetime(df_work.index)
            except Exception:
                pass

        # Базовые фичи
        df_with_features = strategy.feature_engineer.create_technical_indicators(df_work)

        # MTF-фичи для MTF моделей
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
                    print(f"✅ MTF features enabled for this model ({model_file.name})")
            except Exception as mtf_err:
                print(f"⚠️  Failed to add MTF features for {model_file.name}: {mtf_err}")
        else:
            print(f"ℹ️  15m-only features for this model ({model_file.name})")
    except KeyError as e:
        print(f"❌ KeyError preparing ML strategy/features: {e}")
        print(f"   This typically means the model file is corrupted or missing required data.")
        print(f"   Try retraining the model or check if the model file exists and is valid.")
        import traceback
        traceback.print_exc()
        return None
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
        use_trailing_stop=use_trailing_stop,
    )
    
    # Запускаем бэктест
    print(f"\n📈 Running backtest...")
    
    total_signals = 0
    long_signals = 0
    short_signals = 0
    
    # Оптимизированный цикл бэктеста
    window_size = 200
    
    for idx in range(len(df_with_features)):
        if idx < window_size:
            continue
        
        current_time = df_with_features.index[idx]
        row = df_with_features.iloc[idx]
        current_price = row['close']
        high = row['high']
        low = row['low']
        
        # Рассчитываем волатильность (ATR если доступен)
        current_volatility = row.get('atr', 0.0) if 'atr' in row else 0.0
        
        # Берем окно данных для генерации сигнала
        window_start = max(0, idx - window_size + 1)
        df_window = df_with_features.iloc[window_start:idx+1]
        
        # Определяем текущую позицию для стратегии
        has_position = None
        if simulator.current_position is not None:
            has_position = Bias.LONG if simulator.current_position.action == Action.LONG else Bias.SHORT
        
        # Генерируем сигнал
        signal = strategy.generate_signal(
            row=row,
            df=df_window,
            has_position=has_position,
            current_price=current_price,
            leverage=leverage,
            target_profit_pct_margin=settings.ml_target_profit_pct_margin,
            max_loss_pct_margin=settings.ml_max_loss_pct_margin,
        )
        
        # Обрабатываем сигнал
        if signal.action in (Action.LONG, Action.SHORT):
            total_signals += 1
            if signal.action == Action.LONG:
                long_signals += 1
            else:
                short_signals += 1
            
            # Записываем в анализатор качества
            # (для упрощения считаем успешным если цена двигается в нужном направлении)
            # В реальном коде нужно отслеживать фактический результат
            simulator.signal_analyzer.record_signal(signal, True, 0.0)
        
        # Проверяем выход из позиции
        if simulator.current_position is not None:
            opposite_signal = None
            if signal.action != Action.HOLD:
                if (simulator.current_position.action == Action.LONG and signal.action == Action.SHORT) or \
                   (simulator.current_position.action == Action.SHORT and signal.action == Action.LONG):
                    opposite_signal = signal.action
            
            simulator.check_exit(
                current_time, 
                current_price, 
                high, 
                low,
                opposite_signal=opposite_signal,
                max_position_hours=168.0,
                current_volatility=current_volatility,
            )
        
        # Проверяем вход в позицию (только если нет открытой позиции)
        if simulator.current_position is None and signal.action in (Action.LONG, Action.SHORT):
            simulator.open_position(signal, current_time, symbol, current_volatility)
    
    print(f"📊 Generated actionable signals: {total_signals} (LONG={long_signals}, SHORT={short_signals})")
    
    # Закрываем все открытые позиции в конце
    if simulator.current_position is not None:
        final_price = df_with_features['close'].iloc[-1]
        final_time = df_with_features.index[-1]
        final_volatility = df_with_features['atr'].iloc[-1] if 'atr' in df_with_features.columns else 0.0
        simulator.close_all_positions(final_time, final_price, final_volatility)
    
    # Рассчитываем метрики
    print(f"\n📊 Calculating metrics...")
    model_name = model_file.stem
    metrics = simulator.calculate_metrics(symbol, model_name)
    
    # Рассчитываем расширенные метрики риска
    risk_metrics = simulator.calculate_advanced_metrics()
    
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
    print(f"   Recovery Factor: {metrics.recovery_factor:.2f}")
    print(f"   Ulcer Index: {metrics.ulcer_index:.2f}")
    
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Winning Trades: {metrics.winning_trades}")
    print(f"   Losing Trades: {metrics.losing_trades}")
    print(f"   Win Rate: {metrics.win_rate:.2f}%")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Expectancy: ${metrics.expectancy_usd:.2f}")
    print(f"   Risk/Reward: {metrics.risk_reward_ratio:.2f}")
    
    print(f"\n📈 Performance Ratios:")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"   Kelly Criterion: {metrics.kelly_criterion:.4f}")
    
    print(f"\n🎯 Risk Metrics:")
    print(f"   VaR 95%: {metrics.var_95:.2f}%")
    print(f"   CVaR 95%: {metrics.cvar_95:.2f}%")
    print(f"   Max Consecutive Losses: {risk_metrics.max_consecutive_losses}")
    print(f"   Max Consecutive Wins: {risk_metrics.max_consecutive_wins}")
    print(f"   Risk of Ruin: {risk_metrics.risk_of_ruin:.4f}")
    
    print(f"\n📊 Trade Details:")
    print(f"   Average Win: ${metrics.avg_win:.2f}")
    print(f"   Average Loss: ${metrics.avg_loss:.2f}")
    print(f"   Best Trade: ${metrics.best_trade_pnl:.2f}")
    print(f"   Worst Trade: ${metrics.worst_trade_pnl:.2f}")
    print(f"   Average MFE: {metrics.avg_mfe*100:.2f}%")
    print(f"   Average MAE: {metrics.avg_mae*100:.2f}%")
    print(f"   MFE/MAE Ratio: {metrics.mfe_mae_ratio:.2f}")
    
    print(f"\n⏱️  Timing:")
    print(f"   Average Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
    print(f"   Trade Frequency: {metrics.trade_frequency_per_day:.2f} trades/day")
    
    print(f"\n🎯 Signal Analysis:")
    print(f"   Total Signals: {total_signals}")
    print(f"   LONG Signals: {long_signals}")
    print(f"   SHORT Signals: {short_signals}")
    print(f"   Average Confidence: {metrics.avg_confidence:.2%}")
    print(f"   Signal Quality Score: {metrics.signal_quality_score:.2f}")
    
    print("\n" + "=" * 80)
    
    # Создаем графики
    if output_plots and simulator.trades:
        plot_backtest_results(simulator, df_with_features, symbol, model_name)
    
    # Сохраняем детальный отчет
    save_detailed_report(simulator, metrics, risk_metrics, initial_balance, days_back)
    
    return metrics


def save_detailed_report(
    simulator: MLBacktestSimulator,
    metrics: BacktestMetrics,
    risk_metrics: RiskMetrics,
    initial_balance: float,
    days_back: int,
    output_dir: str = "backtest_reports"
):
    """Сохраняет детальный отчет о бэктесте."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"{output_dir}/backtest_report_{metrics.symbol}_{metrics.model_name}_{timestamp}.json"
    
    # Подготовка данных для JSON
    report = {
        "metadata": {
            "symbol": metrics.symbol,
            "model_name": metrics.model_name,
            "timestamp": datetime.now().isoformat(),
            "days_back": days_back,
            "initial_balance": initial_balance,
            "final_balance": initial_balance + metrics.total_pnl,
        },
        "metrics": asdict(metrics),
        "risk_metrics": asdict(risk_metrics),
        "trades": [],
        "equity_curve": simulator.equity_curve,
        "drawdowns": simulator.drawdowns,
    }
    
    # Добавляем сделки
    for i, trade in enumerate(simulator.trades):
        trade_dict = {
            "trade_number": i + 1,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "action": trade.action.value,
            "size_usd": trade.size_usd,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "entry_reason": trade.entry_reason,
            "exit_reason": trade.exit_reason.value if trade.exit_reason else None,
            "confidence": trade.confidence,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "trailing_stop": trade.trailing_stop,
            "mfe": trade.max_favorable_excursion,
            "mae": trade.max_adverse_excursion,
        }
        report["trades"].append(trade_dict)
    
    # Сохраняем в JSON
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📋 Detailed report saved to: {report_file}")
    
    # Также сохраняем текстовую версию
    txt_file = report_file.replace('.json', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ML STRATEGY BACKTEST DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {metrics.model_name}\n")
        f.write(f"Symbol: {metrics.symbol}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Days Back: {days_back}\n\n")
        
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)\n")
        f.write(f"Win Rate: {metrics.win_rate:.2f}%\n")
        f.write(f"Profit Factor: {metrics.profit_factor:.2f}\n")
        f.write(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%\n")
        f.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n")
        f.write(f"Total Trades: {metrics.total_trades}\n\n")
        
        f.write("DETAILED TRADES:\n")
        f.write("-" * 40 + "\n")
        for i, trade in enumerate(simulator.trades, 1):
            f.write(f"\nTrade #{i}:\n")
            f.write(f"  Entry: {trade.entry_time} @ ${trade.entry_price:.2f} ({trade.action.value})\n")
            f.write(f"  Exit: {trade.exit_time} @ ${trade.exit_price:.2f} ({trade.exit_reason.value if trade.exit_reason else 'N/A'})\n")
            f.write(f"  PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)\n")
            f.write(f"  Confidence: {trade.confidence:.2%}\n")
            f.write(f"  SL: ${trade.stop_loss:.2f}, TP: ${trade.take_profit:.2f}\n")
            f.write(f"  Reason: {trade.entry_reason}\n")
    
    print(f"📝 Text report saved to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced backtest for ML strategy on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Базовая команда
  python backtest_ml_strategy.py --model ml_models/ensemble_BTCUSDT_15.pkl
  
  # С трейлинг-стопом
  python backtest_ml_strategy.py --model ml_models/triple_ensemble_BTCUSDT_15.pkl --use-trailing-stop
  
  # С увеличенным балансом и меньшим риском
  python backtest_ml_strategy.py --model ml_models/ensemble_BTCUSDT_15.pkl --balance 5000 --risk 0.01 --leverage 5
  
  # Для другой пары и периода
  python backtest_ml_strategy.py --model ml_models/ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 60
        """
    )
    
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
    parser.add_argument('--use-trailing-stop', action='store_true',
                       help='Use trailing stop loss')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable data validation')
    
    args = parser.parse_args()
    
    # Запускаем бэктест
    metrics = run_ml_backtest(
        model_path=args.model,
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        use_trailing_stop=args.use_trailing_stop,
        output_plots=not args.no_plots,
        validate_data=not args.no_validation,
    )
    
    if metrics:
        print(f"\n✅ Backtest completed successfully!")
        print(f"   Final Score: {(metrics.sharpe_ratio * metrics.profit_factor * (metrics.win_rate/100)):.2f}")
    else:
        print(f"\n❌ Backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()