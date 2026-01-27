"""
Бэктест стратегии Z-Score на исторических данных для всех символов.

Тестирует Z-Score стратегию на исторических данных с расчетом:
- Общего PnL
- Винрейта
- Profit Factor
- Максимальной просадки
- Детальных логов для анализа
- Рекомендаций по улучшению стратегии
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings, StrategyParams
from bot.zscore_strategy import build_zscore_signals
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, Signal


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


@dataclass
class BacktestMetrics:
    """Метрики бэктеста."""
    symbol: str
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


@dataclass
class BacktestRecommendation:
    """Рекомендация по улучшению стратегии."""
    category: str  # "risk", "entry", "exit", "filter", "parameter"
    priority: str  # "high", "medium", "low"
    message: str
    suggestion: str


class ZScoreBacktestSimulator:
    """Симулятор для бэктеста Z-Score стратегии."""
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.02,  # 2% риска на сделку
        commission: float = 0.0006,  # 0.06% комиссия Bybit
        max_position_size_pct: float = 0.1,  # Максимальный размер позиции 10%
        max_consecutive_losses: int = 10,  # Увеличено: больше терпения для серий убытков (было 6, слишком строго)
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.max_position_size_pct = max_position_size_pct
        self.max_consecutive_losses = max_consecutive_losses
        
        self.position: Optional[Dict[str, Any]] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.symbol = ""
        self.consecutive_losses = 0  # Счетчик последовательных убытков
        self.trading_paused = False  # Флаг остановки торговли
        
    def _calculate_position_size(self, entry_price: float, stop_loss_price: float, action: Action) -> float:
        """Рассчитывает размер позиции на основе риска."""
        if action == Action.LONG:
            risk_per_unit = abs(entry_price - stop_loss_price)
        else:  # SHORT
            risk_per_unit = abs(stop_loss_price - entry_price)
        
        if risk_per_unit == 0:
            return 0.0
        
        risk_amount = self.balance * self.risk_per_trade
        position_size_usd = risk_amount / (risk_per_unit / entry_price)
        
        # Ограничиваем максимальным размером позиции
        max_size = self.balance * self.max_position_size_pct
        return min(position_size_usd, max_size)
    
    def _calculate_stop_loss(self, entry_price: float, atr: float, action: Action, atr_multiplier: float = 1.0) -> float:
        """Рассчитывает Stop Loss на основе ATR.
        
        ЭКСТРЕННОЕ ИСПРАВЛЕНИЕ: Уменьшено до 1.0 ATR для меньших потерь.
        При Win Rate 37% и TP/SL = 2.0/1.0 = 2.0 нужен Win Rate минимум 33.3% для безубыточности.
        Это должно улучшить соотношение средних прибыль/убыток.
        """
        if action == Action.LONG:
            return entry_price - (atr * atr_multiplier)
        else:  # SHORT
            return entry_price + (atr * atr_multiplier)
    
    def _calculate_take_profit(self, entry_price: float, atr: float, action: Action, atr_multiplier: float = 2.0) -> float:
        """Рассчитывает Take Profit на основе ATR.
        
        ЭКСТРЕННОЕ ИСПРАВЛЕНИЕ: TP/SL = 2.0/1.0 = 2.0 для лучшего соотношения риск/прибыль.
        При Win Rate 37% и соотношении 2.0:1 стратегия должна быть прибыльной.
        TP увеличен для компенсации низкого Win Rate.
        """
        if action == Action.LONG:
            return entry_price + (atr * atr_multiplier)
        else:  # SHORT
            return entry_price - (atr * atr_multiplier)
    
    def open_position(
        self,
        signal: Signal,
        current_price: float,
        current_time: datetime,
        atr: float,
        df: pd.DataFrame,
    ) -> bool:
        """Открывает позицию по сигналу."""
        if self.position is not None:
            return False  # Уже есть открытая позиция
        
        # Рассчитываем SL и TP
        sl_price = self._calculate_stop_loss(current_price, atr, signal.action)
        tp_price = self._calculate_take_profit(current_price, atr, signal.action)
        
        # Если POC указан в reason, используем его как TP
        if "_poc_" in signal.reason:
            try:
                poc_part = signal.reason.split("_poc_")[-1]
                # Убираем уникальный идентификатор если есть
                poc_value = float(poc_part.split("_")[0])
                if signal.action == Action.LONG:
                    tp_price = max(tp_price, poc_value)  # Используем больший из TP и POC
                else:
                    tp_price = min(tp_price, poc_value)  # Используем меньший из TP и POC
            except (ValueError, IndexError):
                pass  # Используем стандартный TP
        
        # Рассчитываем размер позиции
        size_usd = self._calculate_position_size(current_price, sl_price, signal.action)
        
        if size_usd <= 0:
            return False
        
        self.position = {
            "entry_time": current_time,
            "entry_price": current_price,
            "action": signal.action,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "size_usd": size_usd,
            "entry_reason": signal.reason,
            "symbol": self.symbol,
        }
        
        return True
    
    def check_exit(
        self,
        current_price: float,
        current_time: datetime,
        df: pd.DataFrame,
    ) -> Optional[str]:
        """Проверяет условия выхода из позиции."""
        if self.position is None:
            return None
        
        pos = self.position
        
        # Проверка Take Profit
        if pos["action"] == Action.LONG:
            if current_price >= pos["tp_price"]:
                return "TP"
            if current_price <= pos["sl_price"]:
                return "SL"
        else:  # SHORT
            if current_price <= pos["tp_price"]:
                return "TP"
            if current_price >= pos["sl_price"]:
                return "SL"
        
        # Проверка выхода по Z-Score (возврат к среднему)
        # Это будет обработано в основном цикле через сигналы EXIT
        
        return None
    
    def close_position(self, exit_price: float, exit_time: datetime, exit_reason: str):
        """Закрывает позицию и записывает сделку."""
        if self.position is None:
            return
        
        pos = self.position
        
        # Рассчитываем PnL
        if pos["action"] == Action.LONG:
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:  # SHORT
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
        
        # Учитываем комиссию (вход и выход)
        pnl_pct -= (self.commission * 2)
        
        pnl = pos["size_usd"] * pnl_pct
        
        # Обновляем баланс
        self.balance += pnl
        
        # Записываем сделку
        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            action=pos["action"],
            size_usd=pos["size_usd"],
            pnl=pnl,
            pnl_pct=pnl_pct * 100,
            entry_reason=pos["entry_reason"],
            exit_reason=exit_reason,
            symbol=pos["symbol"],
        )
        
        self.trades.append(trade)
        self.equity_curve.append((exit_time, self.balance))
        
        # ОБНОВЛЕНО: Отслеживаем последовательные убытки для защиты от серий
        # Улучшенная логика: останавливаемся при комбинации убытков и просадки
        if pnl < 0:
            self.consecutive_losses += 1
            drawdown_pct = abs(self.initial_balance - self.balance) / self.initial_balance * 100
            
            # Останавливаемся если:
            # 1. Достигнут лимит убытков И просадка > 5% (ослаблено для большего количества сделок)
            # 2. ИЛИ просадка критическая (> 15% - ослаблено)
            if (self.consecutive_losses >= self.max_consecutive_losses and drawdown_pct > 5.0) or drawdown_pct > 15.0:
                self.trading_paused = True
                print(f"   ⚠️ Trading paused after {self.consecutive_losses} consecutive losses (drawdown: {drawdown_pct:.2f}%)")
        else:
            self.consecutive_losses = 0  # Сбрасываем счетчик при прибыльной сделке
        
        # Сбрасываем позицию
        self.position = None
    
    def run(
        self,
        df: pd.DataFrame,
        signals: List[Signal],
        symbol: str,
    ) -> Dict[str, Any]:
        """Запускает симуляцию бэктеста."""
        self.symbol = symbol
        self.trades = []
        self.equity_curve = [(df.index[0], self.initial_balance)]
        self.position = None
        self.consecutive_losses = 0  # Сбрасываем счетчик убытков
        self.trading_paused = False  # Сбрасываем флаг остановки
        
        # ОПТИМИЗАЦИЯ: Сортируем сигналы по времени и создаем индекс для быстрого поиска
        # Нормализуем timestamp всех сигналов
        normalized_signals = []
        for sig in signals:
            if isinstance(sig.timestamp, pd.Timestamp):
                ts = sig.timestamp
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                normalized_signals.append((ts, sig))
        
        # Сортируем по времени для бинарного поиска
        normalized_signals.sort(key=lambda x: x[0])
        
        # Создаем индекс сигналов для быстрого поиска
        signal_times = [ts for ts, _ in normalized_signals]
        signal_index = 0  # Индекс текущего сигнала
        
        # Обрабатываем каждую свечу
        total_candles = len(df)
        for candle_idx, (idx, row) in enumerate(df.iterrows()):
            # Прогресс каждые 5000 свечей
            if candle_idx > 0 and candle_idx % 5000 == 0:
                print(f"   Processing candle {candle_idx}/{total_candles} ({candle_idx/total_candles*100:.1f}%)...")
            
            current_time = idx
            if isinstance(current_time, pd.Timestamp):
                if current_time.tzinfo is None:
                    current_time = current_time.tz_localize('UTC')
                else:
                    current_time = current_time.tz_convert('UTC')
            
            current_price = float(row['close'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            
            # Получаем ATR для расчета SL/TP
            atr = float(row.get('atr', 0.0)) if 'atr' in row else 0.0
            if atr == 0 or pd.isna(atr):
                # Если ATR нет, используем 1% от цены как приближение
                atr = current_price * 0.01
            
            # ОПТИМИЗАЦИЯ: Ищем сигнал для текущей свечи используя отсортированный список
            # Пропускаем сигналы, которые уже прошли (время меньше текущего)
            signal = None
            
            # Пропускаем сигналы, которые слишком далеко в прошлом (более 1 минуты)
            while signal_index < len(signal_times):
                sig_time = signal_times[signal_index]
                time_diff_seconds = (current_time - sig_time).total_seconds()
                
                # Если сигнал слишком далеко в прошлом (более 1 минуты), переходим к следующему
                if time_diff_seconds > 60:
                    signal_index += 1
                    continue
                
                # Если сигнал в пределах 1 минуты от текущей свечи (в прошлом или будущем) - используем его
                if abs(time_diff_seconds) <= 60:
                    signal = normalized_signals[signal_index][1]
                    # Если сигнал в прошлом (более 15 минут), увеличиваем индекс, чтобы не проверять его снова
                    if time_diff_seconds > 900:  # 15 минут
                        signal_index += 1
                    break
                
                # Если сигнал в будущем (более 1 минуты), останавливаемся
                if time_diff_seconds < -60:
                    break
                
                # Если мы здесь, значит сигнал очень близок к текущему времени
                signal = normalized_signals[signal_index][1]
                break
            
            # Если есть открытая позиция, проверяем условия выхода
            if self.position is not None:
                # Проверяем TP/SL на high/low свечи
                exit_reason = None
                exit_price = None
                
                pos = self.position
                if pos["action"] == Action.LONG:
                    if high_price >= pos["tp_price"]:
                        exit_reason = "TP"
                        exit_price = pos["tp_price"]
                    elif low_price <= pos["sl_price"]:
                        exit_reason = "SL"
                        exit_price = pos["sl_price"]
                else:  # SHORT
                    if low_price <= pos["tp_price"]:
                        exit_reason = "TP"
                        exit_price = pos["tp_price"]
                    elif high_price >= pos["sl_price"]:
                        exit_reason = "SL"
                        exit_price = pos["sl_price"]
                
                # Проверяем сигнал EXIT (выход по Z-Score)
                if signal:
                    signal_reason_upper = signal.reason.upper()
                    if "EXIT" in signal_reason_upper or "exit" in signal_reason_upper:
                        # Проверяем, что сигнал EXIT соответствует направлению позиции
                        if (pos["action"] == Action.LONG and "EXIT_LONG" in signal_reason_upper) or \
                           (pos["action"] == Action.SHORT and "EXIT_SHORT" in signal_reason_upper):
                            exit_reason = "Z-Score Exit"
                            exit_price = current_price
                
                if exit_reason:
                    self.close_position(exit_price, current_time, exit_reason)
                    # После закрытия позиции пропускаем открытие новой на этой свече
                    continue
            
            # Если нет позиции и есть сигнал входа, открываем позицию
            # ПРОВЕРКА: Не открываем новые позиции если торговля приостановлена из-за серии убытков
            if self.position is None and signal and not self.trading_paused:
                if signal.action in (Action.LONG, Action.SHORT):
                    self.open_position(signal, current_price, current_time, atr, df)
            
            # Обновляем кривую эквити
            if self.position is None:
                self.equity_curve.append((current_time, self.balance))
            else:
                # Рассчитываем текущую стоимость позиции
                pos = self.position
                if pos["action"] == Action.LONG:
                    unrealized_pnl = (current_price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
                else:
                    unrealized_pnl = (pos["entry_price"] - current_price) / pos["entry_price"] * pos["size_usd"]
                self.equity_curve.append((current_time, self.balance + unrealized_pnl))
        
        # Закрываем последнюю позицию если она открыта
        if self.position is not None:
            last_price = float(df.iloc[-1]['close'])
            last_time = df.index[-1]
            if isinstance(last_time, pd.Timestamp):
                if last_time.tzinfo is None:
                    last_time = last_time.tz_localize('UTC')
                else:
                    last_time = last_time.tz_convert('UTC')
            self.close_position(last_price, last_time, "End of data")
        
        return {
            "trades": self.trades,
            "final_balance": self.balance,
            "equity_curve": self.equity_curve,
        }


def calculate_metrics(
    trades: List[Trade],
    initial_balance: float,
    signals: List[Signal],
    symbol: str,
) -> BacktestMetrics:
    """Рассчитывает метрики бэктеста."""
    if not trades:
        return BacktestMetrics(
            symbol=symbol,
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
            total_signals=len(signals),
            long_signals=len([s for s in signals if s.action == Action.LONG]),
            short_signals=len([s for s in signals if s.action == Action.SHORT]),
            avg_trade_duration_hours=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            largest_win=0.0,
            largest_loss=0.0,
        )
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    total_pnl = sum(t.pnl for t in trades)
    total_pnl_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    total_wins = sum(t.pnl for t in winning_trades)
    total_losses = abs(sum(t.pnl for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
    
    # Расчет максимальной просадки
    cumulative_pnl = np.cumsum([t.pnl for t in trades])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    max_drawdown_pct = (max_drawdown / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    # Расчет Sharpe Ratio
    if len(trades) > 1:
        returns = [t.pnl / initial_balance for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # Средняя длительность сделки
    durations = []
    for t in trades:
        if t.entry_time and t.exit_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600
            durations.append(duration)
    avg_duration = np.mean(durations) if durations else 0.0
    
    # Лучшая и худшая сделки
    best_trade = max(trades, key=lambda t: t.pnl) if trades else None
    worst_trade = min(trades, key=lambda t: t.pnl) if trades else None
    
    # Серии побед и поражений
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for t in trades:
        if t.pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    return BacktestMetrics(
        symbol=symbol,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate=(len(winning_trades) / len(trades)) * 100 if trades else 0.0,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        total_signals=len(signals),
        long_signals=len([s for s in signals if s.action == Action.LONG]),
        short_signals=len([s for s in signals if s.action == Action.SHORT]),
        avg_trade_duration_hours=avg_duration,
        best_trade_pnl=best_trade.pnl if best_trade else 0.0,
        worst_trade_pnl=worst_trade.pnl if worst_trade else 0.0,
        consecutive_wins=max_consecutive_wins,
        consecutive_losses=max_consecutive_losses,
        largest_win=best_trade.pnl if best_trade else 0.0,
        largest_loss=worst_trade.pnl if worst_trade else 0.0,
    )


def generate_recommendations(metrics: BacktestMetrics, trades: List[Trade]) -> List[BacktestRecommendation]:
    """Генерирует рекомендации по улучшению стратегии."""
    recommendations = []
    
    # Анализ винрейта
    if metrics.win_rate < 40:
        recommendations.append(BacktestRecommendation(
            category="entry",
            priority="high",
            message=f"Низкий винрейт: {metrics.win_rate:.1f}%",
            suggestion="Рассмотрите ужесточение фильтров входа (ADX, RSI, объем). Возможно, стоит увеличить пороги Z-Score для более сильных сигналов."
        ))
    elif metrics.win_rate > 60:
        recommendations.append(BacktestRecommendation(
            category="entry",
            priority="low",
            message=f"Высокий винрейт: {metrics.win_rate:.1f}%",
            suggestion="Хороший винрейт. Рассмотрите возможность увеличения размера позиций или уменьшения SL для увеличения прибыли."
        ))
    
    # Анализ Profit Factor
    if metrics.profit_factor < 1.0:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="high",
            message=f"Profit Factor ниже 1.0: {metrics.profit_factor:.2f}",
            suggestion="Стратегия убыточна. Пересмотрите логику входа и выхода. Возможно, стоит увеличить TP/SL соотношение или улучшить фильтры."
        ))
    elif metrics.profit_factor < 1.5:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="medium",
            message=f"Profit Factor низкий: {metrics.profit_factor:.2f}",
            suggestion="Рассмотрите оптимизацию соотношения TP/SL. Увеличьте TP или уменьшите SL для улучшения соотношения риск/прибыль."
        ))
    
    # Анализ максимальной просадки
    if metrics.max_drawdown_pct > 30:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="high",
            message=f"Большая просадка: {metrics.max_drawdown_pct:.1f}%",
            suggestion="Уменьшите размер позиций или увеличьте диверсификацию. Рассмотрите добавление фильтров для избежания торговли в неблагоприятных условиях."
        ))
    
    # Анализ средних прибылей и убытков
    if metrics.avg_loss != 0 and abs(metrics.avg_win / metrics.avg_loss) < 1.5:
        recommendations.append(BacktestRecommendation(
            category="exit",
            priority="medium",
            message=f"Соотношение средних прибыль/убыток низкое: {abs(metrics.avg_win / metrics.avg_loss):.2f}",
            suggestion="Увеличьте Take Profit или уменьшите Stop Loss. Рассмотрите использование трейлинг-стопа для защиты прибыли."
        ))
    
    # Анализ серий убытков
    if metrics.consecutive_losses > 5:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="medium",
            message=f"Длинная серия убытков: {metrics.consecutive_losses}",
            suggestion="Добавьте механизм остановки торговли после серии убытков. Рассмотрите фильтр по рыночным условиям (тренд/флэт)."
        ))
    
    # Анализ количества сигналов
    if metrics.total_signals == 0:
        recommendations.append(BacktestRecommendation(
            category="parameter",
            priority="high",
            message="Нет сигналов",
            suggestion="Стратегия не генерирует сигналы. Проверьте параметры Z-Score (пороги, фильтры). Возможно, условия слишком строгие."
        ))
    elif metrics.total_signals < 10:
        recommendations.append(BacktestRecommendation(
            category="parameter",
            priority="low",
            message=f"Мало сигналов: {metrics.total_signals}",
            suggestion="Рассмотрите смягчение условий входа для увеличения количества сделок, если это не ухудшит качество."
        ))
    
    # Анализ длительности сделок
    if metrics.avg_trade_duration_hours > 48:
        recommendations.append(BacktestRecommendation(
            category="exit",
            priority="low",
            message=f"Долгие сделки: {metrics.avg_trade_duration_hours:.1f} часов",
            suggestion="Рассмотрите более агрессивные условия выхода. Возможно, стоит уменьшить TP или добавить временной фильтр."
        ))
    
    return recommendations


def load_historical_data(symbol: str, timeframe: str = "15m", data_dir: str = "data") -> Optional[pd.DataFrame]:
    """Загружает исторические данные из CSV файла."""
    # Пробуем разные варианты имен файлов
    possible_paths = [
        f"{data_dir}/{symbol.lower()}_{timeframe}.csv",
        f"{data_dir}/{symbol[:3].lower()}_{timeframe}.csv",
        f"{data_dir}/{symbol}_{timeframe}.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Определяем колонку времени
                time_col = None
                for col in ['datetime', 'timestamp', 'Time', 'time', 'open_time']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if time_col:
                    if df[time_col].dtype == object:
                        df['datetime'] = pd.to_datetime(df[time_col])
                    else:
                        # Если это числа (мс), указываем unit='ms'
                        unit = 'ms' if df[time_col].iloc[0] > 1e12 else 's'
                        df['datetime'] = pd.to_datetime(df[time_col], unit=unit)
                    df = df.set_index('datetime')
                else:
                    # Если нет колонки времени, создаем индекс из порядкового номера
                    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
                
                # Проверяем наличие необходимых колонок
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"⚠️  Missing columns for {symbol}: {missing_cols}")
                    return None
                
                # Сортируем по времени
                df = df.sort_index()
                
                return df
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                return None
    
    return None


def run_zscore_backtest(
    symbols: List[str],
    timeframe: str = "15m",
    initial_balance: float = 1000.0,
    risk_per_trade: float = 0.02,
    data_dir: str = "data",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Запускает бэктест Z-Score стратегии для всех символов."""
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"zscore_backtest_{timestamp}.log")
    
    all_results = {}
    all_metrics = []
    all_recommendations = []
    
    print(f"\n{'='*80}")
    print(f"Z-SCORE STRATEGY BACKTEST")
    print(f"{'='*80}")
    print(f"Timestamp: {timestamp}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Risk per Trade: {risk_per_trade*100:.1f}%")
    print(f"{'='*80}\n")
    
    # Загружаем настройки
    settings = load_settings()
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Testing {symbol}")
        print(f"{'='*80}")
        
        # Загружаем данные
        df = load_historical_data(symbol, timeframe, data_dir)
        if df is None or df.empty:
            print(f"⚠️  No data found for {symbol}. Skipping...")
            continue
        
        print(f"✅ Loaded {len(df)} candles for {symbol}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Подготавливаем индикаторы
        try:
            df_ready = prepare_with_indicators(
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
                ema_timeframe=settings.strategy.momentum_ema_timeframe,
            )
            print(f"✅ Indicators prepared")
        except Exception as e:
            print(f"❌ Error preparing indicators for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Генерируем сигналы Z-Score
        # ВАЖНО: В бэктесте всегда тестируем Z-Score, независимо от настроек включения/выключения
        try:
            # Выводим параметры Z-Score для диагностики
            print(f"\n📊 Z-Score Parameters for {symbol}:")
            # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ для улучшения Win Rate:
            # - Ужесточены пороги Z-Score для более сильных сигналов
            # - Ужесточен ADX threshold для фильтрации трендовых рынков
            # - Ужесточен volume factor для лучшего качества сигналов
            # - Ужесточены RSI пороги для более экстремальных условий
            zscore_params = {
                'window': getattr(settings.strategy, 'zscore_window', getattr(settings.strategy, 'sma_length', 20)),
                'z_long': getattr(settings.strategy, 'zscore_long', -2.5),  # Будет переопределено ниже
                'z_short': getattr(settings.strategy, 'zscore_short', 2.5),  # Будет переопределено ниже
                'z_exit': getattr(settings.strategy, 'zscore_exit', 0.5),
                'adx_threshold': getattr(settings.strategy, 'zscore_adx_threshold', 25.0),  # Будет переопределено ниже
                'vol_factor': getattr(settings.strategy, 'zscore_vol_factor', 0.8),  # Будет переопределено ниже
                'rsi_enabled': getattr(settings.strategy, 'zscore_rsi_enabled', True),
                'rsi_long_threshold': 30.0,  # Оригинальное значение: стандартная перепроданность (больше сигналов)
                'rsi_short_threshold': 70.0,  # Оригинальное значение: стандартная перекупленность (больше сигналов)
            }
            
            # ПЕРЕОПРЕДЕЛЕНИЕ параметров для баланса между количеством и качеством
            # СТРАТЕГИЯ: умеренное ужесточение фильтров для улучшения Win Rate при сохранении количества сделок
            # Цель: ~300-400 сделок в год с Win Rate > 35% и Profit Factor > 1.0
            # ⚡ ВРЕМЕННО УПРОЩЕННЫЕ ПАРАМЕТРЫ ДЛЯ ТЕСТИРОВАНИЯ
            # Отключаем фильтры для диагностики проблемы с отсутствием сигналов
            zscore_params['z_long'] = -1.0  # Очень мягкий порог для тестирования
            zscore_params['z_short'] = 1.0   # Очень мягкий порог для тестирования
            zscore_params['adx_threshold'] = 100.0  # Фактически отключаем ADX фильтр
            zscore_params['vol_factor'] = 0.0  # Отключаем фильтр объема
            zscore_params['rsi_enabled'] = False  # Отключаем RSI фильтр
            zscore_params['rsi_long_threshold'] = 28.0  # Не используется при rsi_enabled=False
            zscore_params['rsi_short_threshold'] = 72.0  # Не используется при rsi_enabled=False
            
            print(f"\n   ⚡ TEMPORARY DEBUG PARAMETERS:")
            print(f"   ⚡ z_long={zscore_params['z_long']}, z_short={zscore_params['z_short']}")
            print(f"   ⚡ ADX filter DISABLED (threshold={zscore_params['adx_threshold']})")
            print(f"   ⚡ Volume filter DISABLED (factor={zscore_params['vol_factor']})")
            print(f"   ⚡ RSI filter DISABLED")
            
            for key, value in zscore_params.items():
                print(f"   {key}: {value}")
            
            # Проверяем наличие необходимых колонок для Z-Score
            required_cols = ['close', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in df_ready.columns]
            if missing_cols:
                print(f"⚠️  Missing columns for Z-Score: {missing_cols}")
            
            # Проверяем наличие индикаторов
            indicator_cols = ['sma', 'atr', 'adx', 'rsi']
            available_indicators = [col for col in indicator_cols if col in df_ready.columns]
            print(f"   Available indicators: {available_indicators}")
            
            # ВАЖНО: Для бэктеста используем v2_generate_signals напрямую, чтобы получить все сигналы
            # build_zscore_signals фильтрует только последние 3 свечи, что не подходит для бэктеста
            from bot.zscore_strategy_v2 import generate_signals as v2_generate_signals
            from bot.zscore_strategy import _map_config_to_v2
            
            # Маппим параметры
            v2_params = _map_config_to_v2(settings.strategy)
            
            # ПЕРЕОПРЕДЕЛЯЕМ параметры для максимального количества сигналов
            v2_params.z_long = zscore_params['z_long']
            v2_params.z_short = zscore_params['z_short']
            v2_params.adx_threshold = zscore_params['adx_threshold']
            v2_params.vol_factor = zscore_params['vol_factor']
            v2_params.rsi_enabled = zscore_params['rsi_enabled']  # Отключаем RSI если нужно
            if zscore_params['rsi_enabled']:
                v2_params.rsi_long_threshold = zscore_params['rsi_long_threshold']
                v2_params.rsi_short_threshold = zscore_params['rsi_short_threshold']
            # ⚡ ВРЕМЕННО УПРОЩЕННЫЕ ПАРАМЕТРЫ ДЛЯ ТЕСТИРОВАНИЯ
            v2_params.sma_slope_threshold = 0.01  # Очень либеральный фильтр тренда для тестирования
            
            # ВРЕМЕННО ОТКЛЮЧАЕМ новые фильтры для диагностики
            v2_params.min_volatility = 0.0  # Отключаем фильтр волатильности
            v2_params.exclude_hours = []  # Отключаем фильтр времени (пустой список)
            v2_params.use_dynamic_sl_tp = False  # Отключаем динамические SL/TP для простоты
            v2_params.require_confirmation = False  # ⚠️ ОТКЛЮЧАЕМ подтверждение сигналов - возможно, это блокирует все сигналы!
            
            # Обновляем TP/SL для лучшего соотношения
            v2_params.stop_loss_atr = 1.0  # Уменьшено для меньших потерь
            v2_params.take_profit_atr = 2.0  # Увеличено для лучшего соотношения TP/SL = 2.0
            
            print(f"   ⚡ SMA slope threshold: {v2_params.sma_slope_threshold}")
            print(f"   ⚡ Volatility filter DISABLED (min_volatility={v2_params.min_volatility})")
            print(f"   ⚡ Time filter DISABLED (exclude_hours={v2_params.exclude_hours})")
            print(f"   ⚡ Signal confirmation DISABLED (require_confirmation={v2_params.require_confirmation})")
            
            print(f"   ✅ Applied optimized parameters: z_long={v2_params.z_long}, z_short={v2_params.z_short}, "
                  f"adx_threshold={v2_params.adx_threshold}, vol_factor={v2_params.vol_factor}, "
                  f"rsi_enabled={v2_params.rsi_enabled}, sma_slope_threshold={v2_params.sma_slope_threshold}")
            
            # 🔍 ДИАГНОСТИКА - Проверка расчета Z-Score перед генерацией сигналов
            print(f"\n🔍 DIAGNOSTIC - Z-Score calculation check:")
            print(f"   DataFrame shape: {df_ready.shape}")
            print(f"   Close values: min={df_ready['close'].min():.2f}, max={df_ready['close'].max():.2f}")
            
            # Генерируем сигналы для всего DataFrame
            df_signals = v2_generate_signals(df_ready.copy(), v2_params)
            
            if df_signals is None or df_signals.empty:
                print(f"❌ v2_generate_signals returned None or empty DataFrame")
                signals = []
            else:
                print(f"✅ v2_generate_signals returned DataFrame with shape: {df_signals.shape}")
                if 'z' in df_signals.columns:
                    print(f"   ✅ 'z' column found! Z-Score range: [{df_signals['z'].min():.2f}, {df_signals['z'].max():.2f}]")
                    
                    # 🔍 ДИАГНОСТИКА - Детальная проверка расчета Z-Score
                    if 'sma' in df_signals.columns and 'std' in df_signals.columns:
                        print(f"   SMA values: min={df_signals['sma'].min():.2f}, max={df_signals['sma'].max():.2f}")
                        print(f"   STD values: min={df_signals['std'].min():.2f}, max={df_signals['std'].max():.2f}")
                    
                    # Проверяем кандидатов на вход
                    long_candidates = df_signals['z'] <= v2_params.z_long
                    short_candidates = df_signals['z'] >= v2_params.z_short
                    print(f"   Long candidates (z <= {v2_params.z_long}): {long_candidates.sum()} rows")
                    print(f"   Short candidates (z >= {v2_params.z_short}): {short_candidates.sum()} rows")
                    
                    print(f"   Last 5 Z-Score values:")
                    display_cols = ['close', 'z', 'signal', 'reason']
                    available_cols = [col for col in display_cols if col in df_signals.columns]
                    print(df_signals[available_cols].tail())
                    
                    # Подсчитываем сигналы
                    long_signals_count = len(df_signals[df_signals['signal'] == 'LONG'])
                    short_signals_count = len(df_signals[df_signals['signal'] == 'SHORT'])
                    exit_long_count = len(df_signals[df_signals['signal'] == 'EXIT_LONG'])
                    exit_short_count = len(df_signals[df_signals['signal'] == 'EXIT_SHORT'])
                    print(f"   Signals in DataFrame: LONG={long_signals_count}, SHORT={short_signals_count}, EXIT_LONG={exit_long_count}, EXIT_SHORT={exit_short_count}")
                else:
                    print(f"   ⚠️  'z' column not found. Columns: {list(df_signals.columns)}")
                
                # Преобразуем DataFrame сигналов в список Signal объектов
                # Используем логику из build_zscore_signals, но без фильтрации по последним 3 свечам
                from bot.strategy import Signal as StrategySignal
                signals = []
                
                for idx, row in df_signals.iterrows():
                    sig = str(row.get("signal", "")).upper()
                    if sig == "LONG":
                        action = Action.LONG
                    elif sig == "SHORT":
                        action = Action.SHORT
                    elif sig in ("EXIT_LONG", "EXIT_SHORT"):
                        # Для бэктеста обрабатываем EXIT сигналы как HOLD с специальным reason
                        action = Action.HOLD
                    else:
                        continue
                    
                    raw_reason = row.get("reason") or ""
                    if raw_reason and not raw_reason.startswith("zscore_"):
                        reason = f"zscore_{raw_reason}"
                    elif raw_reason:
                        reason = raw_reason
                    else:
                        reason = f"zscore_{sig.lower()}"
                    
                    price = float(row.get("close", row.get("price", float('nan'))))
                    
                    try:
                        ts = pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx
                    except Exception:
                        ts = pd.Timestamp.now()
                    
                    signals.append(StrategySignal(timestamp=ts, action=action, reason=str(reason), price=price))
                
                print(f"✅ Converted to {len(signals)} Signal objects")
            
            # 🔍 ДИАГНОСТИКА - Проверка сигналов
            print(f"\n🔍 DIAGNOSTIC - Checking signals for {symbol}:")
            print(f"   Total signals generated: {len(signals)}")
            entry_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
            exit_signals = [s for s in signals if s.action == Action.HOLD and "EXIT" in s.reason.upper()]
            print(f"   Entry signals (LONG/SHORT): {len(entry_signals)}")
            print(f"      - LONG: {len([s for s in entry_signals if s.action == Action.LONG])}")
            print(f"      - SHORT: {len([s for s in entry_signals if s.action == Action.SHORT])}")
            print(f"   Exit signals (HOLD with EXIT): {len(exit_signals)}")
            
            # Выводим первые 10 сигналов для диагностики
            if signals:
                print(f"\n   First 10 signals:")
                for i, sig in enumerate(signals[:10]):
                    print(f"   {i+1}: Time={sig.timestamp}, Action={sig.action}, Reason={sig.reason}, Price={sig.price}")
            else:
                print("   ⚠️ NO SIGNALS GENERATED!")
            
            # Проверяем df_signals если нет входных сигналов
            if len(entry_signals) == 0 and 'df_signals' in locals() and df_signals is not None:
                print(f"\n   ⚠️ NO ENTRY SIGNALS! Checking df_signals:")
                print(f"   df_signals columns: {list(df_signals.columns)}")
                if 'z' in df_signals.columns:
                    print(f"   'z' column exists. Min={df_signals['z'].min():.2f}, Max={df_signals['z'].max():.2f}")
                    print(f"   'signal' column unique values: {df_signals['signal'].unique()}")
                    
                    # Проверяем кандидатов на вход
                    long_candidates = df_signals['z'] <= v2_params.z_long
                    short_candidates = df_signals['z'] >= v2_params.z_short
                    print(f"   Long candidates (z <= {v2_params.z_long}): {long_candidates.sum()} rows")
                    print(f"   Short candidates (z >= {v2_params.z_short}): {short_candidates.sum()} rows")
                    
                    # Проверяем фильтры
                    if 'market_allowed' in df_signals.columns:
                        print(f"   market_allowed (ADX < {v2_params.adx_threshold}): {df_signals['market_allowed'].sum()} rows")
                    if 'sma_flat' in df_signals.columns:
                        print(f"   sma_flat: {df_signals['sma_flat'].sum()} rows")
                    if 'vol_ok' in df_signals.columns:
                        print(f"   vol_ok: {df_signals['vol_ok'].sum()} rows")
                    if 'volatility_ok' in df_signals.columns:
                        print(f"   volatility_ok: {df_signals['volatility_ok'].sum()} rows")
                    if 'time_ok' in df_signals.columns:
                        print(f"   time_ok: {df_signals['time_ok'].sum()} rows")
                    if 'signal_confirmed' in df_signals.columns:
                        confirmed_signals = df_signals[df_signals['signal'].isin(['LONG', 'SHORT'])]
                        if len(confirmed_signals) > 0:
                            print(f"   signal_confirmed for LONG/SHORT: {confirmed_signals['signal_confirmed'].sum()} rows")
                            print(f"   signal_confirmed FALSE: {(~confirmed_signals['signal_confirmed']).sum()} rows")
                    
                    # Проверяем последние 5 строк
                    print(f"\n   Last 5 rows of df_signals:")
                    cols_to_show = ['close', 'z', 'signal', 'reason', 'adx', 'rsi']
                    if 'signal_confirmed' in df_signals.columns:
                        cols_to_show.append('signal_confirmed')
                    available_cols = [c for c in cols_to_show if c in df_signals.columns]
                    print(df_signals[available_cols].tail())
            
            # Если сигналов нет, выводим диагностику
            if len(signals) == 0:
                print(f"\n⚠️  DIAGNOSTIC: No signals generated for {symbol}")
                print(f"   DataFrame shape: {df_ready.shape}")
                print(f"   Last 5 rows Z-Score values:")
                if 'z' in df_ready.columns:
                    cols_to_show = ['close']
                    for col in ['z', 'sma', 'adx', 'rsi']:
                        if col in df_ready.columns:
                            cols_to_show.append(col)
                    print(df_ready[cols_to_show].tail())
                else:
                    print("   ⚠️  'z' column not found in DataFrame - Z-Score calculation may have failed")
                    print(f"   Available columns: {list(df_ready.columns)[:20]}")
        except Exception as e:
            print(f"❌ Error generating signals for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Запускаем симуляцию
        simulator = ZScoreBacktestSimulator(
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
        )
        
        try:
            result = simulator.run(df_ready, signals, symbol)
            trades = result["trades"]
            print(f"✅ Simulation completed: {len(trades)} trades")
            
            # Добавляем информацию о приостановке торговли
            if simulator.trading_paused:
                print(f"   ⚠️ Trading was paused after {simulator.consecutive_losses} consecutive losses")
                print(f"   Total trades before pause: {len(trades)}")
        except Exception as e:
            print(f"❌ Error running simulation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Рассчитываем метрики
        metrics = calculate_metrics(trades, initial_balance, signals, symbol)
        all_metrics.append(metrics)
        
        # Генерируем рекомендации
        recommendations = generate_recommendations(metrics, trades)
        all_recommendations.extend([(symbol, rec) for rec in recommendations])
        
        # Выводим результаты
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {symbol}")
        print(f"{'='*80}")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Winning Trades: {metrics.winning_trades}")
        print(f"Losing Trades: {metrics.losing_trades}")
        print(f"Win Rate: {metrics.win_rate:.2f}%")
        print(f"Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:.2f}%)")
        print(f"Final Balance: ${result['final_balance']:.2f}")
        print(f"Average Win: ${metrics.avg_win:.2f}")
        print(f"Average Loss: ${metrics.avg_loss:.2f}")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Best Trade: ${metrics.best_trade_pnl:.2f}")
        print(f"Worst Trade: ${metrics.worst_trade_pnl:.2f}")
        print(f"Max Consecutive Wins: {metrics.consecutive_wins}")
        print(f"Max Consecutive Losses: {metrics.consecutive_losses}")
        print(f"Average Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
        
        if recommendations:
            print(f"\n📋 RECOMMENDATIONS FOR {symbol}:")
            for rec in recommendations:
                print(f"   [{rec.priority.upper()}] {rec.category.upper()}: {rec.message}")
                print(f"      → {rec.suggestion}")
        
        # Сохраняем результаты
        all_results[symbol] = {
            "metrics": metrics,
            "trades": trades,
            "signals": len(signals),
            "recommendations": recommendations,
        }
    
    # Агрегированные результаты
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    total_trades = sum(m.total_trades for m in all_metrics)
    total_winning = sum(m.winning_trades for m in all_metrics)
    total_losing = sum(m.losing_trades for m in all_metrics)
    total_pnl = sum(m.total_pnl for m in all_metrics)
    total_pnl_pct = (total_pnl / (initial_balance * len(symbols))) * 100 if symbols else 0
    
    print(f"Total Symbols Tested: {len(all_metrics)}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Winning: {total_winning}")
    print(f"Total Losing: {total_losing}")
    print(f"Overall Win Rate: {(total_winning / total_trades * 100) if total_trades > 0 else 0:.2f}%")
    print(f"Total PnL: ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
    
    # Сохраняем детальный отчет
    report_file = os.path.join(output_dir, f"zscore_backtest_report_{timestamp}.csv")
    report_data = []
    for metrics in all_metrics:
        report_data.append({
            "Symbol": metrics.symbol,
            "Total Trades": metrics.total_trades,
            "Winning Trades": metrics.winning_trades,
            "Losing Trades": metrics.losing_trades,
            "Win Rate %": f"{metrics.win_rate:.2f}",
            "Total PnL": f"${metrics.total_pnl:.2f}",
            "Total PnL %": f"{metrics.total_pnl_pct:.2f}",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Max Drawdown": f"${metrics.max_drawdown:.2f}",
            "Max Drawdown %": f"{metrics.max_drawdown_pct:.2f}",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
            "Avg Win": f"${metrics.avg_win:.2f}",
            "Avg Loss": f"${metrics.avg_loss:.2f}",
            "Best Trade": f"${metrics.best_trade_pnl:.2f}",
            "Worst Trade": f"${metrics.worst_trade_pnl:.2f}",
            "Avg Duration Hours": f"{metrics.avg_trade_duration_hours:.1f}",
            "Total Signals": metrics.total_signals,
            "Long Signals": metrics.long_signals,
            "Short Signals": metrics.short_signals,
        })
    
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(report_file, index=False)
    print(f"\n✅ Detailed report saved: {report_file}")
    
    # Сохраняем все сделки
    all_trades_data = []
    for symbol, result in all_results.items():
        for trade in result["trades"]:
            all_trades_data.append({
                "Symbol": trade.symbol,
                "Entry Time": trade.entry_time.isoformat() if trade.entry_time else "",
                "Exit Time": trade.exit_time.isoformat() if trade.exit_time else "",
                "Action": trade.action.value,
                "Entry Price": trade.entry_price,
                "Exit Price": trade.exit_price,
                "Size USD": trade.size_usd,
                "PnL": trade.pnl,
                "PnL %": trade.pnl_pct,
                "Entry Reason": trade.entry_reason,
                "Exit Reason": trade.exit_reason,
            })
    
    if all_trades_data:
        trades_file = os.path.join(output_dir, f"zscore_backtest_trades_{timestamp}.csv")
        df_trades = pd.DataFrame(all_trades_data)
        df_trades.to_csv(trades_file, index=False)
        print(f"✅ Trades saved: {trades_file}")
    
    # Сохраняем рекомендации
    if all_recommendations:
        rec_file = os.path.join(output_dir, f"zscore_backtest_recommendations_{timestamp}.txt")
        with open(rec_file, 'w', encoding='utf-8') as f:
            f.write("Z-SCORE STRATEGY BACKTEST RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            for symbol, rec in all_recommendations:
                f.write(f"[{symbol}] [{rec.priority.upper()}] {rec.category.upper()}\n")
                f.write(f"  Issue: {rec.message}\n")
                f.write(f"  Suggestion: {rec.suggestion}\n\n")
        
        print(f"✅ Recommendations saved: {rec_file}")
    
    return {
        "results": all_results,
        "aggregated_metrics": {
            "total_trades": total_trades,
            "total_winning": total_winning,
            "total_losing": total_losing,
            "overall_win_rate": (total_winning / total_trades * 100) if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
        },
        "recommendations": all_recommendations,
    }


def main():
    """Главная функция для запуска бэктеста."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Z-Score Strategy Backtest")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to test (default: BTCUSDT ETHUSDT SOLUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe for backtest (default: 15m)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Initial balance (default: 1000.0)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.02,
        help="Risk per trade as fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with historical data (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    
    args = parser.parse_args()
    
    run_zscore_backtest(
        symbols=args.symbols,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
