from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from bot.config import AppSettings
from bot.strategy import Action, Bias, Signal


@dataclass
class Position:
    side: Bias
    size_usd: float = 0.0
    avg_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_reason: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    side: Bias
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    entry_reason: Optional[str]
    exit_reason: str


class Simulator:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []

    def _open(self, side: Bias, price: float, usd: float, ts: pd.Timestamp, reason: str, sl: Optional[float] = None, tp: Optional[float] = None):
        if self.position and self.position.side != side:
            # close and flip
            self._close(price, "flip", ts)
        if not self.position:
            self.position = Position(side=side, size_usd=usd, avg_price=price, entry_time=ts, entry_reason=reason, stop_loss=sl, take_profit=tp)
        else:
            # averaging or pyramiding: recompute average price
            total_usd = self.position.size_usd + usd
            new_avg = (self.position.avg_price * self.position.size_usd + price * usd) / total_usd
            self.position.size_usd = min(total_usd, self.settings.risk.max_position_usd)
            self.position.avg_price = new_avg
            # Update SL/TP if provided in the new signal
            if sl: self.position.stop_loss = sl
            if tp: self.position.take_profit = tp

    def _close(self, price: float, reason: str, ts: pd.Timestamp):
        if not self.position:
            return
        # Сохраняем данные позиции перед закрытием
        position_side = self.position.side
        position_entry_time = self.position.entry_time
        position_avg_price = self.position.avg_price
        position_size_usd = self.position.size_usd
        position_entry_reason = self.position.entry_reason
        
        pnl = self._pnl(price)
        trade = Trade(
            entry_time=position_entry_time,
            exit_time=ts,
            side=position_side,
            entry_price=position_avg_price,
            exit_price=price,
            size_usd=position_size_usd,
            pnl=pnl,
            entry_reason=position_entry_reason,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None

    def _pnl(self, price: float) -> float:
        if not self.position:
            return 0.0
        delta = (price - self.position.avg_price) if self.position.side == Bias.LONG else (self.position.avg_price - price)
        raw_pnl = delta / self.position.avg_price * self.position.size_usd
        # Apply commission (approx 0.06% for entry and 0.06% for exit = 0.12% total)
        commission = self.position.size_usd * 0.0012
        return raw_pnl - commission

    def on_signal(self, sig: Signal):
        # Новая логика: только LONG, SHORT, HOLD
        # Time-exit enforcement: if position exists and sig contains max_hold_bars info,
        # we will check the age of the position in run() loop instead. on_signal keeps behavior minimal.
        if sig.action == Action.LONG:
            if not self.position:
                # Нет позиции → открываем LONG
                self._open(Bias.LONG, sig.price, self.settings.risk.base_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
            elif self.position.side == Bias.LONG:
                # Позиция LONG и сигнал LONG → добавляем к позиции
                self._open(Bias.LONG, sig.price, self.settings.risk.add_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
            elif self.position.side == Bias.SHORT:
                # Позиция SHORT и сигнал LONG → закрываем SHORT и открываем LONG
                self._close(sig.price, f"close_for_{sig.reason}", sig.timestamp)
                self._open(Bias.LONG, sig.price, self.settings.risk.base_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
        elif sig.action == Action.SHORT:
            if not self.position:
                # Нет позиции → открываем SHORT
                self._open(Bias.SHORT, sig.price, self.settings.risk.base_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
            elif self.position.side == Bias.SHORT:
                # Позиция SHORT и сигнал SHORT → добавляем к позиции
                self._open(Bias.SHORT, sig.price, self.settings.risk.add_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
            elif self.position.side == Bias.LONG:
                # Позиция LONG и сигнал SHORT → закрываем LONG и открываем SHORT
                self._close(sig.price, f"close_for_{sig.reason}", sig.timestamp)
                self._open(Bias.SHORT, sig.price, self.settings.risk.base_order_usd, sig.timestamp, sig.reason, sl=sig.stop_loss, tp=sig.take_profit)
        # HOLD - ничего не делаем

    def run(self, candles: pd.DataFrame, signals: List[Signal]) -> dict:
        # iterate over candles and process signals aligned by timestamp; support signals list with timestamps
        sig_by_ts = {sig.timestamp: sig for sig in signals}

        for ts in candles.index:
            # Ensure ts is Timestamp
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp(ts)

            # Check SL/TP and time-exit before processing today's signals
            if self.position:
                current_close = candles.loc[ts, 'close']
                current_high = candles.loc[ts, 'high'] if 'high' in candles.columns else current_close
                current_low = candles.loc[ts, 'low'] if 'low' in candles.columns else current_close

                # 1. Check Stop Loss
                if self.position.stop_loss:
                    if self.position.side == Bias.LONG and current_low <= self.position.stop_loss:
                        self._close(self.position.stop_loss, "SL_hit", ts)
                        continue
                    elif self.position.side == Bias.SHORT and current_high >= self.position.stop_loss:
                        self._close(self.position.stop_loss, "SL_hit", ts)
                        continue

                # 2. Check Take Profit
                if self.position.take_profit:
                    if self.position.side == Bias.LONG and current_high >= self.position.take_profit:
                        self._close(self.position.take_profit, "TP_hit", ts)
                        continue
                    elif self.position.side == Bias.SHORT and current_low <= self.position.take_profit:
                        self._close(self.position.take_profit, "TP_hit", ts)
                        continue

                # 3. Check Time Exit
                if self.position and self.position.entry_time is not None:
                    try:
                        entry_idx = candles.index.get_loc(self.position.entry_time)
                        current_idx = candles.index.get_loc(ts)
                        bars_held = current_idx - entry_idx
                    except Exception:
                        bars_held = None

                    max_hold = None
                    last_sig = sig_by_ts.get(self.position.entry_time)
                    if last_sig and last_sig.indicators_info:
                        max_hold = last_sig.indicators_info.get('max_hold_bars')
                    if max_hold is None:
                        max_hold = getattr(self.settings, 'default_max_hold_bars', 20)

                    if bars_held is not None and max_hold is not None and bars_held > int(max_hold):
                        self._close(current_close, f'time_exit_{int(max_hold)}', ts)
                        continue

            # process a signal if exists for this timestamp
            sig = sig_by_ts.get(ts)
            if sig:
                self.on_signal(sig)
        # Close leftover position at last close
        # Close leftover position at last close
        if self.position:
            last_idx = candles.index[-1]
            last_price = candles["close"].iloc[-1]
            # Убеждаемся, что last_idx - это Timestamp
            if not isinstance(last_idx, pd.Timestamp):
                last_idx = pd.Timestamp(last_idx)
            self._close(last_price, "end_of_backtest", last_idx)

        total_pnl = sum(t.pnl for t in self.trades)
        return {
            "trades": self.trades,
            "total_pnl": total_pnl,
            "positions_closed": len(self.trades),
        }
