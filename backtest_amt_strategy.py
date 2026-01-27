import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from dataclasses import dataclass

from bot.exchange.bybit_client import BybitClient
from bot.config import ApiSettings
from bot.amt_orderflow_strategy import get_signals_for_symbol, calculate_atr, set_amt_debug, get_amt_debug_stats
from bot.strategy import Signal, Action
from bot.logger_config import log as bot_log

class TrailingStop:
    def __init__(self, be_atr_mult=1.0, ts_activation_atr_mult=1.8, trail_atr_mult=1.0):
        self.be_mult = be_atr_mult; self.ts_activation_mult = ts_activation_atr_mult; self.trail_mult = trail_atr_mult
        self.activated = False; self.be_activated = False; self.best_price = 0.0; self.current_stop = 0.0
        self.entry_price = 0.0; self.action = None; self.atr = 0.0

    def reset(self, entry_price: float, action: Action, initial_sl: float, atr: float):
        self.entry_price = entry_price; self.best_price = entry_price; self.action = action; self.current_stop = initial_sl
        self.atr = atr; self.activated = False; self.be_activated = False

    def update(self, current_price: float):
        if self.action == Action.LONG:
            profit = current_price - self.entry_price
            if not self.be_activated and profit >= self.atr * self.be_mult:
                self.be_activated = True; self.current_stop = self.entry_price
            if current_price > self.best_price:
                self.best_price = current_price
                if not self.activated and profit >= self.atr * self.ts_activation_mult: self.activated = True
                if self.activated: self.current_stop = max(self.current_stop, self.best_price - (self.atr * self.trail_mult))
        else:
            profit = self.entry_price - current_price
            if not self.be_activated and profit >= self.atr * self.be_mult:
                self.be_activated = True; self.current_stop = self.entry_price
            if current_price < self.best_price:
                self.best_price = current_price
                if not self.activated and profit >= self.atr * self.ts_activation_mult: self.activated = True
                if self.activated: self.current_stop = min(self.current_stop, self.best_price + (self.atr * self.trail_mult))

    def should_close(self, current_price: float) -> bool:
        return (current_price <= self.current_stop) if self.action == Action.LONG else (current_price >= self.current_stop)

@dataclass
class AMTBacktestPosition:
    entry_price: float; entry_time: datetime; action: Action; tp_price: float; sl_price: float; qty: float; trailing: TrailingStop
    signal_reason: str = ""

class AMTBacktestSimulator:
    def __init__(self, initial_balance=100.0, risk_per_trade=0.15, cooldown_minutes=60):
        self.balance = initial_balance; self.risk_per_trade = risk_per_trade; self.position = None
        self.history = []; self.cooldown_seconds = cooldown_minutes * 60; self.last_close_time = None

    def _check_tp_sl(self, current_price: float, current_time: datetime):
        if not self.position: return
        pos = self.position; pos.trailing.update(current_price)
        if pos.trailing.should_close(current_price): self._close_position(current_price, current_time, "Trailing/BE"); return
        if (pos.action == Action.LONG and current_price >= pos.tp_price) or \
           (pos.action == Action.SHORT and current_price <= pos.tp_price):
            self._close_position(pos.tp_price, current_time, "TP_hit")

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        pos = self.position
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.action == Action.LONG else (pos.entry_price - exit_price) / pos.entry_price
        pnl_val = self.balance * self.risk_per_trade * pnl_pct
        self.balance += pnl_val
        self.history.append({
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "action": pos.action.value,
            "pnl_pct": pnl_pct * 100,
            "pnl_val": pnl_val,
            "exit_reason": reason,
            "signal_reason": pos.signal_reason,
        })
        self.position = None; self.last_close_time = exit_time

    def _on_price(self, price: float, ts: datetime) -> None:
        """Process one intra-candle price touch."""
        if not self.position:
            return
        pos = self.position
        pos.trailing.update(price)
        if pos.trailing.should_close(price):
            self._close_position(price, ts, "Trailing/BE")
            return
        if (pos.action == Action.LONG and price >= pos.tp_price) or (pos.action == Action.SHORT and price <= pos.tp_price):
            self._close_position(pos.tp_price, ts, "TP_hit")
            return

    def run(self, df_candles: pd.DataFrame, signals: List[Signal], symbol: str = ""):
        sig_map = {s.timestamp: s for s in signals}
        for ts, candle in df_candles.iterrows():
            if self.position:
                # More realistic intra-candle ordering:
                # - LONG: price tends to go high before sweeping low on breakouts
                # - SHORT: price tends to go low before sweeping high
                if self.position.action == Action.LONG:
                    self._on_price(candle["high"], ts)
                    if self.position:
                        self._on_price(candle["low"], ts)
                else:
                    self._on_price(candle["low"], ts)
                    if self.position:
                        self._on_price(candle["high"], ts)
                continue
            if self.last_close_time and (ts - self.last_close_time).total_seconds() < self.cooldown_seconds: continue
            if ts in sig_map:
                sig = sig_map[ts]
                atr = candle.get('atr', sig.price * 0.005)
                tp_dist = atr * 3.0; sl_dist = atr * 1.5
                tp = sig.price + tp_dist if sig.action == Action.LONG else sig.price - tp_dist
                sl = sig.price - sl_dist if sig.action == Action.LONG else sig.price + sl_dist
                # Trailing stop настройки: BE на 1.5 ATR, активация на 2.5 ATR, расстояние 1.0 ATR
                # Для BTC: большее расстояние trailing (1.2 ATR) чтобы не так быстро закрывало
                if symbol == "BTCUSDT":
                    tr = TrailingStop(be_atr_mult=1.5, ts_activation_atr_mult=2.5, trail_atr_mult=1.2)
                else:
                    tr = TrailingStop(be_atr_mult=1.5, ts_activation_atr_mult=2.5, trail_atr_mult=1.0)
                tr.reset(sig.price, sig.action, sl, atr)
                self.position = AMTBacktestPosition(
                    entry_price=sig.price,
                    entry_time=ts,
                    action=sig.action,
                    tp_price=tp,
                    sl_price=sl,
                    qty=self.balance * self.risk_per_trade / sig.price,
                    trailing=tr,
                    signal_reason=str(getattr(sig, "reason", "")),
                )

def backtest_amt_strategy(symbol="ETHUSDT", days_back=7, initial_balance=100.0):
    api = ApiSettings(); client = BybitClient(api=api)
    df_candles = client.get_kline_df(symbol, "15", days_back * 24 * 60)
    if df_candles.empty: return None
    df_candles["time"] = pd.to_datetime(df_candles["timestamp"], unit="ms", utc=True); df_candles = df_candles.set_index("time").sort_index()
    
    # ATR
    cp_prev = df_candles['close'].shift(1)
    tr = pd.concat([df_candles['high'] - df_candles['low'], abs(df_candles['high'] - cp_prev), abs(df_candles['low'] - cp_prev)], axis=1).max(axis=1)
    df_candles['atr'] = tr.rolling(14).mean()

    all_signals = []
    for i in range(100, len(df_candles)):
        ts = df_candles.index[i]; c = df_candles.iloc[i]; o, h, l, cl = c['open'], c['high'], c['low'], c['close']
        
        # Симуляция 2-х типов свечей: Обычная и "Поглощение"
        # Поглощение должно выглядеть так:
        # - ABS_LONG: доминируют SELL, но цена НЕ падает (чаще bullish/neutral candle)
        # - ABS_SHORT: доминируют BUY, но цена НЕ растёт (чаще bearish/neutral candle)
        is_absorption_scenario = np.random.random() < 0.25
        trades = []
        if is_absorption_scenario:
            bullish = cl >= o
            # Force “reversal pressure” vs price behavior
            main_side = "Sell" if bullish else "Buy"
            for t in range(30):
                # keep body small: price oscillates near mid, with slight drift toward close
                mid = (h + l) / 2.0
                noise = (np.random.random() - 0.5) * (h - l) * 0.08
                p = mid + noise + (cl - mid) * (t / 30.0) * 0.6
                # create “whale-ish” spike once, but not extreme
                qty_mult = 6.0 if t == 15 else 1.0
                qty = (c['volume'] / 30.0) * (0.8 + np.random.random() * 0.8) * qty_mult
                trades.append({"time": ts - timedelta(seconds=(30-t)*6), "price": p, "qty": qty, "side": main_side})
        else:
            # Обычный тренд
            main_side = "Buy" if cl > o else "Sell"
            for t in range(30):
                # trend path + small noise
                base = o + (cl - o) * (t / 30.0)
                p = base + (np.random.random() - 0.5) * (h - l) * 0.03
                side = main_side if np.random.random() < 0.8 else ("Sell" if main_side == "Buy" else "Buy")
                qty = (c['volume'] / 30.0) * (0.7 + np.random.random() * 0.6)
                trades.append({"time": ts - timedelta(seconds=(30-t)*6), "price": p, "qty": qty, "side": side})
        
        candle_trades = pd.DataFrame(trades); candle_trades["time"] = pd.to_datetime(candle_trades["time"], utc=True)
        sigs = get_signals_for_symbol(client, symbol, cl, df_candles.iloc[max(0, i-100):i+1], None, trades_df=candle_trades, current_time=ts)
        all_signals.extend(sigs)
    
    sim = AMTBacktestSimulator(initial_balance=initial_balance); sim.run(df_candles, all_signals, symbol=symbol); return sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--debug", action="store_true", help="Print AMT signal filter debug counters")
    args = parser.parse_args(); symbols = args.symbols.split(","); summary = []
    if args.debug:
        set_amt_debug(True, reset=True)
    sims_by_symbol = {}
    for sym in symbols:
        sim = backtest_amt_strategy(symbol=sym, days_back=args.days)
        if not sim: continue
        sims_by_symbol[sym] = sim
        pnl = sim.balance - 100
        wins = [h for h in sim.history if h['pnl_pct'] > 0]
        losses = [h for h in sim.history if h['pnl_pct'] <= 0]
        win_rate = len(wins) / len(sim.history) * 100 if sim.history else 0
        pf = abs(sum([h['pnl_val'] for h in wins]) / sum([h['pnl_val'] for h in losses])) if losses else 0
        summary.append({"Symbol": sym, "Trades": len(sim.history), "WinRate": f"{win_rate:.1f}%", "PnL": f"${pnl:.2f}", "PnL %": f"{pnl:.2f}%", "PF": f"{pf:.2f}"})
    print("\n📊 ИТОГОВЫЙ ОТЧЕТ ЗА " + str(args.days) + " ДНЕЙ\n" + "="*80)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n💰 ОБЩИЙ PnL: $" + f"{sum([float(s['PnL'].replace('$', '')) for s in summary]):.2f}")
    if args.debug:
        stats = get_amt_debug_stats(reset=False)
        if stats:
            print("\n" + "="*80)
            print("🧪 AMT DEBUG (почему фильтруются сигналы)")
            print("="*80)
            top = sorted(stats.items(), key=lambda kv: kv[1], reverse=True)[:35]
            for k, v in top:
                print(f"{k:32s} : {v}")

            # Always show these key counters if present (even if not in top)
            print("\n— Key counters —")
            keys_of_interest = [
                "BTCUSDT:brk:accepted", "ETHUSDT:brk:accepted", "SOLUSDT:brk:accepted",
                "BTCUSDT:abs:accepted", "ETHUSDT:abs:accepted", "SOLUSDT:abs:accepted",
                "BTCUSDT:brk:reject:clearance_long", "BTCUSDT:brk:reject:clearance_short",
                "ETHUSDT:brk:reject:clearance_long", "ETHUSDT:brk:reject:clearance_short",
            ]
            for k in keys_of_interest:
                if k in stats:
                    print(f"{k:32s} : {stats[k]}")

    # Extra: PnL by reason prefix (abs vs brk) per symbol
    print("\n" + "="*80)
    print("📌 PnL по типам сигналов (abs/brk)")
    print("="*80)
    for sym in symbols:
        sim = sims_by_symbol.get(sym)
        if not sim or not sim.history:
            print(f"{sym}: no trades")
            continue
        by_type = {"abs": [], "brk": [], "other": []}
        for h in sim.history:
            reason = str(h.get("signal_reason", "")).lower()
            if reason.startswith("abs_") or "abs_" in reason:
                by_type["abs"].append(h["pnl_val"])
            elif reason.startswith("brk_") or "brk_" in reason:
                by_type["brk"].append(h["pnl_val"])
            else:
                by_type["other"].append(h["pnl_val"])
        def ssum(xs): return float(sum(xs)) if xs else 0.0
        print(f"{sym}: abs={ssum(by_type['abs']):+.2f}  brk={ssum(by_type['brk']):+.2f}  other={ssum(by_type['other']):+.2f}")

    # Save all trades to CSV for deeper offline analysis
    all_rows = []
    for sym, sim in sims_by_symbol.items():
        for h in sim.history:
            row = dict(h)
            row["symbol"] = sym
            all_rows.append(row)
    if all_rows:
        out_path = "logs/amt_backtest_trades.csv"
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print("\nSaved trades to:", out_path)
