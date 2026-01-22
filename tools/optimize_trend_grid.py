#!/usr/bin/env python3
"""
Grid search optimizer for TREND strategy parameters.

This script runs a simple grid search by setting environment variables
that `bot.config.load_settings()` reads, then invoking the existing
`test_strategy_silent` function from `generate_report.py` to evaluate
performance for each parameter combination.

Usage:
  python tools/optimize_trend_grid.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 7

Note: This runs the same tests as generate_report and may take time
depending on the grid size and number of symbols.
"""

import os
import sys
import argparse
import itertools
from datetime import datetime
import pathlib

# Ensure project root is on sys.path so we can import generate_report when running
# this script as `python tools/optimize_trend_grid.py` (sys.path[0] would be tools/)
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from generate_report import test_strategy_silent


def run_grid(symbols, days, grid):
    results = []
    total = len(symbols) * len(grid)
    i = 0
    for sym in symbols:
        for params in grid:
            i += 1
            sma, atr, adx, vol, pyr = params
            # Set env vars so load_settings() picks them up
            os.environ['SMA_LENGTH'] = str(sma)
            os.environ['TREND_ATR_MULTIPLIER'] = str(atr)
            os.environ['ADX_THRESHOLD'] = str(adx)
            os.environ['BREAKOUT_VOLUME_MULT'] = str(vol)
            os.environ['TREND_MAX_PYRAMID'] = str(pyr)

            print(f"[{i}/{total}] Testing {sym} with SMA={sma}, ATR={atr}, ADX={adx}, VOL={vol}, PYR={pyr}...", flush=True)
            res = test_strategy_silent('trend', sym, days)
            results.append((sym, params, res))
    return results


def summarize(results, top_n=5):
    by_symbol = {}
    for sym, params, res in results:
        by_symbol.setdefault(sym, []).append((params, res))

    summary = {}
    for sym, items in by_symbol.items():
        # sort by total_pnl desc
        items_sorted = sorted(items, key=lambda x: getattr(x[1], 'total_pnl', float('-inf')), reverse=True)
        summary[sym] = items_sorted[:top_n]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()

    # Define small grid (expand if needed)
    sma_list = [20, 30]
    atr_list = [2.5, 3.0, 3.5]
    adx_list = [20, 25, 30]
    vol_list = [1.0, 1.5]
    pyr_list = [0, 1]

    grid = list(itertools.product(sma_list, atr_list, adx_list, vol_list, pyr_list))

    start = datetime.now()
    results = run_grid(args.symbols, args.days, grid)
    summary = summarize(results, top_n=5)

    print('\n=== GRID SEARCH SUMMARY ===')
    for sym, entries in summary.items():
        print(f'\nTop results for {sym}:')
        for params, res in entries:
            sma, atr, adx, vol, pyr = params
            err = res.error if hasattr(res, 'error') else None
            print(f"SMA={sma} ATR={atr} ADX={adx} VOL={vol} PYR={pyr} -> Trades={res.total_trades} Signals={res.signals_count} PnL={res.total_pnl:+.2f} WR={res.win_rate:.1f}% Error={err}")

    elapsed = datetime.now() - start
    print(f"\nGrid search finished in {elapsed}")


if __name__ == '__main__':
    main()

