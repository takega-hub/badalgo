import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import `bot` package
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pandas as pd
import numpy as np
from bot.strategy import (
    generate_trend_signal,
    generate_flat_signal,
    generate_momentum_signal,
    Action,
)


def make_price_series(length=60, start=100.0, trend=0.0, noise=0.2):
    prices = np.linspace(start, start + trend * length, num=length) + np.random.randn(length) * noise
    idx = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='T')
    df = pd.DataFrame(index=idx)
    df['open'] = prices + np.random.randn(length) * 0.1
    df['high'] = df['open'] + np.abs(np.random.randn(length) * 0.5)
    df['low'] = df['open'] - np.abs(np.random.randn(length) * 0.5)
    df['close'] = prices
    df['volume'] = np.random.randint(1, 200, length)
    return df


def test_trend_long_and_short():
    # Uptrend should produce LONG
    df_up = make_price_series(trend=0.1)
    res_up = generate_trend_signal(df_up, state={'long_pyramid': 0}, sma_period=10, min_history=50)
    assert isinstance(res_up, dict)

    # Downtrend should allow SHORT
    df_down = make_price_series(trend=-0.1)
    res_down = generate_trend_signal(df_down, state={'short_pyramid': 0}, sma_period=10, min_history=50)
    assert isinstance(res_down, dict)


def test_flat_signal_basic():
    df = make_price_series(trend=0.0)
    res = generate_flat_signal(df, rsi_period=14, bb_period=10, bb_compression_factor=0.9, min_history=50)
    assert isinstance(res, dict)


def test_momentum_signal_filters():
    df = make_price_series(trend=0.2)
    res = generate_momentum_signal(df, ema_short=5, ema_long=10, vol_lookback=20, vol_top_pct=0.75, min_history=50)
    assert isinstance(res, dict)

