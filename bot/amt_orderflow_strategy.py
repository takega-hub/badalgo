from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from bot.exchange.bybit_client import BybitClient
from bot.strategy import Signal, Action


@dataclass
class AbsorptionConfig:
    """Параметры детекции сквиза на поглощении (упрощённая версия по order flow)."""

    lookback_seconds: int = 60            # окно анализа по тикам
    min_total_volume: float = 10_000.0    # минимум суммарного объёма в окне
    min_buy_sell_ratio: float = 2.0       # сколько раз покупок больше продаж для bear-squeeze (или наоборот для bull-squeeze)
    max_price_drift_pct: float = 0.05     # максимум движения цены в % при сильном CVD (стоим / топчемся)
    min_cvd_delta: float = 5_000.0        # минимальный прирост CVD за окно


def _parse_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует список тиков Bybit в DataFrame с колонками:
    time (datetime UTC), price, qty, side ('Buy'/'Sell').
    """
    if not raw_trades:
        return pd.DataFrame(columns=["time", "price", "qty", "side"])

    rows = []
    for t in raw_trades:
        try:
            # unified trading формат: time, side, qty/execQty, price/execPrice
            ts = int(t.get("time", t.get("T", 0)))
            price_str = t.get("price") or t.get("execPrice") or t.get("p")
            qty_str = t.get("execQty") or t.get("qty") or t.get("q")
            side = t.get("side") or t.get("S") or ""

            if not price_str or not qty_str or not side:
                continue

            price = float(price_str)
            qty = float(qty_str)
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)

            rows.append({"time": dt, "price": price, "qty": qty, "side": side})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["time", "price", "qty", "side"])

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def detect_absorption_squeeze_short(
    client: BybitClient,
    symbol: str,
    current_price: float,
    config: Optional[AbsorptionConfig] = None,
) -> Optional[Signal]:
    """
    Детекция сценария B: Сквиз на поглощении (Absorption Squeeze) в сторону SHORT.

    Интуиция (упрощённо, без полноценного footprint):
    - За последнее окно по времени:
      • Сильно преобладают рыночные покупки (CVD растёт, buy_volume >> sell_volume)
      • Цена почти не растёт или начинает откатываться (поглощение лимитами продавца)
    → Ожидаем вынос стопов лонгов и импульс вниз → SHORT.
    """
    cfg = config or AbsorptionConfig()

    # 1. Берём последние тики по инструменту
    raw_trades = client.get_recent_trades(symbol, limit=500)
    df_trades = _parse_trades(raw_trades)
    if df_trades.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - pd.Timedelta(seconds=cfg.lookback_seconds)
    window_df = df_trades[df_trades["time"] >= window_start]

    if window_df.empty:
        return None

    # 2. Считаем объёмы покупок/продаж и CVD
    buy_mask = window_df["side"].str.upper() == "BUY"
    sell_mask = window_df["side"].str.upper() == "SELL"

    buy_volume = float(window_df.loc[buy_mask, "qty"].sum())
    sell_volume = float(window_df.loc[sell_mask, "qty"].sum())
    total_volume = buy_volume + sell_volume

    if total_volume < cfg.min_total_volume:
        # Событие слишком слабое по объёму, игнорируем
        return None

    cvd_delta = buy_volume - sell_volume  # положительный → агрессивные покупки
    buy_sell_ratio = (buy_volume / max(sell_volume, 1e-6)) if sell_volume > 0 else np.inf

    # 3. Оцениваем движение цены в окне
    first_price = float(window_df["price"].iloc[0])
    last_price = float(window_df["price"].iloc[-1])
    price_change_pct = (last_price - first_price) / first_price * 100.0

    # 4. Условия "абсорбции" для SHORT:
    #    - Мощный приток покупок (CVD растёт, buy >> sell)
    #    - Цена почти не растёт или начинает откатываться
    if cvd_delta < cfg.min_cvd_delta:
        return None

    if buy_sell_ratio < cfg.min_buy_sell_ratio:
        return None

    if abs(price_change_pct) > cfg.max_price_drift_pct:
        # Цена уже сильно ушла, это не "стоячая" абсорбция
        return None

    # 5. Формируем сигнал SHORT
    ts = pd.Timestamp(now_utc)
    reason = (
        f"amt_of_absorption_short_cvd_{int(cvd_delta)}"
        f"_buy_vs_sell_{int(buy_volume)}/{int(sell_volume)}"
        f"_price_chg_{price_change_pct:.2f}%"
    )


def detect_absorption_squeeze_long(
    client: BybitClient,
    symbol: str,
    current_price: float,
    config: Optional[AbsorptionConfig] = None,
) -> Optional[Signal]:
    """
    Зеркальный сценарий B: Сквиз на поглощении (Absorption Squeeze) в сторону LONG.

    Интуиция:
    - За последнее окно по времени:
      • Сильно преобладают рыночные продажи (CVD падает, sell_volume >> buy_volume)
      • Цена почти не падает или начинает восстанавливаться (поглощение лимитами покупателя)
    → Ожидаем вынос стопов шортов и импульс вверх → LONG.
    """
    cfg = config or AbsorptionConfig()

    raw_trades = client.get_recent_trades(symbol, limit=500)
    df_trades = _parse_trades(raw_trades)
    if df_trades.empty:
        return None

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - pd.Timedelta(seconds=cfg.lookback_seconds)
    window_df = df_trades[df_trades["time"] >= window_start]

    if window_df.empty:
        return None

    # Объёмы покупок/продаж и CVD
    buy_mask = window_df["side"].str.upper() == "BUY"
    sell_mask = window_df["side"].str.upper() == "SELL"

    buy_volume = float(window_df.loc[buy_mask, "qty"].sum())
    sell_volume = float(window_df.loc[sell_mask, "qty"].sum())
    total_volume = buy_volume + sell_volume

    if total_volume < cfg.min_total_volume:
        return None

    cvd_delta = buy_volume - sell_volume  # отрицательный → агрессивные продажи
    sell_buy_ratio = (sell_volume / max(buy_volume, 1e-6)) if buy_volume > 0 else np.inf

    # Движение цены в окне
    first_price = float(window_df["price"].iloc[0])
    last_price = float(window_df["price"].iloc[-1])
    price_change_pct = (last_price - first_price) / first_price * 100.0

    # Условия "абсорбции" для LONG:
    #    - Мощный приток продаж (CVD падает, sell >> buy)
    #    - Цена почти не падает или начинает восстанавливаться
    if -cvd_delta < cfg.min_cvd_delta:
        return None

    if sell_buy_ratio < cfg.min_buy_sell_ratio:
        return None

    if abs(price_change_pct) > cfg.max_price_drift_pct:
        return None

    ts = pd.Timestamp(now_utc)
    reason = (
        f"amt_of_absorption_long_cvd_{int(cvd_delta)}"
        f"_sell_vs_buy_{int(sell_volume)}/{int(buy_volume)}"
        f"_price_chg_{price_change_pct:.2f}%"
    )

    return Signal(
        timestamp=ts,
        action=Action.LONG,
        reason=reason,
        price=current_price,
    )

    return Signal(
        timestamp=ts,
        action=Action.SHORT,
        reason=reason,
        price=current_price,
    )


__all__ = [
    "AbsorptionConfig",
    "detect_absorption_squeeze_short",
    "detect_absorption_squeeze_long",
]

