from dataclasses import dataclass
from enum import Enum
from typing import Optional
import hashlib

import numpy as np
import pandas as pd

from bot.config import StrategyParams


class Bias(Enum):
    LONG = "long"
    SHORT = "short"
    RANGE = "range"


class MarketPhase(Enum):
    TREND = "trend"  # Трендовое движение (ADX > threshold)
    FLAT = "flat"    # Флэтовое движение (ADX <= threshold)


class Action(Enum):
    HOLD = "hold"  # Не показываем в списке, просто игнорируем
    LONG = "long"  # Сигнал на покупку
    SHORT = "short"  # Сигнал на продажу


@dataclass
class Signal:
    timestamp: pd.Timestamp
    action: Action
    reason: str
    price: float
    signal_id: Optional[str] = None  # Уникальный ID сигнала
    indicators_info: Optional[dict] = None  # Детальная информация о показателях для логирования
    stop_loss: Optional[float] = None  # Рекомендуемый Stop Loss
    take_profit: Optional[float] = None  # Рекомендуемый Take Profit
    
    def __post_init__(self):
        """Генерирует уникальный ID сигнала, если он не задан."""
        if self.signal_id is None:
            # Генерируем ID на основе timestamp, action, reason и price
            # ВАЖНО: Используем больше знаков для price и добавляем больше данных для уникальности
            ts_str = str(self.timestamp) if hasattr(self.timestamp, 'isoformat') else str(self.timestamp)
            price_str = f"{self.price:.6f}"  # Увеличено с 4 до 6 знаков для большей точности
            # Добавляем больше данных для уникальности (timestamp уже включает время)
            id_string = f"{ts_str}_{self.action.value}_{self.reason}_{price_str}"
            self.signal_id = hashlib.md5(id_string.encode()).hexdigest()[:16]  # 16 символов MD5


def detect_market_phase(row: pd.Series, params: StrategyParams) -> MarketPhase:
    """
    Определяет фазу рынка: тренд или флэт на основе ADX.
    """
    adx = row.get("adx", np.nan)
    if np.isnan(adx):
        return MarketPhase.FLAT
    if adx > params.adx_threshold:
        return MarketPhase.TREND
    return MarketPhase.FLAT


def infer_bias(row: pd.Series, params: StrategyParams) -> Bias:
    """
    Определяет направление bias на основе ADX и DI.
    Используется только в трендовой фазе.
    """
    adx = row.get("adx", np.nan)
    plus_di = row.get("plus_di", np.nan)
    minus_di = row.get("minus_di", np.nan)
    if np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di):
        return Bias.RANGE
    if adx <= params.adx_threshold:
        return Bias.RANGE
    if plus_di > minus_di:
        return Bias.LONG
    if minus_di > plus_di:
        return Bias.SHORT
    return Bias.RANGE


def _volume_ok(row: pd.Series, mult: float) -> bool:
    vol = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    return np.isfinite(vol) and np.isfinite(vol_sma) and vol > vol_sma * mult


def _fakeout_recovery_long(row: pd.Series) -> bool:
    # Simple approximation: previous low pierced recent support, current close reclaims support and closes above prev close.
    support = row.get("recent_low", np.nan)
    prev_low = row.get("prev_low", np.nan)
    prev_close = row.get("prev_close", np.nan)
    close = row.get("close", np.nan)
    if not np.isfinite([support, prev_low, prev_close, close]).all():
        return False
    return prev_low < support and close > support and close > prev_close


def _fakeout_recovery_short(row: pd.Series) -> bool:
    resistance = row.get("recent_high", np.nan)
    prev_high = row.get("prev_high", np.nan)
    prev_close = row.get("prev_close", np.nan)
    close = row.get("close", np.nan)
    if not np.isfinite([resistance, prev_high, prev_close, close]).all():
        return False
    return prev_high > resistance and close < resistance and close < prev_close


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Безопасное деление с проверкой на ноль и NaN.
    
    Args:
        numerator: Числитель
        denominator: Знаменатель
        default: Значение по умолчанию, если деление невозможно
    
    Returns:
        Результат деления или default
    """
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return default
    result = numerator / denominator
    return result if np.isfinite(result) else default


def _format_vol_ratio(volume: float, vol_sma: float, precision: int = 2) -> Optional[float]:
    """
    Форматирует отношение объема к Volume SMA с проверкой на валидность.
    
    Args:
        volume: Объем
        vol_sma: Volume SMA
        precision: Точность округления
    
    Returns:
        Отношение volume/vol_sma или None если деление невозможно
    """
    if vol_sma > 0 and np.isfinite(vol_sma) and np.isfinite(volume):
        return round(volume / vol_sma, precision)
    return None


def generate_trend_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams) -> Signal:
    """
    Трендовая стратегия: используется при ADX > threshold.
    Логика: breakout, fakeout recovery, усреднение на pullback, пирамидинг на консолидации.
    """
    bias = row["bias"]
    price = row["close"]
    vol_ok = _volume_ok(row, params.breakout_volume_mult)
    
    # Собираем информацию о показателях
    adx = row.get("adx", np.nan)
    plus_di = row.get("plus_di", np.nan)
    minus_di = row.get("minus_di", np.nan)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    recent_high = row.get("recent_high", np.nan)
    recent_low = row.get("recent_low", np.nan)
    
    # MACD анализ
    macd = row.get("macd", np.nan)
    macd_signal = row.get("macd_signal", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    macd_bullish = np.isfinite(macd) and np.isfinite(macd_signal) and macd > macd_signal and macd > 0
    macd_bearish = np.isfinite(macd) and np.isfinite(macd_signal) and macd < macd_signal and macd < 0
    macd_hist_positive = np.isfinite(macd_hist) and macd_hist > 0
    macd_hist_negative = np.isfinite(macd_hist) and macd_hist < 0

    # Breakout entries с подтверждением MACD
    long_breakout = (
        bias == Bias.LONG
        and vol_ok
        and price > row.get("recent_high", np.inf)
        and (macd_bullish or not np.isfinite(macd))  # MACD подтверждает или не доступен
    )
    short_breakout = (
        bias == Bias.SHORT
        and vol_ok
        and price < row.get("recent_low", -np.inf)
        and (macd_bearish or not np.isfinite(macd))  # MACD подтверждает или не доступен
    )

    # Fakeout recovery entries
    long_fakeout = bias == Bias.LONG and _fakeout_recovery_long(row) and vol_ok
    short_fakeout = bias == Bias.SHORT and _fakeout_recovery_short(row) and vol_ok

    # Scaling: averaging on pullback to SMA20
    close = row["close"]
    sma = row.get("sma", np.nan)
    pullback_band = params.pullback_tolerance
    near_sma = np.isfinite(sma) and abs(close - sma) / sma <= pullback_band
    volume_spike = _volume_ok(row, params.volume_spike_mult) or (
        np.isfinite(row.get("prev_volume", np.nan))
        and np.isfinite(row.get("vol_sma", np.nan))
        and max(row["volume"], row["prev_volume"]) > row["vol_sma"] * params.volume_spike_mult
    )

    long_scale_avg = bias == Bias.LONG and has_position == Bias.LONG and near_sma and volume_spike
    short_scale_avg = bias == Bias.SHORT and has_position == Bias.SHORT and near_sma and volume_spike

    # Scaling: pyramiding on consolidation breakout
    rng_high = row.get("consolidation_high", np.nan)
    rng_low = row.get("consolidation_low", np.nan)
    consolidation = row.get("is_consolidating", False)
    rsi = row.get("rsi", np.nan)
    rsi_ok = np.isfinite(rsi) and params.rsi_floor <= rsi <= params.rsi_ceiling
    long_pyramid = (
        bias == Bias.LONG
        and has_position == Bias.LONG
        and consolidation
        and price > rng_high
        and vol_ok
        and rsi_ok
    )
    short_pyramid = (
        bias == Bias.SHORT
        and has_position == Bias.SHORT
        and consolidation
        and price < rng_low
        and vol_ok
        and rsi_ok
    )

    # Определяем сигнал на основе bias и условий
    # Если bias меняется на противоположное или на RANGE при открытой позиции - сигнал на разворот
    if has_position and (bias == Bias.RANGE or (bias != has_position and bias != Bias.RANGE)):
        # Если позиция LONG, а bias стал SHORT → сигнал SHORT (закроет LONG и откроет SHORT)
        if has_position == Bias.LONG and bias == Bias.SHORT:
            indicators_info = {
                "strategy": "TREND",
                "bias": bias.value,
                "previous_bias": has_position.value if has_position else None,
                "adx": round(adx, 2) if np.isfinite(adx) else None,
                "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "reason": "bias_flip",
                "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f})" if all(np.isfinite([adx, plus_di, minus_di, macd, macd_signal, macd_hist])) else (f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}" if all(np.isfinite([adx, plus_di, minus_di])) else "N/A")
            }
            return Signal(timestamp=row.name, action=Action.SHORT, reason="trend_bias_flip_to_short", price=price, indicators_info=indicators_info)
        # Если позиция SHORT, а bias стал LONG → сигнал LONG (закроет SHORT и откроет LONG)
        elif has_position == Bias.SHORT and bias == Bias.LONG:
            indicators_info = {
                "strategy": "TREND",
                "bias": bias.value,
                "previous_bias": has_position.value if has_position else None,
                "adx": round(adx, 2) if np.isfinite(adx) else None,
                "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "reason": "bias_flip",
                "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f})" if all(np.isfinite([adx, plus_di, minus_di, macd, macd_signal, macd_hist])) else (f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}" if all(np.isfinite([adx, plus_di, minus_di])) else "N/A")
            }
            return Signal(timestamp=row.name, action=Action.LONG, reason="trend_bias_flip_to_long", price=price, indicators_info=indicators_info)
        # Если bias стал RANGE → закрываем позицию (сигнал на закрытие)
        elif bias == Bias.RANGE:
            # Закрываем позицию при переходе в RANGE
            if has_position == Bias.LONG:
                indicators_info = {
                    "strategy": "TREND",
                    "bias": bias.value,
                    "previous_bias": has_position.value if has_position else None,
                    "adx": round(adx, 2) if np.isfinite(adx) else None,
                    "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                    "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                    "reason": "bias_to_range",
                    "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}" if all(np.isfinite([adx, plus_di, minus_di])) else "N/A"
                }
                return Signal(row.name, Action.SHORT, "trend_bias_to_range_close_long", price, indicators_info=indicators_info)
            elif has_position == Bias.SHORT:
                indicators_info = {
                    "strategy": "TREND",
                    "bias": bias.value,
                    "previous_bias": has_position.value if has_position else None,
                    "adx": round(adx, 2) if np.isfinite(adx) else None,
                    "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                    "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                    "reason": "bias_to_range",
                    "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}" if all(np.isfinite([adx, plus_di, minus_di])) else "N/A"
                }
                return Signal(row.name, Action.LONG, "trend_bias_to_range_close_short", price, indicators_info=indicators_info)
    
    # Определяем сигнал на основе bias и условий входа
    if bias == Bias.LONG:
        # Условия для сигнала LONG: breakout, fakeout, или pullback (с подтверждением MACD)
        if long_breakout or long_fakeout or (near_sma and volume_spike and (macd_bullish or not np.isfinite(macd))):
            entry_type = []
            if long_breakout:
                entry_type.append("breakout")
            if long_fakeout:
                entry_type.append("fakeout")
            if near_sma and volume_spike:
                entry_type.append("pullback")
            
            indicators_info = {
                "strategy": "TREND",
                "bias": bias.value,
                "entry_type": "+".join(entry_type),
                "adx": round(adx, 2) if np.isfinite(adx) else None,
                "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "macd_bullish": macd_bullish if np.isfinite(macd) and np.isfinite(macd_signal) else None,
                "volume": round(volume, 0) if np.isfinite(volume) else None,
                "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                "price_vs_recent_high": round((price / recent_high - 1) * 100, 2) if np.isfinite(recent_high) and recent_high > 0 else None,
                "sma": round(sma, 2) if np.isfinite(sma) else None,
                "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
                "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([adx, plus_di, minus_di, macd, macd_signal, macd_hist, volume, vol_sma])) and vol_sma > 0 else (f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([adx, plus_di, minus_di, volume, vol_sma])) and vol_sma > 0 else "N/A")
            }
            return Signal(row.name, Action.LONG, "trend_long_signal", price, indicators_info=indicators_info)
    elif bias == Bias.SHORT:
        # Условия для сигнала SHORT: breakout, fakeout, или pullback (с подтверждением MACD)
        if short_breakout or short_fakeout or (near_sma and volume_spike and (macd_bearish or not np.isfinite(macd))):
            entry_type = []
            if short_breakout:
                entry_type.append("breakout")
            if short_fakeout:
                entry_type.append("fakeout")
            if near_sma and volume_spike:
                entry_type.append("pullback")
            
            indicators_info = {
                "strategy": "TREND",
                "bias": bias.value,
                "entry_type": "+".join(entry_type),
                "adx": round(adx, 2) if np.isfinite(adx) else None,
                "plus_di": round(plus_di, 2) if np.isfinite(plus_di) else None,
                "minus_di": round(minus_di, 2) if np.isfinite(minus_di) else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "macd_bearish": macd_bearish if np.isfinite(macd) and np.isfinite(macd_signal) else None,
                "volume": round(volume, 0) if np.isfinite(volume) else None,
                "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                "price_vs_recent_low": round((price / recent_low - 1) * 100, 2) if np.isfinite(recent_low) and recent_low > 0 else None,
                "sma": round(sma, 2) if np.isfinite(sma) else None,
                "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
                "indicators": f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([adx, plus_di, minus_di, macd, macd_signal, macd_hist, volume, vol_sma])) and vol_sma > 0 else (f"ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([adx, plus_di, minus_di, volume, vol_sma])) and vol_sma > 0 else "N/A")
            }
            return Signal(row.name, Action.SHORT, "trend_short_signal", price, indicators_info=indicators_info)
    
    # Если нет четкого сигнала
    return Signal(row.name, Action.HOLD, "trend_hold", price)


def generate_range_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams, entry_price: Optional[float] = None) -> Signal:
    """
    Флэтовая стратегия Mean Reversion + Volume Filter.
    Используется при ADX <= threshold.
    
    Логика входа (LONG):
    1. Касание нижней границы BB (цена <= bb_lower)
    2. RSI <= 30 (перепроданность)
    3. Объем не аномальный (Volume < Volume_SMA * 1.3)
    
    Логика входа (SHORT):
    1. Касание верхней границы BB (цена >= bb_upper)
    2. RSI >= 70 (перекупленность)
    3. Объем не аномальный (Volume < Volume_SMA * 1.3)
    
    Логика выхода:
    - Основной TP: средняя линия BB (bb_middle)
    - Агрессивный TP: противоположная граница BB
    - SL: локальный минимум/максимум или 1.5% от входа
    """
    price = row["close"]
    high = row.get("high", price)
    low = row.get("low", price)
    rsi = row.get("rsi", np.nan)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    
    # Bollinger Bands
    bb_upper = row.get("bb_upper", np.nan)
    bb_middle = row.get("bb_middle", np.nan)
    bb_lower = row.get("bb_lower", np.nan)
    
    # MACD анализ для подтверждения разворота
    macd = row.get("macd", np.nan)
    macd_signal = row.get("macd_signal", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    # Для mean reversion: MACD должен показывать готовность к развороту
    # LONG: MACD близок к нулю или выше сигнала (готовность к росту)
    # SHORT: MACD близок к нулю или ниже сигнала (готовность к падению)
    macd_ready_long = not np.isfinite(macd) or (np.isfinite(macd) and np.isfinite(macd_signal) and (macd >= macd_signal or abs(macd) < abs(macd_signal) * 0.5))
    macd_ready_short = not np.isfinite(macd) or (np.isfinite(macd) and np.isfinite(macd_signal) and (macd <= macd_signal or abs(macd) < abs(macd_signal) * 0.5))
    
    # Проверяем валидность данных
    if not all(np.isfinite([bb_upper, bb_middle, bb_lower, rsi, volume, vol_sma])):
        return Signal(row.name, Action.HOLD, "range_no_data", price)
    
    # Фильтр объема: объем не должен быть аномально высоким
    volume_ok = volume < vol_sma * params.range_volume_mult
    
    # Определяем перекупленность/перепроданность
    rsi_oversold = rsi <= params.range_rsi_oversold  # перепроданность - сигнал на покупку
    rsi_overbought = rsi >= params.range_rsi_overbought  # перекупленность - сигнал на продажу
    
    # Проверяем касание границ BB (цена или тень свечи) с допуском
    bb_tolerance = params.range_bb_touch_tolerance_pct
    touch_lower = low <= bb_lower * (1 + bb_tolerance) or price <= bb_lower * (1 + bb_tolerance)  # касание нижней границы
    touch_upper = high >= bb_upper * (1 - bb_tolerance) or price >= bb_upper * (1 - bb_tolerance)  # касание верхней границы
    
    # Выход из позиций: если достигнут TP/SL, сигнализируем противоположное направление или HOLD
    if has_position == Bias.LONG:
        # Основной TP: средняя линия BB - закрываем и ждем
        if price >= bb_middle:
            return Signal(row.name, Action.HOLD, "range_tp_middle", price)
        # Агрессивный TP: верхняя граница BB - закрываем и ждем
        if params.range_tp_aggressive and price >= bb_upper:
            return Signal(row.name, Action.HOLD, "range_tp_aggressive", price)
        # Стоп-лосс: 1.5% от входа - если достигнут, сигнализируем SHORT (закроет LONG)
        if entry_price is not None:
            stop_loss_price = entry_price * (1 - params.range_stop_loss_pct)
            if low <= stop_loss_price:
                # ВАЖНО: Action.SHORT для закрытия LONG позиции корректно для netting аккаунтов
                # (где покупка и продажа взаимозачитываются). Для хедж-режима нужна отдельная логика закрытия.
                return Signal(row.name, Action.SHORT, "range_sl_fixed", price)
    
    if has_position == Bias.SHORT:
        # Основной TP: средняя линия BB - закрываем и ждем
        if price <= bb_middle:
            return Signal(row.name, Action.HOLD, "range_tp_middle", price)
        # Агрессивный TP: нижняя граница BB - закрываем и ждем
        if params.range_tp_aggressive and price <= bb_lower:
            return Signal(row.name, Action.HOLD, "range_tp_aggressive", price)
        # Стоп-лосс: 1.5% от входа - если достигнут, сигнализируем LONG (закроет SHORT)
        if entry_price is not None:
            stop_loss_price = entry_price * (1 + params.range_stop_loss_pct)
            if high >= stop_loss_price:
                # ВАЖНО: Action.LONG для закрытия SHORT позиции корректно для netting аккаунтов
                # (где покупка и продажа взаимозачитываются). Для хедж-режима нужна отдельная логика закрытия.
                return Signal(row.name, Action.LONG, "range_sl_fixed", price)
    
    # Входы: если нет позиции и есть условия входа
    if not has_position:
        # Проверяем, что объем подтверждает направление движения цены
        # Для mean reversion: объем должен быть выше среднего (подтверждает движение к границе), но не слишком высоким (чтобы не было начала тренда)
        volume_confirms_long = volume > vol_sma * 0.8  # Объем выше 80% от среднего (подтверждает движение вниз к нижней границе)
        volume_confirms_short = volume > vol_sma * 0.8  # Объем выше 80% от среднего (подтверждает движение вверх к верхней границе)
        
        # LONG: касание нижней границы BB + RSI перепродан + нормальный объем (не аномальный) + объем подтверждает движение + MACD готов к развороту
        if touch_lower and rsi_oversold and volume_ok and volume_confirms_long and macd_ready_long:
            indicators_info = {
                "strategy": "FLAT",
                "entry_type": "mean_reversion",
                "adx": round(row.get("adx", np.nan), 2) if np.isfinite(row.get("adx", np.nan)) else None,
                "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
                "bb_lower": round(bb_lower, 2) if np.isfinite(bb_lower) else None,
                "bb_middle": round(bb_middle, 2) if np.isfinite(bb_middle) else None,
                "bb_upper": round(bb_upper, 2) if np.isfinite(bb_upper) else None,
                "price_vs_bb_lower": round((price / bb_lower - 1) * 100, 2) if np.isfinite(bb_lower) and bb_lower > 0 else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "volume": round(volume, 0) if np.isfinite(volume) else None,
                "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                "indicators": f"RSI={rsi:.2f} (oversold), BB_lower={bb_lower:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([rsi, bb_lower, macd, macd_signal, macd_hist, volume, vol_sma])) and vol_sma > 0 else (f"RSI={rsi:.2f} (oversold), BB_lower={bb_lower:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([rsi, bb_lower, volume, vol_sma])) and vol_sma > 0 else "N/A")
            }
            return Signal(row.name, Action.LONG, "range_bb_lower_rsi_oversold", price, indicators_info=indicators_info)
        # SHORT: касание верхней границы BB + RSI перекуплен + нормальный объем (не аномальный) + объем подтверждает движение + MACD готов к развороту
        if touch_upper and rsi_overbought and volume_ok and volume_confirms_short and macd_ready_short:
            indicators_info = {
                "strategy": "FLAT",
                "entry_type": "mean_reversion",
                "adx": round(row.get("adx", np.nan), 2) if np.isfinite(row.get("adx", np.nan)) else None,
                "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
                "bb_lower": round(bb_lower, 2) if np.isfinite(bb_lower) else None,
                "bb_middle": round(bb_middle, 2) if np.isfinite(bb_middle) else None,
                "bb_upper": round(bb_upper, 2) if np.isfinite(bb_upper) else None,
                "price_vs_bb_upper": round((price / bb_upper - 1) * 100, 2) if np.isfinite(bb_upper) and bb_upper > 0 else None,
                "macd": round(macd, 4) if np.isfinite(macd) else None,
                "macd_signal": round(macd_signal, 4) if np.isfinite(macd_signal) else None,
                "macd_hist": round(macd_hist, 4) if np.isfinite(macd_hist) else None,
                "volume": round(volume, 0) if np.isfinite(volume) else None,
                "vol_sma": round(vol_sma, 0) if np.isfinite(vol_sma) else None,
                "vol_ratio": round(volume / vol_sma, 2) if np.isfinite(volume) and np.isfinite(vol_sma) and vol_sma > 0 else None,
                "indicators": f"RSI={rsi:.2f} (overbought), BB_upper={bb_upper:.2f}, MACD={macd:.4f}/{macd_signal:.4f} (hist={macd_hist:.4f}), Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([rsi, bb_upper, macd, macd_signal, macd_hist, volume, vol_sma])) and vol_sma > 0 else (f"RSI={rsi:.2f} (overbought), BB_upper={bb_upper:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if all(np.isfinite([rsi, bb_upper, volume, vol_sma])) and vol_sma > 0 else "N/A")
            }
            return Signal(row.name, Action.SHORT, "range_bb_upper_rsi_overbought", price, indicators_info=indicators_info)
    
    return Signal(row.name, Action.HOLD, "range_wait", price)


def generate_momentum_breakout_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams) -> Signal:
    """
    Стратегия "Импульсный пробой" для тренда.
    Цель: Зайти в сильное движение в самом начале и удерживать позицию, пока тренд не сменится.
    
    Технические параметры:
    - Таймфрейм: 1h или 4h (EMA вычисляются на высоком таймфрейме)
    - Индикаторы: EMA 20 (быстрая), EMA 50 (медленная), ADX (14), Volume SMA (20)
    
    Логика входа:
    - EMA Cross: EMA 20 пересекает EMA 50 снизу вверх (для Long) или сверху вниз (для Short)
    - ADX Confirmation: ADX > 25
    - Volume Spike: Объем текущей свечи в 1.5-2 раза выше Volume SMA(20)
    
    Логика выхода:
    - Trailing Stop: Подтягиваем стоп-лосс по линии EMA 50
    - Обратный сигнал: Если EMA 20 пересекает EMA 50 в обратную сторону - закрываем позицию
    """
    price = row["close"]
    adx = row.get("adx", np.nan)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    
    # Получаем EMA с высокого таймфрейма
    ema_timeframe = params.momentum_ema_timeframe
    ema_fast_col = f"ema_fast_{ema_timeframe}"
    ema_slow_col = f"ema_slow_{ema_timeframe}"
    
    ema_fast = row.get(ema_fast_col, np.nan)
    ema_slow = row.get(ema_slow_col, np.nan)
    
    # Получаем предыдущие значения для определения пересечения
    # Для этого нужно иметь доступ к предыдущей строке, но в текущей архитектуре это сложно
    # Используем текущие значения и проверяем их соотношение
    
    # Проверяем валидность данных
    if not all(np.isfinite([adx, ema_fast, ema_slow, volume, vol_sma])):
        return Signal(row.name, Action.HOLD, "momentum_no_data", price)
    
    # ADX подтверждение тренда
    adx_confirmed = adx > params.momentum_adx_threshold
    
    # Volume Spike: объем должен быть в диапазоне 1.5-2x от Volume SMA
    volume_spike = (volume >= vol_sma * params.momentum_volume_spike_min and 
                    volume <= vol_sma * params.momentum_volume_spike_max)
    
    # Определяем направление тренда по EMA
    ema_bullish = ema_fast > ema_slow  # Быстрая EMA выше медленной = восходящий тренд
    ema_bearish = ema_fast < ema_slow  # Быстрая EMA ниже медленной = нисходящий тренд
    
    # Проверяем, что EMA достаточно разошлись (избегаем ложных сигналов)
    # ВАЖНО: Проверяем деление на ноль и валидность данных
    ema_spread = abs(ema_fast - ema_slow) / ema_slow if (ema_slow > 0 and np.isfinite(ema_slow)) else 0
    ema_spread_ok = ema_spread > 0.001  # Минимальный разброс 0.1%
    
    # Выход из позиций
    if has_position == Bias.LONG:
        # Обратный сигнал: EMA 20 пересекает EMA 50 сверху вниз
        if ema_bearish and ema_spread_ok:
            indicators_info = {
                "strategy": "MOMENTUM",
                "exit_type": "ema_cross_reversal",
                "adx": round(adx, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "indicators": f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f} (bearish cross)"
            }
            return Signal(row.name, Action.SHORT, "momentum_long_exit_ema_reversal", price, indicators_info=indicators_info)
        # Trailing Stop: цена ниже EMA 50
        if params.momentum_trailing_stop_ema and price < ema_slow:
            indicators_info = {
                "strategy": "MOMENTUM",
                "exit_type": "trailing_stop_ema",
                "price_vs_ema50": round((price / ema_slow - 1) * 100, 2) if (ema_slow > 0 and np.isfinite(ema_slow)) else None,
                "indicators": f"Price={price:.2f} < EMA50={ema_slow:.2f}"
            }
            return Signal(row.name, Action.HOLD, "momentum_long_exit_trailing_stop", price, indicators_info=indicators_info)
    
    if has_position == Bias.SHORT:
        # Обратный сигнал: EMA 20 пересекает EMA 50 снизу вверх
        if ema_bullish and ema_spread_ok:
            indicators_info = {
                "strategy": "MOMENTUM",
                "exit_type": "ema_cross_reversal",
                "adx": round(adx, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "indicators": f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f} (bullish cross)"
            }
            return Signal(row.name, Action.LONG, "momentum_short_exit_ema_reversal", price, indicators_info=indicators_info)
        # Trailing Stop: цена выше EMA 50
        if params.momentum_trailing_stop_ema and price > ema_slow:
            indicators_info = {
                "strategy": "MOMENTUM",
                "exit_type": "trailing_stop_ema",
                "price_vs_ema50": round((price / ema_slow - 1) * 100, 2) if (ema_slow > 0 and np.isfinite(ema_slow)) else None,
                "indicators": f"Price={price:.2f} > EMA50={ema_slow:.2f}"
            }
            return Signal(row.name, Action.HOLD, "momentum_short_exit_trailing_stop", price, indicators_info=indicators_info)
    
    # Входы: если нет позиции и есть условия входа
    # Примечание: Пересечение EMA проверяется в build_signals, здесь проверяем только текущее состояние
    if not has_position:
        # LONG: EMA 20 выше EMA 50 + ADX подтверждение + Volume Spike
        # Пересечение снизу вверх уже проверено в build_signals, здесь проверяем только подтверждение условий
        if ema_bullish and adx_confirmed and volume_spike and ema_spread_ok:
            indicators_info = {
                "strategy": "MOMENTUM",
                "entry_type": "breakout",
                "adx": round(adx, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "ema_spread_pct": round(ema_spread * 100, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": _format_vol_ratio(volume, vol_sma),
                "indicators": f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if (vol_sma > 0 and np.isfinite(vol_sma)) else f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f}, Vol={volume:.0f}"
            }
            return Signal(row.name, Action.LONG, "momentum_long_breakout", price, indicators_info=indicators_info)
        
        # SHORT: EMA 20 ниже EMA 50 + ADX подтверждение + Volume Spike
        # Пересечение сверху вниз уже проверено в build_signals, здесь проверяем только подтверждение условий
        # ВАЖНО: Если нет флага пересечения, не генерируем сигнал (избегаем спама)
        # Пересечение уже проверено в build_signals, но для дополнительной защиты проверяем флаг
        ema_cross_down = row.get("ema_cross_down", False)  # Флаг пересечения сверху вниз (установлен в enrich_for_strategy)
        # Если флаг не установлен, но мы в build_signals (где пересечение уже проверено), все равно генерируем сигнал
        if ema_bearish and adx_confirmed and volume_spike and ema_spread_ok:
            indicators_info = {
                "strategy": "MOMENTUM",
                "entry_type": "breakout",
                "adx": round(adx, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "ema_spread_pct": round(ema_spread * 100, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": _format_vol_ratio(volume, vol_sma),
                "indicators": f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if (vol_sma > 0 and np.isfinite(vol_sma)) else f"ADX={adx:.2f}, EMA20={ema_fast:.2f}, EMA50={ema_slow:.2f}, Vol={volume:.0f}"
            }
            return Signal(row.name, Action.SHORT, "momentum_short_breakout", price, indicators_info=indicators_info)
    
    return Signal(row.name, Action.HOLD, "momentum_wait", price)


    """
    Стратегия "Возврат к среднему" (Mean Reversion) для флэта.
    Цель: Заработать на «усталости» продавцов или покупателей, когда цена слишком сильно отклонилась от нормы.
    
    Технические параметры:
    - Таймфрейм: 5m или 15m
    - Индикаторы: Bollinger Bands (20, 2), RSI (14), Volume SMA (20)
    
    Логика входа:
    - Цена касается или выходит за нижнюю/верхнюю полосу Боллинджера
    - RSI в зоне перепроданности (<= 30) или перекупленности (>= 70)
    - Volume Exhaustion: Объем низкий (<= Volume_SMA) - продавцы/покупатели иссякли
    
    Логика выхода:
    - Take Profit: При достижении средней линии BB
    - Stop Loss: Фиксированный 1% от входа или за локальный минимум/максимум
    """
    price = row["close"]
    high = row.get("high", price)
    low = row.get("low", price)
    rsi = row.get("rsi", np.nan)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    
    # Bollinger Bands
    bb_upper = row.get("bb_upper", np.nan)
    bb_middle = row.get("bb_middle", np.nan)
    bb_lower = row.get("bb_lower", np.nan)
    
    # Проверяем валидность данных
    if not all(np.isfinite([bb_upper, bb_middle, bb_lower, rsi, volume, vol_sma])):
        return Signal(row.name, Action.HOLD, "mean_rev_no_data", price)
    
    # Volume Exhaustion: объем должен быть низким (<= Volume_SMA)
    volume_exhausted = volume <= vol_sma * params.mean_rev_volume_exhaustion_mult
    
    # Определяем перекупленность/перепроданность
    rsi_oversold = rsi <= params.mean_rev_rsi_oversold
    rsi_overbought = rsi >= (100 - params.mean_rev_rsi_oversold)  # Симметрично для SHORT
    
    # Проверяем касание границ BB
    touch_lower = low <= bb_lower or price <= bb_lower
    touch_upper = high >= bb_upper or price >= bb_upper
    
    # Выход из позиций
    if has_position == Bias.LONG:
        # Take Profit: средняя линия BB
        if params.mean_rev_tp_bb_middle and price >= bb_middle:
            indicators_info = {
                "strategy": "MEAN_REV",
                "exit_type": "tp_bb_middle",
                "price_vs_bb_middle": round((price / bb_middle - 1) * 100, 2),
                "indicators": f"Price={price:.2f} >= BB_middle={bb_middle:.2f}"
            }
            return Signal(row.name, Action.HOLD, "mean_rev_long_exit_tp", price, indicators_info=indicators_info)
        # Stop Loss: фиксированный процент
        if entry_price is not None:
            stop_loss_price = entry_price * (1 - params.mean_rev_stop_loss_pct)
            if low <= stop_loss_price:
                indicators_info = {
                    "strategy": "MEAN_REV",
                    "exit_type": "sl_fixed",
                    "entry_price": round(entry_price, 2),
                    "stop_loss_price": round(stop_loss_price, 2),
                    "indicators": f"Price={price:.2f} <= SL={stop_loss_price:.2f}"
                }
                return Signal(row.name, Action.SHORT, "mean_rev_long_exit_sl", price, indicators_info=indicators_info)
    
    if has_position == Bias.SHORT:
        # Take Profit: средняя линия BB
        if params.mean_rev_tp_bb_middle and price <= bb_middle:
            indicators_info = {
                "strategy": "MEAN_REV",
                "exit_type": "tp_bb_middle",
                "price_vs_bb_middle": round((price / bb_middle - 1) * 100, 2),
                "indicators": f"Price={price:.2f} <= BB_middle={bb_middle:.2f}"
            }
            return Signal(row.name, Action.HOLD, "mean_rev_short_exit_tp", price, indicators_info=indicators_info)
        # Stop Loss: фиксированный процент
        if entry_price is not None:
            stop_loss_price = entry_price * (1 + params.mean_rev_stop_loss_pct)
            if high >= stop_loss_price:
                indicators_info = {
                    "strategy": "MEAN_REV",
                    "exit_type": "sl_fixed",
                    "entry_price": round(entry_price, 2),
                    "stop_loss_price": round(stop_loss_price, 2),
                    "indicators": f"Price={price:.2f} >= SL={stop_loss_price:.2f}"
                }
                return Signal(row.name, Action.LONG, "mean_rev_short_exit_sl", price, indicators_info=indicators_info)
    
    # Входы: если нет позиции и есть условия входа
    if not has_position:
        # LONG: касание нижней границы BB + RSI перепродан + Volume Exhaustion
        if touch_lower and rsi_oversold and volume_exhausted:
            indicators_info = {
                "strategy": "MEAN_REV",
                "entry_type": "mean_reversion",
                "rsi": round(rsi, 2),
                "bb_lower": round(bb_lower, 2),
                "bb_middle": round(bb_middle, 2),
                "bb_upper": round(bb_upper, 2),
                "price_vs_bb_lower": round((price / bb_lower - 1) * 100, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": round(volume / vol_sma, 2) if vol_sma > 0 else None,
                "indicators": f"RSI={rsi:.2f} (oversold), BB_lower={bb_lower:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x, exhausted)"
            }
            return Signal(row.name, Action.LONG, "mean_rev_long_bb_lower", price, indicators_info=indicators_info)
        
        # SHORT: касание верхней границы BB + RSI перекуплен + Volume Exhaustion
        if touch_upper and rsi_overbought and volume_exhausted:
            indicators_info = {
                "strategy": "MEAN_REV",
                "entry_type": "mean_reversion",
                "rsi": round(rsi, 2),
                "bb_lower": round(bb_lower, 2),
                "bb_middle": round(bb_middle, 2),
                "bb_upper": round(bb_upper, 2),
                "price_vs_bb_upper": round((price / bb_upper - 1) * 100, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": round(volume / vol_sma, 2) if vol_sma > 0 else None,
                "indicators": f"RSI={rsi:.2f} (overbought), BB_upper={bb_upper:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x, exhausted)"
            }
            return Signal(row.name, Action.SHORT, "mean_rev_short_bb_upper", price, indicators_info=indicators_info)
    
    return Signal(row.name, Action.HOLD, "mean_rev_wait", price)


def generate_vwap_breakout_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams, df: pd.DataFrame, entry_price: Optional[float] = None) -> Signal:
    """
    Стратегия "Институциональный тренд: VWAP Breakout".
    Цель: Зайти в движение, когда крупные игроки (институционалы) начинают активно толкать цену.
    
    Технические параметры:
    - Таймфрейм: 5m или 15m (интрадей)
    - Индикаторы: VWAP (с ежедневным сбросом), ATR (14), Volume SMA (20)
    
    Логика входа (LONG):
    - Цена пробивает VWAP снизу вверх и закрывается выше
    - Объем в момент пробоя > 1.8x от среднего (Volume SMA)
    - Подтверждение: Следующая свеча также закрывается выше VWAP
    
    Логика входа (SHORT):
    - Цена пробивает VWAP сверху вниз и закрывается ниже
    - Объем в момент пробоя > 1.8x от среднего (Volume SMA)
    - Подтверждение: Следующая свеча также закрывается ниже VWAP
    
    Логика выхода:
    - Take Profit: 2 * ATR от входа
    - Stop Loss: за линией VWAP
    """
    price = row["close"]
    high = row.get("high", price)
    low = row.get("low", price)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    vwap = row.get("vwap", np.nan)
    atr = row.get("atr", np.nan)
    
    # Проверяем валидность данных
    if not all(np.isfinite([vwap, volume, vol_sma, atr])):
        return Signal(row.name, Action.HOLD, "vwap_no_data", price)
    
    # Volume Spike: объем должен быть > 1.8x от Volume SMA
    volume_spike = volume > vol_sma * params.vwap_volume_spike_mult
    
    # Получаем предыдущие значения для определения пробоя
    # Нужно проверить предыдущую свечу и текущую
    current_idx = df.index.get_loc(row.name) if row.name in df.index else None
    if current_idx is None or current_idx < 1:
        return Signal(row.name, Action.HOLD, "vwap_no_prev_data", price)
    
    prev_row = df.iloc[current_idx - 1]
    prev_close = prev_row.get("close", np.nan)
    prev_vwap = prev_row.get("vwap", np.nan)
    
    # Проверяем пробой VWAP
    # LONG: предыдущая цена была ниже VWAP, текущая закрылась выше VWAP
    vwap_breakout_long = (np.isfinite([prev_close, prev_vwap]).all() and 
                          prev_close < prev_vwap and 
                          price > vwap)
    
    # SHORT: предыдущая цена была выше VWAP, текущая закрылась ниже VWAP
    vwap_breakout_short = (np.isfinite([prev_close, prev_vwap]).all() and 
                           prev_close > prev_vwap and 
                           price < vwap)
    
    # Проверяем подтверждение следующей свечой (если доступна)
    confirmation_ok = True
    if params.vwap_confirmation_candles > 0 and current_idx < len(df) - 1:
        next_row = df.iloc[current_idx + 1]
        next_close = next_row.get("close", np.nan)
        next_vwap = next_row.get("vwap", np.nan)
        
        if vwap_breakout_long:
            # Для LONG: следующая свеча должна закрыться выше VWAP
            confirmation_ok = np.isfinite([next_close, next_vwap]).all() and next_close > next_vwap
        elif vwap_breakout_short:
            # Для SHORT: следующая свеча должна закрыться ниже VWAP
            confirmation_ok = np.isfinite([next_close, next_vwap]).all() and next_close < next_vwap
    
    # Выход из позиций
    if has_position == Bias.LONG:
        # Take Profit: 2 * ATR от входа
        if entry_price is not None and np.isfinite(atr):
            tp_price = entry_price + (atr * params.vwap_tp_atr_mult)
            if high >= tp_price:
                indicators_info = {
                    "strategy": "VWAP",
                    "exit_type": "tp_atr",
                    "entry_price": round(entry_price, 2),
                    "tp_price": round(tp_price, 2),
                    "atr": round(atr, 2),
                    "indicators": f"Price={price:.2f} >= TP={tp_price:.2f} (2*ATR)"
                }
                return Signal(row.name, Action.HOLD, "vwap_long_exit_tp", price, indicators_info=indicators_info)
        
        # Stop Loss: за линией VWAP
        if params.vwap_sl_at_vwap and low < vwap:
            indicators_info = {
                "strategy": "VWAP",
                "exit_type": "sl_vwap",
                "vwap": round(vwap, 2),
                "indicators": f"Price={price:.2f} < VWAP={vwap:.2f}"
            }
            return Signal(row.name, Action.SHORT, "vwap_long_exit_sl", price, indicators_info=indicators_info)
    
    if has_position == Bias.SHORT:
        # Take Profit: 2 * ATR от входа
        if entry_price is not None and np.isfinite(atr):
            tp_price = entry_price - (atr * params.vwap_tp_atr_mult)
            if low <= tp_price:
                indicators_info = {
                    "strategy": "VWAP",
                    "exit_type": "tp_atr",
                    "entry_price": round(entry_price, 2),
                    "tp_price": round(tp_price, 2),
                    "atr": round(atr, 2),
                    "indicators": f"Price={price:.2f} <= TP={tp_price:.2f} (2*ATR)"
                }
                return Signal(row.name, Action.HOLD, "vwap_short_exit_tp", price, indicators_info=indicators_info)
        
        # Stop Loss: за линией VWAP
        if params.vwap_sl_at_vwap and high > vwap:
            indicators_info = {
                "strategy": "VWAP",
                "exit_type": "sl_vwap",
                "vwap": round(vwap, 2),
                "indicators": f"Price={price:.2f} > VWAP={vwap:.2f}"
            }
            return Signal(row.name, Action.LONG, "vwap_short_exit_sl", price, indicators_info=indicators_info)
    
    # Входы: если нет позиции и есть условия входа
    if not has_position:
        # LONG: пробой VWAP снизу вверх + Volume Spike + Подтверждение
        if vwap_breakout_long and volume_spike and confirmation_ok:
            indicators_info = {
                "strategy": "VWAP",
                "entry_type": "breakout",
                "vwap": round(vwap, 2),
                "price_vs_vwap": round((price / vwap - 1) * 100, 2) if (vwap > 0 and np.isfinite(vwap)) else None,
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": round(volume / vol_sma, 2) if vol_sma > 0 else None,
                "atr": round(atr, 2),
                "tp_atr_mult": params.vwap_tp_atr_mult,
                "indicators": f"VWAP={vwap:.2f}, Price={price:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x), ATR={atr:.2f}"
            }
            return Signal(row.name, Action.LONG, "vwap_long_breakout", price, indicators_info=indicators_info)
        
        # SHORT: пробой VWAP сверху вниз + Volume Spike + Подтверждение
        if vwap_breakout_short and volume_spike and confirmation_ok:
            indicators_info = {
                "strategy": "VWAP",
                "entry_type": "breakout",
                "vwap": round(vwap, 2),
                "price_vs_vwap": round((price / vwap - 1) * 100, 2) if (vwap > 0 and np.isfinite(vwap)) else None,
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": round(volume / vol_sma, 2) if vol_sma > 0 else None,
                "atr": round(atr, 2),
                "tp_atr_mult": params.vwap_tp_atr_mult,
                "indicators": f"VWAP={vwap:.2f}, Price={price:.2f}, Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x), ATR={atr:.2f}"
            }
            return Signal(row.name, Action.SHORT, "vwap_short_breakout", price, indicators_info=indicators_info)
    
    return Signal(row.name, Action.HOLD, "vwap_wait", price)


def generate_liquidity_sweep_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams, df: pd.DataFrame, entry_price: Optional[float] = None) -> Signal:
    """
    Стратегия "Liquidity Sweep" (Снятие ликвидности).
    Цель: Охотиться за стоп-лоссами толпы, которые крупные игроки используют как ликвидность.
    
    Технические параметры:
    - Таймфрейм: 15m или 1h
    - Индикаторы: Donchian Channels (20), Volume Spike
    
    Логика входа (LONG):
    - Цена кратковременно пробивает нижнюю границу канала (локальный минимум)
    - Огромный всплеск объема в момент пробоя (вынос стопов) > 2.5x от среднего
    - Разворот: Свеча закрывается внутри канала, оставив длинную тень снизу
    - Тень должна составлять минимум 60% от общей длины свечи
    
    Логика входа (SHORT):
    - Цена кратковременно пробивает верхнюю границу канала (локальный максимум)
    - Огромный всплеск объема в момент пробоя (вынос стопов) > 2.5x от среднего
    - Разворот: Свеча закрывается внутри канала, оставив длинную тень сверху
    
    Логика выхода:
    - LONG: Ближайший локальный максимум (верхняя граница канала)
    - SHORT: Ближайший локальный минимум (нижняя граница канала)
    """
    price = row["close"]
    high = row.get("high", price)
    low = row.get("low", price)
    open_price = row.get("open", price)
    volume = row.get("volume", np.nan)
    vol_sma = row.get("vol_sma", np.nan)
    
    # Donchian Channels
    donchian_upper = row.get("donchian_upper", np.nan)
    donchian_lower = row.get("donchian_lower", np.nan)
    donchian_middle = row.get("donchian_middle", np.nan)
    
    # Проверяем валидность данных
    if not all(np.isfinite([donchian_upper, donchian_lower, volume, vol_sma])):
        return Signal(row.name, Action.HOLD, "liquidity_no_data", price)
    
    # Volume Spike: огромный всплеск объема (вынос стопов)
    volume_spike = volume > vol_sma * params.liquidity_volume_spike_mult
    
    # Вычисляем длину свечи и тени
    candle_body = abs(price - open_price)
    candle_range = high - low
    lower_shadow = min(open_price, price) - low  # Нижняя тень
    upper_shadow = high - max(open_price, price)  # Верхняя тень
    
    # Проверяем пробой нижней границы (для LONG)
    # Цена должна была пробить нижнюю границу (low < donchian_lower)
    # но закрыться внутри канала (close > donchian_lower)
    lower_breakout = low < donchian_lower
    closes_inside = price > donchian_lower
    
    # Проверяем пробой верхней границы (для SHORT)
    # Цена должна была пробить верхнюю границу (high > donchian_upper)
    # но закрыться внутри канала (close < donchian_upper)
    upper_breakout = high > donchian_upper
    closes_inside_upper = price < donchian_upper
    
    # Проверяем длинную тень (для LONG - нижняя тень, для SHORT - верхняя тень)
    # Тень должна составлять минимум liquidity_shadow_ratio от общей длины свечи
    long_lower_shadow = (candle_range > 0 and lower_shadow / candle_range >= params.liquidity_shadow_ratio)
    long_upper_shadow = (candle_range > 0 and upper_shadow / candle_range >= params.liquidity_shadow_ratio)
    
    # Выход из позиций
    if has_position == Bias.LONG:
        # Выход: достижение верхней границы канала (локальный максимум)
        if high >= donchian_upper:
            indicators_info = {
                "strategy": "LIQUIDITY",
                "exit_type": "tp_donchian_upper",
                "donchian_upper": round(donchian_upper, 2),
                "indicators": f"Price={price:.2f} >= Donchian_upper={donchian_upper:.2f}"
            }
            return Signal(row.name, Action.HOLD, "liquidity_long_exit_tp", price, indicators_info=indicators_info)
    
    if has_position == Bias.SHORT:
        # Выход: достижение нижней границы канала (локальный минимум)
        if low <= donchian_lower:
            indicators_info = {
                "strategy": "LIQUIDITY",
                "exit_type": "tp_donchian_lower",
                "donchian_lower": round(donchian_lower, 2),
                "indicators": f"Price={price:.2f} <= Donchian_lower={donchian_lower:.2f}"
            }
            return Signal(row.name, Action.HOLD, "liquidity_short_exit_tp", price, indicators_info=indicators_info)
    
    # Входы: если нет позиции и есть условия входа
    if not has_position:
        # LONG: пробой нижней границы + огромный объем + разворот с длинной тенью
        if (lower_breakout and 
            closes_inside and 
            volume_spike and 
            long_lower_shadow):
            indicators_info = {
                "strategy": "LIQUIDITY",
                "entry_type": "sweep_long",
                "donchian_lower": round(donchian_lower, 2),
                "donchian_upper": round(donchian_upper, 2),
                "donchian_middle": round(donchian_middle, 2),
                "low_vs_donchian_lower": round((low / donchian_lower - 1) * 100, 2),
                "close_vs_donchian_lower": round((price / donchian_lower - 1) * 100, 2),
                "lower_shadow_pct": round((lower_shadow / candle_range * 100) if candle_range > 0 else 0, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": _format_vol_ratio(volume, vol_sma),
                "indicators": f"Donchian_lower={donchian_lower:.2f}, Low={low:.2f} (breakout), Close={price:.2f} (inside), Lower_shadow={lower_shadow:.2f} ({(lower_shadow/candle_range*100):.1f}%)" + (f", Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if (vol_sma > 0 and np.isfinite(vol_sma) and candle_range > 0) else f", Vol={volume:.0f}")
            }
            return Signal(row.name, Action.LONG, "liquidity_long_sweep", price, indicators_info=indicators_info)
        
        # SHORT: пробой верхней границы + огромный объем + разворот с длинной тенью
        if (upper_breakout and 
            closes_inside_upper and 
            volume_spike and 
            long_upper_shadow):
            indicators_info = {
                "strategy": "LIQUIDITY",
                "entry_type": "sweep_short",
                "donchian_lower": round(donchian_lower, 2),
                "donchian_upper": round(donchian_upper, 2),
                "donchian_middle": round(donchian_middle, 2),
                "high_vs_donchian_upper": round((high / donchian_upper - 1) * 100, 2),
                "close_vs_donchian_upper": round((price / donchian_upper - 1) * 100, 2),
                "upper_shadow_pct": round((upper_shadow / candle_range * 100) if candle_range > 0 else 0, 2),
                "volume": round(volume, 0),
                "vol_sma": round(vol_sma, 0),
                "vol_ratio": round(volume / vol_sma, 2) if vol_sma > 0 else None,
                "indicators": f"Donchian_upper={donchian_upper:.2f}, High={high:.2f} (breakout), Close={price:.2f} (inside), Upper_shadow={upper_shadow:.2f} ({upper_shadow/candle_range*100:.1f}%)" + (f", Vol={volume:.0f}/{vol_sma:.0f} ({volume/vol_sma:.2f}x)" if (vol_sma > 0 and np.isfinite(vol_sma)) else f", Vol={volume:.0f}")
            }
            return Signal(row.name, Action.SHORT, "liquidity_short_sweep", price, indicators_info=indicators_info)
    
    return Signal(row.name, Action.HOLD, "liquidity_wait", price)


def generate_signal(
    row: pd.Series, 
    has_position: Optional[Bias], 
    params: StrategyParams, 
    entry_price: Optional[float] = None,
    use_momentum: bool = False,
    use_liquidity: bool = False,
    df: Optional[pd.DataFrame] = None,
) -> Signal:
    """
    Главная функция генерации сигналов.
    Определяет фазу рынка и выбирает соответствующую стратегию.
    
    Args:
        row: Строка DataFrame с данными свечи
        has_position: Текущая позиция (LONG, SHORT, None)
        params: Параметры стратегии
        entry_price: Цена входа для расчета стоп-лосса
        use_momentum: Использовать стратегию импульсного пробоя (вместо старой трендовой)
        use_liquidity: Использовать стратегию Liquidity Sweep (снятие ликвидности)
        df: DataFrame со всеми данными (необходим для Liquidity стратегии)
    """
    # Liquidity Sweep стратегия имеет приоритет (работает независимо от фазы рынка)
    if use_liquidity and df is not None:
        return generate_liquidity_sweep_signal(row, has_position, params, df, entry_price)
    
    market_phase = detect_market_phase(row, params)
    
    if market_phase == MarketPhase.TREND:
        if use_momentum:
            return generate_momentum_breakout_signal(row, has_position, params)
        else:
            return generate_trend_signal(row, has_position, params)
    else:  # MarketPhase.FLAT
        return generate_range_signal(row, has_position, params, entry_price)


def enrich_for_strategy(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_volume"] = df["volume"].shift(1)

    # Consolidation detection (для трендовой стратегии)
    window = params.consolidation_bars
    rng_high = df["high"].rolling(window=window).max()
    rng_low = df["low"].rolling(window=window).min()
    rng_width = rng_high - rng_low
    avg_price = df["close"].rolling(window=window).mean()
    # ВАЖНО: Проверяем деление на ноль
    df["is_consolidating"] = (rng_width / avg_price) <= params.consolidation_range_pct
    # Заменяем NaN/Inf на False (если avg_price = 0, то не консолидация)
    df["is_consolidating"] = df["is_consolidating"].fillna(False).replace([np.inf, -np.inf], False)
    df["consolidation_high"] = rng_high
    df["consolidation_low"] = rng_low

    # Bias from 4H signals (используется только в трендовой фазе)
    df["bias"] = df.apply(lambda row: infer_bias(row, params), axis=1)
    
    # Market phase detection
    df["market_phase"] = df.apply(lambda row: detect_market_phase(row, params), axis=1)
    
    # Добавляем флаги пересечения EMA для momentum стратегии (предотвращение спама сигналов)
    # Это позволяет избежать генерации сигналов на каждой свече внутри одного тренда
    ema_timeframe = params.momentum_ema_timeframe
    ema_fast_col = f"ema_fast_{ema_timeframe}"
    ema_slow_col = f"ema_slow_{ema_timeframe}"
    
    if ema_fast_col in df.columns and ema_slow_col in df.columns:
        # Пересечение снизу вверх: prev_ema_fast <= prev_ema_slow AND ema_fast > ema_slow
        prev_ema_fast = df[ema_fast_col].shift(1)
        prev_ema_slow = df[ema_slow_col].shift(1)
        df["ema_cross_up"] = (prev_ema_fast <= prev_ema_slow) & (df[ema_fast_col] > df[ema_slow_col])
        # Пересечение сверху вниз: prev_ema_fast >= prev_ema_slow AND ema_fast < ema_slow
        df["ema_cross_down"] = (prev_ema_fast >= prev_ema_slow) & (df[ema_fast_col] < df[ema_slow_col])
    else:
        df["ema_cross_up"] = False
        df["ema_cross_down"] = False
    
    # Удаляем только строки, где все значения NaN (не удаляем строки с частичными NaN)
    # Это важно, так как shift() и rolling() создают NaN в начале, но основные данные (OHLCV) должны быть
    key_columns = ["open", "high", "low", "close", "volume"]
    if all(col in df.columns for col in key_columns):
        # Удаляем только строки, где все ключевые колонки NaN
        df = df[df[key_columns].notna().any(axis=1)]
    else:
        # Fallback: удаляем только строки, где все значения NaN
        df = df.dropna(how='all')
    
    return df


def build_signals(
    df: pd.DataFrame, 
    params: StrategyParams,
    use_momentum: bool = False,
    use_liquidity: bool = False,
) -> list[Signal]:
    """
    Строит сигналы на основе данных.
    Возвращает только LONG, SHORT или HOLD сигналы.
    
    Args:
        df: DataFrame с данными свечей и индикаторами
        params: Параметры стратегии
        use_momentum: Использовать стратегию импульсного пробоя (вместо старой трендовой)
        use_liquidity: Использовать стратегию Liquidity Sweep
    """
    signals: list[Signal] = []
    position_bias: Optional[Bias] = None
    entry_price: Optional[float] = None  # отслеживаем цену входа для стоп-лосса в флэтовой стратегии
    
    # Для определения пересечения EMA нужны предыдущие значения
    prev_ema_fast = None
    prev_ema_slow = None
    
    for idx, (_, row) in enumerate(df.iterrows()):
        sig = generate_signal(row, position_bias, params, entry_price, use_momentum=use_momentum, use_liquidity=use_liquidity, df=df)
        
        # Для momentum стратегии проверяем пересечение EMA
        if use_momentum and idx > 0:
            ema_timeframe = params.momentum_ema_timeframe
            ema_fast_col = f"ema_fast_{ema_timeframe}"
            ema_slow_col = f"ema_slow_{ema_timeframe}"
            
            ema_fast = row.get(ema_fast_col, np.nan)
            ema_slow = row.get(ema_slow_col, np.nan)
            
            # Проверяем пересечение EMA для генерации сигнала
            # Проверяем, что все значения валидны (не None и не NaN)
            if (prev_ema_fast is not None and prev_ema_slow is not None and 
                np.isfinite([ema_fast, ema_slow, prev_ema_fast, prev_ema_slow]).all()):
                # Пересечение снизу вверх (бычий сигнал)
                if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
                    # Генерируем LONG сигнал, если условия выполнены
                    adx = row.get("adx", np.nan)
                    volume = row.get("volume", np.nan)
                    vol_sma = row.get("vol_sma", np.nan)
                    
                    if (np.isfinite([adx, volume, vol_sma]).all() and
                        adx > params.momentum_adx_threshold and
                        volume >= vol_sma * params.momentum_volume_spike_min and
                        volume <= vol_sma * params.momentum_volume_spike_max):
                        # Создаем сигнал через generate_momentum_breakout_signal
                        sig = generate_momentum_breakout_signal(row, position_bias, params)
                
                # Пересечение сверху вниз (медвежий сигнал)
                elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
                    # Генерируем SHORT сигнал, если условия выполнены
                    adx = row.get("adx", np.nan)
                    volume = row.get("volume", np.nan)
                    vol_sma = row.get("vol_sma", np.nan)
                    
                    if (np.isfinite([adx, volume, vol_sma]).all() and
                        adx > params.momentum_adx_threshold and
                        volume >= vol_sma * params.momentum_volume_spike_min and
                        volume <= vol_sma * params.momentum_volume_spike_max):
                        # Создаем сигнал через generate_momentum_breakout_signal
                        sig = generate_momentum_breakout_signal(row, position_bias, params)
            
            # Сохраняем текущие значения для следующей итерации
            prev_ema_fast = ema_fast if np.isfinite(ema_fast) else prev_ema_fast
            prev_ema_slow = ema_slow if np.isfinite(ema_slow) else prev_ema_slow
        
        signals.append(sig)

        # Обновляем состояние позиции на основе сигнала
        # LONG сигнал: если позиции нет - открываем LONG, если позиция LONG - добавляем, если позиция SHORT - закрываем SHORT и открываем LONG
        if sig.action == Action.LONG:
            if position_bias is None:
                position_bias = Bias.LONG
                entry_price = sig.price
            elif position_bias == Bias.LONG:
                # Позиция уже LONG - остаемся в LONG (добавление будет обработано в live.py)
                pass
            elif position_bias == Bias.SHORT:
                # Позиция SHORT, сигнал LONG - закрываем SHORT и открываем LONG
                position_bias = Bias.LONG
                entry_price = sig.price
        # SHORT сигнал: аналогично
        elif sig.action == Action.SHORT:
            if position_bias is None:
                position_bias = Bias.SHORT
                entry_price = sig.price
            elif position_bias == Bias.SHORT:
                # Позиция уже SHORT - остаемся в SHORT
                pass
            elif position_bias == Bias.LONG:
                # Позиция LONG, сигнал SHORT - закрываем LONG и открываем SHORT
                position_bias = Bias.SHORT
                entry_price = sig.price
        # HOLD сигнал: ничего не меняем
        elif sig.action == Action.HOLD:
            pass  # Позиция остается как есть
    
    return signals
