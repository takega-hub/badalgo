from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
    
    def __post_init__(self):
        """Генерирует уникальный ID сигнала, если он не задан."""
        if self.signal_id is None:
            # Генерируем ID на основе timestamp, action, reason и price
            # ВАЖНО: Используем больше знаков для price и добавляем больше данных для уникальности
            ts_str = str(self.timestamp) if hasattr(self.timestamp, 'isoformat') else str(self.timestamp)
            price_str = f"{self.price:.6f}"  # Увеличено с 4 до 6 знаков для большей точности
            import hashlib
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
        # Если bias стал RANGE → HOLD (позиция будет закрыта автоматически при следующем сигнале)
        elif bias == Bias.RANGE:
            return Signal(row.name, Action.HOLD, "trend_bias_to_range", price)
    
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
    
    # Проверяем касание границ BB (цена или тень свечи)
    touch_lower = low <= bb_lower or price <= bb_lower  # касание нижней границы
    touch_upper = high >= bb_upper or price >= bb_upper  # касание верхней границы
    
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


def generate_signal(row: pd.Series, has_position: Optional[Bias], params: StrategyParams, entry_price: Optional[float] = None) -> Signal:
    """
    Главная функция генерации сигналов.
    Определяет фазу рынка и выбирает соответствующую стратегию.
    """
    market_phase = detect_market_phase(row, params)
    
    if market_phase == MarketPhase.TREND:
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
    df["is_consolidating"] = (rng_width / avg_price) <= params.consolidation_range_pct
    df["consolidation_high"] = rng_high
    df["consolidation_low"] = rng_low

    # Bias from 4H signals (используется только в трендовой фазе)
    df["bias"] = df.apply(lambda row: infer_bias(row, params), axis=1)
    
    # Market phase detection
    df["market_phase"] = df.apply(lambda row: detect_market_phase(row, params), axis=1)
    
    return df.dropna()


def build_signals(df: pd.DataFrame, params: StrategyParams) -> list[Signal]:
    """
    Строит сигналы на основе данных.
    Возвращает только LONG, SHORT или HOLD сигналы.
    """
    signals: list[Signal] = []
    position_bias: Optional[Bias] = None
    entry_price: Optional[float] = None  # отслеживаем цену входа для стоп-лосса в флэтовой стратегии
    
    for _, row in df.iterrows():
        sig = generate_signal(row, position_bias, params, entry_price)
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
