"""
Smart Money Concepts (SMC) стратегия для торгового бота.

SMC основана на концепции, что рынок двигают крупные капиталы (Smart Money),
которые оставляют следы на графике. Стратегия ищет эти следы и входит в сделки вместе с ними.

Особенности реализации:
1. Высокая производительность через NumPy.
2. Фильтр глобального тренда (EMA 200).
3. Проверка Mitigation (смягчения) зон.
4. Динамический расчет SL/TP на основе границ зон.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bot.strategy import Action, Signal

# Глобальный словарь для отслеживания последних сигналов по символам
_last_signal_times = {}


@dataclass
class SMCZone:
    """Универсальный класс для зон SMC (FVG или Order Block)."""
    bar_index: int
    timestamp: pd.Timestamp
    upper: float
    lower: float
    direction: str  # "bullish" или "bearish"
    zone_type: str   # "FVG" или "OB"


class SMCStrategy:
    """Класс стратегии Smart Money Concepts."""

    def __init__(self, params):
        self.params = params

    def _should_generate_signal(self, symbol: str, current_timestamp: pd.Timestamp) -> bool:
        """
        Проверяет, можно ли генерировать сигнал для данного символа в текущий момент времени.
        Сигналы генерируются только на закрытых свечах каждые 15 минут.
        """
        global _last_signal_times

        # 1. Проверка на 15-минутный интервал (0, 15, 30, 45 минут)
        if current_timestamp.minute % 15 != 0:
            return False

        # 2. Проверка, что в этом 15-минутном блоке сигнала еще не было
        current_bucket = current_timestamp.replace(second=0, microsecond=0)
        
        last_signal_time = _last_signal_times.get(symbol)
        if last_signal_time is not None:
            if isinstance(last_signal_time, pd.Timestamp):
                last_bucket = last_signal_time.replace(second=0, microsecond=0)
            else:
                last_bucket = pd.to_datetime(last_signal_time).replace(second=0, microsecond=0)
                
            if last_bucket == current_bucket:
                return False

        return True

    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Основной метод получения сигналов.
        """
        if len(df) < 200:  # Минимум для EMA 200
            return []

        # Получаем timestamp последней свечи для фильтрации
        last_row = df.iloc[-1]
        last_ts = last_row.get('timestamp', last_row.name)
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.to_datetime(last_ts)

        # Проверка 15-минутного фильтра
        if not self._should_generate_signal(symbol, last_ts):
            return []

        # 1. Подготовка данных в NumPy для скорости
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        
        # Безопасно получаем время
        if 'timestamp' in df.columns:
            times = df['timestamp'].values
        else:
            times = df.index.values
            
        # Рассчитываем индикаторы
        # EMA 200 для определения тренда
        ema_200 = df['close'].ewm(span=200, adjust=False).mean().values
        
        current_idx = len(df) - 1
        close_price = closes[current_idx]
        curr_ema = ema_200[current_idx]
        
        # Определяем контекст тренда
        is_bullish_context = close_price > curr_ema
        is_bearish_context = close_price < curr_ema

        # 2. Поиск зон
        fvg_zones = self._find_fvg(df, highs, lows, opens, closes, times)
        ob_zones = self._find_ob(df, highs, lows, opens, closes, times)
        all_zones = fvg_zones + ob_zones

        signals = []
        
        # Статистика фильтрации для диагностики
        stats = {
            'total_zones': len(all_zones),
            'too_old': 0,
            'trend_filter_failed': 0,
            'mitigated': 0,
            'session_filter_blocked': 0,
            'no_touch': 0,
        }

        # 3. Обработка зон и генерация сигналов
        for zone in all_zones:
            # А) Фильтр по возрасту
            max_age = self.params.smc_max_fvg_age_bars if zone.zone_type == "FVG" else self.params.smc_max_ob_age_bars
            if (current_idx - zone.bar_index) > max_age:
                stats['too_old'] += 1
                continue

            # Б) Фильтр по тренду (EMA 200) - делаем менее строгим
            # Для зон в диапазоне касания тренд-фильтр опциональный
            zone_size = zone.upper - zone.lower
            price_distance = min(abs(close_price - zone.upper), abs(close_price - zone.lower), 
                                abs(close_price - (zone.upper + zone.lower) / 2))
            in_touch_range = price_distance <= zone_size * 0.1  # В пределах 10% от размера зоны
            
            # Если цена близко к зоне, тренд-фильтр не обязателен
            if not in_touch_range:
                if zone.direction == "bullish" and not is_bullish_context:
                    stats['trend_filter_failed'] += 1
                    continue
                if zone.direction == "bearish" and not is_bearish_context:
                    stats['trend_filter_failed'] += 1
                    continue

            # В) Проверка на Mitigation (была ли зона пробита ранее)
            if self._is_mitigated(zone, highs, lows, current_idx):
                stats['mitigated'] += 1
                continue

            # Г) Фильтр торговой сессии (только для входа)
            if getattr(self.params, 'smc_enable_session_filter', True):
                if not self._is_trading_session(last_ts):
                    stats['session_filter_blocked'] += 1
                    continue

            # Д) Логика входа при касании
            signal = self._check_entry(zone, last_row, close_price)
            if signal:
                signals.append(signal)
                # Убрали логирование в CSV - слишком много сообщений
                # self._log_signal_to_csv(signal, symbol)
            else:
                stats['no_touch'] += 1
                # Детальная диагностика для зон в диапазоне касания (только для первых 3)
                if stats['no_touch'] <= 3:
                    zone_size = zone.upper - zone.lower
                    tolerance = max(zone_size * self.params.smc_touch_tolerance_pct, 
                                  close_price * self.params.smc_touch_tolerance_pct)
                    import logging
                    logger = logging.getLogger(__name__)
                    
                    if zone.direction == "bullish":
                        touch_upper = last_row['low'] <= (zone.upper + tolerance)
                        above_lower = close_price >= (zone.lower - tolerance * 5)
                        in_zone = zone.lower <= close_price <= zone.upper
                        near_zone = abs(close_price - zone.upper) <= tolerance * 2 or abs(close_price - zone.lower) <= tolerance * 2
                        
                        logger.debug(f"[SMC] Zone {zone.bar_index} ({zone.zone_type}) BULLISH - "
                                   f"touch_upper={touch_upper}, above_lower={above_lower}, in_zone={in_zone}, near_zone={near_zone}, "
                                   f"low={last_row['low']:.2f}, upper={zone.upper:.2f}, close={close_price:.2f}, lower={zone.lower:.2f}, "
                                   f"tolerance={tolerance:.2f}, zone_size={zone_size:.2f}")
                    elif zone.direction == "bearish":
                        touch_lower = last_row['high'] >= (zone.lower - tolerance)
                        below_upper = close_price <= (zone.upper + tolerance * 5)
                        in_zone = zone.lower <= close_price <= zone.upper
                        near_zone = abs(close_price - zone.upper) <= tolerance * 2 or abs(close_price - zone.lower) <= tolerance * 2
                        
                        logger.debug(f"[SMC] Zone {zone.bar_index} ({zone.zone_type}) BEARISH - "
                                   f"touch_lower={touch_lower}, below_upper={below_upper}, in_zone={in_zone}, near_zone={near_zone}, "
                                   f"high={last_row['high']:.2f}, lower={zone.lower:.2f}, close={close_price:.2f}, upper={zone.upper:.2f}, "
                                   f"tolerance={tolerance:.2f}, zone_size={zone_size:.2f}")
        
        # Логируем статистику если нет сигналов
        if len(signals) == 0 and stats['total_zones'] > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[SMC] {symbol} Filter stats: {stats}")
            # Выводим статистику в консоль для диагностики
            if stats['total_zones'] > 0:
                print(f"      [SMC Debug] Filter breakdown:")
                print(f"         - Too old: {stats['too_old']}")
                print(f"         - Trend filter failed: {stats['trend_filter_failed']}")
                print(f"         - Mitigated: {stats['mitigated']}")
                print(f"         - Session blocked: {stats['session_filter_blocked']}")
                print(f"         - No touch: {stats['no_touch']}")

        if signals:
            global _last_signal_times
            _last_signal_times[symbol] = last_ts

        return signals

    def _find_fvg(self, df, highs, lows, opens, closes, times) -> List[SMCZone]:
        """Поиск зон Fair Value Gap."""
        zones = []
        atrs = df['atr'].values if 'atr' in df.columns else np.zeros(len(df))
        min_gap = self.params.smc_fvg_min_gap_pct
        atr_mult = getattr(self.params, 'smc_fvg_atr_multiplier', 1.5)
        use_atr = getattr(self.params, 'smc_fvg_use_atr_filter', True)
        
        for i in range(2, len(df)):
            # Bullish FVG (Разрыв между High i-2 и Low i)
            if lows[i] > highs[i-2]:
                gap_pct = (lows[i] - highs[i-2]) / highs[i-2] if highs[i-2] > 0 else 0
                if gap_pct >= min_gap:
                    # Фильтр по импульсу (тело свечи i-1)
                    body_size = abs(closes[i-1] - opens[i-1])
                    if not use_atr or body_size >= (atrs[i-1] * atr_mult):
                        zones.append(SMCZone(
                            bar_index=i,
                            timestamp=pd.Timestamp(times[i]),
                            upper=lows[i],
                            lower=highs[i-2],
                            direction="bullish",
                            zone_type="FVG"
                        ))
            
            # Bearish FVG (Разрыв между Low i-2 и High i)
            elif highs[i] < lows[i-2]:
                gap_pct = (lows[i-2] - highs[i]) / lows[i-2] if lows[i-2] > 0 else 0
                if gap_pct >= min_gap:
                    body_size = abs(closes[i-1] - opens[i-1])
                    if not use_atr or body_size >= (atrs[i-1] * atr_mult):
                        zones.append(SMCZone(
                            bar_index=i,
                            timestamp=pd.Timestamp(times[i]),
                            upper=lows[i-2],
                            lower=highs[i],
                            direction="bearish",
                            zone_type="FVG"
                        ))
        return zones

    def _find_ob(self, df, highs, lows, opens, closes, times) -> List[SMCZone]:
        """Поиск зон Order Block на основе BOS."""
        zones = []
        lookback = self.params.smc_ob_lookback
        min_move = self.params.smc_ob_min_move_pct
        require_fvg = getattr(self.params, 'smc_ob_require_fvg', True)
        
        # Быстрый поиск экстремумов
        window = lookback * 2 + 1
        if window >= len(df): window = 3
        
        is_max = df['high'] == df['high'].rolling(window, center=True).max()
        is_min = df['low'] == df['low'].rolling(window, center=True).min()
        
        max_indices = np.where(is_max)[0]
        min_indices = np.where(is_min)[0]

        for i in range(max(lookback, 1), len(df)):
            # BOS Up -> Ищем Bullish OB (последняя медвежья свеча перед ростом)
            valid_max_idxs = max_indices[max_indices < i]
            if len(valid_max_idxs) > 0:
                last_max_idx = valid_max_idxs[-1]
                if highs[i] > highs[last_max_idx]:
                    # Нашли BOS! Ищем OB в диапазоне
                    for j in range(i-1, last_max_idx-1, -1):
                        if closes[j] < opens[j]: # Медвежья свеча
                            move = (highs[i] - lows[j]) / lows[j] if lows[j] > 0 else 0
                            if move >= min_move:
                                # Проверка FVG подтверждения
                                has_fvg = True
                                if require_fvg:
                                    has_fvg = False
                                    for k in range(j + 1, min(j + 11, i + 1)):
                                        if k < 2: continue
                                        if lows[k] > highs[k-2]:
                                            has_fvg = True
                                            break
                                if has_fvg:
                                    zones.append(SMCZone(j, pd.Timestamp(times[j]), highs[j], lows[j], "bullish", "OB"))
                            break

            # BOS Down -> Ищем Bearish OB (последняя бычья свеча перед падением)
            valid_min_idxs = min_indices[min_indices < i]
            if len(valid_min_idxs) > 0:
                last_min_idx = valid_min_idxs[-1]
                if lows[i] < lows[last_min_idx]:
                    for j in range(i-1, last_min_idx-1, -1):
                        if closes[j] > opens[j]: # Бычья свеча
                            move = (highs[j] - lows[i]) / highs[j] if highs[j] > 0 else 0
                            if move >= min_move:
                                has_fvg = True
                                if require_fvg:
                                    has_fvg = False
                                    for k in range(j + 1, min(j + 11, i + 1)):
                                        if k < 2: continue
                                        if highs[k] < lows[k-2]:
                                            has_fvg = True
                                            break
                                if has_fvg:
                                    zones.append(SMCZone(j, pd.Timestamp(times[j]), highs[j], lows[j], "bearish", "OB"))
                            break
        return zones

    def _is_mitigated(self, zone, highs, lows, current_idx) -> bool:
        """Проверка: была ли зона пробита ценой после создания."""
        check_start = zone.bar_index + 1
        if check_start >= current_idx:
            return False
            
        if zone.direction == "bullish":
            # Бычья зона смягчена, если цена ушла ниже её границы
            return np.any(lows[check_start:current_idx] <= zone.lower)
        else:
            # Медвежья зона смягчена, если цена ушла выше её границы
            return np.any(highs[check_start:current_idx] >= zone.upper)

    def _is_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """
        Проверяет, входит ли время свечи в активные торговые сессии (UTC).
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # В выходные волатильность низкая, SMC работает хуже
        if weekday >= 5:
            return False
            
        london_start = getattr(self.params, 'smc_session_london_start', 7)
        london_end = getattr(self.params, 'smc_session_london_end', 10)
        ny_start = getattr(self.params, 'smc_session_ny_start', 12)
        ny_end = getattr(self.params, 'smc_session_ny_end', 15)
        
        is_london = london_start <= hour <= london_end
        is_ny = ny_start <= hour <= ny_end
        
        return is_london or is_ny

    def _check_entry(self, zone, last_row, close_price) -> Optional[Signal]:
        """Проверка условий входа и расчет уровней SL/TP."""
        # Tolerance рассчитывается как процент от размера зоны или от цены
        zone_size = zone.upper - zone.lower
        tolerance = max(zone_size * self.params.smc_touch_tolerance_pct, close_price * self.params.smc_touch_tolerance_pct)
        rr_ratio = getattr(self.params, 'smc_rr_ratio', 3.0)
        
        # Безопасно получаем время последней свечи
        last_ts = last_row.get('timestamp', last_row.name)
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.to_datetime(last_ts)

        if zone.direction == "bullish":
            # Вход при касании верхней границы сверху вниз или ретест зоны
            # Проверяем: low свечи коснулся верхней границы зоны (с tolerance)
            touch_upper = last_row['low'] <= (zone.upper + tolerance)
            
            # Также проверяем ретест зоны - цена прошла выше зоны и вернулась к ней
            # Для ретеста проверяем, что цена выше зоны, но low коснулся верхней границы
            price_above_zone = close_price > zone.upper
            retest_upper = price_above_zone and last_row['low'] <= (zone.upper + tolerance * 3)
            
            # И цена закрытия находится в зоне или выше нижней границы
            above_lower = close_price >= (zone.lower - tolerance * 5)
            
            # Альтернативное условие: цена находится внутри зоны или очень близко к ней
            in_zone = zone.lower <= close_price <= zone.upper
            near_zone = abs(close_price - zone.upper) <= tolerance * 3 or abs(close_price - zone.lower) <= tolerance * 3
            
            # Также проверяем близость к зоне снизу (для зон выше текущей цены)
            near_below = close_price < zone.lower and (zone.lower - close_price) <= (zone_size * 0.2)
            
            if (touch_upper and above_lower) or retest_upper or in_zone or near_zone or near_below:
                sl = zone.lower - (close_price * 0.0005)
                # Защита от слишком узкого стопа
                if (close_price - sl) < close_price * 0.001:
                    sl = close_price * 0.999
                
                tp = close_price + (close_price - sl) * rr_ratio
                return Signal(
                    timestamp=last_ts,
                    action=Action.LONG,
                    reason=f"SMC_{zone.zone_type}_TREND_ENTRY",
                    price=close_price,
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2)
                )
        
        elif zone.direction == "bearish":
            # Вход при касании нижней границы снизу вверх или ретест зоны
            # Проверяем: high свечи коснулся нижней границы зоны (с tolerance)
            touch_lower = last_row['high'] >= (zone.lower - tolerance)
            
            # Также проверяем ретест зоны - цена прошла ниже зоны и вернулась к ней
            # Для ретеста проверяем, что цена ниже зоны, но high коснулся нижней границы
            price_below_zone = close_price < zone.lower
            retest_lower = price_below_zone and last_row['high'] >= (zone.lower - tolerance * 3)
            
            # И цена закрытия находится в зоне или ниже верхней границы
            below_upper = close_price <= (zone.upper + tolerance * 5)
            
            # Альтернативное условие: цена находится внутри зоны или очень близко к ней
            in_zone = zone.lower <= close_price <= zone.upper
            near_zone = abs(close_price - zone.upper) <= tolerance * 3 or abs(close_price - zone.lower) <= tolerance * 3
            
            # Также проверяем близость к зоне сверху (для зон ниже текущей цены)
            near_above = close_price > zone.upper and (close_price - zone.upper) <= (zone_size * 0.2)
            
            if (touch_lower and below_upper) or retest_lower or in_zone or near_zone or near_above:
                sl = zone.upper + (close_price * 0.0005)
                if (sl - close_price) < close_price * 0.001:
                    sl = close_price * 1.001
                    
                tp = close_price - (sl - close_price) * rr_ratio
                return Signal(
                    timestamp=last_ts,
                    action=Action.SHORT,
                    reason=f"SMC_{zone.zone_type}_TREND_ENTRY",
                    price=close_price,
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2)
                )
        
        return None
    
    def _log_signal_to_csv(self, signal: Signal, symbol: str):
        """
        Записывает сигнал в CSV файл для истории и анализа.
        
        Args:
            signal: Сигнал для записи
            symbol: Торговая пара
        """
        import csv
        import os
        from pathlib import Path
        
        file_path = Path(__file__).parent.parent / "smc_trade_history.csv"
        file_exists = file_path.exists()
        
        # Подготовка заголовков
        headers = [
            "timestamp", "symbol", "action", "price", 
            "stop_loss", "take_profit", "reason", "rr_ratio"
        ]
        
        # Форматируем timestamp в читаемый вид
        ts_str = signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp)
        
        # Данные для записи
        row = {
            "timestamp": ts_str,
            "symbol": symbol,
            "action": signal.action.value if hasattr(signal.action, 'value') else str(signal.action),
            "price": signal.price,
            "stop_loss": signal.stop_loss if signal.stop_loss else 0.0,
            "take_profit": signal.take_profit if signal.take_profit else 0.0,
            "reason": signal.reason,
            "rr_ratio": getattr(self.params, 'smc_rr_ratio', 2.5)
        }
        
        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            # Убрали избыточное логирование - слишком много сообщений
            # print(f"📝 SMC signal logged to {file_path.name}: {signal.action.value} {symbol} @ ${signal.price:.2f}")
        except Exception as e:
            print(f"❌ Error logging SMC signal: {e}")


def build_smc_signals(df: pd.DataFrame, params, symbol: str = "Unknown") -> List[Signal]:
    """
    Точка входа для бота. Использует класс SMCStrategy для генерации сигналов.
    
    Args:
        df: DataFrame с данными OHLCV
        params: Параметры стратегии
        symbol: Торговая пара для логирования
        
    Returns:
        Список сигналов SMC
    """
    strategy = SMCStrategy(params)
    return strategy.get_signals(df, symbol=symbol)
