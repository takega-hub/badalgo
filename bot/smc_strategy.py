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


@dataclass
class SMCZone:
    """Универсальный класс для зон SMC (FVG или Order Block)."""
    bar_index: int
    timestamp: pd.Timestamp
    upper: float
    lower: float
    direction: str  # "bullish" или "bearish"
    zone_type: str   # "FVG" или "OB"
    ref_index: Optional[int] = None


class SMCStrategy:
    """Класс стратегии Smart Money Concepts."""
    
    def __init__(self, params):
        self.params = params

    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Основной метод получения сигналов.
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Торговая пара для логирования
            
        Returns:
            Список сигналов SMC
        """
        if len(df) < 200:  # Минимум для EMA 200
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
        last_row = df.iloc[-1]
        
        # Безопасно получаем timestamp последней свечи
        last_ts = last_row.get('timestamp', last_row.name)
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.to_datetime(last_ts)
            
        close_price = closes[current_idx]
        curr_ema = ema_200[current_idx]
        
        # Определяем контекст тренда
        is_bullish_context = close_price > curr_ema
        is_bearish_context = close_price < curr_ema

        # Сохраняем ссылку на df для использования в других методах (tp2, volume и т.д.)
        self._df = df

        # 2. Поиск зон
        fvg_zones = self._find_fvg(df, highs, lows, opens, closes, times)
        ob_zones = self._find_ob(df, highs, lows, opens, closes, times)
        all_zones = fvg_zones + ob_zones

        signals = []

        # 3. Обработка зон и генерация сигналов
        for zone in all_zones:
            # А) Фильтр по возрасту
            # iFVG считается как FVG для целей возраста
            if zone.zone_type in ("FVG", "iFVG"):
                max_age = self.params.smc_max_fvg_age_bars
            else:
                max_age = self.params.smc_max_ob_age_bars
            if (current_idx - zone.bar_index) > max_age:
                continue

            # Б) Фильтр по тренду (EMA 200)
            if zone.direction == "bullish" and not is_bullish_context:
                continue
            if zone.direction == "bearish" and not is_bearish_context:
                continue

            # В) Проверка на Mitigation (была ли зона пробита ранее)
            if self._is_mitigated(zone, highs, lows, current_idx):
                continue

            # Г) Фильтр торговой сессии (только для входа)
            if getattr(self.params, 'smc_enable_session_filter', True):
                if not self._is_trading_session(last_ts):
                    continue

            # Д1) Фильтр Premium/Discount (локальный диапазон)
            smc_range_lookback = getattr(self.params, 'smc_range_lookback', 50)
            try:
                start_idx = max(0, current_idx - smc_range_lookback + 1)
                seg = df.iloc[start_idx: current_idx + 1]
                local_high = seg['high'].max()
                local_low = seg['low'].min()
                midpoint = local_low + (local_high - local_low) * 0.5
                if zone.direction == 'bullish' and not (close_price < midpoint):
                    # Разрешаем LONG только в зоне Discount (ниже 0.5 от диапазона)
                    continue
                if zone.direction == 'bearish' and not (close_price > midpoint):
                    # Разрешаем SHORT только в зоне Premium (выше 0.5 от диапазона)
                    continue
            except Exception:
                # В случае проблем с расчетом — не блокируем сигнал
                pass

            # Д) Логика входа при касании
            signal = self._check_entry(zone, last_row, close_price)
            if signal:
                signals.append(signal)
                # Убрали логирование в CSV - слишком много сообщений
                # self._log_signal_to_csv(signal, symbol)

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
                        # Проверка объема: свеча i-1 или свеча i должна иметь объем >= 1.5 * SMA20
                        vol_ok = True
                        if 'volume' in df.columns:
                            vol_series = df['volume']
                            vol_sma = vol_series.rolling(window=20).mean()
                            vol_ok = False
                            try:
                                if vol_series.iloc[i-1] >= vol_sma.iloc[i-1] * 1.5:
                                    vol_ok = True
                                elif vol_series.iloc[i] >= vol_sma.iloc[i] * 1.5:
                                    vol_ok = True
                            except Exception:
                                vol_ok = True
                        if vol_ok:
                            # Найдём ближайший структурный максимум слева для tp2 (если есть)
                            ref_idx = None
                            try:
                                if i > 0:
                                    left_highs = highs[:i]
                                    if len(left_highs) > 0:
                                        max_val = left_highs.max()
                                        refs = np.where(left_highs == max_val)[0]
                                        if len(refs) > 0:
                                            ref_idx = int(refs[-1])
                            except Exception:
                                ref_idx = None
                            zones.append(SMCZone(
                                bar_index=i,
                                timestamp=pd.Timestamp(times[i]),
                                upper=lows[i],
                                lower=highs[i-2],
                                direction="bullish",
                                zone_type="FVG",
                                ref_index=ref_idx
                            ))
            
            # Bearish FVG (Разрыв между Low i-2 и High i)
            elif highs[i] < lows[i-2]:
                gap_pct = (lows[i-2] - highs[i]) / lows[i-2] if lows[i-2] > 0 else 0
                if gap_pct >= min_gap:
                    body_size = abs(closes[i-1] - opens[i-1])
                    if not use_atr or body_size >= (atrs[i-1] * atr_mult):
                        vol_ok = True
                        if 'volume' in df.columns:
                            vol_series = df['volume']
                            vol_sma = vol_series.rolling(window=20).mean()
                            vol_ok = False
                            try:
                                if vol_series.iloc[i-1] >= vol_sma.iloc[i-1] * 1.5:
                                    vol_ok = True
                                elif vol_series.iloc[i] >= vol_sma.iloc[i] * 1.5:
                                    vol_ok = True
                            except Exception:
                                vol_ok = True
                        if vol_ok:
                            # Найдём ближайший структурный минимум слева для tp2 (если есть)
                            ref_idx = None
                            try:
                                if i > 0:
                                    left_lows = lows[:i]
                                    if len(left_lows) > 0:
                                        min_val = left_lows.min()
                                        refs = np.where(left_lows == min_val)[0]
                                        if len(refs) > 0:
                                            ref_idx = int(refs[-1])
                            except Exception:
                                ref_idx = None
                            zones.append(SMCZone(
                                bar_index=i,
                                timestamp=pd.Timestamp(times[i]),
                                upper=lows[i-2],
                                lower=highs[i],
                                direction="bearish",
                                zone_type="FVG",
                                ref_index=ref_idx
                            ))
        # Реализуем инверсию FVG (iFVG): если медвежий FVG пробит снизу вверх и закрепился — становиться бычьим iFVG
        try:
            if getattr(self.params, 'smc_enable_ifvg', False):
                confirm = getattr(self.params, 'smc_ifvg_confirm_bars', 3)
                for z in zones:
                    if z.zone_type == 'FVG' and z.direction == 'bearish':
                        start = z.bar_index + 1
                        end = min(start + confirm, len(df))
                        if start < end:
                            closes_segment = df['close'].iloc[start:end]
                            # Требуем, чтобы все подтверждающие бары закрылись выше верхней границы FVG
                            if len(closes_segment) >= confirm and (closes_segment > z.upper).all():
                                # Инвертируем зону
                                z.direction = 'bullish'
                                z.zone_type = 'iFVG'
        except Exception:
            pass
        return zones

    def _find_ob(self, df, highs, lows, opens, closes, times) -> List[SMCZone]:
        """Поиск зон Order Block на основе BOS."""
        zones = []
        lookback = self.params.smc_ob_lookback
        min_move = self.params.smc_ob_min_move_pct
        require_fvg = getattr(self.params, 'smc_ob_require_fvg', True)
        
        # Быстрый поиск экстремумов
        window = lookback * 2 + 1
        if window >= len(df):
            window = 3

        # Используем обычный rolling(window) без center=True чтобы исключить look-ahead
        is_max = df['high'] == df['high'].rolling(window).max()
        is_min = df['low'] == df['low'].rolling(window).min()

        max_indices = np.where(is_max)[0]
        min_indices = np.where(is_min)[0]

        # Предрасчет SMA по объему (используется в нескольких местах) — вычисляем один раз
        vol_sma = None
        if 'volume' in df.columns:
            try:
                vol_sma = df['volume'].rolling(window=20).mean()
            except Exception:
                vol_sma = None

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
                                    # Проверка объема: свеча OB (j) или импульсная свеча (i) должна иметь объем >= 1.5 * SMA20
                                    vol_ok = True
                                    if 'volume' in df.columns and vol_sma is not None:
                                        vol_ok = False
                                        try:
                                            if df['volume'].iloc[j] >= vol_sma.iloc[j] * 1.5:
                                                vol_ok = True
                                            elif df['volume'].iloc[i] >= vol_sma.iloc[i] * 1.5:
                                                vol_ok = True
                                        except Exception:
                                            vol_ok = True
                                    if vol_ok:
                                        # Используем диапазон манипуляции (S-to-B или B-to-S) для зон OB
                                        try:
                                            seg_high = float(np.max(highs[j:i+1]))
                                            seg_low = float(np.min(lows[j:i+1]))
                                        except Exception:
                                            seg_high = float(highs[j])
                                            seg_low = float(lows[j])
                                        zones.append(SMCZone(j, pd.Timestamp(times[j]), seg_high, seg_low, "bullish", "OB", ref_index=last_max_idx))
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
                                    vol_ok = True
                                    if 'volume' in df.columns and vol_sma is not None:
                                        vol_ok = False
                                        try:
                                            if df['volume'].iloc[j] >= vol_sma.iloc[j] * 1.5:
                                                vol_ok = True
                                            elif df['volume'].iloc[i] >= vol_sma.iloc[i] * 1.5:
                                                vol_ok = True
                                        except Exception:
                                            vol_ok = True
                                    if vol_ok:
                                        # Используем диапазон манипуляции (S-to-B или B-to-S) для зон OB
                                        try:
                                            seg_high = float(np.max(highs[j:i+1]))
                                            seg_low = float(np.min(lows[j:i+1]))
                                        except Exception:
                                            seg_high = float(highs[j])
                                            seg_low = float(lows[j])
                                        zones.append(SMCZone(j, pd.Timestamp(times[j]), seg_high, seg_low, "bearish", "OB", ref_index=last_min_idx))
                            break
        return zones

    def _is_mitigated(self, zone, highs, lows, current_idx) -> bool:
        """Проверка: была ли зона пробита ценой после создания."""
        check_start = zone.bar_index + 1
        if check_start >= current_idx:
            return False
        # Новая логика: проверяем закрытие свечи (close) за границей зоны вместо тени
        try:
            closes = self._df['close'].values if hasattr(self, '_df') and 'close' in self._df.columns else None
            if closes is not None:
                if zone.direction == "bullish":
                    # Бычья зона смягчена, если свеча закрылась ниже её нижней границы
                    return np.any(closes[check_start:current_idx] <= zone.lower)
                else:
                    # Медвежья зона смягчена, если свеча закрылась выше её верхней границы
                    return np.any(closes[check_start:current_idx] >= zone.upper)
        except Exception:
            pass

        # Fallback — старая логика по тени
        if zone.direction == "bullish":
            return np.any(lows[check_start:current_idx] <= zone.lower)
        else:
            return np.any(highs[check_start:current_idx] >= zone.upper)

    def _is_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """
        Проверяет, входит ли время свечи в активные торговые сессии (UTC).
        """
        # Приводим timestamp к UTC чтобы избежать рассинхрона часовых поясов
        try:
            ts_utc = pd.to_datetime(timestamp).tz_localize(None)
            # если timestamp уже с timezone, конвертируем в UTC
            if getattr(pd.to_datetime(timestamp), 'tzinfo', None) is not None:
                ts_utc = pd.to_datetime(timestamp).tz_convert('UTC')
        except Exception:
            ts_utc = pd.to_datetime(timestamp)
        hour = ts_utc.hour
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
        tolerance = zone.upper * self.params.smc_touch_tolerance_pct
        rr_ratio = getattr(self.params, 'smc_rr_ratio', 3.0)
        
        # Безопасно получаем время последней свечи
        last_ts = last_row.get('timestamp', last_row.name)
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.to_datetime(last_ts)

        spread_pct = getattr(self.params, 'smc_spread_pct', 0.0)
        # Учитываем спред: для лонга увеличиваем цену входа на spread, для шорта уменьшаем
        adj_price = close_price
        if spread_pct and spread_pct > 0:
            if zone.direction == "bullish":
                adj_price = close_price + close_price * spread_pct
            elif zone.direction == "bearish":
                adj_price = close_price - close_price * spread_pct

        if zone.direction == "bullish":
            # Вход при касании верхней границы сверху вниз
            if last_row['low'] <= (zone.upper + tolerance) and adj_price > zone.lower:
                sl = zone.lower - (close_price * 0.0005)
                # Защита от слишком узкого стопа
                if (close_price - sl) < close_price * 0.001:
                    sl = close_price * 0.999
                
                tp = adj_price + (adj_price - sl) * rr_ratio
                sig = Signal(
                    timestamp=last_ts,
                    action=Action.LONG,
                    reason=f"SMC_{zone.zone_type}_TREND_ENTRY",
                    price=round(adj_price, 8),
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2)
                )
                # Добавляем tp2 как ближайший структурный максимум (если есть ref_index)
                try:
                    if hasattr(zone, 'ref_index') and zone.ref_index is not None and hasattr(self, '_df'):
                        df = self._df
                        last_max_idx = int(zone.ref_index)
                        if 0 <= last_max_idx < len(df):
                            last_max = df.iloc[last_max_idx]['high']
                            sig.tp2 = round(float(last_max), 2)
                except Exception:
                    pass
                return sig
        
        elif zone.direction == "bearish":
            # Вход при касании нижней границы снизу вверх
            if last_row['high'] >= (zone.lower - tolerance) and adj_price < zone.upper:
                sl = zone.upper + (adj_price * 0.0005)
                if (sl - close_price) < close_price * 0.001:
                    sl = close_price * 1.001
                    
                tp = adj_price - (sl - adj_price) * rr_ratio
                sig = Signal(
                    timestamp=last_ts,
                    action=Action.SHORT,
                    reason=f"SMC_{zone.zone_type}_TREND_ENTRY",
                    price=round(adj_price, 8),
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2)
                )
                try:
                    if hasattr(zone, 'ref_index') and zone.ref_index is not None and hasattr(self, '_df'):
                        df = self._df
                        last_min_idx = int(zone.ref_index)
                        if 0 <= last_min_idx < len(df):
                            last_min = df.iloc[last_min_idx]['low']
                            sig.tp2 = round(float(last_min), 2)
                except Exception:
                    pass
                return sig
        
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
