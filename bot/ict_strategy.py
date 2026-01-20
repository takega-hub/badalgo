"""
ICT (Inner Circle Trader) стратегия "Silver Bullet" для торгового бота.

Стратегия работает в определенные временные окна (сессии) и ищет:
1. Снятие ликвидности за пределами дневного максимума/минимума
2. Резкий возврат в диапазон (манипуляция)
3. Формирование FVG в сторону нового движения
4. Вход на ретест FVG

Особенности:
- Тайм-фильтр для Лондонской и Нью-Йоркской сессий
- Williams Alligator для фильтрации тренда
- ATR для динамического стоп-лосса
- Автоматический перевод в безубыток при R:R 1:1
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timezone
import numpy as np
import pandas as pd
import pytz

from bot.strategy import Action, Signal, Bias


@dataclass
class ICTFVG:
    """Fair Value Gap для ICT стратегии."""
    bar_index: int
    timestamp: pd.Timestamp
    upper: float
    lower: float
    direction: str  # "bullish" или "bearish"
    liquidity_bar_index: int  # Индекс свечи, которая сняла ликвидность
    active: bool = True       # FVG активен (не митигирован)


@dataclass
class ICTLiquidity:
    """Зона ликвидности (снятие за пределами дневного H/L)."""
    bar_index: int
    timestamp: pd.Timestamp
    price: float
    direction: str  # "above_high" или "below_low"
    daily_high: float
    daily_low: float


class ICTStrategy:
    """Класс стратегии ICT Silver Bullet."""
    
    def __init__(self, params):
        self.params = params
        
        # Таймзона Нью-Йорка для Silver Bullet окон (ICT ориентируется на NY Local Time)
        self.ny_tz = pytz.timezone("America/New_York")
        # Silver Bullet окна (NY local time, ET)
        # London SB: 03:00–04:00, NY AM: 10:00–11:00, NY PM: 14:00–15:00
        self.sb_windows_ny = [
            (time(3, 0), time(4, 0)),   # London SB
            (time(10, 0), time(11, 0)), # NY AM SB
            (time(14, 0), time(15, 0)), # NY PM SB
        ]
    
    def is_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """
        Проверяет, находится ли время в активном Silver Bullet окне (NY local time).
        
        Args:
            timestamp: Временная метка свечи
            
        Returns:
            True если время в Лондонской или Нью-Йоркской сессии
        """
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        
        # Приводим к UTC, затем в America/New_York
        if timestamp.tzinfo is None:
            ts_utc = timestamp.tz_localize('UTC')
        else:
            ts_utc = timestamp.tz_convert('UTC')
        ts_ny = ts_utc.astimezone(self.ny_tz)
        current_time_ny = ts_ny.time()
        
        # Silver Bullet окна в NY local time
        for start, end in self.sb_windows_ny:
            if start <= current_time_ny <= end:
                return True
        return False
    
    def calculate_williams_alligator(
        self, 
        df: pd.DataFrame,
        jaw_period: int = 13,
        teeth_period: int = 8,
        lips_period: int = 5,
        jaw_shift: int = 8,
        teeth_shift: int = 5,
        lips_shift: int = 3
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Рассчитывает индикатор Williams Alligator.
        
        Args:
            df: DataFrame с данными OHLCV
            jaw_period: Период для челюсти (медленная линия)
            teeth_period: Период для зубов (средняя линия)
            lips_period: Период для губ (быстрая линия)
            jaw_shift: Сдвиг челюсти
            teeth_shift: Сдвиг зубов
            lips_shift: Сдвиг губ
            
        Returns:
            Tuple (jaw, teeth, lips) - три линии аллигатора
        """
        # Используем медианную цену (high + low) / 2
        median_price = (df['high'] + df['low']) / 2

        # Быстрый векторизованный SMMA через экспоненциальное сглаживание
        # (ewm с alpha=1/period хорошо аппроксимирует SMMA и в разы быстрее циклов)
        jaw = median_price.ewm(alpha=1 / jaw_period, adjust=False).mean().shift(jaw_shift)
        teeth = median_price.ewm(alpha=1 / teeth_period, adjust=False).mean().shift(teeth_shift)
        lips = median_price.ewm(alpha=1 / lips_period, adjust=False).mean().shift(lips_shift)
        
        return jaw, teeth, lips
    
    def is_alligator_expanded(
        self,
        jaw: pd.Series,
        teeth: pd.Series,
        lips: pd.Series,
        index: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Проверяет, раскрыт ли аллигатор (начало тренда).
        
        Args:
            jaw: Линия челюсти
            teeth: Линия зубов
            lips: Линия губ
            index: Индекс для проверки
            
        Returns:
            Tuple (is_expanded, direction) где direction может быть "bullish", "bearish" или None
        """
        if index < 0 or index >= len(jaw):
            return False, None
        
        # Проверяем текущую свечу и несколько предыдущих для подтверждения тренда
        check_bars = min(3, index + 1)  # Проверяем до 3 свечей назад
        bullish_count = 0
        bearish_count = 0
        
        for i in range(max(0, index - check_bars + 1), index + 1):
            if i >= len(jaw) or i >= len(teeth) or i >= len(lips):
                continue
                
            jaw_val = jaw.iloc[i]
            teeth_val = teeth.iloc[i]
            lips_val = lips.iloc[i]
            
            if not all(pd.notna([jaw_val, teeth_val, lips_val])):
                continue
            
            # Бычий аллигатор: lips > teeth > jaw (линии вверх)
            if lips_val > teeth_val > jaw_val:
                bullish_count += 1
            
            # Медвежий аллигатор: lips < teeth < jaw (линии вниз)
            if lips_val < teeth_val < jaw_val:
                bearish_count += 1
        
        # Тренд подтверждается, если большинство свечей показывают одно направление
        # Делаем проверку менее строгой - достаточно 40% свечей или минимум 1 свеча
        if bullish_count >= max(1, check_bars * 0.4):
            return True, "bullish"
        
        if bearish_count >= max(1, check_bars * 0.4):
            return True, "bearish"
        
        # Если нет четкого тренда, но есть хотя бы одна свеча с раскрытым аллигатором - разрешаем
        # Это позволяет торговать в более широком диапазоне условий
        if bullish_count > 0 or bearish_count > 0:
            # Возвращаем направление большинства или бычий по умолчанию
            if bullish_count >= bearish_count:
                return True, "bullish"
            else:
                return True, "bearish"
        
        return False, None
    
    def find_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        lookback_days: int = 1
    ) -> List[ICTLiquidity]:
        """
        Находит снятия ликвидности за пределами дневного максимума/минимума.
        
        Args:
            df: DataFrame с данными OHLCV
            lookback_days: Количество дней для поиска
            
        Returns:
            Список зон ликвидности
        """
        if len(df) < 100:
            return []
        
        liquidity_zones: List[ICTLiquidity] = []
        
        # Копируем и приводим индекс к DatetimeIndex
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)
        
        # Приводим к UTC, затем в NY, чтобы определить сессии
        if df_copy.index.tzinfo is None:
            df_copy.index = df_copy.index.tz_localize('UTC')
        else:
            df_copy.index = df_copy.index.tz_convert('UTC')
        
        ts_ny = df_copy.index.tz_convert(self.ny_tz)
        df_copy["date_ny"] = ts_ny.date
        
        # Определяем сессию по NY local time
        def _session_for_time(t: time) -> str:
            # Примерное деление: Азия 00:00–08:00, Лондон 08:00–13:00, NY 13:00–21:00
            if 0 <= t.hour < 8:
                return "asia"
            if 8 <= t.hour < 13:
                return "london"
            if 13 <= t.hour < 21:
                return "ny"
            return "off"
        
        df_copy["session"] = [ _session_for_time(t.time()) for t in ts_ny ]
        # Убираем "off"‑часы
        df_copy = df_copy[df_copy["session"] != "off"]
        if df_copy.empty:
            return liquidity_zones
        
        # Ключ сессии: (дата, session)
        df_copy["session_key"] = list(zip(df_copy["date_ny"], df_copy["session"]))
        session_keys = list(dict.fromkeys(df_copy["session_key"].tolist()))  # сохраняем порядок
        
        # Используем последние lookback_days сессий
        session_keys = session_keys[-max(1, lookback_days * 3):]
        
        session_high_low = {}
        for sk in session_keys:
            sess_data = df_copy[df_copy["session_key"] == sk]
            if sess_data.empty:
                continue
            session_high_low[sk] = (
                sess_data["high"].max(),
                sess_data["low"].min(),
            )
        
        # Ищем свечи, которые выносят предыдущие сессионные high/low
        for idx_sk, sk in enumerate(session_keys):
            sess_data = df_copy[df_copy["session_key"] == sk]
            if sess_data.empty or len(sess_data) < 3:
                continue
            
            prev_sk = session_keys[idx_sk - 1] if idx_sk > 0 else None
            if prev_sk is None or prev_sk not in session_high_low:
                continue
            
            prev_high, prev_low = session_high_low[prev_sk]
            if prev_high is None or prev_low is None:
                continue
            
            for idx, row in sess_data.iterrows():
                # Ensure idx is compatible with df.index for get_loc
                search_idx = idx
                if df.index.tzinfo is None and idx.tzinfo is not None:
                    search_idx = idx.tz_localize(None)
                elif df.index.tzinfo is not None and idx.tzinfo is None:
                    search_idx = idx.tz_localize('UTC').tz_convert(df.index.tzinfo)
                
                try:
                    bar_idx = df.index.get_loc(search_idx)  # индекс в оригинальном df
                except (KeyError, TypeError):
                    # Fallback if exact match fails due to minor tz/format differences
                    continue
                
                # Sweep выше high предыдущей сессии
                if prev_high > 0 and row["high"] > prev_high * 0.999:
                    liquidity_zones.append(
                        ICTLiquidity(
                            bar_index=bar_idx,
                            timestamp=idx,
                            price=row["high"],
                            direction="above_high",
                            daily_high=prev_high,
                            daily_low=prev_low,
                        )
                    )
                
                # Sweep ниже low предыдущей сессии
                if prev_low < float("inf") and row["low"] < prev_low * 1.001:
                    liquidity_zones.append(
                        ICTLiquidity(
                            bar_index=bar_idx,
                            timestamp=idx,
                            price=row["low"],
                            direction="below_low",
                            daily_high=prev_high,
                            daily_low=prev_low,
                        )
                    )
        
        return liquidity_zones
    
    def find_fvg(
        self,
        df: pd.DataFrame,
        liquidity_sweeps: List[ICTLiquidity]
    ) -> List[ICTFVG]:
        """
        Находит Fair Value Gaps после снятия ликвидности.
        
        FVG формируется когда:
        1. Была свеча, которая сняла ликвидность
        2. Произошел резкий возврат (манипуляция)
        3. Образовался разрыв между свечами
        
        Args:
            df: DataFrame с данными OHLCV
            liquidity_sweeps: Список зон ликвидности
            
        Returns:
            Список FVG зон
        """
        if len(df) < 10:
            return []
        
        fvg_zones = []
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values
        
        # Для каждой зоны ликвидности ищем FVG
        for liq in liquidity_sweeps:
            if liq.bar_index >= len(df) - 3:
                continue
            
            # Ищем FVG в следующих 20 свечах после снятия ликвидности (увеличиваем окно)
            search_window = min(20, len(df) - liq.bar_index - 1)
            
            for i in range(liq.bar_index + 1, liq.bar_index + search_window):
                if i >= len(df) - 1 or i < 2:
                    break
                
                # Нужно минимум 3 свечи для определения FVG: i-2, i-1, i
                if i < 2:
                    continue
                
                prev_prev_high = highs[i-2]
                prev_prev_low = lows[i-2]
                prev_high = highs[i-1]
                prev_low = lows[i-1]
                curr_low = lows[i]
                curr_high = highs[i]
                curr_open = opens[i]
                curr_close = closes[i]
                prev_close = closes[i-1]
                
                # Бычий FVG: нижняя граница текущей свечи выше верхней границы свечи i-2
                # (между i-2 и i есть разрыв, свеча i-1 может быть любой)
                if curr_low > prev_prev_high:
                    # Проверяем, что после снятия ликвидности вверх произошел возврат
                    # (цена вернулась ниже уровня снятия ликвидности)
                    if liq.direction == "above_high":
                        # Проверяем возврат: текущая цена должна быть ниже уровня снятия ликвидности
                        # или хотя бы ниже максимума свечи, которая сняла ликвидность
                        if curr_close < liq.price or (i > liq.bar_index + 1 and any(closes[j] < liq.price for j in range(liq.bar_index + 1, i))):
                            fvg_zones.append(
                                ICTFVG(
                                    bar_index=i,
                                    timestamp=df.index[i],
                                    upper=curr_low,
                                    lower=prev_prev_high,
                                    direction="bullish",
                                    liquidity_bar_index=liq.bar_index,
                                )
                            )
                
                # Медвежий FVG: верхняя граница текущей свечи ниже нижней границы свечи i-2
                if curr_high < prev_prev_low:
                    # Проверяем, что после снятия ликвидности вниз произошел возврат
                    if liq.direction == "below_low":
                        # Проверяем возврат: текущая цена должна быть выше уровня снятия ликвидности
                        if curr_close > liq.price or (i > liq.bar_index + 1 and any(closes[j] > liq.price for j in range(liq.bar_index + 1, i))):
                            fvg_zones.append(
                                ICTFVG(
                                    bar_index=i,
                                    timestamp=df.index[i],
                                    upper=prev_prev_low,
                                    lower=curr_high,
                                    direction="bearish",
                                    liquidity_bar_index=liq.bar_index,
                                )
                            )
        
        return fvg_zones
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Основной метод получения сигналов ICT Silver Bullet.
        Генерирует сигналы для всех свечей в истории.
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Торговая пара для логирования
            
        Returns:
            Список сигналов ICT
        """
        if len(df) < 200:  # Минимум для индикаторов
            return []
        
        signals = []
        
        # Рассчитываем индикаторы один раз для всего DataFrame
        # 1. Williams Alligator
        jaw, teeth, lips = self.calculate_williams_alligator(df)
        
        # 2. ATR для динамического SL
        if 'atr' not in df.columns:
            try:
                import pandas_ta as ta
                atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
                df['atr'] = atr
            except:
                # Простой расчет ATR если pandas_ta недоступен
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=14).mean()
        
        # 3. Ищем все снятия ликвидности за весь период
        lookback_days = getattr(self.params, 'ict_liquidity_lookback_days', 1)
        # Увеличиваем lookback_days для поиска большего количества снятий ликвидности
        lookback_days = max(lookback_days, 5)
        liquidity_sweeps = self.find_liquidity_sweeps(df, lookback_days=lookback_days)
        
        # Диагностика: логируем количество найденных снятий ликвидности
        if len(liquidity_sweeps) == 0:
            print(f"[ICT] ⚠️  Не найдено снятий ликвидности за последние {lookback_days} дней")
            print(f"[ICT] 💡 Пробуем альтернативный метод поиска ликвидности...")
            # Альтернативный метод: ищем свечи, которые выходят за пределы предыдущих N свечей
            liquidity_sweeps = self.find_liquidity_sweeps_alternative(df, lookback_bars=50)
            # Убрали избыточное логирование - слишком много сообщений
            # if len(liquidity_sweeps) > 0:
            #     print(f"[ICT] ✅ Альтернативным методом найдено {len(liquidity_sweeps)} снятий ликвидности")
        # else:
        #     print(f"[ICT] ✅ Найдено {len(liquidity_sweeps)} снятий ликвидности")
        
        # 4. Ищем все FVG после снятия ликвидности
        fvg_zones = self.find_fvg(df, liquidity_sweeps)
        
        if not fvg_zones:
            # Убрали избыточное логирование
            # print(f"[ICT] ⚠️  Не найдено FVG зон после снятий ликвидности")
            return []  # Нет FVG
        # else:
        #     print(f"[ICT] ✅ Найдено {len(fvg_zones)} FVG зон")
        
        # Подготовка словаря ликвидности по bar_index для MSS / Premium/Discount фильтров
        liq_by_bar = {liq.bar_index: liq for liq in liquidity_sweeps}
        
        # 5. Проходим по всем свечам и генерируем сигналы
        # Начинаем с 200 для индикаторов
        candles_in_session = 0
        candles_with_trend = 0
        fvg_retests_checked = 0
        
        for i in range(200, len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            current_atr = df['atr'].iloc[i] if pd.notna(df['atr'].iloc[i]) else current_price * 0.02
            
            # Получаем timestamp текущей свечи
            current_ts = current_row.get('timestamp', df.index[i])
            if not isinstance(current_ts, pd.Timestamp):
                current_ts = pd.to_datetime(current_ts)
            
            # Проверяем, находимся ли в одном из Silver Bullet окон (NY local time)
            if not self.is_trading_session(current_ts):
                continue  # Пропускаем свечи вне Silver Bullet окон
            
            candles_in_session += 1
            
            # Проверяем, раскрыт ли аллигатор на текущей свече
            is_expanded, alligator_direction = self.is_alligator_expanded(jaw, teeth, lips, i)
            
            if not is_expanded:
                continue  # Аллигатор не раскрыт - нет тренда
            
            candles_with_trend += 1
            
            # Перед проверкой FVG помечаем митигированные зоны как неактивные
            for fvg in fvg_zones:
                if not fvg.active:
                    continue
                # Зона считается полностью митигированной, если минимум и максимум цены
                # внутри диапазона [fvg.lower, fvg.upper] встречались в истории после её формирования
                if i > fvg.bar_index + 1:
                    lows_since = df['low'].iloc[fvg.bar_index + 1 : i + 1].min()
                    highs_since = df['high'].iloc[fvg.bar_index + 1 : i + 1].max()
                    if lows_since <= fvg.lower and highs_since >= fvg.upper:
                        fvg.active = False

            # Проверяем все FVG зоны на ретест
            # Используем set для отслеживания уже обработанных FVG на этой свече
            processed_fvg_ids = set()
            
            for fvg in fvg_zones:
                if not fvg.active:
                    continue  # Пропускаем митигированные FVG
                
                # Проверка MSS (Market Structure Shift) между снятием ликвидности и FVG
                liq = liq_by_bar.get(fvg.liquidity_bar_index)
                if liq is not None:
                    # Простая MSS: после снятия ликвидности цена должна пробить локальный свинг
                    lookback = 20
                    prior_start = max(0, liq.bar_index - lookback)
                    prior_slice = df.iloc[prior_start:liq.bar_index]
                    mid_slice = df.iloc[liq.bar_index + 1 : fvg.bar_index + 1]
                    if not prior_slice.empty and not mid_slice.empty:
                        if fvg.direction == "bullish":
                            # Для бычьего сетапа: цена после лику должна пробить предыдущий high
                            recent_high = prior_slice["high"].max()
                            if mid_slice["close"].max() <= recent_high:
                                continue
                        elif fvg.direction == "bearish":
                            # Для медвежьего сетапа: цена после лику должна пробить предыдущий low
                            recent_low = prior_slice["low"].min()
                            if mid_slice["close"].min() >= recent_low:
                                continue
                # Проверяем возраст FVG (не старше max_age свечей)
                max_age = getattr(self.params, 'ict_fvg_max_age_bars', 20)
                # Увеличиваем max_age для более старых FVG (до 50 баров)
                max_age = max(max_age, 50)
                if (i - fvg.bar_index) > max_age:
                    continue
                
                # FVG должен быть создан до текущей свечи
                if fvg.bar_index >= i:
                    continue
                
                # Проверяем направление аллигатора и FVG (но делаем это менее строго)
                # Разрешаем вход, если направление совпадает или аллигатор нейтрален
                if fvg.direction == "bullish" and alligator_direction == "bearish":
                    continue  # Только блокируем противоположное направление
                if fvg.direction == "bearish" and alligator_direction == "bullish":
                    continue  # Только блокируем противоположное направление
                
                # Создаем уникальный ID для FVG (чтобы избежать дубликатов)
                fvg_id = (fvg.bar_index, fvg.direction, round(fvg.lower, 2), round(fvg.upper, 2))
                if fvg_id in processed_fvg_ids:
                    continue  # Пропускаем уже обработанный FVG
                
                processed_fvg_ids.add(fvg_id)
                fvg_retests_checked += 1
                
                # ФИЛЬТР КАЧЕСТВА 1: Проверяем минимальный размер FVG (слишком маленькие могут быть ложными)
                fvg_size = fvg.upper - fvg.lower
                min_fvg_size = current_atr * 0.2  # Минимум 20% от ATR (ослаблено с 30%)
                if fvg_size < min_fvg_size:
                    continue  # Пропускаем слишком маленькие FVG
                
                # ФИЛЬТР КАЧЕСТВА 2: Проверяем, что цена действительно "вернулась" в FVG
                # Для бычьего FVG: цена должна была быть выше зоны, а затем вернулась
                # Для медвежьего FVG: цена должна была быть ниже зоны, а затем вернулась
                # Делаем проверку менее строгой - разрешаем сигналы даже если цена не сильно вышла за зону
                if fvg.bar_index < i and (i - fvg.bar_index) > 2:  # Только если прошло минимум 2 свечи
                    # Проверяем движение после создания FVG
                    price_after_fvg = df['high'].iloc[fvg.bar_index+1:i+1].max() if fvg.direction == "bullish" else df['low'].iloc[fvg.bar_index+1:i+1].min()
                    
                    if fvg.direction == "bullish":
                        # Для бычьего FVG: цена должна была быть выше зоны (ослаблено - допуск 0.5%)
                        if price_after_fvg < fvg.upper * 1.005:  # Допуск 0.5% (ослаблено с 1%)
                            # Не блокируем полностью, но это слабый сигнал - пропускаем только если цена вообще не вышла
                            if price_after_fvg < fvg.upper * 0.998:  # Если цена даже не достигла зоны
                                continue
                    else:
                        # Для медвежьего FVG: цена должна была быть ниже зоны (ослаблено - допуск 0.5%)
                        if price_after_fvg > fvg.lower * 0.995:  # Допуск 0.5% (ослаблено с 1%)
                            # Не блокируем полностью, но это слабый сигнал - пропускаем только если цена вообще не вышла
                            if price_after_fvg > fvg.lower * 1.002:  # Если цена даже не достигла зоны
                                continue
                
                # Проверяем, находится ли цена в зоне FVG (ретест)
                # Используем более мягкое условие - цена может касаться зоны или быть внутри
                if fvg.direction == "bullish":
                    # Бычий FVG: вход если цена в зоне или касается её
                    # Расширяем зону на небольшой процент для более гибкого входа
                    zone_expansion = (fvg.upper - fvg.lower) * 0.15  # 15% расширение зоны (увеличено)
                    zone_lower = fvg.lower - zone_expansion
                    zone_upper = fvg.upper + zone_expansion
                    
                    # Проверяем, находится ли цена в зоне FVG
                    if zone_lower <= current_price <= zone_upper:
                        # Stop Loss за экстремум свечи, создавшей FVG
                        if fvg.bar_index > 0:
                            sl_price = df['low'].iloc[fvg.bar_index - 1]
                        else:
                            atr_mult = getattr(self.params, 'ict_atr_multiplier_sl', 2.0)
                            sl_price = fvg.lower - current_atr * atr_mult
                        
                        # Проверяем, что SL не выше цены входа
                        if sl_price >= current_price:
                            atr_mult = getattr(self.params, 'ict_atr_multiplier_sl', 2.0)
                            sl_price = current_price - current_atr * atr_mult
                        
                        # ВАЛИДАЦИЯ SL: Проверяем, что SL соответствует требованиям (7-10% от маржи)
                        # Предполагаем leverage = 10x (можно получить из params если доступно)
                        leverage = getattr(self.params, 'leverage', 10)
                        min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
                        max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
                        min_sl_pct_from_price = min_sl_pct_from_margin / leverage  # 0.7% от цены при 10x
                        max_sl_pct_from_price = max_sl_pct_from_margin / leverage  # 1.0% от цены при 10x
                        
                        sl_distance_pct = (current_price - sl_price) / current_price
                        
                        # Если SL слишком близко (< 7% от маржи), используем минимальный SL
                        if sl_distance_pct < min_sl_pct_from_price:
                            sl_price = current_price * (1 - min_sl_pct_from_price)
                            sl_distance_pct = min_sl_pct_from_price
                        # Если SL слишком далеко (> 10% от маржи), используем максимальный SL
                        elif sl_distance_pct > max_sl_pct_from_price:
                            sl_price = current_price * (1 - max_sl_pct_from_price)
                            sl_distance_pct = max_sl_pct_from_price
                        
                        # Take Profit: используем улучшенный R:R (3.0 вместо 2.0 для лучших результатов)
                        risk = current_price - sl_price
                        if risk > 0:
                            # Увеличиваем R:R до 3.0 для ICT стратегии (лучше для FVG ретестов)
                            rr_ratio = getattr(self.params, 'ict_rr_ratio', 3.0)
                            if rr_ratio < 3.0:
                                rr_ratio = 3.0  # Минимум 3.0 для ICT
                            tp_price = current_price + risk * rr_ratio

                            # Premium/Discount фильтр: входим в LONG только в discount‑зоне
                            if liq is not None:
                                rng_high = liq.daily_high
                                rng_low = liq.daily_low
                                if rng_high > rng_low:
                                    rel = (current_price - rng_low) / (rng_high - rng_low)
                                    if rel > 0.5:  # выше 0.5 (premium) — пропускаем LONG
                                        continue
                            
                            # ФИЛЬТР КАЧЕСТВА 3: Проверяем минимальный риск (0.7% от цены = 7% от маржи)
                            if risk < min_sl_pct_from_price * current_price:
                                continue  # Слишком маленький риск - пропускаем
                            
                            signals.append(Signal(
                                timestamp=current_ts,
                                action=Action.LONG,
                                reason=f"ict_silver_bullet_long_fvg_reteest_sl_{sl_price:.2f}_tp_{tp_price:.2f}",
                                price=current_price
                            ))
                
                elif fvg.direction == "bearish":
                    # Медвежий FVG: вход если цена в зоне или касается её
                    # Расширяем зону на небольшой процент для более гибкого входа
                    zone_expansion = (fvg.upper - fvg.lower) * 0.15  # 15% расширение зоны (увеличено)
                    zone_lower = fvg.lower - zone_expansion
                    zone_upper = fvg.upper + zone_expansion
                    
                    # Проверяем, находится ли цена в зоне FVG
                    if zone_lower <= current_price <= zone_upper:
                        # Stop Loss за экстремум свечи, создавшей FVG
                        if fvg.bar_index > 0:
                            sl_price = df['high'].iloc[fvg.bar_index - 1]
                        else:
                            atr_mult = getattr(self.params, 'ict_atr_multiplier_sl', 2.0)
                            sl_price = fvg.upper + current_atr * atr_mult
                        
                        # Проверяем, что SL не ниже цены входа
                        if sl_price <= current_price:
                            atr_mult = getattr(self.params, 'ict_atr_multiplier_sl', 2.0)
                            sl_price = current_price + current_atr * atr_mult
                        
                        # ВАЛИДАЦИЯ SL: Проверяем, что SL соответствует требованиям (7-10% от маржи)
                        # Предполагаем leverage = 10x (можно получить из params если доступно)
                        leverage = getattr(self.params, 'leverage', 10)
                        min_sl_pct_from_margin = 0.07  # Минимум 7% от маржи
                        max_sl_pct_from_margin = 0.10   # Максимум 10% от маржи
                        min_sl_pct_from_price = min_sl_pct_from_margin / leverage  # 0.7% от цены при 10x
                        max_sl_pct_from_price = max_sl_pct_from_margin / leverage  # 1.0% от цены при 10x
                        
                        sl_distance_pct = (sl_price - current_price) / current_price
                        
                        # Если SL слишком близко (< 7% от маржи), используем минимальный SL
                        if sl_distance_pct < min_sl_pct_from_price:
                            sl_price = current_price * (1 + min_sl_pct_from_price)
                            sl_distance_pct = min_sl_pct_from_price
                        # Если SL слишком далеко (> 10% от маржи), используем максимальный SL
                        elif sl_distance_pct > max_sl_pct_from_price:
                            sl_price = current_price * (1 + max_sl_pct_from_price)
                            sl_distance_pct = max_sl_pct_from_price
                        
                        # Take Profit: используем улучшенный R:R (3.0 вместо 2.0 для лучших результатов)
                        risk = sl_price - current_price
                        if risk > 0:
                            # Увеличиваем R:R до 3.0 для ICT стратегии (лучше для FVG ретестов)
                            rr_ratio = getattr(self.params, 'ict_rr_ratio', 3.0)
                            if rr_ratio < 3.0:
                                rr_ratio = 3.0  # Минимум 3.0 для ICT
                            tp_price = current_price - risk * rr_ratio

                            # Premium/Discount фильтр: входим в SHORT только в premium‑зоне
                            if liq is not None:
                                rng_high = liq.daily_high
                                rng_low = liq.daily_low
                                if rng_high > rng_low:
                                    rel = (current_price - rng_low) / (rng_high - rng_low)
                                    if rel < 0.5:  # ниже 0.5 (discount) — пропускаем SHORT
                                        continue
                            
                            # ФИЛЬТР КАЧЕСТВА 3: Проверяем минимальный риск (0.7% от цены = 7% от маржи)
                            if risk < min_sl_pct_from_price * current_price:
                                continue  # Слишком маленький риск - пропускаем
                            
                            signals.append(Signal(
                                timestamp=current_ts,
                                action=Action.SHORT,
                                reason=f"ict_silver_bullet_short_fvg_reteest_sl_{sl_price:.2f}_tp_{tp_price:.2f}",
                                price=current_price
                            ))
        
        # Убрали избыточное логирование статистики - слишком много сообщений
        # Диагностика: выводим статистику
        # print(f"[ICT] 📊 Статистика обработки:")
        # print(f"   - Свечей в торговых сессиях: {candles_in_session}")
        # print(f"   - Свечей с раскрытым аллигатором: {candles_with_trend}")
        # print(f"   - Проверок ретеста FVG: {fvg_retests_checked}")
        # print(f"   - Сгенерировано сигналов: {len(signals)}")
        
        return signals


def build_ict_signals(
    df: pd.DataFrame,
    params,
    symbol: str = "Unknown"
) -> List[Signal]:
    """
    Строит сигналы ICT Silver Bullet для всего DataFrame.
    
    Args:
        df: DataFrame с данными (должен содержать OHLCV)
        params: Параметры стратегии
        symbol: Торговая пара
        
    Returns:
        Список Signal объектов
    """
    strategy = ICTStrategy(params)
    return strategy.get_signals(df, symbol)
