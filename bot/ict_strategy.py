"""
ICT (Inner Circle Trader) стратегия "Silver Bullet" для торгового бота.

Стратегия работает в определенные временные окна (сессии) и ищет:
1. Снятие ликвидности за пределами дневного максимума/минимума
2. Резкий возврат в диапазон (манипуляция)
3. Формирование FVG в сторону нового движения
4. Вход на ретест FVG

Особенности:
- Тайм-фильтр для Лондонской и Нью-Йоркской сессий
- Williams Alligator для фильтрации тренда (опционально)
- ATR для динамического стоп-лосса
- Автоматический перевод в безубыток при R:R 1:1
"""
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, time, timezone, timedelta
import numpy as np
import pandas as pd
import pytz
import logging
import os

from bot.strategy import Action, Signal, Bias
import json
from bot import logger_config

# Настройка системного логгера
logger = logging.getLogger("ict_strategy")


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
    is_institutional: bool = False


class ICTStrategy:
    """Класс стратегии ICT Silver Bullet."""
    
    def __init__(self, params):
        self.params = params
        
        # Таймзона Нью-Йорка для Silver Bullet окон (ICT ориентируется на NY Local Time)
        self.ny_tz = pytz.timezone("America/New_York")
        # Silver Bullet окна (NY local time, ET)
        self.sb_windows_ny = [
            (time(3, 0), time(4, 0)),   # London SB
            (time(10, 0), time(11, 0)), # NY AM SB
            (time(14, 0), time(15, 0)), # NY PM SB
        ]

        # Параметры из params (с fallback на значения по умолчанию)
        self.zone_expansion_mult = getattr(self.params, 'ict_fvg_zone_expansion_mult', 0.15)
        self.fvg_tolerance = getattr(self.params, 'ict_fvg_tolerance', 0.005)
        self.mss_close_required = getattr(self.params, 'ict_mss_close_required', True)
        self.ict_max_slippage_pct = getattr(self.params, 'ict_max_slippage_pct', 0.001)
        self.ict_time_drift_sec = getattr(self.params, 'ict_max_time_drift_sec', 5)
        self.mtf_bias_tf = getattr(self.params, 'ict_mtf_bias_timeframe', '4H')
        
        # V10: Состояние для защиты от тильта
        self.session_results = {} # {session_id: is_win (bool)}

        # Данные для визуализации
        self.viz_data = {
            "fvg": [],
            "liquidity": [],
            "mss": [],
            "signals": []
        }
    
    def get_previous_session_id(self, session_id: str) -> Optional[str]:
        """Вычисляет ID предыдущей торговой сессии для любого из 3-х окон."""
        if not session_id: return None
        try:
            date_str, s_type = session_id.rsplit('_', 1)
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            
            if s_type == "NY PM":
                return f"{date_str}_NY AM"
            elif s_type == "NY AM":
                return f"{date_str}_London"
            else: # London -> NY PM предыдущего дня
                prev_date = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                return f"{prev_date}_NY PM"
        except: return None
    
    def is_trading_session(self, timestamp: pd.Timestamp) -> bool:
        """
        Проверяет, находится ли время в активном Silver Bullet окне (NY local time).
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
        
        for start, end in self.sb_windows_ny:
            if start <= current_time_ny <= end:
                return True
        return False
    
    def get_session_id(self, ts: pd.Timestamp) -> Optional[str]:
        """Возвращает уникальный ID сессии (дата + тип окна)."""
        if not ts or pd.isna(ts): return None
        ts_utc = ts.tz_localize('UTC') if ts.tzinfo is None else ts.astimezone(pytz.UTC)
        ts_ny = ts_utc.astimezone(self.ny_tz)
        current_time_ny = ts_ny.time()
        date_str = ts_ny.strftime('%Y-%m-%d')
        
        for i, (start, end) in enumerate(self.sb_windows_ny):
            if start <= current_time_ny <= end:
                if i == 0: s_type = "London"
                elif i == 1: s_type = "NY AM"
                else: s_type = "NY PM"
                return f"{date_str}_{s_type}"
        return None

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
        """Рассчитывает индикатор Williams Alligator."""
        median_price = (df['high'] + df['low']) / 2
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
        """Проверяет, раскрыт ли аллигатор."""
        if index < 0 or index >= len(jaw):
            return False, None
        
        jaw_val = jaw.iloc[index]
        teeth_val = teeth.iloc[index]
        lips_val = lips.iloc[index]
        
        if not all(pd.notna([jaw_val, teeth_val, lips_val])):
            return False, None

        if lips_val > teeth_val > jaw_val:
            return True, "bullish"
        if lips_val < teeth_val < jaw_val:
            return True, "bearish"
        
        if index > 0:
            prev_lips = lips.iloc[index-1]
            if lips_val > prev_lips and lips_val > teeth_val:
                return True, "bullish"
            if lips_val < prev_lips and lips_val < teeth_val:
                return True, "bearish"
        
        return False, None

    def get_higher_tf_bias(self, df: pd.DataFrame, timeframe: Optional[str] = '1h', end_idx: Optional[int] = None) -> Optional[str]:
        """Определяет глобальный тренд через положение цены относительно EMA 50 на 1H."""
        if timeframe is None: return None
        try:
            df_curr = df.iloc[:end_idx+1] if end_idx is not None else df
            if len(df_curr) < 200: return None # Нужно больше данных для 1H EMA 50
            
            # Ресемплинг в 1h (исправлено)
            df_1h = df_curr.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
            if len(df_1h) < 50: return None
            
            ema50 = df_1h['close'].ewm(span=50, adjust=False).mean()
            last_close = df_1h['close'].iloc[-1]
            last_ema = ema50.iloc[-1]
            
            if last_close > last_ema: return 'bullish'
            if last_close < last_ema: return 'bearish'
        except: pass
        return None

    def find_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        lookback_days: int = 1
    ) -> List[ICTLiquidity]:
        """Находит снятия ликвидности за пределами PDH/PDL и сессионных экстремумов."""
        if len(df) < 100: return []
        
        liquidity_zones: List[ICTLiquidity] = []
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)
        
        if df_copy.index.tzinfo is None:
            df_copy.index = df_copy.index.tz_localize('UTC')
        else:
            df_copy.index = df_copy.index.tz_convert('UTC')
        
        ts_ny = df_copy.index.tz_convert(self.ny_tz)
        df_copy["date_ny"] = ts_ny.date
        
        daily_groups = df_copy.groupby("date_ny")
        daily_high_low = {date: (group["high"].max(), group["low"].min()) for date, group in daily_groups}
        
        def _session_for_time(t: time) -> str:
            if 0 <= t.hour < 8: return "asia"
            if 8 <= t.hour < 13: return "london"
            if 13 <= t.hour < 21: return "ny"
            return "off"
        
        df_copy["session"] = [ _session_for_time(t.time()) for t in ts_ny ]
        df_copy["session_key"] = list(zip(df_copy["date_ny"], df_copy["session"]))
        
        session_groups = df_copy[df_copy["session"] != "off"].groupby("session_key")
        session_high_low = {key: (group["high"].max(), group["low"].min()) for key, group in session_groups}

        for i in range(1, len(df_copy)):
            row = df_copy.iloc[i]
            prev_date = row["date_ny"] - pd.Timedelta(days=1)
            pdh, pdl = daily_high_low.get(prev_date, (None, None))
            
            curr_sess_key = row["session_key"]
            all_keys = list(session_high_low.keys())
            try:
                curr_idx = all_keys.index(curr_sess_key)
                prev_sess_key = all_keys[curr_idx-1] if curr_idx > 0 else None
            except ValueError:
                prev_sess_key = None
            
            psh, psl = session_high_low.get(prev_sess_key, (None, None))
            
            levels_high = [v for v in [pdh, psh] if v is not None]
            levels_low = [v for v in [pdl, psl] if v is not None]
            target_high = max(levels_high) if levels_high else None
            target_low = min(levels_low) if levels_low else None
            
            if target_high and row["high"] > target_high:
                liquidity_zones.append(ICTLiquidity(
                    bar_index=i, timestamp=df_copy.index[i], price=row["high"],
                    direction="above_high", daily_high=target_high, daily_low=target_low or 0, is_institutional=True
                ))
            if target_low and row["low"] < target_low:
                liquidity_zones.append(ICTLiquidity(
                    bar_index=i, timestamp=df_copy.index[i], price=row["low"],
                    direction="below_low", daily_high=target_high or 0, daily_low=target_low, is_institutional=True
                ))
        return liquidity_zones
    
    def find_fvg(self, df: pd.DataFrame, liquidity_sweeps: List[ICTLiquidity]) -> List[ICTFVG]:
        """Находит Fair Value Gaps после снятия ликвидности с Displacement и умеренным объемом."""
        if len(df) < 20: return []
        highs, lows = df['high'].values, df['low'].values
        volumes = df['volume'].values
        
        candle_bodies = np.abs(df['close'].values - df['open'].values)
        avg_body = pd.Series(candle_bodies).rolling(20).mean().values
        avg_volume = pd.Series(volumes).rolling(20).mean().values
        
        # Фильтр: Возвращаем объем 1.1x для отсечения пустого шума
        bullish_mask = (lows[2:] > highs[:-2]) & \
                       (candle_bodies[1:-1] > avg_body[1:-1] * 1.3) & \
                       (volumes[1:-1] > avg_volume[1:-1] * 1.1)
                       
        bearish_mask = (highs[2:] < lows[:-2]) & \
                       (candle_bodies[1:-1] > avg_body[1:-1] * 1.3) & \
                       (volumes[1:-1] > avg_volume[1:-1] * 1.1)
        
        all_fvg = []
        bull_indices = np.where(bullish_mask)[0] + 2
        bear_indices = np.where(bearish_mask)[0] + 2
        
        liq_map = {liq.bar_index: liq for liq in liquidity_sweeps}
        liq_indices = np.array(list(liq_map.keys()))
        search_window = getattr(self.params, 'ict_fvg_search_window', 40)
        
        for i in bull_indices:
            prev_liqs = liq_indices[(liq_indices < i) & (liq_indices >= i - search_window)]
            for liq_idx in prev_liqs:
                if liq_map[liq_idx].direction == "below_low":
                    all_fvg.append(ICTFVG(int(i), df.index[i], float(lows[i]), float(highs[i-2]), "bullish", int(liq_idx)))
                    break
        for i in bear_indices:
            prev_liqs = liq_indices[(liq_indices < i) & (liq_indices >= i - search_window)]
            for liq_idx in prev_liqs:
                if liq_map[liq_idx].direction == "above_high":
                    all_fvg.append(ICTFVG(int(i), df.index[i], float(lows[i-2]), float(highs[i]), "bearish", int(liq_idx)))
                    break
        return all_fvg
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """Основной метод получения сигналов ICT Silver Bullet с лимитом на сессию."""
        if len(df) < 200: return []
        signals = []
        
        if 'atr' not in df.columns:
            df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        # Добавляем локальную EMA 20 для подтверждения краткосрочного тренда
        ema20_15m = df['close'].ewm(span=20, adjust=False).mean()
        
        liquidity_sweeps = self.find_liquidity_sweeps(df)
        if not liquidity_sweeps:
            liquidity_sweeps = self.find_liquidity_sweeps_alternative(df, 50)
        
        fvg_zones = self.find_fvg(df, liquidity_sweeps)
        liq_by_bar = {liq.bar_index: liq for liq in liquidity_sweeps}
        
        processed_fvg_ids = set()
        last_trade_session = None # Для ограничения 1 сделка на окно
        
        for i in range(200, len(df)):
            current_row = df.iloc[i]
            current_ts = df.index[i]
            current_price = current_row['close']
            current_low, current_high = current_row['low'], current_row['high']
            
            # Определение текущей сессии для лимита
            session_id = self.get_session_id(current_ts)
            if not session_id: continue
            
            # V10: Пропуск сессии после убытка (cooldown)
            prev_session_id = self.get_previous_session_id(session_id)
            if self.session_results.get(prev_session_id) == False: # False означает Loss
                continue

            # Митигация FVG
            for fvg in fvg_zones:
                if not fvg.active: continue
                if i > fvg.bar_index:
                    if fvg.direction == "bullish" and current_row['close'] < fvg.lower: fvg.active = False
                    elif fvg.direction == "bearish" and current_row['close'] > fvg.upper: fvg.active = False
            
            # ДИНАМИЧЕСКИЙ БАЙАС
            current_mtf_bias = None
            bias_calculated = False

            for fvg in fvg_zones:
                if not fvg.active or fvg.bar_index >= i or (i - fvg.bar_index) > 20: continue # V8: 20 свечей
                
                # Если в этой сессии уже был сигнал - пропускаем
                if last_trade_session == session_id: continue
                
                if not bias_calculated:
                    current_mtf_bias = self.get_higher_tf_bias(df, self.mtf_bias_tf, end_idx=i)
                    bias_calculated = True

                # Filters
                if current_mtf_bias:
                    # V8: Двойной фильтр (MTF Bias + Local EMA 20)
                    if fvg.direction == "bullish":
                        if current_mtf_bias != "bullish" or current_price < ema20_15m.iloc[i]: continue
                    if fvg.direction == "bearish":
                        if current_mtf_bias != "bearish" or current_price > ema20_15m.iloc[i]: continue

                # MSS & Dealing Range
                liq = liq_by_bar.get(fvg.liquidity_bar_index)
                dealing_range = None
                if liq:
                    prior_start = max(0, liq.bar_index - 20)
                    mid_slice = df.iloc[liq.bar_index + 1 : fvg.bar_index + 2]
                    if not mid_slice.empty:
                        if fvg.direction == "bullish":
                            swing_high = df.iloc[prior_start:liq.bar_index]["high"].max()
                            mss_mask = mid_slice["high"] > swing_high
                            if mss_mask.any():
                                abs_mss_idx = liq.bar_index + 1 + np.where(mss_mask.values)[0][0]
                                high_p = df.iloc[liq.bar_index : abs_mss_idx + 1]["high"].max()
                                dealing_range = {"low": liq.price, "high": high_p, "mid": (liq.price + high_p) / 2}
                        else:
                            swing_low = df.iloc[prior_start:liq.bar_index]["low"].min()
                            mss_mask = mid_slice["low"] < swing_low
                            if mss_mask.any():
                                abs_mss_idx = liq.bar_index + 1 + np.where(mss_mask.values)[0][0]
                                low_p = df.iloc[liq.bar_index : abs_mss_idx + 1]["low"].min()
                                dealing_range = {"low": low_p, "high": liq.price, "mid": (liq.price + low_p) / 2}

                # Filters
                if current_mtf_bias:
                    if fvg.direction == "bullish" and current_mtf_bias == "bearish": continue
                    if fvg.direction == "bearish" and current_mtf_bias == "bullish": continue

                if dealing_range:
                    # Возвращаем строгий 50% P/D фильтр (V6 Precision)
                    l_thr = dealing_range["low"] + (dealing_range["high"] - dealing_range["low"]) * 0.50
                    s_thr = dealing_range["low"] + (dealing_range["high"] - dealing_range["low"]) * 0.50
                    if fvg.direction == "bullish" and current_price > l_thr: continue
                    if fvg.direction == "bearish" and current_price < s_thr: continue
                
                fvg_id = (fvg.bar_index, fvg.direction, round(fvg.lower, 2), round(fvg.upper, 2))
                if fvg_id in processed_fvg_ids: continue
                
                # Retest & Age Filter (V8: входим только если FVG "свежий" - до 20 свечей)
                if (i - fvg.bar_index) > 20: continue
                
                z_exp = (fvg.upper - fvg.lower) * 0.2
                atr = df['atr'].iloc[i]
                
                if fvg.direction == "bullish":
                    if current_low <= fvg.upper + z_exp and current_price >= fvg.lower - z_exp:
                        processed_fvg_ids.add(fvg_id)
                        last_trade_session = session_id 
                        sl = liq.price - (atr * 0.5) if liq else current_price * 0.99
                        
                        # V9: Ограничение максимального риска (1.5%)
                        if abs(current_price - sl) > current_price * 0.015:
                            sl = current_price - (current_price * 0.015)

                        # Расчет TP на основе RR
                        risk = abs(current_price - sl)
                        tp = current_price + (risk * self.params.ict_rr_ratio)
                        
                        signals.append(Signal(
                            timestamp=current_ts, 
                            action=Action.LONG, 
                            reason=f"ict_sb_long_{fvg.bar_index}_pd", 
                            price=current_price, 
                            stop_loss=sl, 
                            take_profit=tp
                        ))
                else:
                    if current_high >= fvg.lower - z_exp and current_price <= fvg.upper + z_exp:
                        processed_fvg_ids.add(fvg_id)
                        last_trade_session = session_id 
                        sl = liq.price + (atr * 0.5) if liq else current_price * 1.01

                        # V9: Ограничение максимального риска (1.5%)
                        if abs(current_price - sl) > current_price * 0.015:
                            sl = current_price + (current_price * 0.015)
                        
                        # Расчет TP на основе RR
                        risk = abs(current_price - sl)
                        tp = current_price - (risk * self.params.ict_rr_ratio)
                        
                        signals.append(Signal(
                            timestamp=current_ts, 
                            action=Action.SHORT, 
                            reason=f"ict_sb_short_{fvg.bar_index}_pd", 
                            price=current_price, 
                            stop_loss=sl, 
                            take_profit=tp
                        ))

        self.export_viz_data(symbol)
        return signals

    def export_viz_data(self, symbol: str):
        try:
            filename = f"dumps/ict_viz_{symbol}.json"
            data_to_save = {
                "fvg": [asdict(f) for f in self.viz_data["fvg"]],
                "liquidity": [asdict(l) for l in self.viz_data["liquidity"]],
                "mss": self.viz_data["mss"],
                "signals": [asdict(s) for s in self.viz_data["signals"]]
            }
            def _default(obj):
                if isinstance(obj, pd.Timestamp): return obj.isoformat()
                if isinstance(obj, (Action, Bias)): return obj.name
                return str(obj)
            os.makedirs("dumps", exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, default=_default, indent=2)
        except: pass

    def find_liquidity_sweeps_alternative(self, df: pd.DataFrame, lookback_bars: int = 50) -> List[ICTLiquidity]:
        if len(df) < lookback_bars + 1: return []
        liquidity_zones = []
        highs, lows = df['high'].values, df['low'].values
        for i in range(lookback_bars, len(df)):
            p_h, p_l = np.max(highs[i-lookback_bars:i]), np.min(lows[i-lookback_bars:i])
            if highs[i] > p_h:
                liquidity_zones.append(ICTLiquidity(i, df.index[i], highs[i], "above_high", p_h, p_l, True))
            if lows[i] < p_l:
                liquidity_zones.append(ICTLiquidity(i, df.index[i], lows[i], "below_low", p_h, p_l, True))
        return liquidity_zones

def build_ict_signals(df: pd.DataFrame, params, symbol: str = "Unknown") -> List[Signal]:
    return ICTStrategy(params).get_signals(df, symbol)
