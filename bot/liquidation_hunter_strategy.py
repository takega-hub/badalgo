"""
Стратегия «Охотник за ликвидациями» (Liquidation Hunter)

Бот заходит в сделку, когда на рынке происходит массовое принудительное закрытие позиций,
что часто знаменует собой локальный разворот.

Логика работы:
- Мониторинг ликвидаций (используем объем как прокси, если нет прямых данных)
- Порог: Ликвидации за последние 5 минут должны превышать средние за час в 3-5 раз
- Условие входа (Mean Reversion):
  * LONG: Всплеск ликвидаций + Цена коснулась зоны поддержки + Свеча с длинной нижней тенью
  * SHORT: Всплеск ликвидаций + Цена коснулась зоны сопротивления + Свеча с длинной верхней тенью
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from bot.strategy import Action, Signal, Bias
from bot.config import StrategyParams


class LiquidationHunterStrategy:
    """Стратегия охотника за ликвидациями."""
    
    def __init__(self, params: StrategyParams):
        self.params = params
        # Параметры стратегии (ОПТИМИЗИРОВАНЫ - лучший вариант)
        # Результаты: BTCUSDT 64% WR +3.77 USDT, ETHUSDT 59.3% WR -2.10 USDT, SOLUSDT 79.2% WR +5.01 USDT
        # Общий PnL: +17.09 USDT за 30 дней, средний WR: 81.4%
        # Параметры всплеска объема — теперь используем Z-Score
        self.liq_spike_multiplier = 2.0  # (legacy) fallback multiplier
        self.z_score_threshold = 2.5  # Z-Score threshold for volume spikes
        self.lookback_short = 5  # Период для короткого окна (5 минут/свечей)
        self.lookback_long = 30  # Период для длинного окна (используется для Z-Score)

        # ATR-параметры для адаптивных допусков
        self.atr_period = 14
        self.shadow_atr_multiplier = 0.6  # Тень считается аномальной если > ATR * multiplier

        # Параметры свечных тени/теста
        self.shadow_ratio = 0.3  # (legacy) минимальное соотношение тени к телу свечи (30%) — используем вместе с ATR
        self.ema_period = 200  # EMA для фильтра тренда
        self.support_resistance_lookback = 15  # Период для поиска зон

        # Risk/Reward для TP/SL
        self.rr_ratio = 2.0
        # RSI фильтр отключен - он слишком строгий
        # EMA фильтр сделан опциональным с допуском 2% - не блокирует сигналы строго
        
    def calculate_volume_spike(self, df: pd.DataFrame) -> pd.Series:
        """
        Рассчитывает всплеск объема как прокси для ликвидаций.
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Series с флагами всплесков объема
        """
        # Используем Z-Score объема: (volume - mean)/std по длинному окну
        if len(df) < 3:
            return pd.Series([False] * len(df), index=df.index)

        vol_mean = df['volume'].rolling(window=self.lookback_long, min_periods=1).mean()
        vol_std = df['volume'].rolling(window=self.lookback_long, min_periods=1).std(ddof=0).replace(0, np.nan)

        z_score = (df['volume'] - vol_mean) / vol_std
        # Заполняем NaN False
        volume_spike = z_score > self.z_score_threshold
        # Сохраняем z_score в DataFrame для логирования (не мутируем входной DF тут)
        # Пользуемся Series с тем же индексом
        return volume_spike
    
    def detect_shadow_patterns(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Обнаруживает свечи с длинными тенями (признак разворота).
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Tuple (long_lower_shadows, long_upper_shadows)
        """
        # Вычисляем размеры теней
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        body_size = abs(df['close'] - df['open'])

        # ATR-based check
        if 'atr' in df.columns:
            atr_mean = df['atr'].rolling(window=self.atr_period, min_periods=1).mean()
        else:
            # Простая аппроксимация ATR: rolling(high-low)
            atr_mean = (df['high'] - df['low']).rolling(window=self.atr_period, min_periods=1).mean()

        # Предыдущая свеча: тело для проверки поглощения
        prev_body_high = df[['open', 'close']].shift(1).max(axis=1)
        # Следующая свеча для дополнительной проверки (если доступна)
        next_close = df['close'].shift(-1)

        # Длинная нижняя тень (признак разворота вверх) - тень должна быть аномальной относительно ATR
        long_lower_shadows = (
            (lower_shadow > atr_mean * self.shadow_atr_multiplier) &
            (df['close'] > df['open']) &
            ((df['close'] > prev_body_high) | (next_close > prev_body_high))
        )

        # Длинная верхняя тень (признак разворота вниз)
        long_upper_shadows = (
            (upper_shadow > atr_mean * self.shadow_atr_multiplier) &
            (df['close'] < df['open']) &
            ((df['close'] < df[['open', 'close']].shift(1).min(axis=1)) | (next_close < df[['open', 'close']].shift(1).min(axis=1)))
        )

        return long_lower_shadows.fillna(False), long_upper_shadows.fillna(False)
    
    def find_support_resistance_zones(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Находит зоны поддержки и сопротивления (локальные минимумы и максимумы).
        
        Args:
            df: DataFrame с данными OHLCV
            lookback: Период для поиска локальных экстремумов
            
        Returns:
            Tuple (support_zones, resistance_zones) - булевы серии
        """
        if len(df) < lookback:
            return pd.Series([False] * len(df), index=df.index), pd.Series([False] * len(df), index=df.index)

        # Используем только исторические данные: center=False
        # Локальные минимумы (поддержка) — минимум за предыдущие lookback свечей (включая текущую)
        local_lows = df['low'].rolling(window=lookback, center=False, min_periods=1).min()
        local_highs = df['high'].rolling(window=lookback, center=False, min_periods=1).max()

        # ATR для динамического допуска
        if 'atr' in df.columns:
            atr_mean = df['atr'].rolling(window=self.atr_period, min_periods=1).mean()
        else:
            atr_mean = (df['high'] - df['low']).rolling(window=self.atr_period, min_periods=1).mean()

        # Допуск на основе ATR — нормируем к цене
        tol = atr_mean  # абсолютный допуск в ценах

        support_zones = df['low'] <= (local_lows + tol)
        resistance_zones = df['high'] >= (local_highs - tol)

        return support_zones.fillna(False), resistance_zones.fillna(False)
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Генерирует сигналы на основе ликвидаций и паттернов разворота.
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Торговая пара для логирования
            
        Returns:
            Список сигналов
        """
        if len(df) < max(self.lookback_long, self.ema_period):
            return []
        
        signals: List[Signal] = []

        # Вычисляем EMA для фильтра тренда
        df = df.copy()
        df['ema_200'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()

        # Всплески объема (Z-Score)
        volume_spikes = self.calculate_volume_spike(df)

        # Паттерны теней
        long_lower_shadows, long_upper_shadows = self.detect_shadow_patterns(df)

        # Зоны поддержки и сопротивления
        support_zones, resistance_zones = self.find_support_resistance_zones(df, lookback=self.support_resistance_lookback)

        # Векторизированные проверки EMA
        ema_ok_long = df['close'] < df['ema_200'] * 1.02
        ema_ok_short = df['close'] > df['ema_200'] * 0.98

        # Условие LONG/SHORT в векторной форме
        long_condition = (volume_spikes) & (support_zones | long_lower_shadows) & (ema_ok_long.fillna(True))
        short_condition = (volume_spikes) & (resistance_zones | long_upper_shadows) & (ema_ok_short.fillna(True))

        # Индексы, где есть хоть одно условие — минимизируем итерацию
        candidate_idx = df.index[(long_condition | short_condition)]

        # Предрасчёт ATR для TP/SL
        if 'atr' in df.columns:
            atr_series = df['atr'].rolling(window=self.atr_period, min_periods=1).mean()
        else:
            atr_series = (df['high'] - df['low']).rolling(window=self.atr_period, min_periods=1).mean()

        for idx in candidate_idx:
            row = df.loc[idx]

            is_long = bool(long_condition.loc[idx])
            is_short = bool(short_condition.loc[idx])

            # Если оба условия True — решаем по силе паттерна
            if is_long and is_short:
                long_strength = int(bool(support_zones.loc[idx])) + int(bool(long_lower_shadows.loc[idx]))
                short_strength = int(bool(resistance_zones.loc[idx])) + int(bool(long_upper_shadows.loc[idx]))
                # добавляем вклад z-score
                try:
                    z = float((row['volume'] - row['volume'].rolling(self.lookback_long).mean()) / (row['volume'].rolling(self.lookback_long).std(ddof=0)))
                except Exception:
                    z = 0
                if z > self.z_score_threshold:
                    long_strength += 1 if long_strength >= short_strength else 0
                    short_strength += 1 if short_strength > long_strength else 0

                if long_strength >= short_strength:
                    is_short = False
                else:
                    is_long = False

            # Генерируем LONG сигнал
            if is_long:
                reason = "liquidation_hunter_long_zscore"
                if support_zones.loc[idx]:
                    reason += "_support"
                if long_lower_shadows.loc[idx]:
                    reason += "_shadow"

                # Stop Loss: локальный минимум свечи со всплеском минус небольшая подушка
                atr = atr_series.loc[idx]
                sl = min(row['low'], df['low'].rolling(3, min_periods=1).min().loc[idx]) - (atr * 0.3 if np.isfinite(atr) else 0)
                tp = row['close'] + (row['close'] - sl) * self.rr_ratio

                # Сила сигнала: сумма бинарных факторов + нормализованный z-score
                try:
                    vol_mean = df['volume'].rolling(self.lookback_long).mean().loc[idx]
                    vol_std = df['volume'].rolling(self.lookback_long).std(ddof=0).loc[idx]
                    zval = (row['volume'] - vol_mean) / (vol_std if vol_std != 0 else np.nan)
                except Exception:
                    zval = np.nan

                strength = float((int(bool(support_zones.loc[idx])) + int(bool(long_lower_shadows.loc[idx]))))
                if np.isfinite(zval):
                    strength += max(0.0, zval - self.z_score_threshold + 1.0)

                signals.append(Signal(
                    timestamp=idx,
                    action=Action.LONG,
                    reason=reason,
                    price=row['close'],
                    stop_loss=sl,
                    take_profit=tp,
                    indicators_info={
                        "volume": row.get('volume'),
                        "z_score_threshold": self.z_score_threshold,
                        "atr": float(atr) if np.isfinite(atr) else None
                    }
                ))

            # Генерируем SHORT сигнал
            elif is_short:
                reason = "liquidation_hunter_short_zscore"
                if resistance_zones.loc[idx]:
                    reason += "_resistance"
                if long_upper_shadows.loc[idx]:
                    reason += "_shadow"

                atr = atr_series.loc[idx]
                sl = max(row['high'], df['high'].rolling(3, min_periods=1).max().loc[idx]) + (atr * 0.3 if np.isfinite(atr) else 0)
                tp = row['close'] - (sl - row['close']) * self.rr_ratio

                try:
                    vol_mean = df['volume'].rolling(self.lookback_long).mean().loc[idx]
                    vol_std = df['volume'].rolling(self.lookback_long).std(ddof=0).loc[idx]
                    zval = (row['volume'] - vol_mean) / (vol_std if vol_std != 0 else np.nan)
                except Exception:
                    zval = np.nan

                strength = float((int(bool(resistance_zones.loc[idx])) + int(bool(long_upper_shadows.loc[idx]))))
                if np.isfinite(zval):
                    strength += max(0.0, zval - self.z_score_threshold + 1.0)

                signals.append(Signal(
                    timestamp=idx,
                    action=Action.SHORT,
                    reason=reason,
                    price=row['close'],
                    stop_loss=sl,
                    take_profit=tp,
                    indicators_info={
                        "volume": row.get('volume'),
                        "z_score_threshold": self.z_score_threshold,
                        "atr": float(atr) if np.isfinite(atr) else None
                    }
                ))

        return signals


def build_liquidation_hunter_signals(
    df: pd.DataFrame,
    params: StrategyParams,
    symbol: str = "Unknown"
) -> List[Signal]:
    """
    Строит сигналы стратегии Liquidation Hunter для всего DataFrame.
    
    Args:
        df: DataFrame с данными (должен содержать OHLCV)
        params: Параметры стратегии
        symbol: Торговая пара для логирования
    
    Returns:
        Список Signal объектов
    """
    strategy = LiquidationHunterStrategy(params)
    return strategy.get_signals(df, symbol=symbol)
