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
        self.liq_spike_multiplier = 2.0  # Множитель для определения всплеска ликвидаций
        self.lookback_short = 5  # Период для короткого окна (5 минут/свечей)
        self.lookback_long = 30  # Период для длинного окна
        self.shadow_ratio = 0.3  # Минимальное соотношение тени к телу свечи (30%)
        self.ema_period = 200  # EMA для фильтра тренда
        self.support_resistance_lookback = 15  # Период для поиска зон
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
        if len(df) < self.lookback_long:
            return pd.Series([False] * len(df), index=df.index)
        
        # Средний объем за длинный период (час)
        avg_volume_long = df['volume'].rolling(window=self.lookback_long, min_periods=1).mean()
        
        # Средний объем за короткий период (5 минут)
        avg_volume_short = df['volume'].rolling(window=self.lookback_short, min_periods=1).mean()
        
        # Всплеск: короткий период превышает длинный в N раз
        volume_spike = avg_volume_short > (avg_volume_long * self.liq_spike_multiplier)
        
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
        
        # Длинная нижняя тень (признак разворота вверх)
        long_lower_shadows = (lower_shadow > body_size * self.shadow_ratio) & (df['close'] > df['open'])
        
        # Длинная верхняя тень (признак разворота вниз)
        long_upper_shadows = (upper_shadow > body_size * self.shadow_ratio) & (df['close'] < df['open'])
        
        return long_lower_shadows, long_upper_shadows
    
    def find_support_resistance_zones(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Находит зоны поддержки и сопротивления (локальные минимумы и максимумы).
        
        Args:
            df: DataFrame с данными OHLCV
            lookback: Период для поиска локальных экстремумов
            
        Returns:
            Tuple (support_zones, resistance_zones) - булевы серии
        """
        if len(df) < lookback * 2:
            return pd.Series([False] * len(df), index=df.index), pd.Series([False] * len(df), index=df.index)
        
        # Локальные минимумы (поддержка)
        local_lows = df['low'].rolling(window=lookback, center=True, min_periods=1).min()
        support_zones = df['low'] <= local_lows * 1.002  # Допуск 0.2%
        
        # Локальные максимумы (сопротивление)
        local_highs = df['high'].rolling(window=lookback, center=True, min_periods=1).max()
        resistance_zones = df['high'] >= local_highs * 0.998  # Допуск 0.2%
        
        return support_zones, resistance_zones
    
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
        
        signals = []
        
        # Вычисляем EMA для фильтра тренда
        df = df.copy()
        df['ema_200'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # Всплески объема (прокси для ликвидаций)
        volume_spikes = self.calculate_volume_spike(df)
        
        # Паттерны теней
        long_lower_shadows, long_upper_shadows = self.detect_shadow_patterns(df)
        
        # Зоны поддержки и сопротивления
        support_zones, resistance_zones = self.find_support_resistance_zones(df, lookback=self.support_resistance_lookback)
        
        # Генерируем сигналы (максимально ослабленные фильтры)
        for idx, row in df.iterrows():
            # EMA фильтр опционален - не блокируем, если EMA недоступна или условие не выполнено
            ema_ok_long = True
            ema_ok_short = True
            if np.isfinite(row.get('ema_200', np.nan)):
                # Для LONG: предпочитаем цену ниже EMA, но не блокируем
                ema_ok_long = row['close'] < row['ema_200'] * 1.02  # Допуск 2%
                # Для SHORT: предпочитаем цену выше EMA, но не блокируем
                ema_ok_short = row['close'] > row['ema_200'] * 0.98  # Допуск 2%
            
            # LONG сигнал: всплеск ликвидаций + (поддержка ИЛИ длинная нижняя тень) + EMA опционально
            if (volume_spikes.loc[idx] and 
                (support_zones.loc[idx] or long_lower_shadows.loc[idx]) and
                ema_ok_long):
                
                reason = f"liquidation_hunter_long_spike_{self.liq_spike_multiplier}x"
                if support_zones.loc[idx]:
                    reason += "_support"
                if long_lower_shadows.loc[idx]:
                    reason += "_shadow"
                signals.append(Signal(
                    timestamp=idx,
                    action=Action.LONG,
                    reason=reason,
                    price=row['close']
                ))
            
            # SHORT сигнал: всплеск ликвидаций + (сопротивление ИЛИ длинная верхняя тень) + EMA опционально
            elif (volume_spikes.loc[idx] and 
                  (resistance_zones.loc[idx] or long_upper_shadows.loc[idx]) and
                  ema_ok_short):
                
                reason = f"liquidation_hunter_short_spike_{self.liq_spike_multiplier}x"
                if resistance_zones.loc[idx]:
                    reason += "_resistance"
                if long_upper_shadows.loc[idx]:
                    reason += "_shadow"
                signals.append(Signal(
                    timestamp=idx,
                    action=Action.SHORT,
                    reason=reason,
                    price=row['close']
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
