"""
Гибридная стратегия BREAKOUT_TREND_HYBRID.

Объединяет волатильный пробой VBO с трендовой фильтрацией TREND
для фильтрации ложных пробоев против тренда и повышения качества сигналов.

Концепция:
- VBO определяет пробой диапазона предыдущего дня
- TREND подтверждает направление тренда (EMA веер, ADX)
- Сигнал генерируется только если пробой в направлении тренда

Преимущества:
- Фильтрация ложных пробоев против тренда
- Работает на SOLUSDT (где обе стратегии успешны)
- Повышение качества сигналов за счет двойной фильтрации
- Использование лучших фильтров от обеих стратегий
"""
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

from bot.strategy import Action, Signal
from bot.vbo_strategy import VBOStrategy
from bot.strategy import generate_trend_signal as _generate_trend_signal

logger = logging.getLogger(__name__)


class BreakoutTrendHybridStrategy:
    """Гибридная стратегия, объединяющая VBO и TREND."""
    
    def __init__(self, params):
        self.params = params
        
        # Инициализируем VBO стратегию
        self.vbo_strategy = VBOStrategy(params)
        
        # Параметры для TREND стратегии
        self.trend_adx_threshold = getattr(params, 'trend_adx_threshold', 25.0)
        self.trend_vol_multiplier = getattr(params, 'trend_vol_multiplier', 1.3)
        
        # Параметры гибридной стратегии
        self.require_trend_confirmation = getattr(params, 'hybrid_require_trend_confirmation', True)
        self.trend_confirmation_strength = getattr(params, 'hybrid_trend_confirmation_strength', 0.7)
        # Минимальная сила тренда для подтверждения (0.0-1.0)
        # 0.7 означает, что тренд должен быть достаточно сильным
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Генерирует сигналы гибридной стратегии.
        
        Логика:
        1. Получаем сигнал от VBO стратегии (пробой диапазона)
        2. Проверяем подтверждение от TREND фильтров (направление тренда)
        3. Генерируем сигнал только если оба условия выполнены
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Торговая пара для логирования
            
        Returns:
            Список подтвержденных сигналов
        """
        if len(df) < 200:  # Минимум для обеих стратегий
            return []
        
        signals = []
        
        # 1. Получаем сигналы от VBO стратегии
        try:
            vbo_signals = self.vbo_strategy.get_signals(df, symbol)
            logger.debug(f"[BreakoutTrendHybrid] {symbol} Found {len(vbo_signals)} VBO signals")
        except Exception as e:
            logger.error(f"[BreakoutTrendHybrid] Error getting VBO signals: {e}")
            return []
        
        if not vbo_signals:
            return []
        
        # 2. Получаем состояние тренда один раз для всех сигналов (кэширование)
        cached_trend_result = None
        if self.require_trend_confirmation:
            try:
                trend_state = {'backtest_mode': False}
                cached_trend_result = _generate_trend_signal(
                    df,
                    state=trend_state,
                    sma_period=getattr(self.params, 'trend_sma_period', 21),
                    atr_period=getattr(self.params, 'trend_atr_period', 14),
                    atr_multiplier=getattr(self.params, 'trend_atr_multiplier', 3.0),
                    max_pyramid=getattr(self.params, 'trend_max_pyramid', 2),
                    min_history=100,
                    adx_threshold=self.trend_adx_threshold,
                    vol_multiplier=self.trend_vol_multiplier,
                    use_mtf_filter=getattr(self.params, 'trend_use_mtf_filter', True),
                    mtf_timeframe=getattr(self.params, 'trend_mtf_timeframe', '1h'),
                    mtf_ema_period=getattr(self.params, 'trend_mtf_ema_period', 50),
                )
            except Exception as e:
                logger.warning(f"[BreakoutTrendHybrid] Error getting trend state: {e}")
                cached_trend_result = None
        
        # 3. Проверяем подтверждение от TREND фильтров для каждого VBO сигнала
        for vbo_signal in vbo_signals:
            trend_confirmed = self._check_trend_confirmation(df, vbo_signal, symbol, cached_trend_result)
            
            if not trend_confirmed:
                logger.debug(
                    f"[BreakoutTrendHybrid] {symbol} VBO signal {vbo_signal.action.value} "
                    f"rejected by TREND filters"
                )
                continue
            
            # 4. Генерируем подтвержденный сигнал
            price = vbo_signal.price
            timestamp = vbo_signal.timestamp
            
            # Используем SL/TP от VBO или рассчитываем на основе ATR
            stop_loss = vbo_signal.stop_loss
            take_profit = vbo_signal.take_profit
            
            # Если SL/TP нет, рассчитываем на основе ATR
            if stop_loss is None or take_profit is None:
                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.005
                if stop_loss is None:
                    stop_loss = price - (3.0 * atr_val) if vbo_signal.action == Action.LONG else price + (3.0 * atr_val)
                if take_profit is None:
                    take_profit = price + (6.0 * atr_val) if vbo_signal.action == Action.LONG else price - (6.0 * atr_val)
            
            reason = f"breakout_trend_hybrid_{vbo_signal.action.value.lower()}"
            if vbo_signal.reason:
                reason += f"_{vbo_signal.reason}"
            
            # Safely get indicators_info, handling None case
            vbo_indicators = getattr(vbo_signal, 'indicators_info', None) or {}
            
            signal = Signal(
                timestamp=timestamp,
                action=vbo_signal.action,
                reason=reason,
                price=price,
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                indicators_info={
                    **vbo_indicators,
                    'hybrid_confirmed': True,
                    'trend_confirmation_strength': trend_confirmed.get('strength', 1.0),
                }
            )
            
            signals.append(signal)
            logger.info(
                f"[BreakoutTrendHybrid] {symbol} Generated confirmed signal: "
                f"{vbo_signal.action.value} @ {price:.2f} (VBO + TREND confirmation)"
            )
        
        return signals
    
    def _check_trend_confirmation(
        self, 
        df: pd.DataFrame, 
        vbo_signal: Signal, 
        symbol: str,
        cached_trend_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Проверяет подтверждение сигнала от TREND фильтров.
        
        Args:
            df: DataFrame с данными
            vbo_signal: Сигнал от VBO
            symbol: Торговая пара
            cached_trend_result: Кэшированный результат trend signal (опционально)
            
        Returns:
            Dict с ключами:
            - 'confirmed': bool - подтвержден ли сигнал
            - 'strength': float - сила подтверждения (0.0-1.0)
        """
        if not self.require_trend_confirmation:
            return {'confirmed': True, 'strength': 1.0}
        
        try:
            # Используем кэшированный результат или получаем новый
            if cached_trend_result is None:
                trend_state = {'backtest_mode': False}
                trend_result = _generate_trend_signal(
                    df,
                    state=trend_state,
                    sma_period=getattr(self.params, 'trend_sma_period', 21),
                    atr_period=getattr(self.params, 'trend_atr_period', 14),
                    atr_multiplier=getattr(self.params, 'trend_atr_multiplier', 3.0),
                    max_pyramid=getattr(self.params, 'trend_max_pyramid', 2),
                    min_history=100,
                    adx_threshold=self.trend_adx_threshold,
                    vol_multiplier=self.trend_vol_multiplier,
                    use_mtf_filter=getattr(self.params, 'trend_use_mtf_filter', True),
                    mtf_timeframe=getattr(self.params, 'trend_mtf_timeframe', '1h'),
                    mtf_ema_period=getattr(self.params, 'trend_mtf_ema_period', 50),
                )
            else:
                trend_result = cached_trend_result
            
            if not trend_result or trend_result.get('signal') is None:
                return {'confirmed': False, 'strength': 0.0, 'reason': 'no_trend_signal'}
            
            trend_signal = trend_result.get('signal')
            trend_action = Action.LONG if trend_signal == 'LONG' else Action.SHORT if trend_signal == 'SHORT' else None
            
            if trend_action is None:
                return {'confirmed': False, 'strength': 0.0, 'reason': 'trend_hold_signal'}
            
            # Проверяем совпадение направлений
            if vbo_signal.action != trend_action:
                return {
                    'confirmed': False, 
                    'strength': 0.0, 
                    'reason': f'opposite_directions (VBO={vbo_signal.action.value}, TREND={trend_action.value})'
                }
            
            # Рассчитываем силу подтверждения на основе индикаторов
            indicators_info = trend_result.get('indicators_info', {})
            adx_val = indicators_info.get('adx', 0)
            adx_strength = min(1.0, adx_val / self.trend_adx_threshold) if adx_val > 0 else 0.0
            
            # Проверяем EMA веер
            ema_confirmation = True
            try:
                ema_short = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                ema_long = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                price = float(df['close'].iloc[-1])
                
                if vbo_signal.action == Action.LONG:
                    ema_confirmation = price > ema_short > ema_long
                else:  # SHORT
                    ema_confirmation = price < ema_short < ema_long
            except Exception:
                ema_confirmation = True  # В случае ошибки не блокируем
            
            if not ema_confirmation:
                return {'confirmed': False, 'strength': 0.0, 'reason': 'ema_fan_not_aligned'}
            
            # Общая сила подтверждения
            strength = (adx_strength * 0.6 + (1.0 if ema_confirmation else 0.0) * 0.4)
            
            if strength >= self.trend_confirmation_strength:
                return {'confirmed': True, 'strength': strength, 'reason': 'trend_confirmed'}
            else:
                return {
                    'confirmed': False, 
                    'strength': strength, 
                    'reason': f'weak_trend_confirmation (strength={strength:.2f} < {self.trend_confirmation_strength})'
                }
            
        except Exception as e:
            logger.error(f"[BreakoutTrendHybrid] Error checking TREND confirmation: {e}")
            # В случае ошибки не блокируем сигнал, но отмечаем слабое подтверждение
            return {'confirmed': True, 'strength': 0.5, 'reason': 'trend_error_fallback'}


def build_breakout_trend_signals(
    df: pd.DataFrame,
    params,
    symbol: str = "Unknown"
) -> List[Signal]:
    """
    Точка входа для бота. Использует BreakoutTrendHybridStrategy для генерации сигналов.
    
    Args:
        df: DataFrame с данными OHLCV
        params: Параметры стратегии
        symbol: Торговая пара для логирования
        
    Returns:
        Список сигналов гибридной стратегии
    """
    strategy = BreakoutTrendHybridStrategy(params)
    return strategy.get_signals(df, symbol=symbol)


__all__ = ["BreakoutTrendHybridStrategy", "build_breakout_trend_signals"]
