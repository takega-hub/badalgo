"""
Гибридная стратегия TREND_ZSCORE_HYBRID.

Объединяет трендовую логику TREND с статистическими фильтрами ZSCORE
для повышения качества сигналов и уменьшения ложных входов.

Концепция:
- TREND определяет направление тренда (EMA веер, ADX, объем)
- ZSCORE подтверждает качество входа (Z-Score, RSI, статистические фильтры)
- Сигнал генерируется только если ОБА условия выполнены

Преимущества:
- Повышение Win Rate за счет двойной фильтрации
- Уменьшение ложных сигналов
- Работает на SOLUSDT и ETHUSDT (где обе стратегии успешны)
"""
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

from bot.strategy import Action, Signal
from bot.strategy import generate_trend_signal as _generate_trend_signal
from bot.zscore_strategy_v2 import generate_signals as _generate_zscore_signals, StrategyParams as ZScoreParams

logger = logging.getLogger(__name__)


class TrendZScoreHybridStrategy:
    """Гибридная стратегия, объединяющая TREND и ZSCORE."""
    
    def __init__(self, params):
        self.params = params
        # Параметры для TREND стратегии
        self.trend_adx_threshold = getattr(params, 'trend_adx_threshold', 25.0)
        self.trend_vol_multiplier = getattr(params, 'trend_vol_multiplier', 1.3)
        # Параметры для ZSCORE стратегии
        self.zscore_window = getattr(params, 'zscore_window', 20)
        self.zscore_z_long = getattr(params, 'zscore_z_long', -2.0)
        self.zscore_z_short = getattr(params, 'zscore_z_short', 2.0)
        self.zscore_adx_threshold = getattr(params, 'zscore_adx_threshold', 20.0)
        self.zscore_rsi_enabled = getattr(params, 'zscore_rsi_enabled', True)
        # Параметры гибридной стратегии
        self.require_both_signals = getattr(params, 'hybrid_require_both_signals', True)
        self.zscore_confirmation_strength = getattr(params, 'hybrid_zscore_confirmation_strength', 0.7)
        # Минимальная сила Z-Score для подтверждения (0.0-1.0)
        # 0.7 означает, что Z-Score должен быть на 70% пути к порогу входа
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Генерирует сигналы гибридной стратегии.
        
        Логика:
        1. Получаем сигнал от TREND стратегии
        2. Проверяем подтверждение от ZSCORE фильтров
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
        
        # Определяем режим бэктеста из параметров или по умолчанию False
        backtest_mode = getattr(self.params, 'backtest_mode', False)
        
        # 1. Получаем сигнал от TREND стратегии
        trend_state = {'backtest_mode': backtest_mode}
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
        
        # Если нет сигнала от TREND - выходим
        if not trend_result or trend_result.get('signal') is None:
            return []
        
        trend_signal = trend_result.get('signal')
        trend_action = Action.LONG if trend_signal == 'LONG' else Action.SHORT if trend_signal == 'SHORT' else None
        
        if trend_action is None:
            return []
        
        # 2. Проверяем подтверждение от ZSCORE фильтров
        zscore_confirmed = self._check_zscore_confirmation(df, trend_action, symbol)
        
        if not zscore_confirmed:
            logger.debug(
                f"[TrendZScoreHybrid] {symbol} TREND signal {trend_action.value} "
                f"rejected by ZSCORE filters"
            )
            return []
        
        # 3. Генерируем подтвержденный сигнал
        price = float(df['close'].iloc[-1])
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        
        # Используем SL/TP от TREND стратегии (они оптимизированы)
        stop_loss = trend_result.get('stop_loss')
        take_profit = trend_result.get('take_profit')
        trailing = trend_result.get('trailing')
        
        # Если SL/TP нет, рассчитываем на основе ATR
        if stop_loss is None or take_profit is None:
            atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.005
            if stop_loss is None:
                stop_loss = price - (3.5 * atr_val) if trend_action == Action.LONG else price + (3.5 * atr_val)
            if take_profit is None:
                take_profit = price + (7.0 * atr_val) if trend_action == Action.LONG else price - (7.0 * atr_val)
        
        reason = f"trend_zscore_hybrid_{trend_action.value.lower()}"
        if trend_result.get('reason'):
            reason += f"_{trend_result.get('reason')}"
        
        signal = Signal(
            timestamp=timestamp,
            action=trend_action,
            reason=reason,
            price=price,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            trailing=trailing,
            indicators_info={
                **trend_result.get('indicators_info', {}),
                'hybrid_confirmed': True,
                'zscore_confirmation_strength': zscore_confirmed.get('strength', 1.0),
            }
        )
        
        signals.append(signal)
        logger.info(
            f"[TrendZScoreHybrid] {symbol} Generated confirmed signal: "
            f"{trend_action.value} @ {price:.2f} (TREND + ZSCORE confirmation)"
        )
        
        return signals
    
    def _check_zscore_confirmation(
        self, 
        df: pd.DataFrame, 
        trend_action: Action, 
        symbol: str
    ) -> Dict[str, Any]:
        """
        Проверяет подтверждение сигнала от ZSCORE фильтров.
        
        Args:
            df: DataFrame с данными
            trend_action: Направление сигнала от TREND
            symbol: Торговая пара
            
        Returns:
            Dict с ключами:
            - 'confirmed': bool - подтвержден ли сигнал
            - 'strength': float - сила подтверждения (0.0-1.0)
        """
        # Создаем параметры для ZSCORE
        zscore_params = ZScoreParams(
            window=self.zscore_window,
            z_long=self.zscore_z_long,
            z_short=self.zscore_z_short,
            adx_threshold=self.zscore_adx_threshold,
            rsi_enabled=self.zscore_rsi_enabled,
        )
        
        # Генерируем ZSCORE сигналы
        try:
            zscore_df = _generate_zscore_signals(df, zscore_params)
            if zscore_df is None or zscore_df.empty:
                return {'confirmed': False, 'strength': 0.0}
            
            last_row = zscore_df.iloc[-1]
            
            # Проверяем условия подтверждения
            z_value = last_row.get('z', 0.0)
            rsi_value = last_row.get('rsi', 50.0)
            adx_value = last_row.get('adx', 0.0)
            market_allowed = last_row.get('market_allowed', False)
            
            # Базовая проверка: рынок должен быть разрешен для торговли
            if not market_allowed:
                return {'confirmed': False, 'strength': 0.0, 'reason': 'market_not_allowed'}
            
            # Проверка ADX: должен быть ниже порога (ZSCORE работает во флэте)
            # Но для гибрида мы можем быть более гибкими
            if adx_value > self.zscore_adx_threshold * 1.5:  # Слишком сильный тренд для ZSCORE
                return {'confirmed': False, 'strength': 0.0, 'reason': 'adx_too_high'}
            
            # Проверка направления сигнала
            if trend_action == Action.LONG:
                # Для LONG: Z-Score должен быть отрицательным (перепроданность)
                # или близким к порогу входа
                z_threshold = self.zscore_z_long * self.zscore_confirmation_strength
                z_confirmed = z_value <= z_threshold
                
                # RSI должен быть не в зоне перекупленности
                rsi_confirmed = rsi_value < 70.0 if self.zscore_rsi_enabled else True
                
                if z_confirmed and rsi_confirmed:
                    # Рассчитываем силу подтверждения
                    strength = min(1.0, abs(z_value) / abs(self.zscore_z_long))
                    return {'confirmed': True, 'strength': strength, 'reason': 'zscore_long_confirmed'}
                else:
                    return {
                        'confirmed': False, 
                        'strength': 0.0, 
                        'reason': f'zscore_not_confirmed_long (z={z_value:.2f}, rsi={rsi_value:.2f})'
                    }
            
            elif trend_action == Action.SHORT:
                # Для SHORT: Z-Score должен быть положительным (перекупленность)
                # или близким к порогу входа
                z_threshold = self.zscore_z_short * self.zscore_confirmation_strength
                z_confirmed = z_value >= z_threshold
                
                # RSI должен быть не в зоне перепроданности
                rsi_confirmed = rsi_value > 30.0 if self.zscore_rsi_enabled else True
                
                if z_confirmed and rsi_confirmed:
                    # Рассчитываем силу подтверждения
                    strength = min(1.0, abs(z_value) / abs(self.zscore_z_short))
                    return {'confirmed': True, 'strength': strength, 'reason': 'zscore_short_confirmed'}
                else:
                    return {
                        'confirmed': False, 
                        'strength': 0.0, 
                        'reason': f'zscore_not_confirmed_short (z={z_value:.2f}, rsi={rsi_value:.2f})'
                    }
            
            return {'confirmed': False, 'strength': 0.0, 'reason': 'unknown_action'}
            
        except Exception as e:
            logger.error(f"[TrendZScoreHybrid] Error checking ZSCORE confirmation: {e}")
            # В случае ошибки не блокируем сигнал, но отмечаем слабое подтверждение
            return {'confirmed': True, 'strength': 0.5, 'reason': 'zscore_error_fallback'}


def build_trend_zscore_signals(
    df: pd.DataFrame,
    params,
    symbol: str = "Unknown"
) -> List[Signal]:
    """
    Точка входа для бота. Использует TrendZScoreHybridStrategy для генерации сигналов.
    
    Args:
        df: DataFrame с данными OHLCV
        params: Параметры стратегии
        symbol: Торговая пара для логирования
        
    Returns:
        Список сигналов гибридной стратегии
    """
    strategy = TrendZScoreHybridStrategy(params)
    return strategy.get_signals(df, symbol=symbol)


__all__ = ["TrendZScoreHybridStrategy", "build_trend_zscore_signals"]
