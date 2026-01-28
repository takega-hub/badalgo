"""
Оптимизированные параметры для V18 MTF на основе анализа V18 MTF
Обновлено после анализа результатов обучения V18 MTF (Win Rate 31.5%, отрицательный PnL)
"""

# ============================================================================
# ПРИОРИТЕТ 1: КРИТИЧНЫЕ ОПТИМИЗАЦИИ (на основе анализа V18 MTF)
# ============================================================================

# 1. VOLATILITY_RATIO - САМЫЙ ВАЖНЫЙ ПРИЗНАК!
# Анализ V18: Win Rate 31.5%, слишком много плохих входов
MTF_MIN_VOLATILITY_RATIO = 0.0025  # Оставляем (уже оптимизировано)
MTF_MAX_VOLATILITY_RATIO = 1.1     # УМЕНЬШЕНО с 1.2 (более строгий фильтр плохих входов)

# 2. ТРЕЙЛИНГ-СТОП - главная проблема V18 (52.3% закрытий по SL_TRAILING)
# Анализ: слишком агрессивный трейлинг → преждевременные закрытия
MTF_TRAILING_ACTIVATION_ATR = 0.50  # УВЕЛИЧЕНО с 0.40 (активация позже, меньше преждевременных закрытий)
MTF_TRAILING_DISTANCE_ATR = 0.60   # УВЕЛИЧЕНО с 0.50 (больше расстояние, меньше стоп-аутов)

# ============================================================================
# ПРИОРИТЕТ 2: ВАЖНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 3. ATR - критически важен (много SL_INITIAL: 26.1%)
# Анализ V18: нужно ужесточить фильтры для лучших входов
MTF_MIN_ABSOLUTE_ATR = 150.0       # УВЕЛИЧЕНО с 120.0 (только сильные движения)
MTF_ATR_PERCENT_MIN = 0.0015       # Оставляем

# 4. VOLUME - очень важен (много VERY_BAD сделок: 28.9%)
# Анализ V18: нужно больше объема для подтверждения входа
MTF_MIN_ABSOLUTE_VOLUME = 900.0    # Оставляем
MTF_MIN_VOLUME_SPIKE = 1.8         # УВЕЛИЧЕНО с 1.6 (более строгий фильтр для LONG)
MTF_MIN_VOLUME_SPIKE_SHORT = 1.3   # Оставляем

# 5. TP УРОВНИ - для лучшего RR (средний RR 1.80, цель ≥1.8)
# Анализ V18: только 6.9% закрываются по TP_LEVEL_1 - нужно увеличить TP1
MTF_TP_LEVELS = [2.8, 3.2, 4.0]    # УВЕЛИЧЕНО: TP1 с 2.5 до 2.8 (лучший RR)

# ============================================================================
# ПРИОРИТЕТ 3: ОПЦИОНАЛЬНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 6. RSI - влияет на WR
# Анализ V18: оставляем как есть (уже оптимизировано)
MTF_LONG_RSI_MIN = 0.15            # Оставляем
MTF_LONG_RSI_MAX = 0.55            # Оставляем
MTF_SHORT_RSI_MIN = 0.40           # Оставляем
MTF_SHORT_RSI_MAX = 0.85           # Оставляем

# 7. ADX - сила тренда (много плохих входов)
# Анализ V18: нужно ужесточить - только сильные тренды
MTF_MIN_ADX = 30.0                 # УВЕЛИЧЕНО с 27.0 (только очень сильные тренды)
MTF_MIN_ADX_SHORT = 25.0           # УВЕЛИЧЕНО с 22.0 для SHORT (более строгий)

# ============================================================================
# СЛОВАРЬ ДЛЯ ПРИМЕНЕНИЯ
# ============================================================================

MTF_OPTIMIZED_PARAMS = {
    # Приоритет 1 (критично - на основе анализа V18)
    'min_volatility_ratio': MTF_MIN_VOLATILITY_RATIO,  # 0.0025
    'max_volatility_ratio': MTF_MAX_VOLATILITY_RATIO,  # 1.1 (уменьшено с 1.2)
    'trailing_activation_atr': MTF_TRAILING_ACTIVATION_ATR,  # 0.50 (увеличено с 0.40)
    'trailing_distance_atr': MTF_TRAILING_DISTANCE_ATR,  # 0.60 (увеличено с 0.50)
    
    # Приоритет 2 (важно)
    'min_absolute_atr': MTF_MIN_ABSOLUTE_ATR,  # 150.0 (увеличено с 120.0)
    'atr_percent_min': MTF_ATR_PERCENT_MIN,  # 0.0015
    'min_absolute_volume': MTF_MIN_ABSOLUTE_VOLUME,  # 900.0
    'min_volume_spike': MTF_MIN_VOLUME_SPIKE,  # 1.8 (увеличено с 1.6)
    'min_volume_spike_short': MTF_MIN_VOLUME_SPIKE_SHORT,  # 1.3
    'tp_levels': MTF_TP_LEVELS,  # [2.8, 3.2, 4.0] (увеличено с [2.5, 3.0, 3.8])
    
    # Приоритет 3 (опционально)
    'long_rsi_min': MTF_LONG_RSI_MIN,  # 0.15
    'long_rsi_max': MTF_LONG_RSI_MAX,  # 0.55
    'short_rsi_min': MTF_SHORT_RSI_MIN,  # 0.40
    'short_rsi_max': MTF_SHORT_RSI_MAX,  # 0.85
    'min_adx': MTF_MIN_ADX,  # 30.0 (увеличено с 27.0)
    'min_adx_short': MTF_MIN_ADX_SHORT,  # 25.0 (увеличено с 22.0)
}

# ============================================================================
# ИНСТРУКЦИЯ ПО ПРИМЕНЕНИЮ
# ============================================================================

"""
Для применения этих параметров в crypto_env_v18_mtf.py:

1. Импортируйте параметры:
   from bot.mtf_optimized_params import MTF_OPTIMIZED_PARAMS

2. В методе __init__ класса CryptoTradingEnvV18_MTF, после super().__init__(),
   переопределите параметры:

   # Приоритет 1 (критично)
   self.min_volatility_ratio = MTF_OPTIMIZED_PARAMS['min_volatility_ratio']
   self.max_volatility_ratio = MTF_OPTIMIZED_PARAMS['max_volatility_ratio']
   self.trailing_activation_atr = MTF_OPTIMIZED_PARAMS['trailing_activation_atr']
   self.trailing_distance_atr = MTF_OPTIMIZED_PARAMS['trailing_distance_atr']
   
   # Приоритет 2 (важно)
   # В методе _check_entry_filters_strict обновите:
   # min_absolute_atr = MTF_OPTIMIZED_PARAMS['min_absolute_atr']
   # min_absolute_volume = MTF_OPTIMIZED_PARAMS['min_absolute_volume']
   # и т.д.
   
   # Приоритет 3 (опционально)
   self.long_config['max_rsi_norm'] = MTF_OPTIMIZED_PARAMS['long_rsi_max']
   self.short_config['min_rsi_norm'] = MTF_OPTIMIZED_PARAMS['short_rsi_min']
   self.min_adx = MTF_OPTIMIZED_PARAMS['min_adx']

3. Обновите tp_levels:
   self.tp_levels = MTF_OPTIMIZED_PARAMS['tp_levels']
"""
