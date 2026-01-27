"""
Оптимизированные параметры для V18 MTF на основе анализа V17
Применяются рекомендации из OPTIMIZATION_RECOMMENDATIONS.md
"""

# ============================================================================
# ПРИОРИТЕТ 1: КРИТИЧНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 1. VOLATILITY_RATIO - САМЫЙ ВАЖНЫЙ ПРИЗНАК!
# Анализ: корреляция 0.4182, разница WR 52.8%
# Прибыльные: 0.0029, Убыточные: 0.0019
# Q1 (низкая) = WR 11.2%, Q4 (высокая) = WR 64.0%
MTF_MIN_VOLATILITY_RATIO = 0.0025  # УВЕЛИЧЕНО с 0.0020 (ближе к прибыльным 0.0029)
MTF_MAX_VOLATILITY_RATIO = 1.2     # УМЕНЬШЕНО с 1.5 (более строгий фильтр)

# 2. ТРЕЙЛИНГ-СТОП - главная проблема (52% закрытий по SL_TRAILING)
# Анализ: слишком агрессивный трейлинг → преждевременные закрытия
MTF_TRAILING_ACTIVATION_ATR = 0.40  # УВЕЛИЧЕНО с 0.35 (активация позже)
MTF_TRAILING_DISTANCE_ATR = 0.50   # УВЕЛИЧЕНО с 0.45 (больше расстояние)

# ============================================================================
# ПРИОРИТЕТ 2: ВАЖНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 3. ATR - критически важен (корреляция 0.3484, разница WR 47.3%)
# Анализ: прибыльные = 216.44, убыточные = 135.16
# Q1 (низкий) = WR 13.6%, Q4 (высокий) = WR 60.9%
MTF_MIN_ABSOLUTE_ATR = 120.0       # УВЕЛИЧЕНО с 85.0 (ближе к прибыльным)
MTF_ATR_PERCENT_MIN = 0.0015       # УВЕЛИЧЕНО с 0.001 (более строгий)

# 4. VOLUME - очень важен (корреляция 0.3371, разница WR 49.3%)
# Анализ: прибыльные = 1024.09, убыточные = 481.24
# Q1 (низкий) = WR 14.3%, Q4 (высокий) = WR 63.5%
MTF_MIN_ABSOLUTE_VOLUME = 900.0    # УВЕЛИЧЕНО с 800.0 (ближе к прибыльным 1024.09)
MTF_MIN_VOLUME_SPIKE = 1.6         # УВЕЛИЧЕНО с 1.5 для LONG
MTF_MIN_VOLUME_SPIKE_SHORT = 1.3   # УВЕЛИЧЕНО с 1.1 для SHORT

# 5. TP УРОВНИ - для лучшего RR
# Анализ: средний RR = 1.69 (цель ≥1.8), только 10.3% закрываются по TP_LEVEL_1
MTF_TP_LEVELS = [2.5, 3.0, 3.8]    # УВЕЛИЧЕНО: TP1 с 2.2 до 2.5

# ============================================================================
# ПРИОРИТЕТ 3: ОПЦИОНАЛЬНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 6. RSI - влияет на WR (разница 48.5%)
# Анализ: Q1 (низкий) = WR 41.7%, Q4 (высокий) = WR 67.6%
MTF_LONG_RSI_MIN = 0.15            # Оставляем как есть
MTF_LONG_RSI_MAX = 0.55            # УМЕНЬШЕНО с 0.60 (более строгий)
MTF_SHORT_RSI_MIN = 0.40           # УВЕЛИЧЕНО с 0.35 (более строгий)
MTF_SHORT_RSI_MAX = 0.85           # Оставляем как есть

# 7. ADX - сила тренда
MTF_MIN_ADX = 27.0                 # УВЕЛИЧЕНО с 25.0 (более сильные тренды)
MTF_MIN_ADX_SHORT = 22.0           # УВЕЛИЧЕНО с 20.0 для SHORT

# ============================================================================
# СЛОВАРЬ ДЛЯ ПРИМЕНЕНИЯ
# ============================================================================

MTF_OPTIMIZED_PARAMS = {
    # Приоритет 1
    'min_volatility_ratio': MTF_MIN_VOLATILITY_RATIO,
    'max_volatility_ratio': MTF_MAX_VOLATILITY_RATIO,
    'trailing_activation_atr': MTF_TRAILING_ACTIVATION_ATR,
    'trailing_distance_atr': MTF_TRAILING_DISTANCE_ATR,
    
    # Приоритет 2
    'min_absolute_atr': MTF_MIN_ABSOLUTE_ATR,
    'atr_percent_min': MTF_ATR_PERCENT_MIN,
    'min_absolute_volume': MTF_MIN_ABSOLUTE_VOLUME,
    'min_volume_spike': MTF_MIN_VOLUME_SPIKE,
    'min_volume_spike_short': MTF_MIN_VOLUME_SPIKE_SHORT,
    'tp_levels': MTF_TP_LEVELS,
    
    # Приоритет 3
    'long_rsi_min': MTF_LONG_RSI_MIN,
    'long_rsi_max': MTF_LONG_RSI_MAX,
    'short_rsi_min': MTF_SHORT_RSI_MIN,
    'short_rsi_max': MTF_SHORT_RSI_MAX,
    'min_adx': MTF_MIN_ADX,
    'min_adx_short': MTF_MIN_ADX_SHORT,
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
