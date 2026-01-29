"""
КРИТИЧЕСКИ ОПТИМИЗИРОВАННЫЕ параметры для V18 MTF
Обновлено после второго анализа: Win Rate 32.5%, SL_TRAILING 50.9%, SL_INITIAL 26.7%
ТРЕБУЕТСЯ РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ ПАРАМЕТРОВ
"""

# ============================================================================
# ПРИОРИТЕТ 1: КРИТИЧНЫЕ ОПТИМИЗАЦИИ (РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ)
# ============================================================================

# 1. VOLATILITY_RATIO - САМЫЙ ВАЖНЫЙ ПРИЗНАК!
# Анализ: Win Rate 32.5%, слишком много плохих входов (29.1% VERY_BAD)
# РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ: увеличиваем минимум, уменьшаем максимум
MTF_MIN_VOLATILITY_RATIO = 0.0030  # УВЕЛИЧЕНО с 0.0025 (только высоковолатильные движения)
MTF_MAX_VOLATILITY_RATIO = 1.05    # УМЕНЬШЕНО с 1.1 (еще более строгий фильтр)

# 2. ТРЕЙЛИНГ-СТОП - главная проблема (50.9% закрытий по SL_TRAILING)
# Анализ: даже после увеличения до 0.50/0.60 все еще слишком агрессивный
# РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ: еще больше увеличиваем
MTF_TRAILING_ACTIVATION_ATR = 0.65  # УВЕЛИЧЕНО с 0.50 (активация намного позже)
MTF_TRAILING_DISTANCE_ATR = 0.75   # УВЕЛИЧЕНО с 0.60 (еще больше расстояние)

# ============================================================================
# ПРИОРИТЕТ 2: ВАЖНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 3. ATR - критически важен (много SL_INITIAL: 26.7%)
# Анализ: нужно РАДИКАЛЬНО ужесточить - только очень сильные движения
MTF_MIN_ABSOLUTE_ATR = 180.0       # УВЕЛИЧЕНО с 150.0 (только очень сильные движения)
MTF_ATR_PERCENT_MIN = 0.0020       # УВЕЛИЧЕНО с 0.0015 (более строгий процент)

# 4. VOLUME - очень важен (много VERY_BAD сделок: 29.1%)
# Анализ: нужно РАДИКАЛЬНО больше объема для подтверждения входа
MTF_MIN_ABSOLUTE_VOLUME = 1100.0   # УВЕЛИЧЕНО с 900.0 (значительно больше объема)
MTF_MIN_VOLUME_SPIKE = 2.0         # УВЕЛИЧЕНО с 1.8 (очень строгий фильтр для LONG)
MTF_MIN_VOLUME_SPIKE_SHORT = 1.5   # УВЕЛИЧЕНО с 1.3 (более строгий для SHORT)

# 5. TP УРОВНИ - для лучшего RR (средний RR 1.82, но только 7.1% по TP_LEVEL_1)
# Анализ: нужно увеличить TP1 еще больше, чтобы больше сделок закрывалось по TP
MTF_TP_LEVELS = [3.0, 3.5, 4.2]    # УВЕЛИЧЕНО: TP1 с 2.8 до 3.0 (больше сделок по TP)

# ============================================================================
# ПРИОРИТЕТ 3: ОПЦИОНАЛЬНЫЕ ОПТИМИЗАЦИИ
# ============================================================================

# 6. RSI - влияет на WR
# Анализ V18: оставляем как есть (уже оптимизировано)
MTF_LONG_RSI_MIN = 0.15            # Оставляем
MTF_LONG_RSI_MAX = 0.55            # Оставляем
MTF_SHORT_RSI_MIN = 0.40           # Оставляем
MTF_SHORT_RSI_MAX = 0.85           # Оставляем

# 7. ADX - сила тренда (много плохих входов: 29.1% VERY_BAD)
# Анализ: нужно РАДИКАЛЬНО ужесточить - только очень сильные тренды
MTF_MIN_ADX = 35.0                 # УВЕЛИЧЕНО с 30.0 (только экстремально сильные тренды)
MTF_MIN_ADX_SHORT = 30.0           # УВЕЛИЧЕНО с 25.0 для SHORT (очень строгий)

# ============================================================================
# СЛОВАРЬ ДЛЯ ПРИМЕНЕНИЯ
# ============================================================================

MTF_OPTIMIZED_PARAMS = {
    # Приоритет 1 (критично - РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ)
    'min_volatility_ratio': MTF_MIN_VOLATILITY_RATIO,  # 0.0030 (увеличено с 0.0025)
    'max_volatility_ratio': MTF_MAX_VOLATILITY_RATIO,  # 1.05 (уменьшено с 1.1)
    'trailing_activation_atr': MTF_TRAILING_ACTIVATION_ATR,  # 0.65 (увеличено с 0.50)
    'trailing_distance_atr': MTF_TRAILING_DISTANCE_ATR,  # 0.75 (увеличено с 0.60)
    
    # Приоритет 2 (важно - РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ)
    'min_absolute_atr': MTF_MIN_ABSOLUTE_ATR,  # 180.0 (увеличено с 150.0)
    'atr_percent_min': MTF_ATR_PERCENT_MIN,  # 0.0020 (увеличено с 0.0015)
    'min_absolute_volume': MTF_MIN_ABSOLUTE_VOLUME,  # 1100.0 (увеличено с 900.0)
    'min_volume_spike': MTF_MIN_VOLUME_SPIKE,  # 2.0 (увеличено с 1.8)
    'min_volume_spike_short': MTF_MIN_VOLUME_SPIKE_SHORT,  # 1.5 (увеличено с 1.3)
    'tp_levels': MTF_TP_LEVELS,  # [3.0, 3.5, 4.2] (увеличено с [2.8, 3.2, 4.0])
    
    # Приоритет 3 (опционально - РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ)
    'long_rsi_min': MTF_LONG_RSI_MIN,  # 0.15
    'long_rsi_max': MTF_LONG_RSI_MAX,  # 0.55
    'short_rsi_min': MTF_SHORT_RSI_MIN,  # 0.40
    'short_rsi_max': MTF_SHORT_RSI_MAX,  # 0.85
    'min_adx': MTF_MIN_ADX,  # 35.0 (увеличено с 30.0)
    'min_adx_short': MTF_MIN_ADX_SHORT,  # 30.0 (увеличено с 25.0)
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
