"""
КРИТИЧЕСКИ ОПТИМИЗИРОВАННЫЕ параметры для V18 MTF
Обновлено после третьего анализа: Win Rate 35.8% (улучшение!), но SL_INITIAL 28.9% (ухудшение!)
ФОКУС: Ужесточение фильтров входа для предотвращения SL_INITIAL и VERY_BAD сделок
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

# 3. ATR - критически важен (SL_INITIAL увеличилось до 28.9%!)
# Анализ: нужно ЕЩЕ БОЛЬШЕ ужесточить - только экстремально сильные движения
MTF_MIN_ABSOLUTE_ATR = 200.0       # УВЕЛИЧЕНО с 180.0 (только экстремально сильные движения)
MTF_ATR_PERCENT_MIN = 0.0025       # УВЕЛИЧЕНО с 0.0020 (еще более строгий процент)

# 4. VOLUME - очень важен (VERY_BAD увеличилось до 30.8%!)
# Анализ: нужно ЕЩЕ БОЛЬШЕ объема для подтверждения входа
MTF_MIN_ABSOLUTE_VOLUME = 1300.0   # УВЕЛИЧЕНО с 1100.0 (еще больше объема)
MTF_MIN_VOLUME_SPIKE = 2.2         # УВЕЛИЧЕНО с 2.0 (экстремально строгий фильтр для LONG)
MTF_MIN_VOLUME_SPIKE_SHORT = 1.7   # УВЕЛИЧЕНО с 1.5 (более строгий для SHORT)

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

# 7. ADX - сила тренда (VERY_BAD увеличилось до 30.8%!)
# Анализ: нужно ЕЩЕ БОЛЬШЕ ужесточить - только экстремально сильные тренды
MTF_MIN_ADX = 38.0                 # УВЕЛИЧЕНО с 35.0 (только экстремально сильные тренды)
MTF_MIN_ADX_SHORT = 32.0           # УВЕЛИЧЕНО с 30.0 для SHORT (очень строгий)

# 8. MTF ФИЛЬТРЫ - усиление проверки конфликта трендов
# Анализ: нужно усилить проверку конфликта между ТФ для предотвращения плохих входов
MTF_TREND_CONFLICT_STRICT = True   # Включить строгую проверку конфликта
MTF_MIN_1H_ADX = 28.0              # Минимальный ADX на 1h для входа
MTF_MIN_1H_ADX_SHORT = 25.0         # Минимальный ADX на 1h для SHORT
MTF_DI_RATIO_1H = 1.20             # УВЕЛИЧЕНО с 1.15 (более строгое требование для DI на 1h)

# ============================================================================
# СЛОВАРЬ ДЛЯ ПРИМЕНЕНИЯ
# ============================================================================

MTF_OPTIMIZED_PARAMS = {
    # Приоритет 1 (критично - РАДИКАЛЬНОЕ УЖЕСТОЧЕНИЕ)
    'min_volatility_ratio': MTF_MIN_VOLATILITY_RATIO,  # 0.0030 (увеличено с 0.0025)
    'max_volatility_ratio': MTF_MAX_VOLATILITY_RATIO,  # 1.05 (уменьшено с 1.1)
    'trailing_activation_atr': MTF_TRAILING_ACTIVATION_ATR,  # 0.65 (увеличено с 0.50)
    'trailing_distance_atr': MTF_TRAILING_DISTANCE_ATR,  # 0.75 (увеличено с 0.60)
    
    # Приоритет 2 (важно - ЕЩЕ БОЛЬШЕ УЖЕСТОЧЕНИЕ)
    'min_absolute_atr': MTF_MIN_ABSOLUTE_ATR,  # 200.0 (увеличено с 180.0)
    'atr_percent_min': MTF_ATR_PERCENT_MIN,  # 0.0025 (увеличено с 0.0020)
    'min_absolute_volume': MTF_MIN_ABSOLUTE_VOLUME,  # 1300.0 (увеличено с 1100.0)
    'min_volume_spike': MTF_MIN_VOLUME_SPIKE,  # 2.2 (увеличено с 2.0)
    'min_volume_spike_short': MTF_MIN_VOLUME_SPIKE_SHORT,  # 1.7 (увеличено с 1.5)
    'tp_levels': MTF_TP_LEVELS,  # [3.0, 3.5, 4.2] (увеличено с [2.8, 3.2, 4.0])
    
    # Приоритет 3 (опционально - ЕЩЕ БОЛЬШЕ УЖЕСТОЧЕНИЕ)
    'long_rsi_min': MTF_LONG_RSI_MIN,  # 0.15
    'long_rsi_max': MTF_LONG_RSI_MAX,  # 0.55
    'short_rsi_min': MTF_SHORT_RSI_MIN,  # 0.40
    'short_rsi_max': MTF_SHORT_RSI_MAX,  # 0.85
    'min_adx': MTF_MIN_ADX,  # 38.0 (увеличено с 35.0)
    'min_adx_short': MTF_MIN_ADX_SHORT,  # 32.0 (увеличено с 30.0)
    
    # Приоритет 4 (MTF фильтры - усиление)
    'trend_conflict_strict': MTF_TREND_CONFLICT_STRICT,  # True
    'min_1h_adx': MTF_MIN_1H_ADX,  # 28.0
    'min_1h_adx_short': MTF_MIN_1H_ADX_SHORT,  # 25.0
    'di_ratio_1h': MTF_DI_RATIO_1H,  # 1.20 (увеличено с 1.15)
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
