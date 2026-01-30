"""
ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ДЛЯ V18 MTF НА ОСНОВЕ АНАЛИЗА ЛОГОВ
Обновлено после детального анализа: Win Rate 35.4%, SL_INITIAL 29.1%, SL_TRAILING 46.0%
ФОКУС: Ужесточение фильтров входа и улучшение MTF синхронизации согласно рекомендациям
"""

# ============================================================================
# ПРИОРИТЕТ 1: КРИТИЧНЫЕ ОПТИМИЗАЦИИ (СОГЛАСНО РЕКОМЕНДАЦИЯМ АНАЛИЗА)
# ============================================================================

# 1. VOLATILITY_RATIO - Ужесточить фильтр волатильности
# Рекомендация: уменьшить MTF_MAX_VOLATILITY_RATIO с 1.2 до 1.1
MTF_MIN_VOLATILITY_RATIO = 0.0030  # Уже достаточно высоко
MTF_MAX_VOLATILITY_RATIO = 1.05    # Уже ниже рекомендации 1.1 (оставляем)

# 2. ТРЕЙЛИНГ-СТОП - Увеличить для уменьшения SL_TRAILING (46%)
# Рекомендация: увеличить trailing_activation_atr с 0.40 до 0.45-0.50
# Рекомендация: увеличить trailing_distance_atr с 0.50 до 0.55-0.60
MTF_TRAILING_ACTIVATION_ATR = 0.65  # Уже выше рекомендации (оставляем)
MTF_TRAILING_DISTANCE_ATR = 0.75   # Уже выше рекомендации (оставляем)

# ============================================================================
# ПРИОРИТЕТ 2: ВАЖНЫЕ ОПТИМИЗАЦИИ (СОГЛАСНО РЕКОМЕНДАЦИЯМ)
# ============================================================================

# 3. ATR - Увеличить минимальный ATR для фильтрации слабых движений
# Рекомендация: увеличить MTF_MIN_ABSOLUTE_ATR с 120.0 до 150.0
MTF_MIN_ABSOLUTE_ATR = 200.0       # Уже выше рекомендации (оставляем)
MTF_ATR_PERCENT_MIN = 0.0025       # Оставляем

# 4. VOLUME - Увеличить требования к объему
# Рекомендация: увеличить MTF_MIN_VOLUME_SPIKE с 1.6 до 1.8
MTF_MIN_ABSOLUTE_VOLUME = 1300.0   # Оставляем
MTF_MIN_VOLUME_SPIKE = 2.2         # Уже выше рекомендации (оставляем)
MTF_MIN_VOLUME_SPIKE_SHORT = 1.7   # Оставляем

# 5. TP УРОВНИ - Оптимизировать для увеличения TP закрытий
MTF_TP_LEVELS = [3.0, 3.5, 4.2]    # Оставляем текущие уровни

# ============================================================================
# ПРИОРИТЕТ 3: УСИЛЕНИЕ MTF ФИЛЬТРОВ (СОГЛАСНО РЕКОМЕНДАЦИЯМ)
# ============================================================================

# 6. ADX - Ужесточить требования к силе тренда
# Рекомендация: ужесточить MTF_MIN_ADX с 27.0 до 30.0
MTF_MIN_ADX = 38.0                 # Уже выше рекомендации (оставляем)
MTF_MIN_ADX_SHORT = 32.0           # Оставляем

# 7. RSI - Оставить текущие настройки
MTF_LONG_RSI_MIN = 0.15            # Оставляем
MTF_LONG_RSI_MAX = 0.55            # Оставляем
MTF_SHORT_RSI_MIN = 0.40           # Оставляем
MTF_SHORT_RSI_MAX = 0.85           # Оставляем

# 8. MTF ФИЛЬТРЫ - Усилить проверку конфликта трендов между ТФ
# Рекомендация: усилить проверку конфликта трендов между ТФ
MTF_TREND_CONFLICT_STRICT = True   # Включить строгую проверку конфликта
MTF_MIN_1H_ADX = 30.0              # Увеличиваем с 28.0 до 30.0 для более строгой фильтрации
MTF_MIN_1H_ADX_SHORT = 27.0        # Увеличиваем с 25.0 до 27.0 для SHORT
MTF_DI_RATIO_1H = 1.25             # Увеличиваем с 1.20 до 1.25 для более строгого требования DI

# 9. ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ УСИЛЕНИЯ КОНФЛИКТА ТРЕНДОВ
MTF_TREND_CONFLICT_MULTIPLIER = 1.10  # Множитель для проверки конфликта (было 1.05 в коде)
MTF_REQUIRE_4H_CONFIRMATION = True    # Требовать подтверждение от 4h таймфрейма

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
    'min_1h_adx': MTF_MIN_1H_ADX,  # 30.0
    'min_1h_adx_short': MTF_MIN_1H_ADX_SHORT,  # 27.0
    'di_ratio_1h': MTF_DI_RATIO_1H,  # 1.25 (увеличено с 1.20)
    'trend_conflict_multiplier': MTF_TREND_CONFLICT_MULTIPLIER,  # 1.10
    'require_4h_confirmation': MTF_REQUIRE_4H_CONFIRMATION,  # True
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
   self.mtf_min_adx_short = MTF_OPTIMIZED_PARAMS['min_adx_short']

3. Обновите tp_levels:
   self.tp_levels = MTF_OPTIMIZED_PARAMS['tp_levels']

4. Обновите MTF фильтры:
   self.mtf_min_1h_adx = MTF_OPTIMIZED_PARAMS['min_1h_adx']
   self.mtf_min_1h_adx_short = MTF_OPTIMIZED_PARAMS['min_1h_adx_short']
   self.mtf_di_ratio_1h = MTF_OPTIMIZED_PARAMS['di_ratio_1h']
   self.mtf_trend_conflict_multiplier = MTF_OPTIMIZED_PARAMS['trend_conflict_multiplier']
   self.mtf_require_4h_confirmation = MTF_OPTIMIZED_PARAMS['require_4h_confirmation']
"""
