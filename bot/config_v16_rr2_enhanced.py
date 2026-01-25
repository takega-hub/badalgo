# СОЗДАТЬ НОВЫЙ файл для настройки параметров:

# config_v16_rr2_enhanced.py
ENHANCED_CONFIG = {
    # Риск-менеджмент
    'MIN_POSITION_SIZE': 0.03,  # 3% минимальный размер позиции
    'MAX_POSITION_SIZE': 0.12,  # 12% максимальный размер позиции
    'MIN_SHARES': 0.001,        # Минимальное количество акций
    'MIN_CLOSE_SHARES': 0.0005, # Минимальное количество для закрытия
    
    # Стоп-лосс
    'ATR_MULTIPLIER': 1.8,      # Уменьшенный множитель для более узких SL
    'MIN_SL_PERCENT': 0.002,    # 0.2% минимальный SL
    'MAX_SL_PERCENT': 0.008,    # 0.8% максимальный SL
    
    # Take Profit
    'TP_LEVELS': [1.2, 1.8, 2.4],
    'TP_CLOSE_PERCENTAGES': [0.40, 0.35, 0.25],
    
    # Трейлинг
    'TRAILING_ACTIVATION': 0.4,  # Активация при 0.4 ATR прибыли
    'TRAILING_DISTANCE': 0.6,    # Дистанция 0.6 ATR
    
    # Время удержания
    'MIN_HOLD_STEPS': 8,
    'MAX_HOLD_STEPS': 50,
    
    # Фильтры входа
    'MIN_TREND_STRENGTH': 0.3,    # Усилен тренд-фильтр
    'MAX_VOLATILITY_RATIO': 2.0,  # Ужесточен волатильность
    
    # Награды
    'TP_BONUS_MULTIPLIER': 5.0,
    'MANUAL_PENALTY': 2.0,        # Увеличен штраф за MANUAL
    'SL_INITIAL_PENALTY': 2.5,    # Большой штраф за начальный SL
}