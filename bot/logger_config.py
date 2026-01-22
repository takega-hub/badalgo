"""
Конфигурация логирования для бота.
Управление verbose логированием через переменную окружения.
"""
import os
import typing as t

# Уровень логирования (можно изменить в .env)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Возможные уровни: DEBUG, INFO, WARNING, ERROR
VERBOSE_LOGGING = LOG_LEVEL == "DEBUG"
INFO_LOGGING = LOG_LEVEL in ("DEBUG", "INFO")
WARNING_LOGGING = LOG_LEVEL in ("DEBUG", "INFO", "WARNING")

# Переключатели отладки для отдельных стратегий (включаются через .env)
import logging as _logging


def _is_true(val: t.Any) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# Флаги доступные для других модулей
DEBUG_TREND_ENABLED = _is_true(os.getenv("DEBUG_TREND_STRATEGY", ""))
DEBUG_FLAT_ENABLED = _is_true(os.getenv("DEBUG_FLAT_STRATEGY", ""))


def _enable_strategy_debug():
    """Enable DEBUG logging for specific strategy modules via env vars:
    DEBUG_TREND_STRATEGY, DEBUG_FLAT_STRATEGY (values: 1/true/yes).

    Additionally, when either flag is set we force LOG_LEVEL to DEBUG so the
    convenience helpers (should_log/log) will allow debug messages as well.
    """
    global LOG_LEVEL, VERBOSE_LOGGING, INFO_LOGGING

    if DEBUG_TREND_ENABLED or DEBUG_FLAT_ENABLED:
        # Ensure global log level reflects debug intent for helper functions
        LOG_LEVEL = "DEBUG"
        VERBOSE_LOGGING = True
        INFO_LOGGING = True

    # Apply python logging level to strategy-related loggers to help libs using logging
    if DEBUG_TREND_ENABLED:
        _logging.getLogger("bot.strategy").setLevel(_logging.DEBUG)
        _logging.getLogger("bot.live").setLevel(_logging.DEBUG)
        _logging.getLogger("bot.logger_config").debug("DEBUG enabled for TREND strategy via DEBUG_TREND_STRATEGY")

    if DEBUG_FLAT_ENABLED:
        _logging.getLogger("bot.strategy").setLevel(_logging.DEBUG)
        _logging.getLogger("bot.live").setLevel(_logging.DEBUG)
        _logging.getLogger("bot.logger_config").debug("DEBUG enabled for FLAT strategy via DEBUG_FLAT_STRATEGY")


_enable_strategy_debug()

# Отключение определенных категорий логов
DISABLE_WEB_LOGS = os.getenv("DISABLE_WEB_LOGS", "true").lower() == "true"
DISABLE_ML_DETAILS = os.getenv("DISABLE_ML_DETAILS", "true").lower() == "true"


def should_log(category: str, level: str = "INFO") -> bool:
    """
    Определяет, нужно ли логировать сообщение.
    
    Args:
        category: Категория лога ('web', 'ml', 'live', 'trade')
        level: Уровень важности ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        True если нужно логировать
    """
    # Веб-логи отключаем если DISABLE_WEB_LOGS=true
    if category == "web" and DISABLE_WEB_LOGS:
        return level in ("WARNING", "ERROR")
    
    # ML детали отключаем если DISABLE_ML_DETAILS=true
    if category == "ml_details" and DISABLE_ML_DETAILS:
        return False
    
    # Торговые логи всегда показываем
    if category in ("trade", "signal"):
        return True
    
    # Остальные логи зависят от LOG_LEVEL
    if level == "DEBUG":
        return VERBOSE_LOGGING
    elif level == "INFO":
        return INFO_LOGGING
    elif level == "WARNING":
        return WARNING_LOGGING
    else:  # ERROR
        return True


def log(message: str, category: str = "info", level: str = "INFO"):
    """
    Логирование с проверкой уровня.
    
    Args:
        message: Сообщение для лога
        category: Категория лога
        level: Уровень важности
    """
    if should_log(category, level):
        print(message)
