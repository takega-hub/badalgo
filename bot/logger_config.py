"""
Конфигурация логирования для бота.
Управление verbose логированием через переменную окружения.
"""
import os

# Уровень логирования (можно изменить в .env)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Возможные уровни: DEBUG, INFO, WARNING, ERROR
VERBOSE_LOGGING = LOG_LEVEL == "DEBUG"
INFO_LOGGING = LOG_LEVEL in ("DEBUG", "INFO")
WARNING_LOGGING = LOG_LEVEL in ("DEBUG", "INFO", "WARNING")

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
