"""
Модуль для хранения общих настроек, доступных из разных потоков.
"""
from typing import Optional
from bot.config import AppSettings

# Глобальный объект настроек
_global_settings: Optional[AppSettings] = None


def get_settings() -> Optional[AppSettings]:
    """Получить текущие настройки."""
    return _global_settings


def set_settings(new_settings: AppSettings) -> None:
    """Установить новые настройки."""
    global _global_settings
    _global_settings = new_settings


def reload_settings() -> Optional[AppSettings]:
    """Перезагрузить настройки из .env."""
    from bot.config import load_settings
    global _global_settings
    _global_settings = load_settings()
    return _global_settings




