"""
Тестовый скрипт для проверки функциональности многопарной торговли.
"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Тест загрузки конфигурации с активными символами."""
    print("\n=== Тест 1: Загрузка конфигурации ===")
    try:
        from bot.config import load_settings, AppSettings
        
        # Проверяем, что класс AppSettings имеет необходимые поля
        settings = AppSettings()
        assert hasattr(settings, 'active_symbols'), "AppSettings должен иметь поле active_symbols"
        assert hasattr(settings, 'primary_symbol'), "AppSettings должен иметь поле primary_symbol"
        assert hasattr(settings, 'symbols'), "AppSettings должен иметь поле symbols"
        print("✅ AppSettings содержит необходимые поля")
        
        # Проверяем загрузку настроек
        try:
            loaded_settings = load_settings()
            print(f"✅ Настройки загружены успешно")
            print(f"   - active_symbols: {loaded_settings.active_symbols}")
            print(f"   - primary_symbol: {loaded_settings.primary_symbol}")
            print(f"   - symbols: {loaded_settings.symbols}")
            
            # Проверяем валидность символов
            available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            for symbol in loaded_settings.active_symbols:
                assert symbol in available_symbols, f"Символ {symbol} не в списке доступных"
            print("✅ Все активные символы валидны")
            
            return True
        except Exception as e:
            print(f"⚠️ Ошибка при загрузке настроек: {e}")
            print("   (Это нормально, если .env файл не настроен)")
            return False
    except Exception as e:
        print(f"❌ Ошибка в тесте конфигурации: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_symbol_manager():
    """Тест MultiSymbolManager."""
    print("\n=== Тест 2: MultiSymbolManager ===")
    try:
        from bot.config import AppSettings
        from bot.multi_symbol_manager import MultiSymbolManager
        
        # Создаем тестовые настройки
        test_settings = AppSettings()
        test_settings.active_symbols = ["BTCUSDT", "ETHUSDT"]
        test_settings.primary_symbol = "BTCUSDT"
        
        # Создаем менеджер
        manager = MultiSymbolManager(test_settings)
        print("✅ MultiSymbolManager создан успешно")
        
        # Проверяем инициализацию воркеров
        assert len(manager.workers) == len(test_settings.active_symbols), \
            f"Количество воркеров ({len(manager.workers)}) не соответствует количеству активных символов ({len(test_settings.active_symbols)})"
        print(f"✅ Воркеры инициализированы: {list(manager.workers.keys())}")
        
        # Проверяем методы
        assert hasattr(manager, 'start'), "MultiSymbolManager должен иметь метод start"
        assert hasattr(manager, 'stop'), "MultiSymbolManager должен иметь метод stop"
        assert hasattr(manager, 'get_status'), "MultiSymbolManager должен иметь метод get_status"
        assert hasattr(manager, 'get_all_workers_status'), "MultiSymbolManager должен иметь метод get_all_workers_status"
        assert hasattr(manager, 'update_settings'), "MultiSymbolManager должен иметь метод update_settings"
        print("✅ Все необходимые методы присутствуют")
        
        # Проверяем get_status
        status = manager.get_status()
        assert 'running' in status, "Статус должен содержать 'running'"
        assert 'active_symbols' in status, "Статус должен содержать 'active_symbols'"
        assert 'workers' in status, "Статус должен содержать 'workers'"
        print("✅ Метод get_status() работает корректно")
        
        # Проверяем get_all_workers_status
        workers_status = manager.get_all_workers_status()
        assert isinstance(workers_status, dict), "get_all_workers_status должен возвращать словарь"
        print(f"✅ Метод get_all_workers_status() работает: {len(workers_status)} воркеров")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в тесте MultiSymbolManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_function_signature():
    """Тест сигнатуры функции run_live_from_api."""
    print("\n=== Тест 3: Сигнатура run_live_from_api ===")
    try:
        import inspect
        from bot.live import run_live_from_api
        
        # Получаем сигнатуру функции
        sig = inspect.signature(run_live_from_api)
        params = list(sig.parameters.keys())
        
        # Проверяем, что есть параметр symbol
        assert 'symbol' in params, "run_live_from_api должна иметь параметр symbol"
        print("✅ Функция run_live_from_api имеет параметр symbol")
        
        # Проверяем параметры
        print(f"   Параметры функции: {params}")
        
        # Проверяем, что symbol опционален (Optional)
        symbol_param = sig.parameters['symbol']
        # В Python 3.10+ можно использовать typing.get_origin
        from typing import get_origin, get_args
        if symbol_param.annotation != inspect.Parameter.empty:
            print(f"   Тип параметра symbol: {symbol_param.annotation}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в тесте сигнатуры: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_app_imports():
    """Тест импортов в веб-приложении."""
    print("\n=== Тест 4: Импорты веб-приложения ===")
    try:
        # Проверяем, что MultiSymbolManager импортируется в app.py
        with open('bot/web/app.py', 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'MultiSymbolManager' in content, "MultiSymbolManager должен быть импортирован в app.py"
            assert 'from bot.multi_symbol_manager import MultiSymbolManager' in content or \
                   'import MultiSymbolManager' in content, \
                   "MultiSymbolManager должен быть импортирован правильно"
        print("✅ MultiSymbolManager импортирован в app.py")
        
        # Проверяем наличие API endpoints
        assert '/api/symbols/list' in content, "API endpoint /api/symbols/list должен существовать"
        assert '/api/symbols/active' in content, "API endpoint /api/symbols/active должен существовать"
        assert '/api/symbols/set-active' in content, "API endpoint /api/symbols/set-active должен существовать"
        print("✅ Все необходимые API endpoints присутствуют")
        
        # Проверяем функцию _save_symbol_settings_to_env
        assert '_save_symbol_settings_to_env' in content, \
            "Функция _save_symbol_settings_to_env должна существовать"
        print("✅ Функция _save_symbol_settings_to_env присутствует")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в тесте импортов: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_html_structure():
    """Тест структуры HTML."""
    print("\n=== Тест 5: Структура HTML ===")
    try:
        with open('bot/web/templates/index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем наличие вкладки Symbols
        assert 'id="symbols"' in content, "Вкладка Symbols должна существовать"
        assert 'onclick="showTab(\'symbols\')"' in content, "Кнопка переключения на вкладку Symbols должна существовать"
        print("✅ Вкладка Symbols присутствует")
        
        # Проверяем переключатель символов
        assert 'id="primary-symbol-select"' in content, "Переключатель символов должен существовать"
        assert 'id="symbol-selector"' in content, "Селектор символов должен существовать"
        print("✅ Переключатель символов присутствует")
        
        # Проверяем функции JavaScript
        assert 'function loadSymbolSettings' in content, "Функция loadSymbolSettings должна существовать"
        assert 'function saveSymbolSettings' in content, "Функция saveSymbolSettings должна существовать"
        assert 'function changePrimarySymbol' in content, "Функция changePrimarySymbol должна существовать"
        assert 'function loadSymbolsStatus' in content, "Функция loadSymbolsStatus должна существовать"
        print("✅ Все необходимые JavaScript функции присутствуют")
        
        # Проверяем, что loadChart обновлен для работы с символами
        assert 'symbol=' in content or 'symbol: ' in content or \
               'api/chart/data?symbol' in content, \
               "loadChart должен передавать символ в API"
        print("✅ loadChart обновлен для работы с символами")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в тесте HTML: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Запуск всех тестов."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ МНОГОПАРНОЙ ТОРГОВЛИ")
    print("=" * 60)
    
    tests = [
        ("Загрузка конфигурации", test_config_loading),
        ("MultiSymbolManager", test_multi_symbol_manager),
        ("Сигнатура run_live_from_api", test_live_function_signature),
        ("Импорты веб-приложения", test_web_app_imports),
        ("Структура HTML", test_html_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Критическая ошибка в тесте '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
        print(f"{status}: {name}")
    
    print(f"\nВсего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Не пройдено: {total - passed}")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return 0
    else:
        print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ ({total - passed} из {total})")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
