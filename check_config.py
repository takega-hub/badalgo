"""
Скрипт для проверки конфигурации перед реальным тестированием.
"""
import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Настройка кодировки для Windows консоли
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Загружаем переменные окружения
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print("[OK] .env файл найден и загружен")
else:
    print("[WARN] .env файл не найден. Используются значения по умолчанию.")

def check_required_settings():
    """Проверка необходимых настроек."""
    print("\n" + "="*60)
    print("ПРОВЕРКА НЕОБХОДИМЫХ НАСТРОЕК")
    print("="*60)
    
    required = {
        'BYBIT_API_KEY': 'API ключ Bybit',
        'BYBIT_API_SECRET': 'API секрет Bybit',
        'BYBIT_BASE_URL': 'Базовый URL Bybit',
    }
    
    all_ok = True
    for key, description in required.items():
        value = os.getenv(key, '')
        if value and value not in ['your_bybit_api_key_here', 'your_bybit_api_secret_here']:
            print(f"[OK] {key}: {'*' * 10} (установлен)")
        else:
            print(f"[ERROR] {key}: НЕ УСТАНОВЛЕН ({description})")
            all_ok = False
    
    # Проверяем базовый URL
    base_url = os.getenv('BYBIT_BASE_URL', '')
    if 'testnet' in base_url.lower():
        print(f"[INFO] Используется TESTNET: {base_url}")
    elif 'api.bybit.com' in base_url.lower():
        print(f"[WARN] Используется PRODUCTION: {base_url}")
    
    return all_ok

def check_symbol_settings():
    """Проверка настроек символов."""
    print("\n" + "="*60)
    print("ПРОВЕРКА НАСТРОЕК СИМВОЛОВ")
    print("="*60)
    
    # Проверяем ACTIVE_SYMBOLS
    active_symbols_str = os.getenv('ACTIVE_SYMBOLS', '')
    if active_symbols_str:
        active_symbols = [s.strip().upper() for s in active_symbols_str.split(',')]
        print(f"[OK] ACTIVE_SYMBOLS: {active_symbols}")
    else:
        # Проверяем TRADING_SYMBOL для обратной совместимости
        trading_symbol = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
        print(f"[INFO] ACTIVE_SYMBOLS не установлен, используется TRADING_SYMBOL: {trading_symbol}")
        active_symbols = [trading_symbol]
    
    # Проверяем PRIMARY_SYMBOL
    primary_symbol = os.getenv('PRIMARY_SYMBOL', '')
    if primary_symbol:
        print(f"[OK] PRIMARY_SYMBOL: {primary_symbol}")
    else:
        print(f"[INFO] PRIMARY_SYMBOL не установлен, будет использован первый активный символ")
    
    available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    valid_symbols = [s for s in active_symbols if s in available_symbols]
    invalid_symbols = [s for s in active_symbols if s not in available_symbols]
    
    if invalid_symbols:
        print(f"[WARN] Невалидные символы будут проигнорированы: {invalid_symbols}")
    
    if valid_symbols:
        print(f"[OK] Валидные активные символы: {valid_symbols}")
    else:
        print(f"[ERROR] Нет валидных активных символов!")
        return False
    
    return True

def check_ml_models():
    """Проверка наличия ML моделей."""
    print("\n" + "="*60)
    print("ПРОВЕРКА ML МОДЕЛЕЙ")
    print("="*60)
    
    models_dir = Path(__file__).parent / "ml_models"
    if not models_dir.exists():
        print(f"[ERROR] Директория ml_models не найдена!")
        return False
    
    available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    models_found = {}
    
    for symbol in available_symbols:
        # Ищем модели для символа
        xgb_model = models_dir / f"xgb_{symbol}_15.pkl"
        rf_model = models_dir / f"rf_{symbol}_15.pkl"
        
        found = []
        if xgb_model.exists():
            found.append("xgb")
        if rf_model.exists():
            found.append("rf")
        
        models_found[symbol] = found
        
        if found:
            print(f"[OK] {symbol}: найдены модели {', '.join(found)}")
        else:
            print(f"[WARN] {symbol}: модели не найдены")
    
    return True

def check_config_import():
    """Проверка импорта конфигурации."""
    print("\n" + "="*60)
    print("ПРОВЕРКА ЗАГРУЗКИ КОНФИГУРАЦИИ")
    print("="*60)
    
    try:
        from bot.config import load_settings, AppSettings
        print("[OK] Импорт bot.config успешен")
        
        try:
            settings = load_settings()
            print("[OK] Загрузка настроек успешна")
            print(f"   - active_symbols: {settings.active_symbols}")
            print(f"   - primary_symbol: {settings.primary_symbol}")
            print(f"   - enable_trend_strategy: {settings.enable_trend_strategy}")
            print(f"   - enable_flat_strategy: {settings.enable_flat_strategy}")
            print(f"   - enable_ml_strategy: {settings.enable_ml_strategy}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке настроек: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"[ERROR] Ошибка импорта: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_multi_symbol_manager():
    """Проверка MultiSymbolManager."""
    print("\n" + "="*60)
    print("ПРОВЕРКА MULTISYMBOLMANAGER")
    print("="*60)
    
    try:
        from bot.config import load_settings, AppSettings
        from bot.multi_symbol_manager import MultiSymbolManager
        
        settings = load_settings()
        print(f"[OK] Загружены настройки: active_symbols={settings.active_symbols}")
        
        manager = MultiSymbolManager(settings)
        print("[OK] MultiSymbolManager создан успешно")
        
        status = manager.get_status()
        print(f"[OK] Статус менеджера:")
        print(f"   - running: {status['running']}")
        print(f"   - active_symbols: {status['active_symbols']}")
        print(f"   - workers: {len(status['workers'])}")
        
        workers_status = manager.get_all_workers_status()
        print(f"[OK] Статусы воркеров: {len(workers_status)} символов")
        for symbol, status in workers_status.items():
            print(f"   - {symbol}: {status['current_status']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке MultiSymbolManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция проверки."""
    print("="*60)
    print("ПРОВЕРКА КОНФИГУРАЦИИ ПЕРЕД РЕАЛЬНЫМ ТЕСТИРОВАНИЕМ")
    print("="*60)
    
    checks = [
        ("Необходимые настройки", check_required_settings),
        ("Настройки символов", check_symbol_settings),
        ("ML модели", check_ml_models),
        ("Загрузка конфигурации", check_config_import),
        ("MultiSymbolManager", check_multi_symbol_manager),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Критическая ошибка в проверке '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ПРОВЕРКИ")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[OK] ПРОЙДЕН" if result else "[ERROR] НЕ ПРОЙДЕН"
        print(f"{status}: {name}")
    
    print(f"\nВсего проверок: {total}")
    print(f"Пройдено: {passed}")
    print(f"Не пройдено: {total - passed}")
    
    if passed == total:
        print("\n[SUCCESS] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! ГОТОВО К ЗАПУСКУ!")
        print("\nДля запуска веб-сервера выполните:")
        print("  python main.py --mode web")
        print("\nИли через Flask напрямую:")
        print("  python -m bot.web.app")
        return 0
    else:
        print(f"\n[WARN] НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ ({total - passed} из {total})")
        print("Пожалуйста, исправьте ошибки перед запуском.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
