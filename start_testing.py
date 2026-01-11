"""
Скрипт для быстрого запуска тестирования многопарной торговли.
"""
import sys
import os
import subprocess
from pathlib import Path

def check_prerequisites():
    """Проверка необходимых условий для запуска."""
    # Настройка кодировки для Windows консоли
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("ПРОВЕРКА УСЛОВИЙ ДЛЯ ЗАПУСКА")
    print("="*60)
    
    # Проверка Python версии
    if sys.version_info < (3, 8):
        print("[ERROR] Требуется Python 3.8 или выше")
        return False
    print(f"[OK] Python версия: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Проверка .env файла
    env_path = Path(".env")
    if not env_path.exists():
        print("[WARN] .env файл не найден. Используются значения по умолчанию.")
        print("   Рекомендуется создать .env файл на основе env.example")
    else:
        print("[OK] .env файл найден")
    
    # Проверка зависимостей
    try:
        import flask
        print("[OK] Flask установлен")
    except ImportError:
        print("[ERROR] Flask не установлен. Установите: pip install flask")
        return False
    
    try:
        from bot.config import load_settings
        print("[OK] bot.config импортируется")
    except ImportError as e:
        print(f"[ERROR] Ошибка импорта bot.config: {e}")
        return False
    
    try:
        from bot.multi_symbol_manager import MultiSymbolManager
        print("[OK] MultiSymbolManager импортируется")
    except ImportError as e:
        print(f"[ERROR] Ошибка импорта MultiSymbolManager: {e}")
        return False
    
    return True

def run_config_check():
    """Запуск проверки конфигурации."""
    print("\n" + "="*60)
    print("ЗАПУСК ПРОВЕРКИ КОНФИГУРАЦИИ")
    print("="*60)
    
    try:
        # Устанавливаем кодировку UTF-8 для subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, "check_config.py"], 
                               capture_output=True, text=True, 
                               encoding='utf-8', errors='replace',
                               env=env)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Ошибка при запуске check_config.py: {e}")
        return False

def start_web_server():
    """Запуск веб-сервера."""
    print("\n" + "="*60)
    print("ЗАПУСК ВЕБ-СЕРВЕРА")
    print("="*60)
    print("Для остановки нажмите Ctrl+C")
    print("="*60)
    
    try:
        # Импортируем и запускаем веб-сервер
        from bot.web.app import run_web_server
        print("\n[START] Запуск веб-сервера на http://127.0.0.1:5000")
        print("[INFO] Откройте браузер и перейдите на http://localhost:5000")
        print("[INFO] Логин и пароль указаны в .env (ADMIN_USERNAME, ADMIN_PASSWORD)")
        print("\nДля остановки нажмите Ctrl+C\n")
        
        # Запускаем в режиме отладки для тестирования
        run_web_server(host="127.0.0.1", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n[STOP] Веб-сервер остановлен пользователем")
    except Exception as e:
        print(f"\n[ERROR] Ошибка при запуске веб-сервера: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Основная функция."""
    print("="*60)
    print("ЗАПУСК ТЕСТИРОВАНИЯ МНОГОПАРНОЙ ТОРГОВЛИ")
    print("="*60)
    
    # Проверка условий
    if not check_prerequisites():
        print("\n[ERROR] Не все условия выполнены. Исправьте ошибки и попробуйте снова.")
        return 1
    
    # Запуск проверки конфигурации
    if not run_config_check():
        print("\n[WARN] Проверка конфигурации выявила проблемы.")
        try:
            response = input("\nПродолжить запуск веб-сервера? (y/n): ")
            if response.lower() != 'y':
                print("Запуск отменен.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nЗапуск отменен.")
            return 1
    
    # Запуск веб-сервера
    try:
        start_web_server()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
