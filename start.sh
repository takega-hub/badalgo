#!/bin/bash

# Скрипт запуска Crypto Trading Bot
# Использование: ./start.sh [--host HOST] [--port PORT]

# Получение директории скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Параметры по умолчанию
HOST="0.0.0.0"
PORT=5000

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Неизвестный параметр: $1"
            echo "Использование: $0 [--host HOST] [--port PORT]"
            exit 1
            ;;
    esac
done

# Проверка наличия виртуального окружения
if [ ! -d "venv" ]; then
    echo "Ошибка: виртуальное окружение не найдено!"
    echo "Создайте его командой: python3 -m venv venv"
    exit 1
fi

# Активация виртуального окружения
source venv/bin/activate

# Исправление проблемы с Numba кэшированием в pandas_ta
export NUMBA_CACHE_DIR="/tmp/numba_cache"
mkdir -p /tmp/numba_cache 2>/dev/null || true

# Проверка наличия .env файла
if [ ! -f ".env" ]; then
    echo "Предупреждение: файл .env не найден!"
    echo "Создайте его из .env.example: cp .env.example .env"
    echo "Заполните все необходимые переменные перед запуском."
    read -p "Продолжить запуск? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Проверка установки зависимостей
if ! python -c "import flask" 2>/dev/null; then
    echo "Установка зависимостей..."
    pip install -r requirements.txt
fi

# Запуск бота
echo "Запуск Crypto Trading Bot..."
echo "Админ-панель будет доступна на: http://$HOST:$PORT"
echo "Для остановки нажмите Ctrl+C"
echo ""

python main.py --mode web --web-host "$HOST" --web-port "$PORT"
