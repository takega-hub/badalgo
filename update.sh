#!/bin/bash
# Скрипт для обновления crypto_bot с GitHub

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Конфигурация
PROJECT_DIR="/opt/crypto_bot"
BRANCH="main"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${GREEN}=== Обновление Crypto Bot ===${NC}"

# Проверка существования проекта
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Ошибка: Проект не найден в $PROJECT_DIR${NC}"
    echo -e "${YELLOW}Запустите deploy.sh для первоначальной установки${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

# Остановка бота перед обновлением
echo -e "${YELLOW}Остановка бота...${NC}"
supervisorctl stop crypto-bot || echo "Бот не запущен или supervisor не настроен"

# Обновление кода из репозитория
echo -e "${YELLOW}Обновление кода из GitHub...${NC}"
git fetch origin
git reset --hard origin/"$BRANCH"
git clean -fd

# Обновление зависимостей
echo -e "${YELLOW}Обновление зависимостей...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Обновление конфигурации supervisor, если изменилась
if [ -f "$PROJECT_DIR/supervisor/crypto-bot.conf" ]; then
    echo -e "${YELLOW}Проверка конфигурации supervisor...${NC}"
    cp "$PROJECT_DIR/supervisor/crypto-bot.conf" /etc/supervisor/conf.d/crypto-bot.conf
    supervisorctl reread
    supervisorctl update
fi

# Запуск бота
echo -e "${YELLOW}Запуск бота...${NC}"
supervisorctl start crypto-bot

# Проверка статуса
sleep 2
supervisorctl status crypto-bot

echo -e "${GREEN}=== Обновление завершено ===${NC}"
echo -e "${YELLOW}Просмотр логов: tail -f $PROJECT_DIR/logs/crypto-bot.log${NC}"
