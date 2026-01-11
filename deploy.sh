#!/bin/bash
# Скрипт для деплоя crypto_bot на сервер через GitHub

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Конфигурация
PROJECT_DIR="/opt/crypto_bot"
REPO_URL="https://github.com/takega-hub/badalgo.git"
BRANCH="main"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"

echo -e "${GREEN}=== Деплой Crypto Bot ===${NC}"

# Проверка прав root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Ошибка: Запустите скрипт с правами root (sudo)${NC}"
    exit 1
fi

# Создание директории проекта, если не существует
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Создание директории проекта: $PROJECT_DIR${NC}"
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Клонирование или обновление репозитория
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Клонирование репозитория...${NC}"
    git clone -b "$BRANCH" "$REPO_URL" .
else
    echo -e "${YELLOW}Обновление кода из репозитория...${NC}"
    git fetch origin
    git reset --hard origin/"$BRANCH"
    git clean -fd
fi

# Создание виртуального окружения, если не существует
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Создание виртуального окружения...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Активация виртуального окружения и обновление зависимостей
echo -e "${YELLOW}Обновление зависимостей...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Создание необходимых директорий
echo -e "${YELLOW}Создание необходимых директорий...${NC}"
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/ml_models"
mkdir -p "$PROJECT_DIR/ml_data"
mkdir -p /tmp/numba_cache
chmod 755 /tmp/numba_cache

# Проверка .env файла
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Создание .env файла из примера...${NC}"
    if [ -f "$PROJECT_DIR/env.example" ]; then
        cp "$PROJECT_DIR/env.example" "$PROJECT_DIR/.env"
        echo -e "${RED}ВАЖНО: Отредактируйте файл .env и заполните все необходимые переменные!${NC}"
    else
        echo -e "${RED}ВНИМАНИЕ: Файл env.example не найден. Создайте .env вручную!${NC}"
    fi
else
    echo -e "${GREEN}.env файл уже существует${NC}"
fi

# Установка прав на файлы
echo -e "${YELLOW}Установка прав доступа...${NC}"
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR"
chmod +x "$PROJECT_DIR/deploy.sh"
chmod +x "$PROJECT_DIR/update.sh"

# Обновление конфигурации supervisor
if [ -f "$PROJECT_DIR/supervisor/crypto-bot.conf" ]; then
    echo -e "${YELLOW}Обновление конфигурации supervisor...${NC}"
    
    # Замена пути и пользователя в конфигурации
    sed -i "s|/opt/crypto_bot|$PROJECT_DIR|g" "$PROJECT_DIR/supervisor/crypto-bot.conf"
    sed -i "s|user=your_username|user=$SUDO_USER|g" "$PROJECT_DIR/supervisor/crypto-bot.conf"
    
    # Копирование конфигурации в supervisor
    cp "$PROJECT_DIR/supervisor/crypto-bot.conf" /etc/supervisor/conf.d/crypto-bot.conf
    
    # Перезагрузка supervisor
    supervisorctl reread
    supervisorctl update
    echo -e "${GREEN}Конфигурация supervisor обновлена${NC}"
else
    echo -e "${YELLOW}Конфигурация supervisor не найдена, пропускаем...${NC}"
fi

echo -e "${GREEN}=== Деплой завершен ===${NC}"
echo -e "${YELLOW}Следующие шаги:${NC}"
echo -e "1. Отредактируйте .env файл: nano $PROJECT_DIR/.env"
echo -e "2. Запустите бота: supervisorctl start crypto-bot"
echo -e "3. Проверьте статус: supervisorctl status crypto-bot"
echo -e "4. Просмотр логов: tail -f $LOG_DIR/crypto-bot.log"
