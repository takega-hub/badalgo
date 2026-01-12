#!/bin/bash
# Скрипт для создания файлов деплоя на сервере
# Запустите: bash create_deploy_files.sh

cat > /opt/crypto_bot/deploy.sh << 'DEPLOY_EOF'
#!/bin/bash
# Скрипт для деплоя crypto_bot на сервер через GitHub

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="/opt/crypto_bot"
REPO_URL="https://github.com/takega-hub/badalgo.git"
BRANCH="main"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"

echo -e "${GREEN}=== Деплой Crypto Bot ===${NC}"

if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Ошибка: Запустите скрипт с правами root (sudo)${NC}"
    exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Создание директории проекта: $PROJECT_DIR${NC}"
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Клонирование репозитория...${NC}"
    git clone -b "$BRANCH" "$REPO_URL" .
else
    echo -e "${YELLOW}Обновление кода из репозитория...${NC}"
    git fetch origin
    git reset --hard origin/"$BRANCH"
    git clean -fd
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Создание виртуального окружения...${NC}"
    python3 -m venv "$VENV_DIR"
fi

echo -e "${YELLOW}Обновление зависимостей...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}Создание необходимых директорий...${NC}"
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/ml_models"
mkdir -p "$PROJECT_DIR/ml_data"
mkdir -p /tmp/numba_cache
chmod 755 /tmp/numba_cache

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Создание .env файла из примера...${NC}"
    if [ -f "$PROJECT_DIR/env.example" ]; then
        cp "$PROJECT_DIR/env.example" "$PROJECT_DIR/.env"
        echo -e "${RED}ВАЖНО: Отредактируйте файл .env и заполните все необходимые переменные!${NC}"
    fi
fi

chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR" 2>/dev/null || true
chmod +x "$PROJECT_DIR/deploy.sh" 2>/dev/null || true
chmod +x "$PROJECT_DIR/update.sh" 2>/dev/null || true

if [ -f "$PROJECT_DIR/supervisor/crypto-bot.conf" ]; then
    echo -e "${YELLOW}Обновление конфигурации supervisor...${NC}"
    USERNAME=$(whoami)
    sed "s|/opt/crypto_bot|$PROJECT_DIR|g; s|user=your_username|user=$USERNAME|g" \
        "$PROJECT_DIR/supervisor/crypto-bot.conf" > /etc/supervisor/conf.d/crypto-bot.conf
    supervisorctl reread
    supervisorctl update
    echo -e "${GREEN}Конфигурация supervisor обновлена${NC}"
fi

echo -e "${GREEN}=== Деплой завершен ===${NC}"
DEPLOY_EOF

cat > /opt/crypto_bot/update.sh << 'UPDATE_EOF'
#!/bin/bash
# Скрипт для обновления crypto_bot с GitHub

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="/opt/crypto_bot"
BRANCH="main"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${GREEN}=== Обновление Crypto Bot ===${NC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Ошибка: Проект не найден в $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

echo -e "${YELLOW}Остановка бота...${NC}"
supervisorctl stop crypto-bot || echo "Бот не запущен"

echo -e "${YELLOW}Обновление кода из GitHub...${NC}"
git fetch origin
git reset --hard origin/"$BRANCH"
git clean -fd

echo -e "${YELLOW}Обновление зависимостей...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

if [ -f "$PROJECT_DIR/supervisor/crypto-bot.conf" ]; then
    echo -e "${YELLOW}Проверка конфигурации supervisor...${NC}"
    USERNAME=$(whoami)
    sed "s|/opt/crypto_bot|$PROJECT_DIR|g; s|user=your_username|user=$USERNAME|g" \
        "$PROJECT_DIR/supervisor/crypto-bot.conf" > /etc/supervisor/conf.d/crypto-bot.conf
    supervisorctl reread
    supervisorctl update
fi

echo -e "${YELLOW}Запуск бота...${NC}"
supervisorctl start crypto-bot

sleep 2
supervisorctl status crypto-bot

echo -e "${GREEN}=== Обновление завершено ===${NC}"
UPDATE_EOF

chmod +x /opt/crypto_bot/deploy.sh
chmod +x /opt/crypto_bot/update.sh

echo "Файлы созданы:"
echo "  - /opt/crypto_bot/deploy.sh"
echo "  - /opt/crypto_bot/update.sh"
