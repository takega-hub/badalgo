#!/bin/bash
# Скрипт для диагностики и исправления проблем с supervisor

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="/opt/crypto_bot"
CONF_FILE="/etc/supervisor/conf.d/crypto-bot.conf"

echo -e "${GREEN}=== Диагностика Supervisor ===${NC}"

# 1. Проверка файлов
echo -e "${YELLOW}1. Проверка файлов...${NC}"
if [ ! -f "$PROJECT_DIR/venv/bin/python" ]; then
    echo -e "${RED}ОШИБКА: Python в venv не найден!${NC}"
    echo "Создание виртуального окружения..."
    cd "$PROJECT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ Python найден${NC}"
fi

if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo -e "${RED}ОШИБКА: main.py не найден!${NC}"
    exit 1
else
    echo -e "${GREEN}✓ main.py найден${NC}"
fi

# 2. Создание директорий
echo -e "${YELLOW}2. Создание директорий...${NC}"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p /tmp/numba_cache
chmod 755 /tmp/numba_cache
chmod 755 "$PROJECT_DIR/logs"
echo -e "${GREEN}✓ Директории созданы${NC}"

# 3. Проверка конфигурации
echo -e "${YELLOW}3. Проверка конфигурации...${NC}"
if [ ! -f "$CONF_FILE" ]; then
    echo -e "${RED}ОШИБКА: Конфигурация не найдена!${NC}"
    if [ -f "$PROJECT_DIR/supervisor/crypto-bot.conf" ]; then
        echo "Создание конфигурации из шаблона..."
        USERNAME=$(whoami)
        sed "s|/opt/crypto_bot|$PROJECT_DIR|g; s|user=your_username|user=$USERNAME|g" \
            "$PROJECT_DIR/supervisor/crypto-bot.conf" > "$CONF_FILE"
        echo -e "${GREEN}✓ Конфигурация создана${NC}"
    else
        echo -e "${RED}Шаблон конфигурации не найден!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Конфигурация существует${NC}"
fi

# 4. Проверка путей в конфигурации
echo -e "${YELLOW}4. Проверка путей в конфигурации...${NC}"
PYTHON_PATH=$(grep "^command=" "$CONF_FILE" | cut -d' ' -f1 | sed 's|command=||')
if [ -n "$PYTHON_PATH" ] && [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}ОШИБКА: Python путь в конфигурации неверный: $PYTHON_PATH${NC}"
    echo "Исправление конфигурации..."
    USERNAME=$(whoami)
    cat > "$CONF_FILE" << EOF
[program:crypto-bot]
command=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/main.py --mode web --web-host 0.0.0.0 --web-port 5000
directory=$PROJECT_DIR
user=$USERNAME
autostart=true
autorestart=true
startretries=3
startsecs=10
stdout_logfile=$PROJECT_DIR/logs/crypto-bot.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stderr_logfile=$PROJECT_DIR/logs/crypto-bot-error.log
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=10
environment=PATH="$PROJECT_DIR/venv/bin:%(ENV_PATH)s",NUMBA_CACHE_DIR="/tmp/numba_cache"
stopasgroup=true
killasgroup=true
stopwaitsecs=10
EOF
    echo -e "${GREEN}✓ Конфигурация исправлена${NC}"
fi

# 5. Проверка пользователя
echo -e "${YELLOW}5. Проверка пользователя...${NC}"
CONF_USER=$(grep "^user=" "$CONF_FILE" | cut -d'=' -f2)
CURRENT_USER=$(whoami)
if [ "$CONF_USER" != "$CURRENT_USER" ] && [ "$CONF_USER" != "root" ]; then
    echo -e "${YELLOW}Пользователь в конфигурации ($CONF_USER) отличается от текущего ($CURRENT_USER)${NC}"
    echo "Исправление..."
    sed -i "s|^user=.*|user=$CURRENT_USER|g" "$CONF_FILE"
    echo -e "${GREEN}✓ Пользователь исправлен${NC}"
fi

# 6. Тест ручного запуска
echo -e "${YELLOW}6. Тест ручного запуска...${NC}"
cd "$PROJECT_DIR"
source venv/bin/activate
export NUMBA_CACHE_DIR="/tmp/numba_cache"
timeout 3 python main.py --mode web --web-host 127.0.0.1 --web-port 5001 > /dev/null 2>&1 &
TEST_PID=$!
sleep 2
if ps -p $TEST_PID > /dev/null 2>&1; then
    kill $TEST_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Ручной запуск успешен${NC}"
else
    echo -e "${RED}ОШИБКА: Ручной запуск не удался!${NC}"
    echo "Проверьте .env файл и зависимости"
fi

# 7. Обновление supervisor
echo -e "${YELLOW}7. Обновление supervisor...${NC}"
supervisorctl reread
supervisorctl update

echo -e "${GREEN}=== Диагностика завершена ===${NC}"
echo ""
echo "Попробуйте запустить:"
echo "  sudo supervisorctl start crypto-bot"
echo ""
echo "Проверьте логи:"
echo "  tail -f $PROJECT_DIR/logs/crypto-bot-error.log"
