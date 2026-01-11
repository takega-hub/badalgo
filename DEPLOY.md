# Инструкция по деплою на сервер

## Подготовка сервера

### 1. Требования к системе
- Ubuntu 20.04+ / Debian 11+ или другая Linux-система
- Python 3.10+
- pip
- systemd (для запуска как службы)

### 2. Установка зависимостей системы

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python и pip (если не установлены)
sudo apt install -y python3 python3-pip python3-venv git

# Установка зависимостей для ML библиотек (если нужны)
sudo apt install -y build-essential python3-dev
```

### 3. Клонирование проекта (если используется Git)

```bash
cd /opt  # или другая директория
sudo git clone <your-repo-url> crypto_bot
sudo chown -R $USER:$USER crypto_bot
cd crypto_bot
```

Или загрузите проект на сервер другим способом (scp, rsync, и т.д.)

## Установка приложения

### 1. Создание виртуального окружения

```bash
cd /opt/crypto_bot  # или путь к проекту
python3 -m venv venv
source venv/bin/activate
```

### 2. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Все зависимости уже включены в requirements.txt
# waitress - для кроссплатформенного запуска (Windows/Linux)
# gunicorn - для Linux (опционально, лучше для продакшена)
# numba - для pandas_ta (уже включен)

# Исправление проблемы с Numba кэшированием (автоматически обрабатывается в коде)
# Если возникает ошибка "cannot cache function", установите:
export NUMBA_CACHE_DIR="/tmp/numba_cache"
mkdir -p /tmp/numba_cache
```

### 3. Создание .env файла

```bash
cp .env.example .env
nano .env  # или другой редактор
```

**ВАЖНО:** Заполните все необходимые переменные:
- `BYBIT_API_KEY` и `BYBIT_API_SECRET` - ваши API ключи Bybit
- `BYBIT_BASE_URL` - URL Bybit (testnet или production)
- `ADMIN_USERNAME` и `ADMIN_PASSWORD` - логин и пароль для админ-панели
- `FLASK_SECRET_KEY` - случайный секретный ключ для Flask сессий
- Остальные настройки по необходимости

### 4. Проверка структуры директорий

Убедитесь, что существуют необходимые директории:

```bash
mkdir -p ml_models ml_data
# Если есть ML модели, скопируйте их в ml_models/
```

## Настройка systemd службы

### 1. Создание service файла

```bash
sudo nano /etc/systemd/system/crypto-bot.service
```

Содержимое файла:

```ini
[Unit]
Description=Crypto Trading Bot with Web Admin Panel
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/opt/crypto_bot
Environment="PATH=/opt/crypto_bot/venv/bin"
# Вариант 1: Использование встроенного запуска (с waitress)
ExecStart=/opt/crypto_bot/venv/bin/python main.py --mode web --web-host 0.0.0.0 --web-port 5000

# Вариант 2: Использование gunicorn напрямую (только Linux, лучше для продакшена)
# ExecStart=/opt/crypto_bot/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 --access-logfile - --error-logfile - 'bot.web.app:app'
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Ограничения ресурсов (опционально)
# LimitNOFILE=65536
# MemoryLimit=2G

[Install]
WantedBy=multi-user.target
```

**ВАЖНО:** Замените:
- `your_username` - на ваше имя пользователя на сервере
- `/opt/crypto_bot` - на реальный путь к проекту

### 2. Загрузка и запуск службы

```bash
# Перезагрузка systemd
sudo systemctl daemon-reload

# Включение автозапуска при загрузке системы
sudo systemctl enable crypto-bot

# Запуск службы
sudo systemctl start crypto-bot

# Проверка статуса
sudo systemctl status crypto-bot

# Просмотр логов
sudo journalctl -u crypto-bot -f
```

## Настройка Nginx (опционально, для проксирования)

Если хотите использовать Nginx как reverse proxy с SSL:

### 1. Установка Nginx

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### 2. Создание конфигурации

```bash
sudo nano /etc/nginx/sites-available/crypto-bot
```

Содержимое:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Замените на ваш домен

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (если нужно в будущем)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. Активация конфигурации

```bash
sudo ln -s /etc/nginx/sites-available/crypto-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Настройка SSL (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com
```

## Настройка файрвола

```bash
# Если используется UFW
sudo ufw allow 5000/tcp
# Или для Nginx
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Управление службой

```bash
# Остановка
sudo systemctl stop crypto-bot

# Запуск
sudo systemctl start crypto-bot

# Перезапуск
sudo systemctl restart crypto-bot

# Статус
sudo systemctl status crypto-bot

# Просмотр логов
sudo journalctl -u crypto-bot -f
sudo journalctl -u crypto-bot -n 100  # Последние 100 строк
```

## Обновление приложения

```bash
cd /opt/crypto_bot

# Остановка службы
sudo systemctl stop crypto-bot

# Обновление кода (если используется Git)
git pull

# Обновление зависимостей
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Запуск службы
sudo systemctl start crypto-bot
```

## Мониторинг и логирование

### Просмотр логов в реальном времени

```bash
sudo journalctl -u crypto-bot -f
```

### Просмотр ошибок

```bash
sudo journalctl -u crypto-bot -p err
```

### Настройка ротации логов

Логи systemd автоматически ротируются. Для кастомных логов можно настроить logrotate.

## Безопасность

1. **Измените пароль администратора** через веб-интерфейс после первого входа
2. **Используйте сильный FLASK_SECRET_KEY** (можно сгенерировать: `python -c "import secrets; print(secrets.token_hex(32))"`)
3. **Настройте файрвол** для ограничения доступа к порту 5000 (только с доверенных IP или через Nginx)
4. **Используйте HTTPS** через Nginx с Let's Encrypt
5. **Регулярно обновляйте** зависимости: `pip install -r requirements.txt --upgrade`

## Устранение проблем

### Бот не запускается

```bash
# Проверьте логи
sudo journalctl -u crypto-bot -n 50

# Проверьте .env файл
cat .env | grep -v "SECRET\|PASSWORD\|KEY"  # Не показывайте секреты!

# Попробуйте запустить вручную
cd /opt/crypto_bot
source venv/bin/activate
export NUMBA_CACHE_DIR="/tmp/numba_cache"
mkdir -p /tmp/numba_cache
python main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

### Ошибка "RuntimeError: cannot cache function" (Numba/pandas_ta)

Если возникает ошибка с кэшированием Numba:

```bash
# Решение 1: Установите переменную окружения (уже в systemd service)
export NUMBA_CACHE_DIR="/tmp/numba_cache"
mkdir -p /tmp/numba_cache

# Решение 2: Или используйте домашнюю директорию
export NUMBA_CACHE_DIR="$HOME/.numba_cache"
mkdir -p "$HOME/.numba_cache"

# Решение 3: Переустановите зависимости
pip install --upgrade numba pandas-ta

# После исправления перезапустите службу
sudo systemctl restart crypto-bot
```

### Порт уже занят

```bash
# Проверьте, что использует порт 5000
sudo lsof -i :5000

# Или измените порт в systemd service файле
```

### Проблемы с правами доступа

```bash
# Проверьте права на файлы
ls -la /opt/crypto_bot

# Установите правильные права
sudo chown -R $USER:$USER /opt/crypto_bot
chmod +x start.sh
```

## Резервное копирование

Рекомендуется регулярно делать бэкапы:
- `.env` файл (с секретами)
- `bot_history.json` (история сделок и сигналов)
- `ml_models/` (ML модели, если используются)
- Конфигурационные файлы

```bash
# Пример скрипта бэкапа
tar -czf backup-$(date +%Y%m%d).tar.gz .env bot_history.json ml_models/ processed_signals.json
```
