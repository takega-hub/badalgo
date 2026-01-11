# Деплой Crypto Bot на сервер с Supervisor

## Требования

- Ubuntu 20.04+ / Debian 11+ или другая Linux-система
- Python 3.10+
- Git
- Supervisor

## Шаг 1: Установка зависимостей системы

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python, Git и Supervisor
sudo apt install -y python3 python3-pip python3-venv git supervisor

# Установка зависимостей для ML библиотек
sudo apt install -y build-essential python3-dev
```

## Шаг 2: Первоначальный деплой

### Вариант A: Использование скрипта деплоя (рекомендуется)

1. **Скачайте скрипт деплоя на сервер:**

```bash
cd /opt
sudo git clone https://github.com/takega-hub/badalgo.git crypto_bot
cd crypto_bot
sudo chmod +x deploy.sh
```

2. **Запустите скрипт деплоя:**

```bash
sudo ./deploy.sh
```

Скрипт автоматически:
- Клонирует/обновит репозиторий
- Создаст виртуальное окружение
- Установит зависимости
- Создаст необходимые директории
- Настроит supervisor

### Вариант B: Ручная установка

```bash
# 1. Клонирование репозитория
cd /opt
sudo git clone https://github.com/takega-hub/badalgo.git crypto_bot
sudo chown -R $USER:$USER crypto_bot
cd crypto_bot

# 2. Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# 3. Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# 4. Создание необходимых директорий
mkdir -p logs ml_models ml_data
mkdir -p /tmp/numba_cache
chmod 755 /tmp/numba_cache

# 5. Создание .env файла
cp env.example .env
nano .env  # Заполните все необходимые переменные
```

## Шаг 3: Настройка Supervisor

### 1. Редактирование конфигурации

Откройте файл конфигурации:

```bash
nano supervisor/crypto-bot.conf
```

**ВАЖНО:** Замените:
- `your_username` на ваше имя пользователя на сервере
- `/opt/crypto_bot` на реальный путь к проекту (если отличается)

### 2. Копирование конфигурации в supervisor

```bash
sudo cp supervisor/crypto-bot.conf /etc/supervisor/conf.d/crypto-bot.conf
```

### 3. Обновление supervisor

```bash
sudo supervisorctl reread
sudo supervisorctl update
```

## Шаг 4: Настройка .env файла

```bash
nano /opt/crypto_bot/.env
```

Заполните все необходимые переменные:
- `BYBIT_API_KEY` и `BYBIT_API_SECRET` - ваши API ключи Bybit
- `BYBIT_BASE_URL` - URL Bybit (testnet или production)
- `ADMIN_USERNAME` и `ADMIN_PASSWORD` - логин и пароль для админ-панели
- `FLASK_SECRET_KEY` - случайный секретный ключ (можно сгенерировать: `python -c "import secrets; print(secrets.token_hex(32))"`)

## Шаг 5: Запуск бота

```bash
# Запуск
sudo supervisorctl start crypto-bot

# Проверка статуса
sudo supervisorctl status crypto-bot

# Просмотр логов
tail -f /opt/crypto_bot/logs/crypto-bot.log
```

## Управление ботом через Supervisor

```bash
# Запуск
sudo supervisorctl start crypto-bot

# Остановка
sudo supervisorctl stop crypto-bot

# Перезапуск
sudo supervisorctl restart crypto-bot

# Статус
sudo supervisorctl status crypto-bot

# Просмотр всех процессов
sudo supervisorctl status

# Перезагрузка конфигурации (после изменений в crypto-bot.conf)
sudo supervisorctl reread
sudo supervisorctl update
```

## Обновление бота

### Вариант A: Использование скрипта обновления (рекомендуется)

```bash
cd /opt/crypto_bot
sudo ./update.sh
```

Скрипт автоматически:
- Остановит бота
- Обновит код из GitHub
- Обновит зависимости
- Перезапустит бота

### Вариант B: Ручное обновление

```bash
cd /opt/crypto_bot

# Остановка бота
sudo supervisorctl stop crypto-bot

# Обновление кода
git pull origin main

# Обновление зависимостей
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Запуск бота
sudo supervisorctl start crypto-bot
```

## Просмотр логов

```bash
# Основной лог (stdout)
tail -f /opt/crypto_bot/logs/crypto-bot.log

# Лог ошибок (stderr)
tail -f /opt/crypto_bot/logs/crypto-bot-error.log

# Последние 100 строк
tail -n 100 /opt/crypto_bot/logs/crypto-bot.log

# Поиск ошибок
grep -i error /opt/crypto_bot/logs/crypto-bot-error.log
```

## Настройка Nginx (опционально, для проксирования)

### 1. Установка Nginx

```bash
sudo apt install -y nginx
```

### 2. Создание конфигурации

```bash
sudo nano /etc/nginx/sites-available/crypto-bot
```

Содержимое:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Замените на ваш домен или IP

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Таймауты
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
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
sudo apt install -y certbot python3-certbot-nginx
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

## Устранение проблем

### Бот не запускается

```bash
# Проверьте логи
tail -n 50 /opt/crypto_bot/logs/crypto-bot-error.log

# Проверьте статус supervisor
sudo supervisorctl status crypto-bot

# Проверьте .env файл
cat /opt/crypto_bot/.env | grep -v "SECRET\|PASSWORD\|KEY"

# Попробуйте запустить вручную
cd /opt/crypto_bot
source venv/bin/activate
export NUMBA_CACHE_DIR="/tmp/numba_cache"
python main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

### Supervisor не видит конфигурацию

```bash
# Проверьте синтаксис конфигурации
sudo supervisorctl reread

# Если есть ошибки, проверьте файл
sudo cat /etc/supervisor/conf.d/crypto-bot.conf

# Убедитесь, что файл существует
ls -la /etc/supervisor/conf.d/crypto-bot.conf
```

### Порт уже занят

```bash
# Проверьте, что использует порт 5000
sudo lsof -i :5000

# Или измените порт в конфигурации supervisor
```

### Проблемы с правами доступа

```bash
# Проверьте права на файлы
ls -la /opt/crypto_bot

# Установите правильные права
sudo chown -R $USER:$USER /opt/crypto_bot
chmod +x /opt/crypto_bot/*.sh
```

## Резервное копирование

Рекомендуется регулярно делать бэкапы:

```bash
# Создание бэкапа
tar -czf backup-$(date +%Y%m%d).tar.gz \
    /opt/crypto_bot/.env \
    /opt/crypto_bot/bot_history.json \
    /opt/crypto_bot/ml_models/ \
    /opt/crypto_bot/processed_signals*.json

# Восстановление из бэкапа
tar -xzf backup-YYYYMMDD.tar.gz -C /
```

## Автоматическое обновление (опционально)

Можно настроить cron для автоматического обновления:

```bash
# Редактирование crontab
crontab -e

# Добавьте строку для ежедневного обновления в 3:00
0 3 * * * cd /opt/crypto_bot && /opt/crypto_bot/update.sh >> /opt/crypto_bot/logs/update.log 2>&1
```

## Безопасность

1. **Измените пароль администратора** через веб-интерфейс после первого входа
2. **Используйте сильный FLASK_SECRET_KEY**
3. **Настройте файрвол** для ограничения доступа
4. **Используйте HTTPS** через Nginx с Let's Encrypt
5. **Регулярно обновляйте** зависимости и систему
6. **Ограничьте доступ к .env файлу:**
   ```bash
   chmod 600 /opt/crypto_bot/.env
   ```
