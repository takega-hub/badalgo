# Устранение ошибки "spawn error" в Supervisor

## Проблема
```
crypto-bot: ERROR (spawn error)
```

Эта ошибка означает, что supervisor не может запустить процесс.

## Диагностика

### 1. Проверьте логи supervisor

```bash
# Просмотр логов ошибок
sudo tail -n 50 /var/log/supervisor/supervisord.log

# Или для конкретной программы
sudo supervisorctl tail -f crypto-bot stderr
```

### 2. Проверьте конфигурацию

```bash
# Просмотр конфигурации
sudo cat /etc/supervisor/conf.d/crypto-bot.conf

# Проверка синтаксиса
sudo supervisorctl reread
```

### 3. Проверьте существование файлов и путей

```bash
# Проверьте, что все пути существуют
ls -la /opt/crypto_bot/venv/bin/python
ls -la /opt/crypto_bot/main.py

# Проверьте права доступа
ls -la /opt/crypto_bot/
```

### 4. Проверьте пользователя в конфигурации

```bash
# Узнайте текущего пользователя
whoami

# Проверьте, что пользователь в конфигурации правильный
grep "user=" /etc/supervisor/conf.d/crypto-bot.conf
```

## Решения

### Решение 1: Исправить конфигурацию

Отредактируйте конфигурацию:

```bash
sudo nano /etc/supervisor/conf.d/crypto-bot.conf
```

Убедитесь, что:
1. Все пути правильные
2. Пользователь существует
3. Команда может быть выполнена

Пример правильной конфигурации:

```ini
[program:crypto-bot]
command=/opt/crypto_bot/venv/bin/python /opt/crypto_bot/main.py --mode web --web-host 0.0.0.0 --web-port 5000
directory=/opt/crypto_bot
user=root
autostart=true
autorestart=true
startretries=3
startsecs=10
stdout_logfile=/opt/crypto_bot/logs/crypto-bot.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stderr_logfile=/opt/crypto_bot/logs/crypto-bot-error.log
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=10
environment=PATH="/opt/crypto_bot/venv/bin:%(ENV_PATH)s",NUMBA_CACHE_DIR="/tmp/numba_cache"
stopasgroup=true
killasgroup=true
stopwaitsecs=10
```

После редактирования:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start crypto-bot
```

### Решение 2: Проверить виртуальное окружение

```bash
# Проверьте, что venv существует
ls -la /opt/crypto_bot/venv/bin/python

# Если нет, создайте
cd /opt/crypto_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Решение 3: Проверить main.py

```bash
# Проверьте, что main.py существует и исполняемый
ls -la /opt/crypto_bot/main.py

# Попробуйте запустить вручную
cd /opt/crypto_bot
source venv/bin/activate
python main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

Если ручной запуск работает, проблема в конфигурации supervisor.

### Решение 4: Использовать абсолютные пути

Убедитесь, что в конфигурации используются абсолютные пути:

```ini
command=/opt/crypto_bot/venv/bin/python /opt/crypto_bot/main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

Не используйте относительные пути или переменные окружения в команде.

### Решение 5: Проверить права доступа

```bash
# Установите правильные права
sudo chown -R root:root /opt/crypto_bot
chmod +x /opt/crypto_bot/main.py
chmod +x /opt/crypto_bot/venv/bin/python
```

### Решение 6: Создать директорию для логов

```bash
mkdir -p /opt/crypto_bot/logs
chmod 755 /opt/crypto_bot/logs
```

## Быстрая диагностика

Выполните все команды по порядку:

```bash
# 1. Проверка файлов
echo "=== Проверка файлов ==="
ls -la /opt/crypto_bot/venv/bin/python
ls -la /opt/crypto_bot/main.py
ls -la /opt/crypto_bot/.env

# 2. Проверка конфигурации
echo "=== Конфигурация ==="
cat /etc/supervisor/conf.d/crypto-bot.conf

# 3. Проверка пользователя
echo "=== Пользователь ==="
whoami
grep "user=" /etc/supervisor/conf.d/crypto-bot.conf

# 4. Ручной запуск
echo "=== Ручной запуск ==="
cd /opt/crypto_bot
source venv/bin/activate
python main.py --mode web --web-host 0.0.0.0 --web-port 5000 &
sleep 2
ps aux | grep python
pkill -f "main.py"
```

## После исправления

```bash
# Перезагрузите конфигурацию
sudo supervisorctl reread
sudo supervisorctl update

# Запустите бота
sudo supervisorctl start crypto-bot

# Проверьте статус
sudo supervisorctl status crypto-bot

# Просмотр логов
tail -f /opt/crypto_bot/logs/crypto-bot.log
```
