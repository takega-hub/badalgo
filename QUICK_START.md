# Быстрый старт - Деплой на сервер

## Минимальные шаги для деплоя

### 1. Загрузите проект на сервер
```bash
cd /opt
git clone <your-repo> crypto_bot
# или загрузите через scp/rsync
```

### 2. Настройте окружение
```bash
cd crypto_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Создайте .env файл
```bash
cp env.example .env
nano .env
```

**Обязательно заполните:**
- `BYBIT_API_KEY` и `BYBIT_API_SECRET`
- `ADMIN_USERNAME` и `ADMIN_PASSWORD`
- `FLASK_SECRET_KEY` (сгенерируйте: `python -c "import secrets; print(secrets.token_hex(32))"`)

### 4. Настройте systemd службу
```bash
# Создайте директорию для Numba кэша (исправление проблемы с pandas_ta)
sudo mkdir -p /tmp/numba_cache
sudo chmod 755 /tmp/numba_cache

# Создайте service файл
sudo nano /etc/systemd/system/crypto-bot.service
```

Скопируйте содержимое из `crypto-bot.service` и измените:
- `User=your_username` (замените `%i` на ваше имя пользователя, например `User=crypto` или удалите строку `User=%i`)
- `WorkingDirectory=/opt/crypto_bot` (замените `/opt/crypto_bot` на ваш путь, например `/home/crypto_bot`)
- Все пути в `ExecStart` и `ExecStartPre` также измените на ваш путь

### 5. Запустите службу
```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-bot
sudo systemctl start crypto-bot
sudo systemctl status crypto-bot
```

**Примечание:** Если возникает ошибка с Numba кэшированием, убедитесь, что в `crypto-bot.service` установлена переменная окружения `NUMBA_CACHE_DIR=/tmp/numba_cache` (уже добавлена по умолчанию).

### 6. Проверьте работу
Откройте в браузере: `http://your-server-ip:5000`

## Альтернативный запуск (без systemd)

```bash
cd /opt/crypto_bot
source venv/bin/activate
./start.sh --host 0.0.0.0 --port 5000
```

## Использование Gunicorn (рекомендуется для продакшена)

```bash
# В systemd service файле замените ExecStart на:
ExecStart=/opt/crypto_bot/venv/bin/gunicorn -c /opt/crypto_bot/gunicorn.conf.py 'bot.web.app:app'

# Или запустите вручную:
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 'bot.web.app:app'
```

## Безопасность

1. **Измените пароль администратора** после первого входа
2. **Используйте сильный FLASK_SECRET_KEY**
3. **Настройте файрвол**: `sudo ufw allow 5000/tcp`
4. **Рекомендуется использовать Nginx с SSL** (см. `DEPLOY.md`)

## Мониторинг

```bash
# Просмотр логов
sudo journalctl -u crypto-bot -f

# Статус службы
sudo systemctl status crypto-bot

# Перезапуск
sudo systemctl restart crypto-bot
```

## Полная документация

См. [DEPLOY.md](DEPLOY.md) для подробных инструкций.
