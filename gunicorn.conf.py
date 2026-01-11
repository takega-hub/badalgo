# Конфигурация Gunicorn для Crypto Trading Bot
# Использование: gunicorn -c gunicorn.conf.py 'bot.web.app:app'

# Количество воркеров (рекомендуется: (2 x CPU cores) + 1)
workers = 4

# Привязка к адресу и порту
bind = "0.0.0.0:5000"

# Таймауты
timeout = 120
keepalive = 5

# Логирование
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Имя процесса
proc_name = "crypto-bot"

# Пользователь и группа (раскомментируйте и укажите свои)
# user = "your_user"
# group = "your_group"

# Ограничения
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Перезапуск воркеров при превышении памяти (в байтах)
# max_requests = 1000
# max_requests_jitter = 50
