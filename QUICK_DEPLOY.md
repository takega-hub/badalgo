# Быстрое создание файлов деплоя на сервере

## Проблема
Файлы `deploy.sh` и `update.sh` не найдены на сервере, потому что они еще не закоммичены в репозиторий.

## Решение 1: Создать файлы напрямую на сервере

### Шаг 1: Создайте скрипт для создания файлов

На сервере выполните:

```bash
cd /opt/crypto_bot
```

Затем создайте файл `create_deploy_files.sh`:

```bash
nano create_deploy_files.sh
```

Скопируйте содержимое из файла `create_deploy_files.sh` (который я создал) или выполните команды ниже.

### Шаг 2: Создайте deploy.sh

```bash
cat > /opt/crypto_bot/deploy.sh << 'EOF'
#!/bin/bash
set -e
PROJECT_DIR="/opt/crypto_bot"
REPO_URL="https://github.com/takega-hub/badalgo.git"
BRANCH="main"
cd "$PROJECT_DIR"
if [ ! -d ".git" ]; then
    git clone -b "$BRANCH" "$REPO_URL" .
else
    git fetch origin
    git reset --hard origin/"$BRANCH"
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs ml_models ml_data
mkdir -p /tmp/numba_cache
chmod 755 /tmp/numba_cache
EOF

chmod +x /opt/crypto_bot/deploy.sh
```

### Шаг 3: Создайте update.sh

```bash
cat > /opt/crypto_bot/update.sh << 'EOF'
#!/bin/bash
set -e
PROJECT_DIR="/opt/crypto_bot"
cd "$PROJECT_DIR"
supervisorctl stop crypto-bot || true
git fetch origin
git reset --hard origin/main
git clean -fd
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ -f supervisor/crypto-bot.conf ]; then
    USERNAME=$(whoami)
    sed "s|user=your_username|user=$USERNAME|g" supervisor/crypto-bot.conf > /etc/supervisor/conf.d/crypto-bot.conf
    supervisorctl reread
    supervisorctl update
fi
supervisorctl start crypto-bot
EOF

chmod +x /opt/crypto_bot/update.sh
```

## Решение 2: Добавить файлы в репозиторий (рекомендуется)

### На вашем локальном компьютере:

```bash
# Добавьте файлы
git add deploy.sh update.sh supervisor/crypto-bot.conf DEPLOY_SUPERVISOR.md

# Создайте коммит
git commit -m "Add deployment scripts and supervisor configuration"

# Отправьте в GitHub
git push origin main
```

### На сервере:

```bash
cd /opt/crypto_bot
sudo git pull origin main
sudo chmod +x deploy.sh update.sh
```

## Решение 3: Использовать готовый скрипт

Если у вас есть доступ к файлу `create_deploy_files.sh`:

```bash
cd /opt/crypto_bot
# Загрузите create_deploy_files.sh на сервер (через scp или создайте вручную)
bash create_deploy_files.sh
```

## После создания файлов

Проверьте, что файлы созданы:

```bash
ls -la /opt/crypto_bot/deploy.sh
ls -la /opt/crypto_bot/update.sh
```

Если файлы существуют, можно использовать их:

```bash
sudo ./deploy.sh  # Для первоначального деплоя
sudo ./update.sh  # Для обновления
```
