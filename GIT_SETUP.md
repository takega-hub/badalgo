# Настройка Git и подключение к GitHub

## Быстрая настройка

### Вариант 1: Использование скрипта (рекомендуется)

1. **Запустите скрипт настройки:**
   - Для PowerShell: `.\setup_git.ps1`
   - Для CMD: `setup_git.bat`

2. **Следуйте инструкциям в скрипте**

### Вариант 2: Ручная настройка

1. **Инициализируйте Git репозиторий:**
   ```bash
   git init
   ```

2. **Добавьте remote репозиторий GitHub:**
   ```bash
   git remote add origin https://github.com/ВАШ_USERNAME/crypto_bot.git
   ```
   Замените `ВАШ_USERNAME` на ваш GitHub username.

3. **Проверьте подключение:**
   ```bash
   git remote -v
   ```

4. **Добавьте файлы:**
   ```bash
   git add .
   ```

5. **Создайте первый коммит:**
   ```bash
   git commit -m "Initial commit"
   ```

6. **Отправьте в GitHub:**
   ```bash
   git push -u origin main
   ```
   Если используется ветка `master` вместо `main`:
   ```bash
   git push -u origin master
   ```

## Создание репозитория на GitHub

Если у вас еще нет репозитория на GitHub:

1. Перейдите на https://github.com/new
2. Введите название репозитория (например, `crypto_bot`)
3. Выберите **Public** или **Private**
4. **НЕ** создавайте README, .gitignore или лицензию (они уже есть в проекте)
5. Нажмите **Create repository**
6. Скопируйте URL репозитория и используйте его в шаге 2 выше

## Настройка Git (если еще не настроено)

Если это первый раз использования Git на этом компьютере:

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
```

## Полезные команды

- **Проверить статус:** `git status`
- **Посмотреть изменения:** `git diff`
- **Добавить все файлы:** `git add .`
- **Создать коммит:** `git commit -m "Описание изменений"`
- **Отправить изменения:** `git push`
- **Получить изменения:** `git pull`
- **Посмотреть историю:** `git log`

## Важно

- Файл `.env` с секретами **НЕ** будет загружен в GitHub (он в .gitignore)
- Модели ML (`*.pkl`) **НЕ** будут загружены (они в .gitignore)
- Данные (`ml_data/`) **НЕ** будут загружены (они в .gitignore)
- История бота (`bot_history.json`) **НЕ** будет загружена (она в .gitignore)
