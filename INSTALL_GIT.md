# Установка Git и настройка для работы с GitHub

## Проблема
Git не найден в PATH системы. Нужно установить Git или добавить его в PATH.

## Решение 1: Установка Git (если не установлен)

### Шаг 1: Скачать Git
1. Перейдите на https://git-scm.com/download/win
2. Скачайте установщик Git для Windows
3. Запустите установщик

### Шаг 2: Установка
1. **Важно:** При установке выберите опцию **"Add Git to PATH"** или **"Use Git from the Windows Command Prompt"**
2. Остальные настройки можно оставить по умолчанию
3. Завершите установку

### Шаг 3: Перезапустите PowerShell/CMD
После установки закройте и снова откройте PowerShell или CMD.

### Шаг 4: Проверка
В PowerShell или CMD выполните:
```bash
git --version
```

Если видите версию Git (например, `git version 2.42.0`), значит всё установлено правильно.

---

## Решение 2: Добавить Git в PATH (если Git уже установлен)

Если Git уже установлен, но не работает в PowerShell, нужно добавить его в PATH.

### Вариант A: Через графический интерфейс

1. **Найдите, где установлен Git:**
   - Обычно: `C:\Program Files\Git\bin\git.exe`
   - Или: `C:\Program Files (x86)\Git\bin\git.exe`
   - Или: `C:\Users\ВАШ_ПОЛЬЗОВАТЕЛЬ\AppData\Local\Programs\Git\bin\git.exe`

2. **Добавьте в PATH:**
   - Нажмите `Win + R`
   - Введите `sysdm.cpl` и нажмите Enter
   - Перейдите на вкладку **"Дополнительно"**
   - Нажмите **"Переменные среды"**
   - В разделе **"Системные переменные"** найдите переменную `Path`
   - Нажмите **"Изменить"**
   - Нажмите **"Создать"**
   - Добавьте путь к папке `bin` Git (например: `C:\Program Files\Git\bin`)
   - Нажмите **"ОК"** во всех окнах

3. **Перезапустите PowerShell/CMD**

### Вариант B: Через PowerShell (если PowerShell работает)

Откройте PowerShell **от имени администратора** и выполните:

```powershell
# Найдите путь к Git
$gitPath = "C:\Program Files\Git\bin"
if (Test-Path $gitPath) {
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$gitPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$gitPath", "User")
        Write-Host "Git добавлен в PATH. Перезапустите PowerShell."
    } else {
        Write-Host "Git уже в PATH"
    }
} else {
    Write-Host "Git не найден по пути: $gitPath"
    Write-Host "Проверьте, где установлен Git"
}
```

---

## Решение 3: Использовать Git Bash (альтернатива)

Если Git установлен, но не работает в PowerShell:

1. Найдите **Git Bash** в меню Пуск
2. Откройте Git Bash
3. Перейдите в папку проекта:
   ```bash
   cd /c/Users/takeg/OneDrive/Документы/vibecodding/crypto_bot
   ```
4. Используйте Git команды в Git Bash

---

## После установки/настройки Git

### 1. Настройте Git (если первый раз):
```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
```

### 2. Инициализируйте репозиторий:
```bash
cd "C:\Users\takeg\OneDrive\Документы\vibecodding\crypto_bot"
git init
```

### 3. Добавьте remote репозиторий:
```bash
git remote add origin https://github.com/ВАШ_USERNAME/crypto_bot.git
```

### 4. Добавьте файлы и создайте коммит:
```bash
git add .
git commit -m "Initial commit"
```

### 5. Отправьте в GitHub:
```bash
git push -u origin main
```

---

## Проверка установки

Выполните в PowerShell или CMD:
```bash
git --version
```

Если команда работает, Git установлен и настроен правильно.

---

## Полезные ссылки

- Официальный сайт Git: https://git-scm.com/
- Скачать Git для Windows: https://git-scm.com/download/win
- Документация Git: https://git-scm.com/doc
