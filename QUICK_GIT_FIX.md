# Быстрое решение проблемы с Git

## Проблема
```
git : Имя "git" не распознано...
```

Это означает, что Git либо не установлен, либо не добавлен в PATH.

---

## Решение 1: Установить Git (если не установлен)

### Шаг 1: Скачать Git
1. Откройте браузер
2. Перейдите на: **https://git-scm.com/download/win**
3. Скачайте установщик (автоматически определит 64-bit или 32-bit)

### Шаг 2: Установить Git
1. Запустите скачанный установщик
2. **ВАЖНО:** На шаге "Adjusting your PATH environment" выберите:
   - ✅ **"Git from the command line and also from 3rd-party software"**
   - ИЛИ
   - ✅ **"Use Git and optional Unix tools from the Command Prompt"**
3. Остальные настройки можно оставить по умолчанию
4. Нажмите "Next" до завершения установки

### Шаг 3: Перезапустить PowerShell
1. Закройте текущее окно PowerShell
2. Откройте новое окно PowerShell
3. Проверьте: `git --version`

Если видите версию (например, `git version 2.42.0`), значит всё готово!

---

## Решение 2: Использовать Git Bash (если Git уже установлен)

Если Git уже установлен, но не работает в PowerShell:

1. **Найдите Git Bash:**
   - Нажмите `Win` (клавиша Windows)
   - Введите: `Git Bash`
   - Откройте "Git Bash"

2. **В Git Bash перейдите в папку проекта:**
   ```bash
   cd /c/Users/takeg/OneDrive/Документы/vibecodding/crypto_bot
   ```

3. **Используйте команды Git в Git Bash:**
   ```bash
   git init
   git remote add origin https://github.com/ВАШ_USERNAME/crypto_bot.git
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

---

## Решение 3: Добавить Git в PATH вручную

Если Git установлен, но не в PATH:

### Шаг 1: Найти Git
Обычно Git находится в одном из этих мест:
- `C:\Program Files\Git\bin\git.exe`
- `C:\Program Files (x86)\Git\bin\git.exe`
- `C:\Users\ВАШ_ПОЛЬЗОВАТЕЛЬ\AppData\Local\Programs\Git\bin\git.exe`

Проверьте эти папки в Проводнике.

### Шаг 2: Добавить в PATH
1. Нажмите `Win + R`
2. Введите: `sysdm.cpl` → Enter
3. Вкладка **"Дополнительно"**
4. Нажмите **"Переменные среды"**
5. В разделе **"Системные переменные"** найдите переменную `Path`
6. Нажмите **"Изменить"**
7. Нажмите **"Создать"**
8. Добавьте путь к папке `bin` Git (например: `C:\Program Files\Git\bin`)
9. Нажмите **"ОК"** во всех окнах
10. **Перезапустите PowerShell**

---

## Решение 4: Использовать полный путь к Git

Если Git установлен, но не в PATH, можно использовать полный путь:

```powershell
# Вместо: git init
& "C:\Program Files\Git\bin\git.exe" init

# Вместо: git add .
& "C:\Program Files\Git\bin\git.exe" add .

# И так далее...
```

Или создайте алиас в PowerShell:
```powershell
Set-Alias git "C:\Program Files\Git\bin\git.exe"
```

---

## Проверка после установки/настройки

Выполните в PowerShell:
```powershell
git --version
```

Если команда работает и показывает версию, значит всё готово!

---

## После того, как Git заработает

1. **Настройте Git (если первый раз):**
   ```bash
   git config --global user.name "Ваше Имя"
   git config --global user.email "your.email@example.com"
   ```

2. **Инициализируйте репозиторий:**
   ```bash
   git init
   ```

3. **Добавьте remote (замените на ваш GitHub URL):**
   ```bash
   git remote add origin https://github.com/ВАШ_USERNAME/crypto_bot.git
   ```

4. **Добавьте файлы и создайте коммит:**
   ```bash
   git add .
   git commit -m "Initial commit"
   ```

5. **Отправьте в GitHub:**
   ```bash
   git push -u origin main
   ```

---

## Быстрая проверка

Запустите файл `check_and_install_git.bat` - он автоматически проверит, установлен ли Git и где он находится.
