# Настройка идентификации Git

## Проблема
```
Author identity unknown
*** Please tell me who are you.
```

Git требует знать ваше имя и email для создания коммитов.

---

## Быстрое решение

### Вариант 1: Использовать скрипт (рекомендуется)

Запустите в PowerShell:
```powershell
.\setup_git_user.bat
```

Скрипт запросит ваше имя и email, затем настроит Git.

---

## Вариант 2: Ручная настройка

### Шаг 1: Настройте имя и email

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
```

**Пример:**
```bash
git config --global user.name "Takeg"
git config --global user.email "takeg@example.com"
```

### Шаг 2: Проверьте настройки

```bash
git config --global user.name
git config --global user.email
```

### Шаг 3: Создайте коммит

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## Важные замечания

### Глобальная настройка (--global)

Команда с `--global` настраивает Git для всех репозиториев на вашем компьютере.

### Локальная настройка (без --global)

Если хотите настроить только для этого репозитория:

```bash
git config user.name "Ваше Имя"
git config user.email "your.email@example.com"
```

### Email и GitHub

- Email может быть любым (не обязательно связан с GitHub)
- Если хотите, чтобы коммиты отображались на GitHub, используйте email, привязанный к вашему аккаунту GitHub
- Или добавьте email в настройках GitHub: https://github.com/settings/emails

---

## После настройки

Выполните:

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## Проверка всех настроек Git

```bash
git config --list
```

Покажет все настройки Git.
