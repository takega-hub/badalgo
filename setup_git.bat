@echo off
chcp 65001 >nul
echo === Настройка Git для crypto_bot ===

REM Проверяем, установлен ли Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Git не установлен. Установите Git с https://git-scm.com/
    pause
    exit /b 1
)

echo Git найден
git --version

REM Проверяем, инициализирован ли репозиторий
if exist .git (
    echo Git репозиторий уже инициализирован
) else (
    echo Инициализация Git репозитория...
    git init
    echo Git репозиторий инициализирован
)

REM Проверяем, есть ли remote
echo.
echo Текущие remote репозитории:
git remote -v

REM Запрашиваем URL GitHub репозитория
echo.
echo Введите URL вашего GitHub репозитория (например: https://github.com/username/crypto_bot.git)
echo Или нажмите Enter, чтобы пропустить этот шаг
set /p githubUrl="GitHub URL: "

if not "%githubUrl%"=="" (
    REM Проверяем, есть ли уже origin
    git remote get-url origin >nul 2>&1
    if errorlevel 1 (
        git remote add origin "%githubUrl%"
        echo Remote 'origin' добавлен: %githubUrl%
    ) else (
        echo Remote 'origin' уже существует
        set /p replace="Заменить? (y/n): "
        if /i "%replace%"=="y" (
            git remote set-url origin "%githubUrl%"
            echo Remote 'origin' обновлен
        )
    )
    
    echo.
    echo Проверка подключения к GitHub...
    git remote -v
)

REM Показываем статус
echo.
echo === Статус Git ===
git status

echo.
echo === Следующие шаги ===
echo 1. Добавьте файлы: git add .
echo 2. Создайте коммит: git commit -m "Initial commit"
echo 3. Отправьте в GitHub: git push -u origin main
echo    (или 'git push -u origin master' если используется ветка master)

pause
