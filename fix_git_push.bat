@echo off
chcp 65001 >nul
echo ========================================
echo Исправление проблемы с git push
echo ========================================
echo.

REM Проверяем статус
echo Проверка статуса Git...
git status
echo.

REM Проверяем, есть ли коммиты
echo Проверка коммитов...
git log --oneline -1 >nul 2>&1
if %errorlevel% neq 0 (
    echo [ПРОБЛЕМА] Нет коммитов в репозитории
    echo.
    echo Нужно создать первый коммит:
    echo   1. git add .
    echo   2. git commit -m "Initial commit"
    echo   3. git push -u origin main
    echo.
    echo Создать коммит сейчас?
    set /p create="(y/n): "
    if /i "%create%"=="y" (
        echo.
        echo Добавление файлов...
        git add .
        echo.
        echo Создание коммита...
        git commit -m "Initial commit"
        echo.
        echo Коммит создан!
    ) else (
        echo Отменено.
        pause
        exit /b 1
    )
) else (
    echo [OK] Коммиты найдены
    git log --oneline -1
)

echo.
echo Проверка веток...
git branch
echo.

REM Проверяем, какая ветка используется
git branch --show-current >nul 2>&1
if %errorlevel% neq 0 (
    echo [ПРОБЛЕМА] Нет активной ветки
    echo.
    echo Создание ветки main...
    git checkout -b main
) else (
    setlocal enabledelayedexpansion
    for /f "delims=" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
    echo Текущая ветка: !CURRENT_BRANCH!
    
    if "!CURRENT_BRANCH!"=="master" (
        echo.
        echo [INFO] Используется ветка master, а не main
        echo.
        echo ВАРИАНТ 1: Переименовать master в main
        echo   git branch -M main
        echo   git push -u origin main
        echo.
        echo ВАРИАНТ 2: Использовать master
        echo   git push -u origin master
        echo.
        set /p rename="Переименовать master в main? (y/n): "
        if /i "!rename!"=="y" (
            git branch -M main
            echo Ветка переименована в main
        )
    )
    endlocal
)

echo.
echo Проверка remote...
git remote -v
echo.

echo.
echo ========================================
echo Попытка push...
echo ========================================
echo.

REM Пробуем push в main
git push -u origin main 2>&1
if %errorlevel% == 0 (
    echo.
    echo [УСПЕХ] Код успешно отправлен в GitHub!
) else (
    echo.
    echo [ОШИБКА] Push в main не удался
    echo.
    echo Пробуем push в master...
    git push -u origin master 2>&1
    if %errorlevel% == 0 (
        echo.
        echo [УСПЕХ] Код успешно отправлен в GitHub (ветка master)!
    ) else (
        echo.
        echo [ОШИБКА] Push не удался
        echo.
        echo Возможные причины:
        echo   1. Нет коммитов (создайте: git add . ^&^& git commit -m "Initial commit")
        echo   2. Неправильное имя ветки (используйте: git branch --show-current)
        echo   3. Проблемы с доступом к GitHub (проверьте права доступа)
        echo.
        echo Текущая ветка:
        git branch --show-current
        echo.
        echo Коммиты:
        git log --oneline -5
    )
)

echo.
pause
