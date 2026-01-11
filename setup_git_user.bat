@echo off
chcp 65001 >nul
echo ========================================
echo Настройка идентификации Git
echo ========================================
echo.

echo Введите ваше имя (будет использоваться в коммитах):
set /p GIT_NAME="Имя: "

echo.
echo Введите ваш email (будет использоваться в коммитах):
set /p GIT_EMAIL="Email: "

if "%GIT_NAME%"=="" (
    echo Ошибка: Имя не может быть пустым
    pause
    exit /b 1
)

if "%GIT_EMAIL%"=="" (
    echo Ошибка: Email не может быть пустым
    pause
    exit /b 1
)

echo.
echo Настройка Git...
git config --global user.name "%GIT_NAME%"
git config --global user.email "%GIT_EMAIL%"

echo.
echo [OK] Git настроен:
git config --global user.name
git config --global user.email

echo.
echo Теперь вы можете создать коммит:
echo   git add .
echo   git commit -m "Initial commit"
echo   git push -u origin main

echo.
pause
