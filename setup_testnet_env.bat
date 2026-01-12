@echo off
REM Скрипт для настройки .env файла для тестового сервера Bybit

echo ========================================
echo Настройка .env для Bybit Testnet
echo ========================================
echo.

REM Проверяем, существует ли .env
if exist .env (
    echo Файл .env уже существует.
    echo Создаю резервную копию .env.backup...
    copy .env .env.backup >nul
    echo Резервная копия создана.
    echo.
)

REM Копируем env.example в .env
if exist env.example (
    copy env.example .env >nul
    echo Файл .env создан из env.example
    echo.
) else (
    echo ОШИБКА: Файл env.example не найден!
    pause
    exit /b 1
)

REM Обновляем API ключи для testnet
echo Обновляю API ключи для testnet...
powershell -Command "(Get-Content .env) -replace 'BYBIT_API_KEY=.*', 'BYBIT_API_KEY=Oqe7oPIvtBs60iIoIz' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'BYBIT_API_SECRET=.*', 'BYBIT_API_SECRET=BmJvjEknyMgq8MWGybVPG8OopxUy2qzFaxdc' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'BYBIT_BASE_URL=.*', 'BYBIT_BASE_URL=https://api-testnet.bybit.com' | Set-Content .env"

echo.
echo ========================================
echo Настройка завершена!
echo ========================================
echo.
echo Файл .env настроен для работы с Bybit Testnet.
echo.
echo Следующие шаги:
echo 1. Проверьте файл .env и при необходимости отредактируйте настройки
echo 2. Запустите бота: python main.py --mode web
echo.
pause
