@echo off
chcp 65001 >nul
echo ========================================
echo Проверка секретов перед публикацией
echo ========================================
echo.

echo Проверка файлов, которые могут содержать секреты...
echo.

REM Проверяем .env
if exist .env (
    echo [ВНИМАНИЕ] Найден файл .env
    echo Проверяем, игнорируется ли он...
    git check-ignore .env >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] .env в .gitignore - безопасно
    ) else (
        echo [ОПАСНО] .env НЕ в .gitignore!
        echo Нужно добавить .env в .gitignore
    )
) else (
    echo [OK] .env не найден
)

echo.

REM Проверяем другие файлы с секретами
echo Проверка других потенциально опасных файлов...
git ls-files | findstr /i "\.env secret key password token api" >nul 2>&1
if %errorlevel% == 0 (
    echo [ВНИМАНИЕ] Найдены файлы, которые могут содержать секреты:
    git ls-files | findstr /i "\.env secret key password token api"
    echo.
    echo Проверьте эти файлы перед публикацией!
) else (
    echo [OK] Потенциально опасные файлы не найдены в репозитории
)

echo.

REM Проверяем .gitignore
echo Проверка .gitignore...
if exist .gitignore (
    echo [OK] .gitignore существует
    echo.
    echo Содержимое .gitignore:
    type .gitignore | findstr /i "\.env"
    if %errorlevel% == 0 (
        echo [OK] .env упомянут в .gitignore
    ) else (
        echo [ВНИМАНИЕ] .env не найден в .gitignore
    )
) else (
    echo [ОПАСНО] .gitignore не существует!
)

echo.
echo ========================================
echo Рекомендации:
echo ========================================
echo.
echo 1. Убедитесь, что .env в .gitignore
echo 2. Проверьте, что API ключи не в коде
echo 3. Используйте переменные окружения
echo 4. После проверки можно делать репозиторий публичным
echo.
pause
