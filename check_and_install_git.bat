@echo off
chcp 65001 >nul
echo ========================================
echo Проверка и установка Git
echo ========================================
echo.

REM Проверяем, установлен ли Git
where git >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Git найден в PATH
    git --version
    echo.
    echo Git уже установлен и работает!
    echo Вы можете использовать команды git.
    pause
    exit /b 0
)

echo [INFO] Git не найден в PATH
echo.

REM Проверяем стандартные места установки
echo Проверка стандартных мест установки...
echo.

set "GIT_FOUND=0"

if exist "C:\Program Files\Git\bin\git.exe" (
    echo [НАЙДЕНО] Git найден: C:\Program Files\Git\bin\git.exe
    set "GIT_PATH=C:\Program Files\Git\bin"
    set "GIT_FOUND=1"
)

if exist "C:\Program Files (x86)\Git\bin\git.exe" (
    echo [НАЙДЕНО] Git найден: C:\Program Files (x86)\Git\bin\git.exe
    set "GIT_PATH=C:\Program Files (x86)\Git\bin"
    set "GIT_FOUND=1"
)

if exist "%LOCALAPPDATA%\Programs\Git\bin\git.exe" (
    echo [НАЙДЕНО] Git найден: %LOCALAPPDATA%\Programs\Git\bin\git.exe
    set "GIT_PATH=%LOCALAPPDATA%\Programs\Git\bin"
    set "GIT_FOUND=1"
)

if %GIT_FOUND% == 1 (
    echo.
    echo ========================================
    echo Git установлен, но не в PATH!
    echo ========================================
    echo.
    echo Путь к Git: %GIT_PATH%
    echo.
    echo ВАРИАНТ 1: Добавить Git в PATH вручную
    echo   1. Нажмите Win+R
    echo   2. Введите: sysdm.cpl
    echo   3. Вкладка "Дополнительно" ^> "Переменные среды"
    echo   4. В "Системные переменные" найдите Path ^> "Изменить"
    echo   5. "Создать" ^> Добавьте: %GIT_PATH%
    echo   6. "ОК" во всех окнах
    echo   7. Перезапустите PowerShell/CMD
    echo.
    echo ВАРИАНТ 2: Использовать Git Bash
    echo   - Найдите "Git Bash" в меню Пуск
    echo   - Откройте Git Bash
    echo   - Используйте команды git там
    echo.
    echo ВАРИАНТ 3: Использовать полный путь
    echo   "%GIT_PATH%\git.exe" --version
    echo   "%GIT_PATH%\git.exe" init
    echo.
    pause
    exit /b 1
) else (
    echo [НЕ НАЙДЕНО] Git не установлен
    echo.
    echo ========================================
    echo Нужно установить Git
    echo ========================================
    echo.
    echo 1. Скачайте Git с: https://git-scm.com/download/win
    echo 2. Запустите установщик
    echo 3. ВАЖНО: При установке выберите опцию:
    echo    "Add Git to PATH" или
    echo    "Use Git from the Windows Command Prompt"
    echo 4. Завершите установку
    echo 5. Перезапустите PowerShell/CMD
    echo.
    echo Открыть страницу загрузки Git?
    set /p open="(y/n): "
    if /i "%open%"=="y" (
        start https://git-scm.com/download/win
    )
    pause
    exit /b 1
)
