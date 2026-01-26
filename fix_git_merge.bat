@echo off
REM Скрипт для исправления проблемы с git merge
REM Файл dumps/ict_viz_BTCUSDT.json - временный файл дампа

echo ========================================
echo Fixing git merge issue
echo ========================================
echo.

echo Step 1: Resetting changes in dumps/ict_viz_BTCUSDT.json...
git checkout -- dumps/ict_viz_BTCUSDT.json
if %errorlevel% neq 0 (
    echo Warning: Could not reset file. It may not be tracked.
)

echo.
echo Step 2: Removing dumps/ from git tracking (if tracked)...
git rm -r --cached dumps/ 2>nul
if %errorlevel% equ 0 (
    echo dumps/ removed from git tracking
) else (
    echo dumps/ is not tracked by git (this is OK)
)

echo.
echo Step 3: Pulling latest changes...
git pull origin main

echo.
echo ========================================
echo Done! You can now continue working.
echo ========================================
pause
