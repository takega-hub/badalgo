# Скрипт для деплоя изменений на удаленный сервер через GitHub
# Запускать: .\deploy_to_server.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEPLOY TO SERVER VIA GITHUB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ШАГ 1: Проверка изменений
Write-Host "`n[1/5] Проверка изменений..." -ForegroundColor Yellow
git status

# ШАГ 2: Добавление всех изменений
Write-Host "`n[2/5] Добавление изменений в git..." -ForegroundColor Yellow
git add .

# ШАГ 3: Коммит
Write-Host "`n[3/5] Создание коммита..." -ForegroundColor Yellow
$commitMessage = "Fix strategy signal generation: lower ADX threshold, increase volume limit, add BB touch tolerance"
git commit -m $commitMessage

# ШАГ 4: Push в GitHub
Write-Host "`n[4/5] Отправка в GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ УСПЕШНО ОТПРАВЛЕНО В GITHUB!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# ШАГ 5: Инструкции для сервера
Write-Host "`n[5/5] СЛЕДУЮЩИЕ ШАГИ НА СЕРВЕРЕ:" -ForegroundColor Cyan
Write-Host "1. Подключитесь к серверу:" -ForegroundColor White
Write-Host "   ssh user@your-server" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Перейдите в директорию проекта:" -ForegroundColor White
Write-Host "   cd /path/to/crypto_bot" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Получите изменения:" -ForegroundColor White
Write-Host "   git pull origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Обновите настройки (если нужно):" -ForegroundColor White
Write-Host "   python fix_strategy_settings.py" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Перезапустите бота:" -ForegroundColor White
Write-Host "   sudo systemctl restart crypto-bot" -ForegroundColor Gray
Write-Host "   # или" -ForegroundColor Gray
Write-Host "   sudo supervisorctl restart crypto-bot" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Проверьте статус:" -ForegroundColor White
Write-Host "   sudo systemctl status crypto-bot" -ForegroundColor Gray
Write-Host "   # или" -ForegroundColor Gray
Write-Host "   sudo supervisorctl status crypto-bot" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ГОТОВО!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
