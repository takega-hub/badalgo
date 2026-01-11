# Скрипт для инициализации Git и подключения к GitHub
# Запустите этот скрипт в PowerShell: .\setup_git.ps1

Write-Host "=== Настройка Git для crypto_bot ===" -ForegroundColor Cyan

# Проверяем, установлен ли Git
try {
    $gitVersion = git --version
    Write-Host "Git найден: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ОШИБКА: Git не установлен. Установите Git с https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# Проверяем, инициализирован ли репозиторий
if (Test-Path .git) {
    Write-Host "Git репозиторий уже инициализирован" -ForegroundColor Yellow
} else {
    Write-Host "Инициализация Git репозитория..." -ForegroundColor Yellow
    git init
    Write-Host "Git репозиторий инициализирован" -ForegroundColor Green
}

# Проверяем, есть ли remote
$remotes = git remote -v
if ($remotes) {
    Write-Host "Текущие remote репозитории:" -ForegroundColor Cyan
    Write-Host $remotes
} else {
    Write-Host "Remote репозитории не настроены" -ForegroundColor Yellow
}

# Запрашиваем URL GitHub репозитория
Write-Host "`nВведите URL вашего GitHub репозитория (например: https://github.com/username/crypto_bot.git)" -ForegroundColor Cyan
Write-Host "Или нажмите Enter, чтобы пропустить этот шаг" -ForegroundColor Gray
$githubUrl = Read-Host "GitHub URL"

if ($githubUrl) {
    # Проверяем, есть ли уже origin
    $originExists = git remote get-url origin 2>$null
    if ($originExists) {
        Write-Host "Remote 'origin' уже существует: $originExists" -ForegroundColor Yellow
        $replace = Read-Host "Заменить? (y/n)"
        if ($replace -eq "y" -or $replace -eq "Y") {
            git remote set-url origin $githubUrl
            Write-Host "Remote 'origin' обновлен" -ForegroundColor Green
        }
    } else {
        git remote add origin $githubUrl
        Write-Host "Remote 'origin' добавлен: $githubUrl" -ForegroundColor Green
    }
    
    # Проверяем подключение
    Write-Host "`nПроверка подключения к GitHub..." -ForegroundColor Yellow
    git remote -v
}

# Показываем статус
Write-Host "`n=== Статус Git ===" -ForegroundColor Cyan
git status

Write-Host "`n=== Следующие шаги ===" -ForegroundColor Cyan
Write-Host "1. Добавьте файлы: git add ." -ForegroundColor Yellow
Write-Host "2. Создайте коммит: git commit -m 'Initial commit'" -ForegroundColor Yellow
Write-Host "3. Отправьте в GitHub: git push -u origin main" -ForegroundColor Yellow
Write-Host "   (или 'git push -u origin master' если используется ветка master)" -ForegroundColor Gray
