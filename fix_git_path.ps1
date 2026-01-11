# Скрипт для добавления Git в PATH (запустите от имени администратора)
# Правой кнопкой мыши -> "Запустить от имени администратора"

Write-Host "=== Добавление Git в PATH ===" -ForegroundColor Cyan

# Возможные пути установки Git
$possiblePaths = @(
    "C:\Program Files\Git\bin",
    "C:\Program Files (x86)\Git\bin",
    "$env:LOCALAPPDATA\Programs\Git\bin",
    "$env:USERPROFILE\AppData\Local\Programs\Git\bin"
)

$gitPath = $null

# Ищем Git
foreach ($path in $possiblePaths) {
    if (Test-Path "$path\git.exe") {
        $gitPath = $path
        Write-Host "Git найден: $gitPath" -ForegroundColor Green
        break
    }
}

if (-not $gitPath) {
    Write-Host "Git не найден в стандартных местах." -ForegroundColor Red
    Write-Host "Пожалуйста, установите Git с https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Или укажите путь к Git вручную:" -ForegroundColor Yellow
    $manualPath = Read-Host "Введите путь к папке bin Git (например: C:\Program Files\Git\bin)"
    if ($manualPath -and (Test-Path "$manualPath\git.exe")) {
        $gitPath = $manualPath
    } else {
        Write-Host "Неверный путь. Выход." -ForegroundColor Red
        exit 1
    }
}

# Получаем текущий PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Проверяем, есть ли уже Git в PATH
if ($currentPath -like "*$gitPath*") {
    Write-Host "Git уже добавлен в PATH пользователя" -ForegroundColor Yellow
} else {
    # Добавляем Git в PATH пользователя
    $newPath = "$currentPath;$gitPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Git добавлен в PATH пользователя: $gitPath" -ForegroundColor Green
    Write-Host "Перезапустите PowerShell, чтобы изменения вступили в силу" -ForegroundColor Yellow
}

# Также добавляем в системный PATH (требует прав администратора)
try {
    $systemPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($systemPath -notlike "*$gitPath*") {
        $newSystemPath = "$systemPath;$gitPath"
        [Environment]::SetEnvironmentVariable("Path", $newSystemPath, "Machine")
        Write-Host "Git добавлен в системный PATH" -ForegroundColor Green
    }
} catch {
    Write-Host "Не удалось добавить в системный PATH (требуются права администратора)" -ForegroundColor Yellow
}

# Проверяем, работает ли Git
Write-Host "`nПроверка работы Git..." -ForegroundColor Cyan
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
try {
    $gitVersion = & "$gitPath\git.exe" --version
    Write-Host "Git работает: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git не работает. Перезапустите PowerShell." -ForegroundColor Yellow
}

Write-Host "`nГотово!" -ForegroundColor Green
