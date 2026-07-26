<#
.SYNOPSIS
    Создаёт venv на Python 3.11 и ставит зависимости ML-сервиса.

.DESCRIPTION
    Компилятор C++ больше не нужен: insightface убран, локальных весов нет.
    Замена лица выполняется на fal.ai — задайте FAL_KEY перед запуском.

.EXAMPLE
    .\scripts\setup_venv.ps1
    .\scripts\setup_venv.ps1 -Recreate
#>
[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$NoDev
)

$ErrorActionPreference = 'Stop'

$serviceRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $serviceRoot '.venv'
$python = Join-Path $venvPath 'Scripts\python.exe'

Write-Host "=> Каталог сервиса: $serviceRoot" -ForegroundColor Cyan

# --- 1. Интерпретатор 3.11 ---
if (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = 'py'
    $baseArgs = @('-3.11')
} else {
    $basePython = 'python'
    $baseArgs = @()
    $version = & python -c 'import sys; print("%d.%d" % sys.version_info[:2])'
    if ($version -ne '3.11') {
        Write-Warning "Найден Python $version, сервис рассчитан на 3.11 (mediapipe)."
    }
}

# --- 2. venv ---
if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host '=> Удаляю существующее окружение' -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (Test-Path $python) {
    Write-Host '=> venv уже существует, переиспользую' -ForegroundColor DarkGray
} else {
    Write-Host '=> Создаю виртуальное окружение .venv' -ForegroundColor Cyan
    & $basePython @baseArgs -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'venv create failed' }
}

# --- 3. Зависимости ---
Write-Host '=> Обновляю pip' -ForegroundColor Cyan
& $python -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

Write-Host '=> Устанавливаю requirements.txt' -ForegroundColor Cyan
& $python -m pip install -r (Join-Path $serviceRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

# --- 4. .env ---
$envFile = Join-Path $serviceRoot '.env'
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $serviceRoot '.env.example') $envFile
    Write-Host '=> Создан .env из .env.example' -ForegroundColor Cyan
}

# --- 5. Dev-зависимости (pytest, httpx, ruff) ---
if (-not $NoDev) {
    Write-Host '=> Устанавливаю dev-зависимости' -ForegroundColor Cyan
    & $python -m pip install -r (Join-Path $serviceRoot 'requirements-dev.txt') --quiet
}

# --- 6. Проверка ---
# ErrorActionPreference=Stop превращает stderr нативной команды в исключение,
# поэтому блок проверок выполняем в режиме Continue.
$ErrorActionPreference = 'Continue'

Write-Host '=> Проверяю установку' -ForegroundColor Cyan
& $python -c "import fastapi, mediapipe, cv2, fal_client; print('  mediapipe', mediapipe.__version__, '| opencv', cv2.__version__)"

Write-Host ''
Write-Host 'Готово. Запуск сервиса:' -ForegroundColor Green
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  $env:FAL_KEY = "..."'
Write-Host '  uvicorn app.main:app --reload --port 8000'
