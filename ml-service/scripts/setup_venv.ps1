<#
.SYNOPSIS
    Создаёт venv на Python 3.11 и ставит зависимости ML-сервиса.

.DESCRIPTION
    insightface собирается из исходников и требует Microsoft C++ Build Tools.
    Если компилятора нет, скрипт ставит остальные зависимости и завершается
    с кодом 2: сервис поднимется, /health будет отвечать, а face-swap вернёт
    503 MODEL_NOT_LOADED до установки insightface.

.EXAMPLE
    .\scripts\setup_venv.ps1
    .\scripts\setup_venv.ps1 -Gpu
    .\scripts\setup_venv.ps1 -SkipInsightface
    .\scripts\setup_venv.ps1 -Recreate
#>
[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Recreate,
    [switch]$SkipInsightface,
    [switch]$NoDev
)

$ErrorActionPreference = 'Stop'

$serviceRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $serviceRoot '.venv'
$python = Join-Path $venvPath 'Scripts\python.exe'

Write-Host "=> Каталог сервиса: $serviceRoot" -ForegroundColor Cyan

function Test-MsvcBuildTools {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$found
}

# --- 1. Интерпретатор 3.11 ---
if (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = 'py'
    $baseArgs = @('-3.11')
} else {
    $basePython = 'python'
    $baseArgs = @()
    $version = & python -c 'import sys; print("%d.%d" % sys.version_info[:2])'
    if ($version -ne '3.11') {
        Write-Warning "Найден Python $version, сервис рассчитан на 3.11 (insightface/onnxruntime)."
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

# --- 3. Проверка компилятора ---
$hasMsvc = Test-MsvcBuildTools
if (-not $hasMsvc -and -not $SkipInsightface) {
    Write-Warning 'Microsoft C++ Build Tools не найдены — сборка insightface невозможна.'
    Write-Warning 'Установить: winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools"'
    Write-Warning 'Продолжаю без insightface (face-swap будет недоступен).'
    $SkipInsightface = $true
}

# --- 4. Зависимости ---
Write-Host '=> Обновляю pip' -ForegroundColor Cyan
& $python -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

$requirements = if ($Gpu) { 'requirements-gpu.txt' } else { 'requirements.txt' }
$requirementsPath = Join-Path $serviceRoot $requirements
$tempRequirements = $null

if ($SkipInsightface) {
    $tempRequirements = Join-Path $env:TEMP 'ml-service-requirements.txt'
    Get-Content $requirementsPath | Where-Object { $_ -notmatch '^\s*insightface' } |
        Set-Content -Encoding utf8 $tempRequirements
    $requirementsPath = $tempRequirements
}

Write-Host "=> Устанавливаю $requirements (это надолго: torch ~200 МБ)" -ForegroundColor Cyan
if ($Gpu) { & $python -m pip uninstall -y onnxruntime 2>$null | Out-Null }

& $python -m pip install -r $requirementsPath
$installCode = $LASTEXITCODE
if ($tempRequirements) { Remove-Item $tempRequirements -ErrorAction SilentlyContinue }

if ($installCode -ne 0) {
    Write-Host ''
    Write-Warning 'pip install завершился с ошибкой. Если падает сборка insightface —'
    Write-Warning 'установите C++ Build Tools или запустите скрипт с -SkipInsightface.'
    exit 1
}

# --- 5. .env и каталоги ---
$envFile = Join-Path $serviceRoot '.env'
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $serviceRoot '.env.example') $envFile
    Write-Host '=> Создан .env из .env.example' -ForegroundColor Cyan
}
New-Item -ItemType Directory -Force (Join-Path $serviceRoot 'models') | Out-Null

# --- 6. Dev-зависимости (pytest, httpx, ruff) ---
if (-not $NoDev) {
    Write-Host '=> Устанавливаю dev-зависимости' -ForegroundColor Cyan
    & $python -m pip install pytest==8.3.3 httpx==0.27.2 ruff==0.6.8 --quiet
}

# --- 7. Проверка ---
# ErrorActionPreference=Stop превращает stderr нативной команды в исключение,
# поэтому блок проверок выполняем в режиме Continue.
$ErrorActionPreference = 'Continue'

Write-Host '=> Проверяю установку' -ForegroundColor Cyan
& $python -c "import fastapi, uvicorn, torch, diffusers; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('  diffusers', diffusers.__version__)"

& $python -c "import insightface" *> $null
$hasInsightface = ($LASTEXITCODE -eq 0)

if ($hasInsightface) {
    & $python -c "import insightface; print('  insightface', insightface.__version__)"
} else {
    Write-Host ''
    Write-Warning 'insightface НЕ установлен: /health отвечает ok, /face-swap вернёт 503 MODEL_NOT_LOADED.'
    Write-Host 'После установки C++ Build Tools допоставьте его:' -ForegroundColor DarkGray
    Write-Host '  .\.venv\Scripts\python.exe -m pip install insightface==0.7.3' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host 'Остальное окружение готово. Запуск: uvicorn app.main:app --reload --port 8000' -ForegroundColor Green
    exit 2
}

Write-Host ''
Write-Host 'Готово. Запуск сервиса:' -ForegroundColor Green
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  uvicorn app.main:app --reload --port 8000'
Write-Host ''
Write-Host 'Веса моделей — в ml-service\models (см. README.md).' -ForegroundColor DarkGray
