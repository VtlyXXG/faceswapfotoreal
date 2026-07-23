#!/usr/bin/env bash
# Создаёт venv на Python 3.11 и ставит зависимости ML-сервиса (Linux/macOS).
#   ./scripts/setup_venv.sh [--gpu] [--recreate] [--skip-insightface] [--no-dev]
#
# insightface собирается из исходников и требует компилятор C++ (build-essential).
# Без него используйте --skip-insightface: сервис поднимется, /health ответит ok,
# а /face-swap вернёт 503 MODEL_NOT_LOADED. Код выхода 2 = установлено частично.
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$SERVICE_ROOT/.venv"
REQUIREMENTS="requirements.txt"
RECREATE=0
SKIP_INSIGHTFACE=0

for arg in "$@"; do
  case "$arg" in
    --gpu) REQUIREMENTS="requirements-gpu.txt" ;;
    --recreate) RECREATE=1 ;;
    --skip-insightface) SKIP_INSIGHTFACE=1 ;;
    --no-dev) NO_DEV=1 ;;
    *) echo "неизвестный аргумент: $arg" >&2; exit 1 ;;
  esac
done

# --- 1. Интерпретатор ---
if command -v python3.11 >/dev/null 2>&1; then
  BASE_PYTHON="python3.11"
else
  BASE_PYTHON="python3"
  version="$("$BASE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [ "$version" = "3.11" ] || echo "ВНИМАНИЕ: найден Python $version, сервис рассчитан на 3.11" >&2
fi

# --- 2. venv ---
if [ "$RECREATE" -eq 1 ] && [ -d "$VENV" ]; then
  echo "=> Удаляю существующее окружение"
  rm -rf "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
  echo "=> venv уже существует, переиспользую"
else
  echo "=> Создаю виртуальное окружение .venv"
  "$BASE_PYTHON" -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"

# --- 3. Компилятор для insightface ---
if [ "$SKIP_INSIGHTFACE" -eq 0 ] && ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  echo "ВНИМАНИЕ: компилятор C++ не найден — сборка insightface невозможна." >&2
  echo "         Debian/Ubuntu: sudo apt install build-essential python3.11-dev" >&2
  echo "         Продолжаю без insightface (face-swap будет недоступен)." >&2
  SKIP_INSIGHTFACE=1
fi

# --- 4. Зависимости ---
echo "=> Обновляю pip"
"$PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

REQ_PATH="$SERVICE_ROOT/$REQUIREMENTS"
TEMP_REQ=""
if [ "$SKIP_INSIGHTFACE" -eq 1 ]; then
  TEMP_REQ="$(mktemp)"
  grep -v '^[[:space:]]*insightface' "$REQ_PATH" > "$TEMP_REQ"
  REQ_PATH="$TEMP_REQ"
fi

echo "=> Устанавливаю $REQUIREMENTS (это надолго: torch ~200 МБ)"
[ "$REQUIREMENTS" = "requirements-gpu.txt" ] && "$PYTHON" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true

set +e
"$PYTHON" -m pip install -r "$REQ_PATH"
INSTALL_CODE=$?
set -e
[ -n "$TEMP_REQ" ] && rm -f "$TEMP_REQ"

if [ "$INSTALL_CODE" -ne 0 ]; then
  echo "pip install завершился с ошибкой. Если падает сборка insightface —" >&2
  echo "поставьте build-essential или запустите скрипт с --skip-insightface." >&2
  exit 1
fi

# --- 5. .env и каталоги ---
[ -f "$SERVICE_ROOT/.env" ] || { cp "$SERVICE_ROOT/.env.example" "$SERVICE_ROOT/.env"; echo "=> Создан .env"; }
mkdir -p "$SERVICE_ROOT/models"

# --- 6. Dev-зависимости (pytest, httpx, ruff) ---
if [ "${NO_DEV:-0}" -eq 0 ]; then
  echo "=> Устанавливаю dev-зависимости"
  "$PYTHON" -m pip install pytest==8.3.3 httpx==0.27.2 ruff==0.6.8 --quiet
fi

# --- 7. Проверка ---
echo "=> Проверяю установку"
"$PYTHON" -c "import fastapi, uvicorn, torch, diffusers; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('  diffusers', diffusers.__version__)"

if ! "$PYTHON" -c "import insightface" >/dev/null 2>&1; then
  echo ""
  echo "ВНИМАНИЕ: insightface НЕ установлен: /health отвечает ok, /face-swap вернёт 503." >&2
  echo "Допоставить: .venv/bin/python -m pip install insightface==0.7.3" >&2
  exit 2
fi
"$PYTHON" -c "import insightface; print('  insightface', insightface.__version__)"

cat <<'EOF'

Готово. Запуск сервиса:
  source .venv/bin/activate
  uvicorn app.main:app --reload --port 8000
EOF
