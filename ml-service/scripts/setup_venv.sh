#!/usr/bin/env bash
# Создаёт venv на Python 3.11 и ставит зависимости ML-сервиса (Linux/macOS).
#   ./scripts/setup_venv.sh [--recreate] [--no-dev]
#
# Компилятор C++ больше не нужен: insightface убран, локальных весов нет.
# Замена лица выполняется на fal.ai — не забудьте задать FAL_KEY.
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$SERVICE_ROOT/.venv"
RECREATE=0

for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
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

# --- 3. Зависимости ---
echo "=> Обновляю pip"
"$PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

echo "=> Устанавливаю requirements.txt"
"$PYTHON" -m pip install -r "$SERVICE_ROOT/requirements.txt"

# --- 4. .env ---
[ -f "$SERVICE_ROOT/.env" ] || { cp "$SERVICE_ROOT/.env.example" "$SERVICE_ROOT/.env"; echo "=> Создан .env"; }

# --- 5. Dev-зависимости (pytest, httpx, ruff) ---
if [ "${NO_DEV:-0}" -eq 0 ]; then
  echo "=> Устанавливаю dev-зависимости"
  "$PYTHON" -m pip install -r "$SERVICE_ROOT/requirements-dev.txt" --quiet
fi

# --- 6. Проверка ---
echo "=> Проверяю установку"
"$PYTHON" -c "import fastapi, mediapipe, cv2, fal_client; print('  mediapipe', mediapipe.__version__, '| opencv', cv2.__version__)"

cat <<'EOF'

Готово. Запуск сервиса:
  source .venv/bin/activate
  export FAL_KEY=...
  uvicorn app.main:app --reload --port 8000
EOF
