#!/usr/bin/env bash
#
# Развёртывание gpu-inference-server на чистой Ubuntu 22.04 с картой NVIDIA.
# Выполняется от root. Идемпотентен: повторный запуск не ломает уже сделанное.
#
#   scp -r gpu-inference-server root@<host>:/opt/
#   ssh root@<host> 'bash /opt/gpu-inference-server/setup_server.sh'
#
# Что делает по шагам:
#   1. проверяет карту и драйвер          5. ставит зависимости
#   2. ставит системные пакеты            6. проверяет, что torch видит карту
#   3. ставит Python 3.11 (deadsnakes)    7. качает веса
#   4. создаёт venv                       8. поднимает systemd-юнит
#
# CUDA Toolkit НЕ НУЖЕН. Колёса torch везут рантайм CUDA внутри себя, nvcc здесь
# ничего не компилирует. Значение имеет только версия ДРАЙВЕРА — см. шаг 1.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/gpu-inference-server}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$APP_DIR/weights}"
SERVICE_NAME="${SERVICE_NAME:-gpu-inference}"
PORT="${GPU_PORT:-8100}"
# Куда слушать. localhost по умолчанию НАМЕРЕННО: у сервера нет аутентификации,
# и открытый в интернет порт означает, что жечь вашу карту может кто угодно.
# См. шаг 8 — там про доступ с ml-service
BIND_HOST="${GPU_HOST:-127.0.0.1}"
# 24 ГБ на 4090 против 48 у A40: на 1024 связка SDXL+ControlNet занимает ~11 ГБ
# весов плюс активации, и это проходит с запасом. 1536 — уже нет
MAX_SIDE="${GPU_MAX_SIDE:-1024}"

PYTHON_BIN=python3.11
MIN_DRIVER_MAJOR=525

say() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "запускать от root"
[ -f "$APP_DIR/server.py" ] || die "нет $APP_DIR/server.py — сначала скопируйте файлы в $APP_DIR"


# --- 1. Карта и драйвер ------------------------------------------------------
#
# Единственная системная величина, которая здесь действительно важна. Колёса
# cu124 работают на любом драйвере ветки CUDA 12 (>= 525.60.13) благодаря
# minor version compatibility; 550+ — родная для 12.4 и предпочтительна.

say "1. Карта и драйвер"
command -v nvidia-smi >/dev/null || die "нет nvidia-smi: образ без драйвера NVIDIA"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
DRIVER_MAJOR="${DRIVER%%.*}"
if [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]; then
    die "драйвер $DRIVER старше $MIN_DRIVER_MAJOR — колёса cu124 не заработают.
     Либо возьмите образ с драйвером 550+, либо переведите requirements.txt
     на cu118 (замените строку --extra-index-url на .../whl/cu118)"
fi
echo "драйвер $DRIVER — подходит для cu124"

FREE_GB="$(df -BG --output=avail "$APP_DIR" | tail -1 | tr -dc '0-9')"
[ "$FREE_GB" -ge 40 ] || echo "ВНИМАНИЕ: свободно ${FREE_GB} ГБ. Весам нужно ~12 ГБ плюс
столько же временно на кэш загрузки и колёса. Рекомендуется 40 ГБ."


# --- 2. Системные пакеты -----------------------------------------------------
#
# libglib2.0-0 нужен opencv-python-headless: headless-сборка обходится без
# libGL и GTK, но libgthread тянет всё равно. libgl1 и libgomp1 — страховка:
# facexlib на шаге 9 может подтянуть полный opencv, которому libGL обязателен.

say "2. Системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git \
    libglib2.0-0 libgl1 libgomp1 \
    build-essential


# --- 3. Python 3.11 ----------------------------------------------------------
#
# Ubuntu 22.04 везёт Python 3.10, а сервер требует 3.11 — и это не про
# «пусть будет посвежее». В 3.11 asyncio.TimeoutError стал псевдонимом
# встроенного TimeoutError; на 3.10 это РАЗНЫЕ классы, и `except TimeoutError`
# в GpuQueue (server.py:233) таймаут очереди не поймает — вместо честного 503
# «перегрузка» клиент получит 500, а тот его не повторяет.

say "3. Python 3.11"
if ! command -v "$PYTHON_BIN" >/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev
fi
"$PYTHON_BIN" --version


# --- 4. Окружение ------------------------------------------------------------

say "4. Виртуальное окружение"
[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
"$PIP" install -q --upgrade pip wheel setuptools


# --- 5. Зависимости ----------------------------------------------------------
#
# --no-cache-dir намеренно: колёса torch и nvidia-* весят вместе около трёх
# гигабайт, и кэш pip держал бы их вторым экземпляром на диске, который здесь
# и так под весами.

say "5. Зависимости (торч ~2.5 ГБ, это долго)"
"$PIP" install --no-cache-dir -r "$APP_DIR/requirements.txt"


# --- 6. Торч видит карту -----------------------------------------------------
#
# Отдельным шагом и с жёстким падением. Молча установившийся CPU-торч — самая
# дорогая из возможных ошибок здесь: сервер поднимется, ответит на /health и
# будет считать SDXL на процессоре минутами вместо секунд.

say "6. Проверка CUDA"
"$PY" - <<'PYCODE'
import sys, torch
print("torch   :", torch.__version__)
print("cuda    :", torch.version.cuda)
print("доступна:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("torch не видит карту — проверьте драйвер и что установлена cu124-сборка")
print("карта   :", torch.cuda.get_device_name(0))
print("память  : %.1f ГБ" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
PYCODE


# --- 7. Веса -----------------------------------------------------------------

say "7. Веса (~12 ГБ)"
cd "$APP_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" download_weights.py --root "$WEIGHTS_DIR" || \
    echo "часть источников не отдалась — смотрите вывод выше; необязательные не блокируют запуск"
"$PY" download_weights.py --root "$WEIGHTS_DIR" --check || true


# --- 8. Служба ---------------------------------------------------------------

say "8. systemd-юнит $SERVICE_NAME"
cat >"/etc/systemd/system/${SERVICE_NAME}.service" <<UNIT
[Unit]
Description=projectx GPU inference (SDXL + ControlNet, LaMa, CodeFormer)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
Environment=GPU_WEIGHTS_ROOT=$WEIGHTS_DIR
Environment=GPU_HOST=$BIND_HOST
Environment=GPU_PORT=$PORT
Environment=GPU_MAX_SIDE=$MAX_SIDE
Environment=HF_HUB_OFFLINE=1
ExecStart=$VENV_DIR/bin/python -m uvicorn server:app --host $BIND_HOST --port $PORT
# Загрузка весов и прогрев занимают до полутора минут; без запаса systemd
# успевает счесть службу зависшей и убить её посреди прогрева
TimeoutStartSec=600
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"


# --- 9. Проверка -------------------------------------------------------------

say "9. Проверка"
for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health/ready" | "$PY" -m json.tool || \
    die "сервис не поднялся: journalctl -u $SERVICE_NAME -n 50"

cat <<INFO

Готово. Дальше:

  журнал          journalctl -u $SERVICE_NAME -f
  перезапуск      systemctl restart $SERVICE_NAME
  состояние       curl -s localhost:$PORT/health/ready | python3 -m json.tool

ДОСТУП. Сервер слушает $BIND_HOST и аутентификации не имеет. Открывать порт
$PORT в интернет нельзя — это чужие заказы на вашей карте. Варианты:
  приватная сеть  GPU_HOST=<адрес в приватной сети> в юните, ufw allow from <IP ml-service>
  туннель         ssh -N -L $PORT:127.0.0.1:$PORT root@<host>  (с машины ml-service)

ВОССТАНОВЛЕНИЕ ЛИЦА (необязательно). CodeFormer сейчас выключен: facexlib не
ставится вместе с остальным, потому что тянет полный opencv-python и ссорится
с headless-сборкой за модуль cv2. Порядок, который это разводит:

  $PIP install --no-cache-dir facexlib==0.3.0
  $PIP install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84
  $PY download_weights.py --root $WEIGHTS_DIR --only codeformer facexlib-detection facexlib-parsing
  systemctl restart $SERVICE_NAME

Сам CodeFormer к тому же требует кода архитектуры — см. докстринг FaceRestorer
в server.py. Без него restore выключается, остальное работает.
INFO
