# syntax=docker/dockerfile:1.7
#
# Образ gpu-inference-server: SDXL + ControlNet inpaint + IP-Adapter Plus Face,
# LaMa, CodeFormer. Веса запекаются внутрь — сервер работает оффлайн
# (HF_HUB_OFFLINE=1 ставит себе сам, см. server.py).
#
# БАЗОВЫЙ ОБРАЗ ВЫБРАН ПОД ПИНЫ ПРОЕКТА, А НЕ НАОБОРОТ.
#
# requirements.txt требует Python 3.11 (граница та же, что у ml-service),
# torch==2.4.1 и torchvision==0.19.1 с индекса cu124. Образ ниже несёт ровно
# это: pip видит требования удовлетворёнными и НЕ качает второй торч. На
# python3.12-образе с торчем 2.11 произошло бы обратное — pip всё равно откатил
# бы торч до 2.4.1 по нашему же пину, то есть мы скачали бы его дважды и
# получили питон вне заявленной границы.
#
# cu124 на драйвере 570.211.01 работает: 12.8 — это потолок драйвера, а не
# требование. RTX 3090 (sm_86) обслуживается сборкой cu124 штатно.
#
# Сменить базу, если понадобится, — одной строкой; тогда вернутся и
# --break-system-packages (тот образ externally-managed), и путь до
# site-packages поменяется — он ниже вычисляется, а не прописан константой.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GPU_WEIGHTS_ROOT=/app/weights \
    HF_HOME=/app/weights/.hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

# --- 1. Системные библиотеки -------------------------------------------------
#
# libgl1 + libglib2.0-0 — под opencv. Headless-сборка обходится без libGL, но
# facexlib и basicsr ниже тянут ПОЛНЫЙ opencv-python как зависимость, и на
# первом импорте cv2 он ищет libGL. Ставим обе, чтобы порядок установки pip не
# решал, поднимется контейнер или нет.
# curl — вендоринг архитектуры CodeFormer и HEALTHCHECK.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 2. Зависимости Python ---------------------------------------------------
#
# Отдельным слоем и до всего тяжёлого: правка server.py не должна пересобирать
# ни pip, ни веса.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch, torchvision, diffusers, numpy; \
print('torch', torch.__version__, '| tv', torchvision.__version__, \
'| diffusers', diffusers.__version__, '| numpy', numpy.__version__)"

# --- 3. Стек восстановления лица (CodeFormer) --------------------------------
#
# Вынесено из requirements.txt намеренно — там это закомментировано с
# объяснением: facexlib объявляет зависимостью ПОЛНЫЙ opencv-python и кладёт
# свой cv2 поверх нашего headless. basicsr делает то же самое. Поэтому оба
# ставятся первыми, а headless возвращается принудительной переустановкой
# ПОСЛЕ них — порядок ровно как в setup_server.sh, шаг 9.
#
# basicsr нужен не сам по себе: codeformer_arch.py импортирует из него
# ARCH_REGISTRY, get_root_logger и vqgan_arch.
RUN pip install --no-cache-dir basicsr==1.4.2 facexlib==0.3.0 \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84 \
    && python -c "import cv2; print('cv2', cv2.__version__)"

# --- 4. Патч basicsr под torchvision 0.19 ------------------------------------
#
# basicsr 1.4.2 импортирует torchvision.transforms.functional_tensor, который
# выпилен начиная с torchvision 0.17. Без правки падает ЛЮБОЙ импорт basicsr,
# в том числе транзитом через codeformer_arch.
#
# Путь до пакета вычисляется, а не прописан: он зависит от базового образа
# (conda-окружение против системного dist-packages), и константа здесь — это
# тихая поломка при смене базы. Импорт после sed — проверка, что починилось.
RUN BASICSR="$(python -c 'import basicsr, pathlib; print(pathlib.Path(basicsr.__file__).parent)')" \
    && echo "basicsr: $BASICSR" \
    && sed -i \
        's/from torchvision\.transforms\.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
        "$BASICSR/data/degradations.py" \
    && python -c "import basicsr.data.degradations; print('basicsr patched OK')"

# --- 5. Вендоринг архитектуры CodeFormer -------------------------------------
#
# Два файла и в РАЗНЫЕ места, потому что их ищут разные механизмы:
#
#   vqgan_arch.py      -> внутрь пакета basicsr/archs/. Так его требует сам
#                         codeformer_arch.py строкой
#                         `from basicsr.archs.vqgan_arch import *`.
#   codeformer_arch.py -> рядом с server.py, то есть в /app. Так его ищет НАШ
#                         код: FaceRestorer._arch() делает
#                         sys.path.insert(0, Path(__file__).parent) и затем
#                         `from codeformer_arch import CodeFormer`.
#
# Класть codeformer_arch.py в basicsr/archs (как в упрощённой версии) здесь
# нельзя: наш загрузчик туда не смотрит, восстановление лица молча выключится с
# сообщением «нет кода архитектуры».
RUN BASICSR="$(python -c 'import basicsr, pathlib; print(pathlib.Path(basicsr.__file__).parent)')" \
    && curl -fsSL -o "$BASICSR/archs/vqgan_arch.py" \
        https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/vqgan_arch.py \
    && curl -fsSL -o /app/codeformer_arch.py \
        https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/codeformer_arch.py \
    && python -c "import sys; sys.path.insert(0, '/app'); \
from codeformer_arch import CodeFormer; \
CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, \
connect_list=['32','64','128','256']); print('CodeFormer arch OK')"

# --- 6. Веса ------------------------------------------------------------------
#
# Самый тяжёлый слой (~14 ГБ) и потому предпоследний: всё, что меняется часто,
# лежит ниже. Копируется только манифест — server.py сюда не входит намеренно.
#
# Токен приходит секретом buildkit, а не ARG/ENV: ARG оседает в метаданных
# слоя. required=false — ни один репозиторий манифеста сейчас не gated, сборка
# без секрета проходит.
COPY download_weights.py ./
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python download_weights.py --root "$GPU_WEIGHTS_ROOT" \
    && python download_weights.py --root "$GPU_WEIGHTS_ROOT" --check

# --- 7. Код -------------------------------------------------------------------
#
# Последним слоем. Правка server.py пересобирает только его.
COPY server.py ./

EXPOSE 8100

# start-period большой намеренно: на старте грузятся SDXL, ControlNet,
# IP-Adapter, LaMa и CodeFormer, затем идёт прогревочный кадр. На 3090 это
# минуты, и без запаса докер убил бы контейнер посреди загрузки.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8100/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
