# Python 3.11 — верхняя граница из pyproject (>=3.11,<3.12)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Компилятор больше не нужен: insightface из исходников не собирается.
# libglib2.0-0 и libgl1 требуются opencv/mediapipe, curl — для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Веса сегментатора кладутся в образ на сборке. Иначе их скачивает первый же
# заказ: ~176 МБ на модель, минуты ожидания в задаче, которая должна была
# просто выполниться, — и так в каждом новом контейнере при масштабировании.
ENV U2NET_HOME=/opt/rembg
RUN python -c "from rembg import new_session; [new_session(m) for m in ('u2net_human_seg', 'u2net')]"

# Веса семантической разметки человека (~16 МБ): лицо, волосы, кожа тела,
# одежда. Кладутся на этапе сборки по той же причине, что и веса rembg. Без
# файла сервис работает — разметка просто отключается, и код падает обратно на
# цветовые эвристики, — но кожу от бежевой одежды они не отличают, замерено.
ENV MEDIAPIPE_HOME=/opt/mediapipe
RUN python -c "import urllib.request, os; os.makedirs('/opt/mediapipe', exist_ok=True);     urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',     '/opt/mediapipe/selfie_multiclass_256x256.tflite')"

COPY pyproject.toml ./
COPY app ./app

# Диффузия идёт на fal.ai; локально работают только mediapipe (везёт свою
# модель внутри пакета) и сегментатор rembg с весами из /opt/rembg.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
