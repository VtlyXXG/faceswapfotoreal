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

# Веса семантической разметки человека (~16 МБ): лицо, волосы, кожа тела,
# одежда. По ним строится маска головы на шаблоне. Кладутся на этапе сборки,
# иначе их качает первый же заказ — в каждом новом контейнере при
# масштабировании. Без файла сервис работает: маска строится эллипсом по сетке
# лица, но формы причёски эллипс не знает и берёт её с запасом.
ENV MEDIAPIPE_HOME=/opt/mediapipe
RUN python -c "import urllib.request, os; os.makedirs('/opt/mediapipe', exist_ok=True);     urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',     '/opt/mediapipe/selfie_multiclass_256x256.tflite')"

COPY pyproject.toml ./
COPY app ./app

# Генерация идёт на fal.ai; локально работает только mediapipe — сетка лица
# (модель внутри пакета) и разметка человека с весами из /opt/mediapipe.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
