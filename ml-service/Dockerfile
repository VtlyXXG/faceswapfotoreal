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

COPY pyproject.toml ./
COPY app ./app

# Локальных весов больше нет: инференс выполняется на fal.ai, а маска лица
# считается mediapipe, который везёт свою модель внутри пакета.
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
