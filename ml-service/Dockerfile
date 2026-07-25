# Python 3.11 — верхняя граница из pyproject (>=3.11,<3.12)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential/cmake — insightface собирается из исходников (Cython + C++).
# libglib2.0-0 нужен opencv даже в headless-сборке, curl — для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# insightface требует numpy и Cython уже на этапе сборки, а изоляция сборки pip
# их туда не передаёт — ставим заранее и собираем без изоляции.
RUN pip install numpy==1.26.4 cython \
    && pip install --no-build-isolation insightface==0.7.3

COPY requirements.txt ./
# CPU-сборка torch: с обычного PyPI подтянулись бы CUDA-библиотеки (+2 ГБ к образу).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml ./
COPY app ./app

# Веса моделей в образ не кладём: 855 МБ и они в .gitignore — монтируются томом.
EXPOSE 8000

# uvicorn напрямую, минуя app.main:main(): там при ML_ENV=development включается
# reload, который в контейнере не нужен.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
