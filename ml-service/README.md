# ml-service — ML-микросервис (Python 3.11 + FastAPI)

Face-swap для персонализации иллюстраций: лицо получателя книги переносится
на сгенерированные картинки. Вызывается из Node.js API по HTTP.

> Сервис предназначен для работы с фотографиями, на использование которых
> получено согласие (заказчик книги и её получатель).

## Установка

```powershell
cd ml-service
.\scripts\setup_venv.ps1                   # CPU
.\scripts\setup_venv.ps1 -Gpu              # CUDA 12.1
.\scripts\setup_venv.ps1 -SkipInsightface  # без face-swap
.\scripts\setup_venv.ps1 -Recreate         # пересоздать окружение
.\scripts\setup_venv.ps1 -NoDev            # без pytest/httpx/ruff
```

Linux/macOS:

```bash
./scripts/setup_venv.sh [--gpu] [--recreate] [--skip-insightface] [--no-dev]
```

Скрипт: находит Python 3.11 → создаёт `.venv` → проверяет наличие C++ компилятора →
ставит `requirements.txt` + dev-зависимости → копирует `.env.example` в `.env` →
создаёт `models/` → проверяет импорты.

Коды выхода: `0` — всё установлено, `2` — установлено без `insightface`,
`1` — установка не удалась.

### insightface требует C++ компилятор

У пакета нет готового wheel: он собирается из исходников.

- **Windows** — [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
  workload «Desktop development with C++»:
  ```powershell
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools"
  .\.venv\Scripts\python.exe -m pip install insightface==0.7.3
  ```
- **Debian/Ubuntu** — `sudo apt install build-essential python3.11-dev`

Компилятора нет → скрипт не падает, а ставит всё остальное и предупреждает.
Сервис при этом работает: `/health` отвечает `ok`, `/face-swap` — `503 MODEL_NOT_LOADED`.

> `setup_venv.ps1` сохранён в UTF-8 **с BOM**: Windows PowerShell 5.1 читает `.ps1`
> без BOM как ANSI и ломается на кириллице в строках. При правке файла редактором
> сохраняйте кодировку.

## Запуск

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Swagger: <http://localhost:8000/docs>

## Веса моделей

Кладутся в `ml-service/models/` (в git и в контекст ассистента не попадают):

| Файл | Назначение |
| --- | --- |
| `models/models/buffalo_l/` | детекция лиц и эмбеддинги, скачивается insightface автоматически при первом запросе |
| `models/inswapper_128.onnx` | модель переноса лица, кладётся вручную |

До появления весов `/health` отвечает `ok`, а `/health/ready` — `503 degraded`.
Сервис при этом не падает.

`/health/ready` разделяет две причины деградации — нет пакетов или нет весов:

```json
{
  "status": "degraded",
  "device": "cpu",
  "runtime": { "insightface": true, "onnxruntime": true, "torch": true },
  "detector": { "name": "buffalo_l", "loaded": false, "available": false },
  "swapper": { "name": "inswapper_128.onnx", "loaded": false, "available": false },
  "reason": "нет весов детектора buffalo_l; нет весов inswapper_128.onnx"
}
```

## Эндпоинты

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/health` | liveness — процесс жив, веса не проверяются |
| `GET` | `/health/ready` | readiness — устройство и наличие весов, `503` если не готов |
| `POST` | `/face-swap/analyse` | детекция лиц (multipart `image`) |
| `POST` | `/face-swap` | перенос лица, возвращает изображение + заголовок `X-Swap-Meta` |

Параметры `/face-swap`: `source`, `target`, `target_face_index`, `swap_all_faces`,
`enhance` (постобработка), `style_strength` (0–2), `output_format`.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/face-swap \
  -F source=@photo.jpg -F target=@illustration.png \
  -o result.png -D -
```

## Постобработка: согласование лица с иллюстрацией

`enhance=true` включает стадию, которая подгоняет перенесённое лицо под
стилистику обложки. Baseline работает на OpenCV, без весов и за миллисекунды.

Три стадии, каждая регулируется отдельно:

| Стадия | Что делает | Параметр |
| --- | --- | --- |
| Цвет | переносит статистики LAB из кольца вокруг лица | `ML_STYLE_COLOR` |
| Микротекстура | убирает поры и фотографический микроконтраст | `ML_STYLE_SMOOTH` |
| Фактура | заменяет высокие частоты лица фактурой иллюстрации | `ML_STYLE_GRAIN`, `ML_STYLE_SHARPNESS` |

Ключевые решения, добытые замерами:

- **Разброс кольца не является эталоном контраста.** Кольцо содержит листву,
  небо, животных, и его дисперсия отражает разнородность сюжета, а не фактуру
  кожи. Подтягивание контраста лица к ней ухудшало метрику вдвое, поэтому
  переносится прежде всего среднее, а масштаб ограничен диапазоном 0.85–1.2.
- **Резкость и зерно — одна стадия, а не две.** Раздельные «согласовать
  резкость» и «добавить зерно» складывали энергии и уводили лицо дальше от
  окружения, чем было до обработки. Теперь высокие частоты смешиваются и
  нормируются по энергии кольца: сколько бы ни было зерна, итог совпадает
  с окружением, а повторный прогон ничего не наращивает.
- **Донорская фактура клиппится по перцентилю.** Без этого на кожу переносятся
  структурные края (пряди волос, контуры предметов) и читаются как призрачные
  штрихи.

### Метрики

Считаются на каждом запросе и возвращаются в `X-Swap-Meta`:

- `identity_similarity` — косинус ArcFace-эмбеддингов донора и результата.
  Порог узнаваемости 0.55, ниже 0.45 — уже другой человек. Падение ниже порога
  логируется как warning.
- `texture_before` / `texture_after` в логе — рассогласование фактуры лица и
  окружения (энергия высоких частот + моменты LAB). 0 означает, что область
  лица статистически неотличима от иллюстрации вокруг.

Замер на 4K-обложке: рассогласование **0.1285 → 0.045** (−65%),
узнаваемость **0.847**.

### Что baseline не делает

Мазок кисти он не рисует: контур маски на границе лба и скул при 100% всё ещё
различим. Это задача диффузионной стадии — она встанет на то же место через
`BaseStylizer`, конфигурацией `ML_STYLE_PROVIDER`.

### A/B-стенд для подбора параметров

`python -m bench` гоняет набор обложек через пайплайн при нескольких значениях
силы, считает метрики и собирает HTML со сравнением. Нужен, чтобы параметры
постобработки (а затем диффузии) подбирались по числам и картинке на наборе, а
не на глаз по одному кадру. Подробности — в `bench/README.md`.

## Логирование и трассировка

JSON-логи пишутся в `logs/`:

| Файл | Содержимое |
| --- | --- |
| `logs/app.log` | активный файл: все записи, включая логи uvicorn |
| `logs/error.log` | активный файл: только `error` и выше |
| `logs/app.2026-07-23.1.log` | ротированные файлы |

### Ротация

`app/core/rotation.py` — `TimedRotatingFileHandler`, расширенный проверкой
размера, чтобы политика совпадала с Node (`pino-roll`):

| Переменная | По умолчанию | Значение |
| --- | --- | --- |
| `ML_LOG_ROTATE_WHEN` | `midnight` | когда ротировать (`midnight`, `H`, `D`, `S`) |
| `ML_LOG_FILE_MAX_MB` | `20` | досрочная ротация по объёму |
| `ML_LOG_BACKUP_COUNT` | `14` | сколько ротированных файлов хранить |

Файл роняется по наступлению суток либо раньше при превышении объёма; имя
ротированного файла — `app.<дата>.<номер>.log`, как у Node. Индекс нужен для
нескольких ротаций по размеру за одни сутки.

Два отличия от Node — следствие устройства stdlib:

- активным всегда остаётся `app.log`, дата появляется только у ротированных
  файлов (у `pino-roll` датирован и активный файл);
- удаление старых файлов реализовано в `_prune()`: штатное опирается на схему
  имён stdlib и не находит файлы, переименованные `namer`. Сортировка идёт по
  дате и индексу из имени, а не по `mtime` — на Windows у файлов одной серии
  ротаций метки времени совпадают.

В консоль по умолчанию идёт читаемый формат; `ML_LOG_JSON=true` включает JSON и
там. Поля совпадают с Node-сервисом (`time`, `level`, `service`, `scope`,
`message`, `request_id`), поэтому логи двух сервисов грепаются одинаково:

```json
{"time": "2026-07-23T03:49:04.369Z", "level": "info", "service": "ml-service",
 "scope": "http", "message": "GET /health/ready → 503",
 "request_id": "trace-e2e-1784778544", "status_code": 503, "duration_ms": 3.84}
```

`RequestIdMiddleware` принимает `X-Request-ID` от Node.js API (или генерирует
UUID), кладёт его в `contextvar`, возвращает в заголовке ответа и пишет
access-лог со статусом и длительностью. Штатный access-лог uvicorn приглушён до
`WARNING`, чтобы не дублировать записи без `request_id`.

Произвольные поля добавляются через `extra`:

```python
log.info("глава обработана", extra={"chapter": 3, "duration_ms": 812})
```

## Структура

```
app/
  main.py               точка входа, lifespan, CORS, обработчики ошибок
  config.py             все настройки из ENV (префикс ML_)
  api/
    routes/health.py    /health и /health/ready — связь с Node.js API
    routes/face_swap.py /face-swap, /face-swap/analyse
    schemas.py          контракт ответов для Node.js
    middleware/request_id.py  приём/генерация X-Request-ID + access-лог
  core/
    logging.py          JSON-формат, хендлеры, фильтр request_id
    rotation.py         ротация по времени и размеру, уборка старых файлов
    context.py          contextvar с request_id
    errors.py           доменные ошибки → HTTP-коды
  pipelines/
    registry.py         ленивая загрузка моделей, потокобезопасные синглтоны
    face_swap/
      detector.py       детекция и выбор лица
      swapper.py        перенос (inswapper)
      enhancer.py       обёртка над стилизатором
    style/
      base.py           контракт BaseStylizer + StyleOptions
      classical.py      baseline на OpenCV (текущий), noop
      masking.py        маски: кожа, зона идентичности, кольцо-эталон
      color.py          согласование цвета, микротекстуры и фактуры
      metrics.py        узнаваемость и рассогласование фактуры
bench/                  A/B-стенд подбора параметров постобработки (см. bench/README.md)
      pipeline.py       оркестрация, единственный вход для API-слоя
  utils/image.py        кодирование/декодирование, BGR ndarray внутри
models/                 веса (в .gitignore и .claudeignore)
scripts/                setup_venv.ps1 / setup_venv.sh
tests/                  pytest, работают без весов
```

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Тесты `/health` и `/health/ready` проходят без весов и без `insightface`.

## Интеграция с Node.js API

Node-сервис ходит на `http://localhost:8000`. Проверка готовности перед
включением персонализированных иллюстраций:

```js
const res = await fetch('http://localhost:8000/health/ready');
const ready = res.ok; // 503 → face-swap отключён, книга генерируется без него
```
