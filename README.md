# projectx — сервис персонализации обложек

Node.js (ESM) + Express. Единственная содержательная операция — **замена лица
на обложке**: заказчик присылает своё фото и обложку, сервис возвращает обложку
с его лицом. Вся тяжёлая работа живёт в Python ML-сервисе (`ml-service/`):
голова заказчика вместе с причёской переносится локально жёсткой аппликацией, а
инпейнтинг на fal.ai сводит только стык — подробности в `ml-service/README.md`.
Node ничего не генерирует сам: он принимает заказы, ведёт их состояние и
оркестрирует вызовы ML.

## Быстрый старт

```bash
cp .env.example .env
npm install
npm run dev
```

Node-части нужен только запущенный ML-сервис (`ML_SERVICE_URL`); как поднять
его — см. `ml-service/README.md`.

### Локальная инфраструктура (Postgres + Redis)

Для запуска в автономном режиме `docker-compose.yml` поднимает Postgres и Redis:

```bash
docker compose up -d postgres redis   # только базы (без сборки образов)
cp .env.example .env                  # .env уже указывает на localhost:5432 / :6379
npm run check:infra                   # прогнать драйверы БД и очереди на живых базах
npm start                             # API + воркер против Postgres и Redis
```

`npm run check:infra` (`scripts/check-infra.mjs`) гоняет `PostgresCoverStore`
(полный контракт хранилища) и `BullMqDriver` (жизненный цикл задачи, повторы,
stats) против контейнеров и печатает PASS/FAIL по шагам.

Полный стек (сборка образов + все сервисы):

```bash
docker compose up -d --build                        # api, worker, postgres, redis
docker compose up -d --scale worker=3               # три воркера
```

`api` и `worker` — один образ, разные команды; оба ждут healthcheck
`postgres`/`redis`. Адрес ML-сервиса задаётся `ML_SERVICE_URL`.

**Роли жёстко разделены:** контейнер `api` поднимается с `WORKER_IN_API=false`
и только принимает запросы, отдавая `202`. Все задачи выполняют контейнеры
`worker` — их и масштабируют под нагрузку. Состояние заказов общее в Postgres,
очередь в Redis, артефакты — на общем томе.

## Структура

```
src/
  api/
    routes/             маршруты /api/v1
    controllers/        HTTP-слой, без бизнес-логики
    middleware/         requestId, asyncHandler, errorHandler
  domain/
    cover.js            заказ обложки: спецификация + состояние выполнения
    coverSpec.js        zod-схема заказа — единственный источник правды
  services/
    mlClient.js         клиент ML-сервиса, пробрасывает X-Request-ID
    faceSwapService.js  сама замена лица: пути, вызов ML, задача ml.faceSwap
    coverService.js     приём заказа и его пайплайн (задача cover.faceSwap)
  storage/
    coverRepository.js  репозиторий заказов за контрактом драйвера
    drivers/memoryCoverStore.js   in-process (Map)
    drivers/postgresCoverStore.js БД: jsonb-документ + индексированные колонки
    fileStorage.js      файловые артефакты
  queue/                асинхронная очередь задач
    task.js             модель задачи + статусы
    taskQueue.js        фасад: выбор драйвера, регистрация обработчиков
    drivers/memoryDriver.js  in-process: параллелизм, повторы, backoff
    drivers/bullmqDriver.js  BullMQ + Redis: durable, распределённо
  utils/
    logger.js           pino: pretty в консоль, JSON в logs/ с ротацией
    requestContext.js   AsyncLocalStorage с request_id
    errors.js, id.js
  config/index.js       вся конфигурация из ENV в одном месте
  app.js                сборка Express-приложения
  server.js             API; воркер в этом же процессе — только при WORKER_IN_API=true
  worker.js             автономный воркер (для QUEUE_DRIVER=redis)
ml-service/             Python (FastAPI): детекция, аппликация головы, сведение стыка
tests/                  node:test
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `POST` | `/api/v1/covers` | создать заказ обложки, `202` + `taskId` |
| `GET` | `/api/v1/covers` | список заказов |
| `GET` | `/api/v1/covers/:id` | полное состояние заказа |
| `GET` | `/api/v1/covers/:id/status` | краткий статус и прогресс |
| `GET` | `/api/v1/covers/:id/files` | список готовых файлов |
| `GET` | `/api/v1/covers/:id/files/:filename` | скачать файл |
| `POST` | `/api/v1/personalize` | разовый face-swap без заказа, `202` + `taskId` |
| `GET` | `/api/v1/tasks/:id` | полное состояние задачи |
| `GET` | `/api/v1/tasks/:id/status` | краткий статус задачи (для опроса) |
| `GET` | `/api/v1/health` | liveness |
| `GET` | `/api/v1/health/ready` | readiness + доступность ML-сервиса |
| `GET` | `/api/v1/stats` | состояние очереди |

Пример заказа:

```bash
curl -X POST http://localhost:3000/api/v1/covers \
  -H 'content-type: application/json' \
  -d '{
    "title": "Путешествие к звёздам",
    "source": "uploads/face.jpg",
    "target": "uploads/cover.png",
    "options": { "enhance": true, "style_strength": 1.0, "art_style": "watercolor" }
  }'
# → 202 { "id": "cover_…", "taskId": "task_…", "status": "pending",
#         "statusUrl": "/api/v1/tasks/task_…/status" }
```

`source` — фото с лицом, `target` — обложка. Оба поля это ссылки на файлы под
`STORAGE_ROOT` (выход за его пределы отклоняется). `options` уходят в ML-сервис
как есть. Готовая обложка ложится в артефакты заказа:
`GET /api/v1/covers/:id/files/cover.png`.

### Два входа, одна работа

`POST /covers` и `POST /personalize` делают одно и то же — замену лица; общий код
живёт в `faceSwapService.js`. Разница в учёте:

| | `/covers` | `/personalize` |
| --- | --- | --- |
| Доменный заказ в БД | да (`cover_…`, статус, прогресс, история) | нет |
| Результат | артефакт заказа | артефакт задачи |
| Когда | обычный путь: заказ надо потом найти и отдать | разовый прогон, отладка |

## Асинхронная очередь задач

Сервис **никогда не держит клиента на долгой операции**: face-swap на Python
ML-сервисе уходит в фоновую задачу, а API сразу отвечает `202 Accepted` +
`taskId`. Клиент опрашивает `GET /tasks/:id/status`.

### Драйверы

Очередь скрыта за фасадом `src/queue/taskQueue.js`; драйвер выбирается
`QUEUE_DRIVER`:

| Драйвер | Когда | Свойства |
| --- | --- | --- |
| `memory` (по умолчанию) | один узел, разработка | in-process, без инфраструктуры; параллелизм, повторы с экспоненциальным backoff. Очередь теряется при перезапуске |
| `redis` (BullMQ) | автономность, большие нагрузки | задачи переживают перезапуск; воркеры масштабируются **отдельно** от API. `bullmq`/`ioredis` подгружаются лениво |

Тот же приём, что у хранилища: контракт один, реализации подключаются конфигом.
Перейти на Redis — это `QUEUE_DRIVER=redis` без правок кода.

### Разделение ролей и масштабирование

`WORKER_IN_API` определяет, выполняет ли процесс API задачи:

| Значение | Поведение |
| --- | --- |
| `true` (по умолчанию) | API сам обрабатывает задачи — удобно в разработке: `npm start` делает всё |
| `false` | **Жёсткое разделение:** API только принимает запросы и отдаёт `202`, ни одна задача в его процессе не выполняется |

```bash
# Продакшен: приём и выполнение — разные процессы
QUEUE_DRIVER=redis WORKER_IN_API=false npm start   # несколько инстансов за LB
QUEUE_DRIVER=redis npm run worker                  # масштабируется отдельно
```

Гарантия не декларативная: воркер запускается только явным `startWorker()`, а
драйвер `memory` без него задачи не разбирает — они остаются `pending`. Поэтому
`WORKER_IN_API=false` не может «случайно» обработать задачу inline.

Отсюда следствие: `WORKER_IN_API=false` осмысленно только с
`QUEUE_DRIVER=redis` — у процессов с `memory` очередь своя, и отдельный воркер
до задач API не доберётся. Такую конфигурацию сервис принимает, но пишет
предупреждение при старте.

### Обращение к ML-сервису — в фоне

`POST /api/v1/personalize` ставит задачу `ml.faceSwap` и сразу отвечает `202`
(заказ обложки ставит `cover.faceSwap` — та же работа, но с записью в БД).
Долгий вызов Python-сервиса (диффузия — минуты) идёт в воркере: его таймаут
(`ML_FACE_SWAP_TIMEOUT_MS`) ждёт воркер, а не HTTP-клиент.

```bash
curl -X POST http://localhost:3000/api/v1/personalize \
  -H 'content-type: application/json' \
  -d '{ "source": "uploads/face.jpg", "target": "uploads/cover.png",
        "options": { "enhance": true, "style_strength": 1.0 } }'
# → 202 { "taskId": "task_…", "status": "pending" }
```

Результат сохраняется в артефакты задачи.

### Надёжность

- **Повторы.** Упавшая задача повторяется до `QUEUE_MAX_ATTEMPTS` раз с
  экспоненциальным backoff (`QUEUE_BACKOFF_MS · 2^попытка`). У `redis` то же
  делает BullMQ.
- **Трассировка.** `request_id` едет в задаче (`meta`) и восстанавливается в
  воркере — фоновые логи остаются в трассе исходного запроса даже в другом
  процессе через Redis.
- **Graceful shutdown.** По SIGTERM/SIGINT воркер и соединения закрываются.

## Хранилище заказов

Состояние заказов скрыто за контрактом драйвера (`src/storage/coverRepository.js`),
драйвер выбирается `DB_DRIVER`:

| Драйвер | Когда | Свойства |
| --- | --- | --- |
| `memory` (по умолчанию) | один узел, разработка | in-process (Map); теряется при перезапуске, не разделяется между процессами |
| `postgres` | полная автономность | общее состояние между процессами API и воркеров, переживает перезапуск |

Заказ — вложенный документ (спецификация, результат, артефакты), поэтому
целиком лежит в колонке `data` (`jsonb`), а `id`/`status`/`created_at`
продублированы отдельными колонками под выборки и сортировку — без ORM и
миграций. Схема создаётся на старте (`CREATE TABLE IF NOT EXISTS`).

```bash
DB_DRIVER=postgres DATABASE_URL=postgres://user:pass@host:5432/projectx npm start
```

Полная автономность = `DB_DRIVER=postgres` (заказы) + `QUEUE_DRIVER=redis`
(задачи): тогда несколько инстансов API и воркеров делят одно состояние, а
перезапуск любого процесса ничего не теряет. Оба — переключение конфигом, код
не меняется. `pg` подгружается лениво: узлы на memory-драйвере драйвер БД не
тянут.

## Логирование и трассировка

Структурированные JSON-логи (pino) пишутся в `logs/` с ротацией через `pino-roll`:

| Файл | Содержимое |
| --- | --- |
| `logs/app.2026-07-23.1.log` | все записи уровня `LOG_LEVEL` и выше, JSON построчно |
| `logs/error.2026-07-23.1.log` | только `error` и `fatal` |

Имя формируется как `<базовое>.<дата>.<номер>.log`, поэтому фиксированного
`app.log` не существует — активный файл отдаёт хелпер `latestLogFile()` из
`src/utils/logger.js`.

### Ротация

| Переменная | По умолчанию | Значение |
| --- | --- | --- |
| `LOG_ROTATE_FREQUENCY` | `daily` | `daily`, `hourly` или число мс |
| `LOG_ROTATE_SIZE` | `20m` | досрочная ротация по объёму (`k`/`m`/`g`) |
| `LOG_ROTATE_LIMIT` | `14` | сколько ротированных файлов хранить сверх активного |
| `LOG_ROTATE_DATE_FORMAT` | `yyyy-MM-dd` | формат даты в имени файла |

Частота и размер работают вместе: файл роняется по наступлению суток либо
раньше, если превысил `LOG_ROTATE_SIZE`. Старые файлы удаляются автоматически,
внешний `logrotate` не нужен. Перезапуск сервиса в тот же день дописывает
существующий файл, а не плодит новый.

`pino-roll` подключён как поток напрямую, а не через `pino.transport`: transport
поднимает worker-поток, который удерживает event loop и не даёт завершиться
`node --test`.

В консоли в dev — человекочитаемый вывод (pino-pretty), в production — тот же
JSON, что и в файле. Формат записи общий с ml-service:

```json
{"level":"info","time":"2026-07-23T03:49:04.385Z","service":"cover-service",
 "request_id":"trace-e2e-1784778544","res":{"status_code":200},
 "message":"GET /health/ready → 200"}
```

### X-Request-ID

`src/api/middleware/requestId.js` принимает входящий `X-Request-ID` (или
генерирует UUID), кладёт его в `AsyncLocalStorage` и возвращает тем же
заголовком. Логгер подмешивает `request_id` автоматически — сервисам и очереди
не нужно его прокидывать параметрами.

- Некорректный входящий id (не `[A-Za-z0-9_.:-]{8,128}`) заменяется на свой —
  иначе переносы строк в заголовке ломали бы построчный разбор логов.
- `bindContext()` переносит трассу в фоновую задачу: она переживает HTTP-ответ,
  но её логи остаются в той же трассе.
- `src/services/mlClient.js` пробрасывает id в ML-сервис, поэтому одна цепочка
  видна в логах обоих сервисов:

```bash
grep -h "$TRACE" logs/app.*.log ml-service/logs/app.log
```

## Где живут модели

В Node-части моделей нет вообще: она не грузит веса и ничего не генерирует.
Единственная точка выхода наружу — `src/services/mlClient.js`, который ходит в
Python ML-сервис по HTTP и пробрасывает `X-Request-ID`. Всё, что касается
детекции лиц, переноса головы и сведения стыка, настраивается на стороне
`ml-service/` (см. его README и `ml-service/.env.example`).

Поля из `options` уходят в форму `/face-swap` как есть — в том числе `emotion`,
параметр будущей трансформации мимики. Клиент менять не потребуется, когда
трансформеры появятся.

Практическое следствие: заменить модель свапа или добавить стиль — правка в
Python-сервисе, Node пересобирать не нужно. Обратное тоже верно — очередь,
хранилище и трассировка ничего не знают про ML, кроме таймаута.

## Что осознанно оставлено заглушками

- Приём файлов — по ссылкам под `STORAGE_ROOT`; загрузка multipart (upload) не
  реализована, файлы кладутся в хранилище вне сервиса.
- Файловые артефакты (`storage/output/`) — на локальном диске; при нескольких
  узлах их выносят в общий том или объектное хранилище (S3).
- Аутентификации и rate limiting нет.
