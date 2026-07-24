# projectx — сервис генерации кастомизированных книг

Node.js (ESM) + Express. Текст и иллюстрации генерируются **локальными** моделями
(Ollama / llama.cpp / Stable Diffusion WebUI) через слой абстракции — внешние API не используются.

## Быстрый старт

```bash
cp .env.example .env
npm install
npm run dev
```

Без запущенной модели поставьте `AI_TEXT_PROVIDER=mock` — весь пайплайн отработает на заглушках.

С Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
```

### Локальная инфраструктура (Postgres + Redis)

Для запуска в автономном режиме `docker-compose.yml` поднимает Postgres и Redis:

```bash
docker compose up -d postgres redis   # только базы (без сборки образов)
cp .env.example .env                  # .env уже указывает на localhost:5432 / :6379
npm run check:infra                   # прогнать драйверы БД и очереди на живых базах
npm start                             # API + воркер против Postgres и Redis
```

`npm run check:infra` (`scripts/check-infra.mjs`) гоняет `PostgresBookStore`
(полный контракт хранилища) и `BullMqDriver` (жизненный цикл задачи, повторы,
stats) против контейнеров и печатает PASS/FAIL по шагам.

`docker compose up` целиком поднимает и `api`, и `worker` (масштабируется
`--scale worker=3`), и `postgres`/`redis` — API стартует после их healthcheck.

## Структура

```
src/
  ai/                   слой ИИ — единственное место, знающее про модели
    providers/          реализации: Ollama, llama.cpp, Automatic1111, mock
      BaseTextProvider.js    контракт: generate() / stream() / healthCheck()
      BaseImageProvider.js
    prompts/            шаблоны промптов (версионируемые артефакты)
    registry.js         выбор провайдера по конфигу + регистрация своих
  api/
    routes/             маршруты /api/v1
    controllers/        HTTP-слой, без бизнес-логики
    middleware/         requestId, asyncHandler, errorHandler
  domain/               модели предметной области + zod-схема заказа
  services/
    mlClient.js         клиент ML-сервиса, пробрасывает X-Request-ID
    bookService.js      приём заказа и оркестрация пайплайна
    generationService.js шаги генерации: структура → главы → иллюстрации
    renderService.js    сборка md/html (pdf/epub — точка расширения)
  storage/
    bookRepository.js   репозиторий заказов за контрактом драйвера
    drivers/memoryBookStore.js   in-process (Map)
    drivers/postgresBookStore.js БД: jsonb-документ + индексированные колонки
    fileStorage.js      файловые артефакты
  queue/                асинхронная очередь задач
    task.js             модель задачи + статусы
    taskQueue.js        фасад: выбор драйвера, регистрация обработчиков
    drivers/memoryDriver.js  in-process: параллелизм, повторы, backoff
    drivers/bullmqDriver.js  BullMQ + Redis: durable, распределённо
  services/personalizeService.js  фоновый вызов ML face-swap (задача ml.faceSwap)
  utils/
    logger.js           pino: pretty в консоль, JSON в logs/ с ротацией
    requestContext.js   AsyncLocalStorage с request_id
    errors.js, id.js
  config/index.js       вся конфигурация из ENV в одном месте
  app.js                сборка Express-приложения
  server.js             API + воркер в одном процессе (memory)
  worker.js             автономный воркер (для QUEUE_DRIVER=redis)
tests/                  node:test
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `POST` | `/api/v1/books` | создать заказ, `202` + `taskId` |
| `GET` | `/api/v1/books` | список заказов |
| `GET` | `/api/v1/books/:id` | полное состояние заказа |
| `GET` | `/api/v1/books/:id/status` | краткий статус и прогресс |
| `GET` | `/api/v1/books/:id/files` | список готовых файлов |
| `GET` | `/api/v1/books/:id/files/:filename` | скачать файл |
| `POST` | `/api/v1/personalize` | face-swap на ML-сервисе, `202` + `taskId` |
| `GET` | `/api/v1/tasks/:id` | полное состояние задачи |
| `GET` | `/api/v1/tasks/:id/status` | краткий статус задачи (для опроса) |
| `GET` | `/api/v1/health` | liveness |
| `GET` | `/api/v1/health/ready` | readiness + доступность моделей |
| `GET` | `/api/v1/stats` | состояние очереди |

Пример заказа:

```bash
curl -X POST http://localhost:3000/api/v1/books \
  -H 'content-type: application/json' \
  -d '{
    "title": "Путешествие к звёздам",
    "chapterCount": 5,
    "recipient": { "name": "Аня", "age": 8, "interests": ["космос", "динозавры"] },
    "output": { "formats": ["md", "html"] }
  }'
# → 202 { "id": "book_…", "taskId": "task_…", "status": "pending",
#         "statusUrl": "/api/v1/tasks/task_…/status" }
```

## Асинхронная очередь задач

Сервис **никогда не держит клиента на долгой операции**: тяжёлая работа
(генерация книги, face-swap на Python ML-сервисе) уходит в фоновую задачу, а
API сразу отвечает `202 Accepted` + `taskId`. Клиент опрашивает
`GET /tasks/:id/status`.

### Драйверы

Очередь скрыта за фасадом `src/queue/taskQueue.js`; драйвер выбирается
`QUEUE_DRIVER`:

| Драйвер | Когда | Свойства |
| --- | --- | --- |
| `memory` (по умолчанию) | один узел, разработка | in-process, без инфраструктуры; параллелизм, повторы с экспоненциальным backoff. Очередь теряется при перезапуске |
| `redis` (BullMQ) | автономность, большие нагрузки | задачи переживают перезапуск; воркеры масштабируются **отдельно** от API. `bullmq`/`ioredis` подгружаются лениво |

Тот же приём, что у ИИ-провайдеров и хранилища: контракт один, реализации
подключаются конфигом. Перейти на Redis — это `QUEUE_DRIVER=redis` без правок
кода.

### Масштабирование

```bash
QUEUE_DRIVER=redis npm start      # приём запросов (можно несколько инстансов за LB)
QUEUE_DRIVER=redis npm run worker # выполнение задач (масштабируется отдельно)
```

С `memory` API и воркер живут в одном процессе (`npm start`), отдельный воркер
не нужен. С `redis` API только ставит задачи, а `npm run worker` их выполняет —
столько воркеров, сколько нужно под нагрузку.

### Обращение к ML-сервису — в фоне

`POST /api/v1/personalize` ставит задачу `ml.faceSwap` и сразу отвечает `202`.
Долгий вызов Python-сервиса (диффузия — минуты) идёт в воркере: его таймаут
(`ML_FACE_SWAP_TIMEOUT_MS`) ждёт воркер, а не HTTP-клиент.

```bash
curl -X POST http://localhost:3000/api/v1/personalize \
  -H 'content-type: application/json' \
  -d '{ "source": "uploads/face.jpg", "target": "uploads/cover.png",
        "options": { "enhance": true, "style_strength": 1.0 } }'
# → 202 { "taskId": "task_…", "status": "pending" }
```

Пути `source`/`target` — ссылки на файлы под `STORAGE_ROOT` (выход за его
пределы отклоняется). Результат сохраняется в артефакты задачи.

### Надёжность

- **Повторы.** Упавшая задача повторяется до `QUEUE_MAX_ATTEMPTS` раз с
  экспоненциальным backoff (`QUEUE_BACKOFF_MS · 2^попытка`). У `redis` то же
  делает BullMQ.
- **Трассировка.** `request_id` едет в задаче (`meta`) и восстанавливается в
  воркере — фоновые логи остаются в трассе исходного запроса даже в другом
  процессе через Redis.
- **Graceful shutdown.** По SIGTERM/SIGINT воркер и соединения закрываются.

## Хранилище заказов

Состояние заказов скрыто за контрактом драйвера (`src/storage/bookRepository.js`),
драйвер выбирается `DB_DRIVER`:

| Драйвер | Когда | Свойства |
| --- | --- | --- |
| `memory` (по умолчанию) | один узел, разработка | in-process (Map); теряется при перезапуске, не разделяется между процессами |
| `postgres` | полная автономность | общее состояние между процессами API и воркеров, переживает перезапуск |

Заказ — вложенный документ (спецификация, оглавление, главы, артефакты),
поэтому целиком лежит в колонке `data` (`jsonb`), а `id`/`status`/`created_at`
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
{"level":"info","time":"2026-07-23T03:49:04.385Z","service":"book-service",
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
- `bindContext()` переносит трассу в фоновую задачу генерации: она переживает
  HTTP-ответ, но её логи остаются в той же трассе.
- `src/services/mlClient.js` пробрасывает id в ML-сервис, поэтому одна цепочка
  видна в логах обоих сервисов:

```bash
grep -h "$TRACE" logs/app.*.log ml-service/logs/app.log
```

## Архитектура ИИ-слоя

Доменный код никогда не обращается к модели напрямую — только к контракту
`BaseTextProvider` / `BaseImageProvider`. Провайдер выбирается в `registry.js` по
переменным `AI_TEXT_PROVIDER` / `AI_IMAGE_PROVIDER`.

Добавление своей локальной модели:

```js
import { BaseTextProvider } from './src/ai/providers/BaseTextProvider.js';
import { registerTextProvider } from './src/ai/registry.js';

class VllmProvider extends BaseTextProvider {
  async generate({ prompt }) { /* ... */ }
  async healthCheck() { return true; }
}

registerTextProvider('vllm', VllmProvider);
```

## Что осознанно оставлено заглушками

- Рендеры `pdf` и `epub` пока не реализованы — формат тихо пропускается с записью в лог.
- Приём файлов для `/personalize` — по ссылкам под `STORAGE_ROOT`; загрузка
  multipart (upload) не реализована.
- Файловые артефакты (`storage/output/`) — на локальном диске; при нескольких
  узлах их выносят в общий том или объектное хранилище (S3).
- Аутентификации и rate limiting нет.
