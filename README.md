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
  storage/              репозиторий заказов + файловые артефакты
  queue/                очередь задач с ограничением параллелизма
  utils/
    logger.js           pino: pretty в консоль, JSON в logs/ с ротацией
    requestContext.js   AsyncLocalStorage с request_id
    errors.js, id.js
  config/index.js       вся конфигурация из ENV в одном месте
  app.js                сборка Express-приложения
  server.js             точка входа, graceful shutdown
tests/                  node:test
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `POST` | `/api/v1/books` | создать заказ, возвращает `202` и `id` |
| `GET` | `/api/v1/books` | список заказов |
| `GET` | `/api/v1/books/:id` | полное состояние заказа |
| `GET` | `/api/v1/books/:id/status` | краткий статус и прогресс |
| `GET` | `/api/v1/books/:id/files` | список готовых файлов |
| `GET` | `/api/v1/books/:id/files/:filename` | скачать файл |
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
    "withIllustrations": false,
    "recipient": { "name": "Аня", "age": 8, "interests": ["космос", "динозавры"] },
    "style": { "genre": "приключения", "tone": "тёплый и добрый", "language": "ru" },
    "output": { "formats": ["md", "html"] }
  }'
```

Генерация асинхронная: заказ ставится в очередь, прогресс опрашивается через `/status`.

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

- Хранилище заказов — in-memory (`src/storage/bookRepository.js`); интерфейс async,
  замена на БД не затрагивает вызывающий код.
- Очередь — in-process (`src/queue/jobQueue.js`); при горизонтальном масштабировании
  заменяется на BullMQ/Redis с теми же методами.
- Рендеры `pdf` и `epub` пока не реализованы — формат тихо пропускается с записью в лог.
- Аутентификации и rate limiting нет.
