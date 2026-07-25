import 'dotenv/config';
import path from 'node:path';

const int = (value, fallback) => {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const bool = (value, fallback) => {
  if (value === undefined || value === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase());
};

export const config = {
  env: process.env.NODE_ENV ?? 'development',
  isProduction: process.env.NODE_ENV === 'production',

  server: {
    host: process.env.HOST ?? '0.0.0.0',
    port: int(process.env.PORT, 3000),
  },

  logger: {
    level: process.env.LOG_LEVEL ?? 'info',
    dir: path.resolve(process.cwd(), process.env.LOG_DIR ?? './logs'),
    // Имена без расширения: pino-roll добавит дату, номер и .log
    file: process.env.LOG_FILE ?? 'app',
    errorFile: process.env.LOG_ERROR_FILE ?? 'error',
    rotation: {
      // daily | hourly | число миллисекунд
      frequency: process.env.LOG_ROTATE_FREQUENCY ?? 'daily',
      // размер, при котором файл роняется досрочно: '20m', '500k', '1g'
      size: process.env.LOG_ROTATE_SIZE ?? '20m',
      // сколько ротированных файлов хранить сверх активного
      limitCount: int(process.env.LOG_ROTATE_LIMIT, 14),
      dateFormat: process.env.LOG_ROTATE_DATE_FORMAT ?? 'yyyy-MM-dd',
    },
  },

  mlService: {
    baseUrl: process.env.ML_SERVICE_URL ?? 'http://127.0.0.1:8000',
    timeoutMs: int(process.env.ML_SERVICE_TIMEOUT_MS, 180_000),
    // face-swap с диффузией может идти минутами; ждёт воркер, не клиент
    faceSwapTimeoutMs: int(process.env.ML_FACE_SWAP_TIMEOUT_MS, 600_000),
  },

  storage: {
    driver: process.env.STORAGE_DRIVER ?? 'fs',
    root: path.resolve(process.cwd(), process.env.STORAGE_ROOT ?? './storage'),
  },

  db: {
    // memory — in-process (по умолчанию); postgres — общее состояние заказов
    // между процессами API и воркеров, переживает перезапуск
    driver: process.env.DB_DRIVER ?? 'memory',
    postgres: {
      url: process.env.DATABASE_URL ?? 'postgres://127.0.0.1:5432/projectx',
      table: process.env.DB_TABLE ?? 'covers',
      ssl: (process.env.DB_SSL ?? 'false') === 'true',
    },
  },

  queue: {
    // memory — in-process, без инфраструктуры; redis — BullMQ, durable/распределённо
    driver: process.env.QUEUE_DRIVER ?? 'memory',
    // Выполняет ли процесс API задачи. false — жёсткое разделение ролей:
    // API только принимает запросы и отдаёт 202, работу делают воркеры
    workerInApi: bool(process.env.WORKER_IN_API, true),
    concurrency: int(process.env.QUEUE_CONCURRENCY, 2),
    maxAttempts: int(process.env.QUEUE_MAX_ATTEMPTS, 3),
    backoffMs: int(process.env.QUEUE_BACKOFF_MS, 2000),
    redis: {
      url: process.env.REDIS_URL ?? 'redis://127.0.0.1:6379',
      prefix: process.env.QUEUE_PREFIX ?? 'projectx',
      queue: process.env.QUEUE_NAME ?? 'tasks',
    },
  },
};

export default config;
