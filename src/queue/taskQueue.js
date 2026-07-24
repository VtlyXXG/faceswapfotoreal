import { config } from '../config/index.js';
import { getRequestId } from '../utils/requestContext.js';
import { createLogger } from '../utils/logger.js';
import { createTask } from './task.js';
import { MemoryDriver } from './drivers/memoryDriver.js';
import { BullMqDriver } from './drivers/bullmqDriver.js';

const log = createLogger('queue');

// Реестр обработчиков общий для процесса. Регистрация — побочный эффект
// импорта сервисов, поэтому хендлеры, добавленные после создания драйвера,
// всё равно видны воркеру (ссылка на одну и ту же Map).
const handlers = new Map();

let driver;

const buildDriver = () => {
  if (config.queue.driver === 'redis') {
    log.info({ url: config.queue.redis.url }, 'очередь: драйвер redis (BullMQ)');
    return new BullMqDriver({
      ...config.queue.redis,
      concurrency: config.queue.concurrency,
      maxAttempts: config.queue.maxAttempts,
      backoffMs: config.queue.backoffMs,
    });
  }
  log.info('очередь: драйвер memory (in-process)');
  return new MemoryDriver({
    concurrency: config.queue.concurrency,
    backoffMs: config.queue.backoffMs,
  });
};

const getDriver = () => {
  driver ??= buildDriver();
  driver.setHandlers(handlers);
  return driver;
};

/** Регистрирует обработчик типа задачи. handler(payload, ctx) → result. */
export const registerHandler = (type, handler) => {
  handlers.set(type, handler);
};

/**
 * Создаёт задачу, но НЕ ставит в очередь. Нужно, когда доменную сущность
 * надо связать с task.id до старта выполнения (иначе воркер может начать
 * работу раньше, чем связь сохранится).
 */
export const buildTask = (type, payload, options = {}) =>
  createTask(type, payload, {
    maxAttempts: options.maxAttempts ?? config.queue.maxAttempts,
    requestId: getRequestId() ?? null,
  });

/** Ставит уже созданную задачу в очередь. */
export const submit = async (task) => {
  await getDriver().add(task);
  log.debug({ taskId: task.id, type: task.type }, 'задача поставлена в очередь');
  return task;
};

/**
 * Создаёт и сразу ставит задачу в очередь, возвращает её (статус pending).
 * Выполнение идёт в фоне — HTTP-ответ не ждёт результата.
 */
export const enqueue = (type, payload, options = {}) =>
  submit(buildTask(type, payload, options));

export const getTask = (id) => getDriver().get(id);

export const startWorker = () => getDriver().startWorker();

export const queueStats = () => getDriver().stats();

export const closeQueue = async () => {
  if (driver) await driver.close();
  driver = undefined;
};

/** Сброс синглтона — для тестов. */
export const resetQueue = async () => {
  await closeQueue();
  handlers.clear();
};
