import { TaskStatus } from '../task.js';
import { runWithContext } from '../../utils/requestContext.js';
import { createLogger } from '../../utils/logger.js';

const log = createLogger('queue:memory');
const now = () => new Date().toISOString();

/**
 * In-process драйвер: ограничение параллелизма, повторы с экспоненциальным
 * backoff, задачи в памяти. Без внешней инфраструктуры — уместен для одного
 * узла и разработки. При падении процесса очередь теряется: для durability
 * и горизонтального масштабирования используется bullmqDriver.
 */
export class MemoryDriver {
  #tasks = new Map();
  #pending = [];
  #active = 0;
  #timers = new Set();
  #handlers = new Map();
  #workerStarted = false;

  constructor({ concurrency = 2, backoffMs = 2000 } = {}) {
    this.concurrency = Math.max(1, concurrency);
    this.backoffMs = backoffMs;
  }

  setHandlers(handlers) {
    this.#handlers = handlers;
  }

  async add(task) {
    this.#tasks.set(task.id, task);
    this.#pending.push(task.id);
    // Задачи выполняются только если в этом процессе поднят воркер. Иначе они
    // копятся в pending — так WORKER_IN_API=false гарантирует, что процесс API
    // ничего не выполняет, а не «случайно» обрабатывает задачи inline
    this.#drain();
    return task;
  }

  async get(id) {
    return this.#tasks.get(id) ?? null;
  }

  #drain() {
    if (!this.#workerStarted) return;
    while (this.#active < this.concurrency && this.#pending.length > 0) {
      const id = this.#pending.shift();
      this.#active += 1;
      this.#run(id).finally(() => {
        this.#active -= 1;
        this.#drain();
      });
    }
  }

  async #run(id) {
    const task = this.#tasks.get(id);
    if (!task) return;

    const handler = this.#handlers.get(task.type);
    if (!handler) {
      this.#fail(task, new Error(`нет обработчика для задачи ${task.type}`));
      return;
    }

    task.status = TaskStatus.ACTIVE;
    task.attempts += 1;
    task.updatedAt = now();
    log.debug({ taskId: id, type: task.type, attempt: task.attempts }, 'задача взята в работу');

    try {
      const result = await runWithContext({ requestId: task.meta?.requestId }, () =>
        handler(task.payload, this.#context(task)),
      );
      task.status = TaskStatus.COMPLETED;
      task.result = result ?? null;
      task.progress = 100;
      task.updatedAt = now();
      log.info({ taskId: id, type: task.type }, 'задача выполнена');
    } catch (err) {
      this.#onError(task, err);
    }
  }

  #onError(task, err) {
    const failure = { message: err.message, code: err.code ?? 'TASK_ERROR' };

    if (task.attempts < task.maxAttempts) {
      task.status = TaskStatus.RETRYING;
      task.error = failure;
      task.updatedAt = now();
      const delay = this.backoffMs * 2 ** (task.attempts - 1);
      log.warn({ taskId: task.id, attempt: task.attempts, delay, err }, 'повтор задачи');

      const timer = setTimeout(() => {
        this.#timers.delete(timer);
        this.#pending.push(task.id);
        this.#drain();
      }, delay);
      timer.unref?.();
      this.#timers.add(timer);
      return;
    }

    this.#fail(task, err);
  }

  #fail(task, err) {
    task.status = TaskStatus.FAILED;
    task.error = { message: err.message, code: err.code ?? 'TASK_ERROR' };
    task.updatedAt = now();
    log.error({ taskId: task.id, type: task.type, err }, 'задача провалена');
  }

  #context(task) {
    return {
      taskId: task.id,
      attempt: task.attempts,
      updateProgress: (value) => {
        task.progress = Math.max(0, Math.min(100, Math.round(value)));
        task.updatedAt = now();
      },
    };
  }

  async stats() {
    const byStatus = {};
    for (const task of this.#tasks.values()) {
      byStatus[task.status] = (byStatus[task.status] ?? 0) + 1;
    }
    return {
      driver: 'memory',
      worker: this.#workerStarted,
      active: this.#active,
      pending: this.#pending.length,
      concurrency: this.concurrency,
      byStatus,
    };
  }

  /** Включает обработку задач в этом процессе и разбирает накопленную очередь. */
  async startWorker() {
    if (this.#workerStarted) return;
    this.#workerStarted = true;
    log.info({ concurrency: this.concurrency }, 'memory-воркер запущен');
    this.#drain();
  }

  async close() {
    for (const timer of this.#timers) clearTimeout(timer);
    this.#timers.clear();
  }
}

export default MemoryDriver;
