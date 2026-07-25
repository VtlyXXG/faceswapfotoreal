import { TaskStatus } from '../task.js';
import { runWithContext } from '../../utils/requestContext.js';
import { createLogger } from '../../utils/logger.js';

const log = createLogger('queue:bullmq');

/**
 * Драйвер BullMQ + Redis: задачи переживают перезапуск, воркеры масштабируются
 * горизонтально независимо от API. Это ответ на «автономность и большие
 * нагрузки»: приём запросов (API) и выполнение (worker) — разные процессы.
 *
 * bullmq/ioredis подгружаются лениво: узлы на memory-драйвере не тянут Redis.
 */
export class BullMqDriver {
  #queue = null;
  #worker = null;
  #connection = null;
  #handlers = new Map();

  constructor({ url, prefix, queue, concurrency = 2, maxAttempts = 3, backoffMs = 2000 } = {}) {
    this.url = url;
    this.prefix = prefix;
    this.queueName = queue;
    this.concurrency = Math.max(1, concurrency);
    this.maxAttempts = maxAttempts;
    this.backoffMs = backoffMs;
  }

  setHandlers(handlers) {
    this.#handlers = handlers;
  }

  async #lib() {
    try {
      const [{ Queue, Worker }, IORedis] = await Promise.all([
        import('bullmq'),
        import('ioredis'),
      ]);
      return { Queue, Worker, IORedis: IORedis.default };
    } catch (err) {
      throw new Error(
        `QUEUE_DRIVER=redis, но bullmq/ioredis не установлены: ${err.message}. ` +
          'Выполните npm install или переключитесь на QUEUE_DRIVER=memory.',
      );
    }
  }

  async #getConnection() {
    if (this.#connection) return this.#connection;
    const { IORedis } = await this.#lib();
    // BullMQ требует maxRetriesPerRequest: null на соединении
    this.#connection = new IORedis(this.url, { maxRetriesPerRequest: null });
    return this.#connection;
  }

  async #getQueue() {
    if (this.#queue) return this.#queue;
    const { Queue } = await this.#lib();
    this.#queue = new Queue(this.queueName, {
      connection: await this.#getConnection(),
      prefix: this.prefix,
    });
    return this.#queue;
  }

  async add(task) {
    const queue = await this.#getQueue();
    await queue.add(task.type, task, {
      jobId: task.id,
      attempts: task.maxAttempts,
      backoff: { type: 'exponential', delay: this.backoffMs },
      removeOnComplete: false,
      removeOnFail: false,
    });
    return task;
  }

  async get(id) {
    const queue = await this.#getQueue();
    const job = await queue.getJob(id);
    if (!job) return null;

    const state = await job.getState();
    const status = this.#mapState(state);
    const envelope = job.data;
    // Завершённая задача — всегда 100%, как у memory-драйвера (BullMQ хранит
    // последнее переданное значение прогресса, которое может быть меньше)
    const reported = typeof job.progress === 'number' ? job.progress : envelope.progress;
    return {
      ...envelope,
      status,
      attempts: job.attemptsMade,
      progress: status === TaskStatus.COMPLETED ? 100 : reported,
      result: job.returnvalue ?? null,
      error: job.failedReason ? { message: job.failedReason, code: 'TASK_ERROR' } : null,
      updatedAt: new Date(job.finishedOn ?? job.processedOn ?? job.timestamp).toISOString(),
    };
  }

  #mapState(state) {
    switch (state) {
      case 'active':
        return TaskStatus.ACTIVE;
      case 'completed':
        return TaskStatus.COMPLETED;
      case 'failed':
        return TaskStatus.FAILED;
      default: // waiting, delayed, paused, waiting-children
        return TaskStatus.PENDING;
    }
  }

  async startWorker() {
    if (this.#worker) return;
    const { Worker } = await this.#lib();

    this.#worker = new Worker(
      this.queueName,
      async (job) => {
        const handler = this.#handlers.get(job.name);
        if (!handler) throw new Error(`нет обработчика для задачи ${job.name}`);

        const ctx = {
          taskId: job.id,
          attempt: job.attemptsMade + 1,
          updateProgress: (value) => job.updateProgress(Math.max(0, Math.min(100, value))),
        };
        return runWithContext({ requestId: job.data.meta?.requestId }, () =>
          handler(job.data.payload, ctx),
        );
      },
      { connection: await this.#getConnection(), prefix: this.prefix, concurrency: this.concurrency },
    );

    this.#worker.on('failed', (job, err) =>
      log.error({ taskId: job?.id, type: job?.name, err }, 'задача провалена'),
    );
    this.#worker.on('completed', (job) =>
      log.info({ taskId: job.id, type: job.name }, 'задача выполнена'),
    );
    log.info({ queue: this.queueName, concurrency: this.concurrency }, 'BullMQ-воркер запущен');
  }

  async stats() {
    const queue = await this.#getQueue();
    const counts = await queue.getJobCounts(
      'waiting',
      'active',
      'delayed',
      'completed',
      'failed',
    );
    return {
      driver: 'redis',
      worker: this.#worker !== null,
      active: counts.active ?? 0,
      pending: (counts.waiting ?? 0) + (counts.delayed ?? 0),
      concurrency: this.concurrency,
      byStatus: counts,
    };
  }

  async close() {
    await this.#worker?.close();
    await this.#queue?.close();
    await this.#connection?.quit();
    this.#worker = this.#queue = this.#connection = null;
  }
}

export default BullMqDriver;
