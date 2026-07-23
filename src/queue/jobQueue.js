import { config } from '../config/index.js';
import { createLogger } from '../utils/logger.js';

const log = createLogger('queue');

/**
 * Минимальная in-process очередь с ограничением параллелизма.
 * Драйвер вынесен за интерфейс: замена на BullMQ/Redis — это новая реализация
 * с теми же методами enqueue()/stats().
 */
class MemoryQueue {
  #pending = [];
  #active = 0;

  constructor(concurrency) {
    this.concurrency = Math.max(1, concurrency);
  }

  enqueue(name, task) {
    return new Promise((resolve, reject) => {
      this.#pending.push({ name, task, resolve, reject });
      this.#drain();
    });
  }

  #drain() {
    while (this.#active < this.concurrency && this.#pending.length > 0) {
      const item = this.#pending.shift();
      this.#active += 1;
      log.debug({ job: item.name, active: this.#active }, 'задача взята в работу');

      Promise.resolve()
        .then(item.task)
        .then(item.resolve, (err) => {
          log.error({ job: item.name, err }, 'задача завершилась ошибкой');
          item.reject(err);
        })
        .finally(() => {
          this.#active -= 1;
          this.#drain();
        });
    }
  }

  stats() {
    return { active: this.#active, pending: this.#pending.length, concurrency: this.concurrency };
  }
}

export const jobQueue = new MemoryQueue(config.queue.concurrency);

export default jobQueue;
