import { NotFoundError } from '../../utils/errors.js';

/**
 * In-process хранилище заказов (Map). По умолчанию — для одного узла и
 * разработки. Состояние теряется при перезапуске и не разделяется между
 * процессами: для полной автономности используется postgresCoverStore.
 */
export class MemoryCoverStore {
  #jobs = new Map();

  async save(job) {
    const stored = { ...job, updatedAt: new Date().toISOString() };
    this.#jobs.set(stored.id, stored);
    return stored;
  }

  async findById(id) {
    return this.#jobs.get(id) ?? null;
  }

  async getById(id) {
    const job = this.#jobs.get(id);
    if (!job) throw new NotFoundError('Заказ');
    return job;
  }

  async list({ limit = 50, offset = 0 } = {}) {
    return [...this.#jobs.values()]
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(offset, offset + limit);
  }

  async update(id, patch) {
    const job = await this.getById(id);
    return this.save({ ...job, ...patch });
  }

  async clear() {
    this.#jobs.clear();
  }

  async init() {
    // памяти нечего инициализировать
  }

  async close() {}
}

export default MemoryCoverStore;
