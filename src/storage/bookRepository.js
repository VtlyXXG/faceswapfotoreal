import { NotFoundError } from '../utils/errors.js';

/**
 * In-memory репозиторий заказов.
 * Интерфейс намеренно async — замена на Postgres/Mongo не затронет вызывающий код.
 */
const jobs = new Map();

export const bookRepository = {
  async save(job) {
    jobs.set(job.id, { ...job, updatedAt: new Date().toISOString() });
    return jobs.get(job.id);
  },

  async findById(id) {
    return jobs.get(id) ?? null;
  },

  async getById(id) {
    const job = jobs.get(id);
    if (!job) throw new NotFoundError('Заказ');
    return job;
  },

  async list({ limit = 50, offset = 0 } = {}) {
    return [...jobs.values()]
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(offset, offset + limit);
  },

  async update(id, patch) {
    const job = await this.getById(id);
    return this.save({ ...job, ...patch });
  },

  async clear() {
    jobs.clear();
  },
};

export default bookRepository;
