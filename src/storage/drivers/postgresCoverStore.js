import { NotFoundError } from '../../utils/errors.js';
import { createLogger } from '../../utils/logger.js';

const log = createLogger('db:postgres');

// Имя таблицы приходит из конфигурации и подставляется в SQL как идентификатор,
// поэтому допускаем только безопасные символы — защита от инъекции.
const SAFE_IDENT = /^[a-z_][a-z0-9_]*$/;

/**
 * Хранилище заказов обложек в PostgreSQL.
 *
 * Заказ — вложенный документ (спецификация, результат, артефакты), поэтому
 * целиком лежит в колонке data (jsonb), а id/status/created_at продублированы
 * отдельными колонками для выборок и сортировки. Это даёт общее состояние
 * между процессами API и воркеров без ORM и миграций схемы.
 *
 * pg подгружается лениво: узлы на memory-драйвере не тянут драйвер БД.
 */
export class PostgresCoverStore {
  #pool = null;
  #client = null; // инъекция для тестов

  constructor({ url, table = 'covers', ssl = false, connectionTimeoutMs = 10_000, client = null } = {}) {
    if (!SAFE_IDENT.test(table)) {
      throw new Error(`недопустимое имя таблицы: ${table}`);
    }
    this.url = url;
    this.table = table;
    this.ssl = ssl;
    this.connectionTimeoutMs = connectionTimeoutMs;
    this.#client = client;
  }

  async #getPool() {
    if (this.#pool) return this.#pool;
    const { default: pg } = await import('pg');
    this.#pool = new pg.Pool({
      connectionString: this.url,
      ssl: this.ssl ? { rejectUnauthorized: false } : undefined,
      // По умолчанию pg ждёт соединения бесконечно: недоступный Postgres вешает
      // старт процесса и тесты навсегда, вместо внятной ошибки. Ограничиваем.
      connectionTimeoutMillis: this.connectionTimeoutMs,
    });
    return this.#pool;
  }

  async #query(text, params) {
    if (this.#client) return this.#client.query(text, params);
    const pool = await this.#getPool();
    return pool.query(text, params);
  }

  async init() {
    await this.#query(
      `CREATE TABLE IF NOT EXISTS ${this.table} (
         id text PRIMARY KEY,
         task_id text,
         status text NOT NULL,
         created_at timestamptz NOT NULL,
         updated_at timestamptz NOT NULL,
         data jsonb NOT NULL
       )`,
    );
    await this.#query(
      `CREATE INDEX IF NOT EXISTS ${this.table}_created_idx ON ${this.table} (created_at DESC)`,
    );
    log.info({ table: this.table }, 'схема БД готова');
  }

  async save(job) {
    const row = { ...job, updatedAt: new Date().toISOString() };
    await this.#query(
      `INSERT INTO ${this.table} (id, task_id, status, created_at, updated_at, data)
       VALUES ($1, $2, $3, $4, $5, $6::jsonb)
       ON CONFLICT (id) DO UPDATE SET
         task_id = EXCLUDED.task_id,
         status = EXCLUDED.status,
         updated_at = EXCLUDED.updated_at,
         data = EXCLUDED.data`,
      [row.id, row.taskId ?? null, row.status, row.createdAt, row.updatedAt, JSON.stringify(row)],
    );
    return row;
  }

  async findById(id) {
    const { rows } = await this.#query(`SELECT data FROM ${this.table} WHERE id = $1`, [id]);
    return rows[0]?.data ?? null;
  }

  async getById(id) {
    const job = await this.findById(id);
    if (!job) throw new NotFoundError('Заказ');
    return job;
  }

  async list({ limit = 50, offset = 0 } = {}) {
    const { rows } = await this.#query(
      `SELECT data FROM ${this.table} ORDER BY created_at DESC LIMIT $1 OFFSET $2`,
      [limit, offset],
    );
    return rows.map((r) => r.data);
  }

  async update(id, patch) {
    // Каждый заказ обрабатывает одна задача, поэтому конкурентных изменений
    // одной записи нет — read-modify-write безопасен и даёт паритет с memory
    const job = await this.getById(id);
    return this.save({ ...job, ...patch });
  }

  async clear() {
    await this.#query(`DELETE FROM ${this.table}`);
  }

  async close() {
    await this.#pool?.end();
    this.#pool = null;
  }
}

export default PostgresCoverStore;
