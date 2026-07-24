import { config } from '../config/index.js';
import { createLogger } from '../utils/logger.js';
import { MemoryBookStore } from './drivers/memoryBookStore.js';
import { PostgresBookStore } from './drivers/postgresBookStore.js';

const log = createLogger('db');

let store;

const buildStore = () => {
  if (config.db.driver === 'postgres') {
    log.info({ table: config.db.postgres.table }, 'хранилище заказов: postgres');
    return new PostgresBookStore(config.db.postgres);
  }
  log.info('хранилище заказов: memory (in-process)');
  return new MemoryBookStore();
};

const getStore = () => (store ??= buildStore());

/**
 * Репозиторий заказов за контрактом драйвера. Вызывающий код (bookService,
 * контроллеры) не знает, память под ним или БД — переход на postgres это
 * DB_DRIVER=postgres без правок кода.
 */
export const bookRepository = {
  save: (job) => getStore().save(job),
  findById: (id) => getStore().findById(id),
  getById: (id) => getStore().getById(id),
  list: (options) => getStore().list(options),
  update: (id, patch) => getStore().update(id, patch),
  clear: () => getStore().clear(),
};

/** Готовит хранилище (для postgres — создаёт схему). Вызывается на старте. */
export const initBookStore = () => getStore().init();

export const closeBookStore = async () => {
  if (store) await store.close();
  store = undefined;
};

/** Сброс синглтона — для тестов. */
export const resetBookStore = async () => {
  await closeBookStore();
};

export default bookRepository;
