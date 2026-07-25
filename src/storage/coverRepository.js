import { config } from '../config/index.js';
import { createLogger } from '../utils/logger.js';
import { MemoryCoverStore } from './drivers/memoryCoverStore.js';
import { PostgresCoverStore } from './drivers/postgresCoverStore.js';

const log = createLogger('db');

let store;

const buildStore = () => {
  if (config.db.driver === 'postgres') {
    log.info({ table: config.db.postgres.table }, 'хранилище заказов: postgres');
    return new PostgresCoverStore(config.db.postgres);
  }
  log.info('хранилище заказов: memory (in-process)');
  return new MemoryCoverStore();
};

const getStore = () => (store ??= buildStore());

/**
 * Репозиторий заказов за контрактом драйвера. Вызывающий код (coverService,
 * контроллеры) не знает, память под ним или БД — переход на postgres это
 * DB_DRIVER=postgres без правок кода.
 */
export const coverRepository = {
  save: (job) => getStore().save(job),
  findById: (id) => getStore().findById(id),
  getById: (id) => getStore().getById(id),
  list: (options) => getStore().list(options),
  update: (id, patch) => getStore().update(id, patch),
  clear: () => getStore().clear(),
};

/** Готовит хранилище (для postgres — создаёт схему). Вызывается на старте. */
export const initCoverStore = () => getStore().init();

export const closeCoverStore = async () => {
  if (store) await store.close();
  store = undefined;
};

/** Сброс синглтона — для тестов. */
export const resetCoverStore = async () => {
  await closeCoverStore();
};

export default coverRepository;
