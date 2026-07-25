/**
 * Проверка реальных драйверов на живых Postgres и Redis.
 *
 *   DB_DRIVER=postgres QUEUE_DRIVER=redis node scripts/check-infra.mjs
 *
 * Гоняет PostgresCoverStore (полный контракт хранилища заказов) и BullMqDriver
 * (жизненный цикл задачи, повторы, stats) против баз из docker-compose.
 * Печатает PASS/FAIL по шагам и завершается ненулевым кодом при ошибке.
 */
import { config } from '../src/config/index.js';
import { PostgresCoverStore } from '../src/storage/drivers/postgresCoverStore.js';
import { BullMqDriver } from '../src/queue/drivers/bullmqDriver.js';
import { createTask, TaskStatus } from '../src/queue/task.js';

let failures = 0;

const check = (label, ok, detail = '') => {
  const mark = ok ? 'PASS' : 'FAIL';
  if (!ok) failures += 1;
  console.log(`  [${mark}] ${label}${detail ? ` — ${detail}` : ''}`);
};

const waitFor = async (predicate, timeoutMs = 15000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  return false;
};

const sampleJob = (id, createdAt) => ({
  id,
  taskId: `task_${id}`,
  status: 'pending',
  progress: 0,
  spec: { source: 'uploads/face.jpg', target: 'uploads/cover.png', options: {} },
  result: null,
  createdAt,
  updatedAt: createdAt,
});

async function checkPostgres() {
  console.log(`\n== PostgreSQL (${config.db.postgres.url}) ==`);
  const store = new PostgresCoverStore(config.db.postgres);

  await store.init();
  check('init(): CREATE TABLE/INDEX', true);
  await store.clear();

  const saved = await store.save(sampleJob('cover_pg_1', '2026-07-24T10:00:00.000Z'));
  check('save() вернул документ', saved.id === 'cover_pg_1');

  const found = await store.findById('cover_pg_1');
  check('findById() маппит jsonb → объект', found?.spec?.target === 'uploads/cover.png');
  check('findById() отсутствующего → null', (await store.findById('нет')) === null);

  let threw = false;
  try {
    await store.getById('нет');
  } catch {
    threw = true;
  }
  check('getById() отсутствующего → NotFoundError', threw);

  const updated = await store.update('cover_pg_1', { status: 'completed', progress: 100 });
  const reread = await store.findById('cover_pg_1');
  check('update() смержил и сохранил', updated.status === 'completed' && reread.progress === 100);

  await store.save(sampleJob('cover_pg_2', '2026-07-24T12:00:00.000Z'));
  await store.save(sampleJob('cover_pg_3', '2026-07-24T11:00:00.000Z'));
  const page = await store.list({ limit: 2, offset: 0 });
  check(
    'list() сортирует по created_at DESC',
    page.length === 2 && page[0].id === 'cover_pg_2' && page[1].id === 'cover_pg_3',
  );

  await store.clear();
  check('clear() очистил таблицу', (await store.list({})).length === 0);

  await store.close();
}

async function checkRedis() {
  console.log(`\n== Redis / BullMQ (${config.queue.redis.url}) ==`);
  const driver = new BullMqDriver({
    ...config.queue.redis,
    concurrency: 4,
    maxAttempts: 3,
    backoffMs: 300,
  });

  let flakyCalls = 0;
  driver.setHandlers(
    new Map([
      ['echo', async (payload) => ({ echoed: payload.value })],
      [
        'flaky',
        async () => {
          flakyCalls += 1;
          if (flakyCalls < 2) throw new Error('временный сбой');
          return 'ok';
        },
      ],
    ]),
  );

  await driver.startWorker();
  check('startWorker(): подключение к Redis', true);

  const echo = createTask('echo', { value: 42 }, { maxAttempts: 3 });
  await driver.add(echo);
  const done = await waitFor(async () => (await driver.get(echo.id))?.status === TaskStatus.COMPLETED);
  const echoState = await driver.get(echo.id);
  check('задача выполнена и результат сохранён', done && echoState.result?.echoed === 42);

  const flaky = createTask('flaky', {}, { maxAttempts: 3 });
  await driver.add(flaky);
  const recovered = await waitFor(
    async () => (await driver.get(flaky.id))?.status === TaskStatus.COMPLETED,
  );
  const flakyState = await driver.get(flaky.id);
  check('повтор после сбоя → completed', recovered && flakyState.attempts >= 2);

  const stats = await driver.stats();
  check('stats() читает счётчики очереди', stats.driver === 'redis' && stats.byStatus != null,
    JSON.stringify(stats.byStatus));

  await driver.close();
  check('close(): соединения закрыты', true);
}

async function main() {
  console.log('Проверка инфраструктуры: драйверы на реальных базах');
  try {
    await checkPostgres();
  } catch (err) {
    check('PostgreSQL', false, err.message);
  }
  try {
    await checkRedis();
  } catch (err) {
    check('Redis', false, err.message);
  }

  console.log(`\n${failures === 0 ? '✅ всё зелёное' : `❌ провалов: ${failures}`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
