import { test, describe, before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from '../src/config/index.js';
import {
  coverRepository,
  initCoverStore,
  closeCoverStore,
} from '../src/storage/coverRepository.js';
import { createCoverJob, JobStatus } from '../src/domain/cover.js';
import { createCover, getCover, runPipeline } from '../src/services/coverService.js';
import { startWorker, closeQueue } from '../src/queue/taskQueue.js';

const SWAPPED = Buffer.from('swapped-image-bytes');
// Внутри STORAGE_ROOT, но в игнорируемом tmp/ — фикстуры не попадают в репозиторий
const SPEC = { source: 'tmp/test-face.jpg', target: 'tmp/test-cover.png', options: {} };
const realFetch = globalThis.fetch;

/** Ответ ML-сервиса: настоящая диффузия тесту не нужна, важен контракт. */
const stubMlService = (response) => {
  globalThis.fetch = async () => response();
};

const waitFor = async (predicate, timeoutMs = 5000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error('условие не выполнилось за отведённое время');
};

describe('пайплайн обложки', () => {
  // С DB_DRIVER=postgres таблицы может ещё не быть: тест готовит схему сам,
  // для memory-драйвера init — пустышка
  before(() => initCoverStore());

  // Тест поднимает те же синглтоны, что и боевой процесс, поэтому обязан их и
  // закрыть. С QUEUE_DRIVER=redis сокеты ioredis и пул pg — открытые хендлы:
  // без этого node --test виснет после последнего теста, а не падает.
  // closeQueue() внутри доводит до worker.close() → queue.close() → quit().
  after(async () => {
    await closeQueue(); // сначала воркер: его обработчики пишут в хранилище
    await closeCoverStore();
  });

  beforeEach(async () => {
    await coverRepository.clear();

    // Исходники лежат в хранилище: пайплайн читает их по ссылкам из заказа
    await fs.mkdir(path.join(config.storage.root, 'tmp'), { recursive: true });
    await fs.writeFile(path.join(config.storage.root, SPEC.source), 'face');
    await fs.writeFile(path.join(config.storage.root, SPEC.target), 'cover');

    stubMlService(
      () =>
        new Response(SWAPPED, {
          headers: { 'content-type': 'image/png', 'x-swap-meta': JSON.stringify({ faces: 1 }) },
        }),
    );

    // Тест играет роль процесса-воркера: без него задачи только копятся
    await startWorker();
  });

  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  test('заказ доходит до статуса completed', async () => {
    const job = await createCover({ title: 'Путешествие к звёздам', ...SPEC });
    assert.equal(job.status, JobStatus.PENDING);

    await waitFor(async () => (await getCover(job.id)).status === JobStatus.COMPLETED);

    const finished = await getCover(job.id);
    assert.equal(finished.progress, 100);
    assert.equal(finished.result.filename, 'cover.png');
    assert.deepEqual(finished.result.meta, { faces: 1 });
    assert.ok(finished.artifacts.some((a) => a.filename === 'cover.png'));
  });

  test('заказ с путём вне хранилища отклоняется сразу', async () => {
    await assert.rejects(() => createCover({ ...SPEC, source: '../secrets.txt' }), /вне хранилища/);
  });

  // Через runPipeline напрямую, минуя очередь: иначе тест ждал бы все повторы
  test('ошибка ML-сервиса переводит заказ в failed', async () => {
    stubMlService(() => new Response('boom', { status: 500 }));

    const job = await coverRepository.save(createCoverJob(SPEC));
    await assert.rejects(() => runPipeline(job.id));

    const failed = await getCover(job.id);
    assert.equal(failed.status, JobStatus.FAILED);
    assert.equal(failed.error.code, 'ML_FACE_SWAP_FAILED');
  });
});
