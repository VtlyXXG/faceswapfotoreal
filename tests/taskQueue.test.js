import { test, describe, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { TaskStatus, createTask, toPublicTask } from '../src/queue/task.js';
import { MemoryDriver } from '../src/queue/drivers/memoryDriver.js';

const settle = (ms = 30) => new Promise((resolve) => setTimeout(resolve, ms));

const waitFor = async (predicate, timeoutMs = 2000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await settle(10);
  }
  throw new Error('условие не выполнилось за отведённое время');
};

describe('модель задачи', () => {
  test('createTask задаёт pending и переносит requestId', () => {
    const task = createTask('demo', { a: 1 }, { maxAttempts: 5, requestId: 'req-1' });
    assert.equal(task.status, TaskStatus.PENDING);
    assert.equal(task.maxAttempts, 5);
    assert.equal(task.meta.requestId, 'req-1');
    assert.match(task.id, /^task_/);
  });

  test('toPublicTask скрывает payload', () => {
    const task = createTask('demo', { secret: 'x' });
    const pub = toPublicTask(task);
    assert.equal(pub.payload, undefined);
    assert.equal(pub.id, task.id);
  });
});

describe('memory-драйвер', () => {
  let driver;

  afterEach(() => driver?.close());

  /** Драйвер с поднятым воркером — процесс, который выполняет задачи. */
  const run = (handlers) => {
    driver = new MemoryDriver({ concurrency: 2, backoffMs: 10 });
    driver.setHandlers(new Map(Object.entries(handlers)));
    driver.startWorker();
    return driver;
  };

  test('выполняет задачу и сохраняет результат', async () => {
    const d = run({ 'echo': async (payload) => ({ echoed: payload.value }) });
    const task = createTask('echo', { value: 42 });

    await d.add(task);
    await waitFor(async () => (await d.get(task.id)).status === TaskStatus.COMPLETED);

    const done = await d.get(task.id);
    assert.deepEqual(done.result, { echoed: 42 });
    assert.equal(done.progress, 100);
  });

  test('прогресс отражается через ctx', async () => {
    const d = run({
      'progressive': async (_payload, ctx) => {
        ctx.updateProgress(50);
        return 'ok';
      },
    });
    const task = createTask('progressive', {});

    await d.add(task);
    await waitFor(async () => (await d.get(task.id)).status === TaskStatus.COMPLETED);
    // после завершения прогресс 100, но обработчик успел выставить промежуточный
    assert.equal((await d.get(task.id)).result, 'ok');
  });

  test('повторяет упавшую задачу и в итоге завершает', async () => {
    let calls = 0;
    const d = run({
      'flaky': async () => {
        calls += 1;
        if (calls < 3) throw new Error('временный сбой');
        return 'наконец';
      },
    });
    const task = createTask('flaky', {}, { maxAttempts: 3 });

    await d.add(task);
    await waitFor(async () => (await d.get(task.id)).status === TaskStatus.COMPLETED, 3000);

    const done = await d.get(task.id);
    assert.equal(calls, 3);
    assert.equal(done.attempts, 3);
    assert.equal(done.result, 'наконец');
  });

  test('после исчерпания попыток задача падает', async () => {
    const d = run({
      'broken': async () => {
        const err = new Error('всегда падает');
        err.code = 'BOOM';
        throw err;
      },
    });
    const task = createTask('broken', {}, { maxAttempts: 2 });

    await d.add(task);
    await waitFor(async () => (await d.get(task.id)).status === TaskStatus.FAILED, 3000);

    const failed = await d.get(task.id);
    assert.equal(failed.attempts, 2);
    assert.equal(failed.error.code, 'BOOM');
  });

  test('нет обработчика — задача сразу падает', async () => {
    const d = run({});
    const task = createTask('unknown', {});

    await d.add(task);
    await waitFor(async () => (await d.get(task.id)).status === TaskStatus.FAILED);

    assert.match((await d.get(task.id)).error.message, /обработчик/);
  });

  test('соблюдает предел параллелизма', async () => {
    let active = 0;
    let peak = 0;
    const d = run({
      'slow': async () => {
        active += 1;
        peak = Math.max(peak, active);
        await settle(40);
        active -= 1;
      },
    });

    const ids = [];
    for (let i = 0; i < 5; i += 1) {
      const task = createTask('slow', { i });
      ids.push(task.id);
      await d.add(task);
    }
    await waitFor(
      async () =>
        (await Promise.all(ids.map((id) => d.get(id)))).every(
          (t) => t.status === TaskStatus.COMPLETED,
        ),
      3000,
    );

    assert.ok(peak <= 2, `параллелизм превышен: ${peak}`);
  });

  test('stats считает задачи по статусам', async () => {
    const d = run({ 'echo': async () => 'ok' });
    await d.add(createTask('echo', {}));
    await settle(40);

    const stats = await d.stats();
    assert.equal(stats.driver, 'memory');
    assert.equal(stats.worker, true);
    assert.equal(stats.concurrency, 2);
    assert.equal(stats.byStatus[TaskStatus.COMPLETED], 1);
  });
});

describe('разделение ролей: процесс без воркера', () => {
  let driver;

  afterEach(() => driver?.close());

  test('без startWorker задача принимается, но не выполняется', async () => {
    let executed = false;
    driver = new MemoryDriver({ concurrency: 2, backoffMs: 10 });
    driver.setHandlers(
      new Map([
        [
          'echo',
          async () => {
            executed = true;
            return 'ok';
          },
        ],
      ]),
    );

    const task = createTask('echo', {});
    await driver.add(task);
    await settle(80);

    // Это гарантия WORKER_IN_API=false: процесс API принимает задачу (202),
    // но сам её не выполняет
    assert.equal(executed, false, 'обработчик не должен вызываться без воркера');
    assert.equal((await driver.get(task.id)).status, TaskStatus.PENDING);

    const stats = await driver.stats();
    assert.equal(stats.worker, false);
    assert.equal(stats.pending, 1);
  });

  test('startWorker разбирает накопленную очередь', async () => {
    driver = new MemoryDriver({ concurrency: 2, backoffMs: 10 });
    driver.setHandlers(new Map([['echo', async () => 'ok']]));

    const task = createTask('echo', {});
    await driver.add(task);
    await settle(30);
    assert.equal((await driver.get(task.id)).status, TaskStatus.PENDING);

    await driver.startWorker();
    await waitFor(async () => (await driver.get(task.id)).status === TaskStatus.COMPLETED);

    assert.equal((await driver.get(task.id)).result, 'ok');
  });
});
