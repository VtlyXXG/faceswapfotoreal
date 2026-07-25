import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MemoryCoverStore } from '../src/storage/drivers/memoryCoverStore.js';
import { PostgresCoverStore } from '../src/storage/drivers/postgresCoverStore.js';

const sampleJob = (id, overrides = {}) => ({
  id,
  taskId: `task_${id}`,
  status: 'pending',
  progress: 0,
  spec: { source: 'uploads/face.jpg', target: 'uploads/cover.png', options: {} },
  result: null,
  createdAt: overrides.createdAt ?? '2026-07-24T10:00:00.000Z',
  updatedAt: '2026-07-24T10:00:00.000Z',
  ...overrides,
});

// --- контракт на реальном memory-драйвере ---------------------------------

describe('MemoryCoverStore (контракт)', () => {
  let store;
  beforeEach(() => {
    store = new MemoryCoverStore();
  });

  test('save и findById', async () => {
    await store.save(sampleJob('cover_1'));
    const found = await store.findById('cover_1');
    assert.equal(found.id, 'cover_1');
  });

  test('findById возвращает null для отсутствующего', async () => {
    assert.equal(await store.findById('нет'), null);
  });

  test('getById бросает NotFoundError', async () => {
    await assert.rejects(() => store.getById('нет'), /не найден/);
  });

  test('update мержит и сохраняет', async () => {
    await store.save(sampleJob('cover_1'));
    const updated = await store.update('cover_1', { status: 'completed', progress: 100 });
    assert.equal(updated.status, 'completed');
    assert.equal((await store.findById('cover_1')).progress, 100);
  });

  test('list сортирует по createdAt убыванию и пагинирует', async () => {
    await store.save(sampleJob('a', { createdAt: '2026-07-24T10:00:00.000Z' }));
    await store.save(sampleJob('b', { createdAt: '2026-07-24T12:00:00.000Z' }));
    await store.save(sampleJob('c', { createdAt: '2026-07-24T11:00:00.000Z' }));

    const page = await store.list({ limit: 2, offset: 0 });
    assert.deepEqual(
      page.map((j) => j.id),
      ['b', 'c'],
    );
  });
});

// --- postgres-драйвер на подставном клиенте --------------------------------

/** Фейковый pg-клиент: пишет запросы и отдаёт заранее заданные ответы. */
const fakeClient = (responses = []) => {
  const calls = [];
  let i = 0;
  return {
    calls,
    query: async (text, params) => {
      calls.push({ text, params });
      return responses[i++] ?? { rows: [] };
    },
  };
};

describe('PostgresCoverStore (SQL и маппинг)', () => {
  test('отклоняет небезопасное имя таблицы', () => {
    assert.throws(() => new PostgresCoverStore({ table: 'covers; DROP TABLE x' }), /имя таблицы/);
  });

  test('save делает upsert с корректными параметрами', async () => {
    const client = fakeClient();
    const store = new PostgresCoverStore({ table: 'covers', client });

    const returned = await store.save(sampleJob('cover_1'));

    const { text, params } = client.calls[0];
    assert.match(text, /INSERT INTO covers/);
    assert.match(text, /ON CONFLICT \(id\) DO UPDATE/);
    assert.equal(params[0], 'cover_1');
    assert.equal(params[1], 'task_cover_1'); // task_id колонкой
    assert.equal(params[2], 'pending'); // status колонкой
    assert.equal(JSON.parse(params[5]).id, 'cover_1'); // весь документ в data
    assert.equal(returned.id, 'cover_1');
  });

  test('findById маппит data и отдаёт null на пустом ответе', async () => {
    const job = sampleJob('cover_1');
    const withRow = new PostgresCoverStore({ client: fakeClient([{ rows: [{ data: job }] }]) });
    assert.equal((await withRow.findById('cover_1')).id, 'cover_1');

    const empty = new PostgresCoverStore({ client: fakeClient([{ rows: [] }]) });
    assert.equal(await empty.findById('cover_1'), null);
  });

  test('getById бросает NotFoundError на пустом ответе', async () => {
    const store = new PostgresCoverStore({ client: fakeClient([{ rows: [] }]) });
    await assert.rejects(() => store.getById('нет'), /не найден/);
  });

  test('list сортирует по created_at и передаёт limit/offset', async () => {
    const rows = [{ data: sampleJob('b') }, { data: sampleJob('c') }];
    const client = fakeClient([{ rows }]);
    const store = new PostgresCoverStore({ client });

    const result = await store.list({ limit: 2, offset: 5 });

    assert.match(client.calls[0].text, /ORDER BY created_at DESC/);
    assert.deepEqual(client.calls[0].params, [2, 5]);
    assert.deepEqual(
      result.map((j) => j.id),
      ['b', 'c'],
    );
  });

  test('update читает, мержит и апсертит', async () => {
    const existing = sampleJob('cover_1');
    // 1-й запрос — SELECT (getById), 2-й — INSERT (save)
    const client = fakeClient([{ rows: [{ data: existing }] }, { rows: [] }]);
    const store = new PostgresCoverStore({ client });

    await store.update('cover_1', { status: 'completed' });

    const saveCall = client.calls[1];
    assert.match(saveCall.text, /INSERT INTO/);
    assert.equal(saveCall.params[2], 'completed');
    assert.equal(JSON.parse(saveCall.params[5]).status, 'completed');
  });

  test('init создаёт таблицу и индекс идемпотентно', async () => {
    const client = fakeClient();
    await new PostgresCoverStore({ table: 'covers', client }).init();

    assert.match(client.calls[0].text, /CREATE TABLE IF NOT EXISTS covers/);
    assert.match(client.calls[1].text, /CREATE INDEX IF NOT EXISTS/);
  });
});
