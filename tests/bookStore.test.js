import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MemoryBookStore } from '../src/storage/drivers/memoryBookStore.js';
import { PostgresBookStore } from '../src/storage/drivers/postgresBookStore.js';

const sampleJob = (id, overrides = {}) => ({
  id,
  taskId: `task_${id}`,
  status: 'pending',
  progress: 0,
  spec: { chapterCount: 3 },
  chapters: [],
  createdAt: overrides.createdAt ?? '2026-07-24T10:00:00.000Z',
  updatedAt: '2026-07-24T10:00:00.000Z',
  ...overrides,
});

// --- контракт на реальном memory-драйвере ---------------------------------

describe('MemoryBookStore (контракт)', () => {
  let store;
  beforeEach(() => {
    store = new MemoryBookStore();
  });

  test('save и findById', async () => {
    await store.save(sampleJob('book_1'));
    const found = await store.findById('book_1');
    assert.equal(found.id, 'book_1');
  });

  test('findById возвращает null для отсутствующего', async () => {
    assert.equal(await store.findById('нет'), null);
  });

  test('getById бросает NotFoundError', async () => {
    await assert.rejects(() => store.getById('нет'), /не найден/);
  });

  test('update мержит и сохраняет', async () => {
    await store.save(sampleJob('book_1'));
    const updated = await store.update('book_1', { status: 'completed', progress: 100 });
    assert.equal(updated.status, 'completed');
    assert.equal((await store.findById('book_1')).progress, 100);
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

describe('PostgresBookStore (SQL и маппинг)', () => {
  test('отклоняет небезопасное имя таблицы', () => {
    assert.throws(() => new PostgresBookStore({ table: 'books; DROP TABLE x' }), /имя таблицы/);
  });

  test('save делает upsert с корректными параметрами', async () => {
    const client = fakeClient();
    const store = new PostgresBookStore({ table: 'books', client });

    const returned = await store.save(sampleJob('book_1'));

    const { text, params } = client.calls[0];
    assert.match(text, /INSERT INTO books/);
    assert.match(text, /ON CONFLICT \(id\) DO UPDATE/);
    assert.equal(params[0], 'book_1');
    assert.equal(params[1], 'task_book_1'); // task_id колонкой
    assert.equal(params[2], 'pending'); // status колонкой
    assert.equal(JSON.parse(params[5]).id, 'book_1'); // весь документ в data
    assert.equal(returned.id, 'book_1');
  });

  test('findById маппит data и отдаёт null на пустом ответе', async () => {
    const job = sampleJob('book_1');
    const withRow = new PostgresBookStore({ client: fakeClient([{ rows: [{ data: job }] }]) });
    assert.equal((await withRow.findById('book_1')).id, 'book_1');

    const empty = new PostgresBookStore({ client: fakeClient([{ rows: [] }]) });
    assert.equal(await empty.findById('book_1'), null);
  });

  test('getById бросает NotFoundError на пустом ответе', async () => {
    const store = new PostgresBookStore({ client: fakeClient([{ rows: [] }]) });
    await assert.rejects(() => store.getById('нет'), /не найден/);
  });

  test('list сортирует по created_at и передаёт limit/offset', async () => {
    const rows = [{ data: sampleJob('b') }, { data: sampleJob('c') }];
    const client = fakeClient([{ rows }]);
    const store = new PostgresBookStore({ client });

    const result = await store.list({ limit: 2, offset: 5 });

    assert.match(client.calls[0].text, /ORDER BY created_at DESC/);
    assert.deepEqual(client.calls[0].params, [2, 5]);
    assert.deepEqual(
      result.map((j) => j.id),
      ['b', 'c'],
    );
  });

  test('update читает, мержит и апсертит', async () => {
    const existing = sampleJob('book_1');
    // 1-й запрос — SELECT (getById), 2-й — INSERT (save)
    const client = fakeClient([{ rows: [{ data: existing }] }, { rows: [] }]);
    const store = new PostgresBookStore({ client });

    await store.update('book_1', { status: 'completed' });

    const saveCall = client.calls[1];
    assert.match(saveCall.text, /INSERT INTO/);
    assert.equal(saveCall.params[2], 'completed');
    assert.equal(JSON.parse(saveCall.params[5]).status, 'completed');
  });

  test('init создаёт таблицу и индекс идемпотентно', async () => {
    const client = fakeClient();
    await new PostgresBookStore({ table: 'books', client }).init();

    assert.match(client.calls[0].text, /CREATE TABLE IF NOT EXISTS books/);
    assert.match(client.calls[1].text, /CREATE INDEX IF NOT EXISTS/);
  });
});
