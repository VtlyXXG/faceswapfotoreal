import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import { createApp } from '../src/app.js';
import { config } from '../src/config/index.js';
import { initCoverStore, closeCoverStore } from '../src/storage/coverRepository.js';
import { latestLogFile } from '../src/utils/logger.js';

const flush = () => new Promise((resolve) => setTimeout(resolve, 200));

/** Строки, дописанные в активный файл лога с момента offset (в байтах). */
const readNewLines = async (file, offset) => {
  // offset — в байтах, поэтому режем буфер: в логах кириллица, и slice()
  // по символам разошёлся бы с размером файла
  const content = (await fs.readFile(file)).subarray(offset).toString('utf8');
  return content
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
};

const sizeOf = (file) => {
  try {
    return fsSync.statSync(file).size;
  } catch {
    return 0;
  }
};

describe('структурированное логирование', () => {
  let server;
  let baseUrl;

  before(async () => {
    // Тест ниже дёргает /covers/:id, а значит и хранилище: с DB_DRIVER=postgres
    // таблицы может ещё не быть, и вместо ожидаемого 404 вернулась бы 500
    await initCoverStore();
    server = createApp().listen(0);
    await new Promise((resolve) => server.once('listening', resolve));
    baseUrl = `http://127.0.0.1:${server.address().port}`;
  });

  // Пул pg — открытый хендл: без закрытия процесс не завершится после тестов
  after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await closeCoverStore();
  });

  test('входящий X-Request-ID возвращается и попадает в лог', async () => {
    const activeFile = latestLogFile();
    const offset = activeFile ? sizeOf(activeFile) : 0;
    const incoming = 'node-test-' + Date.now();

    const response = await fetch(`${baseUrl}/api/v1/health`, {
      headers: { 'x-request-id': incoming },
    });
    assert.equal(response.headers.get('x-request-id'), incoming);

    await flush();
    const lines = await readNewLines(latestLogFile(), offset);
    const entry = lines.find((l) => l.request_id === incoming);

    assert.ok(entry, 'запись с request_id не найдена в файле лога');
    assert.equal(entry.service, 'cover-service');
    assert.equal(entry.level, 'info');
    assert.ok(entry.time.endsWith('Z'));
    assert.equal(entry.res.status_code, 200);
  });

  test('без заголовка request_id генерируется', async () => {
    const response = await fetch(`${baseUrl}/api/v1/health`);
    const generated = response.headers.get('x-request-id');

    assert.ok(generated);
    assert.match(generated, /^[0-9a-f-]{36}$/);
  });

  test('некорректный request_id заменяется на сгенерированный', async () => {
    const response = await fetch(`${baseUrl}/api/v1/health`, {
      headers: { 'x-request-id': 'bad id!' },
    });

    assert.notEqual(response.headers.get('x-request-id'), 'bad id!');
  });

  test('404 логируется как warn и не попадает в error.log', async () => {
    const errorFile = latestLogFile(config.logger.errorFile);
    const before = errorFile ? sizeOf(errorFile) : 0;

    await fetch(`${baseUrl}/api/v1/covers/does-not-exist`);
    await flush();

    const after = latestLogFile(config.logger.errorFile);
    assert.equal(after ? sizeOf(after) : 0, before);
  });
});

describe('ротация логов', () => {
  test('имя активного файла соответствует схеме pino-roll', async () => {
    const file = latestLogFile();

    assert.ok(file, 'активный файл лога не найден');
    // app.2026-07-23.1.log — дата и порядковый номер обязательны
    assert.match(path.basename(file), /^app\.\d{4}-\d{2}-\d{2}\.\d+\.log$/);
  });

  test('ротация по размеру создаёт новый файл и удаляет лишние', async () => {
    const { default: roll } = await import('pino-roll');
    const dir = path.join(config.logger.dir, 'rotation-test');
    await fs.rm(dir, { recursive: true, force: true });

    const stream = await roll({
      file: path.join(dir, 'probe'),
      extension: '.log',
      size: '1k',
      limit: { count: 1 },
      mkdir: true,
    });

    // ~4 КБ при пороге 1 КБ: гарантированно несколько ротаций
    for (let i = 0; i < 40; i += 1) {
      stream.write(`${JSON.stringify({ i, payload: 'x'.repeat(100) })}\n`);
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    await flush();

    const files = (await fs.readdir(dir)).filter((n) => n.endsWith('.log')).sort();
    stream.end();

    assert.ok(files.length > 1, `ротации не произошло: ${files.join(', ')}`);
    // limit.count=1 → активный файл + один ротированный
    assert.ok(files.length <= 2, `лишние файлы не удалены: ${files.join(', ')}`);

    await fs.rm(dir, { recursive: true, force: true });
  });
});
