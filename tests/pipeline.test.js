import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { registerTextProvider, registerImageProvider, resetProviders } from '../src/ai/registry.js';
import { BaseTextProvider } from '../src/ai/providers/BaseTextProvider.js';
import { MockImageProvider } from '../src/ai/providers/MockImageProvider.js';
import { bookRepository } from '../src/storage/bookRepository.js';
import { JobStatus } from '../src/domain/book.js';
import { createBook, getBook } from '../src/services/bookService.js';
import { startWorker } from '../src/queue/taskQueue.js';

/** Провайдер, отдающий валидную структуру и короткие главы. */
class StubTextProvider extends BaseTextProvider {
  async generate({ prompt }) {
    if (prompt.includes('Верни строго JSON')) {
      return {
        text: JSON.stringify({
          title: 'Тестовая книга',
          chapters: [{ index: 1, title: 'Начало', summary: 'знакомство' }],
        }),
        model: 'stub',
      };
    }
    return { text: 'Текст главы.', model: 'stub' };
  }
  async healthCheck() {
    return true;
  }
}

const waitFor = async (predicate, timeoutMs = 5000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error('условие не выполнилось за отведённое время');
};

describe('пайплайн генерации', () => {
  beforeEach(async () => {
    await bookRepository.clear();
    resetProviders();
    registerTextProvider(process.env.AI_TEXT_PROVIDER ?? 'mock', StubTextProvider);
    registerImageProvider(process.env.AI_IMAGE_PROVIDER ?? 'mock', MockImageProvider);
    // Тест играет роль процесса-воркера: без него задачи только копятся
    await startWorker();
  });

  test('заказ доходит до статуса completed', async () => {
    const job = await createBook({ recipient: { name: 'Аня' }, chapterCount: 1 });
    assert.equal(job.status, JobStatus.PENDING);

    await waitFor(async () => (await getBook(job.id)).status === JobStatus.COMPLETED);

    const finished = await getBook(job.id);
    assert.equal(finished.progress, 100);
    assert.equal(finished.chapters.length, 1);
    assert.ok(finished.artifacts.some((a) => a.filename === 'book.md'));
  });
});
