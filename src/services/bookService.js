import { createBookJob, JobStatus } from '../domain/book.js';
import { parseBookSpec } from '../domain/bookSpec.js';
import { bookRepository } from '../storage/bookRepository.js';
import { saveArtifact } from '../storage/fileStorage.js';
import { buildTask, submit, registerHandler } from '../queue/taskQueue.js';
import { createLogger } from '../utils/logger.js';
import {
  generateOutline,
  generateChapter,
  generateIllustration,
} from './generationService.js';
import { renderBook } from './renderService.js';

const log = createLogger('book');

export const BOOK_TASK = 'book.generate';

// Генерация книги — фоновая задача. Обработчик регистрируется при импорте
// модуля, поэтому доступен воркеру любого драйвера очереди.
registerHandler(BOOK_TASK, (payload, ctx) => runPipeline(payload.bookId, ctx));

/** Принимает заказ, ставит его в очередь и сразу возвращает id (генерация асинхронна). */
export const createBook = async (input) => {
  const spec = parseBookSpec(input);
  const bookJob = createBookJob(spec);

  // Связываем книгу с задачей ДО постановки в очередь: воркер memory-драйвера
  // может начать выполнение сразу, и taskId должен быть уже сохранён
  const task = buildTask(BOOK_TASK, { bookId: bookJob.id });
  bookJob.taskId = task.id;
  const saved = await bookRepository.save(bookJob);

  await submit(task);
  return saved;
};

export const getBook = (id) => bookRepository.getById(id);
export const listBooks = (options) => bookRepository.list(options);

/**
 * Полный пайплайн: структура → главы → иллюстрации → рендер файлов.
 * Запускается воркером очереди; ctx позволяет отражать прогресс в задаче.
 */
export const runPipeline = async (jobId, ctx = null) => {
  const job = await bookRepository.getById(jobId);
  const { spec } = job;

  await bookRepository.update(jobId, { status: JobStatus.RUNNING, progress: 0 });

  try {
    const outline = await generateOutline(spec);
    await bookRepository.update(jobId, { outline, progress: 5 });

    const chapters = [];
    for (const chapter of outline.chapters) {
      const previousSummary = chapters.at(-1)?.summary;
      const written = await generateChapter(spec, chapter, previousSummary);

      if (spec.withIllustrations) {
        const illustration = await generateIllustration(spec, written);
        if (illustration) {
          const filename = `chapter-${written.index}.png`;
          await saveArtifact(jobId, filename, illustration.data);
          written.illustration = { filename, model: illustration.model };
        }
      }

      chapters.push(written);
      const progress = 5 + Math.round((chapters.length / outline.chapters.length) * 85);
      await bookRepository.update(jobId, { chapters, progress });
      ctx?.updateProgress(progress);
      log.info({ jobId, chapter: written.index }, 'глава готова');
    }

    const book = { title: outline.title, chapters };
    const artifacts = [];
    for (const format of spec.output.formats) {
      const rendered = renderBook(book, format);
      if (!rendered) {
        log.warn({ jobId, format }, 'формат пока не поддерживается, пропущен');
        continue;
      }
      artifacts.push(await saveArtifact(jobId, rendered.filename, rendered.content));
    }

    await bookRepository.update(jobId, {
      status: JobStatus.COMPLETED,
      progress: 100,
      chapters,
      artifacts,
    });
    // Компактный результат задачи; полные данные книги — по GET /books/:id
    return { bookId: jobId, title: book.title, artifacts: artifacts.map((a) => a.filename) };
  } catch (err) {
    await bookRepository.update(jobId, {
      status: JobStatus.FAILED,
      error: { message: err.message, code: err.code ?? 'PIPELINE_ERROR' },
    });
    throw err;
  }
};
