import { createBookJob, JobStatus } from '../domain/book.js';
import { parseBookSpec } from '../domain/bookSpec.js';
import { bookRepository } from '../storage/bookRepository.js';
import { saveArtifact } from '../storage/fileStorage.js';
import { jobQueue } from '../queue/jobQueue.js';
import { createLogger } from '../utils/logger.js';
import { bindContext } from '../utils/requestContext.js';
import {
  generateOutline,
  generateChapter,
  generateIllustration,
} from './generationService.js';
import { renderBook } from './renderService.js';

const log = createLogger('book');

/** Принимает заказ, ставит его в очередь и сразу возвращает id (генерация асинхронна). */
export const createBook = async (input) => {
  const spec = parseBookSpec(input);
  const job = await bookRepository.save(createBookJob(spec));

  // bindContext переносит request_id в фоновую задачу: генерация переживает
  // HTTP-ответ, но её логи остаются в той же трассе, что и запрос клиента
  jobQueue
    .enqueue(job.id, bindContext(() => runPipeline(job.id)))
    .catch((err) => log.error({ jobId: job.id, err }, 'пайплайн генерации упал'));

  return job;
};

export const getBook = (id) => bookRepository.getById(id);
export const listBooks = (options) => bookRepository.list(options);

/** Полный пайплайн: структура → главы → иллюстрации → рендер файлов. */
export const runPipeline = async (jobId) => {
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

    return bookRepository.update(jobId, {
      status: JobStatus.COMPLETED,
      progress: 100,
      chapters,
      artifacts,
    });
  } catch (err) {
    await bookRepository.update(jobId, {
      status: JobStatus.FAILED,
      error: { message: err.message, code: err.code ?? 'PIPELINE_ERROR' },
    });
    throw err;
  }
};
