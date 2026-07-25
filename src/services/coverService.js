import { createCoverJob, JobStatus } from '../domain/cover.js';
import { parseCoverSpec } from '../domain/coverSpec.js';
import { coverRepository } from '../storage/coverRepository.js';
import { saveArtifact } from '../storage/fileStorage.js';
import { buildTask, submit, registerHandler } from '../queue/taskQueue.js';
import { createLogger } from '../utils/logger.js';
import { runFaceSwap, extensionFor, resolveInStorage } from './faceSwapService.js';

const log = createLogger('cover');

export const COVER_TASK = 'cover.faceSwap';

// Обработка обложки — фоновая задача. Обработчик регистрируется при импорте
// модуля, поэтому доступен воркеру любого драйвера очереди.
registerHandler(COVER_TASK, (payload, ctx) => runPipeline(payload.coverId, ctx));

/** Принимает заказ, ставит его в очередь и сразу возвращает id (обработка асинхронна). */
export const createCover = async (input) => {
  const spec = parseCoverSpec(input);
  // Проверяем ссылки заранее, чтобы клиент узнал об ошибке сразу, а не из задачи
  resolveInStorage(spec.source);
  resolveInStorage(spec.target);

  const coverJob = createCoverJob(spec);

  // Связываем заказ с задачей ДО постановки в очередь: воркер memory-драйвера
  // может начать выполнение сразу, и taskId должен быть уже сохранён
  const task = buildTask(COVER_TASK, { coverId: coverJob.id });
  coverJob.taskId = task.id;
  const saved = await coverRepository.save(coverJob);

  await submit(task);
  return saved;
};

export const getCover = (id) => coverRepository.getById(id);
export const listCovers = (options) => coverRepository.list(options);

/**
 * Пайплайн заказа: исходники из хранилища → замена лица на ML-сервисе → артефакт.
 * Запускается воркером очереди; ctx позволяет отражать прогресс в задаче.
 */
export const runPipeline = async (jobId, ctx = null) => {
  const job = await coverRepository.getById(jobId);

  await coverRepository.update(jobId, { status: JobStatus.RUNNING, progress: 0 });

  try {
    const { image, meta, mimeType } = await runFaceSwap(job.spec, ctx);

    const filename = `cover.${extensionFor(mimeType)}`;
    const artifact = await saveArtifact(jobId, filename, image);

    await coverRepository.update(jobId, {
      status: JobStatus.COMPLETED,
      progress: 100,
      result: { filename, meta },
      artifacts: [{ filename: artifact.filename, size: artifact.size }],
    });
    log.info({ jobId, meta }, 'обложка готова');

    // Компактный результат задачи; полные данные заказа — по GET /covers/:id
    return { coverId: jobId, artifact: filename };
  } catch (err) {
    await coverRepository.update(jobId, {
      status: JobStatus.FAILED,
      error: { message: err.message, code: err.code ?? 'PIPELINE_ERROR' },
    });
    throw err;
  }
};
