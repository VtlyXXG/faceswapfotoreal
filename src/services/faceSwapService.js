/**
 * Замена лица — единственная тяжёлая операция сервиса.
 *
 * Здесь живут и разбор путей под STORAGE_ROOT, и сам вызов ML-сервиса.
 * Точек входа две — разовая задача (`POST /personalize`) и доменный заказ
 * обложки (`coverService`), — но выполнение общее: runFaceSwap().
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from '../config/index.js';
import { coverSpecSchema } from '../domain/coverSpec.js';
import { enqueue, registerHandler } from '../queue/taskQueue.js';
import { faceSwap } from './mlClient.js';
import { saveArtifact } from '../storage/fileStorage.js';
import { ValidationError } from '../utils/errors.js';
import { createLogger } from '../utils/logger.js';

const log = createLogger('face-swap');

export const PERSONALIZE_TASK = 'ml.faceSwap';

// Разовому вызову ярлык заказа не нужен — остальное совпадает с заказом обложки
const personalizeSchema = coverSpecSchema.omit({ title: true });

/** Разрешает путь строго внутри STORAGE_ROOT — защита от выхода за пределы. */
export const resolveInStorage = (relative) => {
  const resolved = path.resolve(config.storage.root, relative);
  const root = path.resolve(config.storage.root);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new ValidationError('Путь вне хранилища', { path: relative });
  }
  return resolved;
};

/**
 * Читает исходники из хранилища и отдаёт их ML-сервису.
 * Вызывается только из фоновой задачи: диффузия идёт минутами.
 *
 * @param {{ source: string, target: string, options?: object }} spec
 * @returns {Promise<{ image: Buffer, meta: object, mimeType: string }>}
 */
export const runFaceSwap = async ({ source, target, options }, ctx = null) => {
  const [sourceFile, targetFile] = await Promise.all([
    fs.readFile(resolveInStorage(source)),
    fs.readFile(resolveInStorage(target)),
  ]);
  ctx?.updateProgress(20);

  const result = await faceSwap({ source: sourceFile, target: targetFile, options });
  ctx?.updateProgress(90);

  return result;
};

/** Расширение файла результата по MIME-типу ответа ML-сервиса. */
export const extensionFor = (mimeType) => (mimeType.includes('jpeg') ? 'jpg' : 'png');

// Фоновый обработчик разового вызова: результат кладётся в артефакты задачи.
registerHandler(PERSONALIZE_TASK, async (payload, ctx) => {
  const { image, meta, mimeType } = await runFaceSwap(payload, ctx);

  const artifact = await saveArtifact(ctx.taskId, `result.${extensionFor(mimeType)}`, image);
  log.info({ taskId: ctx.taskId, meta }, 'замена лица готова');

  return { artifact: { filename: artifact.filename, size: artifact.size }, meta };
});

/** Ставит разовую замену лица в очередь, сразу возвращает задачу (202 + task id). */
export const createPersonalization = async (input) => {
  const parsed = personalizeSchema.parse(input);
  // Проверяем ссылки заранее, чтобы клиент узнал об ошибке сразу, а не из задачи
  resolveInStorage(parsed.source);
  resolveInStorage(parsed.target);
  return enqueue(PERSONALIZE_TASK, parsed);
};
