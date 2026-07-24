import fs from 'node:fs/promises';
import path from 'node:path';
import { z } from 'zod';
import { config } from '../config/index.js';
import { enqueue, registerHandler } from '../queue/taskQueue.js';
import { faceSwap } from './mlClient.js';
import { saveArtifact } from '../storage/fileStorage.js';
import { ValidationError } from '../utils/errors.js';
import { createLogger } from '../utils/logger.js';

const log = createLogger('personalize');

export const PERSONALIZE_TASK = 'ml.faceSwap';

const personalizeSchema = z.object({
  // Пути к уже загруженным файлам под STORAGE_ROOT (server-side ссылки)
  source: z.string().min(1),
  target: z.string().min(1),
  options: z
    .object({
      enhance: z.boolean().optional(),
      style_strength: z.number().min(0).max(2).optional(),
      art_style: z.string().max(120).optional(),
      output_format: z.enum(['png', 'jpg']).optional(),
    })
    .strict()
    .optional(),
});

/** Разрешает путь строго внутри STORAGE_ROOT — защита от выхода за пределы. */
const resolveInStorage = (relative) => {
  const resolved = path.resolve(config.storage.root, relative);
  const root = path.resolve(config.storage.root);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new ValidationError('Путь вне хранилища', { path: relative });
  }
  return resolved;
};

// Фоновый обработчик: тяжёлый вызов Python ML-сервиса идёт здесь, вне HTTP.
registerHandler(PERSONALIZE_TASK, async (payload, ctx) => {
  const [source, target] = await Promise.all([
    fs.readFile(resolveInStorage(payload.source)),
    fs.readFile(resolveInStorage(payload.target)),
  ]);
  ctx?.updateProgress(20);

  const { image, meta, mimeType } = await faceSwap({ source, target, options: payload.options });
  ctx?.updateProgress(90);

  const ext = mimeType.includes('jpeg') ? 'jpg' : 'png';
  const artifact = await saveArtifact(ctx.taskId, `result.${ext}`, image);
  log.info({ taskId: ctx.taskId, meta }, 'персонализация готова');

  return { artifact: { filename: artifact.filename, size: artifact.size }, meta };
});

/** Ставит персонализацию в очередь, сразу возвращает задачу (202 + task id). */
export const createPersonalization = async (input) => {
  const parsed = personalizeSchema.parse(input);
  // Проверяем ссылки заранее, чтобы клиент узнал об ошибке сразу, а не из задачи
  resolveInStorage(parsed.source);
  resolveInStorage(parsed.target);
  return enqueue(PERSONALIZE_TASK, parsed);
};
