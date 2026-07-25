import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from '../config/index.js';

const OUTPUT_DIR = path.join(config.storage.root, 'output');
// Исходники, загруженные через панель: заказ ссылается на них путём внутри
// STORAGE_ROOT, поэтому каталог общий для API и воркеров (том cover-output).
const INPUT_DIR = path.join(config.storage.root, 'input');

const jobDir = (jobId) => path.join(OUTPUT_DIR, jobId);

/**
 * Сохраняет загруженный исходник и возвращает путь относительно STORAGE_ROOT —
 * ровно в таком виде его ждёт coverSpec (source/target).
 */
export const saveInput = async (filename, data) => {
  await fs.mkdir(INPUT_DIR, { recursive: true });
  await fs.writeFile(path.join(INPUT_DIR, filename), data);
  return { filename, path: `input/${filename}`, size: Buffer.byteLength(data) };
};

/** Сохраняет артефакт заказа (готовая обложка). */
export const saveArtifact = async (jobId, filename, data) => {
  const dir = jobDir(jobId);
  await fs.mkdir(dir, { recursive: true });
  const filePath = path.join(dir, filename);
  await fs.writeFile(filePath, data);
  return { filename, path: filePath, size: Buffer.byteLength(data) };
};

export const readArtifact = (jobId, filename) => fs.readFile(path.join(jobDir(jobId), filename));

export const listArtifacts = async (jobId) => {
  try {
    return await fs.readdir(jobDir(jobId));
  } catch {
    return [];
  }
};

export const removeJobArtifacts = (jobId) => fs.rm(jobDir(jobId), { recursive: true, force: true });

export const ensureStorage = async () => {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(INPUT_DIR, { recursive: true });
};
