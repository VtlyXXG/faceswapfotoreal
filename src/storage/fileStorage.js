import fs from 'node:fs/promises';
import path from 'node:path';
import { config } from '../config/index.js';

const OUTPUT_DIR = path.join(config.storage.root, 'output');

const jobDir = (jobId) => path.join(OUTPUT_DIR, jobId);

/** Сохраняет артефакт генерации (текст, изображение, готовый файл книги). */
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

export const ensureStorage = () => fs.mkdir(OUTPUT_DIR, { recursive: true });
