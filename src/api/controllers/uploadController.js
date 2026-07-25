import { saveInput } from '../../storage/fileStorage.js';
import { newId } from '../../utils/id.js';
import { ValidationError } from '../../utils/errors.js';

// Панель шлёт файл сырым телом с его Content-Type — multipart не нужен,
// а вместе с ним не нужна и лишняя зависимость в проекте.
const EXTENSIONS = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
};

/** Принимает изображение и возвращает путь под STORAGE_ROOT для source/target. */
export const create = async (req, res) => {
  const contentType = (req.headers['content-type'] ?? '').split(';')[0].trim();
  const extension = EXTENSIONS[contentType];

  if (!extension) {
    throw new ValidationError('Неподдерживаемый тип изображения', {
      contentType,
      supported: Object.keys(EXTENSIONS),
    });
  }

  if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
    throw new ValidationError('Пустое тело запроса');
  }

  const saved = await saveInput(`${newId('src')}.${extension}`, req.body);
  res.status(201).json({ path: saved.path, size: saved.size });
};
