import { ZodError } from 'zod';
import { AppError } from '../../utils/errors.js';
import { logger } from '../../utils/logger.js';

export const notFoundHandler = (req, res) => {
  res.status(404).json({ error: { code: 'NOT_FOUND', message: `Маршрут ${req.path} не найден` } });
};

// eslint-disable-next-line no-unused-vars -- Express определяет обработчик ошибок по арности
export const errorHandler = (err, req, res, next) => {
  if (err instanceof ZodError) {
    return res.status(422).json({
      error: { code: 'VALIDATION_ERROR', message: 'Некорректная спецификация книги', details: err.issues },
    });
  }

  if (err instanceof AppError) {
    return res
      .status(err.status)
      .json({ error: { code: err.code, message: err.message, details: err.details } });
  }

  logger.error({ err }, 'необработанная ошибка');
  res.status(500).json({ error: { code: 'INTERNAL_ERROR', message: 'Внутренняя ошибка сервера' } });
};
