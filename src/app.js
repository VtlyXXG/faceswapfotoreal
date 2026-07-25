import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import pinoHttp from 'pino-http';
import { apiRoutes } from './api/routes/index.js';
import { errorHandler, notFoundHandler } from './api/middleware/errorHandler.js';
import { requestId } from './api/middleware/requestId.js';
import { logger } from './utils/logger.js';

// Путь от модуля, а не от cwd: панель должна находиться и в контейнере,
// и при запуске из произвольного каталога
const PUBLIC_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../public');

export const createApp = () => {
  const app = express();

  app.disable('x-powered-by');

  // Порядок важен: request_id должен существовать до первой строки лога
  app.use(requestId);
  app.use(
    pinoHttp({
      logger,
      genReqId: (req) => req.id,
      customSuccessMessage: (req, res) => `${req.method} ${req.url} → ${res.statusCode}`,
      customErrorMessage: (req, res, err) => `${req.method} ${req.url} → ${err.message}`,
      customLogLevel: (_req, res, err) => {
        if (err || res.statusCode >= 500) return 'error';
        if (res.statusCode >= 400) return 'warn';
        return 'info';
      },
      serializers: {
        req: (req) => ({ method: req.method, url: req.url }),
        res: (res) => ({ status_code: res.statusCode }),
      },
    }),
  );

  app.use(express.json({ limit: '1mb' }));

  app.use('/api/v1', apiRoutes);

  // Панель локального тестирования: GET / отдаёт public/index.html.
  // Смонтирована после API, чтобы статика не перехватывала маршруты.
  app.use(express.static(PUBLIC_DIR));

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
};

export default createApp;
