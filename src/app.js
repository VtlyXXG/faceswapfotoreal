import express from 'express';
import pinoHttp from 'pino-http';
import { apiRoutes } from './api/routes/index.js';
import { errorHandler, notFoundHandler } from './api/middleware/errorHandler.js';
import { requestId } from './api/middleware/requestId.js';
import { logger } from './utils/logger.js';

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

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
};

export default createApp;
