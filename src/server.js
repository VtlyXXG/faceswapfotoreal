import { createApp } from './app.js';
import { config } from './config/index.js';
import { ensureStorage } from './storage/fileStorage.js';
import { checkProviders } from './ai/registry.js';
import { logger } from './utils/logger.js';

const start = async () => {
  await ensureStorage();

  const app = createApp();
  const server = app.listen(config.server.port, config.server.host, async () => {
    logger.info(
      { url: `http://${config.server.host}:${config.server.port}/api/v1`, env: config.env },
      'сервис запущен',
    );

    const models = await checkProviders();
    logger.info({ models }, 'состояние локальных ИИ-моделей');
  });

  const shutdown = (signal) => {
    logger.info({ signal }, 'остановка сервиса');
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 10_000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
};

start().catch((err) => {
  logger.fatal({ err }, 'не удалось запустить сервис');
  process.exit(1);
});
