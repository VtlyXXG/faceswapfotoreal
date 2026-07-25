import { createApp } from './app.js';
import { config } from './config/index.js';
import { ensureStorage } from './storage/fileStorage.js';
import { initCoverStore, closeCoverStore } from './storage/coverRepository.js';
import { startWorker, closeQueue } from './queue/taskQueue.js';
// Импорт ради побочного эффекта — регистрации обработчиков задач в очереди
import './services/coverService.js';
import './services/faceSwapService.js';
import { logger } from './utils/logger.js';

const start = async () => {
  await ensureStorage();
  await initCoverStore();

  // WORKER_IN_API=false — жёсткое разделение ролей: процесс API только
  // принимает запросы и отдаёт 202, задачи выполняют отдельные воркеры
  // (npm run worker), масштабируемые независимо.
  if (config.queue.workerInApi) {
    await startWorker();
  } else {
    logger.info('воркер в процессе API отключён (WORKER_IN_API=false)');
    if (config.queue.driver !== 'redis') {
      logger.warn(
        { driver: config.queue.driver },
        'воркер отключён при драйвере не-redis: очередь у каждого процесса своя, ' +
          'задачи никто не выполнит. Нужен QUEUE_DRIVER=redis',
      );
    }
  }

  const app = createApp();
  const server = app.listen(config.server.port, config.server.host, () => {
    logger.info(
      {
        url: `http://${config.server.host}:${config.server.port}/api/v1`,
        env: config.env,
        queue: config.queue.driver,
        workerInApi: config.queue.workerInApi,
        mlService: config.mlService.baseUrl,
      },
      'сервис запущен',
    );
  });

  const shutdown = async (signal) => {
    logger.info({ signal }, 'остановка сервиса');
    server.close(async () => {
      await closeQueue();
      await closeCoverStore();
      process.exit(0);
    });
    setTimeout(() => process.exit(1), 10_000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
};

start().catch((err) => {
  logger.fatal({ err }, 'не удалось запустить сервис');
  process.exit(1);
});
