/**
 * Автономный воркер очереди — процесс, который только выполняет задачи и не
 * принимает HTTP. Имеет смысл при QUEUE_DRIVER=redis: воркеры масштабируются
 * горизонтально независимо от API (несколько `npm run worker` на общий Redis).
 *
 * С QUEUE_DRIVER=memory очередь живёт внутри процесса API, поэтому отдельный
 * воркер работал бы с собственной пустой очередью — запускать его нет смысла.
 */
import { config } from './config/index.js';
import { startWorker, closeQueue } from './queue/taskQueue.js';
import { initCoverStore, closeCoverStore } from './storage/coverRepository.js';
// Регистрация обработчиков задач
import './services/coverService.js';
import './services/faceSwapService.js';
import { logger } from './utils/logger.js';

const start = async () => {
  if (config.queue.driver !== 'redis') {
    logger.warn(
      { driver: config.queue.driver },
      'воркер имеет смысл только при QUEUE_DRIVER=redis; с memory очередь в процессе API',
    );
  }

  // Воркер пишет статусы заказов, поэтому ему нужно то же хранилище, что и API.
  // С DB_DRIVER=memory у отдельного процесса будет своя пустая память —
  // общее состояние требует DB_DRIVER=postgres
  await initCoverStore();
  await startWorker();
  logger.info({ queue: config.queue.driver, concurrency: config.queue.concurrency }, 'воркер запущен');

  const shutdown = async (signal) => {
    logger.info({ signal }, 'остановка воркера');
    await closeQueue();
    await closeCoverStore();
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
};

start().catch((err) => {
  logger.fatal({ err }, 'не удалось запустить воркер');
  process.exit(1);
});
