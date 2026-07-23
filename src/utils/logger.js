import fs from 'node:fs';
import path from 'node:path';
import pino from 'pino';
import roll from 'pino-roll';
import { config } from '../config/index.js';
import { getRequestId } from './requestContext.js';

fs.mkdirSync(config.logger.dir, { recursive: true });

const { rotation } = config.logger;

/**
 * Ротируемый файловый поток.
 * pino-roll подключается напрямую (не через pino.transport), чтобы не поднимать
 * worker-поток: он держал бы event loop и `node --test` не завершался бы.
 * Внутренний таймер ротации у pino-roll .unref()-нут, процессу он не мешает.
 */
const rollingStream = (basename) =>
  roll({
    file: path.join(config.logger.dir, basename),
    extension: '.log',
    frequency: rotation.frequency,
    size: rotation.size,
    dateFormat: rotation.dateFormat,
    limit: { count: rotation.limitCount },
    mkdir: true,
  });

/** Консоль: pretty в разработке, тот же JSON в проде. */
let consoleStream = process.stdout;
if (!config.isProduction) {
  try {
    const { default: pretty } = await import('pino-pretty');
    consoleStream = pretty({
      colorize: true,
      translateTime: 'HH:MM:ss',
      messageKey: 'message',
      ignore: 'pid,service,hostname',
    });
  } catch {
    // pino-pretty не установлен — пишем в консоль тот же JSON, что и в файл
  }
}

const [appStream, errorStream] = await Promise.all([
  rollingStream(config.logger.file),
  rollingStream(config.logger.errorFile),
]);

const streams = [
  { level: config.logger.level, stream: consoleStream },
  { level: config.logger.level, stream: appStream },
  { level: 'error', stream: errorStream },
];

export const logger = pino(
  {
    level: config.logger.level,
    base: { service: 'book-service', pid: process.pid },
    messageKey: 'message',
    timestamp: pino.stdTimeFunctions.isoTime,
    // Уровень словом, а не числом — так логи читаются одинаково с ml-service
    formatters: { level: (label) => ({ level: label }) },
    // request_id подмешивается автоматически из AsyncLocalStorage
    mixin() {
      const requestId = getRequestId();
      return requestId ? { request_id: requestId } : {};
    },
    redact: {
      paths: ['req.headers.authorization', 'req.headers.cookie', '*.password'],
      remove: true,
    },
  },
  pino.multistream(streams, { dedupe: false }),
);

/**
 * Актуальный файл лога: pino-roll именует файлы как `app.2026-07-23.1.log`,
 * поэтому фиксированного имени не существует. Нужен ops-скриптам и тестам.
 *
 * @param {string} basename `app` или `error` (без расширения)
 * @returns {string|null} абсолютный путь или null, если ничего не записано
 */
export const latestLogFile = (basename = config.logger.file) => {
  const prefix = `${basename}.`;
  const candidates = fs
    .readdirSync(config.logger.dir)
    .filter((name) => name.startsWith(prefix) && name.endsWith('.log'))
    .map((name) => {
      const file = path.join(config.logger.dir, name);
      return { file, mtime: fs.statSync(file).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);

  return candidates[0]?.file ?? null;
};

export const createLogger = (scope) => logger.child({ scope });

export default logger;
