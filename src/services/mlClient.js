/**
 * Клиент ML-микросервиса (FastAPI).
 * Единственное место, где Node ходит в Python — и единственное, где нужно
 * не забыть пробросить X-Request-ID, чтобы цепочку можно было собрать по логам.
 */
import { config } from '../config/index.js';
import { getRequestId } from '../utils/requestContext.js';
import { createLogger } from '../utils/logger.js';
import { REQUEST_ID_HEADER } from '../api/middleware/requestId.js';

const log = createLogger('ml-client');

export const mlFetch = async (pathname, { method = 'GET', body, headers = {}, timeoutMs } = {}) => {
  const url = new URL(pathname, config.mlService.baseUrl);
  const requestId = getRequestId();
  const startedAt = Date.now();

  const response = await fetch(url, {
    method,
    body,
    headers: { ...headers, ...(requestId ? { [REQUEST_ID_HEADER]: requestId } : {}) },
    signal: AbortSignal.timeout(timeoutMs ?? config.mlService.timeoutMs),
  });

  log.debug(
    { url: url.pathname, method, status: response.status, duration_ms: Date.now() - startedAt },
    'запрос к ml-service',
  );

  return response;
};

/** Готовность ML-сервиса для /health/ready. Никогда не бросает. */
export const checkMlService = async () => {
  try {
    const response = await mlFetch('/health/ready', { timeoutMs: 5000 });
    const body = await response.json().catch(() => ({}));

    return {
      available: response.ok,
      status: body.status ?? (response.ok ? 'ready' : 'degraded'),
      reason: body.reason ?? null,
      device: body.device ?? null,
    };
  } catch (err) {
    log.warn({ err: err.message }, 'ml-service недоступен');
    return { available: false, status: 'unreachable', reason: err.message, device: null };
  }
};
