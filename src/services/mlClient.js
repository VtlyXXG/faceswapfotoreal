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

/**
 * Face-swap на Python ML-сервисе. Долгая операция (диффузия — минуты),
 * поэтому вызывается только из фоновой задачи, а не из HTTP-обработчика.
 *
 * @param {{ source: Buffer, target: Buffer, options?: object }} input
 * @returns {Promise<{ image: Buffer, meta: object, mimeType: string }>}
 */
export const faceSwap = async ({ source, target, options = {} }) => {
  const form = new FormData();
  form.append('source', new Blob([source]), 'source.jpg');
  form.append('target', new Blob([target]), 'target.png');
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined && value !== null) form.append(key, String(value));
  }

  const response = await mlFetch('/face-swap', {
    method: 'POST',
    body: form,
    // Постобработка диффузией долгая — таймаут щедрый, но он на стороне воркера,
    // клиент API его не ждёт
    timeoutMs: config.mlService.faceSwapTimeoutMs,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    const err = new Error(`ml-service вернул ${response.status}`);
    err.code = 'ML_FACE_SWAP_FAILED';
    err.details = detail.slice(0, 500);
    throw err;
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  const meta = safeJson(response.headers.get('x-swap-meta'));
  return { image: buffer, meta, mimeType: response.headers.get('content-type') ?? 'image/png' };
};

const safeJson = (value) => {
  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
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
