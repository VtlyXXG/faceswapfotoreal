import { AsyncLocalStorage } from 'node:async_hooks';

/**
 * Контекст запроса, доступный на всей глубине стека без проброса параметров.
 * Благодаря ему логгер добавляет request_id к записям из сервисов и очереди,
 * которые ничего не знают об HTTP.
 */
const storage = new AsyncLocalStorage();

export const runWithContext = (context, fn) => storage.run({ ...context }, fn);

export const getContext = () => storage.getStore();

export const getRequestId = () => storage.getStore()?.requestId;

/**
 * Привязывает уже существующий контекст к отложенной работе (задача в очереди
 * переживает HTTP-ответ, но должна логироваться с тем же request_id).
 */
export const bindContext = (fn) => {
  const context = storage.getStore();
  return context ? (...args) => storage.run(context, () => fn(...args)) : fn;
};
