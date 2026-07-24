import { newId } from '../utils/id.js';

/**
 * Универсальная задача очереди. Доменные сущности (заказ книги, face-swap)
 * ссылаются на неё по id — статус любой фоновой работы читается единообразно
 * через GET /tasks/:id, независимо от драйвера.
 */
export const TaskStatus = Object.freeze({
  PENDING: 'pending', // в очереди, ещё не взята
  ACTIVE: 'active', // выполняется
  RETRYING: 'retrying', // упала, ждёт повтора
  COMPLETED: 'completed',
  FAILED: 'failed', // исчерпаны попытки
});

export const isTerminal = (status) =>
  status === TaskStatus.COMPLETED || status === TaskStatus.FAILED;

export const createTask = (type, payload, { maxAttempts = 3, requestId = null } = {}) => {
  const now = new Date().toISOString();
  return {
    id: newId('task'),
    type,
    payload,
    status: TaskStatus.PENDING,
    progress: 0,
    attempts: 0,
    maxAttempts,
    result: null,
    error: null,
    // requestId путешествует с задачей, чтобы фоновые логи оставались в трассе
    // исходного HTTP-запроса — в том числе через Redis в другой процесс
    meta: { requestId },
    createdAt: now,
    updatedAt: now,
  };
};

/** Публичное представление задачи для API — без внутреннего payload. */
export const toPublicTask = (task) => {
  if (!task) return null;
  return {
    id: task.id,
    type: task.type,
    status: task.status,
    progress: task.progress,
    attempts: task.attempts,
    maxAttempts: task.maxAttempts,
    result: task.result,
    error: task.error,
    createdAt: task.createdAt,
    updatedAt: task.updatedAt,
  };
};
