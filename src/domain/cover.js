import { newId } from '../utils/id.js';

export const JobStatus = Object.freeze({
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
});

/** Заказ обложки: спецификация + состояние выполнения. */
export const createCoverJob = (spec) => ({
  id: newId('cover'),
  taskId: null, // задача очереди, выполняющая замену лица
  status: JobStatus.PENDING,
  progress: 0,
  spec,
  result: null, // { filename, meta } — что вернул ML-сервис
  artifacts: [],
  error: null,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});
