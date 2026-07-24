import { newId } from '../utils/id.js';

export const JobStatus = Object.freeze({
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
});

/** Заказ на генерацию: спецификация + состояние выполнения. */
export const createBookJob = (spec) => ({
  id: newId('book'),
  taskId: null, // задача очереди, выполняющая генерацию
  status: JobStatus.PENDING,
  progress: 0,
  spec,
  outline: null,
  chapters: [],
  artifacts: [],
  error: null,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});

export const createChapter = ({ index, title, summary, content = '', illustration = null }) => ({
  index,
  title,
  summary,
  content,
  illustration,
});
