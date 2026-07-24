import * as bookService from '../../services/bookService.js';
import { listArtifacts, readArtifact } from '../../storage/fileStorage.js';
import { NotFoundError } from '../../utils/errors.js';

export const create = async (req, res) => {
  const job = await bookService.createBook(req.body);
  // 202: заказ принят, генерация в фоне. Клиент опрашивает статус по taskId
  // (общий для всех задач) или по id книги (доменный ресурс).
  res.status(202).json({
    id: job.id,
    taskId: job.taskId,
    status: job.status,
    statusUrl: `/api/v1/tasks/${job.taskId}/status`,
    createdAt: job.createdAt,
  });
};

export const list = async (req, res) => {
  const books = await bookService.listBooks({
    limit: Number(req.query.limit) || 50,
    offset: Number(req.query.offset) || 0,
  });
  res.json({ items: books.map(summary) });
};

export const getOne = async (req, res) => {
  const job = await bookService.getBook(req.params.id);
  res.json(job);
};

export const getStatus = async (req, res) => {
  const job = await bookService.getBook(req.params.id);
  res.json(summary(job));
};

export const listFiles = async (req, res) => {
  await bookService.getBook(req.params.id);
  res.json({ files: await listArtifacts(req.params.id) });
};

export const downloadFile = async (req, res) => {
  await bookService.getBook(req.params.id);
  const { filename } = req.params;

  const files = await listArtifacts(req.params.id);
  if (!files.includes(filename)) throw new NotFoundError('Файл');

  res.type(filename).send(await readArtifact(req.params.id, filename));
};

const summary = (job) => ({
  id: job.id,
  status: job.status,
  progress: job.progress,
  title: job.outline?.title ?? job.spec.title ?? null,
  chapters: job.chapters.length,
  error: job.error,
  createdAt: job.createdAt,
  updatedAt: job.updatedAt,
});
