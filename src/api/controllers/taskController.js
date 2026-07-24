import { getTask } from '../../queue/taskQueue.js';
import { toPublicTask } from '../../queue/task.js';
import { NotFoundError } from '../../utils/errors.js';
import { createPersonalization } from '../../services/personalizeService.js';

/** Полное состояние задачи. */
export const getOne = async (req, res) => {
  const task = await getTask(req.params.id);
  if (!task) throw new NotFoundError('Задача');
  res.json(toPublicTask(task));
};

/** Краткий статус — для частого опроса клиентом. */
export const getStatus = async (req, res) => {
  const task = await getTask(req.params.id);
  if (!task) throw new NotFoundError('Задача');
  res.json({
    id: task.id,
    status: task.status,
    progress: task.progress,
    updatedAt: task.updatedAt,
  });
};

/** Приём персонализации: 202 + task id, работа уходит в фон. */
export const createPersonalize = async (req, res) => {
  const task = await createPersonalization(req.body);
  res.status(202).json({ taskId: task.id, status: task.status });
};
