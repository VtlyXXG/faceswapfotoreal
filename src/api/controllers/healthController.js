import { queueStats } from '../../queue/taskQueue.js';
import { checkMlService } from '../../services/mlClient.js';

export const liveness = (_req, res) => res.json({ status: 'ok', uptime: process.uptime() });

export const readiness = async (_req, res) => {
  const ml = await checkMlService();

  // ML-сервис критичен: вся работа сервиса — замена лица на обложке,
  // без него принимать заказы бессмысленно
  res.status(ml.available ? 200 : 503).json({
    status: ml.available ? 'ready' : 'degraded',
    ml_service: ml,
  });
};

export const stats = async (_req, res) => res.json({ queue: await queueStats() });
