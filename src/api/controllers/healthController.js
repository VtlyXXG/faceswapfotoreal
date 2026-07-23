import { checkProviders } from '../../ai/registry.js';
import { jobQueue } from '../../queue/jobQueue.js';
import { checkMlService } from '../../services/mlClient.js';

export const liveness = (_req, res) => res.json({ status: 'ok', uptime: process.uptime() });

export const readiness = async (_req, res) => {
  const [models, ml] = await Promise.all([checkProviders(), checkMlService()]);

  // ML-сервис не критичен: без него книга генерируется без face-swap
  const ready = models.text.available;
  res.status(ready ? 200 : 503).json({
    status: ready ? 'ready' : 'degraded',
    models,
    ml_service: ml,
  });
};

export const stats = (_req, res) => res.json({ queue: jobQueue.stats() });
