import { Router, raw } from 'express';
import { coverRoutes } from './coverRoutes.js';
import { taskRoutes } from './taskRoutes.js';
import * as health from '../controllers/healthController.js';
import * as tasks from '../controllers/taskController.js';
import * as uploads from '../controllers/uploadController.js';
import { config } from '../../config/index.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const apiRoutes = Router();

apiRoutes.get('/health', health.liveness);
apiRoutes.get('/health/ready', asyncHandler(health.readiness));
apiRoutes.get('/stats', asyncHandler(health.stats));

// Лимит общий с ML-сервисом (ML_MAX_UPLOAD_MB): 4K-обложка весит 25-30 МБ
apiRoutes.post(
  '/uploads',
  raw({ type: ['image/png', 'image/jpeg', 'image/webp'], limit: config.uploads.limit }),
  asyncHandler(uploads.create),
);

apiRoutes.use('/covers', coverRoutes);
apiRoutes.use('/tasks', taskRoutes);
apiRoutes.post('/personalize', asyncHandler(tasks.createPersonalize));

export default apiRoutes;
