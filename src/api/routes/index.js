import { Router } from 'express';
import { coverRoutes } from './coverRoutes.js';
import { taskRoutes } from './taskRoutes.js';
import * as health from '../controllers/healthController.js';
import * as tasks from '../controllers/taskController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const apiRoutes = Router();

apiRoutes.get('/health', health.liveness);
apiRoutes.get('/health/ready', asyncHandler(health.readiness));
apiRoutes.get('/stats', asyncHandler(health.stats));

apiRoutes.use('/covers', coverRoutes);
apiRoutes.use('/tasks', taskRoutes);
apiRoutes.post('/personalize', asyncHandler(tasks.createPersonalize));

export default apiRoutes;
