import { Router } from 'express';
import { bookRoutes } from './bookRoutes.js';
import { taskRoutes } from './taskRoutes.js';
import * as health from '../controllers/healthController.js';
import * as tasks from '../controllers/taskController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const apiRoutes = Router();

apiRoutes.get('/health', health.liveness);
apiRoutes.get('/health/ready', asyncHandler(health.readiness));
apiRoutes.get('/stats', asyncHandler(health.stats));

apiRoutes.use('/books', bookRoutes);
apiRoutes.use('/tasks', taskRoutes);
apiRoutes.post('/personalize', asyncHandler(tasks.createPersonalize));

export default apiRoutes;
