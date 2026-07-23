import { Router } from 'express';
import { bookRoutes } from './bookRoutes.js';
import * as health from '../controllers/healthController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const apiRoutes = Router();

apiRoutes.get('/health', health.liveness);
apiRoutes.get('/health/ready', asyncHandler(health.readiness));
apiRoutes.get('/stats', health.stats);
apiRoutes.use('/books', bookRoutes);

export default apiRoutes;
