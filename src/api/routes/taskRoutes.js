import { Router } from 'express';
import * as controller from '../controllers/taskController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const taskRoutes = Router();

taskRoutes.get('/:id', asyncHandler(controller.getOne));
taskRoutes.get('/:id/status', asyncHandler(controller.getStatus));

export default taskRoutes;
