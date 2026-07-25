import { Router } from 'express';
import * as controller from '../controllers/coverController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const coverRoutes = Router();

coverRoutes.post('/', asyncHandler(controller.create));
coverRoutes.get('/', asyncHandler(controller.list));
coverRoutes.get('/:id', asyncHandler(controller.getOne));
coverRoutes.get('/:id/status', asyncHandler(controller.getStatus));
coverRoutes.get('/:id/files', asyncHandler(controller.listFiles));
coverRoutes.get('/:id/files/:filename', asyncHandler(controller.downloadFile));

export default coverRoutes;
