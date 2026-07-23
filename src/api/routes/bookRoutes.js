import { Router } from 'express';
import * as controller from '../controllers/bookController.js';
import { asyncHandler } from '../middleware/asyncHandler.js';

export const bookRoutes = Router();

bookRoutes.post('/', asyncHandler(controller.create));
bookRoutes.get('/', asyncHandler(controller.list));
bookRoutes.get('/:id', asyncHandler(controller.getOne));
bookRoutes.get('/:id/status', asyncHandler(controller.getStatus));
bookRoutes.get('/:id/files', asyncHandler(controller.listFiles));
bookRoutes.get('/:id/files/:filename', asyncHandler(controller.downloadFile));

export default bookRoutes;
