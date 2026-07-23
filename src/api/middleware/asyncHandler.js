/** Оборачивает async-обработчик, чтобы отклонённые промисы попадали в errorHandler. */
export const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

export default asyncHandler;
