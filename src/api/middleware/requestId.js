import { randomUUID } from 'node:crypto';
import { runWithContext } from '../../utils/requestContext.js';

export const REQUEST_ID_HEADER = 'x-request-id';

// Входящий id принимаем только в безопасном виде: он попадёт в логи и в
// заголовок ответа, поэтому переносы строк и произвольная длина недопустимы.
const VALID = /^[A-Za-z0-9_.:-]{8,128}$/;

/**
 * Берёт X-Request-ID от вызывающей стороны или генерирует новый,
 * кладёт в контекст запроса и возвращает клиенту тем же заголовком.
 */
export const requestId = (req, res, next) => {
  const incoming = req.get(REQUEST_ID_HEADER);
  const id = incoming && VALID.test(incoming) ? incoming : randomUUID();

  req.id = id;
  res.setHeader('X-Request-ID', id);

  runWithContext({ requestId: id }, () => next());
};

export default requestId;
