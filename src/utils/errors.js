/** Базовая ошибка приложения с HTTP-статусом. */
export class AppError extends Error {
  constructor(message, { status = 500, code = 'INTERNAL_ERROR', details } = {}) {
    super(message);
    this.name = new.target.name;
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export class ValidationError extends AppError {
  constructor(message, details) {
    super(message, { status: 422, code: 'VALIDATION_ERROR', details });
  }
}

export class NotFoundError extends AppError {
  constructor(resource = 'Ресурс') {
    super(`${resource} не найден`, { status: 404, code: 'NOT_FOUND' });
  }
}

/** Ошибка на стороне локальной ИИ-модели (недоступна, таймаут, некорректный ответ). */
export class AIProviderError extends AppError {
  constructor(message, details) {
    super(message, { status: 502, code: 'AI_PROVIDER_ERROR', details });
  }
}
