"""Доменные ошибки ML-сервиса и их отображение в HTTP-ответы."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class MLServiceError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NoFaceDetectedError(MLServiceError):
    """На изображении не найдено ни одного лица."""

    status_code = 422
    code = "NO_FACE_DETECTED"


class InvalidImageError(MLServiceError):
    status_code = 400
    code = "INVALID_IMAGE"


class PayloadTooLargeError(MLServiceError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MLServiceError)
    async def _handle(_: Request, exc: MLServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )
