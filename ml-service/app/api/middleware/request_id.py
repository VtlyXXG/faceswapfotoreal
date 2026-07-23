"""Проброс X-Request-ID: принимаем от Node.js API или генерируем свой."""

from __future__ import annotations

import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import reset_request_id, set_request_id
from app.core.logging import get_logger

REQUEST_ID_HEADER = "x-request-id"

# Тот же формат, что валидирует Node: id попадает в логи и в заголовок ответа
_VALID = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")

log = get_logger("http")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _VALID.match(incoming) else str(uuid.uuid4())

        token = set_request_id(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "%s %s → необработанная ошибка",
                request.method,
                request.url.path,
                extra={"method": request.method, "path": request.url.path},
            )
            raise
        finally:
            reset_request_id(token)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        log.info(
            "%s %s → %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                # request_id вне contextvar: он уже сброшен в finally
                "request_id": request_id,
            },
        )

        return response
