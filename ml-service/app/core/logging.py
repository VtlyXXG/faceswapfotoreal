"""
Структурированное логирование.

Формат JSON-записи совпадает с Node-сервисом (pino):
    {"time","level","service","scope","message","request_id",...}
— поэтому логи обоих сервисов читаются и грепаются одинаково.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from app.config import settings
from app.core.context import get_request_id
from app.core.rotation import TimedSizedRotatingFileHandler

# Атрибуты LogRecord, которые не нужно тащить в JSON как extra
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}

_PLAIN_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


class RequestIdFilter(logging.Filter):
    """
    Подмешивает request_id из contextvar.

    Явно переданный extra={"request_id": ...} не затирается: middleware логирует
    итог запроса уже после reset() contextvar, и его id должен пережить фильтр.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "service": "ml-service",
            "scope": record.name,
            "message": record.getMessage(),
            "pid": record.process,
        }

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        # Всё, что передали через logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            if key == "request_id":  # уже обработан выше; None не пишем
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def _console_handler() -> logging.Handler:
    # Консоль Windows по умолчанию cp1251: без UTF-8 любая стрелка или кириллица
    # в сообщении роняет вывод в UnicodeEncodeError внутри logging
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # поток не TextIOWrapper (тесты, пайпы)
        pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.log_json else logging.Formatter(_PLAIN_FORMAT, "%H:%M:%S")
    )
    return handler


def _file_handler(filename: str, level: int) -> logging.Handler:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = TimedSizedRotatingFileHandler(
        settings.log_dir / filename,
        when=settings.log_rotate_when,
        max_bytes=settings.log_file_max_mb * 1024 * 1024,
        backup_count=settings.log_backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())  # в файл — всегда JSON
    return handler


def setup_logging() -> None:
    level = settings.log_level.upper()
    request_id_filter = RequestIdFilter()

    handlers = [
        _console_handler(),
        _file_handler(settings.log_file, logging.NOTSET),
        _file_handler(settings.log_error_file, logging.ERROR),
    ]
    for handler in handlers:
        handler.addFilter(request_id_filter)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)

    # uvicorn ставит свои хендлеры — снимаем, чтобы его записи шли
    # через наш JSON-формат, а не двумя разными форматами в один поток
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    # Access-лог пишет RequestIdMiddleware — со статусом, длительностью и
    # request_id. Штатный лог uvicorn дублировал бы его без этих полей.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
