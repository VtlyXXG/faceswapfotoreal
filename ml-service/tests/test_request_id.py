"""Проброс X-Request-ID и формат JSON-логов."""

import json
import logging
import uuid

from fastapi.testclient import TestClient

from app.core.context import get_request_id, set_request_id
from app.core.logging import JsonFormatter, RequestIdFilter
from app.main import app

client = TestClient(app)


def test_incoming_request_id_is_echoed():
    incoming = "book-" + uuid.uuid4().hex
    response = client.get("/health", headers={"X-Request-ID": incoming})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == incoming


def test_request_id_is_generated_when_absent():
    response = client.get("/health")

    generated = response.headers.get("X-Request-ID")
    assert generated
    uuid.UUID(generated)  # сгенерирован валидный UUID


def test_malformed_request_id_is_replaced():
    # Перенос строки в заголовке сломал бы построчный разбор логов
    response = client.get("/health", headers={"X-Request-ID": "short"})

    assert response.headers["X-Request-ID"] != "short"
    uuid.UUID(response.headers["X-Request-ID"])


def test_context_is_cleared_between_requests():
    client.get("/health", headers={"X-Request-ID": "leak-" + uuid.uuid4().hex})
    assert get_request_id() is None


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="привет %s", args=("мир",), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_shape():
    token = set_request_id("req-" + uuid.uuid4().hex)
    try:
        record = _record(duration_ms=12.5)
        RequestIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        set_request_id(None) if token is None else None

    assert payload["message"] == "привет мир"
    assert payload["level"] == "info"
    assert payload["service"] == "ml-service"
    assert payload["scope"] == "test"
    assert payload["request_id"].startswith("req-")
    assert payload["duration_ms"] == 12.5
    assert payload["time"].endswith("Z")
