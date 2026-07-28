"""
Консольный формат: поля extra обязаны доезжать до терминала.

Тест не про красоту вывода. Весь разбор полётов в этом сервисе построен на
числах, которые логируются через `extra={...}`: масштаб вклейки, доли масок,
площадь стёртого. Пока консольный формат печатал одно лишь сообщение, эти числа
существовали только в JSON-файле внутри контейнера — в `docker logs` их не было
вовсе, и на вопрос «какой вышел масштаб» ответить было нечем.
"""

import logging

from app.core.logging import _PLAIN_FORMAT, JsonFormatter, PlainFormatter


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.pipelines.collage",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="аппликация собрана",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_plain_format_shows_extra_fields():
    record = _record(scale=0.744, neck_source="frame")
    line = PlainFormatter(_PLAIN_FORMAT, "%H:%M:%S").format(record)

    assert "аппликация собрана" in line
    assert "scale=0.744" in line
    assert "neck_source=frame" in line


def test_plain_format_keeps_a_bare_message_bare():
    line = PlainFormatter(_PLAIN_FORMAT, "%H:%M:%S").format(_record())

    assert line.rstrip().endswith("аппликация собрана")


def test_plain_format_skips_empty_request_id():
    """
    request_id подмешивается фильтром всегда и вне запроса пуст. Печатать
    `request_id=None` в каждой строке — значит утопить в шуме то, ради чего
    формат и правился.
    """
    line = PlainFormatter(_PLAIN_FORMAT, "%H:%M:%S").format(_record(request_id=None, scale=1.0))

    assert "request_id" not in line
    assert "scale=1.0" in line


def test_json_format_still_carries_the_same_fields():
    import json

    payload = json.loads(JsonFormatter().format(_record(scale=0.744)))

    assert payload["message"] == "аппликация собрана"
    assert payload["scale"] == 0.744
    assert payload["scope"] == "app.pipelines.collage"
