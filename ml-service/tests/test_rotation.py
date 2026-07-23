"""Ротация логов: по размеру, по времени и удаление старых файлов."""

import logging
import re

import pytest

from app.core.rotation import TimedSizedRotatingFileHandler

NAME_PATTERN = re.compile(r"^app\.\d{4}-\d{2}-\d{2}\.\d+\.log$")


def _handler(tmp_path, **kwargs) -> TimedSizedRotatingFileHandler:
    handler = TimedSizedRotatingFileHandler(tmp_path / "app.log", **kwargs)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _emit(handler, message: str) -> None:
    handler.emit(
        logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
    )


@pytest.fixture
def logs_dir(tmp_path):
    return tmp_path


def test_size_rollover_creates_dated_files(logs_dir):
    handler = _handler(logs_dir, max_bytes=200, backup_count=10)
    try:
        for i in range(40):
            _emit(handler, f"строка номер {i} " + "x" * 50)
    finally:
        handler.close()

    rotated = sorted(p.name for p in logs_dir.iterdir() if p.name != "app.log")

    assert rotated, "ротации по размеру не произошло"
    assert all(NAME_PATTERN.match(name) for name in rotated), rotated
    # активный файл всегда app.log — так работает stdlib
    assert (logs_dir / "app.log").exists()


def test_backup_count_is_enforced(logs_dir):
    handler = _handler(logs_dir, max_bytes=200, backup_count=3)
    try:
        for i in range(60):
            _emit(handler, f"строка номер {i} " + "x" * 50)
    finally:
        handler.close()

    rotated = [p for p in logs_dir.iterdir() if p.name != "app.log"]
    assert len(rotated) == 3, [p.name for p in rotated]


def test_index_increments_within_same_day(logs_dir):
    handler = _handler(logs_dir, max_bytes=200, backup_count=10)
    try:
        for i in range(40):
            _emit(handler, f"строка номер {i} " + "x" * 50)
    finally:
        handler.close()

    indexes = sorted(
        int(p.name.split(".")[-2]) for p in logs_dir.iterdir() if p.name != "app.log"
    )
    # Нумерация без пропусков: удаляются самые старые, а не произвольные файлы
    assert indexes == list(range(indexes[0], indexes[0] + len(indexes)))
    assert indexes[-1] >= len(indexes)


def test_time_rollover_still_works(logs_dir):
    """when='S' — ротация по времени без превышения размера."""
    import time

    handler = _handler(logs_dir, when="S", max_bytes=0, backup_count=5)
    try:
        _emit(handler, "до ротации")
        time.sleep(1.1)
        _emit(handler, "после ротации")
    finally:
        handler.close()

    rotated = [p for p in logs_dir.iterdir() if p.name != "app.log"]
    assert len(rotated) == 1, [p.name for p in rotated]
    assert "до ротации" in rotated[0].read_text(encoding="utf-8")
