"""
Семантическая разметка человека.

Сама модель здесь не запускается — веса весят 16 МБ и лежат вне репозитория.
Проверяется наша часть: что сервис работает без весов, что классы разбираются в
маски и что отсутствие файла сообщается один раз, а не на каждом заказе.
"""

import numpy as np
import pytest

from app.pipelines import parsing


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(parsing, "_SEGMENTER", None)
    monkeypatch.setattr(parsing, "_MISSING_REPORTED", False)
    yield


def test_missing_weights_do_not_break_the_order(monkeypatch, tmp_path):
    """
    Разметка — уточнение, а не условие работы: без неё вызывающий код падает
    обратно на цветовые эвристики. Отказывать заказчику из-за отсутствия
    необязательных весов нельзя, пайплайн до сих пор обходился без них вовсе.
    """
    monkeypatch.setattr(parsing, "model_path", lambda: tmp_path / "нет.tflite")

    assert parsing.available() is False
    assert parsing.parse(np.zeros((8, 8, 3), dtype=np.uint8)) is None


def test_missing_weights_are_reported_once(monkeypatch, tmp_path):
    """Иначе строка повторится на каждом заказе и утопит остальной лог."""
    monkeypatch.setattr(parsing, "model_path", lambda: tmp_path / "нет.tflite")
    said = []
    monkeypatch.setattr(parsing.log, "warning", lambda *a, **kw: said.append(a))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    parsing.parse(frame)
    parsing.parse(frame)

    assert len(said) == 1


def test_model_path_prefers_the_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(parsing.settings, "parsing_model", str(tmp_path / "своя.tflite"))

    assert parsing.model_path().name == "своя.tflite"


def test_failure_inside_the_model_is_not_fatal(monkeypatch, tmp_path):
    """Битые веса или отказ tflite на кадре — это None, а не 500 заказчику."""
    weights = tmp_path / "есть.tflite"
    weights.write_bytes(b"not a model")
    monkeypatch.setattr(parsing, "model_path", lambda: weights)

    assert parsing.parse(np.zeros((8, 8, 3), dtype=np.uint8)) is None


def test_bare_skin_joins_face_and_body():
    face = np.zeros((4, 4), dtype=np.uint8)
    face[0] = 255
    skin = np.zeros((4, 4), dtype=np.uint8)
    skin[3] = 255

    parsed = parsing.Parsed(face=face, hair=face * 0, skin=skin, clothes=face * 0)

    assert int(np.count_nonzero(parsed.bare_skin)) == 8
