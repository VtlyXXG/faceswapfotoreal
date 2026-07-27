"""
Мимика: реестр трансформеров и поведение на незарегистрированной эмоции.

Реализации пока нет — есть точка расширения. Тесты фиксируют её контракт:
нейтральное выражение проходит насквозь, чужое имя получает честный отказ, а
зарегистрированный трансформер действительно вызывается.
"""

import numpy as np
import pytest

from app.pipelines import expression
from tests.conftest import face_mesh


@pytest.fixture
def registry():
    """Реестр глобальный — после теста его надо вернуть как было."""
    saved = dict(expression._REGISTRY)
    yield
    expression._REGISTRY.clear()
    expression._REGISTRY.update(saved)


def _face() -> expression.Face:
    return expression.Face(
        image=np.full((40, 40, 3), 7, dtype=np.uint8),
        alpha=np.full((40, 40), 255, dtype=np.uint8),
        points=face_mesh(),
    )


def test_neutral_passes_the_face_through_unchanged():
    face = _face()

    result = expression.transform(face, expression.NEUTRAL)

    assert np.array_equal(result.image, face.image)
    assert np.array_equal(result.alpha, face.alpha)
    assert result.points == face.points


@pytest.mark.parametrize("emotion", ["", "  ", "NEUTRAL", "Neutral "])
def test_empty_emotion_means_neutral(emotion):
    """Пустое значение — обычный случай: Node не передаёт параметр вовсе."""
    assert expression.transform(_face(), emotion) is not None


def test_unknown_emotion_is_refused_as_not_implemented():
    """
    501, а не 400: запрос корректен, возможности пока нет. Молча вернуть
    неподвижное лицо нельзя — на развороте это обнаружится уже в тираже.
    """
    with pytest.raises(expression.ExpressionNotSupportedError) as exc_info:
        expression.transform(_face(), "smile")

    error = exc_info.value
    assert error.status_code == 501
    assert error.code == "EXPRESSION_NOT_SUPPORTED"
    assert expression.NEUTRAL in error.details["available"]


def test_registered_transformer_becomes_available_and_runs(registry):
    """Добавление мимики — регистрация объекта, больше правок не нужно."""

    class Grin:
        name = "grin"

        def apply(self, face):
            return expression.Face(
                image=face.image + 1, alpha=face.alpha, points=face.points
            )

    expression.register(Grin())

    assert "grin" in expression.available()
    assert expression.transform(_face(), "grin").image[0, 0, 0] == 8
