"""
Мимика: реестр описаний и поведение на незарегистрированной эмоции.

Смена смысла. Раньше мимика была геометрией — деформацией вырезанного лица, — и
реализации не имела вовсе. Теперь голова рисуется генеративно, и мимика стала
текстом промпта. Умолчание при этом обратное прежнему: НИЧЕГО не указывать,
потому что выражение и поворот головы берутся с шаблона.
"""

import pytest

from app.pipelines import expression


@pytest.fixture
def registry():
    """Реестр глобальный — после теста его надо вернуть как было."""
    saved = dict(expression._REGISTRY)
    yield
    expression._REGISTRY.clear()
    expression._REGISTRY.update(saved)


def test_neutral_says_nothing_about_the_expression():
    """
    Пустой текст — не пробел в реализации, а решение: персонаж на развороте уже
    смеётся или спит, и задача модели это выражение сохранить, а не выдумать.
    """
    assert expression.prompt(expression.NEUTRAL) == ""


@pytest.mark.parametrize("emotion", ["", "  ", "NEUTRAL", "Neutral "])
def test_empty_emotion_means_neutral(emotion):
    """Пустое значение — обычный случай: Node не передаёт параметр вовсе."""
    assert expression.prompt(emotion) == ""


def test_known_emotion_overrides_the_scene():
    """
    Явная эмоция спорит с картинкой намеренно — значит, и сказано это должно
    быть прямо, иначе модель послушает сцену.
    """
    text = expression.prompt("laugh").lower()

    assert "override the expression" in text
    assert "laughing" in text


def test_unknown_emotion_is_refused_as_not_implemented():
    """
    501, а не 400: запрос корректен, возможности пока нет. Молча вернуть
    выражение со сцены нельзя — на развороте это обнаружится уже в тираже.
    """
    with pytest.raises(expression.ExpressionNotSupportedError) as exc_info:
        expression.prompt("телепатия")

    error = exc_info.value
    assert error.status_code == 501
    assert error.code == "EXPRESSION_NOT_SUPPORTED"
    assert expression.NEUTRAL in error.details["available"]


def test_registered_emotion_becomes_available(registry):
    """Добавление мимики — регистрация объекта, больше правок не нужно."""
    expression.register(expression.Expression("wink", "the child is winking"))

    assert "wink" in expression.available()
    assert expression.prompt("wink") == "the child is winking"
