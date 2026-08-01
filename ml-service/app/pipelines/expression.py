"""
Мимика: указание модели, с каким выражением рисовать голову.

Что изменилось. Раньше мимика была геометрией: вырезанное лицо донора нужно
было деформировать до переноса в шаблон, и делать это, не потеряв сходство,
было нечем — отсюда заглушка и отказ на любой эмоции, кроме нейтральной.

Теперь голова рисуется генеративно, и мимика стала текстом. Более того, **по
умолчанию её задавать не нужно**: выражение, поворот и наклон головы берутся с
шаблона — персонаж на развороте уже смеётся, спит или пугается, и задача
модели это выражение сохранить, а не выдумать. Ровно поэтому у нейтрального
значения пустой текст: любое указание мимики здесь спорит с картинкой.

Значение остаётся ручкой на случай, когда выражение персонажа нужно
переопределить — например, серия рисовалась под другой сценарий.

Как добавить эмоцию:

    expression.register(Expression("wink", "the child is winking with one eye"))

После регистрации значение становится допустимым в параметре `emotion`
эндпоинта `/face-swap` — больше ничего править не нужно.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import MLServiceError
from app.core.logging import get_logger

log = get_logger(__name__)

# Значение по умолчанию: выражение лица берётся со сцены.
NEUTRAL = "neutral"


class ExpressionNotSupportedError(MLServiceError):
    """
    Запрошена эмоция, для которой нет описания.

    501, а не 400: запрос корректен, просто возможности пока нет. Разница
    существенная — по 400 вызывающая сторона чинит запрос, по 501 ждёт релиза.
    """

    status_code = 501
    code = "EXPRESSION_NOT_SUPPORTED"


@dataclass(frozen=True)
class Expression:
    """
    Эмоция и её описание для промпта.

    :param prompt: фраза на английском — язык промптов эндпоинта. Пусто
        означает «ничего не указывать», то есть довериться шаблону
    """

    name: str
    prompt: str = ""


_REGISTRY: dict[str, Expression] = {}


def register(expression: Expression) -> None:
    """Добавляет эмоцию в реестр. Повторная регистрация имени — замена."""
    _REGISTRY[expression.name] = expression


def available() -> list[str]:
    return sorted(_REGISTRY)


def prompt(emotion: str = "") -> str:
    """
    Текст мимики для промпта.

    :param emotion: имя эмоции; пусто — то же, что `neutral`
    :raises ExpressionNotSupportedError: если эмоции с таким именем нет
    """
    # Сначала обрезка, потом умолчание: Node подставляет в форму пустую строку,
    # а из неё после strip() получается не «neutral», а несуществующее имя.
    name = (emotion or "").strip().lower() or NEUTRAL

    expression = _REGISTRY.get(name)
    if expression is None:
        raise ExpressionNotSupportedError(
            f"Мимика «{name}» пока не поддерживается",
            {"emotion": name, "available": available()},
        )

    if expression.prompt:
        log.info("мимика задана явно и перекроет выражение персонажа", extra={"emotion": name})
    return expression.prompt


register(Expression(NEUTRAL))
register(
    Expression(
        "smile",
        "override the expression: the child is smiling warmly, lips together, eyes bright",
    )
)
register(
    Expression(
        "laugh",
        "override the expression: the child is laughing, mouth open, cheeks raised",
    )
)
register(
    Expression(
        "surprise",
        "override the expression: the child looks surprised, eyes wide, eyebrows raised",
    )
)
register(
    Expression(
        "calm",
        "override the expression: the child looks calm and thoughtful, mouth relaxed",
    )
)
