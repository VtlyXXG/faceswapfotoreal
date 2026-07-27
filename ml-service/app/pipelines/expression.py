"""
Мимика донора: точка расширения пайплайна перед вклейкой.

Зачем это здесь. Обложка — не одна картинка, а разворот за разворотом, и на
каждом из них сейчас оказывается одно и то же лицо с одной и той же
фотографии. Лицо получается статичным: герой радуется, пугается и засыпает с
неизменным выражением. Значит, в пайплайне должно быть место, где выражение
меняется — по параметру, приходящему вместе с заказом.

Место это ровно одно: сразу после вырезки головы и до переноса в шаблон.
Раньше — нельзя, ещё нет вырезанного лица; позже — лицо уже вклеено, и любая
правка мимики поедет вместе с фоном обложки.

Здесь заложен интерфейс и заглушка, а не реализация. Причина не в экономии:
любая переделка мимики — это изменение геометрии лица, то есть ровно то, от
чего мы уходили, вводя жёсткий коллаж. Сначала нужен инструмент, сохраняющий
личность (кандидаты: LivePortrait на fal, локальный варп по сетке mediapipe с
ограничением смещений), и способ проверить, что заказчик остался узнаваем.
До тех пор честнее отказать на неизвестной эмоции, чем молча вернуть
неподвижное лицо: молчаливый отказ обнаружится уже на печати тиража.

Как добавить трансформер:

    class Smile:
        name = "smile"
        def apply(self, face: Face) -> Face: ...

    expression.register(Smile())

После регистрации значение становится допустимым в параметре `emotion`
эндпоинта `/face-swap` — больше ничего править не нужно.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from app.core.errors import MLServiceError
from app.core.logging import get_logger

log = get_logger(__name__)

# Значение по умолчанию: лицо переносится как снято.
NEUTRAL = "neutral"


class ExpressionNotSupportedError(MLServiceError):
    """
    Запрошена эмоция, для которой нет трансформера.

    501, а не 400: запрос корректен, просто возможности пока нет. Разница
    существенная — по 400 вызывающая сторона чинит запрос, по 501 ждёт релиза.
    """

    status_code = 501
    code = "EXPRESSION_NOT_SUPPORTED"


@dataclass
class Face:
    """
    Вырезанная голова донора — вход и выход любого трансформера.

    Идёт тройкой, потому что менять мимику, не трогая остальное, невозможно:
    вслед за пикселями двигается и силуэт (открывшийся рот, поднявшаяся бровь),
    и сетка, по которой лицо потом совмещается с шаблоном.
    """

    image: Any  # BGR numpy.ndarray — кадр фотографии целиком
    alpha: Any  # одноканальная маска головы того же размера
    points: list[tuple[int, int]]  # сетка mediapipe в координатах кадра


class ExpressionTransformer(Protocol):
    """Контракт трансформера мимики."""

    name: str

    def apply(self, face: Face) -> Face:
        """Возвращает лицо с изменённым выражением, сохраняя личность."""
        ...


class KeepAsIs:
    """Нейтральное выражение — лицо переносится как снято на фотографии."""

    name = NEUTRAL

    def apply(self, face: Face) -> Face:
        return replace(face)


_REGISTRY: dict[str, ExpressionTransformer] = {}


def register(transformer: ExpressionTransformer) -> None:
    """Добавляет трансформер в реестр. Повторная регистрация имени — замена."""
    _REGISTRY[transformer.name] = transformer


def available() -> list[str]:
    return sorted(_REGISTRY)


def transform(face: Face, emotion: str = "") -> Face:
    """
    Применяет трансформер мимики к вырезанной голове.

    :param emotion: имя эмоции; пусто — то же, что `neutral`
    :raises ExpressionNotSupportedError: если трансформера с таким именем нет
    """
    # Сначала обрезка, потом умолчание: Node подставляет в форму пустую строку,
    # а из неё после strip() получается не «neutral», а несуществующее имя.
    name = (emotion or "").strip().lower() or NEUTRAL

    transformer = _REGISTRY.get(name)
    if transformer is None:
        raise ExpressionNotSupportedError(
            f"Мимика «{name}» пока не поддерживается",
            {"emotion": name, "available": available()},
        )

    if name != NEUTRAL:
        log.info("применяется трансформер мимики", extra={"emotion": name})
    return transformer.apply(face)


register(KeepAsIs())
