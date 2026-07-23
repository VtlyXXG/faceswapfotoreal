"""
Постобработка результата — согласование лица с иллюстрацией.

Тонкая обёртка над стилизатором из app/pipelines/style: сам алгоритм
выбирается конфигурацией (classical сейчас, диффузионный — следующим шагом).
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipelines.registry import get_stylizer
from app.pipelines.style.base import StyleOptions

log = get_logger(__name__)


def enhance(image: Any, faces: list[Any], enabled: bool = False, strength: float = 1.0) -> Any:
    """
    :param image: BGR-изображение после переноса лица
    :param faces: лица на этом изображении, которые нужно согласовать
    :param enabled: выключатель на уровне запроса
    :param strength: общий множитель силы стадий, 1.0 — значения из конфигурации
    """
    if not enabled or not faces:
        return image

    stylizer = get_stylizer()
    options: StyleOptions = getattr(stylizer, "default_options", StyleOptions()).scaled(strength)

    result = image
    for face in faces:
        result = stylizer.stylize(result, face, options)
    return result


def is_available() -> bool:
    return get_stylizer().is_available()
