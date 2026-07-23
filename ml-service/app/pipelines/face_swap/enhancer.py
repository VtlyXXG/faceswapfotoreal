"""
Постобработка результата — согласование лица с иллюстрацией.

Тонкая обёртка над стилизатором из app/pipelines/style: сам алгоритм
выбирается конфигурацией (classical, diffusion, noop).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.pipelines.registry import get_stylizer
from app.pipelines.style.base import StyleOptions

log = get_logger(__name__)


def enhance(
    image: Any,
    faces: list[Any],
    enabled: bool = False,
    strength: float = 1.0,
    identity_embedding: np.ndarray | None = None,
    art_style: str = "",
) -> Any:
    """
    :param image: BGR-изображение после переноса лица
    :param faces: лица на этом изображении, которые нужно согласовать
    :param enabled: выключатель на уровне запроса
    :param strength: общий множитель силы стадий, 1.0 — значения из конфигурации
    :param identity_embedding: ArcFace-эмбеддинг донора (нужен диффузии для FaceID)
    :param art_style: стиль обложки для промпта диффузии
    """
    if not enabled or not faces:
        return image

    stylizer = get_stylizer()
    base: StyleOptions = getattr(stylizer, "default_options", StyleOptions())
    options = base.scaled(strength)
    options.art_style = art_style

    result = image
    for face in faces:
        result = stylizer.stylize(result, face, options, identity_embedding=identity_embedding)
    return result


def is_available() -> bool:
    return get_stylizer().is_available()
