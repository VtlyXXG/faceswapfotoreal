"""Перенос лица источника на целевое изображение."""

from __future__ import annotations

from typing import Any

from app.pipelines.registry import get_face_swapper


def swap_face(target_image: Any, target_face: Any, source_face: Any) -> Any:
    """
    :param target_image: BGR-изображение, куда вставляем лицо
    :param target_face: лицо на target_image, которое заменяем
    :param source_face: лицо-донор (из другого изображения)
    :return: новое BGR-изображение
    """
    swapper = get_face_swapper()
    return swapper.get(target_image, target_face, source_face, paste_back=True)


def swap_all(target_image: Any, target_faces: list[Any], source_face: Any) -> Any:
    """Последовательная замена нескольких лиц на одном изображении."""
    result = target_image
    for face in target_faces:
        result = swap_face(result, face, source_face)
    return result
