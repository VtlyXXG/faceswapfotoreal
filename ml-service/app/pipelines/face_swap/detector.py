"""Детекция лиц и выбор целевого лица на изображении."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.core.errors import NoFaceDetectedError
from app.pipelines.registry import get_face_analyser


def detect_faces(image: Any) -> list[Any]:
    """
    Возвращает лица, отсортированные слева направо (стабильный порядок между вызовами).

    :param image: BGR-изображение numpy.ndarray
    """
    analyser = get_face_analyser()
    faces = analyser.get(image)
    return sorted(faces, key=lambda f: f.bbox[0])[: settings.max_faces]


def require_faces(image: Any) -> list[Any]:
    faces = detect_faces(image)
    if not faces:
        raise NoFaceDetectedError("На изображении не найдено ни одного лица")
    return faces


def largest_face(image: Any) -> Any:
    """Крупнейшее лицо — обычно главный герой на исходном фото."""
    faces = require_faces(image)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def select_face(faces: list[Any], index: int | None) -> Any:
    """index=None → крупнейшее лицо; иначе лицо по порядковому номеру слева направо."""
    if index is None:
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if index < 0 or index >= len(faces):
        raise NoFaceDetectedError(
            f"Лицо с индексом {index} отсутствует", {"detected": len(faces)}
        )
    return faces[index]


def describe(face: Any) -> dict:
    """Сериализуемое описание лица для ответа API."""
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    return {
        "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
        "det_score": float(getattr(face, "det_score", 0.0)),
        "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
        "age": int(face.age) if getattr(face, "age", None) is not None else None,
    }
