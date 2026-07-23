"""
Построение масок лица.

Все маски — float32 в диапазоне [0, 1] и в координатах кропа, а не полного
изображения: постобработка работает с окрестностью лица, а не с 4K-холстом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Группы точек 106-точечной разметки insightface (2d106det)
_BROW_LEFT = list(range(43, 52))
_BROW_RIGHT = list(range(97, 106))
_EYE_LEFT = list(range(33, 43))
_EYE_RIGHT = list(range(87, 97))
_NOSE = list(range(72, 87))
_MOUTH = list(range(52, 72))

_IDENTITY_GROUPS = (_EYE_LEFT, _EYE_RIGHT, _BROW_LEFT, _BROW_RIGHT, _NOSE, _MOUTH)


@dataclass
class FaceMasks:
    skin: np.ndarray  # вся область лица, растушёвана
    identity: np.ndarray  # глаза, брови, нос, рот — их трогаем осторожнее
    ring: np.ndarray  # кольцо иллюстрации вокруг лица — эталон цвета и фактуры
    box: tuple[int, int, int, int]  # кроп в координатах исходного изображения


def expand_bbox(
    bbox: np.ndarray, shape: tuple[int, ...], margin: float
) -> tuple[int, int, int, int]:
    """Расширяет bbox лица в margin раз, не выходя за границы изображения."""
    height, width = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * margin / 2
    half_h = (y2 - y1) * margin / 2

    return (
        max(0, int(cx - half_w)),
        max(0, int(cy - half_h)),
        min(width, int(cx + half_w)),
        min(height, int(cy + half_h)),
    )


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (_odd(radius), _odd(radius)), 0)
    return (blurred.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _landmarks(face: Any) -> np.ndarray | None:
    points = getattr(face, "landmark_2d_106", None)
    if points is None:
        return None
    return np.asarray(points, dtype=np.float32)


def build_masks(face: Any, shape: tuple[int, ...], margin: float = 1.8) -> FaceMasks:
    box = expand_bbox(face.bbox, shape, margin)
    x0, y0, x1, y1 = box
    crop_h, crop_w = y1 - y0, x1 - x0

    skin = np.zeros((crop_h, crop_w), np.uint8)
    identity = np.zeros((crop_h, crop_w), np.uint8)

    points = _landmarks(face)
    if points is not None:
        local = points - np.array([x0, y0], dtype=np.float32)
        cv2.fillConvexPoly(skin, cv2.convexHull(local.astype(np.int32)), 255)

        for group in _IDENTITY_GROUPS:
            group_points = local[group].astype(np.int32)
            if len(group_points) >= 3:
                cv2.fillConvexPoly(identity, cv2.convexHull(group_points), 255)
    else:
        # Без 106-точечной разметки — эллипс по bbox: грубее, но работает
        fx1, fy1, fx2, fy2 = (int(v) for v in face.bbox)
        center = ((fx1 + fx2) // 2 - x0, (fy1 + fy2) // 2 - y0)
        axes = (max(1, (fx2 - fx1) // 2), max(1, (fy2 - fy1) // 2))
        cv2.ellipse(skin, center, axes, 0, 0, 360, 255, -1)

    face_width = max(1.0, float(face.bbox[2] - face.bbox[0]))
    feather_radius = int(face_width * 0.04)

    # Кольцо: полоса иллюстрации вокруг лица шириной ~15% его размера.
    # Из неё берутся эталонные цвет, зерно и резкость.
    outer = cv2.dilate(skin, np.ones((_odd(int(face_width * 0.15)),) * 2, np.uint8))
    ring = cv2.subtract(outer, cv2.dilate(skin, np.ones((_odd(feather_radius),) * 2, np.uint8)))

    return FaceMasks(
        skin=_feather(skin, feather_radius),
        identity=_feather(identity, max(3, feather_radius // 2)),
        ring=(ring.astype(np.float32) / 255.0),
        box=box,
    )
