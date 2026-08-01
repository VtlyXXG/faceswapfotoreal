"""
Общая заглушка сетки лица.

mediapipe в тестах не запускается: он медленный, требует настоящего лица в
кадре и проверяет совсем не то, что нам нужно. Модулю маски достаточно списка
из 468 точек, и заглушка отдаёт именно его.
"""

import numpy as np
import pytest

from app.pipelines import head_mask

# Канонические смещения относительно центра лица, до масштаба и поворота.
# Высота лица (подбородок → переносица) равна 80, ширина (глаз → глаз, ×2) — 120.
_OFFSETS = {
    152: (0, 60),  # подбородок
    9: (0, -20),  # переносица, она же верх «оси лица»
    33: (-30, -10), 133: (-12, -10),  # правый глаз: внешний и внутренний углы
    362: (12, -10), 263: (30, -10),  # левый глаз
    234: (-50, 0), 454: (50, 0),  # скулы на уровне ушей
    168: (0, -14), 6: (0, -6), 195: (0, 2), 4: (0, 12), 1: (0, 18),  # спинка носа
    98: (-10, 22), 327: (10, 22),  # крылья носа
    61: (-18, 38), 291: (18, 38),  # углы рта
    70: (-34, -24), 300: (34, -24),  # внешние края бровей
}


def face_mesh(centre=(200, 200), scale=1.0, angle=0.0) -> list[tuple[int, int]]:
    """
    468 точек-заглушек: лицо в заданном месте, размере и наклоне.

    :param centre: куда поставить центр лица
    :param scale: множитель размера (1.0 — высота лица 80 пикселей)
    :param angle: наклон головы в градусах
    """
    radians = np.radians(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )

    def place(dx: float, dy: float) -> tuple[int, int]:
        x, y = rotation @ (np.array([dx, dy], dtype=np.float64) * scale)
        return int(round(centre[0] + x)), int(round(centre[1] + y))

    points = [place(0, 0)] * 468
    for index, (dx, dy) in _OFFSETS.items():
        points[index] = place(dx, dy)

    # Дуга челюсти — полуокружность от скулы через подбородок ко второй скуле
    for offset, index in enumerate(head_mask._JAW_ARC):
        step = np.pi * offset / (len(head_mask._JAW_ARC) - 1)
        points[index] = place(-50 * np.cos(step), 60 * np.sin(step))

    # Линия бровей
    for offset, index in enumerate(head_mask._BROW_ARC):
        share = offset / (len(head_mask._BROW_ARC) - 1)
        points[index] = place(-34 + 68 * share, -22)

    return points


@pytest.fixture
def mesh():
    """Фабрика сеток — тесты сами решают, где и какого размера лицо."""
    return face_mesh
