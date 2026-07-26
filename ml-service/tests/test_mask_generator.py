"""Маска лица: геометрия полигона и обработка кадра без лица."""

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import mask_generator


def _fake_landmarks() -> list[tuple[int, int]]:
    """
    468 точек-заглушек: лицо-эллипс в центре кадра 400x400.

    Настоящую сетку даёт mediapipe, но геометрию полигона можно проверить и
    без него — важно, что берутся нужные индексы и брови поднимаются вверх.
    """
    points = [(200, 200)] * 468
    for offset, index in enumerate(mask_generator._JAW_ARC):
        # Дуга челюсти: нижняя половина кадра
        points[index] = (100 + offset * 10, 300)
    for offset, index in enumerate(mask_generator._BROW_ARC):
        # Линия бровей: верхняя часть
        points[index] = (120 + offset * 15, 150)
    return points


def test_polygon_uses_jaw_and_brows_only():
    polygon = mask_generator.face_polygon(_fake_landmarks())

    assert len(polygon) == len(mask_generator._JAW_ARC) + len(mask_generator._BROW_ARC)
    # Лоб и волосы в контур не входят: верхняя точка — приподнятая бровь,
    # а не макушка (индекс 10 канонического овала остался на месте заглушки)
    assert polygon[:, 1].min() < 150


def test_brows_are_lifted_to_include_eyebrow():
    polygon = mask_generator.face_polygon(_fake_landmarks())
    brow_points = polygon[len(mask_generator._JAW_ARC) :]

    # Подъём пропорционален высоте лица (300 - 150 = 150 по дуге челюсти)
    assert (brow_points[:, 1] < 150).all()


def test_generate_mask_rejects_frame_without_face():
    blank = np.zeros((240, 240, 3), dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError):
        mask_generator.generate_mask(blank)


def test_generate_mask_rejects_bad_kernel():
    blank = np.zeros((240, 240, 3), dtype=np.uint8)

    with pytest.raises(InvalidImageError):
        mask_generator.generate_mask(blank, blur_kernel=0)


def test_mask_is_single_channel_and_blurred(monkeypatch):
    """Маска: один канал, размер кадра, мягкие края вместо ступеньки."""
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: _fake_landmarks())

    image = np.zeros((400, 400, 3), dtype=np.uint8)
    mask = mask_generator.generate_mask(image, blur_kernel=51)

    assert mask.shape == (400, 400)
    assert mask.dtype == np.uint8
    assert mask.max() > 0, "полигон должен быть залит белым"

    # Размытие даёт полутона — именно они обеспечивают бесшовный переход
    intermediate = np.count_nonzero((mask > 10) & (mask < 245))
    assert intermediate > 0

    # Чёткая копия того же полигона полутонов не имеет — значит размытие сработало
    sharp = np.zeros((400, 400), dtype=np.uint8)
    cv2.fillPoly(sharp, [mask_generator.face_polygon(_fake_landmarks())], 255)
    assert np.count_nonzero((sharp > 10) & (sharp < 245)) == 0
