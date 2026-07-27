"""Сетка лица, профиль краёв и маска стыка для вклеенной аппликации."""

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import mask_generator
from tests.conftest import face_mesh

_SHAPE = (400, 400)
_FACE_HEIGHT = 80.0


def _head_alpha(radius: int = 130) -> np.ndarray:
    """Силуэт вклеенной головы: круг вокруг лица заглушки."""
    alpha = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.circle(alpha, (200, 190), radius, 255, -1)
    return alpha


def _blend(**overrides) -> np.ndarray:
    kwargs = {
        "head_alpha": _head_alpha(),
        "face_polygon": mask_generator.face_polygon(face_mesh()),
        "neck_line": ((120, 296), (280, 296)),
        "erased": np.zeros(_SHAPE, dtype=np.uint8),
        "face_height": _FACE_HEIGHT,
    }
    kwargs.update(overrides)
    return mask_generator.blend_mask(_SHAPE, **kwargs)


# --- Геометрия полигона ---


def test_polygon_uses_jaw_and_brows_only():
    points = face_mesh()
    polygon = mask_generator.face_polygon(points)

    assert len(polygon) == len(mask_generator._JAW_ARC) + len(mask_generator._BROW_ARC)
    # Лоб и волосы в контур не входят: верхняя точка — приподнятая бровь
    assert polygon[:, 1].min() < points[9][1]


def test_brows_are_lifted_to_include_eyebrow():
    polygon = mask_generator.face_polygon(face_mesh())
    brow_points = polygon[len(mask_generator._JAW_ARC) :]

    # Точки сетки лежат по нижнему краю брови (y = 178), подъём — вверх от неё
    assert (brow_points[:, 1] < 178).all()


def test_face_landmarks_reject_a_frame_without_face():
    blank = np.zeros((240, 240, 3), dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError):
        mask_generator.face_landmarks(blank)


# --- Профиль краёв ---


def test_soften_keeps_the_filled_area_opaque():
    """
    Растушёвка живёт снаружи залитой области и не съедает её. На этом профиле
    держится край аппликации: подъеденный полутонами контур волос дал бы по
    периметру полупрозрачную кайму.
    """
    filled = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.circle(filled, (200, 200), 100, 255, -1)

    softened = mask_generator.soften(filled, _FACE_HEIGHT, 0.05, 0.05)

    assert softened[filled == 255].min() == 255
    assert np.count_nonzero(softened) > np.count_nonzero(filled)
    assert softened.min() == 0


@pytest.mark.parametrize("padding,feather", [(-0.1, 0.1), (0.1, -0.1)])
def test_soften_rejects_negative_ratios(padding, feather):
    with pytest.raises(InvalidImageError):
        mask_generator.soften(np.zeros(_SHAPE, np.uint8), _FACE_HEIGHT, padding, feather)


# --- Маска стыка: что открыто модели ---


def test_face_is_completely_closed():
    """
    Главная гарантия нового пайплайна: до черт лица модель не дотягивается
    вовсе. Не «почти не трогает» — ноль.
    """
    mask = _blend()

    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(face_mesh())], 255)
    core = cv2.erode(face, np.ones((9, 9), np.uint8)) > 0

    assert mask[core].max() == 0


def test_hair_contour_is_open():
    """Внешний контур волос — та самая зона, ради которой маска и строится."""
    mask = _blend()

    alpha = _head_alpha()
    kernel = np.ones((5, 5), np.uint8)
    ring = cv2.dilate(alpha, kernel) - cv2.erode(alpha, kernel)
    # Кольцо берётся сверху, подальше от лица и от шеи
    top = ring.copy()
    top[120:, :] = 0

    assert mask[top > 0].max() == 255


def test_neck_seam_is_covered_wider_than_the_hair_edge():
    """
    Оторванную шею не сглаживают, а закрывают: модель должна дорисовать там
    воротник или тень, и полосы в ширину контура волос на это не хватит.
    """
    mask = _blend()

    column = mask[:, 200]
    seam = np.nonzero(column[250:] > 127)[0]
    assert len(seam) > _FACE_HEIGHT * 0.3


def test_erased_area_is_included():
    """Стёртая причёска персонажа затянута локально — мазок кладёт модель."""
    erased = np.zeros(_SHAPE, dtype=np.uint8)
    erased[40:70, 40:70] = 255

    mask = _blend(erased=erased)

    assert mask[50:60, 50:60].max() == 255


def test_untouched_background_stays_black():
    mask = _blend()

    assert mask[10:30, 350:390].max() == 0


def test_guard_reopens_towards_the_jaw():
    """
    У самой кромки челюсти маска гасится лишь частично: иначе фотографическая
    щека встречалась бы с живописью встык, без единого пикселя на переход.
    """
    polygon = mask_generator.face_polygon(face_mesh())
    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [polygon], 255)

    guard = mask_generator.face_guard(_SHAPE, polygon, _FACE_HEIGHT)

    core = cv2.erode(face, np.ones((15, 15), np.uint8)) > 0
    assert guard[core].min() == 255, "черты защищены полностью"
    assert np.count_nonzero((guard > 10) & (guard < 245)) > 0, "у кромки — полутона"


def test_zero_ratios_leave_only_the_raw_seam():
    """Вырожденный случай: без растушёвок остаются ровно кольцо и полоса шеи."""
    mask = _blend(edge_ratio=0.0, neck_ratio=0.0, guard_ratio=0.0, feather_ratio=0.0)

    assert mask.max() == 255
    assert np.count_nonzero((mask > 10) & (mask < 245)) == 0


@pytest.mark.parametrize(
    "ratios",
    [{"edge_ratio": -0.1}, {"neck_ratio": -0.1}, {"feather_ratio": -0.1}, {"guard_ratio": -0.1}],
)
def test_negative_ratios_are_rejected(ratios):
    with pytest.raises(InvalidImageError):
        _blend(**ratios)
