"""Маска лица: геометрия полигона, профиль краёв и кадр без лица."""

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


@pytest.fixture
def faked(monkeypatch):
    """Кадр 400x400 и заглушка детектора: тесты про маску, а не про mediapipe."""
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: _fake_landmarks())
    return np.zeros((400, 400, 3), dtype=np.uint8)


def _mask(image, **ratios):
    """Маска по одному кадру — то, что пайплайн строит для лица шаблона."""
    polygon = mask_generator.face_polygon(mask_generator.face_landmarks(image))
    return mask_generator.mask_from_polygons(image.shape[:2], [polygon], **ratios)


def _sharp_polygon() -> np.ndarray:
    """Тот же контур без паддинга и растушёвки — эталон для сравнения."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.fillPoly(mask, [mask_generator.face_polygon(_fake_landmarks())], 255)
    return mask


# --- Геометрия полигона ---


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


# --- Детекция и валидация ---


def test_generate_mask_rejects_frame_without_face():
    blank = np.zeros((240, 240, 3), dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError):
        _mask(blank)


@pytest.mark.parametrize("padding,feather", [(-0.1, 0.1), (0.1, -0.1)])
def test_generate_mask_rejects_negative_ratios(faked, padding, feather):
    with pytest.raises(InvalidImageError):
        _mask(faked, padding_ratio=padding, feather_ratio=feather)


# --- Профиль краёв: паддинг и растушёвка ---


def test_mask_is_single_channel_of_frame_size(faked):
    mask = _mask(faked)

    assert mask.shape == (400, 400)
    assert mask.dtype == np.uint8


def test_face_stays_fully_opaque(faked):
    """
    Главное свойство профиля: растушёвка живёт снаружи контура и не съедает
    лицо. До паддинга размытие гасило края полигона до полутонов, и модель
    перерисовывала лицо не целиком — отсюда и вклеенный вид результата.
    """
    mask = _mask(faked)

    assert mask[_sharp_polygon() == 255].min() == 255


def test_padding_widens_mask_beyond_face(faked):
    """Паддинг даёт модели поле, на котором она сводит лицо с иллюстрацией."""
    mask = _mask(faked)

    assert np.count_nonzero(mask) > np.count_nonzero(_sharp_polygon())


def test_feathering_produces_gradient(faked):
    """Полутона по краю — то, ради чего всё и затевалось: границы не видно."""
    mask = _mask(faked)

    assert np.count_nonzero((mask > 10) & (mask < 245)) > 0
    # Чёткая копия того же полигона полутонов не имеет
    sharp = _sharp_polygon()
    assert np.count_nonzero((sharp > 10) & (sharp < 245)) == 0


def test_feathering_reaches_zero(faked):
    """Спад доходит до нуля: фон обложки остаётся неприкосновенным."""
    mask = _mask(faked)

    assert mask.min() == 0


def test_ratios_scale_the_mask(faked):
    """Доли высоты лица, а не пиксели: шире доля — шире и кайма."""
    narrow = _mask(faked, padding_ratio=0.02, feather_ratio=0.03)
    wide = _mask(faked, padding_ratio=0.12, feather_ratio=0.18)

    assert np.count_nonzero(wide) > np.count_nonzero(narrow)


def test_zero_ratios_give_sharp_polygon(faked):
    """Вырожденный случай — ровно прежнее поведение без паддинга и размытия."""
    mask = _mask(faked, padding_ratio=0.0, feather_ratio=0.0)

    assert np.array_equal(mask, _sharp_polygon())


# --- Объединение контуров: лицо шаблона + вклейка коллажа ---


def test_mask_covers_every_polygon():
    """
    Пайплайн подаёт сюда два контура — лица обложки и вклеенного коллажа. Если
    маска накроет только первый, шов склейки окажется вне зоны инпейнтинга и
    останется виден: сглаживать его будет некому.
    """
    face = mask_generator.face_polygon(_fake_landmarks())
    paste = face + np.array([60, 0], dtype=np.int32)

    mask = mask_generator.mask_from_polygons(
        (400, 400), [face, paste], padding_ratio=0.0, feather_ratio=0.0
    )

    for polygon in (face, paste):
        area = np.zeros((400, 400), dtype=np.uint8)
        cv2.fillPoly(area, [polygon], 255)
        assert mask[area == 255].min() == 255, "пересечение контуров не должно стать дырой"


def test_union_mask_is_wider_than_a_single_contour():
    face = mask_generator.face_polygon(_fake_landmarks())
    paste = face + np.array([60, 0], dtype=np.int32)

    single = mask_generator.mask_from_polygons((400, 400), [face])
    union = mask_generator.mask_from_polygons((400, 400), [face, paste])

    assert np.count_nonzero(union) > np.count_nonzero(single)
