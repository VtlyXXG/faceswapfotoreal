"""
Первый шаг пайплайна: перенос лица и жёсткая вклейка.

Главное свойство, которое здесь проверяется, — геометрия донора не меняется.
Ради него весь двухшаговый пайплайн и появился: заказчику нужно портретное
сходство, а не «похожее» лицо, нарисованное моделью.

mediapipe не запускается: сетка подменяется заглушкой, потому что тесты про
перенос и вклейку, а не про детектор.
"""

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import collage, mask_generator


def _mesh(centre=(200, 200), scale=1.0, angle=0.0) -> list[tuple[int, int]]:
    """
    468 точек-заглушек: лицо-эллипс, повёрнутое и промасштабированное как надо.

    Точки раскладываются по кругу, чтобы контур получался выпуклым, а опорные
    индексы — различимыми: заглушка «все точки в одной координате» не дала бы
    проверить ни поворот, ни масштаб.
    """
    points = [(centre[0], centre[1])] * 468
    radians = np.radians(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )

    def place(index: int, dx: float, dy: float) -> None:
        x, y = rotation @ (np.array([dx, dy]) * scale)
        points[index] = (int(round(centre[0] + x)), int(round(centre[1] + y)))

    arc = mask_generator._JAW_ARC
    for offset, index in enumerate(arc):
        angle_step = np.pi * offset / (len(arc) - 1)
        place(index, 90 * np.cos(angle_step), 90 * np.sin(angle_step))

    brows = mask_generator._BROW_ARC
    for offset, index in enumerate(brows):
        place(index, -70 + offset * 15, -60)

    # Опорные точки совмещения: без них similarity_transform считает по
    # вырожденному набору (все индексы указывают в центр).
    for offset, index in enumerate(collage._ALIGN_POINTS):
        if points[index] != (centre[0], centre[1]):
            continue
        place(index, -40 + (offset % 5) * 20, -30 + (offset // 5) * 25)

    return points


@pytest.fixture
def photo() -> np.ndarray:
    """Шумовая «фотография»: на равномерной заливке вклейку не увидеть."""
    rng = np.random.default_rng(seed=17)
    return rng.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)


@pytest.fixture
def cover() -> np.ndarray:
    """«Обложка» ровного цвета — сразу видно, что от неё осталось."""
    return np.full((400, 400, 3), 60, dtype=np.uint8)


@pytest.fixture
def same_pose(monkeypatch):
    """Лица донора и шаблона совпадают: преобразование должно выйти единичным."""
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: _mesh())


# --- Преобразование подобия ---


def test_transform_recovers_known_pose():
    source = np.array(_mesh(), dtype=np.float64)[list(collage._ALIGN_POINTS)]
    target = np.array(_mesh(centre=(300, 250), scale=0.5, angle=20), dtype=np.float64)[
        list(collage._ALIGN_POINTS)
    ]

    matrix, scale = collage.similarity_transform(source, target)

    assert scale == pytest.approx(0.5, abs=0.02)
    projected = source @ matrix[:, :2].T + matrix[:, 2]
    assert np.abs(projected - target).max() < 2


def test_transform_cannot_distort_the_face():
    """
    Четыре степени свободы вместо шести — то, что физически запрещает модели
    геометрии подогнать донора под форму чужого лица. У матрицы [s·R | t]
    столбцы ортогональны и равны по длине; у аффина это не так, и лицо едет.
    """
    source = np.array(_mesh(), dtype=np.float64)[list(collage._ALIGN_POINTS)]
    # Приёмник намеренно «сплюснут» по вертикали: полный аффин сжал бы лицо
    stretched = np.array(_mesh(centre=(210, 190)), dtype=np.float64)
    stretched[:, 1] *= 0.6
    target = stretched[list(collage._ALIGN_POINTS)]

    matrix, _ = collage.similarity_transform(source, target)

    linear = matrix[:, :2]
    assert float(linear[:, 0] @ linear[:, 1]) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(linear[:, 0]) == pytest.approx(np.linalg.norm(linear[:, 1]))


def test_transform_rejects_degenerate_points():
    same = np.zeros((len(collage._ALIGN_POINTS), 2), dtype=np.float64)

    with pytest.raises(InvalidImageError):
        collage.similarity_transform(same, same)


# --- Вклейка ---


def test_donor_pixels_are_pasted_verbatim(same_pose, photo, cover):
    """
    Сердце всей затеи: внутри контура остаются пиксели фотографии, а не их
    интерпретация. Проверяется на совпадающих позах и без коррекции цвета —
    тогда перенос обязан быть тождественным.
    """
    result = collage.build(photo, cover, grow_ratio=0.0, feather_ratio=0.0, colour_match=0.0)

    inside = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.fillPoly(inside, [result.paste_polygon], 255)
    # Край контура сглажен антиалиасингом заливки — сравниваем заведомо
    # внутренние пиксели
    core = cv2.erode(inside, np.ones((5, 5), np.uint8)) > 0

    assert np.abs(result.image[core].astype(int) - photo[core].astype(int)).max() <= 1


def test_cover_outside_the_paste_is_untouched(same_pose, photo, cover):
    result = collage.build(photo, cover, grow_ratio=0.0, feather_ratio=0.0, colour_match=0.0)

    outside = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.fillPoly(outside, [result.paste_polygon], 255)
    untouched = cv2.dilate(outside, np.ones((5, 5), np.uint8)) == 0

    assert np.array_equal(result.image[untouched], cover[untouched])


def test_paste_covers_the_template_face(same_pose, photo, cover):
    """Черты персонажа обложки должны уйти под вклейку целиком."""
    result = collage.build(photo, cover)

    assert result.meta["coverage"] >= 0.9


def test_grow_widens_the_paste(same_pose, photo, cover):
    narrow = collage.build(photo, cover, grow_ratio=0.0, feather_ratio=0.0)
    wide = collage.build(photo, cover, grow_ratio=0.08, feather_ratio=0.0)

    changed = lambda result: np.count_nonzero(  # noqa: E731
        (result.image != cover).any(axis=2)
    )
    assert changed(wide) > changed(narrow)


def test_scaled_donor_keeps_its_proportions(monkeypatch, photo, cover):
    """
    Лицо донора вдвое крупнее лица на обложке: после переноса контур вклейки
    обязан совпасть с лицом шаблона по обеим осям одинаково — иначе где-то
    затесался неравномерный масштаб.
    """
    meshes = iter([_mesh(scale=2.0), _mesh(centre=(180, 220), scale=1.0)])
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: next(meshes))

    result = collage.build(photo, cover, grow_ratio=0.0, feather_ratio=0.0)

    paste, template = result.paste_polygon, result.face_polygon
    for axis in (0, 1):
        size_paste = paste[:, axis].max() - paste[:, axis].min()
        size_template = template[:, axis].max() - template[:, axis].min()
        assert size_paste == pytest.approx(size_template, abs=3)
    assert result.meta["scale"] == pytest.approx(0.5, abs=0.02)


# --- Приведение цвета ---


def test_colour_match_pulls_tone_towards_the_cover(same_pose, cover):
    """Свет и тон подгоняются локально: при strength ~0.2 модель не успевает."""
    bright = np.full((400, 400, 3), 230, dtype=np.uint8)

    plain = collage.build(bright, cover, colour_match=0.0)
    matched = collage.build(bright, cover, colour_match=1.0)

    region = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.fillPoly(region, [matched.paste_polygon], 255)
    core = cv2.erode(region, np.ones((9, 9), np.uint8)) > 0

    assert abs(float(matched.image[core].mean()) - 60) < abs(float(plain.image[core].mean()) - 60)


def test_colour_match_ratio_is_validated(same_pose, photo, cover):
    with pytest.raises(InvalidImageError):
        collage.build(photo, cover, colour_match=1.5)


# --- Отказы ---


def test_missing_face_reports_which_image(monkeypatch, photo, cover):
    """
    Раньше 422 всегда означала «нет лица на обложке». Теперь лицо ищется в двух
    кадрах, и без пометки заказчик не поймёт, что переснимать.
    """

    def _fail(_):
        raise NoFaceDetectedError("нет лица")

    monkeypatch.setattr(mask_generator, "face_landmarks", _fail)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        collage.build(photo, cover)

    assert exc_info.value.details["image"] == "фотография заказчика"
