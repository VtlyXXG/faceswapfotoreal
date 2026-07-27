"""
Первый шаг пайплайна: перенос головы и жёсткая аппликация.

Главных свойств два, и оба — про то, ради чего пайплайн переписывали:
геометрия донора не меняется, а волосы сохраняют свой цвет. Ни mediapipe, ни
rembg не запускаются: сетка и силуэт подменяются заглушками.
"""

import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import collage, expression, mask_generator, segmentation
from tests.conftest import face_mesh

# Цвет «волос» на тестовой фотографии — намеренно ядовитый: любую перекраску
# видно сразу.
_HAIR = (30, 200, 40)
_SKIN = (170, 180, 200)


@pytest.fixture
def photo() -> np.ndarray:
    """
    Фотография: зелёные волосы вокруг светлого лица.

    Разделение на кожу и волосы здесь принципиально — на них держится проверка,
    что LAB-коррекция трогает только кожу.
    """
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:] = _HAIR
    cv2.fillPoly(image, [mask_generator.face_polygon(face_mesh())], _SKIN)
    return image


@pytest.fixture
def cover() -> np.ndarray:
    """Обложка ровного цвета — сразу видно, что от неё осталось."""
    return np.full((400, 400, 3), 60, dtype=np.uint8)


@pytest.fixture
def head_silhouette(monkeypatch):
    """Сегментатор, отдающий круглую голову вокруг лица заглушки."""

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(alpha, (200, 190), 130, 255, -1)
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)


@pytest.fixture
def same_pose(monkeypatch, head_silhouette):
    """Лица донора и шаблона совпадают: преобразование должно выйти единичным."""
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())


def _build(photo, cover, **overrides):
    """Аппликация без стирания причёски шаблона — оно проверяется отдельно."""
    kwargs = {"erase_template_head": False, "feather_ratio": 0.0, "colour_match": 0.0}
    kwargs.update(overrides)
    return collage.build(photo, cover, **kwargs)


# --- Преобразование подобия ---


def test_transform_recovers_known_pose():
    source = np.array(face_mesh(), dtype=np.float64)[list(collage._ALIGN_POINTS)]
    target = np.array(face_mesh(centre=(300, 250), scale=0.5, angle=20), dtype=np.float64)[
        list(collage._ALIGN_POINTS)
    ]

    matrix, scale = collage.similarity_transform(source, target)

    assert scale == pytest.approx(0.5, abs=0.02)
    projected = source @ matrix[:, :2].T + matrix[:, 2]
    assert np.abs(projected - target).max() < 2


def test_transform_cannot_distort_the_head():
    """
    Четыре степени свободы вместо шести — то, что физически запрещает модели
    геометрии подогнать донора под форму чужого лица. У матрицы [s·R | t]
    столбцы ортогональны и равны по длине; у аффина это не так, и лицо едет.
    """
    source = np.array(face_mesh(), dtype=np.float64)[list(collage._ALIGN_POINTS)]
    # Приёмник намеренно «сплюснут» по вертикали: полный аффин сжал бы лицо
    stretched = np.array(face_mesh(centre=(210, 190)), dtype=np.float64)
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


# --- Аппликация ---


def test_hair_is_transferred_with_the_head(same_pose, photo, cover):
    """
    Главное новое требование: причёска донора переносится вместе с лицом. Над
    лбом на обложке должны оказаться его волосы, а не персонажа.
    """
    result = _build(photo, cover)

    above_brow = result.image[110, 200]
    assert tuple(int(v) for v in above_brow) == _HAIR


def test_donor_pixels_are_pasted_verbatim(same_pose, photo, cover):
    """
    Сердце всей затеи: внутри силуэта остаются пиксели фотографии, а не их
    интерпретация. Проверяется на совпадающих позах и без коррекции цвета —
    тогда перенос обязан быть тождественным.
    """
    result = _build(photo, cover)

    core = cv2.erode(result.head_alpha, np.ones((7, 7), np.uint8)) > 250
    assert np.abs(result.image[core].astype(int) - photo[core].astype(int)).max() <= 1


def test_feathered_edge_stays_inside_the_cutout(same_pose, photo, cover):
    """
    Растушёвка края ведётся внутрь силуэта. Симметричная вернула бы наружу те
    самые пиксели фона фотографии, которые срезала эрозия в segmentation, —
    только с половинной прозрачностью, то есть тем же ореолом вокруг волос.
    """
    hard = _build(photo, cover, feather_ratio=0.0)
    soft = _build(photo, cover, feather_ratio=0.05)

    assert not np.any((soft.head_alpha > 0) & (hard.head_alpha == 0))
    assert 0 < int((soft.head_alpha > 0).sum()) < int((hard.head_alpha > 0).sum())


def test_cover_outside_the_head_is_untouched(same_pose, photo, cover):
    result = _build(photo, cover)

    outside = cv2.dilate(result.head_alpha, np.ones((7, 7), np.uint8)) == 0
    assert np.array_equal(result.image[outside], cover[outside])


def test_scaled_donor_keeps_its_proportions(monkeypatch, head_silhouette, photo, cover):
    """
    Лицо донора вдвое крупнее лица на обложке: после переноса контур обязан
    совпасть с лицом шаблона по обеим осям одинаково — иначе где-то затесался
    неравномерный масштаб.
    """
    meshes = iter([face_mesh(scale=2.0), face_mesh(centre=(180, 220))])
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: next(meshes))

    result = _build(photo, cover)

    face, template = result.face_polygon, mask_generator.face_polygon(face_mesh(centre=(180, 220)))
    for axis in (0, 1):
        assert (face[:, axis].max() - face[:, axis].min()) == pytest.approx(
            template[:, axis].max() - template[:, axis].min(), abs=3
        )
    assert result.meta["scale"] == pytest.approx(0.5, abs=0.02)


# --- Коррекция тона: кожа да, волосы нет ---


def test_colour_match_pulls_skin_towards_the_cover(same_pose, photo, cover):
    plain = _build(photo, cover, colour_match=0.0)
    matched = _build(photo, cover, colour_match=1.0)

    skin = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.fillPoly(skin, [matched.face_polygon], 255)
    core = cv2.erode(skin, np.ones((15, 15), np.uint8)) > 0

    assert abs(float(matched.image[core].mean()) - 60) < abs(float(plain.image[core].mean()) - 60)


def test_colour_match_leaves_the_hair_alone(same_pose, photo, cover):
    """
    Волосы переносят ради того, чтобы они остались волосами заказчика.
    Подтянуть их к палитре персонажа — значит перекрасить донора.
    """
    matched = _build(photo, cover, colour_match=1.0)

    assert tuple(int(v) for v in matched.image[110, 200]) == _HAIR


def test_colour_match_ratio_is_validated(same_pose, photo, cover):
    with pytest.raises(InvalidImageError):
        _build(photo, cover, colour_match=1.5)


# --- Стирание причёски персонажа ---


def test_template_hair_outside_the_paste_is_erased(monkeypatch, photo, cover):
    """
    У длинноволосого персонажа вокруг вклеенной головы остаётся его
    собственная шевелюра — на развороте это читается как два человека.
    Понизить strength и попросить модель убрать её нельзя: на 0.2 она ничего
    не убирает.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        # Голова персонажа заметно шире вклеиваемой: у заказчика стрижка,
        # у героя обложки грива до плеч
        radius = 70 if model == "photo" else 190
        cv2.circle(alpha, (200, 190), radius, 255, -1)
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(
        photo, cover, model_photo="photo", model_cover="cover", colour_match=0.0
    )

    assert result.erased.any(), "торчащая причёска персонажа должна попасть в стирание"
    assert result.meta["erased_ratio"] > 0


def test_erased_hair_is_not_smeared_back_into_the_hole(monkeypatch, photo, cover):
    """
    Тёмное кольцо вокруг вклейки. Пока стиралась только торчащая часть головы
    персонажа, дыра выходила кольцом, и её внутренней границей были его же
    тёмные волосы: любой локальный метод тянет цвет от границы внутрь, и они
    размазывались обратно. Поэтому голова стирается целиком, а сам персонаж
    исключается из источников цвета — заливка берётся только из фона.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    dark = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.circle(dark, (200, 190), 150, 255, -1)
    painted = cover.copy()
    painted[dark > 0] = (10, 10, 10)  # грива персонажа — почти чёрная

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(alpha, (200, 190), 70 if model == "photo" else 150, 255, -1)
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(
        photo, painted, model_photo="photo", model_cover="cover", colour_match=0.0
    )

    # Полоса между вклейкой и краем стёртой гривы — то, что увидит заказчик
    band = (result.erased > 0) & (result.head_alpha == 0)
    assert band.any()
    assert float(result.image[band].mean()) > 40, "фон вместо размазанных тёмных волос"


def test_erasing_does_not_touch_the_rest_of_the_character(monkeypatch, photo, cover):
    """
    Стирается голова, а не персонаж: руки, одежда и всё ниже воротника обязаны
    остаться нетронутыми. Заливка знает про них только как про запрещённый
    источник цвета.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    painted = cover.copy()
    painted[330:400, :] = (0, 0, 200)  # «одежда» персонажа внизу кадра

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(alpha, (200, 190), 70 if model == "photo" else 150, 255, -1)
        alpha[330:400, :] = 255
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(
        photo, painted, model_photo="photo", model_cover="cover", colour_match=0.0
    )

    assert np.array_equal(result.image[330:400, :], painted[330:400, :])


def test_full_bleed_cover_does_not_fill_the_hole_with_black(monkeypatch, photo, cover):
    """
    Разворот, где персонаж занимает почти весь кадр: фона, из которого можно
    брать цвет, почти нет. Пирамида без затравки заливает дыру чёрным — а
    чёрное пятно на месте головы хуже любого ореола.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    def _silhouette(image, model):
        if model == "photo":
            return cv2.circle(np.zeros(image.shape[:2], np.uint8), (200, 190), 70, 255, -1)
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        alpha[:6, :6] = 0  # весь кадр — персонаж, кроме уголка
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(photo, cover, model_photo="photo", model_cover="cover", colour_match=0.0)

    band = (result.erased > 0) & (result.head_alpha == 0)
    assert band.any()
    assert float(result.image[band].mean()) == pytest.approx(60, abs=2), "цвет уцелевшего фона"


def test_unknown_erase_method_is_refused(monkeypatch, photo, cover):
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())
    monkeypatch.setattr(
        segmentation,
        "silhouette",
        lambda image, model: cv2.circle(
            np.zeros(image.shape[:2], np.uint8), (200, 190), 150, 255, -1
        ),
    )

    with pytest.raises(InvalidImageError):
        collage.build(photo, cover, erase_method="poisson")


def test_segmenter_failure_on_the_cover_does_not_kill_the_order(monkeypatch, photo, cover):
    """
    Рисованный персонаж — не тот материал, на котором учили U²-Net. Промах на
    обложке обязан стоить предупреждения в логе, а не отказа заказчику.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    def _silhouette(image, model):
        if model == "cover":
            return np.zeros(image.shape[:2], dtype=np.uint8)
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(alpha, (200, 190), 130, 255, -1)
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(photo, cover, model_photo="photo", model_cover="cover")

    assert result.meta["erased_ratio"] is None
    assert not result.erased.any()


# --- Мимика и отказы ---


def test_expression_runs_before_the_paste(same_pose, photo, cover, monkeypatch):
    """
    Точка расширения должна получать вырезанную голову и влиять на результат:
    иначе параметр эмоции окажется декоративным.
    """
    seen = {}

    def _spy(face, emotion=""):
        seen["emotion"] = emotion
        seen["has_alpha"] = bool(face.alpha.any())
        return expression.Face(image=face.image, alpha=face.alpha, points=face.points)

    monkeypatch.setattr(collage.expression, "transform", _spy)

    _build(photo, cover, emotion="grin")

    assert seen == {"emotion": "grin", "has_alpha": True}


def test_unknown_emotion_is_refused(same_pose, photo, cover):
    with pytest.raises(expression.ExpressionNotSupportedError):
        _build(photo, cover, emotion="smile")


def test_missing_face_reports_which_image(monkeypatch, photo, cover):
    """
    Лицо ищется в двух кадрах, и без пометки заказчик не поймёт, что
    переснимать: фотографию или подбирать другой шаблон.
    """

    def _fail(_):
        raise NoFaceDetectedError("нет лица")

    monkeypatch.setattr(mask_generator, "face_landmarks", _fail)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        _build(photo, cover)

    assert exc_info.value.details["image"] == "фотография заказчика"
