"""
Первый шаг пайплайна: перенос головы и жёсткая аппликация.

Главных свойств два, и оба — про то, ради чего пайплайн переписывали:
геометрия донора не меняется, а волосы сохраняют свой цвет. Ни mediapipe, ни
rembg не запускаются: сетка и силуэт подменяются заглушками.
"""

import logging

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
    """
    Аппликация без стирания причёски шаблона — оно проверяется отдельно.

    Якорь шеи тоже выключен: он двигает вклейку по холсту, а здесь проверяются
    свойства самого преобразования. Ему посвящены отдельные тесты ниже.
    """
    kwargs = {
        "erase_template_head": False,
        "feather_ratio": 0.0,
        "colour_match": 0.0,
        "anchor_neck": False,
    }
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
    """Прежнее направление осталось доступным: вклейка подтягивается к персонажу."""
    plain = _build(photo, cover, colour_match=0.0, colour_direction="to_template")
    matched = _build(photo, cover, colour_match=1.0, colour_direction="to_template")

    skin = np.zeros(cover.shape[:2], dtype=np.uint8)
    cv2.fillPoly(skin, [matched.face_polygon], 255)
    core = cv2.erode(skin, np.ones((15, 15), np.uint8)) > 0

    assert abs(float(matched.image[core].mean()) - 60) < abs(float(plain.image[core].mean()) - 60)


def test_reverse_direction_leaves_the_paste_alone(same_pose, photo, cover):
    """
    Направление по умолчанию — обратное: голова остаётся реалистичной, и
    подкрашивается тело персонажа, а не фотография. Значит, пиксели вклейки
    коррекция трогать не должна вовсе.
    """
    plain = _build(photo, cover, colour_match=0.0)
    matched = _build(photo, cover, colour_match=1.0)

    core = cv2.erode(matched.head_alpha, np.ones((7, 7), np.uint8)) > 250
    assert np.array_equal(matched.image[core], plain.image[core])
    assert matched.meta["colour_direction"] == "to_donor"


def test_reverse_direction_recolours_the_painted_body(monkeypatch, photo):
    """
    Открытая кожа персонажа — руки и грудь — подтягивается к тону вклейки.

    Здесь проверяется запасной путь, по цвету: семантическая разметка на
    синтетическом кадре человека не находит, да и суть проверки не в ней.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())
    monkeypatch.setattr(collage.parsing, "parse", lambda _: None)

    # Обложка: телесное тело персонажа снизу, фон того же оттенка по краям
    cover = np.zeros((400, 400, 3), dtype=np.uint8)
    cover[:] = (60, 60, 60)
    cover[300:, 120:280] = (120, 130, 150)  # кожа персонажа, темнее донорской

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        if model == "cover":
            cv2.circle(alpha, (200, 190), 120, 255, -1)
            alpha[300:, 120:280] = 255
        else:
            cv2.circle(alpha, (200, 190), 130, 255, -1)
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    plain = collage.build(photo, cover, model_photo="photo", model_cover="cover", colour_match=0.0)
    matched = collage.build(
        photo, cover, model_photo="photo", model_cover="cover", colour_match=1.0
    )

    body = (slice(340, 380), slice(160, 240))
    assert matched.meta["body_skin_px"] > 0, "кожа персонажа должна найтись"
    # Тело поехало в сторону донорского тона, а не осталось прежним
    assert not np.array_equal(matched.image[body], plain.image[body])
    assert float(matched.image[body].mean()) > float(plain.image[body].mean())


def test_unknown_colour_direction_is_refused(same_pose, photo, cover):
    with pytest.raises(InvalidImageError):
        _build(photo, cover, colour_direction="sideways")


def test_colour_match_leaves_the_hair_alone(same_pose, photo, cover):
    """
    Волосы переносят ради того, чтобы они остались волосами заказчика.
    Подтянуть их к палитре персонажа — значит перекрасить донора.
    """
    matched = _build(photo, cover, colour_match=1.0, colour_direction="to_template")

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


# --- Масштаб: строго по лицу ---


def test_scale_never_sees_the_hair():
    """
    Главная гарантия: причёска в масштаб не входит. Донор с копной до плеч и
    он же стриженый обязаны дать одинаковый масштаб — меняется силуэт, а не
    лицо.
    """
    donor, template = face_mesh(), face_mesh(centre=(180, 220), scale=0.5)

    _, scale = collage.similarity_transform(
        [donor[i] for i in collage._ALIGN_POINTS],
        [template[i] for i in collage._ALIGN_POINTS],
    )

    assert scale == pytest.approx(0.5, abs=0.02)
    # Все опорные точки — из сетки лица, а она заканчивается на бровях
    assert max(collage._ALIGN_POINTS) < 468


def test_biometric_marks_are_measured_separately():
    """
    Мерки независимы и на стилизованном лице расходятся. Считаем их все и
    кладём в метаданные: по одному числу потом не понять, почему голова вышла
    такой.
    """
    ratios = collage.biometric_ratios(face_mesh(), face_mesh(scale=0.5))

    assert {"eyes", "cheeks", "jaw", "face_height"} <= set(ratios)
    for mark, value in ratios.items():
        assert value == pytest.approx(0.5, abs=0.02), mark


def test_median_mark_ignores_one_distorted_feature():
    """
    У рисованного персонажа глаза вдвое больше человеческих. Медиана обязана
    это пережить — одна уехавшая мерка не должна тянуть масштаб за собой.
    """
    ratios = {"eyes": 1.6, "cheeks": 1.05, "jaw": 1.0, "face_height": 0.95}

    assert collage.biometric_scale(ratios, "median") == pytest.approx(1.025, abs=0.01)
    assert collage.biometric_scale(ratios, "eyes") == 1.6
    assert collage.biometric_scale(ratios, "umeyama") is None


def test_unknown_mark_is_refused():
    with pytest.raises(InvalidImageError):
        collage.biometric_scale({"eyes": 1.0}, "nose_length")


def test_chosen_mark_sets_the_size(same_pose, photo, cover):
    """Выбранная мерка должна действительно менять масштаб, а не только логи."""
    default = _build(photo, cover)
    forced = _build(photo, cover, scale_mark="eyes")

    assert forced.meta["scale_mark"] == "eyes"
    assert forced.meta["scale_marks"]["eyes"] == pytest.approx(1.0, abs=0.01)
    assert default.meta["scale_mark"] == "umeyama"


# --- Якорь шеи ---


def _cover_with_collar(collar_y: int) -> np.ndarray:
    """
    Обложка, где под подбородком персонажа есть кожа, а с collar_y — одежда.

    Силуэт-заглушка вклейки кончается на y=320 (круг r=130 вокруг (200, 190)),
    поэтому воротник ниже 320 означает зазор, а выше — нахлёст.
    """
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:] = (60, 60, 60)
    cv2.rectangle(image, (140, 150), (260, 400), (170, 180, 210), -1)  # лицо и шея
    cv2.rectangle(image, (100, collar_y), (300, 400), (40, 90, 220), -1)  # воротник
    return image


@pytest.fixture
def cover_with_collar() -> np.ndarray:
    return _cover_with_collar(330)


def test_anchor_drops_the_head_onto_the_collar(same_pose, photo, cover_with_collar):
    """
    Шея на фотографии короче, чем у персонажа: голова, посаженная строго по
    лицу, повисает над воротником. Якорь опускает её до нахлёста.
    """
    floating = _build(photo, cover_with_collar)
    anchored = _build(photo, cover_with_collar, anchor_neck=True)

    assert anchored.meta["anchor_px"] > 0
    # Низ вклейки опустился ровно на величину сдвига
    assert int(np.nonzero(anchored.head_alpha.any(axis=1))[0].max()) > int(
        np.nonzero(floating.head_alpha.any(axis=1))[0].max()
    )


def test_anchor_measures_the_silhouette_not_the_promised_cut(same_pose, photo):
    """
    Регрессия с боевого прогона. Когда линия одежды донора не нашлась и взят
    запасной отступ, расчётный отрезок среза уезжает на 0.55 высоты лица ниже
    подбородка — независимо от того, есть ли там пиксели. Якорь мерил этот
    отрезок, считал, что шея уже достала до воротника, и не двигал ничего, а на
    обложке висела голова с коротким обрубком шеи.

    Здесь ровно та расстановка: срез обещан ниже воротника (260 + 80 = 340 при
    воротнике на 330), а силуэт кончается на 320. Мерить надо силуэт.
    """
    result = _build(photo, _cover_with_collar(330), anchor_neck=True, neck_ratio=1.0)

    assert result.meta["anchor_px"] > 0, "зазор есть, и его должно быть видно"


def test_anchor_does_not_lift_an_overlapping_neck(same_pose, photo):
    """
    Сдвиг только вниз. Если шея уже перекрыла воротник — это нахлёст, ровно то,
    что нужно, и поднимать её обратно незачем.
    """
    result = _build(
        photo, _cover_with_collar(300), anchor_neck=True, take_neck=True, neck_ratio=1.0
    )

    assert result.meta["anchor_px"] == 0.0


def test_anchor_is_off_when_the_neck_is_not_taken(same_pose, photo, cover_with_collar):
    """
    Якорь сажал в воротник именно перенесённую шею. Её больше нет, и тянуть лицо
    вниз незачем: оно должно стоять там, где его нарисовал художник, а шею
    дорисует модель.
    """
    result = _build(photo, cover_with_collar)

    assert result.meta["anchor_px"] == 0.0


def test_anchor_is_capped(same_pose, photo, cover_with_collar):
    """
    Потолок обязателен: дотянуть шею любой ценой означает уронить лицо ниже,
    чем его нарисовал художник.
    """
    result = _build(photo, cover_with_collar, anchor_neck=True)

    template_face = mask_generator.face_polygon(face_mesh())
    height = float(template_face[:, 1].max() - template_face[:, 1].min())
    assert result.meta["anchor_px"] <= collage._ANCHOR_MAX_RATIO * height + 1


def test_anchor_is_skipped_when_the_collar_is_not_found(same_pose, photo, cover):
    """
    Ровная обложка: воротника не видно. Двигать лицо заказчика по холсту на
    основании догадки нельзя — пусть лучше останется зазор, его хотя бы видно.
    """
    result = _build(photo, cover, anchor_neck=True)

    assert result.meta["anchor_px"] == 0.0
    assert result.meta["anchor_collar"] != "skin"


# --- Диагностика: цифры должны доезжать до терминала ---


def test_geometry_is_logged_as_plain_numbers(same_pose, photo, cover):
    """
    Строка [geometry] дублирует то, что и так лежит в meta, и это намеренно:
    поля extra доезжают не до всякого формата и не до всякого сборщика логов, а
    именно эти числа спрашивают первыми, когда голова вышла не того размера.

    Ловим своим хендлером, а не caplog: тот живёт в плагине, который можно
    отключить ключом запуска, а проверка должна работать всегда.
    """
    lines: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    logger = logging.getLogger("app.pipelines.collage")
    handler = _Catch()
    logger.addHandler(handler)
    # Именно setLevel, а не присваивание .level: он сбрасывает кэш isEnabledFor,
    # в котором после первого же build лежит «INFO выключен»
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        _build(photo, cover)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    line = next(m for m in lines if "[geometry]" in m)
    for field in ("scale=", "mark=", "marks=", "anchor=", "neck=", "face:", "head:"):
        assert field in line, field
    # Мерки перечислены поимённо, а не одним числом
    assert "face_height" in line and "cheeks" in line


# --- Художественный множитель размера головы ---


def test_multiplier_shrinks_the_head(same_pose, photo, cover):
    """
    Ручка эстетическая: 1.0 геометрически верно (лицо донора совпадает с лицом
    персонажа), но у мультяшной серии это читается как большая голова на
    маленьком туловище.
    """
    plain = _build(photo, cover)
    smaller = _build(photo, cover, scale_multiplier=0.85)

    assert smaller.meta["scale"] == pytest.approx(plain.meta["scale"] * 0.85, rel=0.01)
    assert int((smaller.head_alpha > 127).sum()) < int((plain.head_alpha > 127).sum())
    assert smaller.meta["scale_multiplier"] == 0.85


def test_multiplier_applies_to_a_chosen_mark_too(same_pose, photo, cover):
    """Множитель поверх мерки, а не вместо неё: работать должны обе ручки."""
    plain = _build(photo, cover, scale_mark="cheeks")
    smaller = _build(photo, cover, scale_mark="cheeks", scale_multiplier=0.5)

    assert smaller.meta["scale"] == pytest.approx(plain.meta["scale"] * 0.5, rel=0.01)


def test_multiplier_keeps_the_face_proportions(same_pose, photo, cover):
    """
    Голова уменьшается целиком: пропорции самого лица меняться не должны, иначе
    ручка перестала бы быть безобидной для сходства.
    """
    smaller = _build(photo, cover, scale_multiplier=0.7)
    plain = _build(photo, cover)

    def shape(polygon):
        width = polygon[:, 0].max() - polygon[:, 0].min()
        height = polygon[:, 1].max() - polygon[:, 1].min()
        return width / height

    assert shape(smaller.face_polygon) == pytest.approx(shape(plain.face_polygon), rel=0.02)


@pytest.mark.parametrize("value", [0.0, -0.5])
def test_non_positive_multiplier_is_refused(same_pose, photo, cover, value):
    with pytest.raises(InvalidImageError):
        _build(photo, cover, scale_multiplier=value)


# --- Цветокоррекция по зонам ---


def _flat(top: tuple, bottom: tuple) -> np.ndarray:
    """Кадр из двух ровных половин: «лицо» сверху, «шея» снизу."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50] = top
    image[50:] = bottom
    return image


def _halves() -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros((100, 100), dtype=np.uint8)
    top[:50] = 255
    bottom = np.zeros((100, 100), dtype=np.uint8)
    bottom[50:] = 255
    return top, bottom


def test_each_skin_zone_reaches_its_own_reference():
    """
    Лицо на фотографии освещено, шея под подбородком лежит в собственной тени.
    Одна поправка на всю кожу подгоняет статистику по площади — то есть по
    лицу, — и шея остаётся фотографически тёмной.
    """
    donor = _flat((200, 200, 200), (40, 40, 40))  # светлое лицо, тёмная шея
    template = _flat((150, 150, 150), (120, 120, 120))
    face, neck = _halves()

    result = collage._match_skin(donor, template, [(face, face), (neck, neck)], 1.0, 1.0)

    # На ровной заливке разброс нулевой, значит поправка — чистый сдвиг среднего
    assert result[10, 50, 0] == pytest.approx(150, abs=3), "лицо пришло к лицу"
    assert result[90, 50, 0] == pytest.approx(120, abs=3), "шея пришла к шее"


def test_one_zone_for_everything_leaves_the_neck_off():
    """
    Проверка от противного: та самая ошибка, из-за которой шея отрывалась по
    цвету. Одна зона на всю кожу подгоняет статистику по площади — по лицу, — и
    шея приходит куда угодно, только не к шее персонажа.
    """
    donor = _flat((200, 200, 200), (40, 40, 40))
    template = _flat((150, 150, 150), (120, 120, 120))
    face, neck = _halves()

    single = collage._match_skin(donor, template, [(np.maximum(face, neck), face)], 1.0, 1.0)
    split = collage._match_skin(donor, template, [(face, face), (neck, neck)], 1.0, 1.0)

    assert abs(int(split[90, 50, 0]) - 120) < abs(int(single[90, 50, 0]) - 120)


def test_reference_comes_from_the_character_not_from_under_the_paste():
    """
    Эталон берётся по коже персонажа. Пока он считался по пикселям ПОД маской
    донора, после сдвига и уменьшения там оказывались воротник и фон, и «тон
    кожи шаблона» вычислялся по чему угодно, кроме кожи.
    """
    donor = _flat((200, 200, 200), (200, 200, 200))
    # Под вклейкой — тёмная одежда, а кожа персонажа светлая и лежит в стороне
    template = _flat((20, 20, 20), (20, 20, 20))
    template[:, 70:] = (180, 180, 180)

    zone = np.zeros((100, 100), dtype=np.uint8)
    zone[:, :50] = 255
    reference = np.zeros((100, 100), dtype=np.uint8)
    reference[:, 70:] = 255

    result = collage._match_skin(donor, template, [(zone, reference)], 1.0, 1.0)

    assert result[50, 20, 0] == pytest.approx(180, abs=5), "тон взят с кожи персонажа"


def test_skin_zones_split_the_paste_at_the_chin(same_pose, photo, cover):
    """
    Разделение на лицо и шею осталось для режима с перенесённой шеей: там у
    донорской шеи своя тень и свой эталон.
    """
    result = _build(photo, cover, take_neck=True, neck_ratio=0.5)

    assert result.meta["skin_face_px"] > 0
    assert result.meta["skin_neck_px"] > 0, "шея обязана попасть в цветокоррекцию"


def test_paste_has_no_neck_by_default(same_pose, photo, cover):
    """Срез по челюсти: шеи в аппликации нет вовсе, её рисует инпейнтинг."""
    result = _build(photo, cover)

    assert result.meta["skin_neck_px"] == 0


def test_mask_base_follows_the_paste_not_the_character(same_pose, photo, cover):
    """
    Доли масок меряются от вклеенного лица. При множителе меньше единицы голова
    меньше персонажной, и кольца, посчитанные от лица персонажа, оказались бы
    шире нужного — ровно там, где потом виден грязный контур.
    """
    plain = _build(photo, cover)
    smaller = _build(photo, cover, scale_multiplier=0.7)

    assert smaller.meta["face_height_target"] == plain.meta["face_height_target"]
    assert smaller.meta["face_height_paste"] == pytest.approx(
        plain.meta["face_height_paste"] * 0.7, rel=0.05
    )


# --- Стык по линии одежды ---


def test_character_neck_is_erased_down_to_the_collar(monkeypatch, photo):
    """
    Шея донора садится прямо в воротник, а нарисованная шея персонажа
    стирается. Сажать одну шею поверх другой значит получить две шеи и шов на
    горле — самое заметное место портрета; стык по линии одежды прячут
    воротник и плечи.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())

    # Обложка: кожа под подбородком до y=330, ниже одежда
    cover = np.zeros((400, 400, 3), dtype=np.uint8)
    cover[:] = (60, 60, 60)
    cv2.rectangle(cover, (140, 150), (260, 400), (170, 180, 210), -1)
    cv2.rectangle(cover, (100, 330), (300, 400), (40, 90, 220), -1)

    def _silhouette(image, model):
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(alpha, (200, 190), 120 if model == "cover" else 130, 255, -1)
        alpha[260:400, 140:260] = 255
        return alpha

    monkeypatch.setattr(segmentation, "silhouette", _silhouette)

    result = collage.build(
        photo, cover, model_photo="photo", model_cover="cover", colour_match=0.0
    )

    assert result.meta["erase_neck_source"] == "collar"
    # Затирка дошла до воротника, а не остановилась под подбородком
    assert result.meta["erase_neck_ratio"] > 0.5


def test_erase_depth_can_be_fixed_by_hand(monkeypatch, photo, cover):
    """Число вместо поиска: обложка, где линию одежды видно только человеку."""
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())
    monkeypatch.setattr(
        segmentation,
        "silhouette",
        lambda image, model: cv2.circle(
            np.zeros(image.shape[:2], np.uint8), (200, 190), 150, 255, -1
        ),
    )

    result = collage.build(photo, cover, colour_match=0.0, erase_neck_ratio=0.2)

    assert result.meta["erase_neck_ratio"] == 0.2
    assert result.meta["erase_neck_source"] == "fixed"


def test_body_correction_stays_near_the_paste(same_pose, photo, cover):
    """
    Радиус — не «сколько нужно», а «докуда безопасно». На этих обложках кожу
    персонажа от декораций не отделить ни цветом, ни связностью, поэтому
    коррекция сознательно локальная.
    """
    near = _build(photo, cover, colour_match=1.0, body_reach=0.5)
    far = _build(photo, cover, colour_match=1.0, body_reach=3.0)

    assert near.meta["body_skin_px"] <= far.meta["body_skin_px"]


def test_body_correction_can_be_switched_off(same_pose, photo, cover):
    """Радиус ноль — тело не трогаем вовсе, вклейка всё равно остаётся своей."""
    result = _build(photo, cover, colour_match=1.0, body_reach=0.0)

    assert result.meta["body_skin_px"] == 0


def test_body_skin_comes_from_parsing_when_available(monkeypatch, photo, cover):
    """
    Когда разметка есть, кожу берут у неё, а не у цветовой эвристики: цвет
    отличить кожу от бежевой ткани не может, замерено.
    """
    monkeypatch.setattr(mask_generator, "face_landmarks", lambda _: face_mesh())
    monkeypatch.setattr(
        segmentation,
        "silhouette",
        lambda image, model: cv2.circle(
            np.zeros(image.shape[:2], np.uint8), (200, 190), 150, 255, -1
        ),
    )

    skin = np.zeros((400, 400), dtype=np.uint8)
    skin[330:380, 150:250] = 255
    monkeypatch.setattr(
        collage.parsing,
        "parse",
        lambda _: collage.parsing.Parsed(
            face=np.zeros_like(skin), hair=np.zeros_like(skin), skin=skin,
            clothes=np.zeros_like(skin),
        ),
    )

    result = collage.build(photo, cover, colour_match=1.0)

    assert result.meta["body_skin_source"] == "parsing"
    assert result.meta["body_skin_px"] > 0
