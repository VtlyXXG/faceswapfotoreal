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
    """Зона 2 — стыки."""
    kwargs = {
        "head_alpha": _head_alpha(),
        "face_polygon": mask_generator.face_polygon(face_mesh()),
        "neck_line": ((120, 296), (280, 296)),
        "face_height": _FACE_HEIGHT,
    }
    kwargs.update(overrides)
    return mask_generator.seam_mask(_SHAPE, **kwargs)


def _erased(radius: int = 190) -> np.ndarray:
    """Дыра от чужой причёски: что стёрто и не закрыто вклейкой."""
    old_hair = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.circle(old_hair, (200, 190), radius, 255, -1)
    return np.where(_head_alpha() > 0, 0, old_hair).astype(np.uint8)


def _hole(**overrides) -> np.ndarray:
    """Зона 3 — дыра в фоне."""
    kwargs = {
        "head_alpha": _head_alpha(),
        "face_polygon": mask_generator.face_polygon(face_mesh()),
        "erased": _erased(),
        "face_height": _FACE_HEIGHT,
    }
    kwargs.update(overrides)
    return mask_generator.hole_mask(_SHAPE, **kwargs)


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


def test_neck_seam_is_a_narrow_strip():
    """
    Полоса на стыке шеи узкая и лежит на месте. Шея донора теперь доезжает до
    воротника, и закрывать оторванный край больше не нужно — нужно место под
    переход тона и контактную тень, не больше.
    """
    mask = _blend()

    # Непрерывный кусок, накрывающий сам стык (y=296): ниже по этой же колонке
    # идёт кольцо вдоль контура волос, и складывать их вместе нельзя
    open_rows = mask[:, 200] > 127
    assert open_rows[296], "стык обязан быть открыт"
    top = bottom = 296
    while top > 0 and open_rows[top - 1]:
        top -= 1
    while bottom < len(open_rows) - 1 and open_rows[bottom + 1]:
        bottom += 1

    assert (bottom - top) <= _FACE_HEIGHT * 0.25, "полоса не должна расползаться"


def test_hole_is_not_in_the_seam_zone():
    """
    Дыра от чужой причёски ушла в зону 3. Пока она попадала сюда, маска стыка
    раздувалась на пол-неба — при том, что сделать с этим небом на strength
    0.26 модель всё равно ничего не могла.
    """
    mask = _blend()

    # Далеко от вклейки, но внутри стёртой области
    assert mask[190, 30] == 0
    assert _erased()[190, 30] > 0, "проверяем именно стёртое место"


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
    [
        {"edge_ratio": -0.1},
        {"neck_ratio": -0.1},
        {"feather_ratio": -0.1},
        {"guard_ratio": -0.1},
        {"gradient_ratio": -0.1},
    ],
)
def test_negative_ratios_are_rejected(ratios):
    with pytest.raises(InvalidImageError):
        _blend(**ratios)


# --- Градиент: маска для высокого strength ---


def test_gradient_keeps_the_seam_fully_open():
    """
    Вершина конуса — сам стык. Если градиент сбивает её ниже 255, зона, ради
    которой всё затевалось, открывается модели лишь частично.
    """
    seam = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.circle(seam, (200, 200), 60, 255, 4)

    ramped = mask_generator.gradient(seam, _FACE_HEIGHT, 0.2)

    assert ramped[seam > 0].min() == 255


def test_gradient_falls_off_with_distance():
    """Склон линейный и заданной ширины: 16 пикселей при 0.2 от лица в 80."""
    seam = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.circle(seam, (200, 200), 60, 255, 4)

    ramped = mask_generator.gradient(seam, _FACE_HEIGHT, 0.2)

    # Наружу от кольца радиуса 60: чем дальше, тем темнее, за 16 px — ноль
    assert ramped[200, 200 + 66] > ramped[200, 200 + 72] > 0
    assert ramped[200, 200 + 80] == 0


def test_gradient_widens_the_zone_but_not_the_core():
    """
    Градиент шире растушёвки и мягче: полутонов должно стать заметно больше, а
    полностью открытых пикселей — не больше прежнего.
    """
    raw = _blend(feather_ratio=0.0)  # плато без растушёвки — это и есть сам стык
    ramped = _blend(gradient_ratio=0.2)

    def halftones(mask):
        return int(np.count_nonzero((mask > 10) & (mask < 245)))

    assert halftones(ramped) > halftones(raw)
    # Полностью открыт остаётся ровно стык: конус только добавляет склон
    assert int(np.count_nonzero(ramped == 255)) <= int(np.count_nonzero(raw == 255))


def test_gradient_still_spares_the_face():
    """
    Защита лица вычитается последней и при градиенте тоже: черты обязаны
    получить ровный ноль, иначе strength 0.5 перерисует их первыми.
    """
    mask = _blend(gradient_ratio=0.2)

    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(face_mesh())], 255)
    core = cv2.erode(face, np.ones((15, 15), np.uint8)) > 0

    assert mask[core].max() == 0


def test_zero_gradient_keeps_the_previous_mask():
    """Ноль — прежнее поведение ровно: старый режим должен остаться доступным."""
    assert np.array_equal(_blend(), _blend(gradient_ratio=0.0))


# --- Зона 3: дыра в фоне ---


def test_hole_covers_what_the_new_head_does_not():
    """
    Ради этой зоны всё и затевалось: у героя грива до плеч, у заказчика ёжик, и
    вокруг вклейки остаётся кусок неба, стёртый вместе с чужими волосами.
    """
    mask = _hole()

    assert mask.max() == 255
    assert mask[190, 40] == 255, "дальний край дыры открыт целиком"


def test_hole_fades_to_nothing_at_the_hair_but_leaves_no_gap():
    """
    Два требования разом, и раньше их пытались совместить отступом. Отступ
    оставлял вокруг головы кольцо, куда не доставала ни одна зона: заливка там
    светлая, и на тёмном фоне она читалась светящимся контуром — тем самым
    ореолом. Теперь защита мягкая: у самых волос сила ноль, но незакрытой
    полосы нет.
    """
    mask = _hole(margin_ratio=0.15)

    alpha = _head_alpha()
    at_hair = (cv2.dilate(alpha, np.ones((5, 5), np.uint8)) > 0) & (alpha == 0)
    assert mask[at_hair].max() < 40, "у самого контура зона почти погашена"

    # Но чуть дальше она уже работает — незакрытого кольца не остаётся
    band = (cv2.dilate(alpha, np.ones((2 * 40 + 1,) * 2, np.uint8)) > 0) & (alpha == 0)
    assert mask[band].max() > 200


def test_hole_spares_the_face_too():
    """Лицо вычитается из обеих зон: 0.85 по чертам — это другой человек."""
    mask = _hole(erased=np.full(_SHAPE, 255, dtype=np.uint8), margin_ratio=0.0)

    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(face_mesh())], 255)
    core = cv2.erode(face, np.ones((9, 9), np.uint8)) > 0

    assert mask[core].max() == 0


def test_hole_is_empty_without_erasing():
    """Персонаж со стрижкой: стирать нечего, второго вызова быть не должно."""
    mask = _hole(erased=np.zeros(_SHAPE, dtype=np.uint8))

    assert mask.max() == 0


@pytest.mark.parametrize("ratios", [{"margin_ratio": -0.1}, {"feather_ratio": -0.1}])
def test_hole_negative_ratios_are_rejected(ratios):
    with pytest.raises(InvalidImageError):
        _hole(**ratios)


# --- Кольцо стыка: несимметричное ---


def test_ring_reaches_further_out_than_in():
    """
    Внутри контура волосы заказчика, снаружи заливка на месте чужой причёски.
    Заходить внутрь надо минимально, а наружу — до зоны фона, иначе между ними
    останется полоса, которую не трогает ни один проход.
    """
    mask = _blend(edge_ratio=0.02, edge_outer_ratio=0.2, guard_ratio=0.0, feather_ratio=0.0)

    alpha = _head_alpha()
    # По вертикали вверх от центра круга (200, 190) радиуса 130
    outside = mask[190 - 130 - 12, 200]
    inside = mask[190 - 130 + 12, 200]

    assert outside == 255, "наружу кольцо дотягивается"
    assert inside == 0, "внутрь почти не заходит"
    assert alpha[190 - 130 + 12, 200] == 255, "и там действительно волосы"


def test_zones_overlap_and_the_hot_one_keeps_away():
    """
    Две вещи разом, и они тянут в разные стороны: зона фона идёт на 0.85 и
    съедает всё, до чего дотянется, — значит, к волосам её подпускать нельзя;
    но и зазора между зонами быть не должно. Отсюда широкое наружу кольцо.
    """
    seam = _blend(edge_ratio=0.02, edge_outer_ratio=0.25, guard_ratio=0.0, feather_ratio=0.0)
    hole = _hole(margin_ratio=0.15, feather_ratio=0.0)

    alpha = _head_alpha()
    at_hair = (cv2.dilate(alpha, np.ones((5, 5), np.uint8)) > 0) & (alpha == 0)
    assert hole[at_hair].max() < 40, "горячая зона у волос погашена"

    orphan = (alpha == 0) & (_erased() > 0) & (seam < 30) & (hole < 30)
    ring_band = cv2.dilate(alpha, np.ones((2 * 20 + 1,) * 2, np.uint8)) > 0
    assert not (orphan & ring_band).any(), "у самой вклейки полосы-сироты нет"


# --- Зона 4: стилизация вклейки ---


def test_paste_zone_opens_the_skin_and_spares_the_features():
    """
    Ради фактуры зона и заводится: тон подогнать можно, мазок кисти — нет.
    Но черты под ней должны остаться: guard_strength задаёт, какая доля защиты
    лица сохраняется.
    """
    mask = mask_generator.paste_mask(
        _SHAPE, _head_alpha(), mask_generator.face_polygon(face_mesh()), _FACE_HEIGHT
    )

    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(face_mesh())], 255)
    core = cv2.erode(face, np.ones((15, 15), np.uint8)) > 0
    hair = (_head_alpha() > 0) & (face == 0)

    assert mask[hair].max() == 255, "волосы и кожа вне лица открыты полностью"
    assert 0 < mask[core].mean() < 255 * 0.6, "черты под частичной защитой"


def test_paste_zone_can_protect_the_face_completely():
    """guard_strength=1.0 — прежнее поведение: лицо недоступно вовсе."""
    mask = mask_generator.paste_mask(
        _SHAPE,
        _head_alpha(),
        mask_generator.face_polygon(face_mesh()),
        _FACE_HEIGHT,
        guard_strength=1.0,
    )

    face = np.zeros(_SHAPE, dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(face_mesh())], 255)
    core = cv2.erode(face, np.ones((15, 15), np.uint8)) > 0

    assert mask[core].max() == 0


def test_paste_zone_steps_back_from_the_contour():
    """Сам контур — работа зоны 2; трогать край волос дважды незачем."""
    mask = mask_generator.paste_mask(
        _SHAPE,
        _head_alpha(),
        mask_generator.face_polygon(face_mesh()),
        _FACE_HEIGHT,
        inset_ratio=0.15,
    )

    # Полоса заведомо уже отступа (12 px при 0.15 от лица в 80): квадратное
    # ядро съедает больше эллиптического, и мерить надо с запасом
    alpha = _head_alpha()
    rim = (alpha > 0) & (cv2.erode(alpha, np.ones((2 * 7 + 1,) * 2, np.uint8)) == 0)
    assert mask[rim].max() == 0


@pytest.mark.parametrize("kwargs", [{"guard_strength": 1.5}, {"inset_ratio": -0.1}])
def test_paste_zone_validates_its_ratios(kwargs):
    with pytest.raises(InvalidImageError):
        mask_generator.paste_mask(
            _SHAPE, _head_alpha(), mask_generator.face_polygon(face_mesh()), _FACE_HEIGHT, **kwargs
        )


# --- Зона 6: шея и грудь персонажа ---


def _neck(**overrides) -> np.ndarray:
    # Вклейка кончается на подбородке (y=260): донора режут по челюсти, шеи в
    # аппликации нет — её и рисует эта зона
    kwargs = {
        "head_alpha": _head_alpha(radius=70),
        "face_polygon": mask_generator.face_polygon(face_mesh()),
        "chin": (200.0, 260.0),
        "axis": ((0.0, -1.0), (1.0, 0.0)),
        "face_height": _FACE_HEIGHT,
    }
    kwargs.update(overrides)
    return mask_generator.neck_mask(_SHAPE, **kwargs)


def test_neck_zone_opens_the_area_under_the_chin():
    """
    Донора режут по челюсти, шея с фотографии не переносится: фотографичная шея
    на нарисованной груди читалась дешёвой аппликацией. Значит, шею на обложке
    надо нарисовать — вот место, где модель это делает.
    """
    mask = _neck(guard_ratio=0.0)

    assert mask[300, 200] == 255, "прямо под подбородком зона открыта"
    assert mask[10, 200] == 0, "над головой ей делать нечего"


def test_neck_zone_never_touches_the_paste():
    """Подбородок и волосы донора рисовать заново не надо: они и есть цель."""
    mask = _neck(guard_ratio=0.0)

    assert mask[_head_alpha(radius=70) > 0].max() == 0


def test_semantic_skin_extends_the_zone_but_not_to_the_hands():
    """
    Разметка знает, где кожа, но брать её всю нельзя: руки персонажа
    перерисовывать незачем, они и так в материале обложки.
    """
    skin = np.zeros(_SHAPE, dtype=np.uint8)
    skin[300:340, 120:280] = 255  # грудь под подбородком
    skin[300:340, 10:60] = 255  # кисть далеко сбоку

    mask = _neck(body_skin=skin, guard_ratio=0.0)

    assert mask[320, 150] == 255, "грудь рядом с шеей открыта"
    assert mask[320, 30] == 0, "кисть не трогаем"


def test_neck_zone_works_without_parsing():
    """Без весов остаётся геометрия: полоса под подбородком в ширину челюсти."""
    mask = _neck(body_skin=None, guard_ratio=0.0)

    assert mask.max() == 255
