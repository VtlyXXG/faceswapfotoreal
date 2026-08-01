"""
Маска головы на шаблоне: то единственное, что осталось считаться локально.

Главное, что здесь проверяется, — смена знака. В прежнем пайплайне лицо из
маски **вычиталось**: под ней лежала вклеенная фотография заказчика, и трогать
её было нельзя. Теперь под маской лежит голова чужого персонажа, и лицо обязано
быть внутри — иначе модель нарисует новую голову вокруг старого лица.

mediapipe в тестах не запускается: и сетка, и разметка приходят заглушками.
Проверяется геометрия и логика выбора источника, а не качество детектора.
"""

import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import head_mask, parsing

_SHAPE = (400, 400)
_CENTRE = (200, 200)


@pytest.fixture
def image() -> np.ndarray:
    return np.full((*_SHAPE, 3), 120, dtype=np.uint8)


_HEAD_AXES = (60, 90)

# Глубина открытого ворота в заглушке: сколько пикселей кожи между подбородком и
# первой ниткой ткани.
_COLLAR = 40


def _parsed(
    head_centre=_CENTRE, head_axes=_HEAD_AXES, extra_head=None, collar=_COLLAR
) -> parsing.Parsed:
    """
    Разметка-заглушка: у каждого персонажа голова, под ней открытая кожа, ниже —
    ткань. Ровно те четыре класса, которыми оперирует selfie_multiclass.

    Тело рисуется столбцом по ширине своей головы, а не во всю ширину кадра:
    между героями разворота лежит фон, и кожа двух персонажей не смыкается. С
    полосой во всю ширину они были бы связаны в буквальном смысле, и отделить их
    не смог бы никакой разбор на компоненты — заглушка проверяла бы невозможное.

    :param extra_head: вторая голова на развороте — сосед, которого трогать нельзя
    :param collar: глубина открытого ворота; 0 — воротник под самым подбородком
    """
    import cv2

    face = np.zeros(_SHAPE, dtype=np.uint8)
    hair = np.zeros(_SHAPE, dtype=np.uint8)
    skin = np.zeros(_SHAPE, dtype=np.uint8)
    clothes = np.zeros(_SHAPE, dtype=np.uint8)

    def person(centre) -> None:
        # Волосы — весь овал головы, лицо — овал поменьше внутри него
        cv2.ellipse(hair, centre, head_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(face, centre, (head_axes[0] - 20, head_axes[1] - 30), 0, 0, 360, 255, -1)

        left, right = centre[0] - head_axes[0], centre[0] + head_axes[0]
        chin = centre[1] + head_axes[1]
        skin[chin : chin + collar, left:right] = 255
        clothes[chin + collar :, left:right] = 255

    person(head_centre)
    if extra_head is not None:
        person(extra_head)

    return parsing.Parsed(face=face, hair=hair, skin=skin, clothes=clothes)


@pytest.fixture
def stub(monkeypatch, mesh):
    """Подменяет оба источника: сетку лица и семантическую разметку."""

    def _apply(points=..., parsed=...):
        landmarks = mesh(centre=_CENTRE) if points is ... else points
        monkeypatch.setattr(head_mask, "try_landmarks", lambda _: landmarks)
        monkeypatch.setattr(parsing, "parse", lambda _: _parsed() if parsed is ... else parsed)

    return _apply


def _build(image, dilate=0.12, feather=0.10, neck=0.35) -> head_mask.HeadMask:
    return head_mask.build(
        image, dilate_ratio=dilate, feather_ratio=feather, neck_ratio=neck
    )


def test_face_of_the_character_is_inside_the_mask(image, stub):
    """
    Смена знака относительно прежнего пайплайна. Раньше лицо из маски
    вычиталось — оно было фотографией заказчика. Теперь это лицо ЧУЖОГО
    персонажа, и оставить его нетронутым значит получить старого героя с
    новой причёской.
    """
    stub()

    mask = _build(image).mask

    assert mask[_CENTRE[1], _CENTRE[0]] == 255


def test_hair_is_taken_from_the_parsing_not_from_the_face_mesh(image, stub):
    """
    Причёска персонажа обязана попасть в маску целиком: сетка лица про волосы
    не знает ничего, а маска по одному лицу оставляет по краю его старые
    пряди — тот самый «летающий» результат.
    """
    stub()

    mask = _build(image).mask

    # Макушка эллипса волос — на 90 px выше центра, лицо туда не достаёт
    assert mask[_CENTRE[1] - 85, _CENTRE[0]] == 255


def _reach(mask) -> int:
    """Докуда маска доходит вниз по центральному столбцу."""
    column = np.nonzero(mask[:, _CENTRE[0]] > 127)[0]
    return int(column.max())


# Подбородок персонажа в координатах разметки: низ овала головы.
_CHIN_Y = _CENTRE[1] + _HEAD_AXES[1]


def test_open_skin_below_the_chin_is_taken_into_the_mask(image, stub):
    """
    Главное следствие семантического правила. Кожа в вырезе ворота обязана быть
    под маской: голова генерируется целиком, тон ей модель подбирает свой, и
    кожа, оставленная снаружи, ляжет рядом с новой другим оттенком — шов на
    самом видном месте разворота.

    Числа за это не отвечают. Маска растекается по классу SKIN сама и ровно
    настолько, насколько ворот открыт.
    """
    stub()

    mask = _build(image).mask

    assert mask[_CHIN_Y + _COLLAR // 2, _CENTRE[0]] == 255, "кожа шеи обязана быть в маске"


def test_the_garment_is_never_touched(image, stub):
    """
    Обратная половина того же правила и жёсткое требование продакшена: одежда
    персонажа — часть шаблона и обязана дойти до печати без изменений. Инпейнт
    перерисовывает всё, что под маской, поэтому единственный способ сохранить
    полоски рубашки и шевроны на жилетке — не пускать туда маску вовсе.

    Проверяется после расширения и растушёвки: сами они тянут маску вниз на
    десятки пикселей, и последнее слово обязано остаться за тканью.
    """
    stub()

    result = _build(image, dilate=0.30, feather=0.15)
    clothes = _parsed().clothes

    assert result.meta["clothes_guard"] is True
    under_fabric = result.mask[clothes > 0]
    assert not (under_fabric > 127).any(), "ткань не может оказаться под маской"
    # Полутон по самой кромке ворота допустим: спад маски укладывается в поле
    # над тканью, но антиалиасинг кромки съедает у него единицы уровней
    assert int(under_fabric.max()) < 32


def test_a_closed_collar_stops_the_mask_at_the_chin(image, stub):
    """
    Тот же код без единой правки на другом шаблоне: ворот глухой, кожи под
    подбородком нет — и маска обязана остановиться на подбородке сама. Это и
    есть довод против обрезки по Y: одной константой оба случая не описать.
    """
    stub(parsed=_parsed(collar=0))

    mask = _build(image).mask

    assert _reach(mask) <= _CHIN_Y + 4, "под глухим воротом маске идти некуда"


def test_neck_ratio_governs_only_the_geometric_fallback(image, stub):
    """
    `neck_ratio` — параметр запасного пути. Там, где разметки нет, глубину полосы
    шеи назначить больше нечем; там, где есть, её знает модель, и число не
    участвует. Проверяется именно это разделение: на пути без разметки оно
    работает.
    """
    stub(parsed=None)
    tight = _build(image, neck=0.0).mask
    stub(parsed=None)
    deep = _build(image, neck=0.6).mask

    assert _reach(deep) > _reach(tight)


def test_edges_are_feathered(image, stub):
    """
    Жёсткий край маски даёт видимую границу генерации ровно по своей линии.
    Растушёвка — это полутона, и они обязаны в маске быть.
    """
    stub()

    mask = _build(image, feather=0.15).mask

    assert ((mask > 10) & (mask < 245)).any(), "спад 255 → 0 обязан быть плавным"


def test_dilation_grows_the_area(image, stub):
    stub()
    narrow = int((_build(image, dilate=0.02).mask > 127).sum())
    stub()
    wide = int((_build(image, dilate=0.30).mask > 127).sum())

    assert wide > narrow


def test_the_neighbour_on_the_spread_is_left_alone(image, stub):
    """
    На развороте персонаж не один. Маска берёт только те компоненты разметки,
    что связаны с найденным лицом, — перерисовывать причёску соседа мы не
    нанимались.
    """
    stub(parsed=_parsed(extra_head=(360, 200)))

    mask = _build(image, dilate=0.05, feather=0.02).mask

    assert mask[_CENTRE[1], _CENTRE[0]] == 255
    assert mask[200, 360] == 0, "вторая голова обязана остаться нетронутой"


def test_without_parsing_the_head_falls_back_to_an_ellipse(image, stub):
    """
    Веса разметки необязательны. Без них форму причёски взять неоткуда, и
    область головы описывается эллипсом по сетке лица: грубее, но рабоче.
    """
    stub(parsed=None)

    result = _build(image)

    assert result.meta["source"] == "ellipse"
    assert result.mask[_CENTRE[1], _CENTRE[0]] == 255


def test_without_landmarks_the_parsing_carries_the_geometry(image, stub):
    """
    mediapipe не находит лицо мельче ~20% кадра, а на разворотах в полный рост
    оно именно такое. Отказывать в этом случае нельзя: разметка видит персонажа
    и без сетки.
    """
    stub(points=None)

    result = _build(image)

    assert result.meta["landmarks"] is False
    assert result.meta["source"] == "parsing"
    assert result.mask[_CENTRE[1], _CENTRE[0]] == 255


def test_both_sources_silent_is_a_refusal(image, stub):
    """Ни сетки, ни разметки — единственный случай, когда отказ честнее догадки."""
    stub(points=None, parsed=None)

    with pytest.raises(NoFaceDetectedError):
        _build(image)


def test_meta_reports_how_the_mask_was_built(image, stub):
    """
    Числа уезжают в X-Swap-Meta: когда результат вышел странным, смотрят сюда
    первым делом — какой источник сработал и сколько кадра открыто модели.
    """
    stub()

    meta = _build(image).meta

    assert meta["source"] in ("parsing", "ellipse")
    assert meta["parsing"] is True and meta["landmarks"] is True
    assert meta["face_height"] > 0
    assert 0 < meta["open_share"] < 1
    assert meta["open_px"] > 0


@pytest.mark.parametrize(
    "ratios", [{"dilate": -0.1}, {"feather": -0.1}, {"neck": -0.1}]
)
def test_negative_ratios_are_refused(image, stub, ratios):
    stub()

    with pytest.raises(InvalidImageError):
        _build(image, **ratios)
