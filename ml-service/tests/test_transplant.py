"""
Пересадка головы из генерации в шаблон.

Модуль существует потому, что генеративный редактор не умеет инпейнт по маске:
он возвращает свою сцену целиком. Отсюда два обещания, которые здесь и
проверяются: голова садится НА МЕСТО шаблонной, а всё, что не голова,
возвращается из шаблона **побитово**.

mediapipe и разметка в тестах не запускаются — обе приходят заглушками.
Проверяется арифметика подобия и логика вклейки, а не качество детектора.
"""

import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.pipelines import head_mask, parsing, transplant

_SHAPE = (400, 400)
_TEMPLATE_CENTRE = (200, 180)
_HEAD_AXES = (60, 90)

# Метка в углу кадра: по ней заглушка отличает шаблон от генерации. Обе картинки
# иначе неразличимы, а сетку им надо отдавать разную — в этом весь смысл модуля.
_MARK_TEMPLATE = 10
_MARK_GENERATED = 20


def _frame(mark: int, centre=_TEMPLATE_CENTRE, axes=_HEAD_AXES, head_value=200) -> np.ndarray:
    """
    Кадр с головой, отличимой от фона.

    Голова закрашивается своим тоном не для красоты: на одноцветном кадре
    вклейка ничего не меняет, и проверка «шаблон вне головы цел» проходила бы
    даже у модуля, который не делает вообще ничего.
    """
    import cv2

    image = np.full((*_SHAPE, 3), 120, dtype=np.uint8)
    cv2.ellipse(image, centre, axes, 0, 0, 360, (head_value,) * 3, -1)
    image[0, 0] = mark
    return image


def _parsed(centre, axes=_HEAD_AXES) -> parsing.Parsed:
    """Разметка-заглушка: голова, под ней открытая кожа, ниже ткань."""
    import cv2

    face = np.zeros(_SHAPE, dtype=np.uint8)
    hair = np.zeros(_SHAPE, dtype=np.uint8)
    skin = np.zeros(_SHAPE, dtype=np.uint8)
    clothes = np.zeros(_SHAPE, dtype=np.uint8)

    cv2.ellipse(hair, centre, axes, 0, 0, 360, 255, -1)
    # Лицо — доля от головы, а не вычет фиксированных пикселей: на мелкой голове
    # вычет уходит в минус, и заглушка падает там, где проверять надо отказ
    cv2.ellipse(face, centre, (max(1, int(axes[0] * 0.67)), max(1, int(axes[1] * 0.67))),
                0, 0, 360, 255, -1)
    left, right = centre[0] - axes[0], centre[0] + axes[0]
    chin = centre[1] + axes[1]
    skin[chin : chin + 40, left:right] = 255
    clothes[chin + 40 :, left:right] = 255
    return parsing.Parsed(face=face, hair=hair, skin=skin, clothes=clothes)


@pytest.fixture
def scene(monkeypatch, mesh):
    """
    Пара «шаблон и генерация» с управляемой геометрией головы на каждой.

    Возвращает фабрику: она ставит заглушки сетки и разметки так, чтобы каждая
    из двух картинок отдавала СВОЮ голову, и отдаёт готовые кадры.
    """

    def build(gen_centre=_TEMPLATE_CENTRE, gen_scale=1.0, gen_angle=0.0, gen_axes=None):
        # Разметка головы масштабируется вместе с сеткой: иначе кадр, где лицо
        # впятеро мельче, оставался бы с головой прежнего размера, и мерка по
        # габариту головы не увидела бы никакой разницы
        if gen_axes is None:
            gen_axes = (int(_HEAD_AXES[0] * gen_scale), int(_HEAD_AXES[1] * gen_scale))
        template = _frame(_MARK_TEMPLATE, head_value=200)
        generated = _frame(_MARK_GENERATED, gen_centre, gen_axes, head_value=60)
        meshes = {
            _MARK_TEMPLATE: mesh(centre=_TEMPLATE_CENTRE, scale=1.0),
            _MARK_GENERATED: mesh(centre=gen_centre, scale=gen_scale, angle=gen_angle),
        }
        parses = {
            _MARK_TEMPLATE: _parsed(_TEMPLATE_CENTRE),
            _MARK_GENERATED: _parsed(gen_centre, gen_axes),
        }

        # Силуэт головы для ВЫРОВНЕННОЙ генерации: та же геометрия лица, что у
        # шаблона (подобие её туда и посадило), но другая форма причёски. Из
        # этой разницы и берётся зона-сирота — место, которое шаблонная маска
        # накрывает, а новая голова нет. Совпади силуэты, дефекта бы не было, и
        # заглушка проверяла бы не то.
        aligned_parsed = _parsed(_TEMPLATE_CENTRE, (int(_HEAD_AXES[0] * 0.6), _HEAD_AXES[1]))

        def pick(table, fallback):
            def choose(image):
                # Кадров на самом деле три: шаблон, генерация и выровненная
                # генерация. Метку последняя не несёт — варп её сдвигает
                return table.get(int(image[0, 0, 0]), fallback)

            return choose

        monkeypatch.setattr(head_mask, "try_landmarks", pick(meshes, meshes[_MARK_TEMPLATE]))
        monkeypatch.setattr(parsing, "parse", pick(parses, aligned_parsed))
        return template, generated, meshes

    return build


# --- подобие ----------------------------------------------------------------


def _geometry(chin, up=(0.0, -1.0), face_height=80.0, extent=None):
    return {"chin": np.array(chin, dtype=np.float64), "up": np.array(up, dtype=np.float64),
            "face_height": face_height, "head_extent": extent}


def test_identical_heads_give_the_identity_transform():
    same = _geometry((200, 240))
    matrix = transplant.similarity(same, same)
    assert np.allclose(matrix, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-9)


def test_chin_of_the_generation_lands_exactly_on_the_chin_of_the_template():
    """
    Подбородок — неподвижная точка подобия, и это не деталь реализации.

    Маска строится от подбородка вниз (полоса шеи), и промах здесь уводит стык
    головы с телом — то самое место, которое видно на печати в первую очередь.
    """
    source = _geometry((100, 300), face_height=60.0)
    target = _geometry((250, 200), face_height=90.0)
    matrix = transplant.similarity(source, target)

    moved = matrix @ np.array([100.0, 300.0, 1.0])
    assert np.allclose(moved, [250.0, 200.0], atol=1e-6)


def test_scale_comes_from_the_ratio_of_sizes():
    matrix = transplant.similarity(
        _geometry((0, 0), face_height=50.0), _geometry((0, 0), face_height=100.0))
    scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
    assert scale == pytest.approx(2.0, abs=1e-9)


def test_rotation_is_the_angle_between_the_head_axes():
    """Ось генерации наклонена на 30°, шаблонная вертикальна — подобие обязано её выпрямить."""
    radians = np.radians(30.0)
    tilted = (float(np.sin(radians)), -float(np.cos(radians)))
    matrix = transplant.similarity(_geometry((0, 0), up=tilted), _geometry((0, 0)))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    assert abs(angle) == pytest.approx(30.0, abs=0.5)


# --- выбор мерки масштаба ----------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("face", 2.0), ("head", 4.0), ("blend", np.sqrt(8.0))],
)
def test_scale_mode_selects_the_ruler(monkeypatch, mode, expected):
    monkeypatch.setattr(transplant, "SCALE_MODE", mode)
    scale = transplant._scale(
        _geometry((0, 0), face_height=50.0, extent=25.0),
        _geometry((0, 0), face_height=100.0, extent=100.0),
    )
    assert scale == pytest.approx(expected, rel=1e-9)


def test_without_the_head_extent_the_scale_falls_back_to_the_face(monkeypatch):
    """
    Разметки может не быть — тогда габарит головы измерить нечем.

    Откат обязан быть молчаливым и рабочим: мерка по лицу хуже, но кадр она
    соберёт, а отказ здесь означал бы, что без весов разметки не работает вообще
    ничего.
    """
    monkeypatch.setattr(transplant, "SCALE_MODE", "head")
    scale = transplant._scale(
        _geometry((0, 0), face_height=50.0, extent=None),
        _geometry((0, 0), face_height=100.0, extent=None),
    )
    assert scale == pytest.approx(2.0)


# --- габарит головы ----------------------------------------------------------


def test_head_extent_measures_the_head_and_ignores_the_body(monkeypatch, mesh):
    """
    В габарит идут только волосы и лицо.

    Кожа шеи и груди — не голова, и включать её значит мерить глубину выреза
    футболки вместо размера черепа.
    """
    image = _frame(_MARK_TEMPLATE)
    points = mesh(centre=_TEMPLATE_CENTRE)

    monkeypatch.setattr(parsing, "parse", lambda _: _parsed(_TEMPLATE_CENTRE))
    narrow = transplant.head_extent(image, points)

    wide = _parsed(_TEMPLATE_CENTRE)
    wide.skin[:] = 255  # кожи стало во весь кадр, головы это не касается
    monkeypatch.setattr(parsing, "parse", lambda _: wide)
    assert transplant.head_extent(image, points) == pytest.approx(narrow, rel=1e-6)


def test_head_extent_grows_with_the_head(monkeypatch, mesh):
    image = _frame(_MARK_TEMPLATE)
    points = mesh(centre=_TEMPLATE_CENTRE)

    monkeypatch.setattr(parsing, "parse", lambda _: _parsed(_TEMPLATE_CENTRE, (60, 90)))
    small = transplant.head_extent(image, points)
    monkeypatch.setattr(parsing, "parse", lambda _: _parsed(_TEMPLATE_CENTRE, (120, 180)))
    big = transplant.head_extent(image, points)

    # Площадь эллипса вчетверо — корень из неё вдвое
    assert big / small == pytest.approx(2.0, rel=0.05)


def test_without_parsing_the_head_extent_is_unknown(monkeypatch, mesh):
    monkeypatch.setattr(parsing, "parse", lambda _: None)
    assert transplant.head_extent(_frame(_MARK_TEMPLATE), mesh()) is None


# --- вклейка -----------------------------------------------------------------


def test_everything_that_is_not_the_head_survives_bit_for_bit(scene):
    """
    Главное обещание модуля, ради которого он и написан.

    Шаблон нарисован художником и обязан дойти до печати без изменений; сырая
    генерация расходится с ним на два десятка уровней по ВСЕМУ кадру. Проверяется
    не «почти совпадает», а побитовое равенство — и не в одной точке, а везде за
    пределами окна головы.

    Окно берётся заведомо шире маски и считается независимо от продакшн-кода:
    тест, повторяющий формулу, которую проверяет, не проверяет ничего.
    """
    template, generated, _ = scene(gen_centre=(210, 190), gen_scale=0.9)
    result = transplant.transplant(template, generated, 0.12, 0.10, 0.35)

    changed = np.any(result.image != template, axis=2)
    assert changed.any(), "не изменилось вообще ничего — вклейки не было"

    outside = np.ones(_SHAPE, dtype=bool)
    outside[40:350, 90:320] = False  # окно заведомо шире головы с шеей и растушёвкой
    assert not changed[outside].any(), (
        f"за пределами головы изменено {int(changed[outside].sum())} пикселей")


def test_a_head_of_a_wildly_different_size_is_refused_not_squeezed(scene):
    """
    Подобие за пределами правдоподобного означает, что редактор перерисовал
    сцену, а не голову. Подгонять такое значит отдать заказчику коллаж; отказ
    честнее, и причина видна в сообщении.
    """
    template, generated, _ = scene(gen_scale=0.2)
    with pytest.raises(InvalidImageError) as exc:
        transplant.transplant(template, generated, 0.12, 0.10, 0.35)
    assert "подобием" in str(exc.value)


def test_a_head_turned_sideways_is_refused(scene):
    template, generated, _ = scene(gen_angle=60.0)
    with pytest.raises(InvalidImageError):
        transplant.transplant(template, generated, 0.12, 0.10, 0.35)


def test_when_no_face_is_found_anywhere_it_refuses(monkeypatch, scene):
    """
    Отказ, а не тихая сборка чего попало.

    Подменяется именно `try_landmarks` целиком: у `_full_frame_landmarks` есть
    запасной путь через увеличенный кроп, и он обязан быть покрыт тоже — иначе
    тест проверял бы только первую половину функции.
    """
    template, generated, _ = scene()
    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: None)
    with pytest.raises(InvalidImageError):
        transplant.transplant(template, generated, 0.12, 0.10, 0.35)


def test_the_crop_fallback_returns_coordinates_of_the_full_frame(monkeypatch, mesh):
    """
    Координаты из кропа обязаны быть пересчитаны обратно в кадр.

    Промах здесь не заметит ни один замер качества: подобие построится по
    смещённым точкам, голова уедет ровно на смещение кропа, и виноватым будет
    выглядеть генератор.
    """
    image = _frame(_MARK_TEMPLATE)
    expected = mesh(centre=_TEMPLATE_CENTRE)

    calls = {"full": 0}

    def only_on_crop(img):
        # Полный кадр детектор «не видит» — как и живой mediapipe на развороте
        if img.shape[:2] == _SHAPE:
            calls["full"] += 1
            return None
        scale = img.shape[0] / (_SHAPE[0] * 0.6)
        return [(int((x - 100) * scale), int(y * scale)) for x, y in expected]

    monkeypatch.setattr(head_mask, "try_landmarks", only_on_crop)
    points = transplant._full_frame_landmarks(image)

    assert calls["full"] == 1, "запасной путь не должен вызываться раньше основного"
    assert points is not None
    chin = points[152]
    assert abs(chin[0] - expected[152][0]) <= 3 and abs(chin[1] - expected[152][1]) <= 3


def test_the_erase_silhouette_is_much_tighter_than_the_working_mask(monkeypatch, mesh):
    """
    Стирать по рабочей маске нельзя: она расширена на dilate + feather, и в этом
    кольце шаблонный фон ЦЕЛ. Отдать его LaMa значит поменять чёткий фон на
    догадку — на замере так и вышло, хребет за головой размывался.
    """
    image = _frame(_MARK_TEMPLATE)
    points = mesh(centre=_TEMPLATE_CENTRE)
    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: points)
    monkeypatch.setattr(parsing, "parse", lambda _: _parsed(_TEMPLATE_CENTRE))

    silhouette = transplant.head_silhouette(image, points, 80.0)
    working = head_mask.build(image, 0.12, 0.10, 0.35).mask

    assert silhouette is not None
    tight = int((silhouette > 127).sum())
    wide = int((working > 127).sum())
    assert 0 < tight < wide, f"силуэт {tight} не меньше рабочей маски {wide}"


def test_the_erase_silhouette_still_covers_the_head(monkeypatch, mesh):
    """Запас поверх силуэта маленький, но кромку антиалиасинга он обязан накрыть."""
    import cv2

    image = _frame(_MARK_TEMPLATE)
    points = mesh(centre=_TEMPLATE_CENTRE)
    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: points)
    parsed = _parsed(_TEMPLATE_CENTRE)
    monkeypatch.setattr(parsing, "parse", lambda _: parsed)

    silhouette = transplant.head_silhouette(image, points, 80.0)
    hair = np.asarray(parsed.hair) > 127
    covered = (silhouette[hair] > 127).mean()
    assert covered > 0.95, f"силуэт накрыл только {covered:.0%} волос шаблона"


def test_without_parsing_there_is_no_silhouette(monkeypatch, mesh):
    monkeypatch.setattr(parsing, "parse", lambda _: None)
    assert transplant.head_silhouette(_frame(_MARK_TEMPLATE), mesh(), 80.0) is None


def test_under_the_old_head_goes_the_plate_and_not_the_generation(scene):
    """
    Зона-сирота: шаблонная маска её накрывает, новая голова — нет.

    Ради этого места модуль и переделывался. Без стирания туда попадало
    содержимое выровненной генерации — её собственные воротник и плечи, сдвинутые
    подобием, — и на стыке шеи выходило два воротника вместо одного. Проверяется,
    что теперь там лежит именно подложка.
    """
    template, generated, _ = scene(gen_centre=(200, 150), gen_scale=0.8)
    plate = np.full_like(template, 7)  # заведомо неповторимый тон
    plate[0, 0] = template[0, 0]

    result = transplant.transplant(template, generated, 0.12, 0.10, 0.35, plate=plate)

    old = head_mask.build(template, 0.12, 0.10, 0.35).mask
    assert result.meta["orphan_px"] > 0, "сироты нет — проверять нечего"
    assert result.meta["plate"] is True

    # В глубине зоны, куда новая голова точно не дотянулась, обязан быть тон плиты
    deep = (old >= 255) & (np.all(np.abs(result.image.astype(int) - 7) <= 1, axis=2))
    assert deep.sum() > 0, "подложка не попала в кадр вовсе"


def test_the_plate_never_leaks_outside_the_template_mask(scene):
    """Стирание правит только то, что под шаблонной маской, и ни пикселем больше."""
    template, generated, _ = scene(gen_centre=(205, 185), gen_scale=0.95)
    plate = np.full_like(template, 7)
    result = transplant.transplant(template, generated, 0.12, 0.10, 0.35, plate=plate)

    outside = np.ones(_SHAPE, dtype=bool)
    outside[40:350, 90:320] = False
    assert np.array_equal(result.image[outside], template[outside])


def test_a_plate_of_the_wrong_size_is_refused(scene):
    template, generated, _ = scene()
    with pytest.raises(InvalidImageError):
        transplant.transplant(template, generated, 0.12, 0.10, 0.35,
                              plate=np.zeros((10, 10, 3), np.uint8))


def test_without_a_plate_it_still_works_but_warns(scene, caplog):
    """
    Откат обязан остаться рабочим: сервер стирания может быть недоступен, и
    ронять из-за этого весь заказ неправильно. Но молчать тоже нельзя — дефект
    на стыке шеи должен быть виден в логе, а не только на печати.
    """
    import logging

    template, generated, _ = scene(gen_centre=(200, 150), gen_scale=0.8)
    with caplog.at_level(logging.WARNING):
        result = transplant.transplant(template, generated, 0.12, 0.10, 0.35, plate=None)
    assert result.meta["plate"] is False
    assert result.meta["orphan_px"] > 0
    assert any("стирание не передано" in r.message for r in caplog.records)


def test_meta_reports_what_was_done(scene):
    """
    Отчёт читают, когда голова села мимо. Масштаб, поворот и сдвиг обязаны быть
    в нём всегда — по ним сразу видно, подобие промахнулось или генерация.
    """
    template, generated, _ = scene(gen_centre=(215, 190), gen_scale=0.95)
    meta = transplant.transplant(template, generated, 0.12, 0.10, 0.35).meta
    for field in ("scale", "angle", "shift", "mask_px", "changed_px"):
        assert field in meta
    assert 0 < meta["changed_px"] < _SHAPE[0] * _SHAPE[1]
