"""
Стирание старой причёски до отправки редактору.

Проверяется одно свойство и три обещания вокруг него.

Свойство: под маской не остаётся СТРУКТУРЫ. Не «стало светлее» и не «стало
похоже на фон» — именно структуры, потому что цепляется редактор за неё:
контрастная вытянутая линия на щеке читается им как тень скулы, то есть как
лицо, которое ему запрещено трогать. Мерится это лапласианом, и его падение —
единственная честная проверка того, что приём сработал.

Обещания: вне маски не меняется ни пикселя, ноль выключает стирание целиком, а
маска без окружения не превращается в чёрное поле.
"""

import numpy as np

from app.pipelines import erase

# Кадр с прядью: светлый фон и тёмная вытянутая линия внутри будущей маски —
# ровно та геометрия, которую редактор принимает за край челюсти.
_FACE_HEIGHT = 32.0


def _scene() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    image[30:34, 20:44] = 20

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return image, mask


def test_the_strand_loses_its_structure():
    """
    Главное. Прядь не просто светлеет — от неё не остаётся перепада, за который
    можно зацепиться. Порядок падения лапласиана и есть «сохранять нечего».
    """
    image, mask = _scene()

    erased, meta = erase.wipe(image, mask, _FACE_HEIGHT, 0.06)

    assert meta["erased"] is True
    assert meta["structure_after"] < meta["structure_before"] / 10
    assert int(erased[30:34, 20:44].min()) > 150, "тёмных пикселей пряди не осталось"


def test_the_fill_takes_the_colour_from_around_the_mask():
    """
    Почему заливка, а не одно размытие. Размытая тёмная прядь остаётся тёмным
    пятном, и модель дорисует по нему тёмные волосы — то есть ровно то, что мы
    стирали. Цвет обязан прийти из-за границы маски, где лежит настоящая сцена.
    """
    image, mask = _scene()

    erased, _ = erase.wipe(image, mask, _FACE_HEIGHT, 0.06)

    inside = erased[mask > 127].astype(np.float64)
    assert abs(float(inside.mean()) - 200.0) < 12.0, "под маской — продолжение окружения"


def test_outside_the_mask_nothing_moves():
    """
    Стирание — свойство запроса, а не шаблона. Всё, что маска не открыла, обязано
    дойти до редактора побитово: там и лицо, которое ещё ждёт фейссвоп.
    """
    image, mask = _scene()

    erased, _ = erase.wipe(image, mask, _FACE_HEIGHT, 0.06)

    outside = mask == 0
    assert np.array_equal(erased[outside], image[outside])
    assert erased is not image, "правится копия, а не кадр под собой"


def test_zero_switches_the_wiping_off():
    """Ноль возвращает прежнее поведение: редактор получает кадр как есть."""
    image, mask = _scene()

    erased, meta = erase.wipe(image, mask, _FACE_HEIGHT, 0.0)

    assert erased is image
    assert meta == {"erased": False, "reason": "off"}


def test_a_mask_without_surroundings_is_left_alone():
    """
    Заливке нужен цвет из-за границы. Маска, накрывшая окно целиком, вернула бы
    из inpaint чёрное поле — структуру мощнее исходной, да ещё и с подсказкой
    «тут темно». Отказ честнее: кадр уезжает нетронутым, причина уходит в мету.
    """
    image, _ = _scene()
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)

    erased, meta = erase.wipe(image, mask, _FACE_HEIGHT, 0.06)

    assert erased is image
    assert meta["reason"] == "no_context"


def test_an_empty_mask_costs_nothing():
    """Волос не нашлось — стирать нечего, и работы тоже никакой."""
    image, _ = _scene()

    erased, meta = erase.wipe(image, np.zeros(image.shape[:2], np.uint8), _FACE_HEIGHT, 0.06)

    assert erased is image
    assert meta["reason"] == "empty"


def test_a_thin_strand_survives_the_downscale():
    """
    Считается всё на уменьшенной копии, и это ловушка: при ближайшем соседе
    прядь шириной в пиксель выпадает из области — и остаётся в кадре нестёртой,
    то есть ровно тем, из-за чего модуль и написан. Проверяется на 4K, где
    уменьшение восьмикратное.
    """
    image = np.full((1024, 2048, 3), 200, dtype=np.uint8)
    image[500:502, 400:1600] = 20
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[400:700, 300:1700] = 255

    erased, meta = erase.wipe(image, mask, 300.0, 0.06)

    assert meta["work_scale"] < 0.3
    assert int(erased[500:502, 400:1600].min()) > 150
