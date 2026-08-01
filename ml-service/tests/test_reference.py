"""
Подготовка референса: приведение масштаба лица к масштабу шаблона.

Проверяется арифметика отдаления, а не детектор: mediapipe в тестах не
запускается, и замер лица приходит заглушкой. Смысл проверок в том, что доля
лица на холсте после обработки становится РАВНА шаблонной — именно из-за
расхождения этих долей генерация и рисовала голову крупнее маски.
"""

import numpy as np
import pytest

from app.pipelines import head_mask, reference
from app.utils.image import decode_image, encode_image

_SIZE = 400
_FACE_PX = 200.0  # высота лица на фотографии-заглушке: половина кадра


@pytest.fixture
def photo() -> bytes:
    """Фотография заказчика: крупный план, лицо в половину кадра."""
    image = np.full((_SIZE, _SIZE, 3), 90, dtype=np.uint8)
    image[100:300, 100:300] = 200  # светлый прямоугольник на месте лица
    return encode_image(image, "jpeg")[0]


@pytest.fixture
def measured(monkeypatch):
    """Подменяет замер лица: детектор в тестах не работает."""

    def _apply(face_px=_FACE_PX):
        monkeypatch.setattr(head_mask, "face_size", lambda _: face_px)

    return _apply


def _prepare(photo, target_share=0.175, target_aspect=1.0, pad_ratio=0.3, pad_max=2.5):
    return reference.prepare(
        photo,
        "image/jpeg",
        target_share=target_share,
        target_aspect=target_aspect,
        pad_ratio=pad_ratio,
        pad_max=pad_max,
    )


def _share(result: reference.Reference, face_px=_FACE_PX) -> float:
    """
    Какую долю ВЫСОТЫ холста заняло лицо после обработки.

    По высоте, а не по меньшей стороне: когда пропорции референса и шаблона
    совпали, эндпоинт масштабирует референс равномерно, и высота лица
    переносится множителем «высота выхода / высота референса».
    """
    image = decode_image(result.data)
    return face_px / float(image.shape[0])


def _aspect(result: reference.Reference) -> float:
    image = decode_image(result.data)
    return image.shape[1] / float(image.shape[0])


def test_the_face_share_is_brought_down_to_the_template(photo, measured):
    """
    Главное. Лицо на фотографии занимает 0.5 меньшей стороны, лицо персонажа на
    шаблоне — 0.175. Ровно это расхождение и заставляло модель рисовать голову
    больше маски; после обработки доли обязаны сойтись.
    """
    measured()

    result = _prepare(photo, target_share=0.25, pad_max=4.0)

    assert result.meta["reason"] == "measured"
    assert _share(result) == pytest.approx(0.25, abs=0.01)


def test_the_canvas_takes_the_aspect_of_the_template(photo, measured):
    """
    Вторая половина задачи. Фотография квадратная, разворот 1.79:1, и подгоняя
    референс под свой кадр, эндпоинт растягивает квадрат по горизонтали — лицо
    приезжает расплющенным. Совпали пропорции — подгонка стала равномерным
    масштабированием, а оно не искажает ничего.
    """
    measured()

    result = _prepare(photo, target_share=0.25, target_aspect=1.79, pad_max=8.0)

    assert _aspect(result) == pytest.approx(1.79, abs=0.01)
    # И масштаб при этом обязан сойтись тоже: одно не приносится в жертву другому
    assert _share(result) == pytest.approx(0.25, abs=0.01)


def test_a_portrait_template_widens_nothing(photo, measured):
    """
    Пропорция берётся у шаблона, а не «делаем пошире». На вертикальном шаблоне
    холст обязан стать вертикальным — на этом ловится расчёт, где ширина и
    высота перепутаны местами.
    """
    measured()

    result = _prepare(photo, target_share=0.25, target_aspect=0.7, pad_max=8.0)

    assert _aspect(result) == pytest.approx(0.7, abs=0.01)


def test_the_photograph_itself_is_not_resized(photo, measured):
    """
    Отдаление делается полем, а не масштабированием: пиксели лица обязаны
    остаться теми же. Уменьшить фотографию было бы проще, но из неё берётся
    личность, и терять на этом разрешение нельзя.
    """
    measured()

    result = _prepare(photo, target_share=0.25, pad_max=4.0)

    canvas = decode_image(result.data)
    pad_y = (canvas.shape[0] - _SIZE) // 2
    pad_x = (canvas.shape[1] - _SIZE) // 2
    original = decode_image(photo)
    inside = canvas[pad_y : pad_y + _SIZE, pad_x : pad_x + _SIZE]

    # JPEG обеих сторон: сравнение по среднему модулю разности, а не побитовое
    assert float(np.abs(inside.astype(float) - original.astype(float)).mean()) < 3.0


def test_a_photograph_already_at_the_template_scale_is_left_alone(photo, measured):
    """
    Лицо на фото не крупнее, чем на шаблоне, — поле не нужно. Байты обязаны
    уйти теми же: лишнее перекодирование стоило бы качества ни за что.
    """
    measured()

    result = _prepare(photo, target_share=0.5)

    assert result.data is photo
    assert result.meta["scale"] == 1.0


def test_the_growth_is_capped(photo, measured):
    """
    Предел роста холста. Эндпоинт ужимает референс до рабочего разрешения, и
    слишком мелкое лицо перестаёт нести личность: масштаб сойдётся, а сходство
    пропадёт. Упереться в предел законно, промолчать об этом — нет.
    """
    measured()

    result = _prepare(photo, target_share=0.05, pad_max=2.0)

    assert result.meta["scale_wanted"] == pytest.approx(10.0)
    assert result.meta["scale"] == pytest.approx(2.0)


def test_without_a_face_on_the_photo_the_padding_is_blind(photo, measured):
    """
    Фотография в профиль или в темноте: сетки не будет, вычислять нечего.
    Отказывать нельзя — кладётся поле по доле из профиля.
    """
    measured(face_px=None)

    result = _prepare(photo, pad_ratio=0.3)

    assert result.meta["reason"] == "blind"
    assert result.meta["scale"] == pytest.approx(1.6)


def test_the_preparation_can_be_switched_off(photo, measured):
    """
    `pad_max = 1.0` — выключатель целиком. Нужен для подбора: чтобы отличить
    «голова не того размера из-за референса» от «из-за всего остального».
    """
    measured()

    result = _prepare(photo, pad_max=1.0)

    assert result.data is photo
    assert result.meta["reason"] == "disabled"


def test_the_border_has_no_hard_edge(photo, measured):
    """
    Кайма размытая, а не залитая. Ровный прямоугольник заливки даёт на границе
    контур, и модель читает его как рамку, воспроизводя наравне с лицом.
    """
    measured()

    result = _prepare(photo, target_share=0.25, pad_max=4.0)
    canvas = decode_image(result.data).astype(np.int16)
    pad_x = (canvas.shape[1] - _SIZE) // 2

    # Столбец через всё поле слева от фотографии: соседние пиксели каймы не
    # могут отличаться скачком — это и означало бы жёсткую границу
    column = canvas[canvas.shape[0] // 2, : pad_x - 2, 0]
    assert int(np.abs(np.diff(column)).max()) < 40


def test_a_huge_canvas_is_fitted_back(photo, measured):
    """
    Абсолютный предел размера. Поле считается в долях сторон исходника, и от
    донора в 4K холст ушёл бы в десятки тысяч пикселей. Сжатие холста целиком
    не трогает ни долю лица, ни пропорцию: обе — отношения.
    """
    measured(face_px=_FACE_PX)

    result = _prepare(photo, target_share=0.02, target_aspect=1.79, pad_max=50.0)
    canvas = decode_image(result.data)

    assert max(canvas.shape[:2]) <= reference._MAX_CANVAS_PX
    assert result.meta["fitted"] is True
    assert _aspect(result) == pytest.approx(1.79, abs=0.01)


def test_meta_reports_the_scale_and_the_cost(photo, measured):
    """
    Числа уезжают в X-Swap-Meta. Голова не того размера — смотреть на доли;
    сходство поехало — на face_px_at_work: выравнивание пропорции расходует
    рабочий кадр эндпоинта на поле, и лицу остаётся меньше пикселей.
    """
    measured()

    meta = _prepare(photo, target_share=0.25, target_aspect=1.79, pad_max=8.0).meta

    assert meta["own_share"] == pytest.approx(0.5)
    assert meta["target_share"] == pytest.approx(0.25)
    assert meta["face_share_after"] == pytest.approx(0.25, abs=0.01)
    assert meta["aspect"] == pytest.approx(1.79, abs=0.01)
    assert meta["size"] == [1432, 800]
    # Лицо 200 px на холсте шириной 1432 при рабочем кадре 1024
    assert meta["face_px_at_work"] == pytest.approx(143, abs=2)
