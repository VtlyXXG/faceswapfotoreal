"""
Обратная вклейка: из генерации берётся голова, всё остальное — из шаблона.

Проверяется главное обещание безмасочного пути. Модель перерисовывает кадр
целиком, и вместе с головой у неё поедут фактура ткани, надписи и узор обоев;
шаблон нарисован художником и обязан дойти до печати без единого изменения.
Значит, вне маски результат должен совпадать с шаблоном не «на глаз», а
побитово — это и есть предмет тестов.
"""

import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.pipelines import composite

_TEMPLATE_LEVEL = 200
_GENERATED_LEVEL = 40


@pytest.fixture
def scene():
    """Шаблон, генерация другого цвета и маска-квадрат посреди кадра."""
    template = np.full((64, 64, 3), _TEMPLATE_LEVEL, dtype=np.uint8)
    generated = np.full((64, 64, 3), _GENERATED_LEVEL, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    return template, generated, mask


def test_outside_the_mask_the_template_survives_bit_for_bit(scene):
    """
    Не «почти совпадает». Фон, одежда и текст на развороте обязаны остаться теми
    же пикселями, что пришли: иначе диффузия молча переписывает шаблон.
    """
    template, generated, mask = scene

    result = composite.paste(template, generated, mask)

    outside = mask == 0
    assert np.array_equal(result.image[outside], template[outside])


def test_inside_the_mask_the_generation_wins(scene):
    template, generated, mask = scene

    result = composite.paste(template, generated, mask)

    assert tuple(result.image[32, 32]) == (_GENERATED_LEVEL,) * 3


def test_feathered_edge_blends_instead_of_cutting(scene):
    """
    Ради растушёвки всё и делается: жёсткий край дал бы вокруг головы контур,
    видимый на печати. Полутон маски обязан давать полутон результата.
    """
    template, generated, mask = scene
    mask[20:44, 20:44] = 128

    value = int(composite.paste(template, generated, mask).image[32, 32, 0])

    assert _GENERATED_LEVEL < value < _TEMPLATE_LEVEL


def test_generation_of_another_size_is_fitted_to_the_template(scene):
    """
    Эндпоинт работает около мегапикселя, а разворот приходит в 4K: совпадение
    размеров — исключение, а не правило.
    """
    template, _, mask = scene
    small = np.full((32, 32, 3), _GENERATED_LEVEL, dtype=np.uint8)

    result = composite.paste(template, small, mask)

    assert result.image.shape == template.shape
    assert result.meta["scale"] == 2.0
    assert tuple(result.image[32, 32]) == (_GENERATED_LEVEL,) * 3


def test_a_different_aspect_is_covered_and_cropped_not_stretched(scene):
    """
    Пропорция дороже кадрирования: лишняя полоса по краю всё равно уйдёт под
    маской, а растянутое лицо испортит ровно то, ради чего всё делалось.
    """
    template, _, mask = scene
    wide = np.full((32, 96, 3), _GENERATED_LEVEL, dtype=np.uint8)

    result = composite.paste(template, wide, mask)

    assert result.meta["fit"] == "cover"
    assert result.image.shape == template.shape


def test_drift_reports_that_the_generation_moved_away(scene):
    """
    Единственный доступный признак того, что модель перекадрировала сцену. Вне
    маски генерация обязана почти совпадать с шаблоном; разошлась — значит,
    голова уехала из-под маски, и вклейка могла её разрезать.
    """
    template, generated, mask = scene

    same = np.array(template)
    same[20:44, 20:44] = _GENERATED_LEVEL

    assert composite.paste(template, same, mask).meta["drift"] < 1.0
    assert composite.paste(template, generated, mask).meta["drift"] > 100.0


def test_matching_removes_the_editors_global_lift(scene):
    """
    Ровно тот механизм, которым вокруг головы заводится светлое свечение.

    Редактор возвращает кадр не в том тоне, в каком получил, — поднимает весь
    кадр разом. На целом кадре это незаметно, но берём мы из него не кадр, а
    область маски, а маска волос — КОЛЬЦО вокруг головы, выходящее на фон.
    Кольцо получается светлее фона по всему периметру, и это и есть ореол.

    Растушёвка тут бессильна по устройству: она размывает край кольца, а увод
    живёт внутри, где вес маски равен единице.
    """
    _, _, mask = scene
    template = np.full((64, 64, 3), 120, dtype=np.uint8)

    lifted = np.full((64, 64, 3), 138, dtype=np.uint8)  # весь кадр поднят на 18
    lifted[20:44, 20:44] = 60  # и внутри маски нарисована голова

    result = composite.paste(template, lifted, mask, match=True)

    assert result.image[32, 32, 0] == 42, "поднятые 18 уровней сняты и внутри маски"
    assert result.meta["match_before"] == 18.0, "мета показывает увод до поправки"
    assert result.meta["drift"] < 1.0, "после поправки вне маски расхождения нет"


def test_matching_is_a_no_op_on_a_faithful_generation(scene):
    """
    Поправка не «улучшает картинку». Не увёл редактор тон — вклейка обязана
    остаться прежней побитово, иначе она сама станет источником брака.

    Градиент, а не заливка: на ровном фоне наклон считать не по чему, и код
    уходит на путь «только сдвиг». Проверять надо тот, что с наклоном.
    """
    _, _, mask = scene
    column = np.linspace(40, 220, 64, dtype=np.uint8)
    template = np.repeat(np.tile(column, (64, 1))[..., None], 3, axis=2)

    faithful = template.copy()
    faithful[20:44, 20:44] = _GENERATED_LEVEL

    result = composite.paste(template, faithful, mask, match=True)

    assert result.meta["match_gain"] == [1.0, 1.0, 1.0]
    assert result.meta["match_bias"] == [0.0, 0.0, 0.0]
    assert tuple(result.image[32, 32]) == (_GENERATED_LEVEL,) * 3


def test_matching_gives_up_when_the_scene_itself_diverged(scene):
    """
    Поправка держится на допущении «вне маски это одна и та же сцена». Вернул
    редактор чужой кадр — допущения нет, и правильного числа у поправки тоже
    нет: прижатая к пределу прямая добавит наш увод к чужому вместо того, чтобы
    снять чужой. Отказ честнее, а причина видна по drift.
    """
    template, generated, mask = scene  # генерация расходится с шаблоном всюду

    result = composite.paste(template, generated, mask, match=True)

    assert result.meta["matched"] is False
    assert tuple(result.image[32, 32]) == (_GENERATED_LEVEL,) * 3


def test_matching_is_off_unless_asked(scene):
    """
    Умолчание не трогает остальные пути: маска головы накрывает голову целиком,
    и кольца вокруг неё, из-за которого всё написано, там попросту нет.
    """
    template, generated, mask = scene

    assert composite.paste(template, generated, mask).meta.get("matched") is None


def test_mask_of_another_size_is_refused(scene):
    """Маска не от этого кадра — дефект вызывающего кода, а не повод угадывать."""
    template, generated, _ = scene

    with pytest.raises(InvalidImageError):
        composite.paste(template, generated, np.zeros((16, 16), dtype=np.uint8))


def test_three_channel_mask_is_accepted(scene):
    """Маска ездит по пайплайну как PNG, а декодер отдаёт три канала."""
    template, generated, mask = scene

    result = composite.paste(template, generated, np.dstack([mask] * 3))

    assert tuple(result.image[32, 32]) == (_GENERATED_LEVEL,) * 3
