"""
Поля запроса демо-пути, которыми подбирают формулировку и посадку головы.

Оба появились ради одного: перебор вариантов не должен стоить перезапуска
сервиса на боксе. Отсюда и проверки — не на «поле есть», а на том, что
умолчание осталось прежним и что опечатка отказывает громко.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from server import DemoRequest

_PAIR = {"base_image": "AAAA", "donor_photo": "BBBB"}


def test_defaults_change_nothing():
    """
    Прежние вызовы обязаны работать ровно как раньше.

    None здесь не «пусто», а «не просили»: сервер подставит GPU_FLUX_PROMPT и
    SCALE_MODE. Пустая строка означала бы запрос БЕЗ текста вовсе — это другое
    состояние, и путать их значит однажды отправить в генерацию пустой промпт.
    """
    request = DemoRequest(**_PAIR)

    assert request.prompt is None
    assert request.scale_mode is None


def test_the_prompt_is_carried_as_written():
    text = "Replace the facial features and the hairstyle of the child."
    assert DemoRequest(**_PAIR, prompt=text).prompt == text


def test_an_empty_prompt_is_not_the_same_as_no_prompt():
    """Пустая строка доезжает как есть: «текста не давать» — законный опыт."""
    assert DemoRequest(**_PAIR, prompt="").prompt == ""


@pytest.mark.parametrize("mode", ["face", "head", "blend"])
def test_known_scale_modes_pass(mode):
    assert DemoRequest(**_PAIR, scale_mode=mode).scale_mode == mode


def test_scale_mode_is_case_insensitive():
    assert DemoRequest(**_PAIR, scale_mode="FACE").scale_mode == "face"


def test_a_typo_in_scale_mode_is_refused_loudly():
    """
    Неизвестный режим в ml-service просто провалился бы в ветку `head`, и кадр
    вышел бы правдоподобным, но посчитанным не тем способом, о котором просили.
    Такое ловится по числам дорого, поэтому отказ здесь.
    """
    with pytest.raises(ValidationError):
        DemoRequest(**_PAIR, scale_mode="hair")


def test_a_prompt_longer_than_the_limit_is_refused():
    """Ограничение не про вкус: промпт целиком уезжает в мету каждого кадра."""
    with pytest.raises(ValidationError):
        DemoRequest(**_PAIR, prompt="ы" * 2001)


@pytest.mark.parametrize("field", ["dilate", "feather", "neck"])
def test_mask_ratios_default_to_the_server(field):
    """None — «не просили»: сервер подставит GPU_DEMO_* и оставит боевые доли."""
    assert getattr(DemoRequest(**_PAIR), field) is None


@pytest.mark.parametrize("field", ["dilate", "feather", "neck"])
def test_mask_ratios_are_carried(field):
    assert getattr(DemoRequest(**_PAIR, **{field: 0.2}), field) == 0.2


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_a_ratio_outside_the_range_is_refused(value):
    """
    Доля больше высоты лица — почти наверняка опечатка (0.12 против 1.2), и
    молча она означала бы маску во весь кадр: пересадка накрыла бы полразворота.
    """
    with pytest.raises(ValidationError):
        DemoRequest(**_PAIR, dilate=value)


def test_a_zero_feather_is_allowed():
    """Ноль законен: жёсткая кромка — осмысленный опыт при разборе дефекта."""
    assert DemoRequest(**_PAIR, feather=0.0).feather == 0.0


def test_a_single_pass_stays_the_default():
    """
    None — прохода нет. Умолчание обязано остаться однопроходным: второй проход
    удваивает время кадра, и включаться он должен по просьбе, а не сам.
    """
    request = DemoRequest(**_PAIR)

    assert request.prompt2 is None


def test_the_second_pass_prompt_is_carried():
    text = "Replace only the facial features."
    assert DemoRequest(**_PAIR, prompt2=text).prompt2 == text


def test_the_second_pass_is_independent_of_the_first():
    """
    Второй проход можно просить и без переопределения первого: тогда первый
    идёт на умолчании сервера, второй — по заданному тексту.
    """
    request = DemoRequest(**_PAIR, prompt2="only the face")

    assert request.prompt is None
    assert request.prompt2 == "only the face"


def test_a_second_prompt_longer_than_the_limit_is_refused():
    with pytest.raises(ValidationError):
        DemoRequest(**_PAIR, prompt2="ы" * 2001)


def test_passes_defaults_to_the_server():
    """None — «не просили»: сервер подставит GPU_FLUX_PASSES, ныне два."""
    assert DemoRequest(**_PAIR).passes is None


@pytest.mark.parametrize("count", [1, 2])
def test_one_and_two_passes_are_accepted(count):
    assert DemoRequest(**_PAIR, passes=count).passes == count


@pytest.mark.parametrize("count", [0, 3])
def test_other_pass_counts_are_refused(count):
    """
    Третий проход не пробовали и мерок под него нет; ноль означал бы кадр без
    генерации вовсе. И то, и другое — почти наверняка опечатка.
    """
    with pytest.raises(ValidationError):
        DemoRequest(**_PAIR, passes=count)
