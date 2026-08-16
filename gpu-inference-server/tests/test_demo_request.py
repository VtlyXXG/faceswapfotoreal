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
