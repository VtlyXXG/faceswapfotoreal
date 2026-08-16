"""
Выбор числа шагов диффузии по размеру лица.

Замер на 21 кадре: шестнадцать шагов дают прибыль по обеим осям сразу там, где
голова мелкая (резкость +0.09 при неупавшем сходстве), и вредят там, где она
крупная — на развороте резкость улетает за единицу, а сходство валится
монотонно. Разбор чисел — в докстринге `server.demo_steps`.
"""
from __future__ import annotations

import pytest

import server

# Стендовые шаблоны: (высота, ширина, высота лица). Числа настоящие, из меты
# прогонов, и менять их нельзя — на них держится смысл проверок
DINO2 = (768, 1408, 78.5)
DINO1 = (768, 1376, 134.5)
SPREAD = (2048, 2048, 360.3)


@pytest.mark.parametrize("template, name", [(DINO2, "dino2"), (DINO1, "dino1")])
def test_a_small_face_gets_more_steps(template, name):
    """Мелколицые кадры — те самые, где вклейка выходит вдвое мягче живописи."""
    height, width, face = template

    steps, meta = server.demo_steps(face, height, width, None)

    assert steps == server.FLUX_STEPS_SMALL_FACE, f"{name}: {meta}"
    assert meta["steps_reason"] == "мелкое лицо"


def test_a_large_face_keeps_the_default():
    """
    Развороту шаги противопоказаны, и это замер, а не осторожность.

    На spread_08 шестнадцать шагов подняли резкость до 1.415 — голова стала
    резче окружения, чем была у художника, — и уронили сходство с 0.753 до
    0.700. Двадцать восемь довели до 1.739 и 0.671.
    """
    height, width, face = SPREAD

    steps, meta = server.demo_steps(face, height, width, None)

    assert steps == server.FLUX_STEPS
    assert meta["steps_reason"] == "лицо крупное"


def test_the_face_is_measured_where_the_model_works():
    """
    Лицо меряется в ГЕНЕРАЦИИ, а не в шаблоне, и разница решает исход.

    Кадр крупнее потолка FLUX уезжает в генерацию уменьшенным. Здесь лицо в
    шаблоне выше порога, а после уменьшения — ниже, и правило обязано увидеть
    второе: мягкой окажется именно та голова, которую рисовала модель.
    """
    steps, meta = server.demo_steps(210.0, 2048, 2048, None)

    assert meta["face_px_generated"] < 200.0 < 210.0
    assert steps == server.FLUX_STEPS_SMALL_FACE, meta


def test_the_generated_face_size_is_reported():
    """Число, по которому принято решение, обязано быть видно в мете."""
    height, width, face = SPREAD

    _steps, meta = server.demo_steps(face, height, width, None)

    # 360.3 в шаблоне 2048x2048 -> 1840 в генерации -> 323.7
    assert meta["face_px_generated"] == pytest.approx(323.7, abs=0.5)


def test_an_explicit_request_is_never_overridden():
    """
    Названное вызывающим число уважается, каким бы ни было лицо.

    На этом держатся стендовые прогоны: `render_batch.py` передаёт шаги явно, и
    правило не должно молча подменять их — иначе сравнивать станет нечего.
    """
    for face in (78.5, 360.3):
        steps, meta = server.demo_steps(face, 768, 1408, 28)
        assert steps == 28
        assert meta["steps_reason"] == "задано в запросе"
        assert "face_px_generated" not in meta


def test_the_threshold_belongs_to_the_large_side():
    """Ровно на пороге лицо считается крупным: граница включена в «не мелкое»."""
    height = width = 1024
    face = server.FLUX_SMALL_FACE_PX  # кадр под потолком, масштаб единичный

    steps, _meta = server.demo_steps(face, height, width, None)

    assert steps == server.FLUX_STEPS
    assert server.demo_steps(face - 1, height, width, None)[0] == server.FLUX_STEPS_SMALL_FACE


def test_the_request_still_accepts_no_steps_at_all():
    """
    Пустое поле — это «решай сам», а не ошибка разбора.

    Раньше умолчанием стояла восьмёрка, и «не указали» было неотличимо от
    «указали восемь»; правило при таком умолчании не сработало бы ни разу.
    """
    request = server.DemoRequest(base_image="x", donor_photo="y")

    assert request.steps is None

    assert server.DemoRequest(base_image="x", donor_photo="y", steps=12).steps == 12
    with pytest.raises(ValueError):
        server.DemoRequest(base_image="x", donor_photo="y", steps=0)
