"""
Строки журнала, ради которых всё это заводилось.

Поводом был живой случай: пользователь висел на странице шесть минут, и на
вопрос «этот заказ шёл в один проход или в два» ответить оказалось нечем —
журнал знал только код ответа. Ответ пришлось выводить из промежутков между
запросами, и он остался догадкой.

Поэтому здесь проверяется не «логирование вызвано», а состав строки: тест,
проверяющий факт вызова, прошёл бы и на строке, из которой выкинули `passes`.
"""
from __future__ import annotations

import logging

import server
from server import DemoRequest


def test_the_asked_line_carries_what_the_client_actually_sent():
    request = DemoRequest(
        donor_photo="B" * 2048, template_id="dino_pixar_real/spread_05", passes=1
    )
    line = server.order_asked(request)

    assert "source=dino_pixar_real/spread_05" in line
    assert "passes=1" in line


def test_an_unasked_passes_is_logged_as_none_not_as_the_default():
    """
    Различие, на котором стоял весь разбор.

    Старая вкладка со старым JS поля `passes` не шлёт вовсе, и сервер берёт своё
    умолчание — два прохода. Подставить двойку в журнал за клиента значит стереть
    единственный след того, что клиент ничего не просил.
    """
    line = server.order_asked(DemoRequest(donor_photo="B", template_id="book/page"))

    assert "passes=None" in line


def test_the_childs_photo_never_reaches_the_log():
    """В теле запроса детская фотография. В журнале от неё только размер."""
    photo = "СЕКРЕТ" * 1000
    line = server.order_asked(DemoRequest(donor_photo=photo, template_id="book/page"))

    assert "СЕКРЕТ" not in line
    assert "donor_kb=" in line


def test_the_done_line_carries_the_numbers_that_explain_the_wait():
    meta = {
        "template_id": "dino_pixar_real/spread_05",
        "passes": 2,
        "steps": 16,
        "steps_reason": "мелкое лицо",
        "face_px_generated": 187.8,
        "queue_wait_s": 301.4,
        "total_s": 143.2,
        "matched": True,
        "skin_matched": False,
    }
    line = server.order_done(meta)

    # Ровно те четыре числа, по которым «почему так долго» закрывается без
    # захода на машину: сколько простоял в очереди, сколько считался, и на
    # скольких проходах с каким числом шагов
    assert "queue_wait_s=301.4" in line
    assert "total_s=143.2" in line
    assert "passes=2" in line
    assert "steps=16" in line
    # Отказ подгонки тона виден в мете ответа, но не в журнале — а он объясняет
    # «почему лицо как наклейка» ровно так же, как проходы объясняют время
    assert "skin_tone=False" in line


def test_a_line_survives_meta_without_the_optional_fields():
    """
    Мета неполна на любом пути, кроме самого удачного: отказ подгонки тона не
    кладёт своих ключей вовсе. Строка обязана собраться всё равно — журнал,
    падающий на неудачном заказе, бесполезен именно там, где нужен.
    """
    line = server.order_done({"passes": 1})

    assert "passes=1" in line
    assert "tone=None" in line


def test_both_lines_reach_the_journal_through_the_message_not_extra(caplog):
    """
    ФОРМАТ ЖУРНАЛА — `%(message)s`, полей `extra` он не печатает.

    Соседние вызовы в `server.py` передают `extra={...}` и теряют его молча. Тест
    смотрит на отформатированное сообщение, а не на атрибуты записи: только так
    он поймает возврат к `extra`, который выглядит правильно и не логирует
    ничего.
    """
    request = DemoRequest(donor_photo="B", template_id="book/page", passes=1)

    with caplog.at_level(logging.INFO, logger="gpu-inference"):
        server.log.info(server.order_asked(request))

    formatted = logging.Formatter("%(message)s").format(caplog.records[0])
    assert "passes=1" in formatted
    assert "[order]" in formatted
