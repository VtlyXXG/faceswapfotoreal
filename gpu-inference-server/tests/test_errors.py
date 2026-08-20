"""
Отказы, которые клиент обязан понять.

Оба проверяемых здесь дефекта одинаковы по последствию: осмысленная причина
подменялась голым «Internal Server Error». На одиночном кадре это неприятно, на
книге из одиннадцати страниц — заказ, вставший непонятно почему и на какой
странице.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server


# --- Очередь -----------------------------------------------------------------


def test_waiting_too_long_is_a_queue_refusal_not_a_crash():
    """
    Перегрузка обязана выйти наружу как `QueueTimeout` (её эндпоинт отдаёт 503).

    ЭТОТ ТЕСТ ЗНАЧИМ НА БОЕВОЙ МАШИНЕ. Там Python 3.10.12, где
    `asyncio.TimeoutError` — отдельный класс, и прежний `except TimeoutError`
    не ловил ровно ничего: наружу уходил необработанный `asyncio.TimeoutError`,
    то есть 500 без объяснения. На 3.11, где классы слиты, тест проходит при
    любой из двух записей — гонять его надо и на боксе.
    """
    async def scenario():
        queue = server.GpuQueue(slots=1, max_waiting=8, wait_timeout=0.05)
        async with queue.slot():          # единственный слот занят
            async with queue.slot():      # второму ждать нечего
                pass

    with pytest.raises(server.QueueTimeout):
        asyncio.run(scenario())


def test_a_full_queue_is_refused_immediately():
    """Длина очереди — второй предел: отказ сразу, а не через десять минут."""
    async def scenario():
        queue = server.GpuQueue(slots=1, max_waiting=0, wait_timeout=60.0)
        async with queue.slot():
            pass

    with pytest.raises(server.QueueFull):
        asyncio.run(scenario())


def test_the_wait_limit_fits_under_the_proxy_timeout():
    """
    Ожидание и рендер идут в ОДНОМ HTTP-запросе, а прокси автозапуска рвёт
    соединение на 900 с. Значит предел ожидания плюс самая дорогая страница
    (обложка, 167 с) обязаны влезть в 900 — иначе клиент получит 502 от прокси
    вместо честного 503 от очереди, то есть ровно ту непонятную ошибку, ради
    которой всё это правится.
    """
    assert server.QUEUE_WAIT_TIMEOUT + 167.0 < 900.0


# --- Доменные ошибки ml-service ----------------------------------------------


class _FakeMLServiceError(Exception):
    """Форма ошибки из `ml-service/app/core/errors.py`, без самого пакета."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details: dict = {}


def test_a_domain_error_keeps_its_own_status_and_text():
    """
    `InvalidImageError` знает, что он 400 и почему. Именно это и должно доехать
    до клиента вместо «Internal Server Error».
    """
    http = server.domain_error(
        _FakeMLServiceError(400, "INVALID_IMAGE", "маска и кадр разного размера")
    )

    assert http is not None
    assert http.status_code == 400
    assert http.detail == "INVALID_IMAGE: маска и кадр разного размера"


def test_an_ordinary_failure_is_not_dressed_up_as_a_domain_error():
    """
    Настоящий сбой обязан остаться сбоем: 500 с трассировкой в журнале. Выдумать
    ему код значит спрятать поломку сервера за правдоподобным отказом.
    """
    assert server.domain_error(ValueError("что-то пошло не так")) is None
    assert server.domain_error(RuntimeError()) is None


def test_an_http_exception_is_not_mistaken_for_a_domain_error():
    """У HTTPException тоже есть `status_code` — одного поля мало для опознания."""
    assert server.domain_error(HTTPException(422, "лицо не найдено")) is None


# --- Через настоящий эндпоинт ------------------------------------------------


@pytest.fixture()
def demo_client(monkeypatch):
    """Эндпоинт демо-пути с готовой моделью и без единого обращения к карте."""
    monkeypatch.setattr(server.MODELS.flux, "_ready", True)
    return TestClient(server.app, raise_server_exceptions=False)


def _order(**extra) -> dict:
    return {"base_image": "AAAA", "donor_photo": "BBBB", **extra}


def test_a_domain_error_reaches_the_client_with_its_reason(demo_client, monkeypatch):
    """Тот самый случай: страница не собралась, и клиенту сказано, почему."""
    def boom(_request):
        raise _FakeMLServiceError(400, "INVALID_IMAGE", "кадр и маска разного размера")

    monkeypatch.setattr(server, "run_demo", boom)

    response = demo_client.post("/v1/demo-render", json=_order())

    assert response.status_code == 400
    assert response.json()["detail"] == "INVALID_IMAGE: кадр и маска разного размера"


def test_a_missing_face_keeps_its_own_code(demo_client, monkeypatch):
    """422 из ml-service не должен превратиться ни в 400, ни в 500."""
    def boom(_request):
        raise _FakeMLServiceError(422, "NO_FACE_DETECTED", "на фотографии нет лица")

    monkeypatch.setattr(server, "run_demo", boom)

    response = demo_client.post("/v1/demo-render", json=_order())

    assert response.status_code == 422
    assert response.json()["detail"].startswith("NO_FACE_DETECTED: ")


def test_a_refusal_of_the_path_itself_is_left_alone(demo_client, monkeypatch):
    """
    Отказы, сформулированные самим путём («на шаблоне не найден персонаж»),
    проходят насквозь: переписывать их нечем и незачем.
    """
    def boom(_request):
        raise HTTPException(422, "base_image: на шаблоне не найден персонаж")

    monkeypatch.setattr(server, "run_demo", boom)

    response = demo_client.post("/v1/demo-render", json=_order())

    assert response.status_code == 422
    assert response.json()["detail"] == "base_image: на шаблоне не найден персонаж"


def test_a_real_crash_stays_a_crash(demo_client, monkeypatch):
    """
    Неопознанное не маскируется: 500 без подробностей наружу и с трассировкой
    в журнале. Панель открыта всему интернету, и внутренности сервера в ответ
    не уезжают.
    """
    def boom(_request):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(server, "run_demo", boom)

    response = demo_client.post("/v1/demo-render", json=_order())

    assert response.status_code == 500
    assert "CUDA" not in response.text
