"""
Транспорт до fal: загрузка, вызов, разбор ответа.

Схема аргументов проверяется не здесь, а в test_refine.py — вместе со
стратегией, которая её собирает. Этому модулю всё равно, что уезжает: его дело
довезти и не потерять ошибку по дороге.
"""

import pytest

from app.pipelines import fal_api


class _FakeClient:
    """Считает загрузки и запоминает аргументы вызова модели."""

    def __init__(self, result=None):
        self.uploads: list[tuple[bytes, str]] = []
        self.arguments: dict | None = None
        self.model: str | None = None
        self._result = result or {"images": [{"url": "https://cdn/x.png"}], "seed": 7}

    def upload(self, data, content_type):
        self.uploads.append((data, content_type))
        return f"https://cdn/{len(self.uploads)}"

    def subscribe(self, model, arguments, with_logs=False):
        self.model = model
        self.arguments = arguments
        return self._result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(fal_api, "_download", lambda image: b"PNGDATA")
    return _FakeClient()


# --- Загрузка ---


def test_upload_failure_becomes_fal_error():
    """
    Загрузка идёт до инференса и падает первой. Если её не обернуть, httpx-
    исключение доходит до FastAPI как 500 text/plain, и Node.js API получает
    вместо разбираемого JSON голую строку.
    """

    class _Broken:
        def upload(self, data, content_type):
            raise RuntimeError("403 Forbidden")

    with pytest.raises(fal_api.FalError) as exc_info:
        fal_api.upload(_Broken(), b"x" * 10, "image/png")

    error = exc_info.value
    assert error.status_code == 502
    assert error.code == "FAL_REQUEST_FAILED"
    # По деталям в логе видно, какой из файлов не уехал
    assert error.details["content_type"] == "image/png"
    assert error.details["bytes"] == 10


# --- Вызов ---


def test_invoke_passes_arguments_through(client):
    """
    Аргументы собирает стратегия, транспорт их не трогает: любая «умная»
    правка здесь означала бы, что схему эндпоинта знают два места сразу.
    """
    arguments = {"image_url": "https://cdn/1", "strength": 0.5, "custom": [1, 2]}

    fal_api.invoke(client, "fal-ai/whatever", arguments)

    assert client.model == "fal-ai/whatever"
    assert client.arguments == arguments


def test_invoke_failure_becomes_fal_error(client):
    def _boom(model, arguments, with_logs=False):
        raise RuntimeError("model exploded")

    client.subscribe = _boom

    with pytest.raises(fal_api.FalError) as exc_info:
        fal_api.invoke(client, "fal-ai/whatever", {})

    assert exc_info.value.status_code == 502
    assert exc_info.value.details["model"] == "fal-ai/whatever"


def test_invoke_reports_model_seed_and_mime(client):
    _, meta = fal_api.invoke(client, "fal-ai/whatever", {}, {"strength": 0.5})

    assert meta["model"] == "fal-ai/whatever"
    assert meta["strength"] == 0.5, "метаданные стратегии проходят насквозь"
    assert meta["seed"] == 7
    assert meta["mime_type"] == "image/png"


# --- Разбор ответа ---


def test_extract_image_takes_first_of_images():
    result = {"images": [{"url": "https://cdn/a.png"}], "seed": 1}

    assert fal_api._extract_image(result)["url"] == "https://cdn/a.png"


def test_extract_image_understands_a_single_image_response():
    """
    Форм ответа две, и обе живые. Диффузия отдаёт список `images` — она умеет
    несколько вариантов за вызов; фейссвоп и реставраторы отдают одиночный
    `image`, потому что вариант у них ровно один. Не разобрать вторую форму
    значит уронить рабочий путь на РАЗБОРЕ УСПЕШНОГО ответа.
    """
    result = {"image": {"url": "https://cdn/b.png"}, "seed": 2}

    assert fal_api._extract_image(result)["url"] == "https://cdn/b.png"


@pytest.mark.parametrize(
    "response",
    [
        {"images": []},
        {},
        {"images": [{}]},  # объект без ссылки
        {"image": {}},  # одиночный объект без ссылки
        {"video": {"url": "x"}},  # форма чужого эндпоинта
    ],
)
def test_extract_image_rejects_empty_response(response):
    with pytest.raises(fal_api.FalError):
        fal_api._extract_image(response)
