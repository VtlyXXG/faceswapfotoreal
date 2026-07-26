"""
Вызов fal: схема аргументов эндпоинта и обработка ошибок транспорта.

Схему фиксируем тестами намеренно. Именно на ней уже обожглись вживую: fal
принял запрос, но упал на несуществующем имени весов, и выяснилось это только
после боевого прогона. Опечатка в ключе аргумента ловится здесь бесплатно.
"""

import pytest

from app.config import settings
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
    fake = _FakeClient()
    monkeypatch.setattr(fal_api, "_client", lambda: fake)
    monkeypatch.setattr(fal_api, "_download", lambda image: b"PNGDATA")
    return fake


def _call(**overrides):
    kwargs = {
        "target": b"target-bytes",
        "target_mime": "image/png",
        "source": b"source-bytes",
        "source_mime": "image/jpeg",
        "mask": b"mask-bytes",
    }
    kwargs.update(overrides)
    return fal_api.swap_face(**kwargs)


# --- Транспорт ---


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
        fal_api._upload(_Broken(), b"x" * 10, "image/png")

    error = exc_info.value
    assert error.status_code == 502
    assert error.code == "FAL_REQUEST_FAILED"
    # По деталям в логе видно, какой из файлов не уехал
    assert error.details["content_type"] == "image/png"
    assert error.details["bytes"] == 10


# --- Схема запроса ---


def test_arguments_match_endpoint_schema(client):
    _call(output_format="png")

    assert client.model == settings.fal_model
    args = client.arguments
    # Три обязательных ссылки эндпоинта
    assert args["image_url"] and args["mask_url"] and args["reference_image_url"]
    assert args["prompt"] == settings.fal_prompt
    assert args["strength"] == settings.fal_strength
    assert args["guidance_scale"] == settings.fal_guidance_scale
    assert args["num_inference_steps"] == settings.fal_steps


@pytest.mark.parametrize("key", ["ip_adapter_scale", "ip_adapters", "negative_prompt"])
def test_arguments_carry_no_unsupported_keys(client, key):
    """
    Ключей вне схемы эндпоинта быть не должно: лишний параметр он не игнорирует,
    а заворачивает весь запрос. Баланс «личность ↔ стиль» здесь задаётся
    strength и guidance_scale, ip-адаптера у этой модели нет.
    """
    _call()

    assert key not in client.arguments


def test_uploads_target_source_and_mask(client):
    _call()

    assert len(client.uploads) == 3
    assert (b"mask-bytes", "image/png") in client.uploads


def test_missing_mask_fails_before_network(client):
    with pytest.raises(fal_api.MaskMissingError) as exc_info:
        _call(mask=None)

    assert exc_info.value.status_code == 500
    assert client.uploads == [], "до загрузки в CDN дойти не должно"
    assert client.arguments is None, "до вызова модели дойти не должно"


def test_jpeg_alias(client):
    _call(output_format="jpg")

    # Эндпоинт знает только jpeg, но наружу принимаем и jpg
    assert client.arguments["output_format"] == "jpeg"


# --- Разбор ответа ---


def test_extract_image_takes_first_of_images():
    result = {"images": [{"url": "https://cdn/a.png"}], "seed": 1}

    assert fal_api._extract_image(result)["url"] == "https://cdn/a.png"


@pytest.mark.parametrize(
    "response",
    [
        {"images": []},
        {},
        {"images": [{}]},  # объект без ссылки
        {"image": {"url": "x"}},  # форма чужого эндпоинта
    ],
)
def test_extract_image_rejects_empty_response(response):
    with pytest.raises(fal_api.FalError):
        fal_api._extract_image(response)


def test_meta_reports_model_params_and_seed(client):
    _, meta = _call()

    assert meta["model"] == settings.fal_model
    assert meta["strength"] == settings.fal_strength
    assert meta["guidance_scale"] == settings.fal_guidance_scale
    assert meta["steps"] == settings.fal_steps
    assert meta["seed"] == 7
    assert meta["mime_type"] == "image/png"
