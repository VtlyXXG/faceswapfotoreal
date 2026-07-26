"""
Вызов fal: схема аргументов обоих бэкендов и обработка ошибок транспорта.

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


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "fal_backend", "pulid")

    with pytest.raises(fal_api.FalBackendUnknownError):
        _call()


# --- Бэкенд kontext ---


def test_kontext_arguments_match_endpoint_schema(monkeypatch, client):
    monkeypatch.setattr(settings, "fal_backend", "kontext")

    _call(output_format="png")

    assert client.model == settings.fal_kontext_model
    args = client.arguments
    # Три обязательных ссылки эндпоинта
    assert args["image_url"] and args["mask_url"] and args["reference_image_url"]
    assert args["prompt"] == settings.fal_prompt
    assert args["strength"] == settings.fal_strength
    assert args["num_inference_steps"] == settings.fal_steps
    # Эндпоинт не принимает negative_prompt — лишний ключ вызвал бы отказ
    assert "negative_prompt" not in args
    # ip_adapters остались в прошлой схеме
    assert "ip_adapters" not in args


def test_kontext_uploads_mask(monkeypatch, client):
    monkeypatch.setattr(settings, "fal_backend", "kontext")

    _call()

    # Обложка, фото донора и маска
    assert len(client.uploads) == 3
    assert (b"mask-bytes", "image/png") in client.uploads


def test_kontext_without_mask_fails_before_network(monkeypatch, client):
    monkeypatch.setattr(settings, "fal_backend", "kontext")

    with pytest.raises(fal_api.FalBackendUnknownError):
        _call(mask=None)

    assert client.arguments is None, "до вызова модели дойти не должно"


def test_kontext_jpeg_alias(monkeypatch, client):
    monkeypatch.setattr(settings, "fal_backend", "kontext")

    _call(output_format="jpg")

    # Эндпоинт знает только jpeg, но наружу принимаем и jpg
    assert client.arguments["output_format"] == "jpeg"


# --- Бэкенд faceswap ---


def test_faceswap_arguments_match_endpoint_schema(monkeypatch):
    monkeypatch.setattr(settings, "fal_backend", "faceswap")
    fake = _FakeClient(result={"image": {"url": "https://cdn/y.png"}})
    monkeypatch.setattr(fal_api, "_client", lambda: fake)
    monkeypatch.setattr(fal_api, "_download", lambda image: b"PNGDATA")

    _call(donor_gender="female")

    assert fake.model == settings.fal_faceswap_model
    args = fake.arguments
    assert args["target_image"] and args["face_image_0"]
    assert args["gender_0"] == "female"
    assert args["workflow_type"] == settings.fal_faceswap_workflow
    # Маску и промпт эндпоинт не принимает
    assert "mask_url" not in args
    assert "prompt" not in args


def test_faceswap_does_not_upload_mask(monkeypatch):
    """Маска бэкенду не нужна — незачем и грузить её в CDN."""
    monkeypatch.setattr(settings, "fal_backend", "faceswap")
    fake = _FakeClient(result={"image": {"url": "https://cdn/y.png"}})
    monkeypatch.setattr(fal_api, "_client", lambda: fake)
    monkeypatch.setattr(fal_api, "_download", lambda image: b"PNGDATA")

    _call(mask=None)

    assert len(fake.uploads) == 2


@pytest.mark.parametrize(
    "given,expected",
    [
        ("male", "male"),
        ("Female", "female"),
        (" non-binary ", "non-binary"),
        ("", "non-binary"),
        (None, "non-binary"),
        ("мужской", "non-binary"),
    ],
)
def test_gender_normalisation(given, expected):
    """Неизвестное значение не должно уходить в fal и ловить оттуда отказ."""
    assert fal_api.normalise_gender(given) == expected


# --- Разбор ответа ---


def test_extract_image_handles_both_shapes():
    kontext = {"images": [{"url": "https://cdn/a.png"}], "seed": 1}
    faceswap = {"image": {"url": "https://cdn/b.png"}}

    assert fal_api._extract_image(kontext, "kontext")["url"] == "https://cdn/a.png"
    assert fal_api._extract_image(faceswap, "faceswap")["url"] == "https://cdn/b.png"


@pytest.mark.parametrize(
    "response,backend",
    [
        ({"images": []}, "kontext"),
        ({}, "kontext"),
        ({"image": None}, "faceswap"),
        ({"images": [{"url": "x"}]}, "faceswap"),  # форма чужого бэкенда
    ],
)
def test_extract_image_rejects_empty_response(response, backend):
    with pytest.raises(fal_api.FalError):
        fal_api._extract_image(response, backend)


def test_meta_reports_backend_and_seed(monkeypatch, client):
    monkeypatch.setattr(settings, "fal_backend", "kontext")

    _, meta = _call()

    assert meta["backend"] == "kontext"
    assert meta["model"] == settings.fal_kontext_model
    assert meta["seed"] == 7
    assert meta["mime_type"] == "image/png"
