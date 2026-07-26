"""
Вызов замены лица на fal.ai.

Разделение ролей прежнее: этот модуль знает только про транспорт до fal —
загрузку входных изображений, схему аргументов и скачивание результата. Что
именно и по какой маске перерисовывается, решает pipeline.py.

Эндпоинт один: fal-ai/flux-kontext-lora/inpaint — image_url (обложка) +
mask_url (наша маска) + reference_image_url (фото заказчика). Второй бэкенд,
easel-ai/advanced-face-swap, был снят: он ищет лицо своим детектором и на
рисованных обложках его не видит — детектор обучен на фотографиях.

Схема аргументов зафиксирована тестами: эндпоинт отвергает лишние ключи, а
узнаётся это только после боевого прогона.
"""

from __future__ import annotations

import os

from app.config import settings
from app.core.errors import MLServiceError
from app.core.logging import get_logger

log = get_logger(__name__)


class FalError(MLServiceError):
    """Ошибка облачного инференса: неверный ключ, отказ модели, сеть."""

    status_code = 502
    code = "FAL_REQUEST_FAILED"


class FalNotConfiguredError(MLServiceError):
    """Не задан FAL_KEY — сервис не может выполнять замену лица."""

    status_code = 503
    code = "FAL_NOT_CONFIGURED"


class MaskMissingError(MLServiceError):
    """
    Инпейнтинг без маски невозможен. Это дефект вызывающего кода, а не отказ
    fal, поэтому 500 и отдельный код: до сети такой запрос доходить не должен.
    """

    status_code = 500
    code = "MASK_MISSING"


def key_present() -> bool:
    return bool(os.environ.get(settings.fal_key_env, "").strip())


def _client():
    if not key_present():
        raise FalNotConfiguredError(
            f"Не задана переменная окружения {settings.fal_key_env}",
            {"env": settings.fal_key_env},
        )
    try:
        import fal_client

        return fal_client
    except ImportError as exc:  # noqa: BLE001
        raise FalError(
            "fal-client не установлен — выполните pip install -r requirements.txt",
            {"cause": str(exc)},
        ) from exc


def _upload(client, data: bytes, content_type: str) -> str:
    """
    Кладёт изображение в CDN fal и возвращает ссылку.

    Data URI тоже поддерживается, но обложки весят 25-30 МБ: в base64 это
    ~40 МБ в теле запроса на каждый файл, и fal прямо не рекомендует такой
    способ для файлов больше нескольких килобайт.

    Загрузка идёт до инференса и падает первой: отказ авторизации, исчерпанный
    баланс и обрыв сети приходят именно сюда. Без обёртки httpx-исключение
    улетело бы наружу как 500 text/plain и сломало JSON-контракт с Node.js API.
    """
    try:
        return client.upload(data, content_type)
    except Exception as exc:  # noqa: BLE001 — сеть, авторизация, отказ хранилища
        raise FalError(
            f"Не удалось загрузить изображение в CDN fal: {exc}",
            {"content_type": content_type, "bytes": len(data)},
        ) from exc


def _arguments(*, image_url: str, mask_url: str, identity_url: str, fmt: str) -> dict:
    """
    Полная схема запроса. Ключей вне списка входных параметров эндпоинта здесь
    быть не должно: ip_adapter_scale и negative_prompt он не принимает, и
    попытка передать их заворачивает весь запрос.
    """
    return {
        "image_url": image_url,
        "mask_url": mask_url,
        "reference_image_url": identity_url,
        "prompt": settings.fal_prompt,
        "strength": settings.fal_strength,
        "guidance_scale": settings.fal_guidance_scale,
        "num_inference_steps": settings.fal_steps,
        "output_format": fmt,
    }


def _extract_image(result: dict) -> dict:
    """Достаёт объект изображения: эндпоинт отдаёт список images."""
    result = result or {}
    images = result.get("images") or []
    image = images[0] if images else None

    if not image or not image.get("url"):
        raise FalError("Ответ fal не содержит изображения", {"response_keys": list(result)})
    return image


def swap_face(
    *,
    target: bytes,
    target_mime: str,
    source: bytes,
    source_mime: str,
    mask: bytes | None = None,
    output_format: str = "png",
) -> tuple[bytes, dict]:
    """
    Переносит лицо с source на target и возвращает готовое изображение.

    :param target: обложка-шаблон (куда переносим)
    :param source: фотография заказчика (донор личности)
    :param mask: одноканальная маска PNG, белое — зона перерисовки
    :return: (байты готового изображения, метаданные вызова)
    """
    if mask is None:
        raise MaskMissingError("Инпейнтингу нужна маска, но она не построена")

    client = _client()
    fmt = "jpeg" if output_format in ("jpg", "jpeg") else "png"

    image_url = _upload(client, target, target_mime)
    identity_url = _upload(client, source, source_mime)
    mask_url = _upload(client, mask, "image/png")

    arguments = _arguments(
        image_url=image_url, mask_url=mask_url, identity_url=identity_url, fmt=fmt
    )
    meta: dict = {
        "model": settings.fal_model,
        "strength": settings.fal_strength,
        "guidance_scale": settings.fal_guidance_scale,
        "steps": settings.fal_steps,
        "output_format": fmt,
    }

    log.info("запрос замены лица к fal", extra=dict(meta))

    try:
        result = client.subscribe(settings.fal_model, arguments=arguments, with_logs=False)
    except Exception as exc:  # noqa: BLE001 — любая ошибка транспорта или модели
        raise FalError(
            f"Замена лица на fal не выполнена: {exc}",
            {"model": settings.fal_model},
        ) from exc

    image = _extract_image(result)
    content = _download(image)

    meta |= {"seed": (result or {}).get("seed")}
    return content, {**meta, "mime_type": image.get("content_type") or "image/png"}


def _download(image: dict) -> bytes:
    """Скачивает готовое изображение по ссылке из ответа fal."""
    import requests

    try:
        response = requests.get(image["url"], timeout=settings.fal_timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise FalError(f"Не удалось скачать результат: {exc}", {"url": image["url"]}) from exc

    return response.content
