"""
Вызов замены лица на fal.ai.

Разделение ролей прежнее: этот модуль знает только про транспорт до fal —
загрузку входных изображений, схему аргументов и скачивание результата. Что
именно и по какой маске перерисовывается, решает pipeline.py.

Бэкендов два, потому что связки «identity-модель + инпейнтинг по маске» на fal
не существует: и flux-pulid, и ip-adapter-face-id принимают только промпт и
фото лица, без mask_url и без базового изображения. Отсюда развилка:

  kontext  — fal-ai/flux-kontext-lora/inpaint: image_url + mask_url +
             reference_image_url. Наша маска в деле, стиль держится промптом.
  faceswap — easel-ai/advanced-face-swap: target_image + face_image_0.
             Маску не принимает, ищет лицо сам и сохраняет волосы обложки.

Транспорт у них общий, различаются только сборка arguments и разбор ответа.
"""

from __future__ import annotations

import os

from app.config import settings
from app.core.errors import MLServiceError
from app.core.logging import get_logger

log = get_logger(__name__)

# Эндпоинт faceswap принимает только эти три значения
_GENDERS = ("male", "female", "non-binary")


class FalError(MLServiceError):
    """Ошибка облачного инференса: неверный ключ, отказ модели, сеть."""

    status_code = 502
    code = "FAL_REQUEST_FAILED"


class FalNotConfiguredError(MLServiceError):
    """Не задан FAL_KEY — сервис не может выполнять замену лица."""

    status_code = 503
    code = "FAL_NOT_CONFIGURED"


class FalBackendUnknownError(MLServiceError):
    """В настройках указан бэкенд, которого нет."""

    status_code = 500
    code = "FAL_BACKEND_UNKNOWN"


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


def normalise_gender(value: str | None) -> str:
    """
    Пол донора для faceswap. Неизвестное значение не роняет запрос отказом от
    fal, а тихо становится нейтральным дефолтом.
    """
    candidate = (value or "").strip().lower()
    return candidate if candidate in _GENDERS else settings.fal_faceswap_default_gender


def _kontext_arguments(*, image_url: str, mask_url: str, identity_url: str, fmt: str) -> dict:
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


def _faceswap_arguments(*, image_url: str, identity_url: str, gender: str) -> dict:
    return {
        "target_image": image_url,
        "face_image_0": identity_url,
        "gender_0": gender,
        "workflow_type": settings.fal_faceswap_workflow,
        "upscale": settings.fal_faceswap_upscale,
    }


def _extract_image(result: dict, backend: str) -> dict:
    """
    Достаёт объект изображения. Формы ответа разные: kontext отдаёт список
    images, faceswap — одиночный image.
    """
    result = result or {}

    if backend == "kontext":
        images = result.get("images") or []
        image = images[0] if images else None
    else:
        image = result.get("image")

    if not image or not image.get("url"):
        raise FalError(
            "Ответ fal не содержит изображения",
            {"backend": backend, "response_keys": list(result)},
        )
    return image


def swap_face(
    *,
    target: bytes,
    target_mime: str,
    source: bytes,
    source_mime: str,
    mask: bytes | None = None,
    donor_gender: str | None = None,
    output_format: str = "png",
) -> tuple[bytes, dict]:
    """
    Переносит лицо с source на target и возвращает готовое изображение.

    :param target: обложка-шаблон (куда переносим)
    :param source: фотография заказчика (донор личности)
    :param mask: одноканальная маска PNG, белое — зона перерисовки. Обязательна
        для бэкенда kontext и игнорируется бэкендом faceswap
    :param donor_gender: male | female | non-binary, только для faceswap
    :return: (байты готового изображения, метаданные вызова)
    """
    backend = settings.fal_backend
    if backend not in ("kontext", "faceswap"):
        raise FalBackendUnknownError(
            f"Неизвестный бэкенд {backend!r}: допустимы kontext и faceswap",
            {"backend": backend},
        )

    client = _client()
    fmt = "jpeg" if output_format in ("jpg", "jpeg") else "png"

    image_url = _upload(client, target, target_mime)
    identity_url = _upload(client, source, source_mime)

    meta: dict = {"backend": backend, "model": settings.fal_model}

    if backend == "kontext":
        if mask is None:
            raise FalBackendUnknownError(
                "Бэкенду kontext нужна маска, но она не построена",
                {"backend": backend},
            )
        mask_url = _upload(client, mask, "image/png")
        arguments = _kontext_arguments(
            image_url=image_url, mask_url=mask_url, identity_url=identity_url, fmt=fmt
        )
        meta |= {
            "strength": settings.fal_strength,
            "guidance_scale": settings.fal_guidance_scale,
            "steps": settings.fal_steps,
            "output_format": fmt,
        }
    else:
        gender = normalise_gender(donor_gender)
        arguments = _faceswap_arguments(
            image_url=image_url, identity_url=identity_url, gender=gender
        )
        # output_format эндпоинт не принимает — формат определяет он сам,
        # фактический MIME берётся из ответа
        meta |= {
            "gender": gender,
            "workflow_type": settings.fal_faceswap_workflow,
            "upscale": settings.fal_faceswap_upscale,
        }

    log.info("запрос замены лица к fal", extra=dict(meta))

    try:
        result = client.subscribe(settings.fal_model, arguments=arguments, with_logs=False)
    except Exception as exc:  # noqa: BLE001 — любая ошибка транспорта или модели
        raise FalError(
            f"Замена лица на fal не выполнена: {exc}",
            {"backend": backend, "model": settings.fal_model},
        ) from exc

    image = _extract_image(result, backend)
    content = _download(image)

    # seed возвращает только kontext — у faceswap его в ответе нет
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
