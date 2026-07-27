"""
Транспорт до fal.ai — и только он.

Модуль знает, как положить файл в CDN, как дождаться инференса и как забрать
результат. Чего он не знает: какой эндпоинт вызывается, с какими аргументами и
зачем. Это решает стратегия из `refine/` — она получает профиль с числами и
собирает схему запроса сама.

Раньше здесь же лежала и схема, и strength, и промпт. Разделение понадобилось,
когда подходов к стилизации стало больше одного: у инпейнтинга с ControlNet и у
проброса лицевых эмбеддингов общего ровно столько, сколько в этом файле, —
загрузка, subscribe, скачивание.

Ошибки транспорта заворачиваются в FalError: без обёртки httpx-исключение
улетело бы наружу как 500 text/plain и сломало JSON-контракт с Node.js API.
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


def client():
    """Клиент fal. Отсутствие ключа — состояние окружения, а не дефект запроса."""
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


def upload(client, data: bytes, content_type: str) -> str:
    """
    Кладёт изображение в CDN fal и возвращает ссылку.

    Data URI тоже поддерживается, но обложки весят 25-30 МБ: в base64 это
    ~40 МБ в теле запроса на каждый файл, и fal прямо не рекомендует такой
    способ для файлов больше нескольких килобайт.

    Загрузка идёт до инференса и падает первой: отказ авторизации, исчерпанный
    баланс и обрыв сети приходят именно сюда.
    """
    try:
        return client.upload(data, content_type)
    except Exception as exc:  # noqa: BLE001 — сеть, авторизация, отказ хранилища
        raise FalError(
            f"Не удалось загрузить изображение в CDN fal: {exc}",
            {"content_type": content_type, "bytes": len(data)},
        ) from exc


def _extract_image(result: dict) -> dict:
    """Достаёт объект изображения: эндпоинт отдаёт список images."""
    result = result or {}
    images = result.get("images") or []
    image = images[0] if images else None

    if not image or not image.get("url"):
        raise FalError("Ответ fal не содержит изображения", {"response_keys": list(result)})
    return image


def invoke(client, endpoint: str, arguments: dict, meta: dict | None = None) -> tuple[bytes, dict]:
    """
    Вызывает эндпоинт и возвращает готовое изображение.

    Схему аргументов собирает вызывающая стратегия — здесь она проходит
    насквозь. Так и задумано: у разных подходов к стилизации схемы разные, а
    транспорт один.

    :param meta: что стратегия хочет видеть в метаданных вызова
    :return: (байты изображения, метаданные с seed и mime)
    """
    meta = {"model": endpoint, **(meta or {})}
    log.info("запрос к fal", extra={**meta, "arguments": sorted(arguments)})

    try:
        result = client.subscribe(endpoint, arguments=arguments, with_logs=False)
    except Exception as exc:  # noqa: BLE001 — любая ошибка транспорта или модели
        raise FalError(
            f"Замена лица на fal не выполнена: {exc}",
            {"model": endpoint},
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
