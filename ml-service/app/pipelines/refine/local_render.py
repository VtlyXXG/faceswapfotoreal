"""
Рабочая стратегия: свой GPU-сервер, `POST /v1/demo-render`. Ни fal, ни ключей.

Шлём ровно две картинки — шаблон и СЫРОЕ фото заказчика, — и получаем готовый
разворот. Внутри сервера: FLUX.2 рисует кадр с новой головой, LaMa стирает
старую по силуэту, `transplant()` сажает голову в шаблон и согласует тон. Всё,
что вне головы, приезжает из шаблона побитово.

ПОЧЕМУ ЗДЕСЬ НЕТ НИКАКОЙ ЛОКАЛЬНОЙ ПОДГОТОВКИ. Маску, силуэт, геометрию и кроп
донора GPU-сервер считает сам — тем же кодом `app.pipelines`, который он
импортирует напрямую (`GPU_MLSERVICE_PATH`). Посчитать это ещё раз здесь значило
бы завести второй источник тех же чисел, и разойтись они могли бы молча: маска,
собранная по одним долям, и силуэт, собранный по другим, дают срезанную голову
без единой ошибки в логах. Поэтому `request.mask` не читается вовсе (профиль
ставит `needs_mask=False`, и пайплайн его не строит), а фотография уходит теми
же байтами, что прислал заказчик.

ПОЧЕМУ НЕ `/v1/template-render` И НЕ `HostedFluxPuLIDBackend`. Тот контракт
(маска, кроп 112x112, 512-мерный вектор личности, промпты, fidelity, start_step)
описывает ранний путь на SDXL + IP-Adapter. Он требует от вызывающей стороны
собрать полпайплайна и держать его в согласии с сервером. Демо-контракт делает
ту же работу двумя полями, поэтому взят он.

СИНХРОННО, а не через пул: заказ и так выполняется фоновой задачей Node, и
асинхронность здесь добавила бы событийный цикл ради одного запроса.
"""

from __future__ import annotations

import base64
from typing import Any

import requests

from app.config import settings
from app.core.errors import InvalidImageError, MLServiceError, NoFaceDetectedError
from app.core.logging import get_logger
from app.pipelines.refine.base import RefineRequest, RefineResult, register
from app.pipelines.refine.profiles import DEMO_RENDER, RefineProfile

log = get_logger(__name__)

# Путь на сервере. Отдельной настройки у него нет намеренно: адрес сервиса и
# имя его эндпоинта — разные вещи, и переопределять второе значит менять
# контракт, а не конфигурацию
PATH = "/v1/demo-render"

# Сколько текста ответа тащить в детали ошибки. Целиком нельзя: при 500 сервер
# может вернуть трейсбек на десятки килобайт, и он уедет в JSON ошибки Node
_ERROR_TEXT_LIMIT = 300


class RenderNotConfiguredError(MLServiceError):
    """
    Адрес GPU-сервера не задан.

    503, а не 500: сервис исправен, но недонастроен — ровно то же различие, что
    у FAL_NOT_CONFIGURED. Node по 503 отключает иллюстрации, а не помечает заказ
    испорченным.
    """

    status_code = 503
    code = "RENDER_NOT_CONFIGURED"


class RenderRequestFailedError(MLServiceError):
    """Сервер не ответил, ответил ошибкой или прислал не то."""

    status_code = 502
    code = "RENDER_REQUEST_FAILED"


def base_url() -> str:
    """Адрес GPU-сервера без хвостового слэша. Пусто — не настроен."""
    return (settings.render_base_url or "").strip().rstrip("/")


def configured() -> bool:
    return bool(base_url())


class LocalRenderRefiner:
    """Перенос лица своим GPU-сервером: две картинки на вход, разворот на выход."""

    name = "face_swap"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        url = base_url()
        if not url:
            raise RenderNotConfiguredError(
                "Не задан ML_RENDER_BASE_URL — адрес GPU-сервера, на котором "
                "считается генерация",
                {"strategy": self.name, "setting": "ML_RENDER_BASE_URL"},
            )

        if request.expression:
            # Мимику этому пути задать нечем: выражение остаётся от персонажа
            # шаблона — промпт прямо запрещает его менять. Молча проглотить
            # параметр нельзя, иначе заказ уйдёт с ощущением, что эмоция учтена
            log.warning(
                "мимика на этой стратегии не действует: выражение остаётся от шаблона",
                extra={"strategy": self.name, "expression": request.expression[:80]},
            )

        schema = profile.payload
        if schema.name != DEMO_RENDER.name:
            # Профиль собран под чужой эндпоинт — например, остался фейссвопным
            # после ML_REFINE_PAYLOAD. Дешевле упасть здесь, чем отправить
            # 30 МБ и получить 422 с той стороны
            raise InvalidImageError(
                "Схема запроса не для GPU-сервера: он принимает две картинки в base64",
                {
                    "strategy": self.name,
                    "payload": schema.name,
                    "expected": DEMO_RENDER.name,
                },
            )

        payload: dict[str, Any] = {
            schema.image_field: base64.b64encode(request.target).decode("ascii"),
            schema.identity_field: base64.b64encode(request.identity).decode("ascii"),
        }
        if schema.format_field:
            payload[schema.format_field] = (
                "jpg" if request.output_format in {"jpg", "jpeg"} else "png"
            )

        target = f"{url}{PATH}"
        try:
            response = requests.post(target, json=payload, timeout=settings.render_timeout_s)
        except requests.RequestException as exc:
            raise RenderRequestFailedError(
                "GPU-сервер недоступен",
                {"strategy": self.name, "url": target, "error": type(exc).__name__},
            ) from exc

        if response.status_code == 422:
            # Сервер не нашёл персонажа на шаблоне или лица на фотографии. Это
            # тот же случай, что и молчаливый отказ фейссвопа, и код у него
            # обязан быть тот же: 422 чинится другим фото, а не повтором
            raise NoFaceDetectedError(
                "GPU-сервер не нашёл лица: на развороте или на фотографии",
                {
                    "strategy": self.name,
                    "url": target,
                    "detail": response.text[:_ERROR_TEXT_LIMIT],
                },
            )
        if response.status_code >= 400:
            raise RenderRequestFailedError(
                f"GPU-сервер ответил {response.status_code}",
                {
                    "strategy": self.name,
                    "url": target,
                    "status_code": response.status_code,
                    "detail": response.text[:_ERROR_TEXT_LIMIT],
                },
            )

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RenderRequestFailedError(
                "GPU-сервер вернул не JSON",
                {"strategy": self.name, "url": target,
                 "detail": response.text[:_ERROR_TEXT_LIMIT]},
            ) from exc

        image = body.get("image")
        if not image:
            raise RenderRequestFailedError(
                "В ответе GPU-сервера нет картинки",
                {"strategy": self.name, "url": target, "keys": sorted(body)},
            )

        try:
            data = base64.b64decode(image, validate=True)
        except (ValueError, TypeError) as exc:
            raise RenderRequestFailedError(
                "Картинка в ответе GPU-сервера не декодируется из base64",
                {"strategy": self.name, "url": target},
            ) from exc

        # Мета сервера едет как есть, под своим ключом: там числа пересадки
        # (skin_gamma, erase_px, matched, steps, face_height), и разбирать их
        # здесь незачем — они уходят в X-Swap-Meta и читаются глазами при
        # разборе кадра
        remote = body.get("meta") or {}
        meta = {
            **profile.report(),
            "render_url": target,
            "mime_type": body.get("encoding", "image/png"),
            "render": remote,
        }
        log.info(
            "разворот собран на GPU-сервере",
            extra={
                "strategy": self.name,
                "steps": remote.get("steps"),
                "total_s": remote.get("total_s"),
                "queue_wait_s": remote.get("queue_wait_s"),
            },
        )
        return RefineResult(image=data, meta=meta)


register(LocalRenderRefiner())
