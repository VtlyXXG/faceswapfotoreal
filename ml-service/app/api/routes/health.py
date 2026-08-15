"""Эндпоинты состояния — точка связи с Node.js API."""

from __future__ import annotations

import time

from fastapi import APIRouter, Response

from app import __version__
from app.api.schemas import HealthResponse, ReadinessResponse
from app.pipelines import registry

router = APIRouter(tags=["health"])

_STARTED_AT = time.monotonic()


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health() -> HealthResponse:
    """Процесс жив. Веса моделей не проверяются и не загружаются."""
    return HealthResponse(
        status="ok",
        service="ml-service",
        version=__version__,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
    )


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness")
async def readiness(response: Response) -> ReadinessResponse:
    """
    Готовность к работе: зависимости на месте и ключ облачного провайдера задан.
    503 + degraded, если чего-то не хватает — Node.js API может отключить
    иллюстрации с face-swap, не падая целиком.
    """
    state = registry.status()

    missing: list[str] = []
    if not state["runtime"]["mediapipe"]:
        missing.append("не установлен mediapipe — сетку лица построить нечем")
    if not state["runtime"]["opencv"]:
        missing.append("не установлен opencv")
    # Всё, что касается fal, спрашивается только при включённом пути. По
    # умолчанию он выключен, и тогда ни отсутствие fal-client, ни отсутствие
    # ключа готовности не мешают: сервис к fal не обращается вовсе. Пока эта
    # проверка стояла безусловно, свежий клон отвечал 503 и требовал чужой
    # платный ключ, хотя работать собирался локально.
    if state["provider"]["fal_enabled"]:
        if not state["runtime"]["fal_client"]:
            missing.append("не установлен fal-client")
        if not state["provider"]["key_present"]:
            missing.append(f"не задан {state['provider']['key_env']}")
        if state["provider"]["profile_error"]:
            # Несобираемый профиль второго шага — это отказ: заказ дойдёт до
            # fal, потратит три загрузки в CDN и завернётся там же
            missing.append(state["provider"]["profile_error"])

    if state["provider"]["fal_enabled"] and state["hair"]["active"] \
            and not state["runtime"]["parsing_weights"]:
        # Для маски ГОЛОВЫ отсутствие весов — снижение точности, для маски
        # ВОЛОС — потеря задачи: форму причёски знает только разметка, а
        # запасное кольцо вокруг лица до длинных волос попросту не дотянется
        missing.append(
            "нет весов разметки, а профиль правит причёску: форму волос определить нечем"
        )

    ready = not missing
    response.status_code = 200 if ready else 503

    # Отсутствие весов разметки в reason не попадает: без них маска головы
    # строится эллипсом по сетке лица, то есть сервис работает — грубее, но
    # работает. Видно это отдельным флагом runtime.parsing_weights.
    return ReadinessResponse(
        status="ready" if ready else "degraded",
        runtime=state["runtime"],
        provider=state["provider"],
        mask=state["mask"],
        hair=state["hair"],
        expressions=state["expressions"],
        reason="; ".join(missing) or None,
    )
