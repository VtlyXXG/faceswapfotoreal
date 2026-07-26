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
        missing.append("не установлен mediapipe — маску лица построить нечем")
    if not state["runtime"]["opencv"]:
        missing.append("не установлен opencv")
    if not state["runtime"]["fal_client"]:
        missing.append("не установлен fal-client")
    if not state["provider"]["key_present"]:
        missing.append(f"не задан {state['provider']['key_env']}")

    ready = not missing
    response.status_code = 200 if ready else 503

    return ReadinessResponse(
        status="ready" if ready else "degraded",
        runtime=state["runtime"],
        provider=state["provider"],
        mask=state["mask"],
        reason="; ".join(missing) or None,
    )
