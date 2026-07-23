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
    Готовность к работе: наличие весов и выбранное устройство.
    503 + degraded, если веса не найдены — Node.js API может отключить
    иллюстрации с face-swap, не падая целиком.
    """
    state = registry.status()

    missing: list[str] = []
    if not state["runtime"]["insightface"]:
        missing.append("не установлен insightface")
    if not state["detector"]["available"]:
        missing.append(f"нет весов детектора {state['detector']['name']}")
    if not state["swapper"]["available"]:
        missing.append(f"нет весов {state['swapper']['name']}")

    ready = not missing
    response.status_code = 200 if ready else 503

    return ReadinessResponse(
        status="ready" if ready else "degraded",
        device=state["device"],
        models_dir=state["models_dir"],
        runtime=state["runtime"],
        detector=state["detector"],
        swapper=state["swapper"],
        stylizer=state["stylizer"],
        reason="; ".join(missing) or None,
    )
