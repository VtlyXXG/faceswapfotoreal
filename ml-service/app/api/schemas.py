"""Pydantic-схемы ответов. Контракт для Node.js API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceInfo(BaseModel):
    bbox: BBox
    # Сетка mediapipe не даёт ни уверенности детекции, ни пола с возрастом,
    # зато сообщает число найденных точек контура
    landmarks: int


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    service: str
    version: str
    uptime_seconds: float


class RuntimeStatus(BaseModel):
    mediapipe: bool
    opencv: bool
    fal_client: bool


class ProviderStatus(BaseModel):
    model: str
    key_present: bool
    key_env: str


class MaskStatus(BaseModel):
    detector: str
    # Доли высоты лица: сплошное поле вокруг контура и ширина растушёвки за ним
    padding_ratio: float
    feather_ratio: float


class ReadinessResponse(BaseModel):
    status: str = Field(description="ready | degraded")
    runtime: RuntimeStatus
    provider: ProviderStatus
    mask: MaskStatus
    reason: str | None = Field(default=None, description="Почему degraded")


class AnalyseResponse(BaseModel):
    faces: list[FaceInfo]
    count: int


class SwapMeta(BaseModel):
    faces_detected: int
    faces_swapped: int
    model: str | None = None
    seed: int | None = None
