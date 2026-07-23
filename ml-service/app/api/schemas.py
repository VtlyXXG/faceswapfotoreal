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
    det_score: float
    gender: int | None = None
    age: int | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    service: str
    version: str
    uptime_seconds: float


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    available: bool


class RuntimeStatus(BaseModel):
    insightface: bool
    onnxruntime: bool
    torch: bool


class ReadinessResponse(BaseModel):
    status: str = Field(description="ready | degraded")
    device: str
    models_dir: str
    runtime: RuntimeStatus
    detector: ModelStatus
    swapper: ModelStatus
    reason: str | None = Field(default=None, description="Почему degraded")


class AnalyseResponse(BaseModel):
    faces: list[FaceInfo]
    count: int


class SwapMeta(BaseModel):
    faces_detected: int
    faces_swapped: int
    source_face: FaceInfo | None = None
