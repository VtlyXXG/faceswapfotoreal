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
    rembg: bool


class ProviderStatus(BaseModel):
    model: str
    key_present: bool
    key_env: str
    # Главная ручка пайплайна: подбирается из окружения на живом сервисе
    strength: float


class MaskStatus(BaseModel):
    detector: str
    # Доли высоты лица: кольцо вдоль контура волос, полоса на срезе шеи,
    # защита лица и общий спад по краям зоны
    edge_ratio: float
    neck_ratio: float
    guard_ratio: float
    feather_ratio: float


class CollageStatus(BaseModel):
    segmenter_photo: str
    segmenter_cover: str
    # Скачаны ли веса сегментатора: иначе первый заказ ждёт ~176 МБ на модель
    weights_ready: bool
    colour_match: float
    erase_template_head: bool


class ReadinessResponse(BaseModel):
    status: str = Field(description="ready | degraded")
    runtime: RuntimeStatus
    provider: ProviderStatus
    mask: MaskStatus
    collage: CollageStatus
    expressions: list[str] = Field(description="Допустимые значения параметра emotion")
    reason: str | None = Field(default=None, description="Почему degraded")


class AnalyseResponse(BaseModel):
    faces: list[FaceInfo]
    count: int


# Схемы для X-Swap-Meta здесь намеренно нет: заголовок собирается в роутере из
# метаданных вызова fal, состав которых меняется вместе с параметрами модели.
# Прежняя SwapMeta не использовалась и успела разойтись с реальным заголовком.
