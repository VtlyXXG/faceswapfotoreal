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
    """
    Второй шаг: чем и с какими числами он выполняется.

    Числовые поля необязательны, потому что профиль может не собраться —
    например, если в окружении включили ControlNet при эндпоинте, который его не
    принимает. Тогда вместо чисел приходит profile_error, и это ровно то, что
    нужно видеть в /health/ready.
    """

    model: str | None = Field(default=None, description="Идентификатор эндпоинта")
    key_present: bool
    key_env: str
    # Главная ручка пайплайна: подбирается из окружения на живом сервисе
    strength: float | None = None
    profile: str = Field(description="Имя набора гиперпараметров")
    strategy: str | None = Field(default=None, description="Подход: инпейнтинг, эмбеддинги, …")
    controls: list[str] = Field(default_factory=list, description="Карты ControlNet и их веса")
    profile_error: str | None = Field(default=None, description="Почему профиль не собрался")
    profiles: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)


class MaskStatus(BaseModel):
    detector: str
    # Доли высоты лица: кольцо вдоль контура волос, полоса на срезе шеи,
    # защита лица, общий спад по краям зоны и ширина градиента
    edge_ratio: float | None = None
    neck_ratio: float | None = None
    guard_ratio: float | None = None
    feather_ratio: float | None = None
    gradient_ratio: float | None = None


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
