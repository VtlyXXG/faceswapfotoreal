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
    # Веса семантической разметки (~16 МБ). Их отсутствие не отказ: маска
    # головы строится эллипсом по сетке лица — грубее, но рабоче
    parsing_weights: bool


class ProviderStatus(BaseModel):
    """
    Второй шаг: чем и с какими числами он выполняется.

    Поля необязательны, потому что профиль может не собраться — например, если
    в окружении задана несуществующая схема запроса или стиль. Тогда вместо
    чисел приходит profile_error, и это ровно то, что нужно видеть в
    /health/ready.
    """

    model: str | None = Field(default=None, description="Идентификатор эндпоинта")
    # Первое, на что смотреть при разборе «почему сервис не готов». Путь через
    # fal выключен по умолчанию, и тогда ни ключ, ни профиль второго шага к
    # готовности отношения не имеют: сервис к fal не обращается вовсе
    fal_enabled: bool = Field(
        default=False, description="Разрешён ли облачный путь (ML_FAL_ENABLED)"
    )
    key_present: bool
    key_env: str
    # Главная ручка пайплайна: подбирается из окружения на живом сервисе
    strength: float | None = None
    profile: str = Field(description="Имя набора гиперпараметров")
    strategy: str | None = Field(default=None, description="Подход к генерации")
    payload: str | None = Field(default=None, description="Семейство схемы запроса")
    style: str | None = Field(default=None, description="Стиль сцены в промпте")
    identity_field: str | None = Field(
        default=None,
        description="Где эндпоинт ждёт референс: отдельное поле либо массив картинок",
    )
    identity_scale: float | None = None
    # Что схема реально положит в тело запроса. Первое, на что смотрят при 422:
    # лишний ключ эндпоинт не игнорирует, а заворачивает весь запрос
    sends: list[str] = Field(default_factory=list, description="Ключи тела запроса")
    needs_mask: bool | None = Field(
        default=None, description="Строится ли маска головы: фейссвопу она не нужна"
    )
    profile_error: str | None = Field(default=None, description="Почему профиль не собрался")
    profiles: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)


class MaskStatus(BaseModel):
    """Маска головы на шаблоне: чем строится и с какими допусками."""

    detector: str
    parsing_model: str = Field(description="Где сервис ищет веса разметки")
    # Доли высоты лица персонажа: расширение за контур головы, растушёвка
    # краёв и глубина захвата шеи под подбородком
    dilate_ratio: float | None = None
    feather_ratio: float | None = None
    neck_ratio: float | None = None


class HairStatus(BaseModel):
    """
    Шаг причёски: первый из двух вызовов двухшаговой стратегии.

    Работает не всегда — только когда выбран профиль с ней. Поэтому `active`
    отдельным полем: числа в блоке есть всегда, а смысл они имеют лишь при
    active=true.
    """

    active: bool = Field(description="Идёт ли правка причёски перед заменой лица")
    endpoint: str | None = Field(default=None, description="Редактор по двум картинкам")
    payload: str | None = None
    description: str = Field(
        default="",
        description=(
            "Причёска словами, из окружения. Единственный источник для первого "
            "шага: фотография заказчика туда не уезжает"
        ),
    )
    # Доли высоты лица: расширение за контур волос, растушёвка края вклейки и
    # поле защиты вокруг лица
    dilate_ratio: float | None = None
    feather_ratio: float | None = None
    protect_ratio: float | None = None
    # Докуда маска спускается по лбу, в долях высоты лица над бровями: 0 — до
    # самых бровей, и тогда рубца на лбу нет вовсе
    forehead_ratio: float | None = None
    # Сжатие ядра лица поперёк его оси: меньше единицы — сильнее открыты виски
    # и скулы, где застревают пряди старой причёски
    core_ratio: float | None = None
    # Поле вокруг маски волос: редактору уезжает окно с головой, а не разворот
    # целиком — иначе причёска занимает проценты кадра и стричь там нечего
    crop_ratio: float | None = None
    min_changed: float | None = Field(
        default=None, description="Порог «правка состоялась», уровни 0..255"
    )


class RenderStatus(BaseModel):
    """
    Рабочий путь: свой GPU-сервер, на котором считается генерация.

    Адреса по умолчанию у него нет намеренно, поэтому `configured: false` —
    штатное состояние свежего клона и единственная причина, по которой заказ
    ответит 503 RENDER_NOT_CONFIGURED. Видно это до первого заказа, а не после
    его таймаута.
    """

    base_url: str | None = Field(default=None, description="ML_RENDER_BASE_URL")
    configured: bool = Field(description="Задан ли адрес GPU-сервера")
    path: str = Field(description="Эндпоинт генерации на сервере")
    timeout_s: int
    active: bool = Field(description="Идёт ли туда текущий профиль")


class ReadinessResponse(BaseModel):
    status: str = Field(description="ready | degraded")
    runtime: RuntimeStatus
    provider: ProviderStatus
    render: RenderStatus
    mask: MaskStatus
    hair: HairStatus
    expressions: list[str] = Field(description="Допустимые значения параметра emotion")
    reason: str | None = Field(default=None, description="Почему degraded")


class AnalyseResponse(BaseModel):
    faces: list[FaceInfo]
    count: int


# Схемы для X-Swap-Meta здесь намеренно нет: заголовок собирается в роутере из
# метаданных вызова fal, состав которых меняется вместе с параметрами модели.
# Прежняя SwapMeta не использовалась и успела разойтись с реальным заголовком.
