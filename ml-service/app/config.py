"""Конфигурация сервиса. Единственное место, читающее окружение."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ML_",
        extra="ignore",
        protected_namespaces=(),
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    log_dir: Path = Path("./logs")
    log_file: str = "app.log"
    log_error_file: str = "error.log"
    # Политика ротации совпадает с Node: сутки или превышение размера,
    # хранится log_backup_count ротированных файлов сверх активного
    log_rotate_when: str = "midnight"
    log_file_max_mb: int = 20
    log_backup_count: int = 14
    # JSON в консоль (prod) или человекочитаемый формат (dev). В файл — всегда JSON.
    log_json: bool = False

    # --- Шаг 1: коллаж локально (OpenCV + mediapipe) ---
    # Лицо переносится в шаблон преобразованием подобия и вклеивается
    # оригинальными пикселями. Доли — от высоты лица, подробности в collage.py.
    #
    # grow — запас контура вклейки за пределы лица: без него у края может
    #   остаться бровь исходного персонажа, а при низком strength модель её уже
    #   не уберёт. Выше 0.03-0.04 начинает затягивать волосы и фон с фотографии.
    collage_grow_ratio: float = 0.02
    # feather — растушёвка края вклейки. Коллаж намеренно жёсткий: ровно
    #   столько, чтобы убрать ступеньку антиалиасинга. Шов сводит второй шаг.
    collage_feather_ratio: float = 0.01
    # colour_match — доля приведения тона вклейки к лицу шаблона (среднее и
    #   разброс по каналам LAB). Геометрию не трогает, снимает разницу между
    #   светом фотостудии и светом иллюстрации: при strength ~0.2 модель сама
    #   свести освещение не успевает. 0 — цвет фотографии как есть.
    collage_colour_match: float = 0.8

    # --- Шаг 2: облачный инференс на fal.ai ---
    # Ключ читается fal-client из окружения как FAL_KEY — без префикса ML_,
    # поэтому дублируем сюда только для проверки готовности в /health/ready.
    fal_key_env: str = "FAL_KEY"
    fal_timeout_s: int = 600

    # Инпейнтинг по нашей маске MediaPipe. Личность больше не доверяется
    # модели — она приходит готовым коллажем; от эндпоинта нужны только маска и
    # управляемый strength. Единственный рабочий вариант из опробованных.
    #
    # Специализированные лицевые модели fal здесь не подходят. easel-ai/
    # advanced-face-swap ищет лицо сам и на рисованных обложках просто его не
    # находит: детектор обучен на фотографиях. Связки «identity-модель +
    # инпейнтинг по маске» на fal нет вовсе — и flux-pulid, и ip-adapter-face-id
    # это чистый text-to-image, без mask_url и без базового изображения.
    # negative_prompt эндпоинт тоже не принимает: все отрицания идут в промпт.
    fal_model: str = "fal-ai/flux-kontext-lora/inpaint"

    # Под маской уже лежит вклеенное лицо заказчика, поэтому промпт больше не
    # просит его нарисовать — он запрещает трогать геометрию и описывает ровно
    # тот верхний слой, который нужен: мазок, свет, исчезнувший шов. Отрицания
    # идут прямо в текст: negative_prompt эндпоинт не принимает.
    fal_prompt: str = (
        "The face inside the masked area is a photograph collaged onto a painted "
        "illustration. Keep its geometry exactly as it is: the same features, the same "
        "proportions, the same position and size — do not redraw, move or reshape "
        "anything, this must stay the very same person. "
        "Change only the surface: convert the photographic skin into the medium of the "
        "surrounding artwork — the same brush strokes, canvas and paper grain, colour "
        "palette, line work and level of detail — match the direction and temperature "
        "of the light of the cover, and dissolve the collage seam so that no cut-out "
        "edge is left around the face. "
        "A hand-painted portrait that belongs to the cover art: no photographic skin "
        "texture, no visible pores, no visible seam or change of style at the jaw."
    )

    # Ключевой параметр всего пайплайна.
    #
    # strength — насколько сильно зашумляется область под маской. Здесь под ней
    #   лежат оригинальные пиксели фотографии, и задача ровно одна: не дать
    #   модели до них добраться. Рабочий диапазон 0.15–0.30 — шума хватает на
    #   мазок кисти, свет и сведение шва, но не на изменение черт лица. Выше
    #   ~0.35 (см. _SAFE_STRENGTH в fal_api.py) геометрия начинает плыть, и
    #   смысл коллажа теряется; дефолт эндпоинта 0.88 перерисовал бы лицо с нуля.
    #   Ниже 0.15 мазок не набирается и шов остаётся виден.
    fal_strength: float = 0.22
    # guidance_scale — насколько строго выполняется промпт. Прежние 3.5 нужны
    #   были, чтобы перебороть фотофактуру референса; теперь фотография — это
    #   основа кадра, а не соперник, и давить нечего. 2.5 (дефолт эндпоинта) на
    #   низком strength даёт более чистый результат: выше начинают лезть
    #   артефакты контраста на считанных шагах денойза.
    fal_guidance_scale: float = 2.5
    # Шагов больше прежних 40, потому что реально исполняется лишь доля
    # strength от них: 50 × 0.22 ≈ 11 шагов денойза. При 30-40 их остаётся
    # 7-9, и переход в кольце растушёвки набраться уже не успевает.
    fal_steps: int = 50

    # --- Маска лица ---
    # Доли высоты лица: padding — сплошное поле вокруг контура, feather —
    # ширина растушёвки за ним. Подробности профиля — в mask_generator.py.
    # Маска строится по объединению контуров лица шаблона и вклейки коллажа.
    mask_padding_ratio: float = 0.06
    mask_feather_ratio: float = 0.10

    # 4K-обложки занимают 25-30 МБ; значение продублировано в .env.example
    max_upload_mb: int = 40

    allowed_origins: str = "http://localhost:3000"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
