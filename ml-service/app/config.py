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

    # --- Замена лица: облачный инференс на fal.ai ---
    # Ключ читается fal-client из окружения как FAL_KEY — без префикса ML_,
    # поэтому дублируем сюда только для проверки готовности в /health/ready.
    fal_key_env: str = "FAL_KEY"
    fal_timeout_s: int = 600

    # Инпейнтинг по нашей маске MediaPipe: личность задаётся референсом, стиль
    # иллюстрации — промптом. Единственный рабочий вариант из опробованных.
    #
    # Специализированные лицевые модели fal здесь не подходят. easel-ai/
    # advanced-face-swap ищет лицо сам и на рисованных обложках просто его не
    # находит: детектор обучен на фотографиях. Связки «identity-модель +
    # инпейнтинг по маске» на fal нет вовсе — и flux-pulid, и ip-adapter-face-id
    # это чистый text-to-image, без mask_url и без базового изображения.
    # negative_prompt эндпоинт тоже не принимает: все отрицания идут в промпт.
    fal_model: str = "fal-ai/flux-kontext-lora/inpaint"

    # Промпт решает, чем окажется результат — дорисованным лицом или вклеенной
    # фотографией. Поэтому он не описывает портрет, а формулирует задачу:
    # «перерисуй в манере обложки, сохранив личность», и явным перечислением
    # закрывает то, для чего у эндпоинта нет negative_prompt.
    fal_prompt: str = (
        "Repaint the face inside the masked area so that it belongs to this artwork. "
        "Keep the identity, facial features and proportions of the reference person, "
        "but render them entirely in the medium of the surrounding illustration: "
        "the same brush strokes, canvas and paper grain, colour palette, line work, "
        "level of detail and direction of light. "
        "Hand-painted and seamlessly blended into the cover art, "
        "not a photograph pasted on top: no photographic skin texture, no visible "
        "pores, no cut-out edge or seam around the face, no change of style at the jaw."
    )

    # Баланс «личность ↔ стиль». Оба параметра переопределяются из окружения
    # (ML_FAL_STRENGTH, ML_FAL_GUIDANCE_SCALE) — подбирать их всё равно
    # приходится глазами по конкретной обложке.
    #
    # strength — насколько сильно зашумляется область под маской. Дефолт
    #   эндпоинта 0.88 перерисовывает её почти с нуля, и мазок оригинала под
    #   новым лицом не сохраняется. 0.82 оставляет живопись обложки подложкой,
    #   по которой модель ведёт лицо; ниже ~0.7 начинает проступать исходное лицо.
    fal_strength: float = 0.82
    # guidance_scale — насколько строго выполняется промпт, то есть требование
    #   рисовать, а не вклеивать. Дефолт эндпоинта 2.5 оставляет референсу
    #   слишком много воли, и его фотографическая фактура протекает в результат.
    #   3.5 — верх рабочего диапазона Flux; за 4.0 модель начинает «гореть».
    fal_guidance_scale: float = 3.5
    # Шагов больше дефолтных 30: переход в кольце растушёвки маски набирается
    # именно на последних шагах, при 30 он остаётся заметно грубее.
    fal_steps: int = 40

    # --- Маска лица ---
    # Доли высоты лица: padding — сплошное поле вокруг контура, feather —
    # ширина растушёвки за ним. Подробности профиля — в mask_generator.py
    mask_padding_ratio: float = 0.06
    mask_feather_ratio: float = 0.10

    # 4K-обложки занимают 25-30 МБ; значение продублировано в .env.example
    max_upload_mb: int = 40
    max_faces: int = 10

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
