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

    # Какой бэкенд обслуживает замену лица: kontext | faceswap.
    # Связки «identity-модель + инпейнтинг по маске» на fal нет: и flux-pulid,
    # и ip-adapter-face-id — чистый text-to-image без mask_url. Поэтому два
    # разных подхода, и выбор оставлен переключателем.
    #
    #   kontext  — инпейнтинг по нашей маске MediaPipe, личность задаётся
    #              референсом. Держит стиль иллюстрации через промпт, но
    #              Kontext — модель общего референса, не лицевая.
    #   faceswap — специализированный перенос лица. Маска не нужна: модель
    #              сама находит лицо и бережёт волосы обложки. Промпта нет,
    #              поэтому результат тяготеет к фотореализму.
    fal_backend: str = "kontext"

    # --- Бэкенд kontext: инпейнтинг по маске с референсом личности ---
    # negative_prompt эндпоинт не принимает — его здесь намеренно нет.
    fal_kontext_model: str = "fal-ai/flux-kontext-lora/inpaint"
    fal_prompt: str = (
        "A portrait of a person, matching the exact artistic style, "
        "brushstrokes, and lighting of the surrounding image"
    )
    # Дефолт эндпоинта 0.88; в его документации сказано, что этой модели
    # подходят высокие значения strength
    fal_strength: float = 0.88
    # Дефолт эндпоинта 2.5 — заметно ниже привычных для SD 7.5
    fal_guidance_scale: float = 2.5
    fal_steps: int = 30

    # --- Бэкенд faceswap: специализированный перенос лица ---
    fal_faceswap_model: str = "easel-ai/advanced-face-swap"
    # target_hair сохраняет причёску обложки — ровно то, ради чего маска
    # намеренно не заходит на лоб и волосы
    fal_faceswap_workflow: str = "target_hair"
    fal_faceswap_upscale: bool = True
    # Пол донора модель использует для подгонки результата. Значение приходит
    # из заказа; non-binary — нейтральный дефолт, когда его не передали.
    fal_faceswap_default_gender: str = "non-binary"

    # --- Маска лица ---
    # Сильное размытие даёт бесшовный переход к иллюстрации
    mask_blur_kernel: int = 101

    @property
    def fal_model(self) -> str:
        """Активная модель — то, что видно в /health/ready и в логах."""
        return self.fal_kontext_model if self.fal_backend == "kontext" else self.fal_faceswap_model

    @property
    def mask_required(self) -> bool:
        """
        Нужна ли маска. faceswap ищет лицо сам, и строить её для него не просто
        лишний расход: MediaPipe не видит лицо мельче ~20% ширины кадра и
        завернул бы 422 обложку, которую faceswap обработал бы нормально.
        """
        return self.fal_backend == "kontext"

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
