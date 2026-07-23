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

    # cpu | cuda | auto
    device: str = "auto"

    models_dir: Path = Path("./models")
    face_detector: str = "buffalo_l"
    face_swapper: str = "inswapper_128.onnx"
    det_size: int = 640
    lazy_load: bool = True

    # 4K-обложки занимают 25-30 МБ; значение продублировано в .env.example
    # --- Постобработка (согласование лица с иллюстрацией) ---
    # classical | diffusion | noop
    style_provider: str = "classical"
    style_color: float = 0.6
    style_smooth: float = 0.5
    style_grain: float = 0.7
    style_sharpness: float = 0.6
    style_margin: float = 1.8

    # --- Диффузионный стилизатор (SD 1.5 + IP-Adapter FaceID) ---
    # Пути относительно models_dir; веса кладёт scripts/fetch_style_models.py
    style_diffusion_model: str = "sd15"
    style_diffusion_ip_adapter: str = "ip-adapter-faceid"
    style_diffusion_ip_weight: str = "ip-adapter-faceid_sd15.bin"
    style_diffusion_ip_lora: str = "ip-adapter-faceid_sd15_lora.safetensors"
    style_diffusion_ip_scale: float = 0.8
    style_diffusion_texture_strength: float = 0.42
    style_diffusion_identity_keep: float = 0.6  # доля исходной геометрии в зоне глаз/носа/рта
    style_diffusion_steps: int = 30
    style_diffusion_guidance: float = 5.0
    style_diffusion_work_size: int = 512
    style_diffusion_seed: int = 42

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

    def resolve_device(self) -> str:
        """Разрешает 'auto' в реальное устройство, не требуя torch на старте."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
