"""
Состояние обработчика для /health/ready.

Тяжёлых весов больше нет — прогревать нечего, поэтому от прежнего реестра
моделей осталась только диагностика: доступны ли зависимости и настроен ли
ключ fal. Функции warmup/reset сохранены, чтобы не менять app/main.py.
"""

from __future__ import annotations

from importlib.util import find_spec

from app.config import settings
from app.pipelines import fal_api


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def warmup() -> None:
    """Раньше грузила веса в память. Локальных моделей нет — делать нечего."""


def reset() -> None:
    """Симметрична warmup: освобождать тоже нечего."""


def status() -> dict:
    return {
        "runtime": {
            "mediapipe": _installed("mediapipe"),
            "opencv": _installed("cv2"),
            "fal_client": _installed("fal_client"),
        },
        "provider": {
            "model": settings.fal_model,
            "key_present": fal_api.key_present(),
            "key_env": settings.fal_key_env,
        },
        "mask": {
            "detector": "mediapipe/face_mesh",
            "padding_ratio": settings.mask_padding_ratio,
            "feather_ratio": settings.mask_feather_ratio,
        },
    }
