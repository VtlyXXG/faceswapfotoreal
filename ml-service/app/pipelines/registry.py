"""
Состояние обработчика для /health/ready.

Тяжёлых весов больше нет — прогревать нечего, поэтому от прежнего реестра
моделей осталась только диагностика: доступны ли зависимости и настроен ли
ключ fal. Функции warmup/reset сохранены, чтобы не менять app/main.py.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

from app.config import settings
from app.pipelines import expression, fal_api


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _weights_ready() -> bool:
    """
    Скачаны ли веса сегментатора.

    Первый вызов тянет ~176 МБ на модель, и заказ, попавший на эту загрузку,
    ждёт её минутами. В /health/ready это видно заранее — до того, как в
    очередь встанет живой заказ.
    """
    home = os.environ.get("U2NET_HOME") or Path.home() / ".u2net"
    return all(
        (Path(home) / f"{model}.onnx").exists()
        for model in {settings.seg_model_photo, settings.seg_model_cover}
    )


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
            "rembg": _installed("rembg"),
        },
        "provider": {
            "model": settings.fal_model,
            "key_present": fal_api.key_present(),
            "key_env": settings.fal_key_env,
            # Виден в /health/ready намеренно: strength — главная ручка
            # пайплайна, и подбирают её из окружения на живом сервисе
            "strength": settings.fal_strength,
        },
        "mask": {
            "detector": "mediapipe/face_mesh",
            "edge_ratio": settings.mask_edge_ratio,
            "neck_ratio": settings.mask_neck_ratio,
            "guard_ratio": settings.mask_guard_ratio,
            "feather_ratio": settings.mask_feather_ratio,
        },
        # Первый шаг пайплайна виден отдельно: по этим числам сразу понятно,
        # выполняется ли перенос головы локально и с какими допусками.
        "collage": {
            "segmenter_photo": settings.seg_model_photo,
            "segmenter_cover": settings.seg_model_cover,
            "weights_ready": _weights_ready(),
            "colour_match": settings.collage_colour_match,
            "erase_template_head": settings.collage_erase_template_head,
        },
        # Мимика: какие значения параметра emotion эндпоинт сейчас принимает
        "expressions": expression.available(),
    }
