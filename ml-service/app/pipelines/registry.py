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
from app.core.errors import MLServiceError
from app.pipelines import expression, fal_api, refine


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


def _active():
    """
    Активный профиль второго шага и причина, если собрать его не вышло.

    Профиль складывается из пресета и переопределений окружения, то есть может
    оказаться несобираемым — например, карты ControlNet при эндпоинте, который
    их не принимает. Узнать об этом на /health/ready лучше, чем на первом живом
    заказе: тот отвалится уже после трёх загрузок в CDN.
    """
    try:
        return refine.profiles.from_settings(), None
    except MLServiceError as exc:
        return None, exc.message


def status() -> dict:
    profile, profile_error = _active()
    mask = profile.mask if profile else None

    return {
        "runtime": {
            "mediapipe": _installed("mediapipe"),
            "opencv": _installed("cv2"),
            "fal_client": _installed("fal_client"),
            "rembg": _installed("rembg"),
        },
        # Профиль виден целиком намеренно: strength, веса карт и ширина
        # градиента — главные ручки пайплайна, и подбирают их из окружения на
        # живом сервисе. Пустые значения означают, что профиль не собрался, —
        # тогда всё, что о нём известно, лежит в profile_error
        "provider": {
            "model": profile.endpoint if profile else None,
            "key_present": fal_api.key_present(),
            "key_env": settings.fal_key_env,
            "strength": profile.strength if profile else None,
            "profile": settings.refine_profile,
            "strategy": profile.strategy if profile else None,
            "controls": [f"{c.kind}:{c.weight}" for c in profile.controls] if profile else [],
            "profile_error": profile_error,
            "profiles": refine.profiles.available(),
            "strategies": refine.available(),
        },
        "mask": {
            "detector": "mediapipe/face_mesh",
            "edge_ratio": mask.edge_ratio if mask else None,
            "neck_ratio": mask.neck_ratio if mask else None,
            "guard_ratio": mask.guard_ratio if mask else None,
            "feather_ratio": mask.feather_ratio if mask else None,
            "gradient_ratio": mask.gradient_ratio if mask else None,
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
