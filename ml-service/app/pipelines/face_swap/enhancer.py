"""
Постобработка результата (восстановление лица, апскейл).

Точка расширения: GFPGAN / CodeFormer / RealESRGAN. Пока — no-op, чтобы
пайплайн был работоспособен без дополнительных весов.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

_warned = False


def enhance(image: Any, enabled: bool = False) -> Any:
    global _warned
    if not enabled:
        return image

    if not _warned:
        log.warning("постобработка запрошена, но энхансер не подключён — возвращаю оригинал")
        _warned = True
    # TODO: подключить GFPGAN/CodeFormer и вернуть улучшенное изображение
    return image


def is_available() -> bool:
    return False
