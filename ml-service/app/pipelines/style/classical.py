"""
Baseline-стилизатор на классическом CV: без весов, без диффузии, миллисекунды.

Снимает три из четырёх источников «зловещей долины»: шов маски, цветовой
скачок и фотографическую микротекстуру. Мазок кисти он не рисует — это
задача диффузионной стадии, которая встанет на это же место.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.pipelines.style import color
from app.pipelines.style.base import BaseStylizer, StyleOptions
from app.pipelines.style.masking import build_masks
from app.pipelines.style.metrics import texture_distance

log = get_logger(__name__)


class ClassicalStylizer(BaseStylizer):
    name = "classical"

    def stylize(
        self,
        image: Any,
        face: Any,
        options: StyleOptions,
        identity_embedding: Any = None,  # noqa: ARG002 — нужен только диффузии
    ) -> np.ndarray:
        masks = build_masks(face, image.shape, margin=options.margin)
        x0, y0, x1, y1 = masks.box

        if x1 - x0 < 16 or y1 - y0 < 16:
            log.warning("лицо слишком мелкое для постобработки, пропускаю")
            return image

        crop = image[y0:y1, x0:x1].astype(np.float32)
        before = texture_distance(crop, masks.skin, masks.ring)

        crop = color.match_color(crop, masks.skin, masks.ring, options.color)
        crop = color.suppress_micro_texture(crop, masks.skin, masks.identity, options.smooth)
        crop = color.harmonize_texture(
            crop, masks.skin, masks.ring, grain=options.grain, strength=options.sharpness
        )

        after = texture_distance(crop, masks.skin, masks.ring)
        log.info(
            "постобработка лица: рассогласование фактуры %.4f -> %.4f",
            before,
            after,
            extra={"texture_before": before, "texture_after": after},
        )

        return color.paste_back(image, crop, masks.skin, masks.box)

    def is_available(self) -> bool:
        return True


class NoopStylizer(BaseStylizer):
    """Прежнее поведение: постобработки нет."""

    name = "noop"

    def stylize(  # noqa: ARG002
        self,
        image: Any,
        face: Any,
        options: StyleOptions,
        identity_embedding: Any = None,
    ) -> Any:
        return image

    def is_available(self) -> bool:
        return True
