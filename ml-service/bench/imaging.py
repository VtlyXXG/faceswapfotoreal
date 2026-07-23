"""Кодирование кропов в data: URI для встраивания в самодостаточный HTML."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def to_data_uri(image: np.ndarray, max_width: int = 240, quality: int = 88) -> str:
    """BGR-изображение → строка data:image/jpeg;base64,... с уменьшением до max_width."""
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    height, width = img.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, max(1, int(height * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
