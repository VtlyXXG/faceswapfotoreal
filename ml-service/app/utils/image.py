"""Кодирование/декодирование изображений. Внутри пайплайна — BGR numpy.ndarray."""

from __future__ import annotations

from typing import Any

from app.core.errors import InvalidImageError

_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


def _cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:  # noqa: BLE001
        raise InvalidImageError(
            "opencv не установлен — выполните scripts/setup_venv.ps1", {"cause": str(exc)}
        ) from exc


def decode_image(data: bytes) -> Any:
    import numpy as np

    cv2 = _cv2()
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Не удалось прочитать изображение")
    return image


def encode_image(image: Any, fmt: str = "png") -> tuple[bytes, str]:
    cv2 = _cv2()
    fmt = fmt.lower()
    if fmt not in _MIME:
        raise InvalidImageError(f"Неподдерживаемый формат: {fmt}", {"supported": list(_MIME)})

    ok, buffer = cv2.imencode(f".{fmt}", image)
    if not ok:
        raise InvalidImageError(f"Не удалось закодировать изображение в {fmt}")
    return buffer.tobytes(), _MIME[fmt]
