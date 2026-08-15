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


def encode_image(image: Any, fmt: str = "png", quality: int | None = None) -> tuple[bytes, str]:
    """
    Кодирует кадр. Возвращает байты и MIME.

    :param quality: качество для форматов с потерями, 1..100. None — умолчание
        кодека. Для PNG игнорируется: там качества нет, есть степень сжатия, и
        путать их значит менять размер, думая, что меняешь картинку.

        Прореживание цветности НЕ навязывается. Форсировать 4:4:4 казалось
        очевидным улучшением для кожи, но замер на нашем материале показал
        обратное: файл на четверть больше, а ошибка цветности в LAB ВЫШЕ (0.095
        против 0.057 у умолчания). Оставлено кодеку.
    """
    cv2 = _cv2()
    fmt = fmt.lower()
    if fmt not in _MIME:
        raise InvalidImageError(f"Неподдерживаемый формат: {fmt}", {"supported": list(_MIME)})

    params: list[int] = []
    if quality is not None:
        if not 1 <= quality <= 100:
            raise InvalidImageError(
                "Качество кодирования лежит между 1 и 100", {"quality": quality}
            )
        if fmt in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]

    ok, buffer = cv2.imencode(f".{fmt}", image, params)
    if not ok:
        raise InvalidImageError(f"Не удалось закодировать изображение в {fmt}")
    return buffer.tobytes(), _MIME[fmt]
