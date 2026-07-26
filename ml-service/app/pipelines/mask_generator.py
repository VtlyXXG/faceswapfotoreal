"""
Генератор маски лица по сетке MediaPipe Face Mesh.

Маска нужна инпейнтингу: белое — зона, которую модель перерисовывает, чёрное —
неприкосновенный фон. Поэтому полигон строится строго по контуру лица (линия
челюсти, щёки, брови) и намеренно НЕ включает лоб, волосы и фон: перерисованные
волосы дают ореолы и рассогласование с иллюстрацией.

Ключевое отличие от Face Detection: та отдаёт лишь прямоугольник, внутри
которого неизбежно оказываются волосы и фон. Плотная сетка позволяет обвести
именно лицо.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.core.logging import get_logger

log = get_logger(__name__)

# Нижняя дуга овала лица в порядке обхода: от уха через подбородок к другому уху.
# Индексы — из канонического FACEMESH_FACE_OVAL, но верхняя (лобная) часть
# кольца отброшена: именно она заходит на волосы.
_JAW_ARC = (
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152,  # подбородок
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
)

# Верхняя граница — по бровям, от внешнего края одной к внешнему краю другой.
# Замыкает полигон вместо лба.
_BROW_ARC = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)

# Брови приподнимаются на долю высоты лица, чтобы попасть внутрь маски целиком:
# точки сетки лежат по нижнему краю брови.
_BROW_LIFT = 0.06


def _mediapipe():
    try:
        import mediapipe as mp

        return mp
    except ImportError as exc:  # noqa: BLE001
        raise InvalidImageError(
            "mediapipe не установлен — выполните pip install -r requirements.txt",
            {"cause": str(exc)},
        ) from exc


def face_landmarks(image: Any) -> list[tuple[int, int]]:
    """
    Сетка лицевых точек для крупнейшего лица: пиксели BGR-изображения.

    При refine_landmarks=True mediapipe отдаёт 478 точек — канонические 468
    плюс 10 на зрачки. Полигон использует только индексы из первых 468.

    :param image: BGR numpy.ndarray
    """
    import cv2

    mp = _mediapipe()
    height, width = image.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,  # одиночный кадр, а не видеопоток
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not result.multi_face_landmarks:
        raise NoFaceDetectedError("На изображении не найдено ни одного лица")

    landmarks = result.multi_face_landmarks[0].landmark
    return [(int(point.x * width), int(point.y * height)) for point in landmarks]


def face_polygon(points: list[tuple[int, int]]) -> Any:
    """Замкнутый контур лица: дуга челюсти + линия бровей вместо лба."""
    import numpy as np

    jaw = [points[i] for i in _JAW_ARC]
    brows = [points[i] for i in _BROW_ARC]

    # Высота лица нужна, чтобы поднять брови пропорционально размеру лица,
    # а не на фиксированное число пикселей. Меряется от линии бровей до
    # подбородка: у одной лишь дуги челюсти крайние точки лежат на уровне ушей,
    # её вертикальный размах вдвое меньше лица и подъём выходит недостаточным.
    ys = [y for _, y in jaw + brows]
    face_height = max(ys) - min(ys)
    lift = int(face_height * _BROW_LIFT)

    return np.array(jaw + [(x, y - lift) for x, y in brows], dtype=np.int32)


def generate_mask(image: Any, blur_kernel: int = 101) -> Any:
    """
    Маска инпейнтинга: белый полигон лица на чёрном фоне, размытый по Гауссу.

    Размытие делает переход градиентным — модель смешивает новое лицо с
    иллюстрацией постепенно, без жёсткой границы и ореола.

    :param image: BGR numpy.ndarray с лицом
    :param blur_kernel: размер ядра Гаусса; приводится к нечётному
    :return: одноканальная маска uint8 того же размера, что и image
    """
    import cv2
    import numpy as np

    if blur_kernel < 1:
        raise InvalidImageError("Размер ядра размытия должен быть положительным")
    # cv2 требует нечётное ядро
    blur_kernel |= 1

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = face_polygon(face_landmarks(image))
    cv2.fillPoly(mask, [polygon], 255)

    blurred = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

    log.info(
        "маска лица построена",
        extra={
            "image_size": f"{width}x{height}",
            "polygon_points": len(polygon),
            "blur_kernel": blur_kernel,
        },
    )
    return blurred
