"""
Генератор маски лица по сетке MediaPipe Face Mesh.

Маска нужна инпейнтингу: белое — зона, которую модель перерисовывает, чёрное —
неприкосновенный фон. Поэтому полигон строится строго по контуру лица (линия
челюсти, щёки, брови) и намеренно НЕ включает лоб, волосы и фон: перерисованные
волосы дают ореолы и рассогласование с иллюстрацией.

Полигонов на входе может быть несколько: пайплайн подаёт сюда и контур лица
шаблона, и контур вклеенного коллажа (collage.py). Маска обязана накрывать их
объединение — иначе шов склейки окажется снаружи зоны инпейнтинга, и сглаживать
его будет некому.

Ключевое отличие от Face Detection: та отдаёт лишь прямоугольник, внутри
которого неизбежно оказываются волосы и фон. Плотная сетка позволяет обвести
именно лицо.

Профиль краёв маски — причина «пластиковой маски» в результате. Полигон,
залитый белым и просто размытый по Гауссу, даёт градиент, симметричный
относительно контура: половина растушёвки съедает лицо, наружу идёт вторая.
Модели остаётся полоска в один-два десятка пикселей, чтобы согласовать новое
лицо с иллюстрацией, — и она вклеивает его встык. Поэтому маска строится в три
шага:

  1. заливка полигона — контур лица;
  2. дилатация на padding + feather — вокруг лица появляется поле, где модели
     разрешено дорисовывать переход (скулы, подбородок, край лба);
  3. размытие на feather — плавный спад уже целиком за пределами лица.

Ядро размытия берётся ровно по ширине растушёвки, поэтому граница «лицо +
padding» остаётся полностью белой, а спад 255 → 0 укладывается в кольцо
дилатации. Лицо перерисовывается целиком, а сливается с фоном постепенно.
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

# Паддинг и растушёвка — доли высоты лица, а не пиксели: обложки приходят и в 4K,
# и превью-размером, одно и то же ядро на них работает совершенно по-разному.
#
# padding — кольцо вокруг контура, залитое белым наравне с лицом. Даёт модели
#   запас, чтобы дорисовать переход, а не обрывать новое лицо по линии челюсти.
# feather — ширина спада 255 → 0 за этим кольцом. Ровно она и растворяет край
#   маски в иллюстрации.
#
# Сумма (0.06 + 0.10 = 16% высоты лица) подобрана так, чтобы кольцо не доставало
# до волос: лоб отсечён бровями, и запаса до линии роста волос как раз хватает.
_PADDING_RATIO = 0.06
_FEATHER_RATIO = 0.10


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


def soften(
    mask: Any,
    face_height: int,
    padding_ratio: float,
    feather_ratio: float,
) -> Any:
    """
    Профиль краёв: сплошное поле вокруг залитой области и спад за ним.

    Область расширяется на padding + feather и размывается по Гауссу с ядром в
    ширину feather. Полностью белой остаётся исходная область вместе с полем
    padding, а спад до чёрного целиком лежит снаружи — ни один пиксель залитого
    контура не оказывается наполовину прозрачным.

    Используется дважды: для маски инпейнтинга (padding + feather в долях
    высоты лица) и для маски вклейки коллажа (микроскопические доли — там
    растушёвка нужна лишь чтобы убрать ступеньку антиалиасинга).

    :param mask: одноканальная маска uint8, залитая белым
    :param face_height: высота лица в пикселях — база для обеих долей
    :param padding_ratio: сплошное поле вокруг области, доля высоты лица
    :param feather_ratio: ширина спада за полем, доля высоты лица
    """
    import cv2

    if padding_ratio < 0 or feather_ratio < 0:
        raise InvalidImageError(
            "Паддинг и растушёвка маски не могут быть отрицательными",
            {"padding_ratio": padding_ratio, "feather_ratio": feather_ratio},
        )

    padding = round(face_height * padding_ratio)
    feather = round(face_height * feather_ratio)

    # Дилатация на padding + feather: после размытия внешние feather пикселей
    # кольца уйдут в градиент, а padding останется сплошной белой каймой
    grow = padding + feather
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask = cv2.dilate(mask, kernel)

    # Ядро (2*feather+1) обрывается ровно на границе кольца: дальше этой точки
    # свёртка видит только белое, ближе к краю — только чёрное. Отсюда и полная
    # непрозрачность лица, и честный ноль за пределами маски.
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (2 * feather + 1, 2 * feather + 1), 0)

    return mask


def mask_from_polygons(
    shape: tuple[int, int],
    polygons: list[Any],
    padding_ratio: float = _PADDING_RATIO,
    feather_ratio: float = _FEATHER_RATIO,
) -> Any:
    """
    Маска инпейнтинга по готовым контурам: объединение с полем и мягким краем.

    Полигоны заливаются в одну маску, поэтому зона перерисовки накрывает и лицо
    шаблона, и вклеенный коллаж целиком, чем бы они ни различались.

    :param shape: (height, width) кадра
    :param polygons: контуры Nx2 в пикселях кадра
    :param padding_ratio: поле вокруг контуров, доля высоты лица
    :param feather_ratio: ширина растушёвки за полем, доля высоты лица
    :return: одноканальная маска uint8 размера shape
    """
    import cv2
    import numpy as np

    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # По одному вызову на контур: fillPoly со списком считает их частями одной
    # фигуры по правилу чёт-нечет, и пересечение двух контуров осталось бы
    # дырой — ровно в том месте, где лицо шаблона и вклейка совпадают.
    contours = [np.asarray(polygon, dtype=np.int32) for polygon in polygons]
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)

    # Высота лица — по описанному прямоугольнику объединения: полигоны уже
    # включают подъём бровей, то есть меряется ровно перерисовываемая область.
    ys = np.concatenate([contour[:, 1] for contour in contours])
    face_height = int(ys.max() - ys.min())

    result = soften(mask, face_height, padding_ratio, feather_ratio)

    log.info(
        "маска лица построена",
        extra={
            "image_size": f"{width}x{height}",
            "polygons": len(contours),
            "face_height": face_height,
            "padding_px": round(face_height * padding_ratio),
            "feather_px": round(face_height * feather_ratio),
        },
    )
    return result
