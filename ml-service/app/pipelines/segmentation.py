"""
Вырезка головы донора целиком — с волосами.

Раньше в шаблон переносился только овал лица (челюсть + брови), а причёска
оставалась от нарисованного персонажа. Требование изменилось: волосы заказчика
должны сохранить свой цвет, длину и структуру, то есть переносить нужно голову
целиком.

Контур такой головы по сетке mediapipe не построить — она обводит лицо и про
волосы не знает ничего. Поэтому силуэт даёт сегментатор (rembg / U²-Net), а
сетка используется лишь для того, чтобы ограничить его головой:

  1. rembg отделяет человека от фона и возвращает альфу всего силуэта —
     вместе с плечами, руками и всем остальным;
  2. по сетке лица строится «область головы»: эллипс, повёрнутый вместе с
     наклоном головы, с запасом на причёску сверху и по бокам;
  3. снизу эллипс срезается прямой линией — это и есть срез шеи;
  4. пересечение альфы с этой областью и есть голова. Лишние куски силуэта,
     попавшие в эллипс (поднятая рука, второй человек за плечом), отсекаются
     выбором крупнейшей связной компоненты.

Такое разделение труда важно: сегментатор точно знает, где кончаются волосы, но
не знает, где кончается голова; сетка знает, где голова, но не знает про волосы.
Ни один из них по отдельности задачу не решает.

Одна и та же функция работает и для фотографии, и для обложки: на шаблоне тем же
способом находится голова нарисованного персонажа — её надо стереть, иначе её
причёска торчала бы из-под вклеенной.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.errors import MLServiceError
from app.core.logging import get_logger

log = get_logger(__name__)

# Индексы сетки, по которым строится область головы.
_CHIN = 152
_EYE_LEFT, _EYE_RIGHT = 33, 263
_CHEEK_LEFT, _CHEEK_RIGHT = 234, 454
_BROW_MID = 9  # переносица между бровями

# Пропорции головы в долях высоты лица (подбородок → линия бровей). Числа
# соответствуют канону: от подбородка до линии волос примерно полторы такие
# высоты, до макушки — две, ещё немного добавляет объём причёски.
_HAIR_RATIO = 2.3  # вверх от подбородка
_NECK_RATIO = 0.45  # вниз от подбородка — сколько шеи остаётся под срезом
_WIDTH_RATIO = 1.6  # ширина эллипса в долях ширины лица (скула → скула)


class SegmenterUnavailableError(MLServiceError):
    """
    rembg не установлен или не смог загрузить веса.

    503, а не 500: это состояние окружения, а не дефект запроса — ровно как
    отсутствующий FAL_KEY. Повторять запрос бессмысленно, пока не починят сервис.
    """

    status_code = 503
    code = "SEGMENTER_UNAVAILABLE"


@dataclass
class Head:
    """Вырезанная голова: альфа силуэта и геометрия среза шеи."""

    alpha: Any  # одноканальная маска uint8 в координатах исходного кадра
    neck_line: tuple[tuple[int, int], tuple[int, int]]  # отрезок среза шеи
    face_height: float  # подбородок → линия бровей, пиксели
    meta: dict


_SESSIONS: dict[str, Any] = {}


def _session(model: str) -> Any:
    """
    Сессия rembg на модель. Кэшируется: загрузка весов занимает секунды, а
    воркер обрабатывает заказы один за другим в одном процессе.
    """
    if model in _SESSIONS:
        return _SESSIONS[model]

    try:
        from rembg import new_session
    except ImportError as exc:  # noqa: BLE001
        raise SegmenterUnavailableError(
            "rembg не установлен — выполните pip install -r requirements.txt",
            {"cause": str(exc)},
        ) from exc

    try:
        _SESSIONS[model] = new_session(model)
    except Exception as exc:  # noqa: BLE001 — скачивание весов, битый кэш, ошибка onnx
        raise SegmenterUnavailableError(
            f"Не удалось поднять сегментатор {model}: {exc}",
            {"model": model},
        ) from exc

    log.info("сегментатор готов", extra={"model": model})
    return _SESSIONS[model]


def silhouette(image: Any, model: str) -> Any:
    """Альфа переднего плана целиком: человек (или персонаж) без фона."""
    from rembg import remove

    session = _session(model)
    try:
        return remove(image, session=session, only_mask=True)
    except Exception as exc:  # noqa: BLE001 — отказ onnxruntime на нестандартном кадре
        raise SegmenterUnavailableError(
            f"Сегментация не выполнена: {exc}",
            {"model": model},
        ) from exc


def head_region(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    width_ratio: float = _WIDTH_RATIO,
    hair_ratio: float = _HAIR_RATIO,
    neck_ratio: float = _NECK_RATIO,
) -> tuple[Any, tuple[tuple[int, int], tuple[int, int]], float]:
    """
    Область головы по сетке лица: эллипс со срезанным низом.

    Эллипс поворачивается вместе с головой (угол берётся по линии глаз), иначе
    у наклонённой головы он срезал бы висок с одной стороны и захватывал фон с
    другой. Низ срезается прямой, перпендикулярной оси лица: срез шеи должен
    быть ровным — его потом закрывает инпейнтинг, и чем он предсказуемее, тем
    проще его спрятать.

    :return: (маска области uint8, отрезок среза шеи, высота лица в пикселях)
    """
    import cv2
    import numpy as np

    height, width = shape
    chin = np.array(points[_CHIN], dtype=np.float64)
    brow = np.array(points[_BROW_MID], dtype=np.float64)

    eye_left = np.array(points[_EYE_LEFT], dtype=np.float64)
    eye_right = np.array(points[_EYE_RIGHT], dtype=np.float64)

    # Ось лица: единичный вектор от подбородка к бровям. По нему отмеряются и
    # запас на причёску, и срез шеи — то есть всё в системе координат головы,
    # а не кадра.
    axis = brow - chin
    face_height = float(np.linalg.norm(axis))
    face_width = float(np.linalg.norm(eye_right - eye_left)) * 2.0
    if face_height < 1.0 or face_width < 1.0:
        raise MLServiceError("Вырожденная геометрия лица: голову не выделить")

    up = axis / face_height
    roll = float(np.degrees(np.arctan2(up[0], -up[1])))  # 0° — лицо не наклонено

    hair_up = hair_ratio * face_height
    neck_down = neck_ratio * face_height

    centre = chin + up * (hair_up - neck_down) / 2.0
    axes = (int(round(width_ratio * face_width / 2)), int(round((hair_up + neck_down) / 2)))

    region = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(region, (int(centre[0]), int(centre[1])), axes, roll, 0, 360, 255, -1)

    # Срез шеи: всё, что ниже линии, из области выбрасывается. Линия рисуется
    # как большой прямоугольник, чтобы не считать полуплоскость попиксельно —
    # на 4K это заметная разница.
    neck_point = chin - up * neck_down
    side = np.array([-up[1], up[0]])  # перпендикуляр к оси лица
    reach = float(max(height, width)) * 2.0
    below = np.array(
        [
            neck_point + side * reach,
            neck_point - side * reach,
            neck_point - side * reach - up * reach,
            neck_point + side * reach - up * reach,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(region, below, 0)

    # Отрезок среза — по ширине эллипса, он же граница «голова / шея»
    half = width_ratio * face_width / 2
    neck_line = (
        tuple(np.round(neck_point + side * half).astype(int)),
        tuple(np.round(neck_point - side * half).astype(int)),
    )
    return region, neck_line, face_height


def _largest_component(mask: Any) -> Any:
    """
    Оставляет крупнейший связный кусок.

    В эллипс головы попадает не только она: поднятая к лицу рука, плечо
    соседа, ветка за спиной. Всё это отдельные пятна альфы, и вклеивать их в
    обложку не нужно.
    """
    import cv2
    import numpy as np

    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    if count <= 2:  # фон + одна компонента (или пусто) — делить нечего
        return mask

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, mask, 0).astype(np.uint8)


def cutout_head(
    image: Any,
    points: list[tuple[int, int]],
    model: str,
    width_ratio: float = _WIDTH_RATIO,
    hair_ratio: float = _HAIR_RATIO,
    neck_ratio: float = _NECK_RATIO,
) -> Head:
    """
    Голова целиком: силуэт сегментатора, ограниченный областью головы.

    :param image: BGR numpy.ndarray
    :param points: сетка mediapipe того же кадра
    :param model: имя модели rembg
    :return: Head с альфой, отрезком среза шеи и высотой лица
    """
    import numpy as np

    region, neck_line, face_height = head_region(
        points, image.shape[:2], width_ratio, hair_ratio, neck_ratio
    )
    alpha = silhouette(image, model)

    head = _largest_component(np.where(region > 0, alpha, 0).astype(np.uint8))

    area = int(np.count_nonzero(head > 127))
    meta = {
        "model": model,
        "face_height": round(face_height, 1),
        "head_px": area,
        # Насколько силуэт заполнил отведённый эллипс. Близко к нулю — значит
        # сегментатор не нашёл человека (или нашёл не там), и вклеивать нечего.
        "fill": round(area / max(1, int(np.count_nonzero(region))), 3),
    }
    log.info("голова вырезана", extra=meta)

    if area == 0:
        raise SegmenterUnavailableError(
            "Сегментатор не нашёл голову в кадре — вклеивать нечего", meta
        )

    return Head(alpha=head, neck_line=neck_line, face_height=face_height, meta=meta)
