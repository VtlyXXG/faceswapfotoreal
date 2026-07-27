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
  2. альфа чистится: полупрозрачная кайма отбрасывается по порогу, а край
     подрезается эрозией — иначе вместе с волосами переезжает ореол фона
     фотографии;
  3. по сетке лица строится «область головы»: эллипс, повёрнутый вместе с
     наклоном головы, с запасом на причёску сверху и по бокам;
  4. снизу эллипс срезается **по линии челюсти** из той же сетки — ни шея, ни
     воротник, ни футболка в аппликацию не попадают вовсе;
  5. пересечение альфы с этой областью и есть голова. Лишние куски силуэта,
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
from app.pipelines import mask_generator

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
_WIDTH_RATIO = 1.6  # ширина эллипса в долях ширины лица (скула → скула)

# Отступ среза вниз от линии челюсти, доля высоты лица. Ноль — режем ровно по
# челюсти. Прежние 0.45 оставляли под подбородком полосу шеи, а вместе с ней в
# аппликацию заезжали воротник и плечи донора: на рисованной обложке чёрная
# футболка под нарисованным платьем видна сразу. Положительное значение
# опускает срез ниже челюсти, если модели не хватает материала на воротник.
_NECK_RATIO = 0.0

# Подрезка края силуэта, доля высоты лица. 0.006 — это 2-3 пикселя на типичном
# портрете (лицо 350-450 px). Сегментатор ведёт границу по внешнему краю
# волос, и последние пиксели там наполовину состоят из фона фотографии: без
# эрозии этот фон переезжает на обложку тонкой грязной каймой вокруг причёски.
_ERODE_RATIO = 0.006

# Порог «уверенного» переднего плана. U²-Net отдаёт не бинарную маску, а карту
# уверенности, и вокруг волос лежит широкая полутень. Всё, что ниже порога,
# считается фоном: полупрозрачные пиксели в коллаже дают ровно тот же ореол,
# что и лишние — просто слабее.
_ALPHA_SOLID = 160


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

    alpha: Any  # бинарная маска uint8 в координатах исходного кадра
    neck_line: tuple[tuple[int, int], tuple[int, int]]  # отрезок среза под подбородком
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
    follow_jaw: bool = True,
) -> tuple[Any, tuple[tuple[int, int], tuple[int, int]], float]:
    """
    Область головы по сетке лица: эллипс со срезанным по челюсти низом.

    Эллипс поворачивается вместе с головой (угол берётся по линии глаз), иначе
    у наклонённой головы он срезал бы висок с одной стороны и захватывал фон с
    другой.

    Низ отрезается по дуге челюсти из сетки, а не прямой поперёк кадра. Прямая
    через подбородок оставляла бы по бокам от него два треугольника шеи —
    челюсть поднимается к ушам, а линия нет; прямая ниже подбородка тянула бы
    за собой воротник и плечи. Дуга снимает и то и другое: под срезом не
    остаётся ни пикселя шеи, а уши и волосы выше линии челюсти сохраняются.

    Прямой срез (`follow_jaw=False`) нужен там, где голову не вырезают, а
    наоборот стирают: дуга поднимается к ушам и оставляет их кончики, а прямая
    на уровне подбородка забирает ухо целиком и не трогает шею под подбородком.

    :param neck_ratio: отступ среза вниз от челюсти (или от подбородка при
        follow_jaw=False), доля высоты лица
    :param follow_jaw: вести срез по дуге челюсти или прямой поперёк оси лица
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

    # Срез: всё, что ниже линии челюсти, из области выбрасывается. Дуга
    # берётся из той же сетки, что и полигон лица, и опускается на neck_down
    # вдоль оси лица. С боков она продлевается за пределы кадра, чтобы
    # получилась замкнутая фигура «всё, что ниже», — считать полуплоскость
    # попиксельно на 4K заметно дороже.
    side = np.array([-up[1], up[0]])  # перпендикуляр к оси лица
    reach = float(max(height, width)) * 2.0

    if follow_jaw:
        jaw = (
            np.array([points[i] for i in mask_generator._JAW_ARC], dtype=np.float64)
            - up * neck_down
        )
        # Куда продлевать концы дуги, зависит от того, с какой стороны лица
        # лежит её начало: порядок обхода в сетке фиксирован, знак
        # перпендикуляра — нет
        outward = side if float(np.dot(jaw[0] - chin, side)) > 0 else -side
        first, last = jaw[0] + outward * reach, jaw[-1] - outward * reach
        cut = [first, *jaw, last]
    else:
        flat = chin - up * neck_down
        cut = [flat + side * reach, flat - side * reach]

    below = np.array(
        [*cut, cut[-1] - up * reach, cut[0] - up * reach],
        dtype=np.int32,
    )
    cv2.fillPoly(region, [below], 0)  # дуга невыпуклая — fillConvexPoly здесь соврёт

    # Отрезок среза — по ширине эллипса на уровне подбородка. Это самое
    # заметное место шва, и маска второго шага кладёт полосу именно сюда.
    neck_point = chin - up * neck_down
    half = width_ratio * face_width / 2
    neck_line = (
        tuple(np.round(neck_point + side * half).astype(int)),
        tuple(np.round(neck_point - side * half).astype(int)),
    )
    return region, neck_line, face_height


def clean_alpha(alpha: Any, erode_px: int) -> Any:
    """
    Твёрдый край силуэта: порог уверенности плюс эрозия.

    Обе операции борются с одним и тем же — с каймой фона фотографии по контуру
    волос. Сначала отбрасывается полутень сегментатора (в коллаже она даёт
    полупрозрачный ореол), затем край подрезается внутрь на erode_px: граница
    U²-Net проходит по внешним пикселям прядей, а они смешаны с тем, что было за
    головой.

    Маска после этого бинарная. Мягкий край аппликации делает `collage.py` — и
    делает его внутрь, чтобы подрезанное сюда не вернулось.

    :param alpha: карта уверенности uint8 от сегментатора
    :param erode_px: на сколько пикселей подрезать край; 0 — не подрезать
    """
    import cv2
    import numpy as np

    solid = np.where(np.asarray(alpha) >= _ALPHA_SOLID, 255, 0).astype(np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        solid = cv2.erode(solid, kernel)
    return solid


def largest_component(mask: Any) -> Any:
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
    erode_ratio: float = _ERODE_RATIO,
) -> Head:
    """
    Голова целиком: силуэт сегментатора, ограниченный областью головы.

    :param image: BGR numpy.ndarray
    :param points: сетка mediapipe того же кадра
    :param model: имя модели rembg
    :param erode_ratio: подрезка края силуэта, доля высоты лица
    :return: Head с альфой, отрезком среза шеи и высотой лица
    """
    import numpy as np

    region, neck_line, face_height = head_region(
        points, image.shape[:2], width_ratio, hair_ratio, neck_ratio
    )

    # Эрозия меряется от лица, а не в абсолютных пикселях: одни и те же «два
    # пикселя» на превью съедают прядь целиком, а на 4K не делают ничего.
    erode_px = max(1, round(face_height * erode_ratio)) if erode_ratio > 0 else 0
    alpha = clean_alpha(silhouette(image, model), erode_px)

    head = largest_component(np.where(region > 0, alpha, 0).astype(np.uint8))

    area = int(np.count_nonzero(head > 127))
    meta = {
        "model": model,
        "face_height": round(face_height, 1),
        "erode_px": erode_px,
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
