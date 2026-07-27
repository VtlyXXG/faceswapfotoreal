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

# Срез шеи. Резать по челюсти оказалось ошибкой: голова садилась на обложку без
# шеи и висела в воздухе. Шея нужна — и не куском фиксированной длины, а до
# линии одежды донора, чтобы она легла внахлёст на воротник персонажа и переход
# тона кожи было где вести.
#
# Линия одежды ищется по цвету (см. clothing_line): от подбородка вниз идёт
# кожа, а воротник — это первое, что на неё не похоже. По силуэту её не найти:
# рембг не отличает шею от футболки, а на плечах силуэт расширяется на той же
# высоте, что и шея, — сужения, по которому можно было бы опознать воротник, в
# кадре просто нет.
_NECK_RATIO = None  # None — искать линию одежды; число — жёсткий отступ
_NECK_FALLBACK = 0.55  # если линия не нашлась: столько шеи берём вслепую
_NECK_MIN_RATIO = 0.05  # ближе к подбородку срез не имеет смысла
_NECK_MAX_RATIO = 1.10  # дальше — уже грудь, а не шея
_COLLAR_MARGIN = 0.05  # отступ вверх от найденной линии: воротник не забираем

# Допуск по хроме LAB, на который цвет шеи может отличаться от цвета лица.
# Яркость в проверку не входит вовсе: тень под подбородком гасит L вдвое, а
# каналы a и b держит — на них кожа и отличается от ткани.
_SKIN_CHROMA = (9.0, 12.0)
# Разрыв в полосе кожи, который можно перешагнуть, доля высоты лица. Тень под
# подбородком даёт первые 3-5% не-кожи; обрывать поиск на ней означало бы
# срезать шею целиком.
_SKIN_GAP_RATIO = 0.10

# Ширина шеи в долях расстояния между углами челюсти. Ниже челюсти область
# сужается до этой полосы: прямой срез во всю ширину эллипса забирает вместе с
# шеей плечи и воротник — они лежат на той же высоте, что и шея, по бокам.
_NECK_WIDTH_RATIO = 0.85
_JAW_LEFT, _JAW_RIGHT = 172, 397  # углы нижней челюсти
# Насколько эллипс шеи длиннее самой шеи. Он срезается по линии одежды, и запас
# нужен, чтобы у воротника шея не сходилась на нет.
_NECK_TAPER = 1.35

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


def _axis(points: list[tuple[int, int]]) -> tuple[Any, Any, Any, float]:
    """Система координат головы: подбородок, ось вверх, перпендикуляр, высота лица."""
    import numpy as np

    chin = np.array(points[_CHIN], dtype=np.float64)
    brow = np.array(points[_BROW_MID], dtype=np.float64)

    axis = brow - chin
    face_height = float(np.linalg.norm(axis))
    if face_height < 1.0:
        raise MLServiceError("Вырожденная геометрия лица: голову не выделить")

    up = axis / face_height
    return chin, up, np.array([-up[1], up[0]]), face_height


def clothing_line(
    image: Any,
    alpha: Any,
    points: list[tuple[int, int]],
    fallback: float = _NECK_FALLBACK,
) -> tuple[float, dict]:
    """
    Ищет линию одежды донора: докуда вниз от подбородка идёт кожа.

    Признак — цвет, а не форма. По силуэту воротник не найти: сегментатор не
    отличает шею от футболки, а сужения силуэта на шее может не быть вовсе —
    плечи начинаются на той же высоте, и через них полоса переднего плана
    тянется до края кадра.

    Проверяются только каналы a и b: тень под подбородком гасит яркость вдвое,
    а хрому кожи держит. Первые проценты пути эта тень всё же не проходит
    проверку, поэтому короткие разрывы перешагиваются — иначе срез встал бы
    вплотную к подбородку, ради избавления от чего всё и затевалось.

    Отказывать нельзя ни в одном случае: фотографии присылают заказчики. Кадр
    обрезан под подбородком — берём, сколько есть; свитер под горло — fallback.

    :param image: BGR-кадр донора
    :param alpha: его же бинарный силуэт
    :param points: сетка mediapipe того же кадра
    :return: (отступ вниз от подбородка в долях высоты лица, метаданные поиска)
    """
    import cv2
    import numpy as np

    chin, up, _, face_height = _axis(points)
    height, width = np.asarray(alpha).shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    face = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(face, [mask_generator.face_polygon(points)], 255)
    core = cv2.erode(face, np.ones((15, 15), np.uint8)) > 0
    if not core.any():
        return fallback, {"neck_source": "fallback"}

    # Медиана, а не среднее: в контур лица попадают глаза, брови и губы, и
    # среднее уехало бы на них
    reference = np.median(lab[core], axis=0)

    depth = int(round(_NECK_MAX_RATIO * face_height))
    gap = max(1, int(round(_SKIN_GAP_RATIO * face_height)))

    last_skin = 0
    for step in range(depth + 1):
        x, y = np.rint(chin - up * float(step)).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            break

        pixel = lab[y, x]
        skin = (
            np.asarray(alpha)[y, x] > 127
            and abs(float(pixel[1]) - reference[1]) < _SKIN_CHROMA[0]
            and abs(float(pixel[2]) - reference[2]) < _SKIN_CHROMA[1]
        )
        if skin:
            last_skin = step
        elif step - last_skin > gap:
            break

    meta = {"skin_px": last_skin, "skin_gap_px": gap}
    if last_skin == 0:
        return fallback, {**meta, "neck_source": "fallback"}

    return _clamp_neck(last_skin / face_height - _COLLAR_MARGIN), {
        **meta,
        "neck_source": "skin",
    }


def _clamp_neck(ratio: float) -> float:
    return float(min(_NECK_MAX_RATIO, max(_NECK_MIN_RATIO, ratio)))


def head_region(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    width_ratio: float = _WIDTH_RATIO,
    hair_ratio: float = _HAIR_RATIO,
    neck_ratio: float = _NECK_FALLBACK,
    follow_jaw: bool = True,
    neck_column: bool = True,
) -> tuple[Any, tuple[tuple[int, int], tuple[int, int]], float]:
    """
    Область головы по сетке лица: эллипс, срез снизу и колонна шеи.

    Эллипс поворачивается вместе с головой (угол берётся по линии глаз), иначе
    у наклонённой головы он срезал бы висок с одной стороны и захватывал фон с
    другой.

    Собирается из двух частей, и обе нужны:

      1. **голова** — эллипс, срезанный по дуге челюсти. Дуга, а не прямая:
         плечи лежат на той же высоте, что и подбородок, и прямой срез забирает
         их вместе с воротником — на обложку приезжают два оранжевых угла
         футболки;
      2. **шея** — узкая колонна от лица вниз до линии одежды. Ширина берётся
         от углов челюсти: другой мерки шеи в сетке лица нет.

    Раньше низ резался по челюсти и на этом всё заканчивалось. Голова получалась
    чистой, но садилась на обложку без шеи и висела в воздухе — переход тона от
    кожи к телу персонажа вести было негде.

    :param neck_ratio: докуда опускается колонна шеи, доля высоты лица от
        подбородка (при neck_column=False — просто отступ среза вниз)
    :param follow_jaw: вести срез по дуге челюсти или прямой поперёк оси лица
    :param neck_column: добавлять ли колонну шеи под челюстью
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

    # Дуга опускается на neck_down только когда колонны нет: с колонной шею
    # добавляет она, а дуге остаётся отрезать голову от плеч по своему месту
    arc_drop = 0.0 if neck_column else neck_down

    if follow_jaw:
        jaw = (
            np.array([points[i] for i in mask_generator._JAW_ARC], dtype=np.float64)
            - up * arc_drop
        )
        # Куда продлевать концы дуги, зависит от того, с какой стороны лица
        # лежит её начало: порядок обхода в сетке фиксирован, знак
        # перпендикуляра — нет
        outward = side if float(np.dot(jaw[0] - chin, side)) > 0 else -side
        first, last = jaw[0] + outward * reach, jaw[-1] - outward * reach
        cut = [first, *jaw, last]
    else:
        flat = chin - up * arc_drop
        cut = [flat + side * reach, flat - side * reach]

    below = np.array(
        [*cut, cut[-1] - up * reach, cut[0] - up * reach],
        dtype=np.int32,
    )
    cv2.fillPoly(region, [below], 0)  # дуга невыпуклая — fillConvexPoly здесь соврёт

    # Колонна шеи: возвращает под челюсть полосу до линии одежды. Верх уводится
    # внутрь лица, чтобы между головой и шеей не осталось щели там, где дуга
    # челюсти поднимается к ушам.
    #
    # Не прямоугольник, а эллипс со срезанным низом. Прямоугольник давал по
    # бокам шеи две вертикальные прямые во всю её длину — на живописи такая
    # линия читается как наклейка, и мягким инпейнтингом её не убрать: прямая
    # длиной в треть лица слишком заметна для strength 0.26. У эллипса бока
    # сужаются к воротнику, как и положено шее.
    neck_half = _NECK_WIDTH_RATIO * _jaw_width(points) / 2
    if neck_column and neck_down > 0 and neck_half > 0:
        top = chin + up * (0.2 * face_height)
        bottom = chin - up * neck_down
        span = float(np.linalg.norm(top - bottom))

        # Эллипс намеренно длиннее нужного и срезается по линии одежды: иначе
        # у самого воротника шея сходилась бы на нет
        neck_centre = (top + bottom) / 2.0
        cv2.ellipse(
            region,
            (int(neck_centre[0]), int(neck_centre[1])),
            (int(round(neck_half)), int(round(span * _NECK_TAPER / 2))),
            roll,
            0,
            360,
            255,
            -1,
        )
        flat = np.array(
            [
                bottom + side * reach,
                bottom - side * reach,
                bottom - side * reach - up * reach,
                bottom + side * reach - up * reach,
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(region, flat, 0)

    # Отрезок среза — по ширине шеи, а не эллипса: шов проходит там, где шея
    # донора встречается с телом персонажа, и полоса маски ложится туда же.
    neck_point = chin - up * neck_down
    half = max(neck_half, width_ratio * face_width / 6)
    neck_line = (
        tuple(np.round(neck_point + side * half).astype(int)),
        tuple(np.round(neck_point - side * half).astype(int)),
    )
    return region, neck_line, face_height


def _jaw_width(points: list[tuple[int, int]]) -> float:
    """Расстояние между углами нижней челюсти — мерка ширины шеи."""
    import numpy as np

    left = np.array(points[_JAW_LEFT], dtype=np.float64)
    right = np.array(points[_JAW_RIGHT], dtype=np.float64)
    return float(np.linalg.norm(right - left))


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
    neck_ratio: float | None = _NECK_RATIO,
    erode_ratio: float = _ERODE_RATIO,
) -> Head:
    """
    Голова с шеей: силуэт сегментатора, ограниченный областью головы.

    :param image: BGR numpy.ndarray
    :param points: сетка mediapipe того же кадра
    :param model: имя модели rembg
    :param neck_ratio: отступ среза вниз от подбородка; None — искать линию
        одежды по силуэту (`clothing_line`)
    :param erode_ratio: подрезка края силуэта, доля высоты лица
    :return: Head с альфой, отрезком среза шеи и высотой лица
    """
    import numpy as np

    # Эрозия меряется от лица, а не в абсолютных пикселях: одни и те же «два
    # пикселя» на превью съедают прядь целиком, а на 4K не делают ничего.
    _, _, _, face_height = _axis(points)
    erode_px = max(1, round(face_height * erode_ratio)) if erode_ratio > 0 else 0
    alpha = clean_alpha(silhouette(image, model), erode_px)

    # Силуэт нужен до построения области: по нему ищется линия одежды, а по ней
    # проходит срез. Порядок обратный прежнему, где область строилась вслепую.
    neck_meta: dict = {"neck_source": "fixed"}
    if neck_ratio is None:
        neck_ratio, neck_meta = clothing_line(image, alpha, points)

    region, neck_line, face_height = head_region(
        points, image.shape[:2], width_ratio, hair_ratio, neck_ratio
    )
    head = largest_component(np.where(region > 0, alpha, 0).astype(np.uint8))

    area = int(np.count_nonzero(head > 127))
    meta = {
        "model": model,
        "face_height": round(face_height, 1),
        "erode_px": erode_px,
        "neck_ratio": round(float(neck_ratio), 3),
        **neck_meta,
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
