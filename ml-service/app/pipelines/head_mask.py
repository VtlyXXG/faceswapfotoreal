"""
Маска головы **на шаблоне**: единственная локальная работа пайплайна.

Смена парадигмы. Раньше этот слой обслуживал аппликацию: голова заказчика
вырезалась по силуэту, вклеивалась в обложку, а полдюжины масок описывали стыки
вокруг вклейки — кольцо по контуру волос, полоса на шее, дыра в фоне на месте
чужой причёски. Масок было много, потому что задача была одна и та же —
спрятать шов между фотографией и живописью.

Теперь шва нет. Голова заказчика никуда не переносится; в шаблоне вырезается
область головы персонажа, и новая голова рисуется в ней целиком, в материале
сцены, по референсу личности. Маска поэтому описывает не стык, а рабочую
область: где модели можно рисовать.

Область здесь одна — вся голова. Соседний модуль `hair_mask.py` строит вторую,
узкую: одни только волосы, без лица. Общего у них столько, что весь разбор кадра
(сетка, разметка, опорная геометрия, выбор компоненты своего персонажа) живёт
здесь и оттуда вызывается — публичные `try_landmarks`, `face_geometry`,
`parsed_geometry`, `component_of` и `head_ellipse` существуют ради этого.

Форму задаёт **семантика, а не геометрия**. Область — это классы разметки
(`parsing.py`, selfie_multiclass), собранные по одному правилу:

    маска = (HAIR ∪ FACE ∪ SKIN), связные с головой,   минус CLOTHES и BACKGROUND

Что это даёт. Маска сама растекается по всей открытой коже — по шее, по груди в
вырезе футболки — и сама останавливается там, где начинается ткань. Ни одного
числа для этого не нужно: границу знает не наш коэффициент, а модель сегментации.

Почему нельзя было обойтись обрезкой по Y. Одежда персонажа — часть шаблона и
обязана дойти до печати без единого изменения: инпейнт перерисовывает всё, что
под маской, вместе с полосками рубашки и шевронами на жилетке. Но и просто
срезать маску под подбородком нельзя: у персонажа в футболке с открытым воротом
кожа шеи и груди останется вне маски, а тон нового лица модель подберёт свой —
и шов двух оттенков кожи ляжет на самом видном месте. Ткань и кожа лежат под
подбородком вперемешку, и разделить их может только классификатор.

Кожа берётся не вся. В кадре есть ещё руки, а руки — первый источник артефактов
у любой диффузии, и перерисовывать их не просили. Отсекаются они тем же
семантическим правилом: рукав — это CLOTHES, он разрывает связность, и до кисти
маска не доходит. Для случая без рукавов есть предел `_SKIN_REACH_RATIO`.

Поверх области идут только два числа: **расширение** `dilate_ratio` (убирает
остатки старой причёски по краю) и **растушёвка** `feather_ratio` (убирает
видимую границу генерации). Оба — наружу, в фон; со стороны ткани их гасит
защита одежды, которая применяется последней.

Разметки может не быть (веса необязательны) — тогда работает запасной путь на
чистой геометрии: эллипс головы по сетке лица плюс полоса шеи с жёстким срезом
по глубине. Он заведомо грубее и одежду не различает, поэтому и режется
консервативно. Сетки может не быть тоже: mediapipe не находит лицо мельче ~20%
кадра, а на разворотах в полный рост оно именно такое. Тогда работает разметка
без сетки. Отказ приходит, только когда молчат оба.

Размеры задаются долями высоты лица, а не пикселями: обложки приходят и в 4K, и
превью-размером, одно и то же ядро работает на них совершенно по-разному.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.core.logging import get_logger
from app.pipelines import parsing

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
_BROW_ARC = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)

_CHIN = 152
_CHEEK_LEFT, _CHEEK_RIGHT = 234, 454

# Геометрия эллипса головы — запасной путь, когда разметки нет. Доли лица:
# hair — сколько высот лица отложить вверх от подбородка (канон — две до
# макушки, плюс запас на объём причёски), width — ширина в долях ширины лица.
_ELLIPSE_HAIR_RATIO = 2.3
_ELLIPSE_WIDTH_RATIO = 1.7

# Полоса шеи: полуширина у подбородка в долях ширины челюсти и во сколько раз
# она расходится книзу. Расхождение маленькое и намеренно: полоса описывает
# ШЕЮ, а не плечи. Прежние 1.5 накрывали воротник и верх жилетки, и инпейнт
# переписывал бы одежду персонажа вместе с полосками и шевронами.
_NECK_HALF_RATIO = 0.55
_NECK_FLARE = 1.15

# Запас над тканью, доля высоты лица. Одежда вычитается не встык, а с полем:
# по краю воротника идёт полупрозрачная кромка антиалиасинга, и маска,
# доведённая до неё вплотную, всё равно зацепит первый ряд пикселей ткани.
_CLOTHES_MARGIN_RATIO = 0.03

# Докуда от головы берётся открытая кожа, в высотах лица. Единственное
# геометрическое ограничение семантического пути, и стоит оно ради кистей рук:
# у персонажа в майке рукав не разрывает связность, и без предела маска дошла бы
# по голому плечу до пальцев. Груди и шеи хватает 2.5 высот лица с запасом.
_SKIN_REACH_RATIO = 2.5

# Класс FACE разметки покрывает кожу лица от подбородка до линии роста волос;
# высота лица в наших долях меряется от подбородка до бровей, то есть примерно
# три четверти этой высоты. Множитель нужен только на пути без сетки лица.
_PARSED_FACE_TO_HEIGHT = 0.75


@dataclass
class HeadMask:
    """
    Рабочая область на шаблоне и всё, что о ней известно.

    :param mask: одноканальная маска uint8 размера кадра
    :param face_height: высота лица персонажа в пикселях — база всех долей
    :param meta: чем построена, сколько открыто, где центр. Уезжает в
        X-Swap-Meta: когда результат вышел странным, смотрят сюда первым делом
    """

    mask: Any
    face_height: float
    meta: dict = field(default_factory=dict)


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
    плюс 10 на зрачки. Полигоны используют только индексы из первых 468.

    :param image: BGR numpy.ndarray
    :raises NoFaceDetectedError: лица нет или оно мельче порога детектора
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


def try_landmarks(image: Any) -> list[tuple[int, int]] | None:
    """Сетка или None: на разворотах в полный рост её отсутствие — норма."""
    try:
        return face_landmarks(image)
    except NoFaceDetectedError:
        log.info("сетка лица не построена — маска пойдёт по семантической разметке")
        return None


def face_size(image: Any) -> float | None:
    """
    Высота лица на снимке в пикселях. None — лица не видно.

    Нужна не маске, а подготовке референса (`reference.py`): чтобы приравнять
    масштаб лица заказчика к масштабу лица персонажа, обе высоты надо измерить
    одной и той же линейкой. Отказ здесь — не ошибка, а рабочий исход: у
    фотографии в профиль сетки не будет, и поле тогда кладётся вслепую.
    """
    points = try_landmarks(image)
    if points is None:
        return None
    try:
        return float(face_geometry(points)["face_height"])
    except InvalidImageError:
        return None


def face_polygon(points: list[tuple[int, int]]) -> Any:
    """Замкнутый контур лица: дуга челюсти + линия бровей вместо лба."""
    import numpy as np

    return np.array(
        [points[i] for i in _JAW_ARC] + [points[i] for i in _BROW_ARC], dtype=np.int32
    )


def face_geometry(points: list[tuple[int, int]]) -> dict:
    """Опорная геометрия лица: подбородок, ось головы, размеры."""
    import numpy as np

    chin = np.asarray(points[_CHIN], dtype=np.float64)
    brows = np.asarray([points[i] for i in _BROW_ARC], dtype=np.float64).mean(axis=0)

    up = brows - chin
    face_height = float(np.linalg.norm(up))
    if face_height < 1.0:
        raise InvalidImageError("Лицо на шаблоне вырождено: нулевая высота")
    up = up / face_height

    left = np.asarray(points[_CHEEK_LEFT], dtype=np.float64)
    right = np.asarray(points[_CHEEK_RIGHT], dtype=np.float64)

    return {
        "chin": chin,
        "up": up,
        # Поперечная ось строится поворотом продольной, а не по скулам: при
        # наклоне головы линия скул и линия «подбородок → брови» не
        # перпендикулярны, и маска уезжает вбок вслед за этой погрешностью.
        "side": np.array([-up[1], up[0]]),
        "face_height": face_height,
        "face_width": float(np.linalg.norm(right - left)),
    }


def parsed_geometry(parsed: parsing.Parsed) -> dict | None:
    """
    То же самое, но по разметке: путь для лиц, которых не видит детектор.

    Ось считается вертикальной. Наклон головы по одной лишь маске кожи лица
    восстановим, но ненадёжно — момент инерции у пятна с волосами и ушами
    смещён, — а ошибка оси разворачивает полосу шеи мимо шеи.
    """
    import cv2
    import numpy as np

    face = (np.asarray(parsed.face) > 127).astype(np.uint8)
    if not face.any():
        return None

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(face, connectivity=8)
    if count < 2:
        return None

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y = int(stats[largest, cv2.CC_STAT_TOP])
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])

    return {
        "chin": np.array([centroids[largest][0], y + height], dtype=np.float64),
        "up": np.array([0.0, -1.0]),
        "side": np.array([1.0, 0.0]),
        "face_height": max(1.0, height * _PARSED_FACE_TO_HEIGHT),
        "face_width": float(max(1, width)),
        "seed": (labels == largest).astype(np.uint8),
    }


def component_of(mask: Any, seed: Any) -> Any | None:
    """
    Те компоненты маски, на которых лежит затравка. 0/1 uint8 либо None.

    Ответ на вопрос «чьё это» — единственный способ отличить нашего персонажа от
    соседа по развороту: классы разметки приходят одним слоем на всех героев
    сразу и сами по себе никого друг от друга не отделяют.
    """
    import cv2
    import numpy as np

    count, labels = cv2.connectedComponents(mask, connectivity=8)
    if count < 2:
        return None

    touched = {int(label) for label in np.unique(labels[seed > 0]) if label}
    if not touched:
        return None

    return np.isin(labels, list(touched)).astype(np.uint8)


def _region_from_parsing(parsed: parsing.Parsed, seed: Any, face_height: float) -> Any:
    """
    Рабочая область целиком: волосы, лицо и ОТКРЫТАЯ КОЖА, связные с головой.

    Правило одно: `(HAIR ∪ FACE ∪ SKIN) минус CLOTHES`, и из полученного берётся
    компонента, к которой относится найденное лицо. Ткань разрывает связность
    сама, поэтому маска доходит ровно до воротника — при глухом вороте
    останавливается под подбородком, при открытом растекается по шее и груди.
    Ни одного коэффициента для этого не нужно.

    Соседи по развороту отсекаются тем же разбором на компоненты: перерисовывать
    причёску второго персонажа мы не нанимались.

    Руки отсекаются рукавом — он тоже CLOTHES. Там, где рукава нет (майка,
    голые плечи), связность их не остановит, и включается единственное
    геометрическое ограничение этого пути: кожа берётся не дальше
    `_SKIN_REACH_RATIO` высот лица от головы. Предел стоит там, где кончается
    грудь и начинаются кисти: перерисованные пальцы — классический артефакт
    диффузии, и получить его на развороте страшнее, чем недотянуть маску.
    """
    import cv2
    import numpy as np

    hair = (np.asarray(parsed.hair) > 127).astype(np.uint8)
    face = (np.asarray(parsed.face) > 127).astype(np.uint8)
    skin = (np.asarray(parsed.skin) > 127).astype(np.uint8)
    clothes = (np.asarray(parsed.clothes) > 127).astype(np.uint8)

    head = np.maximum(hair, face)
    if not head.any():
        return None

    # Класс волос местами рвётся на пряди — смыкаем их прежде, чем разбирать
    # голову на компоненты, иначе связь с лицом теряется на первом же просвете
    close = max(1, int(round(min(head.shape) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, kernel)

    # Голова НАШЕГО персонажа, а не весь класс волос: на развороте героев
    # несколько, и радиус, отложенный от всех голов сразу, открыл бы кожу вокруг
    # чужой. Затравки может не хватить — разметка нарисованного лица иногда
    # уходит в SKIN целиком, — и тогда работаем по классу целиком, как прежде.
    own = component_of(head, seed)
    if own is None:
        log.info("голова не опознана затравкой — радиус кожи идёт от всего класса волос")
        own = head

    # Кожа ограничивается окрестностью головы — см. про кисти в докстринге.
    # Радиус меряется от головы, а не от подбородка: у персонажа в профиль
    # подбородок смещён, и круг от него срезал бы грудь с одной стороны
    # Через distanceTransform, а не дилатацию: радиус здесь в сотни пикселей, и
    # морфология с ядром такого размера считается минутами, тогда как
    # преобразование расстояния — один проход по кадру
    reach = max(1.0, face_height * _SKIN_REACH_RATIO)
    distance = cv2.distanceTransform((own == 0).astype(np.uint8), cv2.DIST_L2, 3)
    near_head = (distance <= reach).astype(np.uint8)

    region = cv2.morphologyEx(
        np.maximum(own, np.minimum(skin, near_head)), cv2.MORPH_CLOSE, kernel
    )

    # Ткань вычитается ДО разбора на компоненты, а не после: в этом весь смысл.
    # Она и есть та граница, по которой область разваливается на «голова с
    # шеей» и «рука», и, вычтенная позже, разделить их бы уже не смогла.
    region = np.where(clothes > 0, 0, region).astype(np.uint8)
    if not region.any():
        return None

    region = component_of(region, seed)
    return None if region is None else region * 255


def head_ellipse(shape: tuple[int, int], geometry: dict) -> Any:
    """
    Область головы эллипсом по сетке лица — путь без разметки.

    Грубее разметки принципиально: эллипс не знает ни длины волос, ни их формы,
    поэтому берётся с запасом. Запас — меньшее зло: маска чуть шире головы
    заставит модель перерисовать немного фона, маска уже головы оставит по краю
    причёску прежнего героя.
    """
    import cv2
    import numpy as np

    face_height = geometry["face_height"]
    centre = geometry["chin"] + geometry["up"] * (_ELLIPSE_HAIR_RATIO * face_height / 2)
    axes = (
        int(round(_ELLIPSE_WIDTH_RATIO * geometry["face_width"] / 2)),
        int(round(_ELLIPSE_HAIR_RATIO * face_height / 2)),
    )
    # Угол эллипса — наклон продольной оси головы: cv2 меряет его от оси X по
    # часовой стрелке, а «вверх» у нас направлено против неё
    angle = float(np.degrees(np.arctan2(geometry["up"][0], -geometry["up"][1])))

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, (int(centre[0]), int(centre[1])), axes, angle, 0, 360, 255, -1)
    return mask


def _neck_strip(
    shape: tuple[int, int],
    geometry: dict,
    neck_ratio: float,
    parsed: parsing.Parsed | None = None,
) -> Any:
    """
    Полоса под подбородком: место, где новая голова сходится с телом.

    Трапеция, слегка расходящаяся книзу, — и не более того. Расходиться сильнее
    нельзя: под подбородком у персонажа начинается воротник, а за ним плечи, и
    полоса, дотянувшаяся до них, отдаёт инпейнту одежду. Полоски рубашки и
    шевроны на жилетке — часть шаблона, и переписывать их мы не имеем права.

    Там, где есть разметка, полоса вдобавок пересекается с КОЖЕЙ: рисовать надо
    шею, а шея — это кожа. Всё, что полоса зацепила из ткани и фона, из неё
    выпадает ещё до расширения маски.
    """
    import cv2
    import numpy as np

    if neck_ratio <= 0:
        return np.zeros(shape, dtype=np.uint8)

    chin, up, side = geometry["chin"], geometry["up"], geometry["side"]
    half = _NECK_HALF_RATIO * geometry["face_width"]
    depth = neck_ratio * geometry["face_height"]

    # Верх полосы приподнят над подбородком: там она смыкается с областью
    # головы, и зазора между ними быть не должно
    top = chin + up * (0.1 * geometry["face_height"])
    bottom = chin - up * depth

    strip = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        strip,
        np.array(
            [
                top + side * half,
                top - side * half,
                bottom - side * half * _NECK_FLARE,
                bottom + side * half * _NECK_FLARE,
            ],
            dtype=np.int32,
        ),
        255,
    )

    if parsed is not None:
        # Кожа тела плюс лицо: у персонажа с поднятым подбородком верх шеи
        # разметка нередко относит к лицу, и без объединения полоса рвётся
        # ровно там, где должна быть сплошной
        skin = np.maximum(np.asarray(parsed.skin), np.asarray(parsed.face))
        strip = np.minimum(strip, (skin > 127).astype(np.uint8) * 255)

    return strip


def _fabric_guard(shape: tuple[int, int], parsed: parsing.Parsed | None, face_height: float) -> Any:
    """
    Защита одежды: то, что вычитается из маски последним действием.

    Вариант «семантический». Класс CLOTHES разметки — это ровно ткань: рубашка,
    воротник, жилетка. Расширяется на `_CLOTHES_MARGIN_RATIO` и размывается,
    поэтому маска гаснет НАД тканью, а не по её кромке: сама ткань не получает
    ни единицы, а переход от маски к нулю укладывается в поле над ней.

    :return: вес 0..1 той же формы, где 1 — «сюда нельзя», либо None
    """
    import cv2
    import numpy as np

    if parsed is None:
        return None

    clothes = (np.asarray(parsed.clothes) > 127).astype(np.uint8) * 255
    if not clothes.any():
        return None

    margin = max(1, int(round(face_height * _CLOTHES_MARGIN_RATIO)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    guard = cv2.GaussianBlur(cv2.dilate(clothes, kernel), (2 * margin + 1,) * 2, 0)

    return guard.astype(np.float32) / 255.0


def _depth_cutoff(
    shape: tuple[int, int], geometry: dict, neck_ratio: float, feather: int
) -> Any:
    """
    Срез по глубине: ниже линии шеи маски нет вообще.

    Вариант «математический», и он нужен даже там, где сработала разметка.
    Причины две. Первая: без весов разметки вычитать одежду нечем, а
    расширение и растушёвка всё равно тянут маску вниз — на 4K это десятки
    пикселей сверх полосы шеи. Вторая: разметка ошибается на открытой груди и
    на светлой ткани, и тогда единственным ограничителем остаётся геометрия.

    Срез идёт перпендикулярно оси головы, то есть вместе с её наклоном, и
    гасится мягко: жёсткая линия по горлу читалась бы как след ножа.

    :return: вес 0..1, где 1 — «здесь маска разрешена»
    """
    import cv2
    import numpy as np

    height, width = shape
    chin, up, side = geometry["chin"], geometry["up"], geometry["side"]
    bottom = chin - up * (neck_ratio * geometry["face_height"])

    # Полуплоскость выше линии среза: четырёхугольник, заведомо перекрывающий
    # кадр в обе стороны. Диагональ кадра — гарантия, что при любом наклоне оси
    # его углы окажутся внутри
    reach = float(height + width)
    keep = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        keep,
        np.array(
            [
                bottom + side * reach,
                bottom - side * reach,
                bottom - side * reach + up * reach,
                bottom + side * reach + up * reach,
            ],
            dtype=np.int32,
        ),
        255,
    )

    # Спад целиком ВНИЗ от линии: сдвигаем полуплоскость на ширину спада вниз и
    # размываем. Так на самой линии остаётся полная сила, а ноль наступает
    # ровно через feather пикселей под ней
    if feather > 0:
        keep = cv2.dilate(
            keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1,) * 2)
        )
        keep = cv2.GaussianBlur(keep, (2 * feather + 1, 2 * feather + 1), 0)

    return keep.astype(np.float32) / 255.0


def build(
    image: Any,
    dilate_ratio: float,
    feather_ratio: float,
    neck_ratio: float,
) -> HeadMask:
    """
    Строит маску головы персонажа на шаблоне.

    :param image: шаблон, BGR numpy.ndarray
    :param dilate_ratio: расширение за контур головы, доля высоты лица
    :param feather_ratio: ширина растушёвки краёв, доля высоты лица
    :param neck_ratio: насколько маска опускается ниже подбородка
    :raises NoFaceDetectedError: персонажа не нашли ни детектором, ни разметкой
    """
    import cv2
    import numpy as np

    if dilate_ratio < 0 or feather_ratio < 0 or neck_ratio < 0:
        raise InvalidImageError(
            "Доли маски головы не могут быть отрицательными",
            {"dilate": dilate_ratio, "feather": feather_ratio, "neck": neck_ratio},
        )

    shape = image.shape[:2]
    points = try_landmarks(image)
    parsed = parsing.parse(image)

    # Источник геометрии и источник формы независимы. Сетка точнее в геометрии
    # (наклон головы, высота лица), разметка точнее в форме (реальный контур
    # причёски), и лучший результат даёт их сочетание — но работать сервис
    # обязан и на любом из двух источников поодиночке.
    geometry = face_geometry(points) if points else None
    seed = None

    if geometry is None:
        fallback = parsed_geometry(parsed) if parsed else None
        if fallback is None:
            raise NoFaceDetectedError(
                "На шаблоне не найден персонаж: сетка лица пуста, разметка тоже",
                {"parsing_available": parsing.available()},
            )
        seed = fallback.pop("seed")
        geometry = fallback

    if parsed is not None and seed is None:
        # Затравка для разбора на компоненты: контур лица из сетки. Именно он
        # отвечает на вопрос «чья это причёска» на развороте с двумя героями.
        seed = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(seed, [face_polygon(points)], 1)

    face_height = geometry["face_height"]

    # Путь первый и основной: форму задают классы разметки. Ни эллипса, ни
    # полосы шеи здесь нет вовсе — область сама доходит до ткани и сама на ней
    # останавливается, а где ткани нет, растекается по открытой коже.
    region = _region_from_parsing(parsed, seed, face_height) if parsed is not None else None
    source = "parsing"

    if region is None:
        # Путь запасной: чистая геометрия. Разметки нет, отличить кожу от ткани
        # нечем, поэтому маска описывается эллипсом с полосой шеи и режется по
        # глубине — консервативно и заведомо грубее.
        region = np.maximum(
            head_ellipse(shape, geometry), _neck_strip(shape, geometry, neck_ratio, parsed)
        )
        source = "ellipse"
        if points is None:
            # Ни формы, ни сетки — эллипс по геометрии из разметки. Работает,
            # но это худший из путей, и в логе он должен быть виден
            log.warning("маска головы построена эллипсом по разметке: точность снижена")

    mask = region

    # Расширение и растушёвка — в этом порядке. Дилатация на dilate + feather,
    # чтобы спад целиком лёг снаружи: сама голова обязана остаться под сплошными
    # 255, иначе модель получит её полупрозрачной и оставит от старого героя
    # призрак.
    dilate = int(round(face_height * dilate_ratio))
    feather = int(round(face_height * feather_ratio))

    grow = dilate + feather
    if grow > 0:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        )
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (2 * feather + 1, 2 * feather + 1), 0)

    # Защита одежды — ПОСЛЕ расширения, и это принципиально. Вычитание ткани из
    # области (в `_region_from_parsing`) само по себе ничего не гарантирует:
    # дилатация с растушёвкой растят маску во все стороны, в том числе на
    # воротник, и на 4K это десятки пикселей. Последнее слово должно остаться за
    # тканью.
    #
    # Требование, которое здесь обслуживается: одежда персонажа — часть шаблона
    # и обязана дойти до печати без единого изменения. Инпейнт перерисовывает
    # всё, что под маской, поэтому единственный способ сохранить полоски и
    # шевроны — не пускать туда маску вовсе.
    grown = mask

    fabric = _fabric_guard(shape, parsed, face_height)
    if fabric is not None:
        mask = (mask.astype(np.float32) * (1.0 - fabric)).astype(np.uint8)
    else:
        # Ткань не размечена — значит, где она, неизвестно, и единственный
        # ограничитель снизу геометрический. Открытую грудь он тоже срежет, но
        # это честный размен: без разметки выбор стоит между «срезать кожу» и
        # «перерисовать одежду», и второе непоправимо.
        mask = (
            mask.astype(np.float32) * _depth_cutoff(shape, geometry, neck_ratio, feather)
        ).astype(np.uint8)

    trimmed_px = int(np.count_nonzero((grown > 127) & (mask <= 127)))
    open_px = int(np.count_nonzero(mask > 127))
    share = open_px / float(shape[0] * shape[1])

    meta = {
        "source": source,
        "landmarks": points is not None,
        "parsing": parsed is not None,
        "face_height": round(face_height, 1),
        "chin": [round(float(v), 1) for v in geometry["chin"]],
        "dilate_px": dilate,
        "feather_px": feather,
        "neck_px": int(round(face_height * neck_ratio)),
        # Чем ограничена маска снизу: ткань по разметке или срез по глубине.
        # Если в результате перерисован воротник, смотреть надо сюда
        "clothes_guard": fabric is not None,
        "trimmed_px": trimmed_px,
        "open_px": open_px,
        "open_share": round(share, 4),
    }
    log.info("маска головы на шаблоне построена", extra=meta)

    if share > 0.5:
        # Маска в пол-кадра означает, что разметка приняла за голову половину
        # разворота. Не отказ — генерация всё равно пройдёт, — но результат
        # будет объяснять именно эта строка
        log.warning("маска головы накрыла больше половины кадра", extra={"open_share": share})

    return HeadMask(mask=mask, face_height=face_height, meta=meta)
