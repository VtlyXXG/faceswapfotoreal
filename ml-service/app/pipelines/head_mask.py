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
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,  # подбородок
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
)

# Верхняя граница — по бровям, от внешнего края одной к внешнему краю другой.
_BROW_ARC = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)

_CHIN = 152
_CHEEK_LEFT, _CHEEK_RIGHT = 234, 454

# Опорные точки позы головы, см. `head_pose`.
#
# Внешний угол глаза — на той же стороне лица, что и скула: пара нужна затем,
# чтобы полоса вдоль щеки знала, где на ЭТОЙ стороне кончается глаз. Пары
# анатомические, не по номеру: 33 лежит на одной щеке с 234, 263 — с 454.
_NOSE_TIP = 1
_FOREHEAD = 10
_EYE_OUTER = {_CHEEK_LEFT: 33, _CHEEK_RIGHT: 263}

# Нижний контур челюсти. Пара ЗЕРКАЛЬНАЯ, и это важнее, чем кажется: в сетке
# зеркало 377 — это 148, а зеркало 150 — 379, поэтому хорда 150→377, выглядящая
# симметричной, на самом деле идёт наискось. Замерено вращением настоящей сетки:
# 150→377 отклоняется от поперечной оси на 18° уже в фас и гуляет от -5° до -25°
# по наклону, тогда как зеркальные 150→379 и 148→377 держатся в пределах 1.2°
# на том же диапазоне. Косая хорда срезала бы одну щеку выше другой.
_JAW_PAIR = (150, 379)

# Базовый уровень вертикального прокси на фронтальном кадре. Отношение
# «лоб→нос» к «нос→подбородок» в фас равно не единице, а примерно 1.4, потому
# что точка 10 лежит на макушке лба, а не на переносице; ноль коэффициента
# наклона обязан приходиться на фас, иначе поправка работала бы всегда.
#
# Замерено по шести живым кадрам: 0.224, 0.103, 0.207, 0.179, 0.160, 0.179.
# Разброс между лицами и есть шумовая полка прокси — отсюда мёртвая зона ниже.
_PITCH_LEVEL = 0.18

# Мёртвая зона коэффициента наклона. Меньше неё — считаем, что наклона нет:
# разброс базового уровня между лицами (±0.06) отвечает примерно ±7° поворота,
# и без этой зоны фронтальные кадры получали бы поправку на пустом месте.
_PITCH_DEADBAND = 0.06

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

    return np.array([points[i] for i in _JAW_ARC] + [points[i] for i in _BROW_ARC], dtype=np.int32)


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


def head_pose(points: list[tuple[int, int]], geometry: dict) -> dict | None:
    """
    Ориентация головы по проекционным пропорциям. None — сетка не годится.

    Точных градусов здесь нет и не нужно: нужны надёжные нормированные веса, по
    которым полоса вдоль щеки решает, насколько её раздвинуть и не пора ли
    убрать совсем. Тяжёлый солвер (`cv2.solvePnP`) ради этого не заводится —
    он требует трёхмерной модели лица и даёт точность, за которую мы здесь не
    платим.

    **Поворот меряется не отношением расстояний, а знаковой полушириной.**
    Разница принципиальная, и она стоила калибровки. Отношение расстояний от
    кончика носа до скул — хоть евклидовых, хоть поперечных — НЕМОНОТОННО: на
    настоящей сетке, повёрнутой по шагам, оно растёт до 30-40°, а дальше падает
    обратно, потому что дальняя скула заходит за силуэт и её проекция
    возвращается к оси. Замер на живом лице: 0.04 (фас), 0.44 (20°), 0.57
    (30°), 0.57 (40°), 0.52 (45°), 0.33 (60°), 0.09 (80°). Порог по такому
    числу срабатывал бы дважды — на повороте и на возврате, — и профиль в 80°
    выглядел бы для него фасом.

    Знаковая проекция скулы на поперечную ось монотонна на всём диапазоне:
    355, 283, 203, 116, 27, -17 пикселей на тех же углах. Ноль означает ровно
    то, что нам нужно знать, — дальняя щека кончилась; минус — она уже за
    силуэтом. Из неё же берётся и ширина полосы, так что одно измерение
    отвечает на оба вопроса, и рассогласоваться им негде.

    :param geometry: оси и размеры лица, см. `face_geometry`
    :return: словарь позы либо None, если сетка короткая или лицо вырождено.

        ``near``/``far`` — направление вдоль поперечной оси (+1 или -1) для
        ближней к камере и дальней щеки; ``near_half``/``far_half`` — их
        полуширины в пикселях, у дальней возможен ноль и минус;
        ``near_eye``/``far_eye`` — туда же спроецированные внешние углы глаз;
        ``yaw`` — 0 в фас, 1 в момент исчезновения дальней щеки (около 43° на
        замеренном лице), знак — в какую сторону повёрнута голова;
        ``pitch`` — 0 в фас, больше нуля голова опущена, с мёртвой зоной
    """
    import numpy as np

    if len(points) <= max(_FOREHEAD, _CHEEK_RIGHT, *_EYE_OUTER.values()):
        return None

    chin, side = geometry["chin"], geometry["side"]

    def lateral(index: int) -> float:
        return float(np.dot(np.asarray(points[index], dtype=np.float64) - chin, side))

    left, right = lateral(_CHEEK_LEFT), lateral(_CHEEK_RIGHT)

    # Ближняя щека — та, чья проекция шире. Определять её по знаку нельзя:
    # знак говорит, с какой стороны кадра лежит точка, а на сильном повороте
    # дальняя скула переходит на сторону ближней, не переставая быть дальней
    near_index, far_index = (
        (_CHEEK_RIGHT, _CHEEK_LEFT) if abs(right) >= abs(left) else (_CHEEK_LEFT, _CHEEK_RIGHT)
    )
    near_lateral = lateral(near_index)
    near = 1.0 if near_lateral >= 0 else -1.0
    near_half = abs(near_lateral)
    if near_half < 1.0:
        return None

    # Дальняя сторона меряется в СВОЁМ направлении, то есть с обратным знаком.
    # Отрицательный результат — не ошибка, а факт: точка ушла за ось, дальней
    # щеки в кадре больше нет
    far_half = -near * lateral(far_index)
    near_eye = abs(lateral(_EYE_OUTER[near_index]))
    far_eye = -near * lateral(_EYE_OUTER[far_index])

    raw = 0.0
    forehead = np.asarray(points[_FOREHEAD], dtype=np.float64)
    nose = np.asarray(points[_NOSE_TIP], dtype=np.float64)
    upper = float(np.linalg.norm(forehead - nose))
    lower = float(np.linalg.norm(nose - np.asarray(points[_CHIN], dtype=np.float64)))
    if upper + lower > 1.0:
        raw = (upper - lower) / (upper + lower) - _PITCH_LEVEL

    pitch = 0.0 if abs(raw) < _PITCH_DEADBAND else raw - np.sign(raw) * _PITCH_DEADBAND

    return {
        "near": near,
        "far": -near,
        "near_half": near_half,
        "far_half": far_half,
        "near_eye": near_eye,
        "far_eye": far_eye,
        "yaw": near * float(np.clip(1.0 - far_half / near_half, 0.0, 1.0)),
        "pitch": float(pitch),
    }


def jaw_line(points: list[tuple[int, int]], geometry: dict) -> tuple[Any, Any] | None:
    """
    Нижний контур челюсти: точка на прямой и нормаль, смотрящая ВНИЗ от лица.

    Существует ради одного запрета: тон щеки нельзя размазывать по шее. Под
    подбородком лежит его собственная тень, освещение там другое, и заливка
    лицевой медианой ставит на шее светлое пятно — внутри маски, то есть
    пережившее вклейку.

    Прямая строится по зеркальной паре точек челюсти и сдвигается так, чтобы
    пройти ниже их обеих и ниже подбородка: контур челюсти выпуклый, и прямая
    через две его точки срезала бы подбородок.

    :return: (точка, нормаль) либо None. Положительная проекция на нормаль
        означает «ниже челюсти»
    """
    import numpy as np

    first, second = _JAW_PAIR
    if len(points) <= max(first, second, _CHIN):
        return None

    left = np.asarray(points[first], dtype=np.float64)
    right = np.asarray(points[second], dtype=np.float64)
    chord = right - left
    length = float(np.linalg.norm(chord))
    if length < 1.0:
        return None

    normal = np.array([-chord[1], chord[0]]) / length
    if float(np.dot(normal, geometry["up"])) > 0:
        normal = -normal

    anchor = max(
        (np.asarray(points[index], dtype=np.float64) for index in (first, second, _CHIN)),
        key=lambda point: float(np.dot(point, normal)),
    )
    return anchor, normal


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

    region = cv2.morphologyEx(np.maximum(own, np.minimum(skin, near_head)), cv2.MORPH_CLOSE, kernel)

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


def _fabric_guard(
    shape: tuple[int, int],
    parsed: parsing.Parsed | None,
    face_height: float,
    region: Any = None,
) -> Any:
    """
    Защита одежды: то, что вычитается из маски последним действием.

    Вариант «семантический». Класс CLOTHES разметки — это ровно ткань: рубашка,
    воротник, жилетка. Расширяется на `_CLOTHES_MARGIN_RATIO` и размывается,
    поэтому маска гаснет НАД тканью, а не по её кромке: сама ткань не получает
    ни единицы, а переход от маски к нулю укладывается в поле над ней.

    ПОЛЕ НЕ ЗАХОДИТ НА ВОЛОСЫ ПЕРСОНАЖА. Запас над тканью нужен затем, что по
    кромке воротника идёт полупрозрачный антиалиасинг, и маска, доведённая до
    неё вплотную, зацепит первый ряд ткани. Но там, где соседом ткани оказалась
    не кожа, а прядь волос, лежащая ПОВЕРХ одежды, поле обходится дорого:
    расширение съедает прядь на несколько пикселей, маска рвёт её поперёк, верх
    уходит под перерисовку, а низ остаётся от прежнего героя — на плече повисает
    тёмный обрубок. Замерено на dino1: поле гасило маску с 0.02 до 0.53-0.84 как
    раз на тех строках, где разметка ещё уверенно говорит HAIR.

    Волосы персонажа перерисовываются в любом случае — они и есть то, что мы
    заменяем, — поэтому запас над ними не защищает ничего. Сама ткань при этом не
    теряет ни единицы защиты: `region` по построению не содержит класса CLOTHES
    (см. `_region_from_parsing`), и на каждом пикселе ткани поле остаётся ровно
    таким, каким было.

    :param region: рабочая область до расширения. Нужна, чтобы снять поле только
        с волос НАШЕГО персонажа, а не всего класса волос на развороте
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
    guard = guard.astype(np.float32) / 255.0

    if region is not None:
        # Именно ПОСЛЕ размытия: до него вычитание не дало бы ничего — классы
        # HAIR и CLOTHES не пересекаются, и снимать поле было бы не с чего. Всё
        # поле над тканью и есть результат расширения с размытием, и снимается
        # ровно оно
        own_hair = (np.asarray(parsed.hair) > 127) & (np.asarray(region) > 127)
        guard = np.where(own_hair, 0.0, guard).astype(np.float32)

    return guard


def _depth_cutoff(shape: tuple[int, int], geometry: dict, neck_ratio: float, feather: int) -> Any:
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
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
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

    fabric = _fabric_guard(shape, parsed, face_height, region)
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
