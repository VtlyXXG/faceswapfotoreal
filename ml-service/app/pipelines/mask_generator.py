"""
Сетка лица MediaPipe и маска инпейнтинга для вклеенной аппликации.

Смысл маски перевернулся вместе с пайплайном. Пока лицо рисовала модель, белым
отмечалось само лицо — зона, которую надо перерисовать. Теперь лицо приходит
готовым, оригинальными пикселями фотографии, и трогать его нельзя ни в какой
степени. Инпейнтингу остаётся только стык:

  1. **внешний контур волос** — кольцо вдоль силуэта вклейки, где фотография
     встречается с живописью обложки;
  2. **срез шеи** — прямая, по которой голова оторвана от плеч. Самое заметное
     место аппликации, и полоса здесь самая широкая;
  3. **стёртая причёска персонажа** — если она торчала из-под вклейки, её
     затянули фоном локально, и модели надо положить туда мазок.

Всё остальное чёрное. Лицо защищено отдельно и явно: контур из сетки
растушёвывается и **вычитается** из маски. Черты оказываются под нулём, то есть
недоступны модели вовсе, а у самой кромки челюсти маска гасится лишь частично —
иначе фотографическая щека легла бы на живопись встык, без единого пикселя на
переход.

Размеры задаются долями высоты лица, а не пикселями: обложки приходят и в 4K, и
превью-размером, одно и то же ядро работает на них совершенно по-разному.
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

# Доли высоты лица (подбородок → линия бровей).
#
# edge — половина ширины кольца вдоль силуэта волос. Кольцо симметрично: часть
#   ложится на края прядей, часть на фон обложки, и переход модель ведёт по
#   обеим сторонам стыка.
# neck — толщина полосы на срезе шеи. Заметно шире кольца: оторванную шею надо
#   не сгладить, а спрятать, дорисовав воротник или тень под подбородком.
# guard — растушёвка защиты лица. Черты под ней недоступны модели полностью.
# feather — спад 255 → 0 по краям всей зоны.
_EDGE_RATIO = 0.05
_NECK_RATIO = 0.35
_GUARD_RATIO = 0.06
_FEATHER_RATIO = 0.06


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

    Используется и для зоны инпейнтинга, и для края самой аппликации (там доли
    микроскопические — растушёвка нужна лишь чтобы убрать ступеньку
    антиалиасинга по контуру волос).

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


def face_guard(
    shape: tuple[int, int],
    polygon: Any,
    face_height: float,
    guard_ratio: float = _GUARD_RATIO,
) -> Any:
    """
    Защита лица: растушёванный контур, который потом вычитается из маски.

    Размытие здесь симметрично относительно контура — ровно тот профиль, что
    когда-то мешал маске лица и оказался нужен тут. Черты лежат глубоко внутри и
    получают 255, то есть полную защиту. У самой кромки челюсти значение падает
    до половины, и настолько же приоткрывается зона инпейнтинга: без этого
    фотографическая щека встречалась бы с живописью встык, и переход было бы
    некуда вести.

    :param polygon: контур лица Nx2 в координатах кадра
    :param guard_ratio: ширина растушёвки, доля высоты лица
    """
    import cv2
    import numpy as np

    if guard_ratio < 0:
        raise InvalidImageError(
            "Ширина защиты лица не может быть отрицательной", {"guard_ratio": guard_ratio}
        )

    guard = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(guard, [np.asarray(polygon, dtype=np.int32)], 255)

    blur = round(face_height * guard_ratio)
    if blur > 0:
        guard = cv2.GaussianBlur(guard, (2 * blur + 1, 2 * blur + 1), 0)
    return guard


def gradient(mask: Any, face_height: float, gradient_ratio: float) -> Any:
    """
    Превращает маску-плато в конус: 255 на стыке, линейный спад в обе стороны.

    Зачем это понадобилось. Пока strength держали около 0.2, форма маски внутри
    почти не имела значения: модель всё равно едва касалась пикселей, и разница
    между «обрабатывается на 100%» и «на 60%» была неразличима. На 0.45-0.55
    она перерисовывает открытое по-настоящему, и плато означает, что вся
    область под ним переписана одинаково сильно — а по его границе идёт
    ступенька ровно той высоты, на которую подняли strength.

    Спад строится через distanceTransform, а не гауссианом: гауссиан размывает
    и сам стык, теряя на нём полные 255, а расстояние даёт честный конус —
    вершина точно на стыке, склон точно заданной ширины.

    :param mask: бинарная (или почти) маска стыка
    :param gradient_ratio: ширина склона, доля высоты лица
    :return: маска uint8 с градиентом
    """
    import cv2
    import numpy as np

    if gradient_ratio < 0:
        raise InvalidImageError(
            "Ширина градиента не может быть отрицательной",
            {"gradient_ratio": gradient_ratio},
        )

    reach = round(face_height * gradient_ratio)
    if reach <= 0:
        return mask

    core = (np.asarray(mask) > 127).astype(np.uint8)
    if not core.any():
        return mask

    # Расстояние до стыка считается по фону: DIST_L2 с маской 3x3 — приближение,
    # но на ширинах в десятки пикселей его погрешность меньше пикселя
    distance = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3)
    slope = np.clip(1.0 - distance / float(reach), 0.0, 1.0)

    return np.maximum(core * 255, (slope * 255).astype(np.uint8))


def blend_mask(
    shape: tuple[int, int],
    head_alpha: Any,
    face_polygon: Any,
    neck_line: tuple[tuple[int, int], tuple[int, int]],
    erased: Any,
    face_height: float,
    edge_ratio: float = _EDGE_RATIO,
    neck_ratio: float = _NECK_RATIO,
    guard_ratio: float = _GUARD_RATIO,
    feather_ratio: float = _FEATHER_RATIO,
    gradient_ratio: float = 0.0,
) -> Any:
    """
    Зона инпейнтинга для вклеенной аппликации: контур волос, шея, следы стирания.

    Кольцо вдоль силуэта берётся как разность дилатации и эрозии — полоса
    одинаковой ширины по обе стороны от контура, независимо от его формы. Прядям
    волос это подходит куда лучше, чем любой полигон: контур там рваный, и
    описать его многоугольником нельзя.

    :param head_alpha: силуэт вклеенной головы в координатах шаблона
    :param face_polygon: контур лица донора после переноса — его защищаем
    :param neck_line: отрезок среза шеи в координатах шаблона
    :param erased: маска стёртой причёски персонажа (может быть пустой)
    :param face_height: высота лица на шаблоне, база для всех долей
    :param gradient_ratio: ширина градиента от стыка наружу; 0 — прежнее плато
        со спадом по краю. Нужен на высоком strength, см. `gradient`
    :return: одноканальная маска uint8 размера shape
    """
    import cv2
    import numpy as np

    if edge_ratio < 0 or neck_ratio < 0 or feather_ratio < 0 or gradient_ratio < 0:
        raise InvalidImageError(
            "Доли маски не могут быть отрицательными",
            {
                "edge": edge_ratio,
                "neck": neck_ratio,
                "feather": feather_ratio,
                "gradient": gradient_ratio,
            },
        )

    solid = (np.asarray(head_alpha) > 127).astype(np.uint8)

    edge = max(1, round(face_height * edge_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    ring = cv2.subtract(cv2.dilate(solid, kernel), cv2.erode(solid, kernel)) * 255

    # Срез шеи: полоса поперёк отрезка. Толщина здесь несоизмерима с кольцом —
    # шов от оторванной шеи не сглаживают, а закрывают: модель должна дорисовать
    # на этом месте воротник, тень или прядь.
    neck = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, round(face_height * neck_ratio))
    cv2.line(neck, tuple(neck_line[0]), tuple(neck_line[1]), 255, thickness)

    mask = np.maximum(np.maximum(ring, neck), np.asarray(erased, dtype=np.uint8))

    if gradient_ratio > 0:
        # Градиент заменяет растушёвку, а не дополняет: и то и другое описывает
        # край зоны, только конус делает это шире и линейно
        mask = gradient(mask, face_height, gradient_ratio)
    else:
        # soften, а не просто размытие: кольцо вдоль волос узкое, и симметричный
        # гауссиан сбил бы его пик заметно ниже 255 — зона стыка открылась бы
        # модели лишь частично. Здесь спад целиком уходит наружу, а сам стык
        # остаётся полностью доступным.
        mask = soften(mask, face_height, 0.0, feather_ratio)

    # Вычитание защиты — последним действием: что бы ни попало в зону раньше,
    # лицо из неё выпадает. Порядок здесь и есть гарантия, ради которой всё
    # затевалось.
    guard = face_guard(shape, face_polygon, face_height, guard_ratio)
    mask = (mask.astype(np.float32) * (1.0 - guard.astype(np.float32) / 255.0)).astype(np.uint8)

    log.info(
        "маска стыка построена",
        extra={
            "image_size": f"{shape[1]}x{shape[0]}",
            "face_height": round(face_height, 1),
            "edge_px": edge,
            "neck_px": thickness,
            "gradient_px": round(face_height * gradient_ratio),
            "open_px": int(np.count_nonzero(mask > 127)),
        },
    )
    return mask
