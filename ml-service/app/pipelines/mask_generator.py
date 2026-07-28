"""
Сетка лица MediaPipe и маски инпейнтинга для вклеенной аппликации.

Масок две, потому что работы на холсте две, и силы они требуют
противоположной. Одна маска с одним strength обслужить обе не может: то, что
достаточно для шва, не восстановит фон, а то, что восстановит фон, сотрёт лицо.

  **Зона 1 — лицо донора.** Полная защита. Контур из сетки растушёвывается и
  **вычитается** из обеих масок последним действием. Черты получают ровный
  ноль, то есть недоступны модели вовсе; у самой кромки челюсти маска гасится
  лишь наполовину — иначе фотографическая щека легла бы на живопись встык, без
  единого пикселя на переход.

  **Зона 2 — стыки** (`seam_mask`). Узкое кольцо вдоль контура новых волос и
  узкая полоса на стыке шеи с телом персонажа. Здесь нужен мягкий переход тона
  и цвета кожи от шеи донора к телу шаблона плюс контактная тень — работа на
  низком strength, и трогать что-то кроме самого стыка нельзя.

  **Зона 3 — дыра в фоне** (`hole_mask`). Место, где у персонажа была своя
  причёска, а новая голова его не накрыла: у героя с гривой до плеч это
  половина неба вокруг головы. Локальная заливка оставляет там мыло без
  текстуры живописи, и низким strength его не исправить — фон нужно рисовать
  заново, целиком, вместе с тем, что в нём было. Отсюда высокий strength и
  отдельный проход.

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
# neck — толщина полосы на стыке шеи с телом персонажа. Узкая: шея донора
#   теперь доезжает до воротника, и прятать оторванный край больше не нужно —
#   нужно свести тон кожи и положить контактную тень.
# guard — растушёвка защиты лица. Черты под ней недоступны модели полностью.
# feather — спад 255 → 0 по краям всей зоны.
# Кольцо стыка. Сплошная часть узкая с обеих сторон: внутри волосы заказчика,
# снаружи только что восстановленный фон, и полная сила замыливает и то и
# другое. До зоны фона дотягивается не оно, а градиент за ним — там маска уже
# слабая, и мазок ложится касанием.
_EDGE_RATIO = 0.03
_EDGE_OUTER_RATIO = 0.05
_NECK_RATIO = 0.12
_GUARD_RATIO = 0.06
_FEATHER_RATIO = 0.04

# Отступ зоны фона от вклеенной головы, доля высоты лица. Зона 3 работает на
# strength 0.85 с промптом «фон, никаких голов и волос» — подпустишь её к
# контуру причёски, и она его съест, оставив по краю светлую кайму. Отступ
# должен быть настоящим, а перекрытие зон обеспечивает не он, а внешняя
# половина кольца зоны 2 (edge_outer): она дотягивается сюда сверху.
_HOLE_MARGIN_RATIO = 0.12
_HOLE_FEATHER_RATIO = 0.05

# Зона шеи: её ширина в долях высоты лица, глубина вниз от подбородка и спад по
# краям. Ширина берётся щедрой — рисуется не шея-палка, а шея с плечами.
_NECK_WIDTH_RATIO = 0.75
_NECK_DEPTH_RATIO = 0.85
_NECK_FEATHER_RATIO = 0.05
# Насколько кожа из разметки может выходить за колонну шеи. Ровно чтобы забрать
# грудь и плечи по бокам, но не руки: их перерисовывать незачем.
_NECK_SPREAD_RATIO = 0.25


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


def hole_mask(
    shape: tuple[int, int],
    head_alpha: Any,
    face_polygon: Any,
    erased: Any,
    face_height: float,
    margin_ratio: float = _HOLE_MARGIN_RATIO,
    feather_ratio: float = _HOLE_FEATHER_RATIO,
    guard_ratio: float = _GUARD_RATIO,
) -> Any:
    """
    Зона 3: дыра в фоне на месте стёртой причёски персонажа.

    Открывается модели целиком и на высоком strength — здесь она не сводит, а
    рисует заново. Причина в разнице масштабов: у героя обложки грива до плеч, у
    заказчика ёжик, и вклеенная голова накрывает лишь часть стёртого. Всё
    остальное — небо, ветка, крыло птеродактиля, — оказывается затянуто ровной
    заливкой без единого мазка, и на 0.26 модель кладёт поверх неё только
    касание кисти. Восстановить утраченное можно лишь сгенерировав его заново.

    От вклеенной головы зона отступает на margin: под strength 0.85 всё, что она
    накрывает, исчезает, и подпускать её к контуру новых волос нельзя. Полоса
    между зоной и головой достаётся зоне 2, где сила безопасная.

    :param head_alpha: силуэт вклеенной головы в координатах шаблона
    :param face_polygon: контур лица донора — защищаем и здесь, на всякий случай
    :param erased: маска стёртой головы персонажа
    :param margin_ratio: ширина мягкой защиты вклейки, доля высоты лица. Не
        отступ: зона доходит до волос, но её сила у контура падает до нуля
    :return: одноканальная маска uint8 размера shape
    """
    import cv2
    import numpy as np

    if margin_ratio < 0 or feather_ratio < 0:
        raise InvalidImageError(
            "Доли маски не могут быть отрицательными",
            {"margin": margin_ratio, "feather": feather_ratio},
        )

    hole = (np.asarray(erased) > 127).astype(np.uint8)
    if not hole.any():
        return np.zeros(shape, dtype=np.uint8)

    # Вклейка защищается МЯГКО, а не отступом. Жёсткий отступ оставлял вокруг
    # головы кольцо, куда не доставала ни одна зона: заливка там светлая, и на
    # тёмном фоне она читалась светящимся контуром — тем самым ореолом, который
    # не брали ни градиенты, ни сужение колец. Теперь зона доходит до самых
    # волос, но её сила у контура падает до нуля: съесть причёску 0.85 уже не
    # может, а незакрытой полосы не остаётся.
    margin = max(1, round(face_height * margin_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    guard_paste = cv2.GaussianBlur(
        cv2.dilate((np.asarray(head_alpha) > 0).astype(np.uint8) * 255, kernel),
        (2 * margin + 1, 2 * margin + 1),
        0,
    )

    mask = np.where(hole > 0, 255, 0).astype(np.uint8)
    mask = (mask.astype(np.float32) * (1.0 - guard_paste.astype(np.float32) / 255.0)).astype(
        np.uint8
    )
    if not mask.any():
        return mask

    # Спад внутрь: наружу расширять зону нельзя, она и так граничит с холстом,
    # который трогать не просили.
    #
    # Но только по ВНЕШНЕЙ границе. По внутренней зона упирается во вклейку, и
    # спад там означал бы полосу, не накрытую ни одной зоной: зона 3 уже
    # погасла, зона 2 ещё не началась, и между ними остаётся сырая заливка —
    # тот самый грязный контур вокруг головы. Внутренний край поэтому
    # восстанавливается на полную после растушёвки; накроет его кольцо зоны 2,
    # которое по построению заходит сюда же.
    feather = round(face_height * feather_ratio)
    if feather > 0:
        trim = feather + 1
        shrink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * trim + 1, 2 * trim + 1))
        # Спад только по внешней границе: внутренняя уже сведена мягкой
        # защитой вклейки, и второй спад по ней вернул бы незакрытую полосу
        soft = cv2.GaussianBlur(cv2.erode(mask, shrink), (2 * feather + 1, 2 * feather + 1), 0)
        near_paste = cv2.dilate(
            (np.asarray(head_alpha) > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (margin + 2 * feather) + 1,) * 2),
        )
        mask = np.where(near_paste > 0, mask, soft)

    guard = face_guard(shape, face_polygon, face_height, guard_ratio)
    return (mask.astype(np.float32) * (1.0 - guard.astype(np.float32) / 255.0)).astype(np.uint8)


def neck_mask(
    shape: tuple[int, int],
    head_alpha: Any,
    face_polygon: Any,
    chin: Any,
    axis: tuple,
    face_height: float,
    body_skin: Any = None,
    width_ratio: float = _NECK_WIDTH_RATIO,
    depth_ratio: float = _NECK_DEPTH_RATIO,
    feather_ratio: float = _NECK_FEATHER_RATIO,
    guard_ratio: float = _GUARD_RATIO,
) -> Any:
    """
    Зона 6: шея и открытая грудь персонажа — их модель рисует заново.

    Донора теперь режут строго по челюсти, шея с фотографии не переносится
    вовсе. Причина простая: фотографичная шея на нарисованной груди читалась
    дешёвой аппликацией, и спрятать этот стык нечем — воротник рисованный, кожа
    фотографическая, разница текстур колоссальная. Проще не сводить два
    материала, а нарисовать шею целиком в материале обложки: тогда шва нет,
    есть один непрерывный мазок от подбородка до воротника.

    Зона поэтому накрывает всё, что между подбородком вклейки и одеждой
    персонажа: его нарисованную шею, открытую грудь и полосу под самым
    подбородком. Границы даёт семантическая разметка (`parsing.py`) — по цвету
    отличить кожу от бежевого воротника невозможно, замерено. Без разметки
    остаётся геометрия: полоса под подбородком в ширину челюсти.

    :param chin: подбородок вклейки в координатах шаблона
    :param axis: (up, side) шаблона
    :param body_skin: кожа тела персонажа из разметки; None — только геометрия
    :param depth_ratio: насколько глубоко зона уходит вниз без разметки
    """
    import cv2
    import numpy as np

    # Ось приходит и массивами, и парой кортежей: коллаж кладёт её в метаданные,
    # а те переживают сериализацию
    up = np.asarray(axis[0], dtype=np.float64)
    side = np.asarray(axis[1], dtype=np.float64)
    half = width_ratio * face_height / 2
    depth = depth_ratio * face_height

    # Геометрическая основа: трапеция от подбородка вниз. Она есть всегда и
    # накрывает то место, где шея обязана появиться, даже если разметки нет
    column = np.zeros(shape, dtype=np.uint8)
    top = np.asarray(chin, dtype=np.float64) + up * (0.05 * face_height)
    bottom = np.asarray(chin, dtype=np.float64) - up * depth
    cv2.fillConvexPoly(
        column,
        np.array(
            [
                top + side * half,
                top - side * half,
                bottom - side * half * 1.6,
                bottom + side * half * 1.6,
            ],
            dtype=np.int32,
        ),
        255,
    )

    # Кожа из разметки берётся не вся: руки и кисти персонажа рисовать заново
    # не надо, они и так в материале обложки. Нужна только та кожа, что лежит
    # под подбородком, — шея и грудь. Поэтому пересечение с окрестностью
    # колонны, а не объединение со всей кожей.
    mask = column
    if body_skin is not None:
        spread = max(1, round(face_height * _NECK_SPREAD_RATIO))
        reach = cv2.dilate(
            column, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * spread + 1,) * 2)
        )
        mask = np.maximum(column, np.minimum(np.asarray(body_skin), reach))

    # Вклейку зона не трогает: подбородок и волосы донора рисовать заново не
    # надо, они и есть то, ради чего всё делается
    mask = np.where(np.asarray(head_alpha) > 0, 0, mask).astype(np.uint8)

    feather = round(face_height * feather_ratio)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (2 * feather + 1, 2 * feather + 1), 0)

    # Ещё раз после растушёвки: размытие возвращает зону на подбородок, а на
    # силе этого прохода туда нельзя ни на единицу. Жёсткий край у контура
    # вклейки не страшен — его накрывает кольцо зоны 2.
    mask = np.where(np.asarray(head_alpha) > 0, 0, mask).astype(np.uint8)

    guard = face_guard(shape, face_polygon, face_height, guard_ratio)
    mask = (mask.astype(np.float32) * (1.0 - guard.astype(np.float32) / 255.0)).astype(np.uint8)

    log.info(
        "маска шеи построена",
        extra={
            "semantic": body_skin is not None,
            "open_px": int(np.count_nonzero(mask > 127)),
        },
    )
    return mask


def paste_mask(
    shape: tuple[int, int],
    head_alpha: Any,
    face_polygon: Any,
    face_height: float,
    guard_ratio: float = _GUARD_RATIO,
    guard_strength: float = 0.6,
    inset_ratio: float = 0.04,
) -> Any:
    """
    Зона 4: сама вклейка — её надо перевести из фотографии в живопись.

    До сих пор её не существовало, и это было осознанно: лицо вычиталось из
    масок целиком, «модель до него физически не дотягивается». Гарантия
    сходства при этом железная, но и результат честный — фотографическое лицо
    на картине маслом, склейка видна по фактуре, а не по шву. Никакой
    цветокоррекцией это не лечится: тон можно подогнать, мазок кисти — нет.

    Поэтому зона открывается, но не целиком. Защита лица не вычитается
    полностью, а **ослабляется**: `guard_strength` задаёт, какая её доля
    остаётся. Ноль — лицо открыто наравне с остальным (максимум фактуры,
    минимум гарантий), единица — прежняя полная защита. Промежуточное значение
    оставляет черты под частичной маской: мазок ложится, геометрия держится.

    От внешнего контура зона отступает внутрь на `inset_ratio`: сам контур —
    работа зоны 2 со своей силой, и накладывать поверх него ещё один проход
    значит трогать край волос дважды.

    :param guard_strength: доля защиты лица, которая остаётся; 0..1
    :param inset_ratio: отступ внутрь от контура вклейки, доля высоты лица
    """
    import cv2
    import numpy as np

    if not 0.0 <= guard_strength <= 1.0:
        raise InvalidImageError(
            "Доля защиты лица должна лежать в диапазоне 0..1",
            {"guard_strength": guard_strength},
        )
    if inset_ratio < 0:
        raise InvalidImageError("Отступ внутрь не может быть отрицательным", {"inset": inset_ratio})

    solid = (np.asarray(head_alpha) > 127).astype(np.uint8) * 255
    inset = round(face_height * inset_ratio)
    if inset > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
        solid = cv2.erode(solid, kernel)
    if not solid.any():
        return solid

    guard = face_guard(shape, face_polygon, face_height, guard_ratio)
    weight = 1.0 - guard.astype(np.float32) / 255.0 * guard_strength
    mask = (solid.astype(np.float32) * weight).astype(np.uint8)

    log.info(
        "маска стилизации построена",
        extra={
            "guard_strength": guard_strength,
            "inset_px": inset,
            "open_px": int(np.count_nonzero(mask > 127)),
        },
    )
    return mask


def seam_mask(
    shape: tuple[int, int],
    head_alpha: Any,
    face_polygon: Any,
    neck_line: tuple[tuple[int, int], tuple[int, int]],
    face_height: float,
    edge_ratio: float = _EDGE_RATIO,
    neck_ratio: float = _NECK_RATIO,
    guard_ratio: float = _GUARD_RATIO,
    feather_ratio: float = _FEATHER_RATIO,
    gradient_ratio: float = 0.0,
    edge_outer_ratio: float | None = None,
) -> Any:
    """
    Зона 2: стыки — контур новых волос и место, где шея входит в тело персонажа.

    Кольцо вдоль силуэта берётся как разность дилатации и эрозии — полоса
    одинаковой ширины по обе стороны от контура, независимо от его формы. Прядям
    волос это подходит куда лучше, чем любой полигон: контур там рваный, и
    описать его многоугольником нельзя.

    Следов стирания чужой причёски здесь больше нет: они ушли в зону 3, где для
    них хватает силы. Раньше они попадали сюда, и маска раздувалась на пол-неба
    при том, что сделать с этим небом на 0.26 модель ничего не могла.

    :param head_alpha: силуэт вклеенной головы в координатах шаблона
    :param face_polygon: контур лица донора после переноса — его защищаем
    :param neck_line: отрезок стыка шеи в координатах шаблона
    :param face_height: высота лица на шаблоне, база для всех долей
    :param gradient_ratio: ширина градиента от стыка наружу; 0 — плато со
        спадом по краю. См. `gradient`
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

    # Кольцо намеренно НЕсимметричное. Внутрь от контура лежат волосы заказчика,
    # и туда нужно заходить минимально; наружу лежит заливка на месте чужой
    # причёски, и там кольцо должно дотянуться до зоны фона, иначе между ними
    # останется полоса, которую не трогает ни один проход.
    inner = max(1, round(face_height * edge_ratio))
    outer = max(inner, round(face_height * (edge_outer_ratio or edge_ratio)))

    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1))
    shrink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1))
    ring = cv2.subtract(cv2.dilate(solid, grow), cv2.erode(solid, shrink)) * 255
    edge = inner

    # Полоса на стыке шеи с телом. Узкая: шея донора доезжает до воротника, и
    # закрывать оторванный край больше не нужно — нужно свести тон кожи с телом
    # персонажа и дать модели место под контактную тень.
    neck = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, round(face_height * neck_ratio))
    cv2.line(neck, tuple(neck_line[0]), tuple(neck_line[1]), 255, thickness)

    mask = np.maximum(ring, neck)

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
            "edge_outer_px": outer,
            "neck_px": thickness,
            "gradient_px": round(face_height * gradient_ratio),
            "open_px": int(np.count_nonzero(mask > 127)),
        },
    )
    return mask
