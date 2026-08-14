"""
Пересадка головы из генерации в шаблон.

Зачем это нужно. Генеративный редактор (FLUX.2 и подобные) не умеет инпейнт по
маске: он переписывает кадр целиком и возвращает СВОЮ сцену. Замер на 21 паре:
вне маски головы кадр расходится с шаблоном на 22 уровня яркости в среднем,
тогда как текущий путь через SDXL даёт 2.6 — то есть шум перекодирования. Для
книги это неприемлемо: шаблон нарисован художником и обязан дойти до печати без
изменений.

При этом ЛИЧНОСТЬ такой редактор переносит заметно лучше. Отсюда задача модуля:
взять из генерации ровно голову, всё остальное вернуть из шаблона побитово.

ПОЧЕМУ НЕЛЬЗЯ ПРОСТО ВКЛЕИТЬ ПО ШАБЛОННОЙ МАСКЕ. Генерация рисует голову не там
и не того размера: замерено смещение подбородка на 13-16 px и высота лица на
10-13% меньше шаблонной. Маска, построенная по голове ШАБЛОНА, срежет такую
голову с одного края и оставит на другом кольцо чужого фона. Поэтому сначала
подобие, и только потом вклейка.

ПОЧЕМУ ПОДОБИЕ СТРОИТСЯ ПО ГЕОМЕТРИИ, А НЕ ПО ТОЧКАМ СЕТКИ. Соблазн взять 468
точек и посчитать `estimateAffinePartial2D` проверен и отвергнут: на лицах РАЗНЫХ
людей соответствие точек приблизительное, RANSAC оставляет 27-36% инлайеров и
выдаёт масштаб 0.955 там, где отношение высот лиц требует 1.11. Ошибка масштаба
в 15% — это полголовы мимо маски. Три величины из `face_geometry` (подбородок,
ось головы, высота лица) определяют подобие однозначно и устойчиво, и это ровно
те величины, по которым строится сама маска, — то есть источник геометрии один.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InvalidImageError
from app.core.logging import get_logger
from app.pipelines import head_mask, parsing

log = get_logger(__name__)

# Чем меряется масштаб при посадке головы.
#
#   face  — по высоте лица (подбородок → брови). Лицо садится в размер точно, но
#           череп приезжает свой: у ребёнка помладше он относительно лица
#           крупнее, и голова выходит великоватой для плеч шаблона.
#   head  — по корню из площади силуэта «волосы плюс лицо». Голова садится в
#           размер, но мерка зависит от ПРИЧЁСКИ, а она у нового ребёнка своя:
#           пышные волосы заставят ужать лицо, гладкие — раздуть.
#   blend — среднее геометрическое двух.
#
# Значение выбрано замером на 21 кадре, см. `_scale`.
SCALE_MODE = "head"

# Запас поверх силуэта головы при стирании, доли высоты лица. Нужен только на
# полупрозрачную кромку антиалиасинга, которую разметка относит к фону: сам
# силуэт форму головы уже знает. Рабочая маска расширена на 0.12 + 0.10 — в
# двадцать раз больше, и именно поэтому стирать по ней нельзя.
_ERASE_DILATE_RATIO = 0.03
_ERASE_FEATHER_RATIO = 0.03

# Во сколько раз масштаб генерации может разойтись с шаблоном, прежде чем это
# перестаёт быть подгонкой головы и становится признаком, что модель нарисовала
# другую сцену. Полтора раза — это уже не «чуть мельче», а другой план.
_SCALE_MIN, _SCALE_MAX = 0.5, 1.5

# Предельный поворот головы, градусы. Наклон в пределах десятка градусов
# генератор допускает свободно; больше — он поменял позу, и подобием это не
# чинится.
_ANGLE_LIMIT = 25.0


@dataclass
class Transplanted:
    """
    Готовый кадр и всё, что известно о пересадке.

    :param image: BGR numpy.ndarray размера шаблона
    :param meta: уезжает в отчёт. Когда голова села мимо, смотрят сюда первым
        делом: масштаб, поворот и сдвиг сразу говорят, что пошло не так
    """

    image: Any
    meta: dict = field(default_factory=dict)


def _full_frame_landmarks(image: Any) -> list[tuple[int, int]] | None:
    """
    Сетка в координатах ПОЛНОГО кадра, даже если детектор нашёл лицо не сразу.

    MediaPipe не видит лицо мельче ~20% кадра, а на развороте оно занимает
    10-17%. Запасной путь — увеличенное окно вокруг предполагаемой головы, но
    его координаты обязаны быть пересчитаны обратно: подобие, построенное по
    точкам из кропа, увезёт голову на смещение этого кропа.
    """
    import cv2

    points = head_mask.try_landmarks(image)
    if points is not None:
        return points

    height, width = image.shape[:2]
    x0, y0 = int(width * 0.25), 0
    x1, y1 = int(width * 0.75), int(height * 0.6)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    scale = max(1.0, 900.0 / max(1, max(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    points = head_mask.try_landmarks(crop)
    if points is None:
        return None
    return [(int(x / scale) + x0, int(y / scale) + y0) for x, y in points]


def _own_head(image: Any, points: list[tuple[int, int]]) -> Any:
    """
    Силуэт головы НАШЕГО персонажа: классы HAIR и FACE, связные с его лицом.

    Отсюда берут и размер головы (`head_extent`), и область стирания
    (`head_silhouette`) — источник один, и разойтись им негде.

    :return: 0/1 uint8 либо None, если разметки нет или головы на ней не видно
    """
    import cv2
    import numpy as np

    parsed = parsing.parse(image)
    if parsed is None:
        return None

    hair = (np.asarray(parsed.hair) > 127).astype(np.uint8)
    face = (np.asarray(parsed.face) > 127).astype(np.uint8)
    head = np.maximum(hair, face)
    if not head.any():
        return None

    # Смыкаем разрывы между прядями тем же ядром, что и `_region_from_parsing`,
    # иначе голова распадётся на компоненты на первом же просвете
    close = max(1, int(round(min(head.shape) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, kernel)

    seed = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(seed, [head_mask.face_polygon(points)], 1)
    own = head_mask.component_of(head, seed)
    return head if own is None else own


def head_silhouette(image: Any, points: list[tuple[int, int]], face_height: float) -> Any:
    """
    Где стоит стирать старую голову: её собственный силуэт, и ничего сверх.

    ПОЧЕМУ НЕ РАБОЧАЯ МАСКА. Маска из `head_mask.build` расширена на
    `dilate + feather` — на замеренном кадре это 29 пикселей за силуэт. В этом
    кольце шаблонный фон ЦЕЛ, и стирать его незачем: LaMa вернёт туда свою
    догадку, и на месте чёткого горного хребта окажется размытое пятно. Замер
    трёх вариантов области стирания (стык заплатки с нетронутым шаблоном /
    остаток старой головы в пикселях):

        по всей рабочей маске   18.46 / 75
        по эрозии этой маски    12.97 / 260-714
        по силуэту разметки     см. `SCALE_MODE`-подобный выбор ниже

    Эрозия отвергнута замером: она снимает одинаково во все стороны, а голова
    не круг — где-то срезает лишнее, где-то оставляет волосы прежнего героя.
    Силуэт знает форму, и потому запас поверх него нужен маленький: только на
    полупрозрачную кромку антиалиасинга, которую разметка относит к фону.

    :return: маска 0..255 либо None, если разметки нет
    """
    import cv2
    import numpy as np

    own = _own_head(image, points)
    if own is None:
        return None

    grow = max(1, int(round(face_height * _ERASE_DILATE_RATIO)))
    feather = max(1, int(round(face_height * _ERASE_FEATHER_RATIO)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    mask = cv2.dilate((own * 255).astype(np.uint8), kernel)
    return cv2.GaussianBlur(mask, (2 * feather + 1,) * 2, 0)


def head_extent(image: Any, points: list[tuple[int, int]]) -> float | None:
    """
    Размер ГОЛОВЫ в пикселях: корень из площади силуэта «волосы плюс лицо».

    Существует потому, что `face_height` (подбородок → брови) описывает лицо, а
    не голову. У ребёнка помладше черепная коробка относительно лица заметно
    крупнее, и подгонка по высоте лица раздувает вместе с ней весь череп: на
    замере это дало голову, великоватую для плеч шаблона, при коррекции 1.28.

    Корень из площади, а не высота описанного прямоугольника: прямоугольник
    целиком определяется двумя крайними прядями, и одна торчащая прядь меняет
    его сильнее, чем вся остальная голова.

    :return: масштабный измеритель либо None, если разметки нет
    """
    import numpy as np

    own = _own_head(image, points)
    if own is None:
        return None
    area = int(np.count_nonzero(own))
    return float(np.sqrt(area)) if area else None


def _scale(source: dict, target: dict) -> float:
    """
    Во сколько раз увеличить генерацию, чтобы её голова села в шаблонную.

    ЗАМЕР НА 21 КАДРЕ, по которому выбран режим. Мерилось отношение размера
    головы в готовом кадре к размеру головы шаблона (идеал — единица) и число
    кадров с промахом больше 15% хоть по одной мерке:

        режим   голова: среднее / худшая   промахов >15%
        face          0.988 / 1.314              5
        head          0.977 / 0.904              3
        blend         0.970 / 0.805              6

    `face` даёт худший выброс: на кадре с малышом голова выходит на 31% крупнее
    шаблонной и заметно не по плечам. `blend` оказался не компромиссом, а худшим
    вариантом по числу промахов — ошибки не гасят друг друга, а размазываются по
    всем кадрам. `head` и по среднему, и по худшему, и по числу промахов лучше
    обоих, и глазами тоже: на провальном кадре он ставит 0.97 вместо 1.28.

    ОГОВОРКА ПРО ВТОРУЮ МЕРКУ. Отношение высот ЛИЦА в готовом кадре считалось
    тоже, и там `head` показывал худший результат (0.69). Проверка глазами его не
    подтвердила: дефекта на этом кадре нет. Сетка лица на composite ложится
    неустойчиво — вокруг вклейки лежат пиксели шаблона и растушёвка, — поэтому
    мерка по лицу на готовом кадре шумит и решающей не бралась.
    """
    import numpy as np

    by_face = float(target["face_height"]) / max(1e-6, float(source["face_height"]))
    extent_s, extent_t = source.get("head_extent"), target.get("head_extent")
    if SCALE_MODE == "face" or not extent_s or not extent_t:
        return by_face

    by_head = float(extent_t) / max(1e-6, float(extent_s))
    if SCALE_MODE == "head":
        return by_head
    return float(np.sqrt(by_face * by_head))


def similarity(source: dict, target: dict) -> Any:
    """
    Матрица 2x3, сажающая голову `source` на место головы `target`.

    Три величины задают подобие полностью: отношение размеров — масштаб, угол
    между осями головы — поворот, подбородок — неподвижная точка. Поворот
    берётся именно как угол между осями, а не как разность их наклонов к
    вертикали: второе ломается на кадрах, где голова наклонена больше 90°.

    Чем меряется масштаб — см. `_scale`: у мерки по лицу и у мерки по голове
    разные слабые места, и выбор между ними сделан замером, а не вкусом.
    """
    import numpy as np

    scale = _scale(source, target)
    up_s, up_t = np.asarray(source["up"], float), np.asarray(target["up"], float)
    angle = float(np.arctan2(up_t[0] * up_s[1] - up_t[1] * up_s[0],
                             up_t[0] * up_s[0] + up_t[1] * up_s[1]))

    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)

    chin_s = np.asarray(source["chin"], float)
    chin_t = np.asarray(target["chin"], float)
    shift = chin_t - rotation @ chin_s

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = rotation
    matrix[:, 2] = shift
    return matrix


def transplant(
    template: Any,
    generated: Any,
    dilate_ratio: float,
    feather_ratio: float,
    neck_ratio: float,
    match_tone: bool = True,
    plate: Any = None,
) -> Transplanted:
    """
    Возвращает шаблон, в котором заменена только голова.

    ЧТО ЛОЖИТСЯ ПОД СТАРУЮ ГОЛОВУ. Маска шаблона и маска новой головы совпадают
    не полностью: у прежнего героя своя причёска, у нового своя, и между ними
    остаётся зона, которую надо очистить от старой головы, но накрыть новой
    нечем. На замере это 8819 пикселей из 49874 — шестая часть маски.

    Класть туда выровненную генерацию НЕЛЬЗЯ, и это проверено дорого: подобие
    сдвигает её по вертикали (на замеренном кадре на 11 пикселей), вместе с
    головой уезжают её собственные воротник и плечи, и на стыке шеи выходит два
    воротника вместо одного. Замер: 3458 пикселей результата расходятся с
    шаблоном больше чем на 32 уровня, максимум 207.

    Правильное содержимое там — продолжение шаблонной одежды и фона, то есть
    стирание. Его делает LaMa на GPU-сервере (`/v1/erase`), и результат
    передаётся сюда параметром `plate`.

    :param template: шаблон-разворот, BGR numpy.ndarray. Источник всего, что не
        голова, — и источник побитово
    :param generated: ответ генеративного редактора, BGR numpy.ndarray
    :param match_tone: привести тон генерации к шаблону перед вклейкой. Считается
        по пикселям вне ОБЕИХ масок, где обе картинки изображают одно и то же
    :param plate: шаблон со стёртой по маске головой. None — старая голова
        накрывается генерацией, и на стыке шеи появляется описанное выше
        удвоение; путь оставлен рабочим на случай недоступного сервера, но
        сопровождается предупреждением в логе
    :raises InvalidImageError: голову не нашли на шаблоне или на генерации, либо
        подобие вышло за пределы правдоподобного
    """
    import cv2
    import numpy as np

    if template.ndim != 3 or generated.ndim != 3:
        raise InvalidImageError("Пересадка ждёт цветные изображения")

    height, width = template.shape[:2]
    if generated.shape[:2] != (height, width):
        generated = cv2.resize(generated, (width, height), interpolation=cv2.INTER_LANCZOS4)

    points_t = _full_frame_landmarks(template)
    points_g = _full_frame_landmarks(generated)
    if points_t is None:
        raise InvalidImageError("На шаблоне не найдена голова персонажа")
    if points_g is None:
        raise InvalidImageError("На генерации не найдена голова")

    geometry_t = head_mask.face_geometry(points_t)
    geometry_g = head_mask.face_geometry(points_g)
    geometry_t["head_extent"] = head_extent(template, points_t)
    geometry_g["head_extent"] = head_extent(generated, points_g)
    matrix = similarity(geometry_g, geometry_t)

    scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if not _SCALE_MIN <= scale <= _SCALE_MAX or abs(angle) > _ANGLE_LIMIT:
        # Не подгоняем: подобие за этими пределами означает, что редактор
        # перерисовал сцену, а не голову, и вклейка выдаст коллаж
        raise InvalidImageError(
            "Голова генерации не приводится к шаблонной подобием",
            {"scale": round(scale, 3), "angle": round(angle, 1)},
        )

    aligned = cv2.warpAffine(
        generated, matrix, (width, height),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    # Маска строится по ВЫРОВНЕННОЙ генерации, а не по шаблону: заменяем мы ту
    # голову, которая теперь в кадре, и её контур знает только она сама. Маска
    # шаблона нужна тут же, вторым слагаемым: под ней лежит СТАРАЯ голова, и
    # если её не накрыть, от прежнего героя останется ободок причёски
    head_new = head_mask.build(aligned, dilate_ratio, feather_ratio, neck_ratio)
    head_old = head_mask.build(template, dilate_ratio, feather_ratio, neck_ratio)
    mask = np.maximum(head_new.mask, head_old.mask)

    orphan = int(np.count_nonzero((head_old.mask > 127) & (head_new.mask <= 127)))

    meta = {
        "orphan_px": orphan,
        "plate": plate is not None,
        "scale": round(scale, 3),
        "angle": round(angle, 1),
        "shift": [round(float(matrix[0, 2]), 1), round(float(matrix[1, 2]), 1)],
        "face_height_template": round(geometry_t["face_height"], 1),
        "face_height_generated": round(geometry_g["face_height"], 1),
        "mask_new_px": int(np.count_nonzero(head_new.mask > 127)),
        "mask_old_px": int(np.count_nonzero(head_old.mask > 127)),
        "mask_px": int(np.count_nonzero(mask > 127)),
    }

    if match_tone:
        aligned, tone = _match_tone(template, aligned, mask)
        meta.update(tone)

    if plate is None:
        if orphan:
            log.warning(
                "стирание не передано — под старой головой останется содержимое "
                "генерации, на стыке шеи возможно удвоение воротника",
                extra={"orphan_px": orphan},
            )
        result = _paste(template, aligned, mask)
    else:
        if plate.shape != template.shape:
            raise InvalidImageError(
                "Стёртая подложка другого размера, чем шаблон",
                {"plate": list(plate.shape), "template": list(template.shape)},
            )
        # Два действия по очереди, и порядок важен: сперва снимается старая
        # голова, потом на её место садится новая. Вне обеих масок ни один байт
        # не трогается ни на одном из шагов.
        #
        # Стирание идёт по СИЛУЭТУ головы, а не по рабочей маске: та расширена
        # на 29 пикселей, и в этом кольце шаблонный фон цел — отдавать его LaMa
        # значит менять чёткий хребет на размытое пятно. Разметки может не быть,
        # тогда выбора нет и стираем по маске, как раньше
        erase = head_silhouette(template, points_t, geometry_t["face_height"])
        if erase is None:
            log.info("силуэт головы не построен — стираем по рабочей маске")
            erase = head_old.mask
        meta["erase_px"] = int(np.count_nonzero(erase > 127))

        cleared = _paste(template, plate, erase)
        result = _paste(cleared, aligned, head_new.mask)
    meta["changed_px"] = int(np.count_nonzero(np.any(result != template, axis=2)))
    log.info("голова пересажена в шаблон", extra=meta)
    return Transplanted(image=result, meta=meta)


def _match_tone(template: Any, aligned: Any, mask: Any) -> tuple[Any, dict]:
    """
    Приводит тон генерации к шаблону по пикселям вне маски.

    Та же логика, что в `composite.py::_match_levels` и в `composite()` на
    GPU-сервере: вне маски обе картинки изображают одно и то же, расхождение там
    и есть увод. Здесь она нужна сильнее, чем там: генеративный редактор уводит
    тон заметно больше диффузии по маске.
    """
    import cv2
    import numpy as np

    outside = mask <= 8
    if int(np.count_nonzero(outside)) < 64:
        return aligned, {"matched": False}

    fit = []
    for channel in range(template.shape[2]):
        source = aligned[..., channel][outside].astype(np.float64)
        target = template[..., channel][outside].astype(np.float64)
        variance = float(source.var())
        source_mean, target_mean = float(source.mean()), float(target.mean())
        gain = 1.0 if variance < 4.0 else float(
            ((source - source_mean) * (target - target_mean)).mean() / variance)
        fit.append((gain, target_mean - gain * source_mean))

    if any(not 0.8 <= g <= 1.25 or abs(b) > 24.0 for g, b in fit):
        return aligned, {"matched": False, "match_reason": "out_of_range"}

    corrected = aligned.astype(np.float32)
    for channel, (gain, bias) in enumerate(fit):
        corrected[..., channel] = corrected[..., channel] * gain + bias
    return np.rint(np.clip(corrected, 0, 255)).astype(np.uint8), {
        "matched": True,
        "match_gain": [round(g, 3) for g, _ in fit],
        "match_bias": [round(b, 1) for _, b in fit],
    }


def _paste(template: Any, aligned: Any, mask: Any) -> Any:
    """
    Вклейка по растушёванной маске с побитовым возвратом шаблона снаружи.

    Растушёвка уже заложена в саму маску (`feather_ratio`), поэтому здесь только
    веса. Обещание «вне маски не тронут ни один пиксель» держится не арифметикой
    с нулевым весом, а явным копированием: округление не должно решать.
    """
    import numpy as np

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = np.rint(
        template.astype(np.float32) * (1.0 - alpha) + aligned.astype(np.float32) * alpha
    ).astype(np.uint8)
    untouched = mask == 0
    blended[untouched] = template[untouched]
    return blended
