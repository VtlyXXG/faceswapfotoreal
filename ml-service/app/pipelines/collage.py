"""
Шаг 1 замены лица: фото-аппликация головы донора на шаблон.

Что переносится. Голова целиком — лицо вместе с причёской, её цветом, длиной и
структурой. Силуэт даёт сегментатор (`segmentation.py`), а не сетка лица: сетка
про волосы ничего не знает. Раньше переносился только овал лица, и причёска
оставалась от нарисованного персонажа — требование изменилось.

Как переносится. Преобразованием подобия: поворот, единый масштаб, сдвиг.
Четыре степени свободы вместо шести у полного аффина — это не экономия, а
запрет: аффин подогнал бы донора под форму чужого лица, растянув его по одной
оси, и узнаваемость исчезла бы ровно на этом шаге. Матрица вида [s·R | t]
такого выразить не может.

Порядок шагов внутри:

  1. сетки mediapipe для фотографии и для обложки;
  2. вырезка головы донора по силуэту сегментатора;
  3. **мимика** (`expression.py`) — точка расширения: выражение лица меняется
     здесь, на вырезанной голове, до переноса в шаблон;
  4. перенос подобием в координаты обложки;
  5. LAB-коррекция тона — только по коже: волосы обязаны сохранить свой цвет;
  6. стирание причёски нарисованного персонажа, если она торчит из-под
     вклеенной головы;
  7. композит.

Наружу отдаётся не только картинка, но и геометрия для маски второго шага:
силуэт вклейки, контур лица (его инпейнтингу трогать нельзя) и линия среза шеи.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InvalidImageError, MLServiceError, NoFaceDetectedError
from app.core.logging import get_logger
from app.pipelines import expression, mask_generator, segmentation

log = get_logger(__name__)

# Опорные точки совмещения — жёсткий каркас лица: углы глаз, спинка и крылья
# носа, углы рта, подбородок, скулы у ушей, внешние края бровей. Считать
# преобразование по всем 468 точкам сетки нет смысла: щёки и губы подвижны, а
# лишние точки только тянут посадку за мимикой донора. Волосы в совмещении не
# участвуют вовсе — голова сажается по лицу, причёска едет следом.
_ALIGN_POINTS = (
    33, 133, 362, 263,  # внешние и внутренние углы глаз
    168, 6, 195, 4, 1,  # спинка носа сверху вниз до кончика
    98, 327,  # крылья носа
    61, 291,  # углы рта
    152,  # подбородок
    234, 454,  # края скул на уровне ушей
    70, 300,  # внешние края бровей
)

# Растушёвка края аппликации, доля высоты лица. Коллаж намеренно жёсткий: это
# ровно та ширина, которая убирает ступеньку антиалиасинга по контуру волос.
# Настоящее сведение с фоном — работа второго шага.
_FEATHER_RATIO = 0.01

# Приведение тона кожи к шаблону (среднее и разброс по каналам LAB). Считается
# и применяется ТОЛЬКО по лицу: волосы должны сохранить исходный цвет, ради
# этого их и переносят. 1.0 — полностью тон шаблона, 0.0 — как на фотографии.
_COLOUR_MATCH = 0.8

# Ниже этого масштаба уменьшать фотографию в один проход warpAffine нельзя:
# билинейная выборка пропускает пиксели и волосы рассыпаются на алиасинг.
_PRESCALE_THRESHOLD = 0.99

# Радиус, на который стирание причёски шаблона затягивается окружающим фоном.
# Доля высоты лица: Telea тянет цвет от границы дыры внутрь, и слишком большой
# радиус даёт мыло, слишком маленький — не закрывает.
_INPAINT_RADIUS_RATIO = 0.04


@dataclass
class Collage:
    """Результат первого шага: аппликация и геометрия для маски второго."""

    image: Any  # BGR numpy.ndarray — шаблон с вклеенной головой
    head_alpha: Any  # силуэт вклейки в координатах шаблона
    face_polygon: Any  # контур лица донора после переноса — зона, которую нельзя трогать
    neck_line: tuple[tuple[int, int], tuple[int, int]]  # срез шеи в координатах шаблона
    erased: Any  # маска стёртой причёски персонажа (нули, если не стиралась)
    meta: dict = field(default_factory=dict)


def _landmarks(image: Any, role: str) -> list[tuple[int, int]]:
    """
    Сетка лица с пометкой, чей это кадр.

    Лицо ищется в обоих кадрах, и без детали причина отказа неотличима: 422 на
    обложке означает «пришлите другой шаблон», 422 на фотографии — «переснимите».
    """
    try:
        return mask_generator.face_landmarks(image)
    except NoFaceDetectedError as exc:
        raise NoFaceDetectedError(f"Не найдено лицо: {role}", {"image": role}) from exc


def similarity_transform(source: Any, target: Any) -> tuple[Any, float]:
    """
    Преобразование подобия source → target методом наименьших квадратов.

    Классическое решение Умеямы: центрируем оба набора точек, оптимальный
    поворот берём из SVD ковариации, масштаб — как отношение разбросов. Матрица
    получается ровно вида [s·R | t], поэтому переносимая голова может только
    повернуться и изменить размер целиком; ни растяжения по оси, ни сдвига
    (shear) такая матрица выразить не в состоянии.

    Альтернатива — cv2.estimateAffinePartial2D — считает то же самое, но через
    RANSAC со случайными выборками: результат меняется от запуска к запуску, а
    для одинаковых входов пайплайн обязан давать одинаковый коллаж.

    :param source: точки-источники Nx2
    :param target: соответствующие им точки-приёмники Nx2
    :return: (матрица 2x3 float64, масштаб)
    """
    import numpy as np

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centred, dst_centred = src - src_mean, dst - dst_mean

    variance = float((src_centred**2).sum() / len(src))
    if variance <= 0:
        raise InvalidImageError(
            "Опорные точки лица вырождены — преобразование не определено",
            {"points": len(src)},
        )

    u, singular, vt = np.linalg.svd(dst_centred.T @ src_centred / len(src))

    # Отражения быть не должно: детерминант < 0 означает зеркальное лицо
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[1, 1] = -1

    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() / variance)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix, scale


def _warp(image: Any, matrix: Any, scale: float, size: tuple[int, int]) -> Any:
    """
    Переносит кадр донора в систему координат шаблона.

    Лицо на фотографии обычно крупнее, чем на обложке, то есть перенос — это
    уменьшение. Уменьшать одним warpAffine нельзя: интерполяция берёт отдельные
    отсчёты и на коэффициенте вроде 0.3 просто выбрасывает две трети пикселей,
    оставляя рваные пряди волос. Поэтому сначала честное усреднение INTER_AREA
    до нужного размера, а уже потом поворот и сдвиг — тогда в warpAffine
    остаётся масштаб ~1 и алиасингу взяться неоткуда.

    Работает и с трёхканальным кадром, и с одноканальной альфой.
    """
    import cv2
    import numpy as np

    width, height = size
    flags = cv2.INTER_LANCZOS4

    if scale < _PRESCALE_THRESHOLD:
        small_w = max(1, round(image.shape[1] * scale))
        small_h = max(1, round(image.shape[0] * scale))
        image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # Уменьшение уже выполнено — из матрицы его надо убрать
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        matrix[:, :2] /= scale
        flags = cv2.INTER_LINEAR

    return cv2.warpAffine(image, matrix, (width, height), flags=flags, borderValue=0)


def _transform_points(points: Any, matrix: Any) -> Any:
    """Тот же перенос для контура и отрезка: точки, а не пиксели."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    return np.rint(pts @ matrix[:, :2].T + matrix[:, 2]).astype(np.int32)


def _match_skin(donor: Any, template: Any, skin: Any, ratio: float) -> Any:
    """
    Подгоняет тон кожи под лицо шаблона: среднее и разброс по каналам LAB.

    LAB, а не BGR: там яркость отделена от цвета, поэтому подгонка тона кожи не
    задевает светотеневой рисунок лица — а он и есть геометрия, которую нельзя
    трогать.

    Ключевое отличие от прежней версии — коррекция взвешивается маской кожи и
    **не касается волос**. Их цвет переносится ради того, чтобы он остался
    цветом заказчика; подтянуть его к палитре персонажа означало бы перекрасить
    донора в нарисованного героя.

    :param skin: маска кожи uint8, она же вес коррекции
    """
    import cv2
    import numpy as np

    selection = skin > 127
    if ratio <= 0 or not selection.any():
        return donor

    src = cv2.cvtColor(donor, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(template, cv2.COLOR_BGR2LAB).astype(np.float32)

    matched = src.copy()
    for channel in range(3):
        src_values = src[..., channel][selection]
        dst_values = dst[..., channel][selection]

        # Разброс растягивается только если он есть: на ровной заливке остаётся
        # один сдвиг среднего — тон подогнать всё равно нужно.
        src_std = float(src_values.std())
        gain = float(dst_values.std()) / src_std if src_std > 1e-6 else 1.0

        matched[..., channel] = (src[..., channel] - float(src_values.mean())) * gain + float(
            dst_values.mean()
        )

    # Смешивание в BGR, а не в LAB, ровно ради волос: обратный перевод
    # LAB → BGR не побитовый, и пиксели с нулевым весом уехали бы на единицу-две
    # просто оттого, что их прогнали через цветовое пространство. «Волосы не
    # тронуты» должно означать «не тронуты», а не «почти».
    corrected = cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    weight = (skin.astype(np.float32) / 255.0 * ratio)[..., None]
    blended = donor.astype(np.float32) * (1.0 - weight) + corrected.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def _erase_template_head(
    target: Any,
    points: list[tuple[int, int]],
    pasted: Any,
    face_height: float,
    model: str,
) -> tuple[Any, Any, dict]:
    """
    Стирает причёску нарисованного персонажа там, где её не закрыла вклейка.

    Без этого шага перенос причёски виден насквозь: у персонажа с длинными
    волосами вокруг вклеенной головы остаётся его собственная шевелюра, и на
    развороте оказывается два человека сразу. Понизить strength и попросить
    модель убрать её нельзя — на 0.2 она ничего не убирает, только подкрашивает.

    Дыра затягивается Telea по окружающему фону: получается размытое пятно,
    которое затем попадает в маску инпейнтинга и там дорисовывается мазком. Это
    компромисс — идеально восстановить фон за головой локально невозможно.

    Отказ сегментатора на обложке не фатален: рисованный персонаж — не тот
    материал, на котором учили U²-Net, и вероятность промаха здесь выше, чем на
    фотографии. Поэтому шаг пропускается с предупреждением, а не роняет заказ.

    :return: (изображение с затянутой дырой, маска стирания, метаданные)
    """
    import cv2
    import numpy as np

    empty = np.zeros(target.shape[:2], dtype=np.uint8)
    try:
        head = segmentation.cutout_head(target, points, model)
    except MLServiceError as exc:
        log.warning(
            "причёску персонажа стереть не удалось — сегментатор не нашёл голову",
            extra={"cause": exc.message},
        )
        return target, empty, {"erased_ratio": None}

    # Запас вокруг вклейки: стирать вплотную к ней нельзя, иначе Telea затянет
    # дыру цветом самой вклейки и по контуру волос пойдёт ореол.
    grow = max(1, round(face_height * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    covered = cv2.dilate((pasted > 127).astype(np.uint8), kernel)

    erased = np.where((head.alpha > 127) & (covered == 0), 255, 0).astype(np.uint8)
    # Крошка по краям силуэта — не причёска, а погрешность сегментации
    erased = cv2.morphologyEx(erased, cv2.MORPH_OPEN, kernel)

    head_px = max(1, int(np.count_nonzero(head.alpha > 127)))
    ratio = float(np.count_nonzero(erased)) / head_px
    meta = {"erased_ratio": round(ratio, 3), "template_head_px": head_px}

    if not erased.any():
        return target, empty, meta

    radius = max(1, round(face_height * _INPAINT_RADIUS_RATIO))
    filled = cv2.inpaint(target, erased, radius, cv2.INPAINT_TELEA)

    log.info("причёска персонажа стёрта", extra=meta)
    if ratio > 0.5:
        # Половина головы персонажа мимо вклейки — фон за ней Telea честно
        # восстановить не сможет, и на 0.2 модель это не спасёт.
        log.warning(
            "стёрта большая часть головы персонажа — фон придётся домысливать",
            extra=meta,
        )
    return filled, erased, meta


def build(
    source: Any,
    target: Any,
    emotion: str = "",
    model_photo: str = "u2net_human_seg",
    model_cover: str = "u2net",
    width_ratio: float = segmentation._WIDTH_RATIO,
    hair_ratio: float = segmentation._HAIR_RATIO,
    neck_ratio: float = segmentation._NECK_RATIO,
    feather_ratio: float = _FEATHER_RATIO,
    colour_match: float = _COLOUR_MATCH,
    erase_template_head: bool = True,
) -> Collage:
    """
    Вклеивает голову донора в шаблон и возвращает коллаж для инпейнтинга.

    :param source: BGR-фотография заказчика
    :param target: BGR-иллюстрация-шаблон
    :param emotion: имя трансформера мимики; пусто — нейтральное выражение
    :param model_photo: модель сегментации для фотографии
    :param model_cover: модель сегментации для обложки
    :param colour_match: доля приведения тона кожи к шаблону, 0..1
    :param erase_template_head: стирать ли причёску персонажа из-под вклейки
    """
    import cv2
    import numpy as np

    if not 0.0 <= colour_match <= 1.0:
        raise InvalidImageError(
            "Доля приведения цвета должна лежать в диапазоне 0..1",
            {"colour_match": colour_match},
        )

    source_points = _landmarks(source, "фотография заказчика")
    target_points = _landmarks(target, "обложка")

    head = segmentation.cutout_head(
        source, source_points, model_photo, width_ratio, hair_ratio, neck_ratio
    )

    # Мимика — до переноса: на вклеенной голове её правка поехала бы вместе с
    # фоном обложки. Нейтральное выражение проходит насквозь без затрат.
    face = expression.transform(
        expression.Face(image=source, alpha=head.alpha, points=source_points), emotion
    )

    matrix, scale = similarity_transform(
        [face.points[i] for i in _ALIGN_POINTS],
        [target_points[i] for i in _ALIGN_POINTS],
    )

    height, width = target.shape[:2]
    warped = _warp(face.image, matrix, scale, (width, height))
    alpha = _warp(face.alpha, matrix, scale, (width, height))

    face_polygon = _transform_points(mask_generator.face_polygon(face.points), matrix)

    # Все доли маски и растушёвок меряются от лица НА ШАБЛОНЕ: именно его
    # размер определяет, сколько пикселей занимает стык на этой обложке.
    template_polygon = mask_generator.face_polygon(target_points)
    template_face_height = float(template_polygon[:, 1].max() - template_polygon[:, 1].min())

    # Край аппликации: чуть размыть, чтобы контур волос не пилило антиалиасингом
    alpha = mask_generator.soften(alpha, template_face_height, 0.0, feather_ratio)

    # Кожа = лицо внутри силуэта. Волосы сюда не попадают, и коррекция их не
    # трогает — в этом весь смысл переноса причёски.
    skin = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(skin, [face_polygon], 255)
    skin = cv2.GaussianBlur(np.minimum(skin, alpha), (0, 0), max(1.0, template_face_height * 0.03))
    warped = _match_skin(warped, target, skin, colour_match)

    base, erased, erase_meta = (
        _erase_template_head(target, target_points, alpha, template_face_height, model_cover)
        if erase_template_head
        else (target, np.zeros((height, width), dtype=np.uint8), {"erased_ratio": None})
    )

    weight = (alpha.astype(np.float32) / 255.0)[..., None]
    image = (warped.astype(np.float32) * weight + base.astype(np.float32) * (1.0 - weight)).astype(
        np.uint8
    )

    neck_line = _transform_points(np.array(head.neck_line), matrix)
    meta = {
        "scale": round(scale, 3),
        "rotation_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 2),
        "emotion": (emotion or expression.NEUTRAL).strip().lower(),
        "colour_match": colour_match,
        "segmenter": head.meta["model"],
        # Насколько силуэт заполнил отведённый эллипс головы. Близко к нулю —
        # сегментатор промахнулся, близко к единице — причёска упёрлась в
        # границу области, и часть волос могла остаться за кадром вклейки.
        "head_fill": head.meta["fill"],
        "head_px": int(np.count_nonzero(alpha > 127)),
        "face_height_target": round(template_face_height, 1),
        **erase_meta,
    }
    log.info("аппликация собрана", extra={**meta, "image_size": f"{width}x{height}"})

    if scale > 1.0:
        log.warning(
            "лицо на фотографии мельче, чем на обложке — вклейка растянута",
            extra={"scale": meta["scale"]},
        )

    return Collage(
        image=image,
        head_alpha=alpha,
        face_polygon=face_polygon,
        neck_line=(tuple(neck_line[0]), tuple(neck_line[1])),
        erased=erased,
        meta=meta,
    )
