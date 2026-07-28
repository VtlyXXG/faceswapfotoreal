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
  6. стирание головы нарисованного персонажа целиком и заливка её места фоном
     обложки — фон при этом берётся только из фона, сам персонаж источником
     цвета не служит;
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

# Биометрические мерки для масштаба — пары точек сетки, расстояние между
# которыми меряется на обоих лицах. Причёски и габаритов головы здесь нет:
# масштаб считается строго по лицу.
_SCALE_MARKS = {
    "pupils": (468, 473),  # зрачки; есть только при refine_landmarks
    "eyes": (33, 263),  # внешние углы глаз
    "cheeks": (234, 454),  # скулы на уровне ушей
    "jaw": (172, 397),  # углы нижней челюсти
    "face_height": (152, 9),  # подбородок → переносица
}

# На сколько высот лица разрешено опустить вклейку, чтобы шея дотянулась до
# воротника. Сдвиг нужен, когда шеи на фотографии мало: голова, посаженная
# строго по лицу, повисает над воротником. Но и уводить лицо далеко от того
# места, где его нарисовал художник, нельзя — отсюда потолок.
_ANCHOR_MAX_RATIO = 0.25

# Растушёвка края аппликации, доля высоты лица. Коллаж намеренно жёсткий: это
# ровно та ширина, которая убирает ступеньку антиалиасинга по контуру волос.
# Настоящее сведение с фоном — работа второго шага.
#
# Спад ведётся ВНУТРЬ силуэта. Симметричная растушёвка (или тем более
# mask_generator.soften, который сначала расширяет область) вернула бы наружу
# те самые пиксели фона фотографии, ради которых силуэт подрезали эрозией в
# segmentation.clean_alpha, — только с половинной прозрачностью.
_FEATHER_RATIO = 0.01

# Приведение тона кожи к шаблону (среднее и разброс по каналам LAB). Считается
# и применяется ТОЛЬКО по лицу: волосы должны сохранить исходный цвет, ради
# этого их и переносят. 1.0 — полностью тон шаблона, 0.0 — как на фотографии.
_COLOUR_MATCH = 0.8

# Ниже этого масштаба уменьшать фотографию в один проход warpAffine нельзя:
# билинейная выборка пропускает пиксели и волосы рассыпаются на алиасинг.
_PRESCALE_THRESHOLD = 0.99

# Радиус, на который стирание причёски шаблона затягивается окружающим фоном.
# Доля высоты лица; используется вариантами telea и ns.
_INPAINT_RADIUS_RATIO = 0.04

# Способ локального восстановления фона под стёртой причёской персонажа.
# pyramid — заливка пирамидой (push-pull), telea и ns — cv2.inpaint.
_ERASE_METHOD = "pyramid"
_ERASE_METHODS = ("pyramid", "telea", "ns")

# Насколько прямой срез области стирания опускается ниже подбородка персонажа,
# доля высоты лица. Небольшой запас на погрешность сетки — и только: ниже лежит
# шея персонажа, а она нужна. Вклеенная голова обрезана по челюсти, своей шеи у
# неё нет, и садится она ровно на нарисованную; сотрёшь — под подбородком
# останется дыра, которую модели придётся заполнять воротником с нуля.
_ERASE_NECK_RATIO = 0.05

# Запас вокруг силуэта персонажа, доля высоты лица. Стирать впритык нельзя:
# по краю рисованных волос идёт полупрозрачная кромка, и она остаётся тёмной
# каймой ровно там, где её должно было не стать.
_ERASE_PAD_RATIO = 0.03

# Порог «здесь есть персонаж» для маски стирания. Он намеренно много ниже
# segmentation._ALPHA_SOLID: там решается, что взять В аппликацию, и сомнение
# трактуется в пользу фона; здесь — что убрать С обложки, и сомнение
# трактуется в пользу стирания. Любой намёк на персонажа — не фон.
_FOREIGN_FLOOR = 20


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


def biometric_ratios(source: list, target: list) -> dict[str, float]:
    """
    Во сколько раз лицо шаблона больше лица донора — по каждой мерке отдельно.

    Мерки независимы, и на рисованном персонаже они расходятся: у него глаза
    вдвое больше человеческих, а лицо короче. На spread_08 расхождение доходит
    до 18% — зрачки дают 1.60, скулы 1.48, челюсть 1.43, высота лица 1.35.
    Поэтому «масштабировать по биометрии» — это не одно число, а выбор, какой
    мерке верить; выбор живёт в ML_HEAD_SCALE_MARK, а сами числа уезжают в
    метаданные, чтобы расхождение было видно на конкретном заказе.

    Волос ни в одной мерке нет и быть не может: все точки — из сетки лица.
    """
    import numpy as np

    ratios = {}
    for mark, (first, second) in _SCALE_MARKS.items():
        if max(first, second) >= min(len(source), len(target)):
            continue  # зрачки есть только при refine_landmarks
        src = np.linalg.norm(np.array(source[first], float) - np.array(source[second], float))
        dst = np.linalg.norm(np.array(target[first], float) - np.array(target[second], float))
        if src > 1e-6:
            ratios[mark] = float(dst / src)
    return ratios


def biometric_scale(ratios: dict[str, float], mark: str) -> float | None:
    """
    Масштаб по выбранной мерке. None — оставить решение методу наименьших
    квадратов (`umeyama`), то есть подгонку сразу по всем 18 опорным точкам.

    Медиана здесь не «на всякий случай»: она устойчива к одной уехавшей мерке,
    а уезжает на стилизованном лице обычно ровно одна — глаза.
    """
    import numpy as np

    if mark == "umeyama":
        return None
    if not ratios:
        raise InvalidImageError("Биометрические мерки не посчитаны", {"mark": mark})
    if mark == "median":
        return float(np.median(list(ratios.values())))
    if mark not in ratios:
        raise InvalidImageError(
            "Неизвестная биометрическая мерка",
            {"mark": mark, "available": [*sorted(ratios), "median", "umeyama"]},
        )
    return ratios[mark]


def similarity_transform(
    source: Any, target: Any, scale_override: float | None = None
) -> tuple[Any, float]:
    """
    Преобразование подобия source → target методом наименьших квадратов.

    Классическое решение Умеямы: центрируем оба набора точек, оптимальный
    поворот берём из SVD ковариации, масштаб — как отношение разбросов. Матрица
    получается ровно вида [s·R | t], поэтому переносимая голова может только
    повернуться и изменить размер целиком; ни растяжения по оси, ни сдвига
    (shear) такая матрица выразить не в состоянии.

    Опорные точки — только лицевые (`_ALIGN_POINTS`): углы глаз, нос, рот,
    подбородок, скулы, брови. Ни причёска, ни габариты головы в масштаб не
    входят вовсе — ни при каком значении scale_override.

    Альтернатива — cv2.estimateAffinePartial2D — считает то же самое, но через
    RANSAC со случайными выборками: результат меняется от запуска к запуску, а
    для одинаковых входов пайплайн обязан давать одинаковый коллаж.

    :param source: точки-источники Nx2
    :param target: соответствующие им точки-приёмники Nx2
    :param scale_override: взять масштаб отсюда, а не из МНК. Поворот и привязка
        к центру лица остаются прежними — меняется только размер.
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
    if scale_override is not None:
        if scale_override <= 0:
            raise InvalidImageError(
                "Масштаб должен быть положительным", {"scale": scale_override}
            )
        scale = float(scale_override)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix, scale


def _lowest_after_transform(alpha: Any, matrix: Any, down: Any) -> Any:
    """
    Самая дальняя точка силуэта в направлении `down` — после переноса.

    Считается без самого переноса. Матрица подобия линейна, поэтому «какой
    пиксель окажется ниже всех» решается одним скалярным произведением в
    координатах фотографии: максимум `⟨p, Rᵀ·down⟩`. Гонять ради этого warpAffine
    по 4K-кадру незачем — тем более что сдвиг должен войти в ту же матрицу, то
    есть быть известен до переноса.

    :param down: единичный вектор «вниз» в координатах шаблона
    :return: точка в координатах ШАБЛОНА либо None, если силуэт пуст
    """
    import numpy as np

    ys, xs = np.nonzero(np.asarray(alpha) > 127)
    if ys.size == 0:
        return None

    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    direction = linear.T @ np.asarray(down, dtype=np.float64)

    lowest = int(np.argmax(xs * direction[0] + ys * direction[1]))
    point = np.array([xs[lowest], ys[lowest]], dtype=np.float64)
    return point @ linear.T + np.asarray(matrix, dtype=np.float64)[:, 2]


def neck_anchor(
    target: Any,
    target_points: list,
    alpha: Any,
    matrix: Any,
    face_height: float,
    max_ratio: float = _ANCHOR_MAX_RATIO,
) -> tuple[float, dict]:
    """
    На сколько опустить вклейку, чтобы шея донора дошла до воротника шаблона.

    Совместить одним преобразованием подобия и лицо, и низ шеи невозможно:
    длина шеи на фотографии своя, у персонажа своя. Лицо важнее — по нему и
    считается матрица, — а низ шеи после этого оказывается где придётся. Если он
    оказался выше воротника, между шеей и телом остаётся зазор, и голова висит в
    воздухе; сюда и добавляется сдвиг.

    Низ меряется по **самой альфе**, а не по расчётному отрезку среза. Разница
    принципиальная, и она стоила боевого прогона: когда линия одежды донора не
    нашлась и взят запасной отступ (`neck_source: fallback`), отрезок среза
    оказывается на 0.55 высоты лица ниже подбородка независимо от того, есть ли
    там пиксели. Якорь видел эту обещанную длину, считал, что шея уже достаёт до
    воротника, и не двигал ничего — а на обложке висела голова с коротким
    обрубком шеи. По альфе такого не случается: где вклейка кончается, там она и
    кончается.

    Сдвиг только вниз и только по оси лица шаблона. Вверх двигать нечего: если
    шея уже перекрыла воротник — это нахлёст, ровно то, что нужно.

    Линия одежды персонажа ищется тем же способом, что и у донора, но по
    сплошной альфе: сегментатор для этого не запускается — второй прогон на 4K
    стоит секунд, а нужна здесь только граница кожи и ткани.

    :param alpha: силуэт вырезанной головы в координатах ФОТОГРАФИИ
    :param matrix: преобразование донор → шаблон
    :param face_height: высота лица на шаблоне
    :return: (сдвиг в пикселях вдоль оси лица вниз, метаданные)
    """
    import numpy as np

    from app.pipelines import segmentation

    chin, up, _, _ = segmentation._axis(target_points)

    solid = np.full(target.shape[:2], 255, dtype=np.uint8)
    collar_ratio, collar_meta = segmentation.clothing_line(target, solid, target_points)

    if collar_meta["neck_source"] != "collar":
        # Найденного воротника нет: кожа упёрлась в край кадра, не кончилась
        # вовсе или мерить было нечего. Сдвиг двигает лицо заказчика по холсту,
        # и делать это по догадке нельзя — пусть лучше останется зазор, его
        # хотя бы видно и он достаётся зоне стыка
        log.info("якорь шеи пропущен: воротник персонажа не найден", extra=collar_meta)
        return 0.0, {"anchor_px": 0.0, "anchor_collar": collar_meta["neck_source"]}

    collar = chin - up * (collar_ratio * face_height)

    bottom = _lowest_after_transform(alpha, matrix, -up)
    if bottom is None:
        return 0.0, {"anchor_px": 0.0, "anchor_collar": "empty"}

    # Насколько воротник ниже низа вклейки, вдоль оси лица шаблона
    gap = float(np.dot(collar - bottom, -up))
    limit = max_ratio * face_height
    offset = float(min(max(gap, 0.0), limit))

    meta = {
        "anchor_gap_px": round(gap, 1),
        "anchor_px": round(offset, 1),
        "anchor_collar": collar_meta["neck_source"],
    }
    if gap > limit:
        # Шея настолько короче, что дотянуть её до воротника значит уронить лицо
        # ниже, чем его нарисовал художник. Опускаем на сколько можно, остальное
        # достаётся зоне стыка — но в логе это должно быть видно.
        log.warning("шея не дотягивается до воротника даже со сдвигом", extra=meta)

    return offset, meta


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


def _feather_inwards(alpha: Any, face_height: float, feather_ratio: float) -> Any:
    """
    Мягкий край аппликации, целиком лежащий внутри вырезанного силуэта.

    Порядок ровно обратный `mask_generator.soften`: там область сначала
    расширяют, чтобы спад ушёл наружу и залитое осталось непрозрачным, — для
    зоны инпейнтинга это правильно. Здесь наружу уходить некуда: за контуром
    волос лежит фон фотографии, и любой полупрозрачный пиксель там — это ореол.
    Поэтому сначала эрозия, потом размытие: спад укладывается внутрь контура
    целиком, а плато 255 отступает от края на ширину растушёвки.

    Эрозия на пиксель шире ядра размытия — не запас «на всякий случай», а
    ровно то, что делает гарантию строгой: гауссиан с ядром 2f+1 тянется на f
    пикселей, и стартуй он с контура минус f, крайний пиксель контура получил бы
    ненулевую альфу.

    :param face_height: высота лица на шаблоне — база доли
    """
    import cv2

    if feather_ratio < 0:
        raise InvalidImageError(
            "Растушёвка края аппликации не может быть отрицательной",
            {"feather_ratio": feather_ratio},
        )

    feather = round(face_height * feather_ratio)
    if feather <= 0:
        return alpha

    trim = feather + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * trim + 1, 2 * trim + 1))
    return cv2.GaussianBlur(cv2.erode(alpha, kernel), (2 * feather + 1, 2 * feather + 1), 0)


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


def _pyramid_fill(image: Any, unknown: Any) -> Any:
    """
    Заливка дыры пирамидой (push-pull): известное усредняется вниз по уровням и
    поднимается обратно, заполняя пустое.

    Зачем не cv2.inpaint. Telea и Навье-Стокс тянут цвет от границы дыры внутрь
    вдоль изофот — на большой дыре это даёт штрихи от каждой неровности контура.
    Пирамида по построению не может дать ни штриха, ни кольца: на верхних
    уровнях дыра просто исчезает, и вниз возвращается гладкая интерполяция
    окрестности. Ровно то, что нужно на месте стёртой причёски — однородный
    фон, поверх которого второй шаг положит мазок. Плюс на 4K она в 13 раз
    быстрее Telea (1.5 с против 19.5 с): дыра там размером с голову.

    :param unknown: маска uint8 — пиксели, которые НЕ являются источником цвета
    :return: изображение, где заполнено всё unknown (вызывающий берёт нужное)
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    # До уровня, где от кадра остаются единицы пикселей: дыра размером с голову
    # должна на верхних уровнях полностью раствориться в окрестности
    levels = max(1, int(np.log2(max(1, min(height, width)))) - 3)

    visible = unknown == 0
    if not visible.any():
        return np.asarray(image).copy()  # фона в кадре нет — брать цвет неоткуда

    known = visible.astype(np.float32)
    weighted = image.astype(np.float32) * known[..., None]

    stack = [(weighted, known)]
    for _ in range(levels):
        weighted = cv2.pyrDown(weighted)
        known = cv2.pyrDown(known)
        stack.append((weighted, known))

    # Затравка — средний цвет всего видимого фона. Без неё на обложке, где
    # персонаж занимает почти весь разворот, известного не остаётся даже на
    # верхнем уровне пирамиды, и дыра заливается чёрным.
    result = image[visible].mean(axis=0).astype(np.float32).reshape(1, 1, 3)

    # Вверх: на каждом уровне известное берётся как есть, неизвестное — с
    # уровня выше. Деление на вес возвращает цвет из взвешенной суммы.
    for weighted, known in reversed(stack):
        safe = np.maximum(known, 1e-3)[..., None]
        level = np.where(known[..., None] > 1e-3, weighted / safe, 0.0)
        coarse = cv2.resize(
            result, (level.shape[1], level.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        result = np.where(known[..., None] > 1e-3, level, coarse)

    return np.clip(result, 0, 255).astype(np.uint8)


def _restore_background(
    image: Any, hole: Any, foreign: Any, face_height: float, method: str
) -> Any:
    """
    Затягивает дыру фоном обложки, не подмешивая в неё персонажа.

    Ключевое здесь — `foreign`. Любой локальный метод восстановления берёт цвет
    с границы дыры, а граница стёртой причёски — это сам персонаж: его волосы
    сверху, кожа шеи снизу. Отсюда и брались тёмное кольцо вокруг вклейки, и
    розовые пятна на месте ушей. Поэтому персонаж целиком объявляется
    неизвестным наравне с дырой: источником цвета остаётся только настоящий
    фон. Записывается результат при этом ТОЛЬКО в дыру — остальной персонаж
    (руки, одежда, всё ниже воротника) обязан остаться нетронутым.

    :param hole: что заполнить — стёртая голова персонажа
    :param foreign: что нельзя брать за образец — персонаж целиком
    :param method: pyramid, telea или ns
    """
    import cv2
    import numpy as np

    if method not in _ERASE_METHODS:
        raise InvalidImageError(
            "Неизвестный способ восстановления фона",
            {"method": method, "available": list(_ERASE_METHODS)},
        )

    unknown = np.maximum(np.asarray(hole), np.asarray(foreign))
    if method == "pyramid":
        filled = _pyramid_fill(image, unknown)
    else:
        radius = max(1, round(face_height * _INPAINT_RADIUS_RATIO))
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        filled = cv2.inpaint(image, unknown, radius, flag)

    return np.where(np.asarray(hole)[..., None] > 0, filled, image)


def _erase_template_head(
    target: Any,
    points: list[tuple[int, int]],
    pasted: Any,
    face_height: float,
    model: str,
    method: str = _ERASE_METHOD,
    neck_ratio: float = _ERASE_NECK_RATIO,
    pad_ratio: float = _ERASE_PAD_RATIO,
) -> tuple[Any, Any, dict]:
    """
    Стирает голову нарисованного персонажа и затягивает её место фоном.

    Без этого шага перенос причёски виден насквозь: у персонажа с длинными
    волосами вокруг вклеенной головы остаётся его собственная шевелюра, и на
    развороте оказывается два человека сразу. Понизить strength и попросить
    модель убрать её нельзя — на 0.2 она ничего не убирает, только подкрашивает.

    Стирается голова **целиком**, а не только торчащая из-под вклейки часть.
    Разница принципиальная: если оставить закрытую часть на месте, дыра
    получается кольцом, и её внутренняя граница — тёмные волосы персонажа.
    Любой локальный метод восстановления тянет цвет от границы внутрь, поэтому
    ровно эти волосы и размазывались вокруг вклейки тёмным ореолом. Середину
    всё равно закрывает вклеенная голова, так что стирать её ничего не стоит.

    Маски здесь свои, не донорские: срез опускается ниже челюсти (уши
    персонажа), эрозии нет вовсе, а порог «здесь есть персонаж» много ниже —
    сомнение трактуется в пользу стирания, а не в пользу фона.

    Отказ сегментатора на обложке не фатален: рисованный персонаж — не тот
    материал, на котором учили U²-Net, и вероятность промаха здесь выше, чем на
    фотографии. Поэтому шаг пропускается с предупреждением, а не роняет заказ.

    :param pasted: силуэт вклеенной головы донора — что уже закрыто
    :param method: способ восстановления фона (pyramid, telea, ns)
    :param neck_ratio: насколько опустить срез ниже челюсти персонажа
    :param pad_ratio: запас вокруг силуэта персонажа
    :return: (изображение с затянутой дырой, маска стирания, метаданные)
    """
    import cv2
    import numpy as np

    empty = np.zeros(target.shape[:2], dtype=np.uint8)
    try:
        # Срез прямой, а не по дуге челюсти: дуга поднимается к ушам и
        # оставляет их кончики красными лепестками по бокам вклейки, а если
        # опустить её настолько, чтобы их достать, вместе с ними стирается шея
        # персонажа под подбородком — та самая, на которую садится вклеенная
        # голова. Прямая на уровне подбородка забирает ухо целиком и шею не трогает.
        region, _, _ = segmentation.head_region(
            points,
            target.shape[:2],
            neck_ratio=neck_ratio,
            follow_jaw=False,
            # Стираем голову персонажа, а не его шею: она остаётся на
            # месте и служит фоном под шею донора
            neck_column=False,
        )
        silhouette = np.asarray(segmentation.silhouette(target, model))
    except MLServiceError as exc:
        log.warning(
            "причёску персонажа стереть не удалось — сегментатор не нашёл голову",
            extra={"cause": exc.message},
        )
        return target, empty, {"erased_ratio": None}

    pad = max(1, round(face_height * pad_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))

    # Персонаж целиком — он не источник цвета для заливки ни одним пикселем
    foreign = cv2.dilate(np.where(silhouette > _FOREIGN_FLOOR, 255, 0).astype(np.uint8), kernel)
    # Дыра — та его часть, что попала в область головы. Крупнейшая компонента:
    # в эллипс головы могла заехать поднятая рука или ветка за спиной
    head = segmentation.largest_component(np.where(region > 0, foreign, 0).astype(np.uint8))
    hole = cv2.dilate(head, kernel)

    if not hole.any():
        # Пустой силуэт — это промах сегментатора на рисованном персонаже, а не
        # «голова нулевой площади». Тот же случай, что и отказ выше: пропускаем
        # шаг с предупреждением, а не роняем заказ.
        log.warning("причёску персонажа стереть не удалось — силуэт пуст")
        return target, empty, {"erased_ratio": None}

    head_px = int(np.count_nonzero(hole))

    # Запас вокруг вклейки: то, что она закрывает, модели перерисовывать не надо
    grow = max(1, round(face_height * 0.02))
    covered = cv2.dilate(
        (pasted > 127).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)),
    )
    erased = np.where((hole > 0) & (covered == 0), 255, 0).astype(np.uint8)

    ratio = float(np.count_nonzero(erased)) / head_px
    meta = {
        "erased_ratio": round(ratio, 3),
        "template_head_px": head_px,
        "erase_method": method,
    }

    filled = _restore_background(target, hole, foreign, face_height, method)

    log.info("голова персонажа стёрта", extra=meta)
    if ratio > 0.5:
        # Половина головы персонажа мимо вклейки — фон за ней честно
        # восстановить нечем, и на 0.2 модель это не спасёт.
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
    neck_ratio: float | None = segmentation._NECK_RATIO,
    erode_ratio: float = segmentation._ERODE_RATIO,
    scale_mark: str = "umeyama",
    scale_multiplier: float = 1.0,
    anchor_neck: bool = True,
    feather_ratio: float = _FEATHER_RATIO,
    colour_match: float = _COLOUR_MATCH,
    erase_template_head: bool = True,
    erase_method: str = _ERASE_METHOD,
    erase_neck_ratio: float = _ERASE_NECK_RATIO,
    erase_pad_ratio: float = _ERASE_PAD_RATIO,
) -> Collage:
    """
    Вклеивает голову донора в шаблон и возвращает коллаж для инпейнтинга.

    :param source: BGR-фотография заказчика
    :param target: BGR-иллюстрация-шаблон
    :param emotion: имя трансформера мимики; пусто — нейтральное выражение
    :param model_photo: модель сегментации для фотографии
    :param model_cover: модель сегментации для обложки
    :param neck_ratio: докуда брать шею; None — искать линию одежды донора
    :param erode_ratio: подрезка края силуэта, доля высоты лица
    :param scale_mark: по какой биометрической мерке считать масштаб;
        umeyama — подгонка сразу по всем опорным точкам лица
    :param scale_multiplier: множитель поверх посчитанного масштаба. Ручка
        художественная, а не геометрическая: 1.0 — лицо донора точно совпадает
        с лицом персонажа
    :param anchor_neck: опускать ли вклейку до воротника, если шея не дотянулась
    :param colour_match: доля приведения тона кожи к шаблону, 0..1
    :param erase_template_head: стирать ли голову персонажа из-под вклейки
    :param erase_method: чем затягивать её место (pyramid, telea, ns)
    :param erase_neck_ratio: насколько опустить срез ниже челюсти персонажа
    :param erase_pad_ratio: запас вокруг силуэта персонажа
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
        source, source_points, model_photo, width_ratio, hair_ratio, neck_ratio, erode_ratio
    )

    # Мимика — до переноса: на вклеенной голове её правка поехала бы вместе с
    # фоном обложки. Нейтральное выражение проходит насквозь без затрат.
    face = expression.transform(
        expression.Face(image=source, alpha=head.alpha, points=source_points), emotion
    )

    # Масштаб — строго по лицу. Мерки считаются все, чтобы их расхождение было
    # видно в метаданных: на рисованном персонаже они спорят между собой до 18%,
    # и по одному числу потом не понять, почему голова вышла такой.
    if scale_multiplier <= 0:
        raise InvalidImageError(
            "Множитель масштаба должен быть положительным",
            {"scale_multiplier": scale_multiplier},
        )

    ratios = biometric_ratios(face.points, target_points)
    measured = biometric_scale(ratios, scale_mark)
    align_source = [face.points[i] for i in _ALIGN_POINTS]
    align_target = [target_points[i] for i in _ALIGN_POINTS]

    matrix, scale = similarity_transform(
        align_source,
        align_target,
        None if measured is None else measured * scale_multiplier,
    )
    if measured is None and scale_multiplier != 1.0:
        # У umeyama масштаб считается внутри, поэтому множитель применяется
        # вторым проходом — по уже известному числу
        matrix, scale = similarity_transform(
            align_source, align_target, scale * scale_multiplier
        )

    height, width = target.shape[:2]

    # Все доли маски и растушёвок меряются от лица НА ШАБЛОНЕ: именно его
    # размер определяет, сколько пикселей занимает стык на этой обложке.
    template_polygon = mask_generator.face_polygon(target_points)
    template_face_height = float(template_polygon[:, 1].max() - template_polygon[:, 1].min())

    # Якорь шеи — до переноса: сдвиг входит в ту же матрицу, иначе поедут и
    # контур лица, и отрезок стыка, а маски строятся уже по ним
    anchor_meta: dict = {"anchor_px": 0.0}
    if anchor_neck:
        offset, anchor_meta = neck_anchor(
            target, target_points, face.alpha, matrix, template_face_height
        )
        if offset > 0:
            _, up, _, _ = segmentation._axis(target_points)
            matrix = np.asarray(matrix, dtype=np.float64).copy()
            matrix[:, 2] -= up * offset

    warped = _warp(face.image, matrix, scale, (width, height))
    alpha = _warp(face.alpha, matrix, scale, (width, height))

    face_polygon = _transform_points(mask_generator.face_polygon(face.points), matrix)

    # Край аппликации: чуть размыть, чтобы контур волос не пилило антиалиасингом.
    # Эрозия перед размытием сдвигает весь спад внутрь силуэта — снаружи от
    # вырезанного контура не остаётся ни одного полупрозрачного пикселя, то есть
    # ни одного пикселя фона фотографии.
    alpha = _feather_inwards(alpha, template_face_height, feather_ratio)

    # Кожа = лицо внутри силуэта. Волосы сюда не попадают, и коррекция их не
    # трогает — в этом весь смысл переноса причёски.
    skin = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(skin, [face_polygon], 255)
    skin = cv2.GaussianBlur(np.minimum(skin, alpha), (0, 0), max(1.0, template_face_height * 0.03))
    warped = _match_skin(warped, target, skin, colour_match)

    base, erased, erase_meta = (
        _erase_template_head(
            target,
            target_points,
            alpha,
            template_face_height,
            model_cover,
            erase_method,
            erase_neck_ratio,
            erase_pad_ratio,
        )
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
        "erode_px": head.meta["erode_px"],
        # Докуда взята шея и по какому признаку. Первое, на что смотреть, если
        # голова на развороте оказалась висящей в воздухе или, наоборот, в
        # аппликацию приехал воротник
        "neck_ratio": head.meta["neck_ratio"],
        "neck_source": head.meta["neck_source"],
        # Масштаб и его разброс по меркам: если голова вышла велика или мала,
        # смотреть надо сюда, а не на габариты причёски
        "scale_mark": scale_mark,
        "scale_multiplier": scale_multiplier,
        "scale_marks": {name: round(value, 3) for name, value in ratios.items()},
        **anchor_meta,
        # Насколько силуэт заполнил отведённый эллипс головы. Близко к нулю —
        # сегментатор промахнулся, близко к единице — причёска упёрлась в
        # границу области, и часть волос могла остаться за кадром вклейки.
        "head_fill": head.meta["fill"],
        "head_px": int(np.count_nonzero(alpha > 127)),
        "face_height_target": round(template_face_height, 1),
        **erase_meta,
    }
    log.info("аппликация собрана", extra={**meta, "image_size": f"{width}x{height}"})

    # Та же геометрия, но числами прямо в тексте сообщения. Дублирование
    # намеренное: поля `extra` доезжают не до всякого формата и не до всякого
    # сборщика логов, а именно эти цифры спрашивают первыми, когда голова вышла
    # не того размера или повисла над воротником. Тег [geometry] — чтобы строка
    # грепалась одной командой: docker logs ml-service | grep geometry
    donor_face = head.meta["face_height"]
    template_head = erase_meta.get("template_head_px")
    log.info(
        "[geometry] scale=%.3f mark=%s x%.2f marks=%s | anchor=%.1f px (%s) | "
        "neck=%.3f (%s) | face: donor %.0f px -> template %.0f px (x%.3f) | "
        "head: pasted %d px, character %s px (%s) | erased=%s",
        scale,
        scale_mark,
        scale_multiplier,
        {name: round(value, 3) for name, value in sorted(ratios.items())},
        anchor_meta.get("anchor_px", 0.0),
        anchor_meta.get("anchor_collar", "-"),
        head.meta["neck_ratio"],
        head.meta["neck_source"],
        donor_face,
        template_face_height,
        template_face_height / max(donor_face, 1e-6),
        meta["head_px"],
        template_head if template_head is not None else "?",
        f"x{meta['head_px'] / template_head:.2f}" if template_head else "?",
        erase_meta.get("erased_ratio"),
    )

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
