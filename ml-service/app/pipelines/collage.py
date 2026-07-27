"""
Шаг 1 замены лица: жёсткий коллаж из оригинальных пикселей фотографии.

Зачем он вообще нужен. Раньше пайплайн отдавал fal обложку, маску и фото
заказчика как референс личности, а перерисовку вела модель при strength 0.82 —
то есть область под маской зашумлялась почти до нуля и рисовалась заново «по
мотивам» референса. Похожесть при этом держится только на том, насколько хорошо
модель прочитала личность с одного фото, и портретного сходства она не даёт:
геометрия лица уезжает. Заказчику нужно ровно обратное — 100% сохранение
геометрии при лёгкой стилизации.

Поэтому личность теперь переносится не моделью, а локально и буквально:
пиксели лица с фотографии вклеиваются в шаблон как есть. Модели остаётся второй
шаг — пройтись по этому коллажу инпейнтингом с очень низким strength (0.15–0.30,
см. config.py), когда физически невозможно изменить черты и пропорции, но
хватает на мазок кисти, свет и растворение шва.

Как переносится лицо:

  1. mediapipe даёт сетку из 468 точек на фотографии и на шаблоне;
  2. по опорным точкам считается преобразование подобия (поворот, единый
     масштаб, сдвиг) — оно совмещает лица, но не искажает донора;
  3. фотография переносится этим преобразованием в систему координат шаблона;
  4. в шаблон вклеивается только область внутри контура лица донора.

Ключевое — именно подобие, а не полный аффин и не гомография. У аффина шесть
степеней свободы: он подгонит донора под форму лица шаблона, растянув его по
одной оси и завалив сдвигом, и именно это убивает узнаваемость. Подобие имеет
четыре, оно физически не может изменить пропорции лица — только повернуть его и
поменять размер целиком.

rembg здесь не используется намеренно: вырезать фон не нужно, потому что контур
берётся по сетке лица и в него не попадают ни волосы, ни фон, ни одежда. Тот же
контур (челюсть + брови вместо лба), что и у маски — по причинам из
mask_generator.py: вклеенные волосы дают ореол и рассогласование с обложкой.
Лишняя зависимость с отдельной onnx-моделью ради этого не нужна.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.core.logging import get_logger
from app.pipelines import mask_generator

log = get_logger(__name__)

# Опорные точки совмещения — жёсткий каркас лица: углы глаз, спинка и крылья
# носа, углы рта, подбородок, скулы у ушей, внешние края бровей. Считать
# преобразование по всем 468 точкам сетки нет смысла: щёки и губы подвижны, а
# лишние точки только тянут посадку за мимикой донора.
_ALIGN_POINTS = (
    33, 133, 362, 263,  # внешние и внутренние углы глаз
    168, 6, 195, 4, 1,  # спинка носа сверху вниз до кончика
    98, 327,  # крылья носа
    61, 291,  # углы рта
    152,  # подбородок
    234, 454,  # края скул на уровне ушей
    70, 300,  # внешние края бровей
)

# Доли высоты лица (как и всё остальное в пайплайне — обложки приходят и в 4K,
# и превью-размером).
#
# grow — насколько контур вклейки расширяется за пределы лица донора. Нужен,
#   чтобы вклейка гарантированно накрыла черты лица шаблона: формы лиц разные, и
#   без запаса у края может остаться, например, бровь исходного персонажа —
#   а при strength 0.2 модель её уже не уберёт. Больше 3-4% брать нельзя:
#   начинает затягивать в кадр волосы и фон с фотографии.
# feather — растушёвка края вклейки. Коллаж намеренно жёсткий: это ровно та
#   ширина, которая убирает ступеньку антиалиасинга по контуру, и не больше.
#   Настоящее сведение шва — работа второго шага.
_GROW_RATIO = 0.02
_FEATHER_RATIO = 0.01

# Приведение цвета вклейки к цвету лица на шаблоне (среднее и разброс по
# каналам LAB). Геометрию не трогает вообще — это поканальная кривая, — но
# снимает разницу в тоне кожи и температуре света между фотостудией и
# иллюстрацией. Без него при strength 0.2 модель просто не успевает свести
# освещение, и лицо остаётся «фотографией на обложке».
# 1.0 — полностью цвет шаблона, 0.0 — цвет фотографии как есть.
_COLOUR_MATCH = 0.8

# Ниже этого масштаба уменьшать фотографию в один проход warpAffine нельзя:
# билинейная выборка пропускает пиксели и лицо рассыпается на алиасинг.
_PRESCALE_THRESHOLD = 0.99

# Доля лица шаблона, которую вклейка обязана накрыть. Меньше — значит лица
# несовместимы по ракурсу и часть черт персонажа осталась снаружи.
_MIN_COVERAGE = 0.9


@dataclass
class Collage:
    """Результат первого шага: коллаж и контуры для маски второго шага."""

    image: Any  # BGR numpy.ndarray — шаблон с вклеенным лицом
    paste_polygon: Any  # контур вклейки в координатах шаблона
    face_polygon: Any  # контур лица самого шаблона
    meta: dict = field(default_factory=dict)


def _landmarks(image: Any, role: str) -> list[tuple[int, int]]:
    """
    Сетка лица с пометкой, чей это кадр.

    До двухшагового пайплайна 422 всегда означала «нет лица на обложке», теперь
    же лицо ищется в обоих кадрах, и без детали причина отказа неотличима.
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
    получается ровно вида [s·R | t], поэтому переносимое лицо может только
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
    Переносит фотографию в систему координат шаблона.

    Лицо на фотографии обычно крупнее, чем на обложке, то есть перенос — это
    уменьшение. Уменьшать одним warpAffine нельзя: интерполяция берёт отдельные
    отсчёты и на коэффициенте вроде 0.3 просто выбрасывает две трети пикселей,
    оставляя рваные контуры. Поэтому сначала честное усреднение INTER_AREA до
    нужного размера, а уже потом поворот и сдвиг — тогда в warpAffine остаётся
    масштаб ~1 и алиасингу взяться неоткуда.
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

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _transform_points(points: Any, matrix: Any) -> Any:
    """Тот же перенос для контура: точки, а не пиксели."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    return np.rint(pts @ matrix[:, :2].T + matrix[:, 2]).astype(np.int32)


def _match_colour(donor: Any, template: Any, region: Any, ratio: float) -> Any:
    """
    Подгоняет тон вклейки под лицо шаблона: среднее и разброс по каналам LAB.

    LAB, а не BGR: там яркость отделена от цвета, поэтому подгонка тона кожи не
    задевает светотеневой рисунок лица — а он и есть геометрия, которую нельзя
    трогать. Статистики считаются по одной и той же области в обоих кадрах:
    донорское лицо после переноса и то, что было на его месте на обложке.
    """
    import cv2
    import numpy as np

    selection = region > 0
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

        matched[..., channel] = (
            src[..., channel] - float(src_values.mean())
        ) * gain + float(dst_values.mean())

    blended = src * (1.0 - ratio) + matched * ratio
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def build(
    source: Any,
    target: Any,
    grow_ratio: float = _GROW_RATIO,
    feather_ratio: float = _FEATHER_RATIO,
    colour_match: float = _COLOUR_MATCH,
) -> Collage:
    """
    Вклеивает лицо с фотографии в шаблон и возвращает коллаж для инпейнтинга.

    :param source: BGR-фотография заказчика
    :param target: BGR-иллюстрация-шаблон
    :param grow_ratio: запас контура вклейки, доля высоты лица
    :param feather_ratio: растушёвка края вклейки, доля высоты лица
    :param colour_match: доля приведения тона к шаблону, 0..1
    :return: Collage с готовым изображением и обоими контурами
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

    matrix, scale = similarity_transform(
        [source_points[i] for i in _ALIGN_POINTS],
        [target_points[i] for i in _ALIGN_POINTS],
    )

    height, width = target.shape[:2]
    warped = _warp(source, matrix, scale, (width, height))

    template_polygon = mask_generator.face_polygon(target_points)
    paste_polygon = _transform_points(mask_generator.face_polygon(source_points), matrix)

    face_height = int(template_polygon[:, 1].max() - template_polygon[:, 1].min())
    paste = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(paste, [paste_polygon], 255)
    paste = mask_generator.soften(paste, face_height, grow_ratio, feather_ratio)

    warped = _match_colour(warped, target, paste, colour_match)

    # Собственно коллаж: вклейка идёт поверх оригинала, за её пределами шаблон
    # не меняется ни на бит.
    alpha = (paste.astype(np.float32) / 255.0)[..., None]
    image = (warped.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )

    template_area = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(template_area, [template_polygon], 255)
    covered = int(np.count_nonzero((paste > 127) & (template_area > 0)))
    coverage = covered / max(1, int(np.count_nonzero(template_area)))

    meta = {
        "scale": round(scale, 3),
        "rotation_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 2),
        "coverage": round(coverage, 3),
        "colour_match": colour_match,
    }
    log.info("коллаж собран", extra={**meta, "image_size": f"{width}x{height}"})

    if coverage < _MIN_COVERAGE:
        # Не отказ: результат может быть приемлемым, но причина будущей
        # претензии должна быть видна в логах, а не выясняться по картинке.
        log.warning(
            "вклейка накрыла лицо шаблона не полностью — вероятно, разные ракурсы",
            extra={"coverage": meta["coverage"]},
        )
    if scale > 1.0:
        log.warning(
            "лицо на фотографии мельче, чем на обложке — вклейка растянута",
            extra={"scale": meta["scale"]},
        )

    return Collage(
        image=image,
        paste_polygon=paste_polygon,
        face_polygon=template_polygon,
        meta=meta,
    )
