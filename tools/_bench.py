"""
Общее для стендовых мерок: где лежит материал, как строится геометрия шаблона.

Пять скриптов рядом считали одно и то же по-разному, и расхождение мерки уже
один раз обошлось дорого: широкое окно вместо полосы у контура дало ровные 772
«пряди» и там и там, то есть померило тёмный фон и не разделило комплекты.
Поэтому геометрия и пороги живут здесь в одном экземпляре.

Корень проекта берётся от расположения файла, а не из константы: скрипты должны
работать у того, кто клонировал репозиторий к себе, а не только на машине, где
их писали. Переопределяется через ML_BENCH_ROOT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("ML_BENCH_ROOT") or Path(__file__).resolve().parents[1])
OUT = ROOT / "storage" / "output"
sys.path.insert(0, str(ROOT / "ml-service"))

# Стендовый набор: три шаблона на семь личностей — 21 кадр, на которых меряется
# всё. Пары «кадр — донор» менять нельзя, иначе числа не сравнить с прежними
TEMPLATES = {"dino1": "dino1.jpeg", "dino2": "dino2.jpg.jpeg", "spread_08": "spread_08.png"}
DONORS = {
    "id1": "donors/donor2.jpg", "id2": "donors/donor3.jpg", "id3": "donors/donor4.jpg",
    "id4": "donors/donor6.jpg", "id5": "donors/donor8.jpg", "id6": "donor.jpg.jpeg",
    "id7": "source.jpg",
}
STEMS = [f"{key}_{tag}" for key in TEMPLATES for tag in DONORS]

# Полоса вдоль контура стирания, в которой ищутся пряди прежнего героя. Шире
# брать нельзя: на dino2 и spread_08 в полосу попадает детальная листва, и она
# считается «тёмным по фону» наравне с волоском
STRAND_BAND_PX = 16

# Порог «пиксель изменён»: ниже него разница списывается на перекодирование
CHANGED_LEVEL = 8

# Ширина кольца вокруг области правки, доли высоты лица: с ним сравнивается
# резкость вклеенной головы. Четверть лица — это фон вплотную к голове, тот
# самый, рядом с которым мягкость и заметна глазу
_DETAIL_RING_RATIO = 0.25

_cache: dict[str, tuple] = {}


def template_context(key: str) -> tuple:
    """
    Шаблон и всё, что считается по нему один раз на семь кадров.

    Возвращает (кадр, разбор, силуэт стирания, зона поиска прядей).
    """
    if key in _cache:
        return _cache[key]

    from app.pipelines import head_mask, transplant

    T = cv2.imread(str(ROOT / TEMPLATES[key]))
    if T is None:
        raise SystemExit(f"нет шаблона {ROOT / TEMPLATES[key]}")
    points = transplant._full_frame_landmarks(T)
    face_height = head_mask.face_geometry(points)["face_height"]
    own, parsed = transplant._own_head(T, points)

    grow = max(1, int(round(face_height * transplant._ERASE_DILATE_RATIO)))
    feather = max(1, int(round(face_height * transplant._ERASE_FEATHER_RATIO)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    silhouette = cv2.GaussianBlur(
        cv2.dilate((own * 255).astype(np.uint8), kernel), (2 * feather + 1,) * 2, 0
    ) > 127

    background = ~(
        (np.asarray(parsed.clothes) > 127)
        | (np.asarray(parsed.skin) > 127)
        | (np.asarray(parsed.face) > 127)
    )
    # «Тёмное» — темнее медианы самой причёски: абсолютный порог не переносится
    # между шаблонами, у spread_08 сцена светлее dino1 целиком
    dark = T.max(axis=2) < np.median(T[silhouette].reshape(-1, 3).max(axis=1))
    band = cv2.distanceTransform((~silhouette).astype(np.uint8),
                                 cv2.DIST_L2, 5) <= STRAND_BAND_PX

    _cache[key] = (T, parsed, silhouette, dark & ~silhouette & background & band,
                   face_height, own)
    return _cache[key]


def changed_mask(frame, template):
    """
    Какие пиксели кадра отличаются от шаблона БОЛЬШЕ ЧЕМ НА ВОСЕМЬ УРОВНЕЙ.

    Не «дошло побитово», как здесь было написано раньше: допуск в `CHANGED_LEVEL`
    списывает на перекодирование и подогнанную по тону вклейку тоже. Для областей
    правки это верно, а для подсчёта прядей — нет, см. `strand_pixels`.
    """
    return np.abs(frame.astype(np.int16) - template.astype(np.int16)).max(axis=2) > CHANGED_LEVEL


def intact_mask(frame, template):
    """Какие пиксели дошли до печати ПОБИТОВО: расхождение с шаблоном ровно ноль."""
    return np.abs(frame.astype(np.int16) - template.astype(np.int16)).max(axis=2) == 0


def strand_pixels(frame, template, zone):
    """
    Пряди прежнего героя в полосе у контура: (с допуском, строго).

    ЗАЧЕМ ДВА ЧИСЛА. Первое считается через `changed_mask`, то есть с допуском в
    восемь уровней, и в этом его беда: подогнанная по тону вклейка проваливается
    под порог и записывается в «уцелевшие волосы». Замер прямо: на spread_08
    поправка тона подняла счёт с 4015 до 9694 px по всему набору — ровно там, где
    правка сработала лучше всего. Из 311 «прядей» на spread_08_id1 расхождение
    ровно ноль было у НУЛЯ пикселей, из 1396 после правки — у одного.

    Второе число требует побитового совпадения. Логика прямая: раз пиксель не
    отличается от шаблона ни на единицу, пайплайн его не писал, значит вклейка
    прошла мимо и волосок прежнего героя доехал до печати. По этому счёту тот же
    прогон даёт 566 против 623 px, то есть регресса нет.

    Первое число оставлено НЕ для симметрии: им померены все прежние прогоны, и
    заменить его значит потерять сравнимость со всей записанной историей. Читать
    следует строгое, тревожиться — когда расходятся оба.

    ЧЕСТНАЯ ОГОВОРКА. Строгий счёт тоже завышает, просто много меньше. Вклейка
    идёт по маске с мягкой кромкой, и там, где вес почти нулевой, округление
    возвращает ровно шаблонное значение — пиксель записан, а выглядит нетронутым.
    Так набрались 53 px на spread_08_id2 одним пятном 24x81. Отделить это можно
    только маской вклейки, а её в готовом кадре уже нет.
    """
    return (int((zone & ~changed_mask(frame, template)).sum()),
            int((zone & intact_mask(frame, template)).sum()))


def lightness(frame):
    """L* в шкале 0..100."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 2.55


def tone_step(frame, changed, skin, zone):
    """
    Перепад светлоты кожи по обе стороны границы вклейки, L*.

    ВНИМАНИЕ: мерит только СТЫК. Если потемнела вся голова целиком, ступень как
    раз уйдёт в ноль и отчитается об успехе — промах всей головы мимо тона
    персонажа ловится отдельно, скриптом skin_level.py.
    """
    ch = changed.astype(np.uint8)
    d = np.where(changed, cv2.distanceTransform(ch, cv2.DIST_L2, 5),
                 -cv2.distanceTransform(1 - ch, cv2.DIST_L2, 5))
    near = zone & skin & (d >= 1) & (d < 9)
    far = zone & skin & (d > -9) & (d <= -1)
    if near.sum() < 60 or far.sum() < 60:
        return None
    L = lightness(frame)
    return float(np.median(L[near]) - np.median(L[far]))


def detail_ratio(frame, template, changed, face_height):
    """
    Насколько вклеенная голова резче своего окружения — в долях от шаблонной.

    ЗАЧЕМ ЗНАМЕНАТЕЛЬ. Сравнивать резкость головы с резкостью головы шаблона в
    лоб нельзя: у двух детей разные причёски и разное содержание, и число
    померяет их, а не мягкость. Здесь обе картинки сперва делятся на СВОЁ
    окружение — кольцо снаружи области правки, — и только потом сравниваются.
    Кольцо вне правки, то есть у результата и шаблона оно побитово одно и то же,
    и знаменатель по построению общий.

    ЧТО ПОКАЗЫВАЕТ. У шаблона голова — самое резкое место кадра: замер даёт 2.15
    на dino1 и 1.58 на dino2 против размытого фона. После пересадки становится
    1.01 и 0.97, то есть голова перестаёт быть резче фона за ней. Единица здесь
    означала бы «перепад сохранён», 0.48 на dino1 — что от него осталась половина.

    ОГОВОРКА. Мерка не отличает равномерное размытие от отсутствия текстуры, а на
    замере это оказались разные вещи: волосы в генерации детализированы, а кожа
    лица пуста по высоким частотам. Число одно на обе беды, и лечатся они
    по-разному — глазами смотреть всё равно придётся.
    """
    if changed.sum() < 500:
        return None

    def sharpness(image, area):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)[area].var())

    grow = max(3, int(face_height * _DETAIL_RING_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    ring = (cv2.dilate(changed.astype(np.uint8), kernel) > 0) & ~changed
    if ring.sum() < 500:
        return None

    was = sharpness(template, changed) / max(sharpness(template, ring), 1e-6)
    now = sharpness(frame, changed) / max(sharpness(frame, ring), 1e-6)
    return now / max(was, 1e-6)


def lower_zone(changed):
    """Нижняя четверть области правки — там, где вклейка граничит с шеей."""
    ys = np.nonzero(changed)[0]
    zone = np.zeros(changed.shape, bool)
    zone[int(np.percentile(ys, 75)):] = True
    return zone


def frame_path(prefix: str, stem: str) -> Path:
    return OUT / f"{prefix}{stem}.png"


def parse_prefixes(argv, default):
    """Приставки комплектов из аргументов: FINAL_ S1234_ и так далее."""
    return [a if a.endswith("_") else a + "_" for a in argv[1:]] or list(default)
