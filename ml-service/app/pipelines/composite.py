"""
Обратная вклейка: голова из генерации возвращается в шаблон по маске.

Зачем это нужно. Безмасочный эндпоинт перерисовывает кадр ЦЕЛИКОМ — в этом его
сила (модель видит позу, свет и лицо, которое заменяет) и в этом же его цена: в
ответе меняется не только голова. Диффузия проходит по всему изображению, и
после неё поедет фактура ткани, порядок пуговиц, надписи на плакате, рисунок
обоев. Для книги это неприемлемо: шаблон нарисован художником и обязан дойти до
печати без единого изменения.

Отсюда правило этого модуля: **из ответа берётся ровно голова, всё остальное
берётся из шаблона побитово**. Не «почти», а побитово — там, где маска равна
нулю, пиксели шаблона копируются как есть, без арифметики и округлений.

Граница между двумя источниками — растушёванный край той же маски, что строит
`head_mask.py`. Полутона там и решают: жёсткий край дал бы контур вокруг головы,
а мягкий переход через десятки пикселей глазом не читается.

Что модуль умеет сверх этого, и только по просьбе: привести тон генерации к
тону шаблона (`match`). Понадобилось это маске волос — она обходит голову
кольцом и выходит на фон, и общий увод экспозиции у редактора читается там как
светлое свечение вокруг головы. Поправка считается по пикселям ВНЕ маски, где
обе картинки изображают одно и то же, — см. `_match_levels`.

Чего этот модуль НЕ делает — и это его главное ограничение. Он не совмещает
голову с маской. Предполагается, что эндпоинт вернул тот же кадр в той же
композиции; если генерация уехала (модель перекадрировала сцену или сдвинула
персонажа), маска ляжет мимо новой головы и разрежет её. Молча это произойти не
должно, поэтому считается `drift` — насколько разошлись шаблон и генерация ВНЕ
маски. Там они обязаны совпадать почти полностью, и большое значение означает
ровно одно: результату верить нельзя, смотреть надо глазами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InvalidImageError
from app.core.logging import get_logger

log = get_logger(__name__)

# Насколько пропорция генерации может разойтись с шаблоном, прежде чем вместо
# растягивания включится вписывание с кропом. Доли процента — это округление
# сторон эндпоинтом до кратности восьми, и растянуть их безопаснее, чем резать.
_ASPECT_TOLERANCE = 0.01

# Во сколько раз генерацию приходится увеличивать, прежде чем это стоит строки в
# логе. Эндпоинт работает около мегапикселя, шаблон приходит в 4K: голова из
# ответа растягивается вчетверо и на печати будет мягче фона. Не отказ —
# ограничение эндпоинта, — но объяснять «размытое лицо» будет именно эта строка.
_UPSCALE_WARN = 1.5

# Средний модуль разницы ВНЕ маски, уровни 0..255. Порог мягкий: диффузия
# трогает весь кадр, и десяток уровней шума — норма. Заметное превышение
# означает, что генерация уехала от шаблона композицией.
_DRIFT_WARN = 12.0

# Разница считается на уменьшенной копии: нужна оценка, а не точность, а на 4K
# полный проход по трём каналам — сотни мегабайт трафика памяти впустую.
_DRIFT_PREVIEW_PX = 512

# Пределы поправки тона, см. `_match_levels`. Поправка обязана чинить общий увод
# экспозиции и не обязана уметь ничего сверх того: разойдись шаблон с генерацией
# сильнее этих границ — дело не в тоне, а в композиции, и лечится это не
# умножением. Вышедшая за пределы прямая поправку не ограничивает, а ОТМЕНЯЕТ:
# прижатое к границе число заведомо неверно, а причина видна по drift.
_MATCH_GAIN_MIN, _MATCH_GAIN_MAX = 0.8, 1.25
_MATCH_BIAS_MAX = 24.0

# Ниже этой дисперсии (уровни²) наклон не по чему считать: вне маски одно ровное
# небо. Тогда остаётся сдвиг — для плоской области он и есть вся поправка.
_MATCH_MIN_VARIANCE = 4.0


@dataclass
class Composited:
    """
    Готовый разворот и всё, что известно о вклейке.

    :param image: BGR numpy.ndarray размера шаблона
    :param meta: уезжает в X-Swap-Meta. Когда голова вышла срезанной или
        размытой, смотрят сюда первым делом: масштаб генерации и drift
    """

    image: Any
    meta: dict = field(default_factory=dict)


def _fit(generated: Any, shape: tuple[int, int]) -> tuple[Any, dict]:
    """
    Приводит генерацию к размеру шаблона.

    Эндпоинт отдаёт свой рабочий кадр (около мегапикселя), а вклеивать надо в
    исходный разворот — совпадение размеров скорее исключение.

    Пропорция при этом важнее кадрирования. Разошлась она заметно — генерация
    вписывается «по большей стороне» с центральным кропом: лишняя полоса по краю
    всё равно уйдёт под маской, а растянутое лицо испортит именно то, ради чего
    всё делалось.
    """
    import cv2

    height, width = shape
    source_height, source_width = generated.shape[:2]
    scale = max(width / float(source_width), height / float(source_height))
    meta = {
        "generated_size": [source_width, source_height],
        "template_size": [width, height],
        "scale": round(scale, 3),
    }

    if (source_height, source_width) == (height, width):
        return generated, {**meta, "fit": "none"}

    # INTER_AREA корректно усредняет при уменьшении, но при увеличении даёт
    # ступеньку; LANCZOS4 — наоборот. Выбор по направлению, а не «один на все»
    interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA

    target_aspect = width / float(height)
    drift = abs(source_width / float(source_height) - target_aspect) / target_aspect

    if drift <= _ASPECT_TOLERANCE:
        return cv2.resize(
            generated, (width, height), interpolation=interpolation
        ), {**meta, "fit": "stretch", "aspect_drift": round(drift, 4)}

    # Округление вниз оставило бы кадр на пиксель меньше кропа — берём не меньше
    covered = (
        max(width, int(round(source_width * scale))),
        max(height, int(round(source_height * scale))),
    )
    resized = cv2.resize(generated, covered, interpolation=interpolation)
    left = (resized.shape[1] - width) // 2
    top = (resized.shape[0] - height) // 2

    log.warning(
        "пропорция генерации разошлась с шаблоном — вписываем с кропом",
        extra={"aspect_drift": round(drift, 4), **meta},
    )
    return resized[top : top + height, left : left + width], {
        **meta,
        "fit": "cover",
        "aspect_drift": round(drift, 4),
    }


def difference(base: Any, other: Any, mask: Any = None) -> float:
    """
    Средний модуль разницы двух кадров, уровни 0..255. Ноль — они совпадают.

    Существует ради одного вопроса: **а изменилось ли вообще что-нибудь?**
    Фейссвоп, не найдя лица, не отвечает ошибкой — он возвращает присланный
    кадр как есть. Отличить такой ответ от удачного больше нечем: код 200,
    картинка на месте, счёт выставлен. Кадры при этом не побитово равны —
    ответ переехал через перекодирование, — поэтому сравниваются не байты, а
    содержимое, и на уменьшенной копии: нужна оценка, а не точность.

    :param mask: где считать. Пусто — по всему кадру, и это умолчание фейссвопа:
        он правит кадр сам, и где именно, мы не знаем. Правка причёски знает —
        она идёт по своей маске, и разницу вне маски мерить бессмысленно: там
        пиксели шаблона возвращены побитово, и среднее по кадру размажет ответ
        до неразличимости
    """
    import cv2

    if other.shape[:2] != base.shape[:2]:
        # Разный размер — значит, эндпоинт кадр точно тронул
        other = cv2.resize(other, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

    scale = _DRIFT_PREVIEW_PX / float(max(base.shape[:2]))
    if scale < 1.0:
        size = (max(1, int(base.shape[1] * scale)), max(1, int(base.shape[0] * scale)))
        base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        other = cv2.resize(other, size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    if mask is None:
        return float(cv2.absdiff(base, other).mean())

    inside = mask > 127
    if not inside.any():
        return 0.0

    return float(cv2.absdiff(base, other).mean(axis=2)[inside].mean())


def _drift(base: Any, fitted: Any, mask: Any) -> float | None:
    """
    Насколько генерация разошлась с шаблоном ВНЕ маски.

    Единственный доступный признак того, что модель перекадрировала сцену. Прямо
    проверить, попала ли новая голова под маску, нечем — для этого пришлось бы
    искать лицо на генерации ещё раз, — но сдвиг кадра виден по фону, и стоит
    эта проверка один проход по уменьшенной копии.
    """
    import cv2

    scale = _DRIFT_PREVIEW_PX / float(max(base.shape[:2]))
    if scale >= 1.0:
        small_base, small_fitted, small_mask = base, fitted, mask
    else:
        size = (max(1, int(base.shape[1] * scale)), max(1, int(base.shape[0] * scale)))
        small_base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        small_fitted = cv2.resize(fitted, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    outside = small_mask <= 8
    if not outside.any():
        return None

    difference = cv2.absdiff(small_base, small_fitted).mean(axis=2)
    return float(difference[outside].mean())


def _match_levels(base: Any, fitted: Any, mask: Any) -> tuple[Any, dict]:
    """
    Приводит тон генерации к тону шаблона по пикселям ВНЕ маски.

    Написано против светлого ореола вокруг головы. Редакторы общего назначения
    возвращают кадр не в том тоне, в каком получили: поднятые тени, добавленный
    контраст, чуть тёплый или чуть светлый результат — правка глобальная, и
    сама по себе она незаметна, потому что уводит весь кадр разом. Но мы берём
    из этого кадра не весь кадр, а область маски. А маска волос — это КОЛЬЦО
    вокруг головы: там, где лежала грива, она выходит на небо и горы. Кольцо
    посветлее фона по всему периметру и читается ровно как свечение — при том,
    что ни модель, ни вклейка никакого ореола не рисовали.

    Растушёвка от этого не спасает и спасать не может: она размывает КРАЙ
    кольца, а увод тона живёт внутри, где вес маски равен единице и пиксель
    копируется из генерации как есть. Мягкий край делает светлое пятно мягким —
    то есть похожим на свечение ещё больше.

    Чинится это тем, что у нас есть эталон. Вне маски шаблон и генерация
    изображают одно и то же, и там они обязаны совпадать; расхождение там — и
    есть увод, измеренный по десяткам тысяч пикселей той же сцены. По ним на
    канал считается прямая `gain * x + bias` (МНК), и ею правится генерация
    целиком — включая ту часть, что уедет под маску.

    Никакого «улучшения картинки» здесь не происходит: не увёл редактор тон —
    получится единица со сдвигом в ноль, и вклейка останется прежней побитово.

    :return: поправленная генерация и отчёт о поправке
    """
    import cv2
    import numpy as np

    scale = _DRIFT_PREVIEW_PX / float(max(base.shape[:2]))
    if scale >= 1.0:
        small_base, small_fitted, small_mask = base, fitted, mask
    else:
        size = (max(1, int(base.shape[1] * scale)), max(1, int(base.shape[0] * scale)))
        small_base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        small_fitted = cv2.resize(fitted, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    # Порог тот же, что у drift: считать поправку по краю растушёвки нельзя —
    # там пиксели наполовину из генерации, и они утянули бы её на себя
    outside = small_mask <= 8
    if int(np.count_nonzero(outside)) < 64:
        # Маска накрыла окно почти целиком — сравнивать не с чем. Молча
        # перемножить на единицу честнее, чем считать поправку по десятку точек
        log.info("тон генерации не поправлен: вне маски слишком мало пикселей")
        return fitted, {"matched": False}

    fit = []

    for channel in range(base.shape[2]):
        source = small_fitted[..., channel][outside].astype(np.float64)
        target = small_base[..., channel][outside].astype(np.float64)

        source_mean, target_mean = float(source.mean()), float(target.mean())
        variance = float(source.var())

        if variance < _MATCH_MIN_VARIANCE:
            gain = 1.0
        else:
            gain = float(((source - source_mean) * (target - target_mean)).mean() / variance)
        fit.append((gain, target_mean - gain * source_mean))

    # Поправка держится на одном допущении: ВНЕ маски обе картинки изображают
    # одну и ту же сцену. Вышедшая за пределы прямая означает, что допущение не
    # выполнено — редактор перекадрировал сцену, перекрасил её или вернул чужой
    # кадр, — и тогда правильного ответа у поправки нет. Прижать её к границе
    # значит применить заведомо неверное число: не увод тона мы этим исправим, а
    # добавим свой. Отказ честнее, а причина видна по drift
    if any(
        not _MATCH_GAIN_MIN <= gain <= _MATCH_GAIN_MAX or abs(bias) > _MATCH_BIAS_MAX
        for gain, bias in fit
    ):
        log.warning(
            "тон генерации не поправлен: вне маски она расходится с шаблоном "
            "сильнее, чем объясняется уводом тона",
            extra={"fit": [(round(gain, 3), round(bias, 1)) for gain, bias in fit]},
        )
        return fitted, {"matched": False}

    corrected = fitted.astype(np.float32)
    for channel, (gain, bias) in enumerate(fit):
        corrected[..., channel] = corrected[..., channel] * gain + bias

    return np.clip(corrected, 0, 255).astype(np.uint8), {
        "matched": True,
        "match_gain": [round(gain, 3) for gain, _ in fit],
        "match_bias": [round(bias, 1) for _, bias in fit],
        # Насколько генерация расходилась с шаблоном вне маски ДО поправки.
        # Именно это число и превращается в ореол, если поправку не делать
        "match_before": round(_drift(base, fitted, mask) or 0.0, 2),
    }


def paste(base: Any, generated: Any, mask: Any, match: bool = False) -> Composited:
    """
    Вклеивает голову из генерации в шаблон по растушёванной маске.

    :param base: шаблон-разворот, BGR numpy.ndarray. Источник всего, что не
        голова, — и источник побитово
    :param generated: ответ эндпоинта, BGR numpy.ndarray любого размера
    :param mask: маска головы, uint8 размера шаблона. 255 — брать из генерации,
        0 — из шаблона, полутона — переход
    :param match: привести тон генерации к тону шаблона перед вклейкой, см.
        `_match_levels`. Нужно там, где маска обходит голову КОЛЬЦОМ и выходит
        на фон: общий увод тона у редактора превращается там в светлый ореол
    """
    import numpy as np

    if base.ndim != 3 or generated.ndim != 3:
        raise InvalidImageError(
            "Вклейка ждёт цветные изображения",
            {"base": list(base.shape), "generated": list(generated.shape)},
        )

    if mask.ndim == 3:
        # Маска ездит по пайплайну как PNG, а декодер отдаёт три одинаковых
        # канала: берём любой
        mask = mask[..., 0]
    if mask.shape != base.shape[:2]:
        raise InvalidImageError(
            "Маска и шаблон разного размера — вклеивать не по чему",
            {"mask": list(mask.shape), "template": list(base.shape[:2])},
        )

    fitted, meta = _fit(generated, base.shape[:2])

    if match:
        fitted, match_meta = _match_levels(base, fitted, mask)
        meta.update(match_meta)

    alpha = mask.astype(np.float32) / 255.0
    blended = np.rint(
        base.astype(np.float32) * (1.0 - alpha[..., None])
        + fitted.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)

    # Явное восстановление шаблона там, где маска пуста. Арифметика с нулевым
    # весом даёт то же самое, но «даёт то же самое» — это про округление, а
    # обещание «фон нетронут» должно держаться не на нём
    untouched = alpha <= 0.0
    blended[untouched] = base[untouched]

    drift = _drift(base, fitted, mask)
    meta.update(
        {
            "mask_open_px": int(np.count_nonzero(mask > 127)),
            "changed_px": int(np.count_nonzero(np.any(blended != base, axis=2))),
            "untouched_px": int(np.count_nonzero(untouched)),
            "drift": round(drift, 2) if drift is not None else None,
        }
    )

    if meta["scale"] > _UPSCALE_WARN:
        log.warning(
            "генерация мельче шаблона — вклеенная голова будет мягче фона",
            extra={"scale": meta["scale"], **meta},
        )
    if drift is not None and drift > _DRIFT_WARN:
        log.warning(
            "генерация разошлась с шаблоном вне маски: композиция уехала, "
            "вклейка могла срезать голову",
            extra={"drift": round(drift, 2), "warn_at": _DRIFT_WARN},
        )

    log.info("голова вклеена в шаблон", extra=meta)
    return Composited(image=blended, meta=meta)
