"""
Подготовка референса личности: масштаб лица и пропорция кадра под шаблон.

Референс уходит в Kontext третьей картинкой, и эндпоинт приводит его к своему
рабочему кадру сам. Обе беды этого слоя происходят от того, КАК он его приводит.

**Масштаб.** Заказчик присылает портрет — лицо занимает почти весь кадр.
Персонаж на развороте нарисован в полный рост, и его лицо занимает шестую часть.
Kontext переносит в генерацию не только черты референса, но и его композицию:
голова рисуется в том масштабе, в каком была на фотографии, — крупнее маски.
Нижняя челюсть в маску не помещается и срезается по её краю. Замер на нашей
паре: 0.500 высоты кадра у донора против 0.175 у шаблона dino1.

**Пропорция.** Фотография квадратная, разворот — 1.79:1. Подгоняя референс под
свой кадр, эндпоинт растягивает квадрат по горизонтали почти вдвое, и лицо
приезжает расплющенным. Ни промпт, ни вес идентичности с этим ничего не делают:
искажение случается до того, как модель увидит лицо.

Лечится и то и другое одним действием — полем вокруг фотографии. Высота холста
задаётся долей лица, ширина — пропорцией шаблона; сама фотография ложится в
центр нетронутой, режется только её доля площади. Совпали пропорции — подгонка
стала равномерным масштабированием, а оно не искажает ничего.

Поэтому коэффициента здесь нет. Ширина поля не назначается «на глаз в 20-30%»:
все три величины — высота лица на шаблоне, высота лица на фотографии, пропорция
шаблона — известны, и холст вычисляется из них. Число появляется только там, где
мерить нечего: если лица на фотографии не видно (профиль, тёмный кадр), кладётся
слепое поле `pad_ratio`.

Доля лица меряется по ВЫСОТЕ кадра, а не по меньшей стороне. Это следует из того
же вывода: когда пропорции совпали, эндпоинт масштабирует референс равномерно с
коэффициентом «высота выхода / высота референса», и высота лица переносится
ровно этим множителем. На альбомных шаблонах меньшая сторона и есть высота, но
на портретных — нет, и разница вылезла бы молча.

Чем платим. Поле по ширине не несёт ни одного пикселя лица, а рабочее
разрешение эндпоинта расходуется и на него: чем шире холст, тем меньше пикселей
достаётся самому лицу, а из них берётся личность. Управляется это `pad_max`, а
видно — по `face_px_at_work` в мете: если сходство поехало, смотреть надо туда.

Чем заполняется поле. Размытым продолжением краёв, приглушённым к нейтрально
серому, — а не сплошной заливкой. Ровный прямоугольник заливки даёт на границе
жёсткий контур, и он читается моделью как рамка, попадая в генерацию наравне с
лицом. Размытая кайма контура не создаёт вовсе, а личность из неё не берётся:
там нет ни черт, ни деталей.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

# Во сколько пикселей ужимается холст перед размытием каймы. Размывать поле в
# тысячи пикселей ядром той же величины — это минуты счёта; на превью в 128
# пикселей то же самое получается мгновенно, а разницы не видно: кайма и
# существует затем, чтобы в ней ничего нельзя было разглядеть.
_BLUR_PREVIEW_PX = 128

# Насколько кайма приглушается к нейтрально серому. Ноль — размытые края фото
# как есть, единица — ровная серая заливка.
_GREY_BLEND = 0.5
_GREY_LEVEL = 128.0

# Предполагаемая длинная сторона рабочего кадра эндпоинта. Точного числа fal не
# публикует, и берётся пессимистичное: по нему считается прогноз того, сколько
# пикселей достанется лицу. Прогноз нужен не расчёту, а человеку — это
# единственный способ заметить, что масштаб сошёлся ценой сходства.
_ENDPOINT_WORK_PX = 1024

# Ниже этого лицо в рабочем кадре считается мелким для переноса личности.
# Граница мягкая и служит порогом предупреждения, а не отказа.
_FACE_PX_FLOOR = 96

# Предел абсолютного размера холста по длинной стороне. Нужен от донора в 4K:
# поле считается в долях его сторон, и без этого предела холст ушёл бы в
# десятки тысяч пикселей — сотни мегабайт на декодирование и загрузку. Сжатие
# всего холста целиком не меняет ни долю лица, ни пропорцию.
_MAX_CANVAS_PX = 4096


@dataclass
class Reference:
    """
    Готовый референс — то, что уйдёт в CDN.

    :param data: байты изображения
    :param mime: тип содержимого
    :param meta: как считался холст. Уезжает в X-Swap-Meta: если голова опять
        вышла не того размера или лицо расплющено, смотреть надо сюда
    """

    data: bytes
    mime: str
    meta: dict = field(default_factory=dict)


def _pad(image: Any, top: int, bottom: int, left: int, right: int) -> Any:
    """
    Холст с размытой каймой: фотография в центре нетронутой.

    Кайма строится продолжением краёв, а не заливкой, и остаётся мягкой на всём
    протяжении — от края фотографии до границы холста нет ни одной резкой линии.
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    canvas = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # Размытие через уменьшенную копию: см. про стоимость ядра в шапке модуля
    scale = _BLUR_PREVIEW_PX / float(max(canvas.shape[:2]))
    small = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, _BLUR_PREVIEW_PX / 16.0))
    blurred = cv2.resize(
        small, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    surround = (
        blurred.astype(np.float32) * (1.0 - _GREY_BLEND) + _GREY_LEVEL * _GREY_BLEND
    ).astype(np.uint8)

    # Оригинал возвращается на место последним действием: размывать разрешено
    # что угодно, кроме самого лица
    surround[top : top + height, left : left + width] = image
    return surround


def _canvas(
    width: int,
    height: int,
    wanted_height: float,
    aspect: float,
    pad_max: float,
) -> tuple[int, int]:
    """
    Размер холста: высота из доли лица, ширина из пропорции шаблона.

    Обе стороны только растут — фотографию мы обрамляем, а не режем, поэтому
    любая цель, требующая уйти ниже исходного кадра, поднимается до него.

    :param wanted_height: высота, при которой доля лица сойдётся с шаблонной
    :param aspect: ширина/высота целевого шаблона
    :param pad_max: предел роста ДЛИННОЙ стороны относительно длинной стороны
        исходника. Именно она расходует рабочее разрешение эндпоинта
    """
    # Пол: ни высота меньше исходной, ни ширина (через пропорцию) меньше неё же
    floor_height = max(float(height), width / aspect)
    canvas_height = max(wanted_height, floor_height)

    allowed = pad_max * float(max(width, height))
    long_side = max(aspect * canvas_height, canvas_height)
    if long_side > allowed:
        # Упираемся в предел — жертвуем сходимостью по масштабу, но не
        # пропорцией: искажение видно всегда, а промах по размеру головы
        # частично добирает промпт
        canvas_height = max(canvas_height * allowed / long_side, floor_height)

    return int(round(aspect * canvas_height)), int(round(canvas_height))


def prepare(
    data: bytes,
    mime: str,
    target_share: float | None,
    target_aspect: float | None,
    pad_ratio: float,
    pad_max: float,
) -> Reference:
    """
    Приводит референс к масштабу и пропорции шаблона.

    :param data: фотография заказчика, как её прислали
    :param mime: её тип содержимого
    :param target_share: высота лица ПЕРСОНАЖА в долях высоты шаблона. None —
        шаблон своей высоты лица не сообщил, работает слепое поле
    :param target_aspect: ширина/высота шаблона. None — пропорция сохраняется
        как у фотографии
    :param pad_ratio: слепое поле сверху и снизу, доля высоты исходника.
        Применяется, только когда посчитать нечего
    :param pad_max: предел роста длинной стороны. 1.0 и меньше — обработка
        выключена целиком, референс уходит теми же байтами, что пришли
    """
    import cv2

    from app.pipelines import head_mask
    from app.utils.image import decode_image, encode_image

    meta: dict = {
        "target_share": round(target_share, 4) if target_share else None,
        "target_aspect": round(target_aspect, 3) if target_aspect else None,
    }

    if pad_max <= 1.0:
        log.info("подготовка референса выключена", extra={"pad_max": pad_max})
        return Reference(data=data, mime=mime, meta={**meta, "reason": "disabled"})

    image = decode_image(data)
    height, width = image.shape[:2]

    face = head_mask.face_size(image)
    own_share = face / float(height) if face else None
    aspect = target_aspect or (width / float(height))

    if own_share and target_share:
        # Основной путь: высота холста, при которой доля лица станет шаблонной
        wanted_height = face / target_share
        meta["reason"] = "measured"
    else:
        # Мерить нечего — кладём слепое поле. Оно заведомо не попадёт в масштаб
        # шаблона, но крупный портрет отдалит, а это и есть цель
        wanted_height = height * (1.0 + 2.0 * pad_ratio)
        meta["reason"] = "blind" if face is None else "no_target"
        log.info(
            "масштаб референса не измерен — поле кладётся вслепую",
            extra={"face_found": face is not None, "pad_ratio": pad_ratio},
        )

    canvas_width, canvas_height = _canvas(width, height, wanted_height, aspect, pad_max)

    meta["own_share"] = round(own_share, 4) if own_share else None
    meta["own_aspect"] = round(width / float(height), 3)
    meta["scale_wanted"] = round(max(aspect * wanted_height, wanted_height) / max(width, height), 3)

    if canvas_width <= width and canvas_height <= height:
        # Фотография уже и в масштабе, и в пропорции: поле не нужно, а лишнее
        # перекодирование стоило бы качества ни за что
        return Reference(data=data, mime=mime, meta={**meta, "scale": 1.0})

    scale = max(canvas_width, canvas_height) / float(max(width, height))
    if scale < meta["scale_wanted"] - 0.01:
        log.warning(
            "холст референса упёрся в предел — масштаб сойдётся не до конца",
            extra={"wanted": meta["scale_wanted"], "pad_max": pad_max},
        )

    left = (canvas_width - width) // 2
    right = canvas_width - width - left
    top = (canvas_height - height) // 2
    bottom = canvas_height - height - top
    padded = _pad(image, top, bottom, left, right)

    # Абсолютный предел размера — см. про донора в 4K в шапке модуля. Сжатие
    # холста целиком не трогает ни долю лица, ни пропорцию: обе — отношения
    if max(padded.shape[:2]) > _MAX_CANVAS_PX:
        fit = _MAX_CANVAS_PX / float(max(padded.shape[:2]))
        padded = cv2.resize(padded, None, fx=fit, fy=fit, interpolation=cv2.INTER_AREA)
        meta["fitted"] = True

    final_height, final_width = padded.shape[:2]

    # JPEG, а не PNG: референс — фотография, и на холсте вшестеро шире исходного
    # PNG весит десятки мегабайт. Из него берутся черты лица, а не фактура
    encoded, encoded_mime = encode_image(padded, "jpeg")

    face_px_at_work = (
        face / float(canvas_width if aspect >= 1 else canvas_height) * _ENDPOINT_WORK_PX
        if face
        else None
    )

    meta.update(
        {
            "scale": round(scale, 3),
            "size": [final_width, final_height],
            "aspect": round(final_width / float(final_height), 3),
            "face_share_after": round(face / float(canvas_height), 4) if face else None,
            "face_px_at_work": round(face_px_at_work) if face_px_at_work else None,
            "bytes": len(encoded),
        }
    )

    if face_px_at_work and face_px_at_work < _FACE_PX_FLOOR:
        # Масштаб и пропорция сошлись, но лицу досталось мало пикселей. Не
        # отказ: сходство пострадает не всегда, а знать об этом надо заранее
        log.warning(
            "лицу на референсе достаётся мало пикселей рабочего кадра — сходство под вопросом",
            extra={"face_px_at_work": round(face_px_at_work), "floor": _FACE_PX_FLOOR},
        )

    log.info("референс приведён к шаблону", extra=meta)
    return Reference(data=encoded, mime=encoded_mime, meta=meta)
