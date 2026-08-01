"""
Оркестрация замены лица: маска головы на шаблоне → генерация на fal.ai.

Пайплайн стал из двух шагов, и оба простые:

  1. **Локальная геометрия** (`head_mask.py`, `reference.py`) — маска головы
     персонажа на ШАБЛОНЕ и поле вокруг фотографии заказчика. Нужна не всем:
     диффузионным стратегиям без неё нечего отдать модели и некуда вернуть
     ответ, а фейссвопу она только мешает — область он находит своим детектором,
     а кайма вокруг фотографии сбивает поиск лица донора. Профиль отвечает на
     это одним полем `needs_mask`, и весь шаг тогда пропускается целиком.
  2. **Перенос лица** (`refine/`, на fal.ai). Шаблон и фотография уезжают одним
     вызовом; чем именно и с какими ключами — решает стратегия из профиля.

Здесь не знают ни про fal, ни про схему запроса, ни про то, вернётся ли готовый
кадр или его ещё придётся вклеивать. Смена подхода — это одно значение
`strategy` в профиле, и ни строчки в этом файле.

Чего здесь больше нет и почему. Прежняя схема вырезала голову заказчика по
контуру челюсти, вклеивала её в шаблон преобразованием подобия и полудюжиной
проходов инпейнтинга пыталась спрятать шов: фон, шея, фактура, стык, общая
текстура, LAB-коррекция тона. На фотореалистичных шаблонах это провалилось
целиком — «летающая голова», плоский свет и, главное, полная неспособность
перенять мимику и поворот головы персонажа: аппликация переносит геометрию
донора один в один, а на развороте герой смеётся вполоборота.

Генеративный путь решает ровно это: поза и эмоция остаются от иллюстрации,
потому что модель видит их в кадре, а личность приходит второй картинкой.
Пикселей фотографии в результате нет ни одного — и сводить, соответственно,
нечего.

Контракт SwapRequest/SwapResult сохранён прежним: Node.js API получает те же
бинарный ответ и заголовок X-Swap-Meta, что и раньше.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.pipelines import expression, head_mask, refine
from app.pipelines import reference as reference_prep
from app.utils.image import decode_image, encode_image

log = get_logger(__name__)


def _sniff_mime(data: bytes) -> str:
    """MIME по сигнатуре файла: fal ждёт content-type при загрузке."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


@dataclass
class SwapRequest:
    source: bytes  # фотография заказчика — референс личности
    target: bytes  # разворот книги, куда переносим
    # Мимика. Пусто — выражение берётся с шаблона, и это умолчание: поза и
    # эмоция персонажа уже нарисованы. Список значений — expression.available().
    emotion: str = ""
    # Причёска заказчика словами: «короткий светлый ёжик». Читает это одна
    # стратегия — двухшаговая, — и для неё это единственный источник того, что
    # рисовать: фотография на шаг причёски не уезжает вовсе, иначе редактор
    # рисует по ней второе лицо. Пусто — описание берётся из ML_HAIR_DESCRIPTION,
    # а если пусто и там, заказ отклоняется до сети.
    hair: str = ""
    target_face_index: int | None = None
    swap_all_faces: bool = False
    enhance: bool = False
    style_strength: float = 1.0
    art_style: str = ""
    output_format: str = "png"


@dataclass
class SwapResult:
    image: bytes
    mime_type: str
    faces_detected: int
    faces_swapped: int
    meta: dict = field(default_factory=dict)


def _geometry(
    request: SwapRequest, profile: refine.profiles.RefineProfile
) -> tuple[head_mask.HeadMask | None, reference_prep.Reference]:
    """
    Шаг 1 целиком: маска головы персонажа и подготовленный референс.

    Оба вычисления существуют ради диффузии и вместе с ней и включаются.

    Маска отвечает на вопрос «где модели можно рисовать» — а фейссвоп рисует не
    по нашей указке и область находит сам. Поле вокруг фотографии лечит перенос
    композиции у kontext: с портрета крупным планом голова выходит больше маски,
    а квадратный кадр эндпоинт растягивает под разворот. Чужому детектору лица
    ни то, ни другое не нужно — ему нужна чистая фотография.

    :return: маска либо None, если профиль её не требует, и референс — он есть
        всегда, разница лишь в том, тронут ли он подготовкой
    """
    if not profile.needs_mask:
        log.info(
            "локальная геометрия пропущена: стратегии не нужны ни маска, ни поле референса",
            extra={"strategy": profile.strategy},
        )
        # pad_max=1.0 в профиле этой стратегии вернёт фотографию теми же
        # байтами; проверка размеров ей не понадобится, поэтому и цели нет
        return None, reference_prep.prepare(
            request.source,
            _sniff_mime(request.source),
            target_share=None,
            target_aspect=None,
            pad_ratio=profile.reference_pad_ratio,
            pad_max=profile.reference_pad_max,
        )

    # Отсутствие персонажа — 422 отсюда: ни детектор, ни разметка его не нашли
    target_image = decode_image(request.target)
    head = head_mask.build(
        target_image,
        dilate_ratio=profile.mask.dilate_ratio,
        feather_ratio=profile.mask.feather_ratio,
        neck_ratio=profile.mask.neck_ratio,
    )

    # Референс равняется на шаблон и по масштабу, и по пропорции. Считается
    # здесь, потому что здесь известна вторая половина обоих отношений — высота
    # лица ПЕРСОНАЖА и размер шаблона.
    target_height, target_width = target_image.shape[:2]
    return head, reference_prep.prepare(
        request.source,
        _sniff_mime(request.source),
        target_share=head.face_height / float(target_height),
        target_aspect=target_width / float(target_height),
        pad_ratio=profile.reference_pad_ratio,
        pad_max=profile.reference_pad_max,
    )


def run(request: SwapRequest) -> SwapResult:
    # Все гиперпараметры второго шага приходят одним набором — профилем. Здесь
    # он берётся один раз и передаётся дальше целиком: и маска, и запрос
    # обязаны собираться из одних и тех же чисел. Ширина растушёвки маски и
    # strength подбираются вместе, и разъехаться они не должны.
    profile = refine.profiles.from_settings()

    # Мимика проверяется до сети: неизвестная эмоция — 501, и узнать об этом
    # лучше здесь, чем после трёх загрузок в CDN
    emotion_prompt = expression.prompt(request.emotion)

    # Шаг 1. Локальная геометрия — маска на шаблоне и поле вокруг фотографии.
    # Нужна не всем: у фейссвопа своя область и свой детектор, и оба наших
    # вычисления ему только мешают. Три секунды на 4K и лишний повод отказать
    # заказу (`NoFaceDetectedError` от нашего детектора) — не та цена, которую
    # стоит платить за неиспользуемое число.
    head, reference = _geometry(request, profile)

    # Шаг 2. Шаблон уходит теми же байтами, что пришли: перекодировать его
    # незачем — мы в нём ничего не меняли, а 4K-разворот на пережатии теряет
    # ровно ту фактуру, которую модель просят повторить.
    mask_png, mask_mime = encode_image(head.mask, "png") if head else (b"", "image/png")
    result = refine.run(
        refine.RefineRequest(
            target=request.target,
            target_mime=_sniff_mime(request.target),
            mask=mask_png,
            mask_mime=mask_mime,
            identity=reference.data,
            identity_mime=reference.mime,
            expression=emotion_prompt,
            hair=request.hair,
            output_format=request.output_format,
        ),
        profile,
    )

    image, call_meta = result.image, dict(result.meta)
    mime_type = call_meta.pop("mime_type", "image/png")

    log.info(
        "замена лица выполнена",
        extra={
            "model": call_meta.get("model"),
            "strategy": profile.strategy,
            "bytes": len(image),
            **(head.meta if head else {}),
        },
    )

    # Перерисовывается ровно одна голова — та, что нашлась на шаблоне, —
    # поэтому счётчики всегда 1: поля сохранены ради неизменного формата
    # X-Swap-Meta.
    return SwapResult(
        image=image,
        mime_type=mime_type,
        faces_detected=1,
        faces_swapped=1,
        meta={
            **call_meta,
            "mask": head.meta if head else None,
            "reference": reference.meta,
            "emotion": request.emotion or expression.NEUTRAL,
        },
    )


def analyse(image_bytes: bytes) -> list[dict]:
    """Только детекция — используется Node.js API для предпросмотра."""
    image = decode_image(image_bytes)
    points = head_mask.face_landmarks(image)

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [
        {
            "bbox": {
                "x": min(xs),
                "y": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            },
            "landmarks": len(points),
        }
    ]
