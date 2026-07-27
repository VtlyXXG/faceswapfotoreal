"""
Оркестрация замены лица: фото-аппликация локально → сведение стыка на fal.ai.

Пайплайн из двух шагов, и порядок в нём принципиален:

  1. **Аппликация** (`collage.py`, локально: rembg + OpenCV + mediapipe).
     Голова заказчика — лицо вместе с причёской — вырезается по силуэту
     сегментатора и вклеивается в шаблон преобразованием подобия. Геометрия
     переносится один в один, цвет и структура волос сохраняются.
  2. **Стилизация** (`refine/`, на fal.ai). Коллаж уходит в инпейнтинг по
     градиентной маске. Маска накрывает внешний контур волос, срез шеи и следы
     стирания чужой причёски — лицо из неё вычтено явно, модель до него
     физически не дотягивается.

Оркестратор не знает, чем именно выполняется второй шаг. Он берёт профиль
(`refine.profiles.from_settings()`), собирает по нему маску и отдаёт запрос в
`refine.run` — а инпейнтинг там с ControlNet, проброс лицевых эмбеддингов или
что-то третье, решает поле `strategy` в профиле. Ровно поэтому маска строится
по тому же профилю: ширина её градиента и strength подбираются вместе, и
разъехаться они не должны.

Прежние схемы и почему они не подошли: при strength 0.82 модель рисовала лицо
заново по референсу и портретного сходства не давала; версия с переносом одного
лишь овала лица сохраняла сходство, но оставляла заказчику причёску
нарисованного персонажа.

Контракт SwapRequest/SwapResult сохранён прежним: Node.js API получает те же
бинарный ответ и заголовок X-Swap-Meta, что и раньше. Добавилось одно
необязательное поле `emotion` — параметр будущей трансформации мимики.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.core.logging import get_logger
from app.pipelines import collage as collage_builder
from app.pipelines import mask_generator, refine
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
    source: bytes
    target: bytes
    # Мимика вклеиваемого лица. Пусто — нейтральное выражение, то есть лицо как
    # снято. Список доступных значений — expression.available().
    emotion: str = ""
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


def run(request: SwapRequest) -> SwapResult:
    # Шаг 1. Голова переносится локально: отсутствие лица в любом из двух
    # кадров — 422 из детектора, с пометкой, какой именно кадр не подошёл.
    target_image = decode_image(request.target)
    source_image = decode_image(request.source)

    collage = collage_builder.build(
        source_image,
        target_image,
        emotion=request.emotion,
        model_photo=settings.seg_model_photo,
        model_cover=settings.seg_model_cover,
        width_ratio=settings.head_width_ratio,
        hair_ratio=settings.head_hair_ratio,
        neck_ratio=settings.head_neck_ratio,
        erode_ratio=settings.head_erode_ratio,
        feather_ratio=settings.collage_feather_ratio,
        colour_match=settings.collage_colour_match,
        erase_template_head=settings.collage_erase_template_head,
        erase_method=settings.collage_erase_method,
        erase_neck_ratio=settings.collage_erase_neck_ratio,
        erase_pad_ratio=settings.collage_erase_pad_ratio,
    )

    # Все гиперпараметры второго шага приходят одним набором — профилем. Здесь
    # он берётся один раз и передаётся дальше целиком: и маска, и стратегия
    # обязаны собираться из одних и тех же чисел, иначе градиент маски и
    # strength разъезжаются молча.
    profile = refine.profiles.from_settings()

    shape = target_image.shape[:2]
    face_height = collage.meta["face_height_target"]

    # Зона 2 — стыки: узкое кольцо по контуру новых волос и узкая полоса там,
    # где шея донора входит в тело персонажа. Лицо вычитается внутри.
    seam = mask_generator.seam_mask(
        shape,
        collage.head_alpha,
        collage.face_polygon,
        collage.neck_line,
        face_height,
        edge_ratio=profile.mask.edge_ratio,
        neck_ratio=profile.mask.neck_ratio,
        guard_ratio=profile.mask.guard_ratio,
        feather_ratio=profile.mask.feather_ratio,
        gradient_ratio=profile.mask.gradient_ratio,
    )
    masks = {"seam": encode_image(seam, "png")[0]}

    # Зона 3 — дыра в фоне на месте чужой причёски. Отдельным проходом и на
    # высоком strength: заливка оставляет там мыло, и сводить его с чем-либо
    # бессмысленно, фон нужно рисовать заново.
    if profile.background is not None:
        hole = mask_generator.hole_mask(
            shape,
            collage.head_alpha,
            collage.face_polygon,
            collage.erased,
            face_height,
            margin_ratio=profile.mask.hole_margin_ratio,
            feather_ratio=profile.mask.hole_feather_ratio,
            guard_ratio=profile.mask.guard_ratio,
        )
        # Мелкая дыра второго вызова не стоит: у героя со стрижкой её почти нет
        share = float((hole > 127).sum()) / float(shape[0] * shape[1])
        if share >= profile.background.min_area_ratio:
            masks["background"] = encode_image(hole, "png")[0]
        else:
            log.info("зона фона пропущена: дыра мала", extra={"hole_share": round(share, 5)})

    # PNG, а не JPEG: коллаж — это оригинальные пиксели фотографии, и терять их
    # на артефактах сжатия перед единственным шагом, который их сохраняет,
    # бессмысленно.
    collage_png, collage_mime = encode_image(collage.image, "png")

    # Шаг 2. Референсом остаётся исходное фото: оно подсказывает модели, чьё
    # лицо она обводит. Коллаж уезжает и массивом тоже — по нему стратегия
    # строит карты управления, а декодировать PNG второй раз незачем.
    result = refine.run(
        refine.RefineRequest(
            collage=collage_png,
            collage_mime=collage_mime,
            reference=request.source,
            reference_mime=_sniff_mime(request.source),
            masks=masks,
            collage_image=collage.image,
            output_format=request.output_format,
        ),
        profile,
    )

    image, call_meta = result.image, dict(result.meta)
    mime_type = call_meta.pop("mime_type", "image/png")

    log.info(
        "замена лица выполнена: коллаж + стилизация на fal",
        extra={"model": call_meta.get("model"), "bytes": len(image), **collage.meta},
    )

    # Перерисовывается ровно одно лицо — крупнейшее найденное, — поэтому
    # счётчики всегда 1: поля сохранены ради неизменного формата X-Swap-Meta.
    return SwapResult(
        image=image,
        mime_type=mime_type,
        faces_detected=1,
        faces_swapped=1,
        meta={**call_meta, "collage": collage.meta},
    )


def analyse(image_bytes: bytes) -> list[dict]:
    """Только детекция — используется Node.js API для предпросмотра."""
    image = decode_image(image_bytes)
    points = mask_generator.face_landmarks(image)

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
