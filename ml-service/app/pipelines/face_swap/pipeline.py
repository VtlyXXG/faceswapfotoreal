"""
Оркестрация замены лица: фото-аппликация локально → сведение стыка на fal.ai.

Пайплайн из двух шагов, и порядок в нём принципиален:

  1. **Аппликация** (`collage.py`, локально: rembg + OpenCV + mediapipe).
     Голова заказчика — лицо вместе с причёской — вырезается по силуэту
     сегментатора и вклеивается в шаблон преобразованием подобия. Геометрия
     переносится один в один, цвет и структура волос сохраняются.
  2. **Сведение стыка** (`fal_api.py`, на fal.ai). Коллаж уходит в инпейнтинг
     по маске с экстремально низким strength. Маска накрывает только внешний
     контур волос, срез шеи и следы стирания чужой причёски — лицо из неё
     вычтено явно, модель до него физически не дотягивается.

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
from app.pipelines import fal_api, mask_generator
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

    # Маска — только стык: контур волос, срез шеи и следы стирания чужой
    # причёски. Лицо из неё вычитается внутри blend_mask.
    mask = mask_generator.blend_mask(
        target_image.shape[:2],
        collage.head_alpha,
        collage.face_polygon,
        collage.neck_line,
        collage.erased,
        collage.meta["face_height_target"],
        edge_ratio=settings.mask_edge_ratio,
        neck_ratio=settings.mask_neck_ratio,
        guard_ratio=settings.mask_guard_ratio,
        feather_ratio=settings.mask_feather_ratio,
    )
    mask_png, _ = encode_image(mask, "png")

    # PNG, а не JPEG: коллаж — это оригинальные пиксели фотографии, и терять их
    # на артефактах сжатия перед единственным шагом, который их сохраняет,
    # бессмысленно.
    collage_png, collage_mime = encode_image(collage.image, "png")

    # Шаг 2. Референсом остаётся исходное фото: при strength ~0.2 оно почти ни
    # на что не влияет, но подсказывает модели, чьё лицо она обводит мазком.
    image, call_meta = fal_api.refine_collage(
        collage=collage_png,
        collage_mime=collage_mime,
        reference=request.source,
        reference_mime=_sniff_mime(request.source),
        mask=mask_png,
        output_format=request.output_format,
    )

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
