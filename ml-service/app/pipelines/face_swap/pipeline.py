"""
Оркестрация замены лица: коллаж локально → лёгкая стилизация на fal.ai.

Пайплайн из двух шагов, и порядок в нём принципиален:

  1. **Коллаж** (`collage.py`, локально, OpenCV + mediapipe). Лицо с фотографии
     переносится в шаблон преобразованием подобия и вклеивается оригинальными
     пикселями. Геометрия лица на этом шаге переносится ровно один в один.
  2. **Стилизация** (`fal_api.py`, на fal.ai). Тот же коллаж уходит в
     инпейнтинг по маске с очень низким strength — модель накладывает мазок
     кисти, согласует свет и растворяет шов склейки, но зашумление слишком
     слабое, чтобы она могла изменить черты лица и пропорции.

Прежняя схема — отдать fal чистую обложку и попросить нарисовать лицо по
референсу при strength 0.82 — портретного сходства не давала: при таком шуме
модель рисует лицо заново, а не переносит. Сходство и стилизация здесь
разведены по разным шагам именно поэтому.

Маска обязательна и накрывает объединение двух контуров: лица шаблона и
вклейки. Иначе шов коллажа оказался бы вне зоны инпейнтинга и остался виден.

Контракт SwapRequest/SwapResult сохранён прежним: Node.js API получает те же
бинарный ответ и заголовок X-Swap-Meta, что и раньше.
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
    # Шаг 1. Лицо переносится локально: отсутствие лица в любом из двух кадров —
    # 422 из детектора, с пометкой в деталях, какой именно кадр не подошёл.
    target_image = decode_image(request.target)
    source_image = decode_image(request.source)

    collage = collage_builder.build(
        source_image,
        target_image,
        grow_ratio=settings.collage_grow_ratio,
        feather_ratio=settings.collage_feather_ratio,
        colour_match=settings.collage_colour_match,
    )

    # Маска — по объединению контуров: и лицо шаблона, и вклейка целиком, чтобы
    # шов склейки заведомо оказался внутри зоны инпейнтинга.
    mask = mask_generator.mask_from_polygons(
        target_image.shape[:2],
        [collage.face_polygon, collage.paste_polygon],
        padding_ratio=settings.mask_padding_ratio,
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
