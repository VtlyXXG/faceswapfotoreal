"""
Оркестрация замены лица: маска по target → вызов fal.ai → результат.

Локального инференса больше нет. Здесь остаётся только последовательность
шагов; детали вызова модели живут в app/pipelines/fal_api.py, построение
маски — в app/pipelines/mask_generator.py.

Маска строится не всегда: бэкенду faceswap она не нужна, он ищет лицо сам.
Признак приходит из настроек, чтобы шаг не выполнялся впустую.

Контракт SwapRequest/SwapResult сохранён прежним: Node.js API получает те же
бинарный ответ и заголовок X-Swap-Meta, что и раньше.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.core.logging import get_logger
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
    # male | female | non-binary; используется только бэкендом faceswap
    donor_gender: str | None = None


@dataclass
class SwapResult:
    image: bytes
    mime_type: str
    faces_detected: int
    faces_swapped: int
    meta: dict = field(default_factory=dict)


def run(request: SwapRequest) -> SwapResult:
    mask_png = None
    if settings.mask_required:
        # Маска строится по target: перерисовывается лицо на иллюстрации, а не
        # на фотографии заказчика. Отсутствие лица здесь — 422 из детектора.
        target_image = decode_image(request.target)
        mask = mask_generator.generate_mask(target_image, blur_kernel=settings.mask_blur_kernel)
        mask_png, _ = encode_image(mask, "png")

    image, call_meta = fal_api.swap_face(
        target=request.target,
        target_mime=_sniff_mime(request.target),
        source=request.source,
        source_mime=_sniff_mime(request.source),
        mask=mask_png,
        donor_gender=request.donor_gender,
        output_format=request.output_format,
    )

    mime_type = call_meta.pop("mime_type", "image/png")

    log.info(
        "замена лица выполнена через fal",
        extra={
            "backend": call_meta.get("backend"),
            "model": call_meta.get("model"),
            "bytes": len(image),
        },
    )

    # Оба бэкенда обрабатывают ровно одно лицо, поэтому счётчики всегда 1:
    # поля сохранены ради неизменного формата X-Swap-Meta.
    return SwapResult(
        image=image,
        mime_type=mime_type,
        faces_detected=1,
        faces_swapped=1,
        meta=call_meta,
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
