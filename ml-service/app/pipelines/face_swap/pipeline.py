"""
Оркестрация face-swap: источник → детекция → перенос → постобработка.

Единственная точка входа для API-слоя; роуты не знают про insightface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.pipelines.face_swap import detector, enhancer, swapper
from app.utils.image import decode_image, encode_image

log = get_logger(__name__)


@dataclass
class SwapRequest:
    source: bytes
    target: bytes
    target_face_index: int | None = None
    swap_all_faces: bool = False
    enhance: bool = False
    output_format: str = "png"


@dataclass
class SwapResult:
    image: bytes
    mime_type: str
    faces_detected: int
    faces_swapped: int
    meta: dict = field(default_factory=dict)


def run(request: SwapRequest) -> SwapResult:
    source_image = decode_image(request.source)
    target_image = decode_image(request.target)

    source_face = detector.largest_face(source_image)
    target_faces = detector.require_faces(target_image)

    if request.swap_all_faces:
        selected = target_faces
        result = swapper.swap_all(target_image, selected, source_face)
    else:
        selected = [detector.select_face(target_faces, request.target_face_index)]
        result = swapper.swap_face(target_image, selected[0], source_face)

    result = enhancer.enhance(result, enabled=request.enhance)
    payload, mime_type = encode_image(result, request.output_format)

    log.info(
        "face-swap выполнен: обнаружено %d, заменено %d",
        len(target_faces),
        len(selected),
    )

    return SwapResult(
        image=payload,
        mime_type=mime_type,
        faces_detected=len(target_faces),
        faces_swapped=len(selected),
        meta={"source_face": detector.describe(source_face)},
    )


def analyse(image_bytes: bytes) -> list[dict]:
    """Только детекция — используется Node.js API для предпросмотра."""
    faces: list[Any] = detector.detect_faces(decode_image(image_bytes))
    return [detector.describe(f) for f in faces]
