"""Эндпоинты face-swap пайплайна."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from app.api.schemas import AnalyseResponse
from app.config import settings
from app.core.errors import PayloadTooLargeError
from app.pipelines.face_swap import pipeline

router = APIRouter(prefix="/face-swap", tags=["face-swap"])


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"Файл больше {settings.max_upload_mb} МБ",
            {"filename": upload.filename, "size": len(data)},
        )
    return data


@router.post("/analyse", response_model=AnalyseResponse, summary="Детекция лиц")
async def analyse(image: UploadFile = File(...)) -> AnalyseResponse:
    faces = pipeline.analyse(await _read(image))
    return AnalyseResponse(faces=faces, count=len(faces))


@router.post(
    "",
    summary="Перенос лица",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Готовое изображение"}},
)
async def swap(
    source: UploadFile = File(..., description="Фото-донор лица"),
    target: UploadFile = File(..., description="Иллюстрация, куда переносим"),
    target_face_index: int | None = Form(None),
    swap_all_faces: bool = Form(False),
    enhance: bool = Form(False, description="Согласовать лицо с иллюстрацией"),
    style_strength: float = Form(1.0, ge=0.0, le=2.0, description="Множитель силы постобработки"),
    art_style: str = Form("", description="Стиль обложки для промпта диффузии"),
    output_format: str = Form("png"),
    donor_gender: str | None = Form(
        None, description="Пол заказчика: male | female | non-binary. Только для faceswap"
    ),
) -> Response:
    """
    Результат возвращается бинарно; метаданные — в заголовке `X-Swap-Meta` (JSON),
    чтобы Node.js API мог сразу сохранить файл без base64-обвязки.
    """
    result = pipeline.run(
        pipeline.SwapRequest(
            source=await _read(source),
            target=await _read(target),
            target_face_index=target_face_index,
            swap_all_faces=swap_all_faces,
            enhance=enhance,
            style_strength=style_strength,
            art_style=art_style,
            output_format=output_format,
            donor_gender=donor_gender,
        )
    )

    meta = {
        "faces_detected": result.faces_detected,
        "faces_swapped": result.faces_swapped,
        **result.meta,
    }

    return Response(
        content=result.image,
        media_type=result.mime_type,
        headers={"X-Swap-Meta": json.dumps(meta, ensure_ascii=False)},
    )
