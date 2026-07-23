"""
Прогон пар через реальный пайплайн face-swap + стилизатор.

Swap выполняется один раз на пару и переиспользуется всеми вариантами — это
самая дорогая операция. Стилизатор здесь всегда classical, чтобы стенд мерил
именно алгоритм, а не текущее значение ML_STYLE_PROVIDER.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.core.logging import get_logger
from app.pipelines.face_swap import detector, swapper
from app.pipelines.style.base import StyleOptions
from app.pipelines.style.classical import ClassicalStylizer
from app.pipelines.style.masking import build_masks
from app.pipelines.style.metrics import IDENTITY_THRESHOLD, identity_similarity, texture_distance
from app.utils.image import decode_image
from bench.imaging import to_data_uri
from bench.model import Cell, PairResult, Variant, summarize
from bench.pairs import Pair, discover_pairs

log = get_logger("bench")


def default_variants(strengths: list[float]) -> list[Variant]:
    variants = [Variant(name="plain", enhance=False)]
    for strength in strengths:
        variants.append(Variant(name=f"s{strength:g}", enhance=True, strength=strength))
    return variants


def _style_options() -> StyleOptions:
    return StyleOptions(
        color=settings.style_color,
        smooth=settings.style_smooth,
        grain=settings.style_grain,
        sharpness=settings.style_sharpness,
        margin=settings.style_margin,
    )


def _measure(
    image: np.ndarray, source_embedding: np.ndarray, mask_face: Any
) -> tuple[float | None, float | None, str]:
    """Узнаваемость (повторная детекция), рассогласование фактуры и кроп лица."""
    masks = build_masks(mask_face, image.shape, margin=settings.style_margin)
    x0, y0, x1, y1 = masks.box
    crop = image[y0:y1, x0:x1]

    texture = texture_distance(crop.astype(np.float32), masks.skin, masks.ring)

    identity: float | None = None
    note = ""
    faces = detector.detect_faces(image)
    if faces:
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        identity = round(identity_similarity(source_embedding, largest.normed_embedding), 4)
        if identity < IDENTITY_THRESHOLD:
            note = "ниже порога"
    else:
        note = "лицо не найдено"

    return identity, texture, to_data_uri(crop), note


def run_pair(pair: Pair, variants: list[Variant], stylizer: ClassicalStylizer) -> PairResult | None:
    source_image = decode_image(pair.source.read_bytes())
    target_image = decode_image(pair.target.read_bytes())

    try:
        source_face = detector.largest_face(source_image)
        target_faces = detector.require_faces(target_image)
    except Exception as exc:  # noqa: BLE001 — пропускаем пару, а не роняем стенд
        log.warning("пара %s пропущена: %s", pair.pair_id, exc)
        return None

    target_face = max(
        target_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    swapped = swapper.swap_face(target_image, target_face, source_face)
    source_embedding = source_face.normed_embedding

    # Кроп донора для колонки-эталона
    src_masks = build_masks(source_face, source_image.shape, margin=settings.style_margin)
    sx0, sy0, sx1, sy1 = src_masks.box
    source_uri = to_data_uri(source_image[sy0:sy1, sx0:sx1])
    result = PairResult(pair_id=pair.pair_id, source_uri=source_uri)

    options = _style_options()
    for variant in variants:
        if variant.enhance:
            image = stylizer.stylize(swapped, target_face, options.scaled(variant.strength))
        else:
            image = swapped
        identity, texture, crop_uri, note = _measure(image, source_embedding, target_face)
        result.cells.append(
            Cell(
                variant=variant.name,
                identity=identity,
                texture=texture,
                crop_uri=crop_uri,
                note=note,
            )
        )

    log.info("пара %s обработана", pair.pair_id)
    return result


def run_bench(input_dir: str, strengths: list[float]) -> dict:
    """Возвращает данные отчёта; запись файлов — на стороне CLI."""
    variants = default_variants(strengths)
    pairs = discover_pairs(input_dir)

    if not pairs:
        return {"results": [], "variants": variants, "summaries": [], "pairs_found": 0}

    log.info("найдено пар: %d, вариантов: %d", len(pairs), len(variants))
    stylizer = ClassicalStylizer()

    results = [r for p in pairs if (r := run_pair(p, variants, stylizer)) is not None]
    summaries = summarize(results, variants, IDENTITY_THRESHOLD)

    return {
        "results": results,
        "variants": variants,
        "summaries": summaries,
        "pairs_found": len(pairs),
    }
