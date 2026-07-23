"""Промпты для диффузионной стилизации лица.

negative важнее positive: борьба идёт именно с фотографичностью перенесённого
лица, а не за добавление стиля с нуля.
"""

from __future__ import annotations

_BASE_POSITIVE = (
    "portrait of a child, {art_style}, painterly skin, visible brush strokes, "
    "impasto texture, warm children's book illustration, soft lighting, coherent face"
)

_BASE_NEGATIVE = (
    "photograph, photorealistic, dslr, skin pores, plastic skin, smooth airbrush, "
    "cgi, 3d render, waxy, blurry, deformed, extra eyes, distorted face, text, watermark"
)

_DEFAULT_STYLE = "soft watercolor children's book illustration"


def build_prompt(art_style: str | None) -> tuple[str, str]:
    style = (art_style or "").strip() or _DEFAULT_STYLE
    return _BASE_POSITIVE.format(art_style=style), _BASE_NEGATIVE
