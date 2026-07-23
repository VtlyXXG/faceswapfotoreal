"""
Подготовка ArcFace-эмбеддингов для IP-Adapter FaceID.

Ключевой момент концепта: идентичность передаётся в диффузию не картинкой, а
тем самым 512-мерным вектором, который уже отдаёт детектор (w600k_r50). Никакой
CLIP-энкодер и второй проход не нужны — FaceID проецирует вектор напрямую.

Форма (2, 512) продиктована diffusers 0.30: при classifier-free guidance
pipeline делает chunk(2) по нулевой оси, ожидая [негатив, позитив].
"""

from __future__ import annotations

from typing import Any

import numpy as np

FACEID_DIM = 512


def build_faceid_embeds(embedding: np.ndarray, torch_module: Any, dtype: Any, device: str) -> list:
    """
    :param embedding: ArcFace normed_embedding, форма (512,)
    :param torch_module: модуль torch (внедряется, чтобы файл импортировался без него)
    :return: список из одного тензора формы (2, 512) на IP-Adapter
    """
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape[0] != FACEID_DIM:
        raise ValueError(f"ожидался эмбеддинг {FACEID_DIM}, получено {vector.shape[0]}")

    positive = torch_module.from_numpy(vector).unsqueeze(0)  # (1, 512)
    negative = torch_module.zeros_like(positive)  # (1, 512)
    per_adapter = torch_module.cat([negative, positive], dim=0).to(dtype=dtype, device=device)
    return [per_adapter]
