"""
Метрики качества постобработки.

Без них нельзя отличить «стало лучше» от «стало иначе»: обе величины
считаются до и после стилизации и уходят в ответ API и в лог.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_EPS = 1e-6

# Практическая граница узнаваемости по косинусу ArcFace-эмбеддингов.
# Ниже 0.45 — уже другой человек, 0.55 — безопасный порог.
IDENTITY_THRESHOLD = 0.55


def identity_similarity(source: Any, result: Any) -> float:
    """Косинус между эмбеддингами лица-донора и лица на результате."""
    a = np.asarray(source, dtype=np.float32).ravel()
    b = np.asarray(result, dtype=np.float32).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < _EPS:
        return 0.0
    return float(np.dot(a, b) / denominator)


def texture_distance(crop: np.ndarray, skin: np.ndarray, ring: np.ndarray) -> float:
    """
    Насколько фактура лица отличается от фактуры окружения.

    Складывается из двух рассогласований: энергии высоких частот (мазок против
    гладкой кожи) и цветовых моментов LAB. 0 — область лица статистически
    неотличима от иллюстрации вокруг неё.
    """
    if ring.sum() < _EPS or skin.sum() < _EPS:
        return 0.0

    crop = crop.astype(np.float32)

    # 1. Высокие частоты
    residual = crop - cv2.GaussianBlur(crop, (0, 0), 2.0)
    energy = np.abs(residual).mean(axis=2)
    face_energy = float((energy * skin).sum() / (skin.sum() + _EPS))
    ring_energy = float((energy * ring).sum() / (ring.sum() + _EPS))
    energy_ratio = abs(np.log((face_energy + _EPS) / (ring_energy + _EPS)))

    # 2. Цветовые моменты
    lab = cv2.cvtColor(crop / 255.0, cv2.COLOR_BGR2LAB)
    face_mean, face_std = _stats(lab, skin)
    ring_mean, ring_std = _stats(lab, ring)
    # L в LAB лежит в [0,100], a/b — в [-128,127]; нормируем, чтобы яркость
    # не перевешивала цветовые каналы
    norm = np.array([100.0, 128.0, 128.0], dtype=np.float32)
    color_distance = float(
        np.abs((face_mean - ring_mean) / norm).mean()
        + np.abs((face_std - ring_std) / norm).mean()
    )

    return round(energy_ratio + color_distance, 4)


def _stats(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = mask.astype(np.float32)
    total = float(weights.sum()) + _EPS
    mean = (image * weights[..., None]).sum(axis=(0, 1)) / total
    variance = ((image - mean) ** 2 * weights[..., None]).sum(axis=(0, 1)) / total
    return mean, np.sqrt(np.maximum(variance, 0.0))
