"""
Классические операции согласования: цвет, микротекстура, зерно, резкость.

Все функции чистые: принимают BGR-кроп и маски, возвращают новый кроп.
Работают в float32, чтобы не терять точность на цепочке из четырёх стадий.
"""

from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-6


def _weighted_stats(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и СКО по каналам с весом маски."""
    weights = mask.astype(np.float32)
    total = float(weights.sum()) + _EPS
    weights_3d = weights[..., None]

    mean = (image * weights_3d).sum(axis=(0, 1)) / total
    variance = ((image - mean) ** 2 * weights_3d).sum(axis=(0, 1)) / total
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _blend(original: np.ndarray, modified: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = mask[..., None]
    return original * (1.0 - alpha) + modified * alpha


def match_color(
    crop: np.ndarray, skin: np.ndarray, ring: np.ndarray, strength: float
) -> np.ndarray:
    """
    Переносит статистики LAB из кольца иллюстрации на область лица.

    Именно это убирает «фотографический» цвет кожи: у обложки своя световая
    температура и своя насыщенность, и лицо после swap в них не попадает.
    Полный перенос (strength=1.0) стирает индивидуальный тон кожи, поэтому
    результат подмешивается частично.
    """
    if strength <= 0 or ring.sum() < _EPS:
        return crop

    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    face_mean, face_std = _weighted_stats(lab, skin)
    ring_mean, ring_std = _weighted_stats(lab, ring)

    scale = np.where(face_std > _EPS, ring_std / (face_std + _EPS), 1.0)
    # Дисперсия кольца определяется разнородностью сюжета (листва, небо,
    # животные), а не фактурой кожи, поэтому подтягивать к ней контраст лица
    # нельзя — иначе стадия сама создаёт рассогласование. Переносим прежде
    # всего среднее, разброс правим лишь слегка.
    scale = np.clip(scale, 0.85, 1.2)

    matched = (lab - face_mean) * scale + ring_mean
    matched = _blend(lab, matched, skin * strength)

    result = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR) * 255.0
    return np.clip(result, 0, 255)


def suppress_micro_texture(
    crop: np.ndarray, skin: np.ndarray, identity: np.ndarray, strength: float
) -> np.ndarray:
    """
    Убирает поры и фотографический микроконтраст, сохраняя крупные черты.

    Зона идентичности (глаза, нос, рот) сглаживается вдвое слабее: именно там
    живёт узнаваемость, и её потеря читается как «другой человек».
    """
    if strength <= 0:
        return crop

    smoothed = cv2.bilateralFilter(crop.astype(np.uint8), 9, 45, 9).astype(np.float32)
    weight = skin * strength * (1.0 - 0.5 * identity)
    return _blend(crop, smoothed, weight)


def harmonize_texture(
    crop: np.ndarray,
    skin: np.ndarray,
    ring: np.ndarray,
    grain: float,
    strength: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Заменяет высокочастотную составляющую лица на фактуру иллюстрации.

    Одна стадия вместо двух: раздельные «согласование резкости» и «добавление
    зерна» складывали энергии и в сумме уводили лицо дальше от окружения,
    чем было до обработки. Здесь высокие частоты сначала смешиваются
    (своя деталь <-> донорская фактура), а затем нормируются по энергии
    кольца — сколько бы ни было зерна, итог совпадает с окружением.

    Донор берётся сдвигом остатка на половину кадра: лицо в центре, поэтому
    на его место попадает мазок иллюстрации. Сдвиг детерминирован.
    """
    if strength <= 0 or ring.sum() < _EPS:
        return crop

    low = cv2.GaussianBlur(crop, (0, 0), sigma)
    own_hf = crop - low

    height, width = crop.shape[:2]
    donor_hf = np.roll(own_hf, shift=(height // 2, width // 2), axis=(0, 1))
    # Донор несёт не только фактуру холста, но и структурные края — пряди
    # волос, контуры предметов. Перенесённые на кожу, они читаются как
    # призрачные штрихи, поэтому срезаем выбросы, оставляя мелкое зерно.
    limit = float(np.percentile(np.abs(donor_hf), 85)) + _EPS
    donor_hf = np.clip(donor_hf, -limit, limit)

    mixed_hf = own_hf * (1.0 - grain) + donor_hf * grain

    # Нормировка энергии по кольцу
    ring_energy = float((np.abs(own_hf).mean(axis=2) * ring).sum() / (ring.sum() + _EPS))
    mixed_energy = float((np.abs(mixed_hf).mean(axis=2) * skin).sum() / (skin.sum() + _EPS))
    if mixed_energy < _EPS:
        return crop

    gain = np.clip(ring_energy / mixed_energy, 0.4, 2.5)
    harmonized = low + mixed_hf * gain

    return np.clip(_blend(crop, harmonized, skin * strength), 0, 255)


def paste_back(image: np.ndarray, crop: np.ndarray, mask: np.ndarray, box) -> np.ndarray:
    """Вклейка кропа обратно по растушёванной маске."""
    x0, y0, x1, y1 = box
    result = image.astype(np.float32).copy()
    region = result[y0:y1, x0:x1]
    result[y0:y1, x0:x1] = _blend(region, crop, mask)
    return np.clip(result, 0, 255).astype(np.uint8)
