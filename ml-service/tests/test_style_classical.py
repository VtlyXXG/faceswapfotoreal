"""Baseline-постобработка: маски, согласование цвета, зерно, метрики."""

import cv2
import numpy as np
import pytest

from app.pipelines.style import color
from app.pipelines.style.base import StyleOptions
from app.pipelines.style.masking import build_masks, expand_bbox
from app.pipelines.style.metrics import identity_similarity, texture_distance


class FakeFace:
    """Лицо без insightface: bbox по центру кадра, без 106-точечной разметки."""

    def __init__(self, bbox, landmarks=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        if landmarks is not None:
            self.landmark_2d_106 = landmarks


@pytest.fixture
def scene():
    """Синяя «иллюстрация» 400x400 с красным «лицом» в центре."""
    image = np.full((400, 400, 3), (200, 120, 60), np.uint8)  # BGR: синеватый фон
    cv2.circle(image, (200, 200), 60, (60, 90, 210), -1)  # красноватое лицо
    return image


# --- маски ---------------------------------------------------------------


def test_expand_bbox_stays_inside_image():
    box = expand_bbox(np.array([10, 10, 50, 50]), (100, 100, 3), margin=4.0)
    x0, y0, x1, y1 = box

    assert 0 <= x0 < x1 <= 100
    assert 0 <= y0 < y1 <= 100


def test_masks_are_normalised_and_disjoint(scene):
    masks = build_masks(FakeFace([140, 140, 260, 260]), scene.shape, margin=1.8)

    for mask in (masks.skin, masks.identity, masks.ring):
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() and mask.max() <= 1.0

    # Кольцо — это окружение, оно не должно накрывать середину лица
    height, width = masks.skin.shape
    assert masks.skin[height // 2, width // 2] > 0.9
    assert masks.ring[height // 2, width // 2] < 0.01


# --- стадии --------------------------------------------------------------


def test_match_color_moves_face_towards_surroundings(scene):
    masks = build_masks(FakeFace([140, 140, 260, 260]), scene.shape, margin=1.8)
    x0, y0, x1, y1 = masks.box
    crop = scene[y0:y1, x0:x1].astype(np.float32)

    def distance(image):
        face = image[masks.skin > 0.9].mean(axis=0)
        ring = image[masks.ring > 0.5].mean(axis=0)
        return float(np.abs(face - ring).sum())

    matched = color.match_color(crop, masks.skin, masks.ring, strength=0.8)

    assert distance(matched) < distance(crop)


def test_match_color_disabled_is_identity(scene):
    masks = build_masks(FakeFace([140, 140, 260, 260]), scene.shape, margin=1.8)
    x0, y0, x1, y1 = masks.box
    crop = scene[y0:y1, x0:x1].astype(np.float32)

    assert np.array_equal(color.match_color(crop, masks.skin, masks.ring, 0.0), crop)


def _flat_face_scene():
    """Лицо — плоская заливка в центре, окружение — шум."""
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8).astype(np.float32)
    crop[70:130, 70:130] = 128.0

    skin = np.zeros((200, 200), np.float32)
    skin[70:130, 70:130] = 1.0
    ring = np.zeros((200, 200), np.float32)
    ring[:40, :] = 1.0

    return crop, skin, ring


def _hf_energy(image, mask):
    residual = image - cv2.GaussianBlur(image, (0, 0), 2.0)
    return float((np.abs(residual).mean(axis=2) * mask).sum() / mask.sum())


def test_harmonize_texture_brings_face_energy_to_ring():
    crop, skin, ring = _flat_face_scene()

    result = color.harmonize_texture(crop, skin, ring, grain=0.7, strength=1.0)

    # Плоское лицо получает фактуру окружения...
    assert _hf_energy(result, skin) > _hf_energy(crop, skin)
    # ...ровно столько, сколько её в кольце — это и есть согласование
    assert _hf_energy(result, skin) == pytest.approx(_hf_energy(crop, ring), rel=0.35)
    # За пределами маски изображение не тронуто
    assert np.allclose(result[:40, :40], crop[:40, :40])


def test_harmonize_texture_does_not_stack_energy():
    """Повторный прогон не должен наращивать резкость: стадия нормирующая."""
    crop, skin, ring = _flat_face_scene()

    once = color.harmonize_texture(crop, skin, ring, grain=0.7, strength=1.0)
    twice = color.harmonize_texture(once, skin, ring, grain=0.7, strength=1.0)

    assert _hf_energy(twice, skin) == pytest.approx(_hf_energy(once, skin), rel=0.25)


def test_paste_back_preserves_area_outside_mask(scene):
    masks = build_masks(FakeFace([140, 140, 260, 260]), scene.shape, margin=1.8)
    x0, y0, x1, y1 = masks.box
    crop = np.zeros((y1 - y0, x1 - x0, 3), np.float32)

    result = color.paste_back(scene, crop, masks.skin, masks.box)

    assert np.array_equal(result[:100, :100], scene[:100, :100])
    assert result[200, 200].sum() < scene[200, 200].sum()  # центр лица затемнён


# --- метрики -------------------------------------------------------------


def test_texture_distance_drops_after_harmonisation(scene):
    masks = build_masks(FakeFace([140, 140, 260, 260]), scene.shape, margin=1.8)
    x0, y0, x1, y1 = masks.box
    crop = scene[y0:y1, x0:x1].astype(np.float32)

    before = texture_distance(crop, masks.skin, masks.ring)
    matched = color.match_color(crop, masks.skin, masks.ring, strength=1.0)
    after = texture_distance(matched, masks.skin, masks.ring)

    assert after < before


def test_identity_similarity_bounds():
    vector = np.array([0.3, 0.4, 0.5, 0.7], dtype=np.float32)

    assert identity_similarity(vector, vector) == pytest.approx(1.0, abs=1e-5)
    assert identity_similarity(vector, -vector) == pytest.approx(-1.0, abs=1e-5)
    assert identity_similarity(vector, np.zeros(4, np.float32)) == 0.0


def test_style_options_scaling_is_clamped():
    options = StyleOptions(color=0.6, smooth=0.5, grain=0.7, sharpness=0.6)

    assert options.scaled(0.0).color == 0.0
    assert options.scaled(10.0).grain == 1.0  # верхняя граница
    assert options.scaled(1.0).margin == options.margin  # margin не масштабируется
