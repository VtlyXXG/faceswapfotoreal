"""
Диффузионный стилизатор — части, проверяемые без 5 ГБ весов:
форма FaceID-эмбеддингов, промпт, деградация без весов, выбор в реестре.

Полный прогон SD 1.5 требует весов (scripts/fetch_style_models.py) и на
практике GPU, поэтому здесь не запускается.
"""

import numpy as np
import pytest
import torch

from app.core.errors import ModelNotLoadedError
from app.pipelines.style.base import StyleOptions
from app.pipelines.style.diffusion import DiffusionStylizer, weights_present
from app.pipelines.style.faceid import FACEID_DIM, build_faceid_embeds
from app.pipelines.style.prompts import build_prompt

# --- FaceID-эмбеддинги ---------------------------------------------------


def test_faceid_embeds_shape_matches_diffusers_contract():
    embedding = np.random.default_rng(0).standard_normal(FACEID_DIM).astype(np.float32)

    embeds = build_faceid_embeds(embedding, torch, torch.float32, "cpu")

    # Один тензор на IP-Adapter, форма (2, 512): diffusers делает chunk(2)
    # на [негатив, позитив] при classifier-free guidance
    assert len(embeds) == 1
    assert embeds[0].shape == (2, FACEID_DIM)


def test_faceid_embeds_negative_is_zero_positive_is_embedding():
    embedding = np.arange(FACEID_DIM, dtype=np.float32)

    tensor = build_faceid_embeds(embedding, torch, torch.float32, "cpu")[0]
    negative, positive = tensor.chunk(2)

    assert torch.count_nonzero(negative) == 0
    assert torch.allclose(positive.flatten(), torch.from_numpy(embedding))


def test_faceid_embeds_rejects_wrong_dim():
    with pytest.raises(ValueError, match="512"):
        build_faceid_embeds(np.zeros(128, np.float32), torch, torch.float32, "cpu")


# --- промпт --------------------------------------------------------------


def test_prompt_uses_art_style():
    positive, negative = build_prompt("bold comic ink")

    assert "bold comic ink" in positive
    assert "photograph" in negative  # борьба с фотографичностью


def test_prompt_falls_back_when_style_empty():
    positive, _ = build_prompt("")
    assert "watercolor" in positive  # стиль по умолчанию


# --- деградация без весов ------------------------------------------------


def test_stylizer_unavailable_without_weights(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    assert weights_present() is False
    assert DiffusionStylizer().is_available() is False


def test_stylize_raises_when_weights_missing(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path)

    class FakeFace:
        bbox = np.array([100, 100, 300, 300], np.float32)
        normed_embedding = np.zeros(FACEID_DIM, np.float32)

    image = np.zeros((512, 512, 3), np.uint8)
    with pytest.raises(ModelNotLoadedError, match="не найдены"):
        DiffusionStylizer().stylize(image, FakeFace(), StyleOptions())


# --- переключатель LoRA --------------------------------------------------


def test_use_lora_override_wins_over_config(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "style_diffusion_use_lora", True)
    assert DiffusionStylizer(use_lora=False)._resolve_use_lora() is False
    assert DiffusionStylizer(use_lora=True)._resolve_use_lora() is True


def test_use_lora_falls_back_to_config_when_unset(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "style_diffusion_use_lora", False)
    assert DiffusionStylizer()._resolve_use_lora() is False

    monkeypatch.setattr(settings, "style_diffusion_use_lora", True)
    assert DiffusionStylizer()._resolve_use_lora() is True


# --- реестр --------------------------------------------------------------


def test_registry_selects_diffusion(monkeypatch):
    from app.config import settings
    from app.pipelines import registry

    monkeypatch.setattr(settings, "style_provider", "diffusion")
    registry.reset()
    try:
        stylizer = registry.get_stylizer()
        assert stylizer.name == "diffusion"
    finally:
        registry.reset()


def test_readiness_reports_stylizer_availability(monkeypatch, tmp_path):
    from app.config import settings
    from app.pipelines import registry

    monkeypatch.setattr(settings, "style_provider", "diffusion")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    registry.reset()
    try:
        state = registry.status()
        assert state["stylizer"]["name"] == "diffusion"
        assert state["stylizer"]["available"] is False  # весов нет
    finally:
        registry.reset()


# --- StyleOptions --------------------------------------------------------


def test_scaled_preserves_art_style_and_records_intensity():
    options = StyleOptions(art_style="oil painting")

    scaled = options.scaled(0.5)

    assert scaled.art_style == "oil painting"
    assert scaled.intensity == 0.5
