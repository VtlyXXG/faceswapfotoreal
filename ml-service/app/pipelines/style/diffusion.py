"""
Диффузионный стилизатор: SD 1.5 + IP-Adapter FaceID.

Ставится на то же место в пайплайне, что и classical, через BaseStylizer.
Классика убирает шов, цвет и микротекстуру, но не рисует мазок кисти — здесь
лицо ре-синтезируется под стиль обложки, а идентичность держит IP-Adapter
FaceID по ArcFace-эмбеддингу донора.

Тяжёлые импорты (torch, diffusers) и веса грузятся лениво: как и swapper,
сервис поднимается и без них, а /health/ready сообщает degraded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.core.errors import ModelNotLoadedError
from app.core.logging import get_logger
from app.pipelines.style import color
from app.pipelines.style.base import BaseStylizer, StyleOptions
from app.pipelines.style.faceid import build_faceid_embeds
from app.pipelines.style.masking import build_masks
from app.pipelines.style.prompts import build_prompt

log = get_logger(__name__)


def _model_dir() -> Path:
    return (settings.models_dir / settings.style_diffusion_model).expanduser().resolve()


def _ip_adapter_dir() -> Path:
    return (settings.models_dir / settings.style_diffusion_ip_adapter).expanduser().resolve()


def weights_present() -> bool:
    """Наличие весов без их загрузки — для /health/ready."""
    base_ok = (_model_dir() / "model_index.json").exists()
    ip_ok = (_ip_adapter_dir() / settings.style_diffusion_ip_weight).exists()
    return base_ok and ip_ok


class DiffusionStylizer(BaseStylizer):
    name = "diffusion"

    def __init__(self, use_lora: bool | None = None) -> None:
        self._pipe = None
        self._torch = None
        self._use_lora = use_lora  # None → берём из конфигурации
        self.default_options = StyleOptions()

    # --- загрузка ----------------------------------------------------------

    def is_available(self) -> bool:
        return weights_present()

    def _resolve_use_lora(self) -> bool:
        if self._use_lora is not None:
            return self._use_lora
        return settings.style_diffusion_use_lora

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        if not weights_present():
            raise ModelNotLoadedError(
                "Веса SD 1.5 или IP-Adapter FaceID не найдены",
                {
                    "base": str(_model_dir()),
                    "ip_adapter": str(_ip_adapter_dir()),
                    "hint": "python scripts/fetch_style_models.py",
                },
            )

        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
        except ImportError as exc:  # noqa: BLE001
            raise ModelNotLoadedError(
                "diffusers/torch не установлены", {"cause": str(exc)}
            ) from exc

        device = settings.resolve_device()
        # На CPU только float32: float16 там не поддержан ядрами
        dtype = torch.float16 if device == "cuda" else torch.float32

        log.info("загрузка SD 1.5 (%s, %s)", device, dtype)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(_model_dir()), torch_dtype=dtype, safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )

        pipe.load_ip_adapter(
            str(_ip_adapter_dir()),
            subfolder=None,
            weight_name=settings.style_diffusion_ip_weight,
            image_encoder_folder=None,  # plain FaceID: эмбеддинг вместо CLIP-энкодера
        )

        lora = _ip_adapter_dir() / settings.style_diffusion_ip_lora
        if self._resolve_use_lora() and lora.exists():
            # LoRA-загрузка версионно капризна: при сбое продолжаем на одном
            # адаптере (чуть ниже сходство), а не роняем весь пайплайн
            try:
                pipe.load_lora_weights(str(_ip_adapter_dir()), weight_name=lora.name)
                pipe.fuse_lora()
                log.info("LoRA FaceID подключена")
            except Exception as exc:  # noqa: BLE001
                log.warning("LoRA FaceID не подхватилась, продолжаю без неё: %s", exc)
        elif not self._resolve_use_lora():
            log.info("LoRA FaceID отключена (--no-lora)")

        pipe.set_ip_adapter_scale(settings.style_diffusion_ip_scale)
        pipe = pipe.to(device)
        if device == "cpu":
            pipe.enable_attention_slicing()

        self._torch = torch
        self._pipe = pipe
        log.info("диффузионный стилизатор готов")
        return pipe

    # --- стилизация --------------------------------------------------------

    def stylize(
        self,
        image: Any,
        face: Any,
        options: StyleOptions,
        identity_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        pipe = self._load()
        torch = self._torch

        masks = build_masks(face, image.shape, margin=options.margin)
        x0, y0, x1, y1 = masks.box
        if x1 - x0 < 32 or y1 - y0 < 32:
            log.warning("лицо мелкое для диффузии, пропускаю")
            return image

        embedding = identity_embedding
        if embedding is None:
            embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raise ModelNotLoadedError("нет эмбеддинга для IP-Adapter FaceID")

        crop = image[y0:y1, x0:x1]
        work = self._img2img(pipe, torch, crop, embedding, options)

        # Идентичность держим: в зоне глаз/носа/рта берём меньше диффузии,
        # чтобы не сдвигать геометрию, которая и есть узнаваемость. Общая
        # сила запроса (intensity) дополнительно гасит подмешивание.
        keep = settings.style_diffusion_identity_keep
        gate = min(1.0, options.intensity)
        alpha = (masks.skin * (1.0 - keep * masks.identity) * gate)[..., None]
        composited = crop.astype(np.float32) * (1.0 - alpha) + work * alpha

        return color.paste_back(image, composited, masks.skin, masks.box)

    def _img2img(
        self, pipe, torch, crop: np.ndarray, embedding, options: StyleOptions
    ) -> np.ndarray:
        size = settings.style_diffusion_work_size
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

        from PIL import Image

        init = Image.fromarray(resized)
        positive, negative = build_prompt(options.art_style)
        device = settings.resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        embeds = build_faceid_embeds(embedding, torch, dtype, device)

        strength = min(0.9, settings.style_diffusion_texture_strength * options.intensity)
        generator = torch.Generator(device=device).manual_seed(settings.style_diffusion_seed)

        with torch.no_grad():
            output = pipe(
                prompt=positive,
                negative_prompt=negative,
                image=init,
                strength=strength,
                num_inference_steps=settings.style_diffusion_steps,
                guidance_scale=settings.style_diffusion_guidance,
                ip_adapter_image_embeds=embeds,
                generator=generator,
            ).images[0]

        result = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
        restored = cv2.resize(
            result, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        return restored.astype(np.float32)
