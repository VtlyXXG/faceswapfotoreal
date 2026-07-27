"""
Стратегия «инпейнтинг по маске + ControlNet»: та, которой пайплайн работает.

Личность сюда приезжает готовыми пикселями — вклеенной головой из collage.py, —
и задача эндпоинта не нарисовать лицо, а положить поверх слой живописи: мазок
кисти, зерно холста, тени по объёму. Отсюда три входа вместо одного:

  image_url  — коллаж, основа кадра;
  mask_url   — где работать. Градиентная, а не плато: на strength 0.5 граница
               плато превращается в видимую ступеньку;
  controls   — карты управления. Canny держит рисунок и не даёт поехать чертам,
               depth ставит тени туда, где объём.

Референс-фото уезжает отдельной ссылкой: на низком strength оно почти ни на что
не влияет, но подсказывает модели, чьё лицо она обводит.

Схема аргументов зафиксирована тестами намеренно. Лишний ключ fal не
игнорирует, а заворачивает весь запрос, и узнаётся это только после боевого
прогона — один раз уже узнавали.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.pipelines import fal_api
from app.pipelines.refine import controls as control_maps
from app.pipelines.refine.base import RefineRequest, RefineResult, register
from app.pipelines.refine.profiles import RefineProfile

log = get_logger(__name__)


class InpaintControlNetRefiner:
    """Инпейнтинг готового коллажа по маске стыка, с картами управления."""

    name = "inpaint_controlnet"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        if request.mask is None:
            raise fal_api.MaskMissingError("Инпейнтингу нужна маска, но она не построена")

        if profile.strength > profile.safe_strength:
            # Не отказ: значение переопределяется из окружения именно для
            # подбора. Но выше этой границы шум съедает вклеенные пиксели, и
            # смысл двухшагового пайплайна пропадает — в логе это должно быть видно.
            log.warning(
                "strength выше безопасного для коллажа — черты лица могут поехать",
                extra={"strength": profile.strength, "safe_max": profile.safe_strength},
            )

        client = fal_api.client()
        fmt = "jpeg" if request.output_format in ("jpg", "jpeg") else "png"

        image_url = fal_api.upload(client, request.collage, request.collage_mime)
        identity_url = fal_api.upload(client, request.reference, request.reference_mime)
        mask_url = fal_api.upload(client, request.mask, "image/png")

        arguments = {
            "image_url": image_url,
            "mask_url": mask_url,
            "reference_image_url": identity_url,
            "prompt": profile.prompt,
            "strength": profile.strength,
            "guidance_scale": profile.guidance_scale,
            "num_inference_steps": profile.steps,
            "output_format": fmt,
        }

        maps = self._controls(client, request, profile)
        if maps:
            # Ключ добавляется только когда карты есть: у эндпоинта без
            # ControlNet пустой список — такой же лишний ключ, как и полный
            arguments[profile.control_field] = maps

        meta = {**profile.report(), "output_format": fmt, "controls_sent": len(maps)}
        return RefineResult(*fal_api.invoke(client, profile.endpoint, arguments, meta))

    def _controls(self, client, request: RefineRequest, profile: RefineProfile) -> list[dict]:
        """
        Строит и загружает карты управления.

        Карта считается по коллажу, а не по обложке: держать надо ту геометрию,
        которая уже собрана, включая вклеенную голову. Без картинки коллажа
        (её передаёт pipeline.py) карты просто не строятся — второй раз
        декодировать PNG ради этого незачем, а падать тем более не за чем.
        """
        if not profile.controls:
            return []
        if request.collage_image is None:
            log.warning(
                "карты управления пропущены: коллаж не передан массивом",
                extra={"controls": [c.kind for c in profile.controls]},
            )
            return []

        maps = []
        for spec in profile.controls:
            data, mime = control_maps.build(spec, request.collage_image)
            maps.append(
                {
                    "control_type": spec.kind,
                    "control_image_url": fal_api.upload(client, data, mime),
                    "conditioning_scale": spec.weight,
                    "start_percentage": spec.start,
                    "end_percentage": spec.end,
                }
            )
        return maps


register(InpaintControlNetRefiner())
