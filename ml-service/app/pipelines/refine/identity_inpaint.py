"""
Стратегия «генеративный инпейнт с подмешиванием идентичности».

Один вызов, три картинки на входе:

  image_url  — шаблон-разворот как есть. Ни вклейки, ни коллажа: всё, что было
               локальной аппликацией, отсюда убрано;
  mask_url   — голова и шея персонажа. Внутри маски модель рисует заново,
               снаружи не трогает ни пикселя;
  identity   — фотография заказчика. Из неё берутся черты, тон кожи и цвет
               волос; поза, поворот головы и мимика остаются от персонажа —
               они уже нарисованы вокруг маски и внутри неё видны модели как
               контекст.

Почему так, а не через identity-модели напрямую. `fal-ai/flux-pulid` и
`fal-ai/ip-adapter-face-id` личность держат отлично, но принимают только
prompt + фото лица: ни базового кадра, ни маски. Отдать им разворот означает
получить обратно другой разворот — сгенерированный с нуля, без нашей
иллюстрации. Поэтому идентичность идёт тем каналом, который эндпоинт-инпейнтер
понимает: референсным изображением.

Схема запроса живёт в профиле (`profiles.PayloadSchema`), а не здесь. Имена
полей у эндпоинтов разные, лишний ключ fal не игнорирует, а заворачивает весь
запрос — и менять их правкой кода стратегии значит гарантированно однажды
забыть про второй эндпоинт.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.pipelines import fal_api
from app.pipelines.refine.base import RefineRequest, RefineResult, register
from app.pipelines.refine.profiles import RefineProfile

log = get_logger(__name__)


class IdentityInpaintRefiner:
    """Генерация головы внутри маски шаблона по референсу личности."""

    name = "identity_inpaint"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        if not request.mask:
            raise fal_api.MaskMissingError(
                "Инпейнтингу нужна маска головы, но она не построена",
                {"strategy": self.name},
            )

        if profile.strength < profile.safe_strength:
            # Не отказ: значение переопределяется из окружения именно для
            # подбора. Но под маской лежит голова чужого персонажа, и слабая
            # генерация оставляет от неё черты — смесь двух лиц. В логе это
            # должно быть видно раньше, чем на печати тиража.
            log.warning(
                "strength ниже безопасного: под генерацией могут проступить черты персонажа",
                extra={"strength": profile.strength, "safe_min": profile.safe_strength},
            )

        client = fal_api.client()
        fmt = "jpeg" if request.output_format in ("jpg", "jpeg") else "png"

        # Порядок загрузок соответствует стоимости ошибки: шаблон самый тяжёлый
        # (25-30 МБ на 4K), и если ключ или баланс не в порядке, отказ придёт
        # уже на нём — до маски и референса
        image_url = fal_api.upload(client, request.target, request.target_mime)
        mask_url = fal_api.upload(client, request.mask, request.mask_mime)
        identity_url = fal_api.upload(client, request.identity, request.identity_mime)

        arguments = profile.payload.arguments(
            image_url=image_url,
            mask_url=mask_url,
            identity_url=identity_url,
            prompt=profile.prompt(request.expression),
            strength=profile.strength,
            guidance_scale=profile.guidance_scale,
            steps=profile.steps,
            output_format=fmt,
            identity_scale=profile.identity_scale,
        )

        meta = {**profile.report(), "output_format": fmt}
        image, call_meta = fal_api.invoke(client, profile.endpoint, arguments, meta)

        return RefineResult(image=image, meta={**meta, **call_meta})


register(IdentityInpaintRefiner())
