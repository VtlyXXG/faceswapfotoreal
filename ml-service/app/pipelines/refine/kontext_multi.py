"""
Стратегия «безмасочная генерация кадра целиком + локальная вклейка головы».

Один вызов, две картинки в одном массиве:

  image_urls[0] — шаблон-разворот как есть;
  image_urls[1] — фотография заказчика. Из неё берутся черты, тон кожи и цвет
                  волос; поза, поворот головы и мимика остаются от персонажа.

Почему без маски. Масочный путь (`identity_inpaint.py`) провалился на том, что
маска и создаёт: под ней у модели нет контекста. Она не видит ни глаз, которые
заменяет, ни линии челюсти, ни того, как на скулу падает свет, — и отдаёт
искажённые пропорции, артефакты по краю и чужую личность. Здесь модель видит
кадр целиком, включая лицо, которое заменяет, и весь свет сцены.

Чем за это платим и как расплачиваемся. Перерисовывается ВЕСЬ кадр: вместе с
головой поедут фактура ткани, надписи, узор обоев — а шаблон обязан дойти до
печати неизменным. Поэтому из ответа берётся ровно голова: маска строится
локально, как и раньше, но уезжает не в fal, а в `composite.paste`. Фон и
одежда после этого — побитово исходные пиксели шаблона.

Схема запроса живёт в профиле (`profiles.KONTEXT_MULTI`), а не здесь. У
kontext/max/multi всего четыре ключа: prompt, image_urls, guidance_scale,
output_format. Ни mask_url, ни strength, ни num_inference_steps он не принимает,
и лишний ключ не игнорируется, а заворачивает весь запрос 422-й — до инференса,
но уже после загрузок в CDN.
"""

from __future__ import annotations

from app.core.errors import InvalidImageError
from app.core.logging import get_logger
from app.pipelines import composite, fal_api
from app.pipelines.refine.base import RefineRequest, RefineResult, register
from app.pipelines.refine.profiles import RefineProfile
from app.utils.image import decode_image, encode_image

log = get_logger(__name__)


class KontextMultiRefiner:
    """Генерация кадра целиком по двум картинкам и вклейка головы в шаблон."""

    name = "kontext_multi"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        if not request.mask:
            # Маска не уезжает в fal, но без неё нечем взять из ответа одну
            # голову — а вернуть весь кадр значит отдать в печать перерисованные
            # ткань и фон. Это дефект вызывающего кода, отсюда 500 до сети.
            raise fal_api.MaskMissingError(
                "Безмасочной генерации маска нужна для вклейки, но она не построена",
                {"strategy": self.name},
            )

        if not profile.payload.images_field:
            # Схема без массива картинок означает, что профиль собран из чужих
            # частей: эндпоинт получит image_url вместо image_urls и завернёт
            # запрос. Дешевле упасть здесь, чем после двух загрузок в CDN.
            raise InvalidImageError(
                "Схема запроса не передаёт картинки массивом — она не для этой стратегии",
                {
                    "strategy": self.name,
                    "payload": profile.payload.name,
                    "expected": "kontext_multi",
                },
            )

        client = fal_api.client()
        fmt = "jpeg" if request.output_format in ("jpg", "jpeg") else "png"

        # Порядок загрузок соответствует стоимости ошибки: шаблон самый тяжёлый
        # (25-30 МБ на 4K), и если ключ или баланс не в порядке, отказ придёт
        # уже на нём — до референса
        image_url = fal_api.upload(client, request.target, request.target_mime)
        identity_url = fal_api.upload(client, request.identity, request.identity_mime)

        # Профиль передаётся целиком: strength, steps и маску схема отсечёт сама
        arguments = profile.payload.arguments(
            image_url=image_url,
            mask_url="",
            identity_url=identity_url,
            prompt=profile.prompt(request.expression),
            strength=profile.strength,
            guidance_scale=profile.guidance_scale,
            steps=profile.steps,
            output_format=fmt,
            identity_scale=profile.identity_scale,
        )

        meta = {**profile.report(), "output_format": fmt}
        generated, call_meta = fal_api.invoke(client, profile.endpoint, arguments, meta)

        # Локальный шаг: из ответа берётся голова, всё остальное — из шаблона
        pasted = composite.paste(
            decode_image(request.target), decode_image(generated), decode_image(request.mask)
        )
        image, mime_type = encode_image(pasted.image, fmt)

        # Ссылка на ответ fal переименовывается, а не сохраняется как есть:
        # image_url в мете читается как «вот результат», а результат теперь
        # другой — по этой ссылке лежит кадр ДО вклейки, с перерисованным фоном
        call_meta = dict(call_meta)
        call_meta["generated_url"] = call_meta.pop("image_url", None)

        return RefineResult(
            image=image,
            meta={
                **meta,
                **call_meta,
                # Отдаём мы не то, что вернул fal, а результат вклейки: и тип
                # содержимого, и байты теперь наши
                "mime_type": mime_type,
                "composite": pasted.meta,
            },
        )


register(KontextMultiRefiner())
