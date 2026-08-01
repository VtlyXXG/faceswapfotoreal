"""
Стратегия «специализированный фейссвоп» — `fal-ai/face-swap`.

Модуль назван по эндпоинту, а не по стратегии: имя `face_swap.py` внутри
`refine/` читалось бы как пакет `pipelines/face_swap/`, где лежит оркестрация
всего заказа. Стратегия в реестре зовётся `face_swap`.

Один вызов, две ссылки, ноль настроек:

  base_image_url — шаблон-разворот как есть;
  swap_image_url — фотография заказчика как есть, без поля и подготовки.

Почему настроек нет и почему это хорошо. Четыре предыдущих подхода отличались
именно настройками — сила, шаги, ширина маски, формулировка промпта, — и все
четыре провалились не на них. Заменить лицо по текстовому описанию диффузия не
умеет: широкая маска даёт хоррор-маску, узкая — сплющенную челюсть и чужую
этничность, кадр целиком — гладкий ком кожи. Здесь описывать нечего: модель
обучена ровно этой задаче, и результат зависит только от того, какие две
картинки прислали.

Локальной работы тоже нет. Маска не нужна — область эндпоинт находит своим
детектором, а всё вне неё оставляет нетронутым; вклейка не нужна по той же
причине. Фотография уходит теми же байтами, что прислал заказчик: поле вокруг
неё спасало kontext от переноса композиции, а чужому детектору лица только
мешает.

**Главная особенность этого эндпоинта — молчаливый отказ.** Не найдя лица на
любой из двух картинок, он не отвечает ошибкой: он возвращает присланный шаблон.
Ответ 200, изображение на месте, счёт выставлен, а замены нет — и уехать такой
разворот может прямо в печать. Отличить это от удачи можно только сравнением
ответа с тем, что отправляли, и делается это ниже.
"""

from __future__ import annotations

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.core.logging import get_logger
from app.pipelines import composite, fal_api
from app.pipelines.refine.base import RefineRequest, RefineResult, register
from app.pipelines.refine.profiles import RefineProfile
from app.utils.image import decode_image

log = get_logger(__name__)

# Насколько ответ должен отличаться от шаблона, чтобы считаться заменой. Уровни
# 0..255, среднее по кадру. Порог низкий намеренно: голова занимает несколько
# процентов разворота, и даже удачная замена двигает среднее немного. Всё, что
# ниже, — это перекодирование того же самого кадра, то есть отказ.
_CHANGED_MIN = 0.35


class FalFaceSwapRefiner:
    """Перенос лица специализированной моделью. Ни промпта, ни маски."""

    name = "face_swap"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        if profile.payload.prompt_field or profile.payload.images_field:
            # Схема с промптом или массивом картинок — не для этого эндпоинта:
            # он принимает ровно две ссылки поимённо. Дешевле упасть здесь, чем
            # получить 422 после двух загрузок в CDN
            raise InvalidImageError(
                "Схема запроса не для фейссвопа: он принимает две ссылки и ничего больше",
                {
                    "strategy": self.name,
                    "payload": profile.payload.name,
                    "expected": "face_swap",
                },
            )

        if request.expression:
            # Мимику здесь задать нечем: она приезжает с фотографии донора и
            # правится только тем, какое фото прислали. Молча проглотить
            # параметр нельзя — заказ уйдёт с ощущением, что эмоция учтена
            log.warning(
                "мимика на этой стратегии не действует: выражение приходит с фотографии",
                extra={"strategy": self.name, "expression": request.expression[:80]},
            )

        client = fal_api.client()

        # Порядок загрузок соответствует стоимости ошибки: шаблон самый тяжёлый
        # (25-30 МБ на 4K), и если ключ или баланс не в порядке, отказ придёт
        # уже на нём — до фотографии
        image_url = fal_api.upload(client, request.target, request.target_mime)
        identity_url = fal_api.upload(client, request.identity, request.identity_mime)

        # Профиль передаётся целиком: промпт, маску, силу и формат схема отсечёт
        # сама — у этого эндпоинта их не существует
        arguments = profile.payload.arguments(
            image_url=image_url,
            mask_url="",
            identity_url=identity_url,
            prompt=profile.prompt(request.expression),
            strength=profile.strength,
            guidance_scale=profile.guidance_scale,
            steps=profile.steps,
            output_format=request.output_format,
            identity_scale=profile.identity_scale,
        )

        meta = {**profile.report()}
        image, call_meta = fal_api.invoke(client, profile.endpoint, arguments, meta)

        changed = composite.difference(decode_image(request.target), decode_image(image))
        if changed < _CHANGED_MIN:
            # Тот самый молчаливый отказ. 422, а не 502: запрос корректен и
            # выполнен, просто на этой паре картинок замена не состоялась —
            # чинится она другим фото или другим шаблоном, а не повтором.
            raise NoFaceDetectedError(
                "Фейссвоп вернул шаблон без изменений: лицо не найдено "
                "на развороте или на фотографии",
                {
                    "strategy": self.name,
                    "endpoint": profile.endpoint,
                    "changed": round(changed, 3),
                    "threshold": _CHANGED_MIN,
                    # Детектор эндпоинта — не наш: MediaPipe находит лица там,
                    # где чужая модель слепнет, и наоборот
                    "hint": "детектор эндпоинта не видит лица; проверьте фотографию "
                    "и фотореалистичность шаблона",
                },
            )

        # Формат ответа выбирает эндпоинт: output_format он не принимает.
        # Перекодировать под запрошенный не станем — это стоило бы ровно той
        # сохранности кадра, ради которой сюда и шли; настоящий тип уезжает в
        # мете, и Node сохраняет файл по нему
        mime_type = call_meta.get("mime_type", "image/png")
        if request.output_format not in mime_type:
            log.info(
                "формат ответа выбран эндпоинтом, а не запросом",
                extra={"requested": request.output_format, "mime_type": mime_type},
            )

        return RefineResult(image=image, meta={**meta, **call_meta, "changed": round(changed, 3)})


register(FalFaceSwapRefiner())
