"""
Стратегия «сначала причёска, потом лицо»: два вызова fal вместо одного.

Откуда взялась. Фейссвоп отработал ровно так, как обещал, и ровно в тех
границах, которые у него есть: лицо перенеслось, а причёска осталась от
шаблона. Настроить это нечем — эндпоинт не читает текста и волос не касается по
устройству. Значит, причёску надо править отдельно, до него.

Порядок шагов — главное решение этого модуля, и он именно такой:

  1. **Причёска.** Редактор получает ОКНО вокруг головы, текст — и больше
     ничего. Из его ответа в шаблон возвращается ровно область причёски
     (`hair_mask` + `composite.paste`); всё остальное — и лицо в том числе —
     остаётся побитово исходным.
  2. **Лицо.** Тот самый `fal-ai/face-swap`, теми же двумя ссылками, той же
     стратегией `face_swap` — она вызывается отсюда из реестра и не знает, что
     шаблон по дороге поправили.

**Фотография заказчика на первый шаг не отправляется, и это не экономия.**
Универсальный редактор понимает портрет крупным планом не как «вот чья
причёска», а как «вот что нарисовать»: получив его вторым файлом, и
nano-banana, и seedream нарисовали лицо донора на затылке персонажа — в пустой
области маски волос, то есть ровно там, где мы разрешили рисовать. Личность
переносит специализированная модель на втором шаге; общему редактору на первом
достаточно слов. Отсюда и требование к описанию причёски: без него первый шаг не
запускается вовсе — рисовать нечего.

Почему не наоборот. Правка волос — это диффузия, и всё, что попадёт под её
маску, вернётся сглаженным. Пусти её после переноса, и она пройдёт по настоящим
фотографическим пикселям лица: сначала по краю, потом по виску, потом по брови —
то есть по единственному, ради чего вся затея. Первым шагом она рискует только
нарисованной головой шаблона, которую всё равно заменят.

И вторая причина, менее очевидная: детектор фейссвопа ищет лицо сам. Шаблон, на
котором он однажды сработал, — проверенный вход, и трогать в нём лицо нельзя,
иначе молчаливый отказ («200, кадр без замены») станет вероятным исходом. Отсюда
защита лица в маске волос: после первого шага кадр отличается от исходного
ТОЛЬКО внутри причёски, а глаза, нос и рот доезжают до фейссвопа нетронутыми.

Почему окно, а не разворот. Первый живой прогон правку сделал — фон на месте
срезанных волос дорисовался безупречно, — но объём причёски остался шаблонным:
вместо жёсткого ёжика вышел гладкий блондинистый шлем. Причина не в силе правки
(её у этих эндпоинтов нет как параметра, схемы проверены), а в масштабе: на
4K-развороте голова занимает проценты кадра, рабочий кадр редактора около
мегапикселя, и на сотню пикселей волос он отдаёт мыльную шапку — стричь там
физически нечего. В окне те же волосы получают тысячи пикселей.

Чем этот путь дороже: двумя вызовами вместо одного, вторым счётом и одним новым
поводом отказать заказу — редактор тоже умеет промолчать, вернув кадр без
правки. Ловится это тем же приёмом, что и у фейссвопа: сравнением с тем, что
отправляли, только внутри маски.
"""

from __future__ import annotations

from dataclasses import replace

from app.core.errors import InvalidImageError, MLServiceError
from app.core.logging import get_logger
from app.pipelines import composite, erase, fal_api, hair_mask
from app.pipelines.refine.base import RefineRequest, RefineResult, get, register
from app.pipelines.refine.profiles import RefineProfile, strategy_defaults
from app.utils.image import decode_image, encode_image

log = get_logger(__name__)

# Ниже какого размера окно вокруг головы отдавать бессмысленно: редактор
# растянет его до своего рабочего кадра и дорисует детали, которых в исходнике
# не было. Порог не запрет, а строка в логе — на превью-размерах разворота
# маленькое окно всё-таки лучше мелкой головы в полном кадре.
_WINDOW_MIN_PX = 256

# Имя стратегии второго шага. Строкой, а не импортом класса: шаг замены лица
# берётся из реестра ровно так же, как его берёт pipeline.py, — то есть работает
# та же стратегия, что и на одношаговом пути, а не её копия.
_SWAP = "face_swap"


class HairEditFailedError(MLServiceError):
    """
    Редактор вернул кадр без правки причёски.

    422, а не 502, по той же логике, что и молчаливый отказ фейссвопа: запрос
    корректен и выполнен, просто на этой паре картинок правка не состоялась.
    Чинится она другой фотографией, другим шаблоном или другим редактором — но
    не повтором того же запроса.
    """

    status_code = 422
    code = "HAIR_EDIT_FAILED"


def _window(
    shape: tuple[int, int], hair: hair_mask.HairMask, crop_ratio: float
) -> list[int] | None:
    """
    Окно вокруг причёски: [left, top, width, height] либо None.

    Считается по маске волос с полем в долях высоты лица. Поле нужно не для
    красоты: редактор должен видеть, во что вписана голова — плечи, воротник,
    фон, — иначе он теряет свет сцены и перспективу. Но видеть весь разворот ему
    незачем, и это стоило первого прогона: на полном кадре модель сохранила
    объём причёски, потому что стричь на сотне пикселей нечего.

    :param crop_ratio: поле вокруг маски, доли высоты лица. 0 — окна нет
    """
    import numpy as np

    if crop_ratio <= 0:
        return None

    rows = np.flatnonzero(np.any(hair.mask > 0, axis=1))
    columns = np.flatnonzero(np.any(hair.mask > 0, axis=0))
    if not rows.size or not columns.size:
        return None

    height, width = shape
    margin = max(1, int(round(hair.face_height * crop_ratio)))

    top = max(0, int(rows[0]) - margin)
    bottom = min(height, int(rows[-1]) + 1 + margin)
    left = max(0, int(columns[0]) - margin)
    right = min(width, int(columns[-1]) + 1 + margin)

    if (right - left, bottom - top) == (width, height):
        # Окно совпало с кадром — головы тут столько же, сколько разворота.
        # Резать нечего, и лишнее перекодирование ни к чему
        return None

    if min(right - left, bottom - top) < _WINDOW_MIN_PX:
        log.warning(
            "окно вокруг головы меньше рабочего кадра редактора — он будет додумывать детали",
            extra={"window": [left, top, right - left, bottom - top], "floor": _WINDOW_MIN_PX},
        )

    return [left, top, right - left, bottom - top]


def _cut(image, mask, window: list[int] | None):
    """Кадр и маска, обрезанные окном. Без окна — они же целиком."""
    if window is None:
        return image, mask

    left, top, width, height = window
    return image[top : top + height, left : left + width], mask[
        top : top + height, left : left + width
    ]


def _restore(target, edited, window: list[int] | None):
    """
    Возвращает поправленное окно на место в разворот.

    Копия, а не правка на месте: `target` — это декодированный шаблон заказа, и
    менять его под собой значит однажды получить фейссвоп по кадру, который уже
    кто-то потрогал. Вне окна пиксели шаблона остаются исходными, вне маски
    внутри окна — тоже: об этом позаботилась вклейка.
    """
    if window is None:
        return edited

    left, top, width, height = window
    restored = target.copy()
    restored[top : top + height, left : left + width] = edited
    return restored


class HairThenFaceSwapRefiner:
    """Правка причёски редактором, затем перенос лица фейссвопом."""

    name = "hair_swap"

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        stage = profile.hair

        if not stage.describes_hair(request.hair):
            # Референса на этом шаге нет, и рисовать модели нечего: «замени
            # причёску» без описания означает случайную причёску. 400, а не
            # умолчание, — заказ чинится одним полем, а не пересъёмкой
            raise InvalidImageError(
                "Не сказано, какую причёску рисовать, а фотографию на этот шаг "
                "не отправляют — редактор нарисовал бы по ней второе лицо",
                {
                    "strategy": self.name,
                    "hint": "передайте поле hair в запросе или задайте "
                    "ML_HAIR_DESCRIPTION на весь тираж",
                },
            )

        # Маска строится ДО сети. Она же — единственная локальная работа этого
        # пути, и если персонажа на шаблоне не нашли, узнать об этом надо здесь,
        # а не после четырёх загрузок в CDN и двух инференсов
        target = decode_image(request.target)
        hair = hair_mask.build(
            target,
            dilate_ratio=stage.dilate_ratio,
            feather_ratio=stage.feather_ratio,
            protect_ratio=stage.protect_ratio,
            forehead_ratio=stage.forehead_ratio,
            core_ratio=stage.core_ratio,
            guard_ratio=stage.guard_ratio,
            cheek_ratio=stage.cheek_ratio,
        )

        # Редактору уезжает не разворот, а окно вокруг головы. Причина
        # измеренная: на полном 4K-развороте причёска занимает проценты кадра,
        # рабочий кадр модели около мегапикселя — и на сотню пикселей волос она
        # отдаёт мыльную шапку, потому что стричь там нечего. В окне те же
        # волосы получают тысячи пикселей, и стрижка становится стрижкой.
        window = _window(target.shape[:2], hair, stage.crop_ratio)
        scene, scene_mask = _cut(target, hair.mask, window)

        # Под маской у редактора не должно остаться СТРУКТУРЫ старых волос, см.
        # `erase.py`. Геометрия маски эту прядь давно отдаёт (hair_left_px
        # считает единицы пикселей), а редактор всё равно возвращает её на
        # место: тёмная линия на щеке читается им как тень скулы или край
        # челюсти, то есть как лицо, которое ему запрещено трогать. Слова против
        # пикселей проигрывают, поэтому уезжают пиксели без пряди
        sent, erase_meta = erase.wipe(scene, scene_mask, hair.face_height, stage.erase_ratio)

        client = fal_api.client()
        if window is None and not erase_meta["erased"]:
            # Разворот уходит теми же байтами, что прислали: перекодировать
            # незачем, мы в нём ничего не меняли
            scene_bytes, scene_mime = request.target, request.target_mime
        else:
            scene_bytes, scene_mime = encode_image(sent, "png")

        image_url = fal_api.upload(client, scene_bytes, scene_mime)

        # Фотография заказчика на этот шаг не уезжает — ни в CDN, ни в запрос.
        # Схема это знает (`images_identity=False`) и в массив её не положит;
        # пустая ссылка здесь означает «нечего класть», а не «забыли загрузить».
        # Проверено дважды и дорого: получив портрет крупным планом, оба живых
        # редактора нарисовали лицо донора на затылке персонажа
        arguments = stage.payload.arguments(
            image_url=image_url,
            mask_url="",
            identity_url="",
            prompt=stage.prompt(request.hair),
            strength=profile.strength,
            guidance_scale=profile.guidance_scale,
            steps=profile.steps,
            output_format="png",
            identity_scale=profile.identity_scale,
        )

        generated, call_meta = fal_api.invoke(
            client,
            stage.endpoint,
            arguments,
            {"stage": "hair", "window": window, **stage.report()},
        )

        # Из ответа берётся ровно причёска. Редактор правит кадр целиком, и без
        # этого шага в печать уехали бы его версии ткани, фона и — главное —
        # лица, которое фейссвопу ещё предстоит найти
        #
        # match=True — против светлого ореола. Маска волос обходит голову
        # кольцом и там, где лежала грива, выходит на небо; общий увод тона у
        # редактора становится в этом кольце свечением вокруг головы. Поправка
        # считается по окну ВНЕ маски, где обе картинки — одна и та же сцена
        # Вклейка идёт в ИСХОДНЫЙ кадр, а не в стёртый: стирание — свойство
        # запроса, и в готовый разворот его пиксели не попадают никогда. Поправка
        # тона от этого не страдает — она считается ВНЕ маски, где стирание по
        # построению не тронуло ни пикселя
        pasted = composite.paste(scene, decode_image(generated), scene_mask, match=True)

        # Изменилось ли хоть что-то ВНУТРИ маски. Среднее по кадру здесь не
        # годится: вне маски пиксели шаблона возвращены побитово, и любая правка
        # утонула бы в этом нуле.
        #
        # Сравнивается ответ с тем, что ОТПРАВЛЯЛИ, а не с шаблоном. Разница
        # появляется вместе со стиранием: молчаливый отказ редактора — это
        # присланный кадр обратно, то есть кадр СТЁРТЫЙ, и от шаблона он
        # отличается сильно. Сравнение с шаблоном объявило бы такой ответ удачной
        # правкой и отправило бы фейссвопу размытое пятно вместо причёски
        changed = composite.difference(sent, pasted.image, scene_mask)
        if changed < stage.min_changed:
            raise HairEditFailedError(
                "Редактор вернул причёску без изменений: волосы остались от шаблона",
                {
                    "strategy": self.name,
                    "endpoint": stage.endpoint,
                    "changed": round(changed, 3),
                    "threshold": stage.min_changed,
                    "hint": "проверьте, видна ли причёска донора на фотографии, "
                    "и попробуйте другой редактор в ML_HAIR_ENDPOINT",
                },
            )

        edited, edited_mime = encode_image(_restore(target, pasted.image, window), "png")
        hair_meta = {
            **hair.meta,
            "changed": round(changed, 3),
            "endpoint": stage.endpoint,
            "window": window,
            "erase": erase_meta,
            "composite": pasted.meta,
            "generated_url": call_meta.get("image_url"),
            "seed": call_meta.get("seed"),
        }
        log.info("причёска перенесена, шаблон готов к замене лица", extra=hair_meta)

        # Шаг второй. Рабочая стратегия вызывается из реестра как есть, и
        # профиль ей достаётся тот же — с её собственными эндпоинтом и схемой.
        # Единственное отличие входа: шаблон теперь поправленный, и молчаливый
        # отказ фейссвоп будет искать сравнением с ним же, что и правильно
        result = get(_SWAP).refine(
            replace(request, target=edited, target_mime=edited_mime),
            self._swap_profile(profile),
        )

        return RefineResult(
            image=result.image,
            meta={
                **result.meta,
                # Отчёт фейссвопа называет своей стратегию второго шага —
                # возвращаем имя настоящей: заказ прошёл двумя вызовами, и по
                # мете это должно быть видно сразу
                "strategy": self.name,
                "hair": hair_meta,
            },
        )

    @staticmethod
    def _swap_profile(profile: RefineProfile) -> RefineProfile:
        """
        Профиль второго шага: тот же самый, но с именем рабочей стратегии.

        Эндпоинт и схема берутся из профиля, а не из умолчаний стратегии, — так
        `ML_REFINE_ENDPOINT` продолжает означать «эндпоинт замены лица» и на
        двухшаговом пути. Умолчания подставляются, только если профиль пришёл с
        чужой схемой: собранный из частей разных стратегий, он стоил бы 422
        после двух загрузок в CDN.
        """
        defaults = strategy_defaults(_SWAP)
        if defaults is None or profile.payload.name == defaults.payload.name:
            return replace(profile, strategy=_SWAP)

        log.warning(
            "схема профиля не фейссвопная — второй шаг идёт на умолчаниях стратегии",
            extra={"payload": profile.payload.name, "expected": defaults.payload.name},
        )
        return replace(
            profile,
            strategy=_SWAP,
            endpoint=defaults.endpoint,
            payload=defaults.payload,
            instruction=defaults.instruction,
        )


register(HairThenFaceSwapRefiner())
