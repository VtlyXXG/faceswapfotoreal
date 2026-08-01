"""
Все гиперпараметры генерации в одном месте.

Здесь лежат числа и тексты, а не логика: сила инпейнтинга, доли маски головы,
идентификаторы эндпоинтов, имена полей запроса и промпты стилей. Профиль —
замороженный dataclass: из него нельзя получить картинку, он ничего не вызывает
и ни от чего не зависит; его можно распечатать в лог, сравнить с другим и
положить в /health/ready целиком.

Смена парадигмы. Раньше пайплайн вырезал голову заказчика и вклеивал её в
шаблон, а полдюжины проходов инпейнтинга пытались этот стык спрятать: фон, шея,
фактура, шов, общая текстура. Эталонные листы показали, что так нужного качества
не получить в принципе — фотографические пиксели на живописи остаются
фотографическими, сколько их ни своди, а мимику и поворот головы персонажа
аппликация не перенимает вовсе.

Дальше личность стали передавать эндпоинту **референсом**, а лицо рисовать
внутри маски. Это тоже провалилось, и по причине, которую маска и создаёт: под
маской у модели нет контекста — ни глаз, ни линии челюсти, ни света на скуле, —
и она отдаёт искажённые пропорции, артефакты по краю и чужую личность.

Безмасочная диффузия по всему кадру (`kontext_multi`) провалилась следом:
общая img2img-модель не умеет хирургически заменить лицо по тексту и на месте
головы отдаёт гладкий бесформенный ком кожи. Стратегия оставлена в реестре —
воспроизвести провал одной переменной окружения дешевле, чем спорить о нём.

Текущая схема — **специализированный фейссвоп** (`refine/easel_face_swap.py`):
модель лица вместо диффузии, никакого промпта, никакой маски и никакой локальной
вклейки. Фотореализм здесь не выпрашивается словами: переносится настоящее лицо,
а всё вне него эндпоинт оставляет своим.

У фейссвопа есть своё ограничение, и оно не настраивается: он переносит ЛИЦО.
Причёска остаётся от шаблона, потому что модель её не касается вовсе. Оттуда
второй пресет фотореалистичной серии — **двухшаговый** (`refine/hair_swap.py`):
редактор по двум картинкам правит одни волосы, результат правки возвращается в
шаблон по маске волос, и уже на него идёт тот же самый фейссвоп, без единого
изменения в нём.

Пресеты:

  pixar_real  — фотореализм: кожа с текстурой, мягкие тени, 85 мм;
  pixar_hair  — то же плюс перенос причёски донора отдельным проходом;
  impasto     — живопись: густой мазок, фактура холста.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from app.config import settings
from app.core.errors import InvalidImageError

# Эндпоинты.
#
# fal-ai/face-swap — рабочий. Специализированный фейссвоп: своя модель лица,
#   никакого промпта и никакой диффузии по кадру. Схема из двух ссылок,
#   base_image_url + swap_image_url, и ничего больше. Фотореализм тут не
#   выпрашивается словами — переносятся настоящие пиксели лица, а всё вне его
#   эндпоинт оставляет нетронутым. Пришёл на смену easel-ai/advanced-face-swap,
#   который fal объявил устаревшим («no longer supported»).
#
#   Две особенности, обе принципиальные. Ответ приходит полем `image`, а не
#   списком `images`, — см. fal_api._extract_image. И, главное: **лица он ищет
#   своим детектором, а не найдя, молча возвращает исходный кадр.** Отказа при
#   этом нет: есть счёт за вызов и разворот без замены. Проверяется это уже у
#   нас, в стратегии, сравнением ответа с шаблоном.
#
# kontext/max/multi — безмасочный img2img. Принимает МАССИВ картинок
#   (шаблон + фотография) и перерисовывает кадр целиком. Маска ему не
#   передаётся вовсе — она нужна локально, чтобы вернуть в шаблон только
#   голову. Ни mask_url, ни strength, ни num_inference_steps схема запроса не
#   знает: лишний ключ заворачивает весь запрос 422-й.
# kontext-inpaint — прежний путь по маске. Оставлен, но провалился: внутри
#   пустой маски у модели нет контекста, и она отдаёт искажённые пропорции,
#   артефакты по краю и потерю личности.
# flux-general/inpainting — альтернатива с ip_adapters. Оставлена схемой, а не
#   удалена: у неё принципиально другой способ подмешивать личность, и когда
#   XLabs-адаптер сменится на identity-совместимый, переключение будет стоить
#   одной переменной окружения. Ставить её по умолчанию нельзя: на нашем
#   доноре она дважды нарисовала чужое лицо — ip-adapter переносил стиль.
#
# Чего в этом списке НЕТ и почему. PuLID (fal-ai/pulid, fal-ai/flux-pulid) жив и
# не устарел, но принимает prompt + фотографию лица и генерирует НОВЫЙ кадр в
# заданном image_size: базовой картинки, которую можно править, у него нет.
# InstantID отдельным эндпоинтом на fal не выставлен вовсе. ControlNet задаёт
# структуру, а не личность, и нашей задачи не решает ни в каком виде: старый
# inpaint_controlnet.py удалён именно поэтому.
_FAL_FACE_SWAP = "fal-ai/face-swap"
_KONTEXT_MAX_MULTI = "fal-ai/flux-pro/kontext/max/multi"
_KONTEXT_INPAINT = "fal-ai/flux-kontext-lora/inpaint"
_FLUX_GENERAL = "fal-ai/flux-general/inpainting"

# Мультикартиночные редакторы — запасной путь стратегии kontext_multi. Схема у
# них одна и та же (prompt + массив image_urls), поэтому переключение стоит двух
# переменных окружения и не требует ни строчки кода. Нужны они там, где голова
# персонажа стилизована: фейссвоп переносит фотографические пиксели как есть, и
# на рендере они могут не сойтись с материалом сцены. Все три проверены живыми,
# схемы сверены с openapi.
_NANO_BANANA_EDIT = "fal-ai/nano-banana/edit"
_SEEDREAM_EDIT = "fal-ai/bytedance/seedream/v4/edit"
_HY_WU_EDIT = "fal-ai/hy-wu-edit"


@dataclass(frozen=True)
class PayloadSchema:
    """
    Имена полей запроса — то, чем эндпоинты отличаются друг от друга.

    Вынесено в данные не ради красоты. Лишний или неверно названный ключ fal не
    игнорирует, а заворачивает весь запрос, и узнаётся это только после боевого
    прогона — уже узнавали. Пока схема живёт здесь, смена эндпоинта стоит
    правки одного объекта, а не поиска по коду стратегии.

    **Имя, равное None, означает «этого ключа у эндпоинта нет».** Ключ тогда не
    попадает в тело вовсе, и стратегия об этом не знает: она передаёт весь
    профиль целиком, а лишнее отсекается здесь. Ради этого правила поля и
    объявлены необязательными — kontext/max/multi не понимает ни маски, ни
    strength, ни числа шагов, и любой из этих ключей означает 422 на весь
    запрос.

    :param images_field: имя поля-МАССИВА, если эндпоинт принимает картинки
        списком. Задано — шаблон и фотография уезжают вместе, ``image_field`` и
        ``identity_field`` не используются вовсе
    :param images_identity: кладётся ли в этот массив фотография заказчика.
        False — уезжает один шаблон, а личность описывается словами.

        Флаг стоил боди-хоррора. Универсальный редактор, получив вторым файлом
        портрет крупным планом, понимает его не как «вот чья причёска», а как
        «вот что нарисовать»: на затылке персонажа появилось второе лицо
        донора — прямо в пустой области маски волос. Специализированному
        фейссвопу фотография нужна, общему редактору на шаге причёски — нет
    :param identity_kind: как передаётся референс личности.
        ``url`` — ссылкой в одном поле (kontext);
        ``list`` — списком объектов со своим весом (ip_adapters, image_prompts).
    :param prompt_field: None у эндпоинтов, которые вообще не читают текст.
        Фейссвоп — именно такой: он переносит лицо моделью, а не по описанию
    :param extra: постоянные ключи, которые эндпоинт требует всегда
    """

    name: str
    image_field: str | None = "image_url"
    images_field: str | None = None
    images_identity: bool = True
    mask_field: str | None = "mask_url"
    identity_field: str | None = "reference_image_url"
    identity_kind: str = "url"
    prompt_field: str | None = "prompt"
    strength_field: str | None = "strength"
    guidance_field: str | None = "guidance_scale"
    steps_field: str | None = "num_inference_steps"
    format_field: str | None = "output_format"
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.image_field and not self.images_field:
            raise InvalidImageError(
                "Схема запроса не называет поля для шаблона: нужен image_field или images_field",
                {"payload": self.name},
            )

    def sent_keys(self) -> list[str]:
        """
        Имена ключей, которые схема положит в тело. Первое, на что смотрят при
        422: у безмасочного эндпоинта их четыре, и лишний он не прощает.
        """
        names = [self.prompt_field, self.images_field or self.image_field]
        if not self.images_field:
            names.append(self.identity_field)
        names += [
            self.mask_field,
            self.strength_field,
            self.guidance_field,
            self.steps_field,
            self.format_field,
            *(name for name, _ in self.extra),
        ]
        return sorted(name for name in names if name)

    def _identity(self, identity_url: str, identity_scale: float) -> Any:
        if self.identity_kind == "url":
            return identity_url
        if self.identity_kind == "list":
            return [{"image_url": identity_url, "scale": identity_scale}]
        raise InvalidImageError(
            f"Неизвестный способ передачи личности «{self.identity_kind}»",
            {"payload": self.name, "supported": ["url", "list"]},
        )

    def arguments(
        self,
        image_url: str,
        mask_url: str,
        identity_url: str,
        prompt: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        output_format: str,
        identity_scale: float,
    ) -> dict:
        """
        Собирает тело запроса. Чистая сборка словаря: ни сети, ни файлов.

        Принимаются все аргументы и всегда — что из них уедет, решает схема.
        Вызывающий не обязан помнить, какой эндпоинт чего не понимает; он и не
        может этого помнить, потому что эндпоинт меняется переменной окружения.
        """
        body: dict = {}
        if self.prompt_field:
            body[self.prompt_field] = prompt

        if self.images_field:
            # Массив вместо пары полей. Порядок — часть контракта с промптом:
            # шаблон первым («the FIRST image»), фотография второй. Либо
            # фотографии нет вовсе — тогда и в промпте её быть не должно
            body[self.images_field] = (
                [image_url, identity_url] if self.images_identity else [image_url]
            )
        else:
            body[self.image_field] = image_url
            if self.identity_field:
                body[self.identity_field] = self._identity(identity_url, identity_scale)

        for name, value in (
            (self.mask_field, mask_url),
            (self.strength_field, strength),
            (self.guidance_field, guidance_scale),
            (self.steps_field, steps),
            (self.format_field, output_format),
        ):
            if name:
                if isinstance(value, str) and not value:
                    # Пустая строка в поле, которое схема объявила, — это 422
                    # после загрузок в CDN. Числа не проверяются: ноль у них
                    # законное значение, а пустота бывает только у строк, и
                    # означает она одно — профиль собран не для этой схемы
                    raise InvalidImageError(
                        f"Схема «{self.name}» требует непустое поле «{name}»",
                        {"payload": self.name, "field": name},
                    )
                body[name] = value

        return {**body, **dict(self.extra)}


KONTEXT = PayloadSchema(name="kontext_inpaint")

# Безмасочная схема: только четыре ключа, и это не экономия, а требование
# эндпоинта. mask_url / strength / num_inference_steps он не принимает —
# каждый из них означает 422 на весь запрос, ещё до инференса.
KONTEXT_MULTI = PayloadSchema(
    name="kontext_multi",
    image_field=None,
    images_field="image_urls",
    identity_field=None,
    mask_field=None,
    strength_field=None,
    steps_field=None,
)

# Схема фейссвопа: две ссылки и ничего больше. Ни промпта, ни маски, ни силы,
# ни формата, ни размера — он не диффузия, и сказать ему нечего. Настраивать тут
# нечего в принципе, и это не бедность схемы, а её честность: результат зависит
# только от того, какие две картинки прислали.
FACE_SWAP = PayloadSchema(
    name="face_swap",
    image_field="base_image_url",
    identity_field="swap_image_url",
    prompt_field=None,
    mask_field=None,
    strength_field=None,
    guidance_field=None,
    steps_field=None,
    format_field=None,
)

# Мультикартиночные редакторы: шаблон и фотография массивом, замена описывается
# промптом. Схема совпадает с kontext/max/multi, поэтому работают они на той же
# стратегии — вместе с локальной вклейкой, которая защищает шаблон. Нужны там,
# где голова персонажа стилизована и фотографические пиксели с ней не сойдутся.
NANO_BANANA = PayloadSchema(
    name="nano_banana",
    image_field=None,
    images_field="image_urls",
    identity_field=None,
    mask_field=None,
    strength_field=None,
    guidance_field=None,
    steps_field=None,
)

# У Seedream своя ценность: единственный из трёх отдаёт 4K. Для разворота,
# который идёт в печать, это разница между мягкой головой и резкой. Формата
# вывода его схема не знает — ключ output_format ему слать нельзя.
SEEDREAM_EDIT = PayloadSchema(
    name="seedream_edit",
    image_field=None,
    images_field="image_urls",
    identity_field=None,
    mask_field=None,
    strength_field=None,
    guidance_field=None,
    steps_field=None,
    format_field=None,
    extra=(("image_size", "auto_4K"),),
)

# HY-WU заявлен как редактор именно для переноса лица, и у него есть режим
# размышления перед правкой: модель сначала разбирает задачу, потом рисует.
HY_WU_EDIT = PayloadSchema(
    name="hy_wu_edit",
    image_field=None,
    images_field="image_urls",
    identity_field=None,
    mask_field=None,
    strength_field=None,
    guidance_field=None,
    # num_inference_steps он принимает, но профильные 40 — число из эпохи Flux,
    # и подставлять его чужой модели незачем: своё умолчание она знает лучше
    steps_field=None,
    extra=(("enable_thinking", True),),
)

# Те же редакторы, но БЕЗ фотографии заказчика: в массив уезжает один шаблон.
# Схемы шага причёски — и единственные, которые он принимает (см. validate).
#
# Почему отдельными объектами, а не флагом в профиле. Состав массива — свойство
# запроса, а не стратегии: перепутав их местами, мы не получим ни ошибки, ни
# 422 — получим успешный ответ с лицом донора, нарисованным на затылке. Такое
# ловится только тем, что неверную связку нельзя собрать.
NANO_BANANA_SOLO = replace(NANO_BANANA, name="nano_banana_solo", images_identity=False)
SEEDREAM_SOLO = replace(SEEDREAM_EDIT, name="seedream_solo", images_identity=False)
HY_WU_SOLO = replace(HY_WU_EDIT, name="hy_wu_solo", images_identity=False)

FLUX_GENERAL = PayloadSchema(
    name="flux_general",
    identity_field="ip_adapters",
    identity_kind="list",
)

_PAYLOADS: dict[str, PayloadSchema] = {}


def register_payload(schema: PayloadSchema) -> None:
    _PAYLOADS[schema.name] = schema


def payloads() -> list[str]:
    return sorted(_PAYLOADS)


def payload(name: str) -> PayloadSchema:
    schema = _PAYLOADS.get((name or "").strip().lower())
    if schema is None:
        raise InvalidImageError(
            f"Неизвестная схема запроса «{name}»", {"payload": name, "available": payloads()}
        )
    return schema


register_payload(KONTEXT)
register_payload(KONTEXT_MULTI)
register_payload(FACE_SWAP)
register_payload(NANO_BANANA)
register_payload(SEEDREAM_EDIT)
register_payload(HY_WU_EDIT)
register_payload(NANO_BANANA_SOLO)
register_payload(SEEDREAM_SOLO)
register_payload(HY_WU_SOLO)
register_payload(FLUX_GENERAL)


@dataclass(frozen=True)
class MaskProfile:
    """
    Геометрия маски головы на ШАБЛОНЕ. Доли — от высоты лица персонажа.

    Маска накрывает лицо и причёску персонажа и расширяется настолько, чтобы
    модели было где связать новую голову с шеей и воротником. Без запаса
    получается голова, приставленная к чужой шее; с чрезмерным — модель
    переписывает плечи и одежду.

    :param dilate_ratio: расширение маски за контур головы
    :param feather_ratio: размытие краёв. Ради него всё и делается: жёсткий
        край маски даёт видимую границу генерации
    :param neck_ratio: насколько маска опускается ниже подбородка персонажа —
        там она захватывает шею, чтобы новая голова села на неё бесшовно
    """

    dilate_ratio: float = 0.12
    feather_ratio: float = 0.10
    neck_ratio: float = 0.35


@dataclass(frozen=True)
class StylePreset:
    """
    Стиль сцены: чем и как рисовать внутри маски.

    Отдельно от профиля, потому что меняется чаще и по другой причине: профиль
    подбирают под эндпоинт, стиль — под серию книг.
    """

    name: str
    prompt: str
    # Отрицания идут в промпт текстом: инпейнтинговые эндпоинты fal, которыми мы
    # пользуемся, negative_prompt не принимают.
    avoid: str = (
        "no collage, no pasted cut-out, no visible mask edge, no seam around the head, "
        "no second head, no duplicated face, no floating head, no mismatched skin tone, "
        # Отрицания масштаба: та же беда, что и в _IDENTITY_INSTRUCTION, но с
        # другой стороны — там сказано, как надо, здесь как не надо
        "no oversized head, no zoomed-in face, no cropped chin, no cropped jaw"
    )

    def text(self, extra: str = "") -> str:
        parts = [self.prompt, extra.strip(), self.avoid]
        return " ".join(part for part in parts if part)


IMPASTO = StylePreset(
    name="impasto",
    prompt=(
        "impasto oil painting style, visible thick brushstrokes, textured canvas, "
        "artistic lighting, painted artwork. "
        "Paint the child's head and hair into this illustration so that it belongs to "
        "the painting: same palette, same brush language, same light direction and "
        "colour temperature as the surrounding artwork."
    ),
)

PIXAR_REAL = StylePreset(
    name="pixar_real",
    prompt=(
        "RAW photo, cinematic photorealism, detailed skin texture, soft shadows, "
        "8k uhd, 85mm lens, natural lighting matching the scene. "
        "Render the child's head and hair as a real photograph of this child inside "
        "the scene: same light direction, same depth of field and same grain as the "
        "surrounding frame."
    ),
)

# Что требуется от генерации независимо от стиля: поза и мимика берутся со
# сцены, черты и тон кожи — с фотографии заказчика. Именно это разделение и
# отличает identity transfer от «нарисуй ребёнка».
_IDENTITY_INSTRUCTION = (
    "Replace the head inside the mask with the child from the reference photograph. "
    "Keep the pose, the head turn, the tilt and the facial expression of the original "
    "character in the scene; take the identity from the reference — face shape, eye "
    "shape and colour, eyebrows, nose, mouth, skin tone, hair colour and hair length. "
    # Масштаб — вторая половина того же требования, что и поза: с фотографии
    # берутся ЧЕРТЫ, а размер головы диктует сцена. Kontext переносит из
    # референса и композицию тоже, и без этой фразы голова выходит крупнее
    # маски, а челюсть срезается по её краю. Само отдаление референса делается
    # полем в pipelines/reference.py — здесь оно только проговорено словами.
    "Strictly preserve the exact size and proportions of the original character's "
    "head, do not enlarge the face. The reference photograph is a close-up: use it "
    "only for identity, never for framing or head size. "
    "Blend the new head into the existing neck, collar and shoulders so that no edge "
    "of the mask is visible anywhere."
)

# Инструкция безмасочного пути. Отличий от масочной три, и все существенные.
#
# Первое: маски у модели нет, поэтому «внутри маски» заменяется прямым адресом
# картинок — первая и вторая. Порядок ссылок в image_urls и порядок слов здесь
# обязаны совпадать, иначе модель перерисует фотографию по мотивам разворота.
#
# Второе: модель отдаёт КАДР ЦЕЛИКОМ, и всё, что не голова, требуется вернуть
# неизменным. Локально мы всё равно возьмём из ответа только голову, но чем
# меньше уехал остальной кадр, тем точнее сядет вклейка: сместившийся фон
# означает смещённую голову, а её вклеивать уже некуда.
#
# Третье: фотореализм требуется словами и подпирается отрицаниями. Kontext на
# иллюстрации охотно продолжает её материал — и отдаёт нарисованное лицо,
# которое формально «похоже», а на печати остаётся рисунком.
_MULTI_INSTRUCTION = (
    "Photorealistic identity transfer. You are given two images. "
    "The FIRST image is the scene: keep it exactly as it is — same composition, same "
    "framing, same crop, same background, same body, same clothing, same pose, and the "
    "same head position, head size and head turn. Do not move, rescale or reframe "
    "anything. "
    "The SECOND image is a close-up portrait of the child whose identity must be used. "
    "Change one thing only: give the character in the first image the face and the hair "
    "of the child from the second image — face shape, eye shape and eye colour, "
    "eyebrows, nose, mouth, skin tone, hair colour and hair length. "
    "Keep the facial expression, the gaze direction and the head angle of the character "
    "in the first image; the second image supplies identity only and must never dictate "
    "framing, head size, pose or expression. "
    "Match the light of the first image exactly: same light direction, same shadows, "
    "same colour temperature, same contrast, same depth of field and same grain. "
    "The result must look like a real photograph of a real child: natural skin texture "
    "with pores, individual hair strands, photographic micro-contrast. "
    "No painting, no oil texture, no brush strokes, no illustration, no cartoon, no 3D "
    "render, no plastic or airbrushed skin, no beautification, no change of age, no "
    "second head, no duplicated face."
)

# Инструкция шага причёски. Пять требований, и все выстраданы прогонами.
#
# Нулевое, и оно же про то, чего в запросе НЕТ. Картинка здесь ровно одна —
# шаблон. Фотография заказчика на этот шаг не уезжает, потому что универсальный
# редактор понимает портрет крупным планом не как «вот чья причёска», а как «вот
# что нарисовать»: оба живых редактора нарисовали лицо донора на затылке
# персонажа — в пустой области маски волос, то есть ровно там, где им разрешили
# рисовать. Причёска поэтому описывается СЛОВАМИ, и слова эти обязательны:
# без них модели неоткуда взять ни длину, ни цвет. Запрет на второе лицо
# продублирован отрицаниями — стоит он дёшево, а стоил дорого.
#
# Первое: правится не лицо. Оно охраняется прямым запретом — «не меняй, не
# подменяй, не омолаживай», — потому что следом идёт фейссвоп, и ему нужны
# нетронутые глаза, нос и рот: своим детектором он ищет их на уже поправленном
# шаблоне, а не найдя, молча вернёт кадр без замены.
#
# Второе: у причёски есть след. Длинные волосы лежат на плечах и закрывают уши,
# шею и одежду; короткая стрижка их открывает, и всё, что было под волосами,
# модель обязана дорисовать. Без этой фразы на месте старой гривы остаётся
# тёмный ореол — и он переживает любую вклейку, потому что лежит внутри маски.
#
# Третье, и это правка после первого живого прогона: **силуэт обязан
# уменьшиться**. Прежний текст просил сохранить «same head size» — фразой,
# написанной против перекадрирования, — и редактор прочитал её буквально:
# голова вместе с копной волос была объявлена неприкосновенной по размеру.
# Получился блондинистый «шлем» — цвет донора при объёме шаблона. Поэтому здесь
# разведены ЧЕРЕП с лицом (не меняются) и КОНТУР ПРИЧЁСКИ (обязан сесть по
# черепу), а старая причёска названа не формой для правки, а формой на удаление.
#
# Четвёртое: фотореализм требуется словами и подпирается отрицаниями. Волосы —
# самое уязвимое место: диффузия охотно отдаёт их слипшейся шапкой или
# нарисованной прядью, и на печати это читается как рисунок раньше, чем кожа.
# Короткая стрижка вдобавок описывается физически — миллиметры, просвечивающая
# кожа головы, резкий край, — иначе «короткие» для модели означают просто
# «покороче прежних».
#
# Пятое, и это правка после прогона с ореолом. Прежний текст велел «перестроить
# фон» и запрещал ореол ТЁМНЫЙ — то есть остаток самих волос. Пришёл светлый:
# на месте срезанной гривы модель положила размытое свечение поверх неба и гор.
# Так эти редакторы и заполняют большую освободившуюся область — мягким пятном
# в тон окружения, — и вклейку оно переживает, потому что лежит внутри маски.
# Поэтому фон здесь описан не как «перестроить», а как ПРОДОЛЖИТЬ: небо,
# горизонт и пейзаж обязаны пройти за головой ровно так же, как проходят слева
# и справа от неё, тем же цветом, той же яркостью, той же резкостью. Свечение
# любого знака названо отдельным отрицанием, потому что «no dark halo» модель
# честно исполнила и нарисовала светлый.
#
# Там же — про прядь на щеке. Она остаётся не только от маски: даже открыв ей
# область, стирать локон надо словами, иначе редактор считает его частью лица.
_HAIR_INSTRUCTION = (
    "Photorealistic hairstyle replacement in this single photograph. "
    "You are given one image and no reference image: the new hairstyle is described in "
    "words at the end of this instruction, and those words are the only source of its "
    "cut, its length, its colour and its texture. "
    "Change the hair of the person and nothing else. "
    "Remove the existing hair completely. Do not keep its volume, its bulk, its length or "
    "its outline: the old hairstyle is not a shape to adjust, it is a shape to delete. If "
    "the new hair is shorter, the head must become visibly smaller in silhouette: the hair "
    "has to follow the skull tightly, the ears, the temples and the neck have to come out "
    "from under it, and the background, the shoulders and the clothing must be rebuilt "
    "everywhere the old hair used to be. "
    "Continue the background straight through the area the old hair occupied: the sky, the "
    "horizon, the clouds, the landscape and everything else behind the head must carry on "
    "across that area exactly as they read immediately to the left and to the right of it — "
    "same colour, same brightness, same gradient, same texture, same sharpness, same "
    "grain — as if the long hair had never been there. That area must be indistinguishable "
    "from the rest of the background: not a patch, not a smudge, not a soft blob. "
    "Nothing glows around this head. There is no halo of any kind, light or dark, no bloom, "
    "no glare, no haze, no mist, no white or bright fringe, no soft blurred aura, no "
    "brightened or washed-out region around the head, no vignette, no outline, no rim light "
    "that is not already in the photograph. "
    "Remove every remaining lock of the old hair, including any strand lying on the cheek, "
    "the cheekbone, the temple, the jaw, the ear or the neck: under such a strand there is "
    "skin, and that skin must be drawn, matching the surrounding complexion exactly. "
    "Short hair means short: the scalp shows through it, the hairline and the sideburns "
    "have a sharp clipped edge, the crown carries no bulk, and the whole layer is "
    "millimetres thick, not centimetres. "
    "Never draw a face, a head, eyes, a nose or a mouth anywhere new. The back of the "
    "head is the back of the head: hair, scalp and skin, nothing else. There is exactly "
    "one person in this image and exactly one face, the one already there. "
    "That face is not yours to touch: same eyes, same eyebrows, same nose, same mouth, "
    "same skin, same expression, same gaze, same position and angle of the head. Do not "
    "swap the face, do not change the age, do not beautify anything. Keep the skull and "
    "the face at exactly the same size — what changes is the hair around them, and only it. "
    "Keep the composition, the framing, the crop, the pose, the body, the clothing and the "
    "background unchanged; do not move, rescale or reframe anything. "
    "The hair must look photographed, not drawn: individual strands and stubble, visible "
    "scalp at the roots, natural specular highlights, photographic micro-contrast and the "
    "same grain as the rest of the frame. "
    "Match the light of the photograph: same direction, same shadows, same colour "
    "temperature, same contrast, same depth of field. "
    "No painting, no oil texture, no brush strokes, no illustration, no cartoon, no 3D "
    "render, no plastic or airbrushed look, no smooth blurry mass of hair, no rounded cap, "
    "no bob, no wig, no helmet hair, no sticker edge, no second face, no face on the back "
    "of the head, no extra head, no portrait inserted into the scene. "
    "No glow, no halo, no bloom, no light leak, no lens flare, no blur and no change of "
    "exposure, brightness, contrast or colour temperature anywhere in the frame. "
    "The hairstyle to give this person, described in words:"
)

# Приписка к промпту, когда старая причёска стёрта до отправки (см.
# pipelines/erase.py). Существует потому, что пятно на месте волос — это тоже
# вход, и без объяснения редактор попытается его ОБЪЯСНИТЬ: размытую область он
# читает как расфокус, туман или дымку и честно дорисовывает их по всему кадру.
# Сказанное здесь превращает пятно из содержимого в разметку задания: не «что
# нарисовано», а «где рисовать».
#
# Идёт первой, а не последней: инструкция кончается словами «the hairstyle to
# give this person, described in words:», после которых стоит только описание
# причёски.
_ERASE_NOTE = (
    "Part of this photograph has been deliberately wiped out before it was given to you: "
    "the area where the old hair used to be is now a flat, featureless smudge. That smudge "
    "is not content and it is not part of the scene — it is a blank marking the area you "
    "have to rebuild from scratch. Everything inside it must be redrawn: the new hair, and, "
    "wherever the new hair does not reach, the background, the skin, the ear, the temple, "
    "the cheek, the jaw, the neck, the collar and the clothing that belong there, "
    "reconstructed from what surrounds the blank and continuous with it. "
    "Never reproduce the smudge, never leave any part of it, and never interpret it as "
    "hair, shadow, shading, fog, haze, mist, glow, blur or depth of field: nothing in the "
    "output is blurred, and the whole frame is as sharp and as detailed as the untouched "
    "part of the photograph. "
)


@dataclass(frozen=True)
class HairStage:
    """
    Первый шаг двухшаговой стратегии: чем и как правится причёска.

    Отдельным набором, а не полями профиля, по той же причине, по какой отделены
    друг от друга стратегии: шагов здесь два, у каждого свой эндпоинт со своей
    схемой, и перепутать их значит послать фейссвопу промпт, а редактору — две
    именованные ссылки. Профиль остаётся про ШАГ ЗАМЕНЫ ЛИЦА (endpoint, payload),
    а всё, что относится к волосам, лежит здесь.

    :param endpoint: мультикартиночный редактор. Все три живых проверены и
        взаимозаменяемы: nano-banana точнее держит остальной кадр, seedream
        единственный отдаёт 4K, hy-wu умеет думать перед правкой
    :param payload: схема запроса — обязательно из «одиночных» (`*_solo`).
        Фотография заказчика на этот шаг не уезжает, см. `images_identity`
    :param description: словами о причёске заказчика. **Единственный источник**
        того, что рисовать: референса на этом шаге нет. Пусто — заказ
        отклоняется до сети, потому что рисовать нечего
    :param dilate_ratio: расширение маски за контур волос, доля высоты лица
    :param feather_ratio: растушёвка краёв маски
    :param protect_ratio: поле защиты вокруг лица. Единственное число, которое
        стоит между диффузией и глазами персонажа
    :param forehead_ratio: докуда маска спускается по лбу, в долях высоты лица
        над бровями. 0 — до самых бровей, и тогда шва на лбу нет вовсе: лоб
        целиком рисует редактор, а стык уходит под растушёвку фейссвопа.
        0.5 и больше — лоб не трогается (умолчание). Крутить это, а не protect,
        когда над бровями видно рубец: protect меняет ширину поля, а не форму
    :param core_ratio: во сколько раз сжать ядро лица поперёк его оси. 1.0 — как
        построено (умолчание); 0.75 сильнее открывает виски и скулы, где
        застревают пряди старой причёски. Углы глаз остаются под защитой:
        дилатация ядра круговая и возвращает накрытие наружу
    :param guard_ratio: какую долю полуширины ЗАЩИТЫ ЛИЦА оставить. 1.0 — всю
        (умолчание). Единственная ручка против пряди, которую разметка отнесла
        к классу лица: такая прядь не входит в маску (она не волосы) и
        вычитается вместе с защитой, а защита отнимается ПОСЛЕ дилатации —
        расширять маску за ней бесполезно. Ядро лица возвращается поверх среза
    :param cheek_ratio: докуда спускается полоса вдоль щеки, доли высоты лица
        ниже подбородка. 0 — полосы нет (умолчание) и область строится ровно по
        разметке.

        Единственное, чем берётся локон, которого сегментатор не видит ВООБЩЕ —
        ни волосами, ни лицом. Такая прядь недостижима ни одним числом выше по
        построению: `dilate` расширяет источник, которого нет, а `guard` и
        `core` только отпускают защиту, области не добавляя. Полоса задаёт
        область геометрией, не спрашивая разметку, см. `hair_mask._cheek_bands`.
        Платим щеками и шеей — они уходят под диффузию; глаза, нос и рот не
        уходят никогда, их держит ядро лица
    :param erase_ratio: насколько размыть заливку, которой стирается старая
        причёска ДО отправки редактору, доля высоты лица. 0 — стирания нет
        (умолчание).

        **Провалившийся путь, оставленный переключателем.** Гипотеза была в том,
        что редактор цепляется за структуру старых волос внутри открытой маски;
        живой прогон её опроверг и заодно доказал обратное. Стирание залило
        мылом всю область — и прядь на щеке осталась резкой, то есть лежала вне
        маски целиком; именно так и выяснилось, что разметка её не видит.
        Заодно выяснилась цена: без силуэта старой копны редактор рисует ёжик
        прямо по черепу (голова выходит лысой), а границы заливки возвращают тот
        самый светлый ореол вокруг головы. Включать это можно, но незачем: цель
        достигается `cheek_ratio`, и без обеих этих бед
    :param crop_ratio: поле вокруг маски волос, доля высоты лица. Редактору
        уезжает не весь разворот, а окно вокруг головы — иначе причёска
        занимает проценты кадра, и на них модель отдаёт мыльную шапку вместо
        стрижки. 0 выключает окно и отправляет разворот целиком
    :param min_changed: насколько правка обязана изменить кадр ВНУТРИ маски,
        уровни 0..255. Ниже — редактор промолчал: причёска осталась прежней
    """

    endpoint: str = _NANO_BANANA_EDIT
    payload: PayloadSchema = NANO_BANANA_SOLO
    instruction: str = _HAIR_INSTRUCTION
    description: str = ""
    dilate_ratio: float = 0.10
    feather_ratio: float = 0.07
    protect_ratio: float = 0.05
    forehead_ratio: float = 0.5
    core_ratio: float = 1.0
    guard_ratio: float = 1.0
    # Полоса вдоль щёк. 0.35 высоты лица ниже подбородка — это воротник: длинная
    # прядь спускается по щеке и кончается на нём, а глубже начинается грудь,
    # где волос не бывает и правке делать нечего
    cheek_ratio: float = 0.35
    # Стирание выключено: прогон показал, что оно не решает задачу (прядь лежит
    # ВНЕ маски, и заливка до неё не достаёт) и вдобавок стоит силуэта причёски
    # и светлого ореола. Оставлено переключателем, как kontext_multi, —
    # воспроизвести провал одной переменной дешевле, чем спорить о нём
    erase_ratio: float = 0.0
    crop_ratio: float = 1.0
    min_changed: float = 3.0

    def prompt(self, description: str = "") -> str:
        """
        Текст запроса шага причёски.

        :param description: описание причёски из заказа. Складывается с
            профильным: одно приходит от оператора на весь тираж, другое —
            от менеджера на конкретный заказ, и оба уместны
        """
        # Отрицания живут в самой инструкции, а не в стиле: пресеты стилей
        # запрещают коллаж и срезанную челюсть, а здесь опасности другие —
        # парик, шлем из волос и второе лицо на затылке.
        #
        # Описание идёт последним, потому что инструкция им заканчивается: «the
        # hairstyle to give this person, described in words:». Перед словами о
        # причёске стоять нечему — они здесь единственные данные.
        #
        # Приписка о стирании — только когда стирание включено. Промпт обязан
        # описывать тот кадр, который уехал: пообещать редактору пятно и
        # прислать нетронутую фотографию значит попросить его стереть волосы,
        # которых он не найдёт.
        parts = [
            _ERASE_NOTE if self.erase_ratio > 0 else "",
            self.instruction,
            self.description.strip(),
            description.strip(),
        ]
        return " ".join(part for part in parts if part)

    def describes_hair(self, description: str = "") -> bool:
        """Есть ли чем описать причёску: профилем или заказом."""
        return bool(self.description.strip() or description.strip())

    def report(self) -> dict:
        return {
            "hair_endpoint": self.endpoint,
            "hair_payload": self.payload.name,
            "hair_dilate_ratio": self.dilate_ratio,
            "hair_feather_ratio": self.feather_ratio,
            "hair_protect_ratio": self.protect_ratio,
            "hair_forehead_ratio": self.forehead_ratio,
            "hair_core_ratio": self.core_ratio,
            "hair_guard_ratio": self.guard_ratio,
            "hair_cheek_ratio": self.cheek_ratio,
            "hair_erase_ratio": self.erase_ratio,
            "hair_crop_ratio": self.crop_ratio,
            "hair_min_changed": self.min_changed,
            "hair_sends": self.payload.sent_keys(),
        }


_STYLES: dict[str, StylePreset] = {}


def register_style(preset: StylePreset) -> None:
    _STYLES[preset.name] = preset


def styles() -> list[str]:
    return sorted(_STYLES)


def style(name: str) -> StylePreset:
    preset = _STYLES.get((name or "").strip().lower())
    if preset is None:
        raise InvalidImageError(
            f"Неизвестный стиль «{name}»", {"style": name, "available": styles()}
        )
    return preset


register_style(IMPASTO)
register_style(PIXAR_REAL)


@dataclass(frozen=True)
class StrategyDefaults:
    """
    Чем стратегия разговаривает с fal: эндпоинт, схема запроса, текст инструкции.

    Эти вещи меняются только вместе. Фейссвоп — это одновременно другой
    эндпоинт, другой набор ключей, отсутствие промпта и отсутствие всей
    локальной геометрии; безмасочная диффузия — другой эндпоинт, другой набор
    ключей и другой промпт («первая картинка» вместо «внутри маски»). Любая их
    комбинация из разных стратегий даёт либо 422, либо молча испорченный кадр.
    Поэтому `ML_REFINE_STRATEGY` тянет за собой остальное, а не требует задать
    его руками.

    Точечное переопределение из окружения по-прежнему сильнее: ML_REFINE_ENDPOINT
    и ML_REFINE_PAYLOAD перебивают умолчания стратегии — иначе новый эндпоинт
    нельзя было бы попробовать без релиза.

    :param needs_mask: нужна ли стратегии локальная геометрия
    :param reference_pad_max: предел поля вокруг фотографии; 1.0 выключает
        подготовку референса целиком. None — оставить как в профиле
    """

    name: str
    endpoint: str
    payload: PayloadSchema
    instruction: str
    needs_mask: bool = True
    reference_pad_max: float | None = None


_STRATEGY_DEFAULTS: dict[str, StrategyDefaults] = {}


def register_strategy_defaults(defaults: StrategyDefaults) -> None:
    _STRATEGY_DEFAULTS[defaults.name] = defaults


def strategy_defaults(name: str) -> StrategyDefaults | None:
    """
    Умолчания стратегии либо None.

    None — не ошибка: реестр самих стратегий живёт в `refine/base.py`, и он же
    решает, существует ли такая. Здесь только про то, чем она говорит с fal.
    """
    return _STRATEGY_DEFAULTS.get((name or "").strip().lower())


# Предел роста холста референса на диффузионных путях. Держится одним числом на
# весь модуль: его же берут умолчания стратегий, и разъехаться им нельзя —
# переключение туда-обратно вернуло бы фотографию без поля. Откуда 5.5, см.
# комментарий у поля reference_pad_max.
_REFERENCE_PAD_MAX = 5.5


@dataclass(frozen=True)
class RefineProfile:
    """
    Полный набор гиперпараметров генерации.

    :param strategy: имя стратегии в реестре `refine`. Их четыре:
        face_swap — специализированный фейссвоп (рабочая);
        hair_swap — правка причёски редактором, затем тот же фейссвоп;
        kontext_multi — диффузия по кадру целиком плюс локальная вклейка;
        identity_inpaint — генерация внутри маски.
    :param instruction: что просят у модели словами. Часть стратегии, а не
        стиля: масочный путь адресует область («inside the mask»), безмасочный —
        картинки по порядку («the FIRST image»). У фейссвопа пусто: он текст не
        читает вовсе.
    :param needs_mask: нужна ли локальная геометрия — маска головы и подгонка
        референса. Фейссвопу не нужна ни та, ни другая: область он находит сам,
        а поле вокруг фотографии мешает его детектору
    :param payload: схема запроса выбранного эндпоинта
    :param identity_scale: насколько сильно тянуть черты с фотографии. Выше —
        ближе к донору и дальше от стиля сцены. Работает только со схемами, где
        у референса есть свой вес (identity_kind="list").
    :param safe_strength: граница, ниже которой в лог уходит предупреждение:
        под маской лежит голова чужого персонажа, и слабая генерация оставляет
        от неё черты.
    """

    name: str
    strategy: str
    endpoint: str
    style: str
    strength: float
    guidance_scale: float
    steps: int
    safe_strength: float
    payload: PayloadSchema = KONTEXT
    instruction: str = _IDENTITY_INSTRUCTION
    needs_mask: bool = True
    identity_scale: float = 0.9
    mask: MaskProfile = field(default_factory=MaskProfile)
    # Шаг причёски. Есть у всех профилей и работает у одного — двухшагового:
    # так `ML_REFINE_STRATEGY=hair_swap` остаётся переключателем, а не поводом
    # собирать профиль заново. Endpoint и payload самого профиля описывают шаг
    # ЗАМЕНЫ ЛИЦА и на причёску не влияют.
    hair: HairStage = field(default_factory=HairStage)
    # Подготовка референса, см. pipelines/reference.py. Ширина поля вокруг
    # фотографии не задаётся: она вычисляется из двух измеренных высот лица.
    # Здесь только границы этого вычисления.
    #
    # pad_max — предел роста ДЛИННОЙ стороны холста. Эндпоинт ужимает референс
    #   до рабочего кадра, и расходуется этот кадр на длинную сторону: чем
    #   шире холст, тем меньше пикселей достаётся лицу, а из них берётся
    #   личность. 1.0 выключает подготовку целиком.
    #
    #   Откуда 5.5. Пока референс равнялся на шаблон только масштабом, нашей
    #   паре хватало роста 2.86. С выравниванием пропорции к нему добавилось
    #   поле по ширине: квадратный донор 1275 px разворачивается в 1.79:1, и
    #   длинная сторона растёт в 5.12 раза. Предел обязан это пропускать, иначе
    #   выравнивание не состоится — а частичное совпадение пропорции означает
    #   частичное искажение лица.
    #
    #   Чем платим, видно в мете: face_px_at_work падает со 179 до 100 px при
    #   пессимистичном рабочем кадре в 1024. Размен сознательный — расплющенное
    #   лицо портит результат всегда, а мелкий референс лишь иногда. Если
    #   сходство поедет, крутить надо именно это число.
    reference_pad_max: float = _REFERENCE_PAD_MAX
    # pad_ratio — слепое поле с каждой стороны в долях стороны исходника. Идёт
    #   в дело, только когда лица на фотографии не нашлось и вычислять нечего.
    reference_pad_ratio: float = 0.3
    # Дополнение к промпту стиля: сюда кладётся описание сцены, если оно есть.
    # Пусто — работает один стиль.
    scene: str = ""

    def prompt(self, expression: str = "") -> str:
        """
        Полный текст запроса: что сделать, с каким выражением, в каком стиле.

        :param expression: указание мимики. Пусто — выражение берётся со сцены,
            и это умолчание: поза и эмоция персонажа для нас исходные данные
        """
        parts = [self.instruction, expression.strip(), style(self.style).text(self.scene)]
        return " ".join(part for part in parts if part)

    def validate(self) -> RefineProfile:
        """
        Проверяет профиль до сети.

        Отдельным шагом, а не в __post_init__: профиль собирают и из пресетов, и
        из окружения, и промежуточные состояния законны. Смысл проверки в том,
        чтобы неверная связка стоила 500 с внятным текстом, а не отказа fal
        после загрузок в CDN.
        """
        if not self.endpoint:
            raise InvalidImageError(
                "Профилю не задан эндпоинт — укажите его в ML_REFINE_ENDPOINT",
                {"profile": self.name},
            )
        if not 0.0 <= self.strength <= 1.0:
            raise InvalidImageError(
                "strength должен лежать в диапазоне 0..1",
                {"profile": self.name, "strength": self.strength},
            )
        if self.steps < 1:
            raise InvalidImageError(
                "Число шагов инференса должно быть положительным",
                {"profile": self.name, "steps": self.steps},
            )
        if not 0.0 <= self.identity_scale <= 2.0:
            raise InvalidImageError(
                "Вес идентичности должен лежать в диапазоне 0..2",
                {"profile": self.name, "identity_scale": self.identity_scale},
            )
        if self.reference_pad_max < 1.0 or self.reference_pad_ratio < 0:
            raise InvalidImageError(
                "Предел поля референса не может быть меньше единицы, а доля — отрицательной",
                {
                    "profile": self.name,
                    "reference_pad_max": self.reference_pad_max,
                    "reference_pad_ratio": self.reference_pad_ratio,
                },
            )
        if self.mask.dilate_ratio < 0 or self.mask.feather_ratio < 0 or self.mask.neck_ratio < 0:
            raise InvalidImageError(
                "Доли маски головы не могут быть отрицательными",
                {"profile": self.name},
            )
        if (
            self.hair.dilate_ratio < 0
            or self.hair.feather_ratio < 0
            or self.hair.protect_ratio < 0
            or self.hair.forehead_ratio < 0
            or self.hair.core_ratio < 0
            or self.hair.guard_ratio < 0
            or self.hair.cheek_ratio < 0
            or self.hair.erase_ratio < 0
            or self.hair.crop_ratio < 0
            or self.hair.min_changed < 0
        ):
            raise InvalidImageError(
                "Доли шага причёски не могут быть отрицательными",
                {"profile": self.name, **self.hair.report()},
            )
        if not self.hair.endpoint:
            raise InvalidImageError(
                "Шагу причёски не задан эндпоинт — укажите его в ML_HAIR_ENDPOINT",
                {"profile": self.name},
            )
        if not self.hair.payload.images_field:
            # Редактор принимает картинки массивом. Схема с двумя именованными
            # полями отправит ему шаблон в image_url — 422 на весь запрос после
            # загрузки в CDN
            raise InvalidImageError(
                "Схема шага причёски не передаёт картинки массивом",
                {"profile": self.name, "payload": self.hair.payload.name},
            )
        if self.hair.payload.images_identity:
            # Единственная проверка здесь, у которой есть цена в браке. Схема,
            # кладущая в массив фотографию заказчика, не даст ни ошибки, ни 422:
            # редактор ответит успехом, нарисовав лицо донора на затылке
            # персонажа. Схемы шага причёски — только «одиночные», *_solo
            raise InvalidImageError(
                "Схема шага причёски отправит редактору фотографию заказчика: "
                "он нарисует по ней второе лицо. Нужна одиночная схема (*_solo)",
                {
                    "profile": self.name,
                    "payload": self.hair.payload.name,
                    "solo": [name for name in payloads() if name.endswith("_solo")],
                },
            )

        style(self.style)  # неизвестный стиль — тоже ошибка конфигурации
        payload(self.payload.name)  # и незарегистрированная схема запроса
        payload(self.hair.payload.name)  # схема шага причёски проверяется так же
        return self

    def report(self) -> dict:
        """Плоская сводка для логов и /health/ready."""
        return {
            "profile": self.name,
            "strategy": self.strategy,
            "endpoint": self.endpoint,
            "payload": self.payload.name,
            "style": self.style,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            # Чем уедет личность: отдельным полем или вторым элементом массива
            "identity_field": self.payload.identity_field or self.payload.images_field,
            "identity_scale": self.identity_scale,
            "needs_mask": self.needs_mask,
            "sends": self.payload.sent_keys(),
            # Шаг причёски — только у двухшаговой стратегии. У остальных эти
            # числа существуют, но не работают, и в отчёте им делать нечего:
            # мета читается как «что произошло с заказом»
            **(self.hair.report() if self.strategy == "hair_swap" else {}),
        }


_PRESETS: dict[str, RefineProfile] = {}


def register(profile: RefineProfile) -> None:
    """Добавляет профиль в реестр. Повторное имя — замена."""
    _PRESETS[profile.name] = profile


def available() -> list[str]:
    return sorted(_PRESETS)


def get(name: str) -> RefineProfile:
    profile = _PRESETS.get((name or "").strip().lower())
    if profile is None:
        raise InvalidImageError(
            f"Неизвестный профиль обработки «{name}»",
            {"profile": name, "available": available()},
        )
    return profile


def from_settings() -> RefineProfile:
    """
    Активный профиль: пресет из ML_REFINE_PROFILE плюс переопределения окружения.

    Пустое значение переопределения означает «взять из пресета» — именно поэтому
    они объявлены как None, а не как числа с дефолтами: иначе невозможно
    отличить «оператор поставил 0.2» от «оператор не трогал».
    """
    profile = get(settings.refine_profile)

    changes: dict = {}
    if settings.refine_endpoint:
        changes["endpoint"] = settings.refine_endpoint
    if settings.refine_payload:
        changes["payload"] = payload(settings.refine_payload)
    if settings.refine_style:
        changes["style"] = settings.refine_style
    if settings.refine_scene:
        changes["scene"] = settings.refine_scene
    if settings.refine_strategy:
        changes["strategy"] = settings.refine_strategy
        # Смена стратегии тянет за собой эндпоинт, схему и текст инструкции: по
        # отдельности они не имеют смысла, см. StrategyDefaults. setdefault, а
        # не присваивание, — заданное в окружении явно остаётся сильнее.
        defaults = strategy_defaults(settings.refine_strategy)
        if defaults is not None and settings.refine_strategy != profile.strategy:
            changes.setdefault("endpoint", defaults.endpoint)
            changes.setdefault("payload", defaults.payload)
            changes.setdefault("instruction", defaults.instruction)
            changes.setdefault("needs_mask", defaults.needs_mask)
            if defaults.reference_pad_max is not None:
                changes.setdefault("reference_pad_max", defaults.reference_pad_max)
    if settings.refine_strength is not None:
        changes["strength"] = settings.refine_strength
    if settings.refine_identity_scale is not None:
        changes["identity_scale"] = settings.refine_identity_scale
    if settings.reference_pad_max is not None:
        changes["reference_pad_max"] = settings.reference_pad_max
    if settings.reference_pad_ratio is not None:
        changes["reference_pad_ratio"] = settings.reference_pad_ratio
    if settings.refine_guidance_scale is not None:
        changes["guidance_scale"] = settings.refine_guidance_scale
    if settings.refine_steps is not None:
        changes["steps"] = settings.refine_steps

    # Шаг причёски правится своим набором переменных: он про другой эндпоинт с
    # другой схемой, и ML_REFINE_ENDPOINT относится не к нему, а к замене лица
    hair_changes: dict = {}
    if settings.hair_endpoint:
        hair_changes["endpoint"] = settings.hair_endpoint
    if settings.hair_payload:
        hair_changes["payload"] = payload(settings.hair_payload)
    if settings.hair_description:
        hair_changes["description"] = settings.hair_description
    for key, value in (
        ("dilate_ratio", settings.hair_dilate_ratio),
        ("feather_ratio", settings.hair_feather_ratio),
        ("protect_ratio", settings.hair_protect_ratio),
        ("forehead_ratio", settings.hair_forehead_ratio),
        ("core_ratio", settings.hair_core_ratio),
        ("guard_ratio", settings.hair_guard_ratio),
        ("cheek_ratio", settings.hair_cheek_ratio),
        ("erase_ratio", settings.hair_erase_ratio),
        ("crop_ratio", settings.hair_crop_ratio),
        ("min_changed", settings.hair_min_changed),
    ):
        if value is not None:
            hair_changes[key] = value
    if hair_changes:
        changes["hair"] = replace(profile.hair, **hair_changes)

    # Имя поля идентичности правится в схеме, а не в профиле: профиль про то,
    # что послать, схема — про то, как это назвать
    if settings.refine_identity_field:
        base = changes.get("payload", profile.payload)
        changes["payload"] = replace(base, identity_field=settings.refine_identity_field)

    mask_changes = {
        key: value
        for key, value in (
            ("dilate_ratio", settings.mask_dilate_ratio),
            ("feather_ratio", settings.mask_feather_ratio),
            ("neck_ratio", settings.mask_neck_ratio),
        )
        if value is not None
    }
    if mask_changes:
        changes["mask"] = replace(profile.mask, **mask_changes)

    return replace(profile, **changes).validate() if changes else profile.validate()


register_strategy_defaults(
    StrategyDefaults(
        name="face_swap",
        endpoint=_FAL_FACE_SWAP,
        payload=FACE_SWAP,
        # Текста эндпоинт не читает. Пустая строка здесь — не забывчивость, а
        # утверждение: у этой стратегии промпта нет и быть не может
        instruction="",
        # Ни маски, ни поля вокруг фотографии. Область эндпоинт находит сам, а
        # серая кайма референса мешает его детектору искать лицо донора
        needs_mask=False,
        reference_pad_max=1.0,
    )
)
# Двухшаговая стратегия говорит с fal дважды, и здесь описан ВТОРОЙ вызов —
# тот самый фейссвоп, слово в слово как у одношаговой. Так и задумано: шаг
# замены лица обязан остаться неотличимым от рабочего пути, иначе двухшаговая
# схема чинила бы причёску ценой того, что уже работает. Первый вызов —
# редактор причёски — описан в HairStage и переключается своими переменными.
#
# Локальная работа при этом всё-таки появляется: маску волос строит сама
# стратегия, из шаблона. В `needs_mask` это не отражается намеренно — то поле
# спрашивает, нужна ли ПАЙПЛАЙНУ маска головы, а она здесь не нужна: голова
# целиком не перерисовывается ни на одном из двух шагов.
register_strategy_defaults(
    StrategyDefaults(
        name="hair_swap",
        endpoint=_FAL_FACE_SWAP,
        payload=FACE_SWAP,
        instruction="",
        needs_mask=False,
        # Поле вокруг фотографии мешает обоим шагам: детектору фейссвопа — искать
        # лицо, редактору причёски — разглядеть волосы
        reference_pad_max=1.0,
    )
)
register_strategy_defaults(
    StrategyDefaults(
        name="kontext_multi",
        endpoint=_KONTEXT_MAX_MULTI,
        payload=KONTEXT_MULTI,
        instruction=_MULTI_INSTRUCTION,
        # Возврат на диффузию обязан вернуть и локальную работу. Обе величины
        # заданы явно, а не оставлены «как в профиле»: профиль по умолчанию
        # теперь фейссвопный, и молчание здесь означало бы пустую маску и
        # фотографию без поля
        needs_mask=True,
        reference_pad_max=_REFERENCE_PAD_MAX,
    )
)
register_strategy_defaults(
    StrategyDefaults(
        name="identity_inpaint",
        endpoint=_KONTEXT_INPAINT,
        payload=KONTEXT,
        instruction=_IDENTITY_INSTRUCTION,
        needs_mask=True,
        reference_pad_max=_REFERENCE_PAD_MAX,
    )
)


register(
    RefineProfile(
        name="pixar_real",
        strategy="face_swap",
        endpoint=_FAL_FACE_SWAP,
        payload=FACE_SWAP,
        # Фотореализм здесь не выпрашивается словами: переносится настоящее
        # лицо, и просить его «быть фотографией» некого. Отсюда и пустая
        # инструкция — эндпоинт текст не принимает вовсе
        instruction="",
        style="pixar_real",
        # Локальной геометрии нет: область эндпоинт находит сам, а поле вокруг
        # фотографии мешает его детектору (pad_max=1.0 выключает подготовку)
        needs_mask=False,
        reference_pad_max=1.0,
        # Ни одно из этих чисел в фейссвоп не уезжает — все они про диффузию. По
        # ним работают kontext_multi и identity_inpaint, если переключиться на
        # них одной переменной окружения, и профиль обязан оставаться
        # собираемым при любом переключении
        strength=0.92,
        guidance_scale=3.5,
        steps=40,
        safe_strength=0.85,
        identity_scale=0.9,
    )
)

# Тот же фотореализм, но с причёской донора. Отличие от pixar_real ровно одно —
# имя стратегии, и это не экономия записи, а утверждение: шаг замены лица здесь
# тот же самый, с тем же эндпоинтом и той же схемой. Двухшаговая схема ДОБАВЛЯЕТ
# проход перед ним и не трогает его самого.
#
# Почему отдельным пресетом, а не заменой умолчания. Проход причёски — это
# второй вызов, второй счёт и вторая возможность испортить кадр: редактор правит
# кадр целиком, и от его правки в шаблоне остаётся ровно то, что попало под
# маску волос. Пока это не проверено на тираже, рабочим путём остаётся
# pixar_real, а переключение стоит одной переменной: ML_REFINE_PROFILE=pixar_hair.
register(
    replace(
        get("pixar_real"),
        name="pixar_hair",
        strategy="hair_swap",
    )
)

# Живописной серии фейссвоп не подходит вдвойне. Во-первых, он переносит
# фотографические пиксели — ровно то, чем провалилась аппликация: на живописи
# они остаются фотографическими. Во-вторых, лицо на картине его детектор просто
# не найдёт. Поэтому здесь остаётся диффузия, и притом масочная: безмасочная
# инструкция требует фотографии и прямым текстом запрещает мазок, а стиль
# impasto требует обратного — в одном промпте это спор двух половин.
register(
    replace(
        get("pixar_real"),
        name="impasto",
        style="impasto",
        strategy="identity_inpaint",
        endpoint=_KONTEXT_INPAINT,
        payload=KONTEXT,
        instruction=_IDENTITY_INSTRUCTION,
        needs_mask=True,
        reference_pad_max=_REFERENCE_PAD_MAX,
    )
)
