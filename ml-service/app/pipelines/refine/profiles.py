"""
Все гиперпараметры второго шага в одном месте.

Здесь лежат числа, а не логика: сила инпейнтинга, пороги Canny, веса ControlNet,
доли маски, идентификаторы эндпоинтов и промпт. Раньше они были размазаны по
config.py, fal_api.py и mask_generator.py, и подобрать связку означало править
три файла и помнить, какое значение с каким сочетается.

Профиль — замороженный dataclass, то есть просто набор чисел с именем. Из него
нельзя получить картинку, он ничего не вызывает и ни от чего не зависит; его
можно распечатать в лог, сравнить с другим и положить в /health/ready целиком.
Стратегия (`refine/base.py`) получает профиль аргументом и решает, что с ним
делать, — поэтому «переключить подход» означает выбрать другое имя, а не
переписать вызов.

Четыре пресета сейчас:

  seam                — консервативный: strength 0.20, маска-плато только по
                        стыку. Режим, на котором пайплайн работал до сих пор;
                        страховка и база сравнения.
  blend               — **по умолчанию**: strength 0.26, градиентная маска.
                        Зона шире и мягче — спад накрывает шею и контур волос
                        целиком, — но сила остаётся в безопасном диапазоне
                        0.25-0.28. Выше ~0.28 контур причёски плывёт, и без
                        карт ControlNet удержать его нечем.
  stylise             — strength 0.5, та же градиентная маска. Настоящая
                        стилизация: мазок по всей аппликации и тени. **Без карт
                        рискует контуром причёски** — брать только вместе со
                        stylise_controlnet или после проверки на своей серии.
  stylise_controlnet  — то же плюс карты Canny и Depth. **Требует эндпоинта,
                        принимающего ControlNet**: kontext-inpaint его не
                        принимает, а лишний ключ fal не игнорирует, а
                        заворачивает весь запрос.

Про выбор по умолчанию. Рабочий эндпоинт — kontext-inpaint с нашей LoRA, и
стиль обложек держится на ней; уходить с него ради ControlNet значит потерять
этот стиль. Поэтому дефолт собран из того, что работает на нём: градиентная
маска (шея и контур волос сводятся мягко) плюс низкий strength (геометрия
причёски не плывёт). Тени на такой силе даёт промпт и LAB-коррекция кожи из
collage.py, а не карта глубины.

Свой профиль добавляется без правки этого файла:

    profiles.register(replace(profiles.get("stylise"), name="soft", strength=0.4))

Переопределения из окружения (ML_REFINE_*) накладываются поверх выбранного
пресета в `from_settings` — ими подбирают значения на живом сервисе, не
выкладывая релиз.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.config import settings
from app.core.errors import InvalidImageError

# Идентификатор эндпоинта инпейнтинга по маске. Специализированные лицевые
# модели fal здесь не подходят: easel-ai/advanced-face-swap ищет лицо своим
# детектором и на рисованной обложке его не находит, а flux-pulid и
# ip-adapter-face-id — чистый text-to-image, без mask_url и базового кадра.
_INPAINT_ENDPOINT = "fal-ai/flux-kontext-lora/inpaint"

# Эндпоинты, про которые известно, что ControlNet они не принимают. Лишний ключ
# в аргументах fal не игнорирует, а заворачивает весь запрос, поэтому такая
# связка ловится до сети — см. RefineProfile.validate.
_NO_CONTROL_ENDPOINTS = frozenset({_INPAINT_ENDPOINT})


@dataclass(frozen=True)
class ControlSpec:
    """
    Одна карта управления для ControlNet.

    :param kind: имя карты для эндпоинта (canny, depth)
    :param source: чем карта строится. `canny` — локально по коллажу через
        cv2.Canny; `image` — картой служит сам коллаж, карту глубины считает
        эндпоинт. Локального инференса глубины у сервиса нет (MiDaS потянул бы
        torch, а от локальных весов уходили осознанно), поэтому depth идёт
        вторым способом.
    :param weight: вес карты. Больше — жёстче держится геометрия оригинала,
        меньше — свободнее мазок.
    :param start: доля шагов денойза, с которой карта включается
    :param end: доля шагов, на которой отключается. Тени берутся с depth в
        начале денойза, когда решается крупная форма; к концу карта только
        мешает класть мазок.
    :param low: нижний порог Canny (только для source="canny")
    :param high: верхний порог Canny
    """

    kind: str
    source: str = "canny"
    weight: float = 0.6
    start: float = 0.0
    end: float = 0.8
    low: int = 100
    high: int = 200


@dataclass(frozen=True)
class MaskProfile:
    """
    Геометрия зон инпейнтинга, доли высоты лица на шаблоне.

    Зона 2 (стыки) — edge, neck, gradient, feather. Зона 3 (дыра в фоне) —
    hole_*. Зона 1 (лицо) — guard: она не открывается, а вычитается из обеих.

    :param edge_ratio: насколько кольцо стыка заходит ВНУТРЬ контура волос
    :param edge_outer_ratio: насколько оно уходит НАРУЖУ. Кольцо несимметрично:
        внутри волосы заказчика, снаружи заливка на месте чужой причёски, и
        дотянуться наружу надо до зоны фона
    :param neck_ratio: толщина полосы на стыке шеи с телом персонажа
    :param guard_ratio: растушёвка защиты лица — она вычитается из обеих масок
    :param feather_ratio: спад по краям зоны стыков
    :param gradient_ratio: ширина градиента от стыка наружу. Ноль — маска-плато
        со спадом по краю. Больше нуля — маска становится конусом: 255 ровно на
        стыке и линейный спад на эту долю в обе стороны. На strength 0.2 разницы
        почти нет, на 0.5 плато означает, что вся область под ним
        перерисовывается одинаково сильно, и по его границе идёт ступенька.
    :param hole_margin_ratio: отступ зоны фона от вклеенной головы. Зона 3 идёт
        на высоком strength и всё под собой стирает — до контура новых волос её
        подпускать нельзя. Но и меньше edge_ratio: зоны обязаны ПЕРЕКРЫВАТЬСЯ, а
        не стыковаться, иначе между ними остаётся полоса сырой заливки, которую
        не трогает ни один проход. Связь проверяется в `validate`.
    :param hole_feather_ratio: спад по краям зоны фона; ведётся только наружу
    :param paste_inset_ratio: отступ зоны стилизации внутрь от контура вклейки
    :param paste_guard_strength: какая доля защиты лица остаётся в зоне
        стилизации. 1.0 — лицо закрыто полностью и фактуру не получает, 0.0 —
        открыто наравне с остальным
    """

    edge_ratio: float = 0.03
    edge_outer_ratio: float = 0.05
    neck_ratio: float = 0.12
    guard_ratio: float = 0.06
    feather_ratio: float = 0.04
    gradient_ratio: float = 0.0
    hole_margin_ratio: float = 0.12
    hole_feather_ratio: float = 0.05
    paste_inset_ratio: float = 0.04
    paste_guard_strength: float = 0.6


@dataclass(frozen=True)
class BackgroundPass:
    """
    Отдельный проход по зоне 3 — дыре в фоне на месте стёртой причёски.

    Почему проходов два, а не один. У инпейнтинга одна сила на вызов, а зонам
    нужна разная: на стыке 0.26 (иначе поедут черты и контур причёски), в дыре
    0.85 (иначе фон не восстановить). Совместить их в одном вызове можно было бы
    только если бы эндпоинт читал маску как карту силы попиксельно — он этого не
    делает. Значит, два вызова: сначала фон, потом стык поверх результата.

    Порядок именно такой. Стык сводит вклейку с тем, что вокруг, и «вокруг»
    должно быть уже нарисовано — иначе второй проход будет сводить края с мылом.

    :param min_area_ratio: ниже этой доли кадра проход пропускается. У героя со
        стрижкой дыры почти нет, и платить за второй вызов не за что.
    """

    strength: float = 0.85
    guidance_scale: float = 4.0
    steps: int = 50
    prompt: str = ""
    min_area_ratio: float = 0.002


@dataclass(frozen=True)
class StylisePass:
    """
    Проход по зоне 4 — по самой вклейке, ради фактуры.

    Коллаж собирается из фотографии, а обложка написана маслом. Тон подогнать
    можно (`collage.colour_match`), мазок кисти — нет: это не цвет, а структура.
    Пока лицо вычиталось из масок целиком, оно оставалось стопроцентной
    фотографией на живописи, и склейка читалась именно по фактуре.

    Сила здесь — компромисс, и он честный: чем выше, тем больше мазка и тем
    сильнее плывут черты. 0.35 добавляет фактуру, оставляя лицо узнаваемым;
    выше 0.45 начинает меняться разрез глаз. Второй рычаг — доля защиты лица в
    маске (`MaskProfile.paste_guard_strength`): им регулируют, насколько черты
    остаются под прикрытием.
    """

    strength: float = 0.35
    guidance_scale: float = 3.0
    steps: int = 50
    prompt: str = ""


@dataclass(frozen=True)
class UnifyPass:
    """
    Финальный проход по всему кадру — ради общей фактуры холста.

    Предыдущие проходы работают по маскам, то есть каждый оставляет за собой
    границу: под маской поверхность одна, вне её другая. По отдельности эти
    границы слабые, вместе складываются в тот самый «эффект аппликации», когда
    видно не шов, а разницу материала.

    Здесь маски нет вовсе — обрабатывается весь разворот целиком, и именно
    поэтому сила должна быть маленькой. 0.15-0.20 хватает, чтобы поверх всего
    легло единое зерно холста; выше начинает уходить портретное сходство, и
    уходит оно сразу везде, потому что защиты лица на этом проходе нет.

    Цена прохода — не только вызов: он трогает и те части обложки, которых
    пайплайн до сих пор не касался вовсе. Текст и мелкие детали на развороте
    после него уже не побитово те же.
    """

    strength: float = 0.18
    guidance_scale: float = 2.0
    steps: int = 50
    prompt: str = ""
    # Выше этого значения проход перестаёт быть «единой фактурой» и становится
    # перерисовкой всего разворота
    safe_strength: float = 0.22


@dataclass(frozen=True)
class RefineProfile:
    """
    Полный набор гиперпараметров второго шага.

    :param strategy: имя стратегии в реестре `refine`. Здесь и происходит
        переключение подхода: inpaint_controlnet сегодня, identity_embedding
        (проброс лицевых эмбеддингов) — когда появится эндпоинт, который их
        принимает вместе с маской.
    :param safe_strength: граница, выше которой в лог уходит предупреждение.
        Это не запрет: значение подбирают из окружения, и отказывать из-за
        превышения нельзя — но в логе разбора полётов оно должно быть видно.
    :param control_field: имя ключа, в котором эндпоинт ждёт список карт.
        Схемы у эндпоинтов разные, а лишний ключ заворачивает весь запрос.
    """

    name: str
    strategy: str
    endpoint: str
    prompt: str
    strength: float
    guidance_scale: float
    steps: int
    safe_strength: float
    mask: MaskProfile = field(default_factory=MaskProfile)
    controls: tuple[ControlSpec, ...] = ()
    control_field: str = "controlnets"
    # Второй проход по зоне фона. None — одна зона стыков, как было
    background: BackgroundPass | None = None
    # Проход по самой вклейке: перевод фотографии в живопись. None — вклейка
    # остаётся фотографической, как было до сих пор
    stylise: StylisePass | None = None
    # Финальный проход по всему кадру: общее зерно холста. None — каждая зона
    # остаётся со своей поверхностью
    unify: UnifyPass | None = None

    def validate(self) -> RefineProfile:
        """
        Проверяет профиль до сети.

        Отдельным шагом, а не в __post_init__: профиль собирают и из пресетов, и
        из окружения, и промежуточные состояния законны. Смысл проверки в том,
        чтобы неверная связка стоила 500 с внятным текстом, а не отказа fal
        после трёх загрузок в CDN.
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
        if self.background and not 0.0 <= self.background.strength <= 1.0:
            raise InvalidImageError(
                "strength прохода по фону должен лежать в диапазоне 0..1",
                {"profile": self.name, "strength": self.background.strength},
            )
        if self.background and self.background.strength <= self.strength:
            # Не опечатка, а бессмыслица: ради генерации фона заново второй
            # вызов и оплачивается. Если сила не выше, чем на стыке, он лишний
            raise InvalidImageError(
                "Проход по фону слабее прохода по стыку — тогда он не нужен",
                {
                    "profile": self.name,
                    "background": self.background.strength,
                    "seam": self.strength,
                },
            )
        if self.mask.hole_margin_ratio >= self.mask.edge_outer_ratio + self.mask.gradient_ratio:
            # Зона 3 отступает от вклейки дальше, чем достаёт кольцо зоны 2.
            # Между ними останется полоса, которую не трогает ни один проход, —
            # и в ней сырая заливка на месте чужой причёски. Ровно этот зазор
            # выглядел на обложке грязным контуром вокруг головы.
            raise InvalidImageError(
                "Зоны маски не перекрываются: отступ зоны фона больше кольца стыка",
                {
                    "profile": self.name,
                    "hole_margin": self.mask.hole_margin_ratio,
                    "edge_outer": self.mask.edge_outer_ratio,
                    "gradient": self.mask.gradient_ratio,
                },
            )
        if self.stylise and not 0.0 <= self.stylise.strength <= 1.0:
            raise InvalidImageError(
                "strength прохода стилизации должен лежать в диапазоне 0..1",
                {"profile": self.name, "strength": self.stylise.strength},
            )
        if self.unify and not 0.0 <= self.unify.strength <= 1.0:
            raise InvalidImageError(
                "strength финального прохода должен лежать в диапазоне 0..1",
                {"profile": self.name, "strength": self.unify.strength},
            )
        if self.controls and not self.control_field:
            raise InvalidImageError(
                "Заданы карты ControlNet, но не задано имя ключа для них",
                {"profile": self.name},
            )
        if self.controls and self.endpoint in _NO_CONTROL_ENDPOINTS:
            raise InvalidImageError(
                "Этот эндпоинт не принимает ControlNet — укажите другой в "
                "ML_REFINE_ENDPOINT либо отключите карты через ML_REFINE_CONTROLS=none",
                {
                    "profile": self.name,
                    "endpoint": self.endpoint,
                    "controls": [c.kind for c in self.controls],
                },
            )
        for control in self.controls:
            if control.source not in ("canny", "image"):
                raise InvalidImageError(
                    "Неизвестный способ построения карты управления",
                    {"kind": control.kind, "source": control.source},
                )
        return self

    def report(self) -> dict:
        """Плоская сводка для логов и /health/ready."""
        return {
            "profile": self.name,
            "strategy": self.strategy,
            "endpoint": self.endpoint,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            "gradient_ratio": self.mask.gradient_ratio,
            "controls": [f"{c.kind}:{c.weight}" for c in self.controls],
            "background_strength": self.background.strength if self.background else None,
            "stylise_strength": self.stylise.strength if self.stylise else None,
            "unify_strength": self.unify.strength if self.unify else None,
        }


# Промпт консервативного режима: маска открывает только стык, поэтому и речь
# идёт только о границе. Шея там ещё отрезана по челюсти, отсюда и просьба
# закрыть край воротником.
_CONSERVATIVE_PROMPT = (
    "A photograph of a head has been collaged onto this painted illustration. "
    "Work only along the seam that is masked: blend the outer edge of the hair "
    "into the painted background, and cover the cut at the neck — paint a collar, "
    "a shadow or a strand of hair there so that no torn edge remains. "
    "Match the brush strokes, canvas and paper grain, colour palette, line work "
    "and the direction and temperature of the light of the surrounding artwork. "
    "Keep the hair colour, length and shape exactly as they are, keep the face "
    "untouched — do not redraw, move or reshape anything, this must stay the very "
    "same person. "
    "Seamless hand-painted cover art: no cut-out edge, no halo around the hair, "
    "no visible collage border, no second head or duplicated hair."
)

# Промпт зоны стыков. Открыты только две узкие полосы — контур волос и место,
# где шея входит в тело персонажа, — и работа там ровно одна: свести тон и цвет
# кожи и поставить контактную тень. Про фон здесь не сказано ни слова: фон — это
# зона 3 и отдельный проход.
_SEAM_PROMPT = (
    "A photograph of a head with its neck has been collaged onto this painted "
    "illustration. Work only inside the narrow masked strips. "
    "At the neck: blend the skin of the photographed neck into the painted body "
    "below it — carry the skin tone, colour and warmth of the illustration across "
    "the join so that no line, no step in colour and no edge of the collar remains "
    "visible, and paint a soft contact shadow where the chin and the jaw meet the "
    "neck and where the neck meets the collar, following the light of the scene. "
    "At the hair: dissolve the outer edge of the hair into whatever lies behind it. "
    "Match the brush strokes, canvas grain, palette and line work of the artwork. "
    "Keep the hair colour, length and shape exactly as they are, keep the face "
    "untouched — do not redraw, move or reshape anything, this must stay the very "
    "same person, and leave the rest of the canvas alone. "
    "Seamless hand-painted cover art: no cut-out edge, no halo around the hair, "
    "no visible collage border, no second head or duplicated hair."
)

# Промпт зоны стилизации. Речь идёт только о фактуре: перевести фотографическую
# кожу в живопись, сохранив человека. Про геометрию сказано отдельно и жёстко —
# на 0.35 модель уже способна двигать черты, и напоминание тут не лишнее.
_STYLISE_ZONE_PROMPT = (
    "This head is a photograph collaged onto a hand-painted illustration. "
    "Repaint it in the medium of the artwork: visible oil brush strokes, canvas "
    "texture and paper grain, the same palette, the same edge quality and the same "
    "level of detail as the painting around it. Replace the photographic skin "
    "texture with painted skin — no pores, no photo grain, no camera sharpness. "
    "Keep the person exactly as they are: same facial proportions, same position "
    "and size of the eyes, nose and mouth, same gaze, same hair colour, length and "
    "shape. Do not redraw, move, rotate or reshape any feature — this must remain "
    "the very same recognisable person, only painted instead of photographed."
)

# Промпт финального прохода. Он идёт по всему кадру без маски, поэтому говорит
# только про поверхность и ничего — про содержание: любая просьба «нарисуй» на
# полном кадре означает «перерисуй разворот».
_UNIFY_PROMPT = (
    "A single hand-painted children's book illustration. Unify the whole image "
    "under one surface: the same oil paint, the same visible brush strokes, the "
    "same canvas weave and the same varnish across every part of the picture, so "
    "that nothing looks pasted in or photographic. "
    "Change nothing else: keep every character, face, pose, object and colour "
    "exactly where and as they are — this is a texture pass, not a redraw."
)

# Промпт зоны фона. Здесь модель не сводит, а рисует заново: под маской лежит
# ровная заливка на месте стёртой причёски персонажа, и восстановить по ней
# нечего. Про лицо и волосы не сказано ничего — маска до них не доходит.
_BACKGROUND_PROMPT = (
    "Repaint the masked area as background of this illustration. "
    "Continue the surrounding scene straight through it — the sky, the clouds, the "
    "landscape, the foliage and any creature or object whose edges enter the masked "
    "area must carry on and be completed naturally, with the same brush strokes, "
    "canvas grain, palette, level of detail and direction of light as the artwork "
    "around it. "
    "This is background only: no person, no head, no hair, no face, no figure. "
    "Hand-painted cover art with no flat blurred patch, no smeared area, no seam "
    "and no trace that anything was ever removed."
)

# Промпт режима стилизации. Отличие не косметическое: на strength 0.5 модель
# действительно перерисовывает то, что открыто маской, поэтому от неё требуется
# не «сгладить шов», а положить мазок по всей аппликации и поставить тени.
# Просьба про тень под подбородком продублирована картой глубины: словами модель
# ставит её куда придётся, картой — туда, где объём.
_STYLISE_PROMPT = (
    "A photograph of a head has been collaged onto this painted illustration. "
    "Repaint it as part of the artwork: same brush strokes, same canvas grain, "
    "same palette and line work as the surrounding illustration. "
    "Relight the head to match the scene — put a soft contact shadow under the "
    "chin and along the jaw where the head meets the body, shade the side of the "
    "hair that faces away from the light source of the painting, and let the rim "
    "light fall on the same side as everywhere else in the picture. "
    "Blend the outer edge of the hair into the background and cover the cut at "
    "the neck with a collar, a shadow or a strand of hair. "
    "Keep the identity intact: same facial proportions, same eyes, nose and "
    "mouth, same hair colour and length — this must stay the very same person. "
    "Seamless hand-painted cover art: no cut-out edge, no halo around the hair, "
    "no visible collage border, no second head or duplicated hair."
)


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


def _controls_from_env(raw: str, base: tuple[ControlSpec, ...]) -> tuple[ControlSpec, ...]:
    """
    Разбирает ML_REFINE_CONTROLS: «none» — выключить все, «canny,depth» —
    оставить перечисленные. Веса и пороги остаются из пресета: подбирать их
    строкой в окружении — верный способ получить набор, который никто потом не
    воспроизведёт.
    """
    wanted = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not wanted or wanted == ["none"]:
        return ()

    known = {control.kind: control for control in base}
    unknown = [kind for kind in wanted if kind not in known]
    if unknown:
        raise InvalidImageError(
            "В ML_REFINE_CONTROLS перечислены карты, которых нет в профиле",
            {"unknown": unknown, "available": sorted(known)},
        )
    return tuple(known[kind] for kind in wanted)


def from_settings() -> RefineProfile:
    """
    Активный профиль: пресет из ML_REFINE_PROFILE плюс переопределения ML_REFINE_*.

    Пустое значение переопределения означает «взять из пресета» — именно
    поэтому они объявлены как None, а не как числа с дефолтами: иначе
    невозможно отличить «оператор поставил 0.2» от «оператор не трогал».
    """
    profile = get(settings.refine_profile)

    changes: dict = {}
    if settings.refine_endpoint:
        changes["endpoint"] = settings.refine_endpoint
    if settings.refine_prompt:
        changes["prompt"] = settings.refine_prompt
    if settings.refine_strength is not None:
        changes["strength"] = settings.refine_strength
    if settings.refine_guidance_scale is not None:
        changes["guidance_scale"] = settings.refine_guidance_scale
    if settings.refine_steps is not None:
        changes["steps"] = settings.refine_steps
    if settings.refine_strategy:
        changes["strategy"] = settings.refine_strategy
    if settings.refine_controls:
        changes["controls"] = _controls_from_env(settings.refine_controls, profile.controls)
    if settings.refine_gradient_ratio is not None:
        changes["mask"] = replace(profile.mask, gradient_ratio=settings.refine_gradient_ratio)

    return replace(profile, **changes).validate() if changes else profile.validate()


register(
    RefineProfile(
        name="seam",
        strategy="inpaint_controlnet",
        endpoint=_INPAINT_ENDPOINT,
        prompt=_CONSERVATIVE_PROMPT,
        # Ниже 0.15 мазок не набирается и стык остаётся виден, выше 0.28 плывёт
        # контур причёски. Режим существовал ровно ради этой узкой полосы.
        strength=0.20,
        guidance_scale=2.5,
        # Реально исполняется доля strength от шагов: 50 × 0.20 = 10 шагов денойза
        steps=50,
        safe_strength=0.28,
        mask=MaskProfile(),
    )
)

register(
    RefineProfile(
        name="blend",
        strategy="inpaint_controlnet",
        # Рабочий эндпоинт: на нём висит наша LoRA, и стиль обложек держится на
        # ней. Переезд ради ControlNet стоил бы этого стиля — от карт отказались
        # осознанно, а геометрию причёски вместо них удерживает низкий strength.
        endpoint=_INPAINT_ENDPOINT,
        prompt=_SEAM_PROMPT,
        # Середина запрошенного диапазона 0.25-0.28. Верхняя граница не
        # круглое число: выше ~0.28 начинает плыть контур причёски, и без карт
        # ControlNet удержать его нечем. 0.26 оставляет запас на то, что сама
        # граница определена приблизительно, и всё равно даёт заметно больше
        # работы по шву, чем прежние 0.20.
        strength=0.26,
        # 2.5 — дефолт эндпоинта. На низком strength реально исполняется
        # 0.26 × 50 ≈ 13 шагов денойза, и на таком их числе более высокая
        # строгость промпта лезет артефактами контраста, а не деталями.
        guidance_scale=2.5,
        steps=50,
        # Предупреждение начинается ровно там, где заканчивается безопасный
        # диапазон: 0.25-0.28 — рабочий режим, а не повод сорить в лог
        safe_strength=0.28,
        # Градиент вместо плато, но зона узкая: только контур волос и полоса на
        # стыке шеи. Всё, что шире, уходит в зону фона со своей силой.
        # Сплошное кольцо узкое, а до зоны фона дотягивается градиент: на
        # полной силе кольцо замыливало бы только что восстановленный фон —
        # крыло птеродактиля рядом с головой, — а под градиентом оно его едва
        # касается.
        mask=MaskProfile(gradient_ratio=0.10),
        # Зона 3: дыра от чужой причёски рисуется заново и отдельным вызовом.
        # 0.85 — сила, на которой модель действительно генерирует содержимое, а
        # не подкрашивает; ниже ~0.7 из-под неё проступает мыло от заливки.
        background=BackgroundPass(prompt=_BACKGROUND_PROMPT),
        # Зона 4: перевод самой вклейки из фотографии в живопись
        stylise=StylisePass(prompt=_STYLISE_ZONE_PROMPT),
        # И финальное зерно холста поверх всего разворота
        unify=UnifyPass(prompt=_UNIFY_PROMPT),
    )
)

register(
    RefineProfile(
        name="stylise",
        strategy="inpaint_controlnet",
        endpoint=_INPAINT_ENDPOINT,
        prompt=_STYLISE_PROMPT,
        # Середина запрошенного диапазона 0.45-0.55. На этих значениях модель
        # уже действительно перерисовывает открытое маской — отсюда и градиент
        # маски: плато на такой силе даёт ступеньку по своей границе.
        strength=0.50,
        guidance_scale=3.5,
        steps=50,
        # Предупреждение начинается за верхней границей запрошенного диапазона:
        # 0.45-0.55 — рабочий режим, а не повод сорить в лог на каждом заказе
        safe_strength=0.55,
        mask=MaskProfile(gradient_ratio=0.12),
    )
)

register(
    replace(
        get("stylise"),
        name="stylise_controlnet",
        # Не kontext-inpaint: тот карты не принимает. flux-general — тот самый
        # эндпоинт, через который пробовали ip_adapters (см. README, таблицу
        # отвергнутых моделей), то есть модульные ключи он берёт. Точную форму
        # списка карт — имя ключа и имена полей внутри — проверять первым же
        # боевым прогоном: схему у fal мы уже один раз узнавали постфактум.
        endpoint="fal-ai/flux-general/inpainting",
        controls=(
            # Canny держит рисунок: контур причёски, линию челюсти, разрез глаз.
            # Он же главный предохранитель личности на высоком strength.
            ControlSpec(kind="canny", source="canny", weight=0.65, start=0.0, end=0.8),
            # Depth отвечает за тени: карта объёма подсказывает, где голова
            # выступает над плечами, и модель кладёт контактную тень туда, а не
            # куда пришлось по промпту. Карту считает эндпоинт — локального
            # инференса глубины у сервиса нет.
            ControlSpec(kind="depth", source="image", weight=0.45, start=0.0, end=0.5),
        ),
    )
)
