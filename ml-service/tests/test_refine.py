"""
Второй шаг: профили гиперпараметров, реестр стратегий и схема запроса.

Проверяется три разных вещи, и путать их не стоит:

  * **профиль** — числа и их связность. Несобираемая связка обязана падать до
    сети, а не после трёх загрузок в CDN;
  * **реестр** — что подход переключается именем, а не правкой pipeline.py;
  * **схема запроса** — что уезжает в fal. Фиксируется тестами намеренно: на
    ней уже обжигались вживую, лишний ключ заворачивает весь запрос.

Про рабочий путь. Он безмасочный: эндпоинт получает массив из шаблона и
фотографии и перерисовывает кадр целиком, а голова возвращается в шаблон
локально. Поэтому здесь же проверяется отсутствие ключей — mask_url, strength и
num_inference_steps kontext/max/multi не принимает, и каждый из них означает 422
на весь запрос.
"""

import base64
from dataclasses import replace

import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import fal_api, hair_mask, refine
from app.pipelines.refine import hair_swap, local_render, profiles
from app.utils.image import decode_image, encode_image

# Картинки настоящие, а не заглушки из байтов: безмасочная стратегия декодирует
# и шаблон, и маску, и ответ модели — вклейка идёт локально.
_TEMPLATE = np.full((32, 32, 3), 200, dtype=np.uint8)
_GENERATED = np.full((32, 32, 3), 40, dtype=np.uint8)
_MASK = np.zeros((32, 32), dtype=np.uint8)
_MASK[8:24, 8:24] = 255

_TEMPLATE_PNG = encode_image(_TEMPLATE, "png")[0]
_MASK_PNG = encode_image(_MASK, "png")[0]
_GENERATED_PNG = encode_image(_GENERATED, "png")[0]


class _FakeClient:
    def __init__(self):
        self.uploads: list[tuple[bytes, str]] = []
        self.arguments: dict | None = None
        self.model: str | None = None
        self.calls: list[dict] = []

    def upload(self, data, content_type):
        self.uploads.append((data, content_type))
        return f"https://cdn/{len(self.uploads)}"

    def subscribe(self, model, arguments, with_logs=False):
        self.model = model
        self.arguments = arguments
        self.calls.append(arguments)
        if "swap_image_url" in arguments:
            # Фейссвоп отдаёт одиночный `image`, а не список: вариант у него
            # ровно один. Форма ответа — часть контракта, и подменять её общей
            # заглушкой значит не проверить разбор вовсе
            return {"image": {"url": f"https://cdn/out{len(self.calls)}.png"}, "seed": 7}
        return {"images": [{"url": f"https://cdn/out{len(self.calls)}.png"}], "seed": 7}


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(fal_api, "client", lambda: fake)
    monkeypatch.setattr(fal_api, "_download", lambda image: _GENERATED_PNG)
    return fake


class _FakeResponse:
    def __init__(self, status_code: int, body, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text or str(body)

    def json(self):
        if self._body is None:
            raise ValueError("не JSON")
        return self._body


class _FakeRender:
    """GPU-сервер, отвечающий готовым разворотом. Записывает, что ему прислали."""

    def __init__(self):
        self.calls: list[dict] = []
        self.urls: list[str] = []
        self.timeout: float | None = None
        self.response: _FakeResponse | None = None
        self.error: Exception | None = None

    def post(self, url, json, timeout):
        self.calls.append(json)
        self.urls.append(url)
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return self.response or _FakeResponse(
            200,
            {
                "image": base64.b64encode(_GENERATED_PNG).decode("ascii"),
                "encoding": "image/png",
                "meta": {"steps": 8, "total_s": 31.2, "matched": 1},
            },
        )


@pytest.fixture(autouse=True)
def render(monkeypatch):
    """
    Фальшивый GPU-сервер на рабочем пути — autouse намеренно.

    Под именем `face_swap` в реестре стоит именно он, и без адреса любой заказ
    отвечал бы 503 RENDER_NOT_CONFIGURED, не дойдя до проверки самого теста.
    Сеть при этом не трогается: подменён `requests.post`.
    """
    fake = _FakeRender()
    monkeypatch.setattr(local_render.settings, "render_base_url", "http://gpu-box:8300")
    monkeypatch.setattr(local_render.requests, "post", fake.post)
    return fake


def _request(**overrides) -> refine.RefineRequest:
    kwargs = {
        "target": _TEMPLATE_PNG,
        "target_mime": "image/png",
        "mask": _MASK_PNG,
        "identity": b"photo-bytes",
        "identity_mime": "image/jpeg",
    }
    kwargs.update(overrides)
    return refine.RefineRequest(**kwargs)


def _profile() -> profiles.RefineProfile:
    """Рабочий профиль: специализированный фейссвоп."""
    return profiles.get("pixar_real")


def _with(strategy: str) -> profiles.RefineProfile:
    """
    Тот же профиль, переведённый на другую стратегию её же умолчаниями.

    Ровно то, что делает `ML_REFINE_STRATEGY`: подход приносит с собой эндпоинт,
    схему, инструкцию и признак локальной геометрии.
    """
    defaults = profiles.strategy_defaults(strategy)
    return replace(
        _profile(),
        strategy=strategy,
        endpoint=defaults.endpoint,
        payload=defaults.payload,
        instruction=defaults.instruction,
        needs_mask=defaults.needs_mask,
    )


def _multi() -> profiles.RefineProfile:
    """Безмасочная диффузия по кадру целиком плюс локальная вклейка."""
    return _with("kontext_multi")


def _inpaint() -> profiles.RefineProfile:
    """Прежний масочный путь — он остался под живописной серией."""
    return profiles.get("impasto")


def _fal_swap() -> profiles.RefineProfile:
    """
    Прежний рабочий путь: специализированный фейссвоп на fal.

    Имя стратегии сменилось с `face_swap` на `fal_face_swap`, когда первое занял
    свой GPU-сервер. Сама схема запроса не изменилась ни на ключ, и тесты ниже
    сторожат именно её: лишний ключ заворачивает весь запрос.
    """
    return _with("fal_face_swap")


# --- Профили: числа и их связность ---


def test_presets_cover_both_series():
    """Две серии книг — два стиля."""
    assert {"pixar_real", "impasto"} <= set(profiles.available())

    assert profiles.get("impasto").style == "impasto"
    assert profiles.get("pixar_real").style == "pixar_real"


def test_default_profile_is_a_face_swap_without_diffusion():
    """
    Смена парадигмы. Обе диффузионные стратегии провалились: инпейнт — на
    отсутствии контекста внутри маски, безмасочный img2img — на том, что общая
    модель не умеет хирургической замены по тексту. Рабочий путь лицо не рисует.
    """
    profile = _profile()

    assert profile.strategy == "face_swap"
    assert profile.endpoint == "local/demo-render", "облака в рабочем пути нет вовсе"
    assert profile.payload.name == "demo_render"
    assert profile.payload.prompt_field is None, "промпт-запрет живёт на GPU-сервере"
    assert profile.payload.mask_field is None
    assert profile.needs_mask is False, "маску и силуэт считает сам сервер"


def test_maskless_diffusion_stays_available_for_comparison():
    """
    Провал воспроизводится одной переменной окружения. Спорить о результате
    дешевле, глядя на два файла, чем на память.
    """
    profile = _multi()

    assert profile.endpoint == "fal-ai/flux-pro/kontext/max/multi"
    assert profile.payload.images_field == "image_urls"
    assert profile.payload.mask_field is None, "эндпоинт маску не принимает"


def test_strength_sits_high_because_there_is_nothing_to_preserve():
    """
    Про масочный путь: там под маской лежит чужой персонаж, и слабая генерация
    оставит от него черты — получится смесь двух лиц. На безмасочном пути
    strength в запрос не уходит вовсе, и число сохранено только ради него.
    """
    profile = _inpaint()

    assert profile.strength >= 0.85
    # Предупреждение уходит, когда сила падает НИЖЕ порога, а не поднимается
    assert profile.safe_strength <= profile.strength


def test_prompt_addresses_the_images_by_order_and_demands_a_photograph():
    """
    Маски у модели нет, поэтому область названа порядком картинок: первая —
    сцена, вторая — личность. Перепутать их местами значит перерисовать
    фотографию заказчика по мотивам разворота, причём молча.

    Фотореализм требуется словами и подпирается отрицаниями: на иллюстрации
    kontext охотно продолжает её материал и отдаёт нарисованное лицо.
    """
    prompt = _multi().prompt().lower()

    assert "first image" in prompt and "second image" in prompt
    assert "real photograph of a real child" in prompt
    assert "no painting" in prompt and "no brush strokes" in prompt
    # Поза и мимика — со сцены, черты — с фотографии. Ради этого разделения всё
    assert "keep the facial expression" in prompt
    assert "identity" in prompt


def test_masked_path_keeps_its_own_instruction():
    """
    Инструкция — часть стратегии, а не стиля: масочный путь адресует область
    («inside the mask»), безмасочный — картинки по порядку. Один текст на оба
    означает, что одна из стратегий говорит модели неправду.
    """
    prompt = _inpaint().prompt().lower()

    assert "inside the mask" in prompt
    assert "take the identity from the reference" in prompt
    assert "first image" not in prompt


def test_expression_is_woven_into_the_prompt():
    prompt = _multi().prompt("override the expression: the child is laughing")

    assert "laughing" in prompt


@pytest.mark.parametrize(
    "changes",
    [{"strength": 1.5}, {"steps": 0}, {"endpoint": ""}, {"style": "не-существует"}],
)
def test_broken_profile_is_refused_before_the_network(changes):
    """
    Профиль проверяется до сети. Иначе неверная связка стоит двух загрузок в
    CDN и отказа fal — с текстом, по которому ничего не понять.
    """
    with pytest.raises(InvalidImageError):
        replace(_profile(), **changes).validate()


def test_default_profile_builds():
    """Профиль по умолчанию обязан собираться без единой переменной окружения."""
    assert profiles.from_settings().report()["profile"] in profiles.available()


def test_environment_overrides_the_preset(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_profile", "impasto")
    monkeypatch.setattr(profiles.settings, "refine_strength", 0.88)
    monkeypatch.setattr(profiles.settings, "mask_dilate_ratio", 0.2)

    profile = profiles.from_settings()

    assert profile.strength == 0.88
    assert profile.mask.dilate_ratio == 0.2
    assert profile.mask.feather_ratio == profiles.get("impasto").mask.feather_ratio
    assert profile.guidance_scale == profiles.get("impasto").guidance_scale


def test_strategy_brings_its_endpoint_schema_and_instruction(monkeypatch):
    """
    Вещи, которые меняются только вместе. Другой подход — это другой эндпоинт,
    другой набор ключей, другой текст промпта и другая локальная работа; любая
    их комбинация из разных стратегий даёт либо 422, либо молча испорченный кадр.
    """
    monkeypatch.setattr(profiles.settings, "refine_strategy", "identity_inpaint")

    profile = profiles.from_settings()

    assert profile.strategy == "identity_inpaint"
    assert profile.endpoint == "fal-ai/flux-kontext-lora/inpaint"
    assert profile.payload.mask_field == "mask_url"
    assert "inside the mask" in profile.prompt()
    assert profile.needs_mask is True, "инпейнту маска нужна — её и рисуют"


def test_diffusion_strategies_bring_back_the_local_geometry(monkeypatch):
    """
    Обратный переход опаснее прямого: профиль по умолчанию маску не строит, и
    без этого признака диффузия получила бы пустую маску вместо рабочей области.
    """
    monkeypatch.setattr(profiles.settings, "refine_strategy", "kontext_multi")

    profile = profiles.from_settings()

    assert profile.needs_mask is True
    assert profile.reference_pad_max > 1.0, "kontext переносит композицию фотографии"


def test_explicit_endpoint_beats_the_strategy_default(monkeypatch):
    """
    Иначе новый эндпоинт нельзя было бы попробовать без релиза, а пробовать их
    приходится — на fal схемы меняются чаще, чем выходят наши версии.
    """
    monkeypatch.setattr(profiles.settings, "refine_strategy", "identity_inpaint")
    monkeypatch.setattr(profiles.settings, "refine_endpoint", "fal-ai/что-нибудь-новое")

    assert profiles.from_settings().endpoint == "fal-ai/что-нибудь-новое"


def test_identity_field_is_overridable_without_touching_the_code(monkeypatch):
    """
    Имя поля референса — то, чем эндпоинты отличаются друг от друга. Смена
    эндпоинта не должна требовать релиза.
    """
    monkeypatch.setattr(profiles.settings, "refine_strategy", "identity_inpaint")
    monkeypatch.setattr(profiles.settings, "refine_identity_field", "face_image_url")

    assert profiles.from_settings().payload.identity_field == "face_image_url"


def test_unknown_payload_schema_is_refused(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_payload", "телепатия")

    with pytest.raises(InvalidImageError):
        profiles.from_settings()


def test_unknown_profile_name_is_refused():
    with pytest.raises(InvalidImageError):
        profiles.get("не-существует")


def test_schema_without_a_template_field_is_refused():
    """Схема, не называющая, куда класть шаблон, — дефект конфигурации."""
    with pytest.raises(InvalidImageError):
        profiles.PayloadSchema(name="пустая", image_field=None, images_field=None)


def test_blank_override_means_from_profile():
    """
    В .env.example ручки перечислены с пустыми значениями — так видно, что они
    есть. Пустая строка обязана означать «не задано»: иначе сервис падает на
    старте, разбирая конфиг, который оператор считает пустым.
    """
    from app.config import Settings

    blank = Settings(refine_strength="", refine_steps="", mask_dilate_ratio="")

    assert blank.refine_strength is None
    assert blank.refine_steps is None
    assert blank.mask_dilate_ratio is None


# --- Реестр стратегий ---


def test_strategy_is_chosen_by_name_from_the_profile(client):
    """
    Смена подхода — это смена имени в профиле. Ни pipeline.py, ни fal_api.py о
    существовании второй стратегии знать не должны.
    """
    seen = {}

    class _Spy:
        name = "spy"

        def refine(self, request, profile):
            seen["profile"] = profile.name
            return refine.RefineResult(image=b"SPY", meta={})

    refine.register(_Spy())
    result = refine.run(_request(), replace(_profile(), strategy="spy"))

    assert result.image == b"SPY"
    assert seen["profile"] == "pixar_real"
    assert client.arguments is None, "до fal дойти не должно"


def test_both_approaches_stay_registered():
    """
    Масочный путь удалять рано: он провалился на наших парах, но сравнивать
    новый результат не с чем, если старый нельзя воспроизвести одной ручкой.
    """
    assert {"face_swap", "hair_swap", "kontext_multi", "identity_inpaint"} <= set(
        refine.available()
    )


def test_unknown_strategy_is_refused():
    with pytest.raises(refine.RefinerNotSupportedError):
        refine.run(_request(), replace(_profile(), strategy="телепатия"))


# --- Схема запроса: фейссвоп на fal (выключенный путь) ---


def test_arguments_match_the_face_swap_schema(client):
    """
    Две ссылки и ничего больше. Проверяется полным сравнением множества, а не
    наличием: лишний ключ здесь стоит столько же, сколько недостающий, — 422 на
    весь запрос после двух загрузок в CDN.
    """
    refine.run(_request(), _fal_swap())

    args = client.arguments
    assert client.model == "fal-ai/face-swap"
    assert set(args) == {"base_image_url", "swap_image_url"}


def test_the_template_is_the_base_and_the_photo_is_the_face(client):
    """
    Перепутать их местами значит вклеить лицо персонажа в фотографию заказчика:
    запрос пройдёт, картинка вернётся, и это будет не разворот книги.
    """
    refine.run(_request(), _fal_swap())

    assert client.arguments["base_image_url"] == "https://cdn/1"
    assert client.arguments["swap_image_url"] == "https://cdn/2"
    assert client.uploads[0] == (_TEMPLATE_PNG, "image/png")
    assert client.uploads[1] == (b"photo-bytes", "image/jpeg")


def test_a_single_image_response_is_understood(client):
    """
    Фейссвоп отдаёт одиночный `image`, диффузия — список `images`. Обе формы
    живые, и разбираться они обязаны обе: иначе рабочий путь падает на разборе
    успешного ответа.
    """
    result = refine.run(_request(), _fal_swap())

    assert result.image == _GENERATED_PNG
    assert result.meta["seed"] == 7


def test_an_unchanged_answer_is_refused(client, monkeypatch):
    """
    Молчаливый отказ — главная ловушка этого эндпоинта. Не найдя лица, он не
    отвечает ошибкой: он возвращает присланный шаблон. Код 200, картинка на
    месте, счёт выставлен, замены нет — и такой разворот уедет прямо в печать.
    """
    monkeypatch.setattr(fal_api, "_download", lambda image: _TEMPLATE_PNG)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        refine.run(_request(), _fal_swap())

    assert exc_info.value.status_code == 422
    assert exc_info.value.details["changed"] < 1.0


def test_a_wrong_schema_never_reaches_the_face_swap(client):
    """Схема с промптом — не для этого эндпоинта: он принимает две ссылки."""
    with pytest.raises(InvalidImageError):
        refine.run(_request(), replace(_fal_swap(), payload=profiles.KONTEXT_MULTI))

    assert client.uploads == []


def test_the_face_swap_needs_no_mask_at_all(client):
    """
    Область эндпоинт находит своим детектором. Маска на этом пути не строится
    вовсе — и её отсутствие в запросе не должно ничего ломать.
    """
    refine.run(_request(mask=b""), _fal_swap())

    assert len(client.uploads) == 2, "в CDN уезжают шаблон и фотография, и только"


@pytest.mark.parametrize(
    "key",
    ["prompt", "mask_url", "strength", "num_inference_steps", "output_format", "image_urls"],
)
def test_face_swap_arguments_carry_no_diffusion_keys(client, key):
    """
    Всё, что было ручками диффузии, здесь не существует. Промпта в том числе:
    фотореализм переносится моделью, а не выпрашивается словами.
    """
    refine.run(_request(), _fal_swap())

    assert key not in client.arguments


@pytest.mark.parametrize(
    ("payload", "endpoint"),
    [
        (profiles.NANO_BANANA, "fal-ai/nano-banana/edit"),
        (profiles.SEEDREAM_EDIT, "fal-ai/bytedance/seedream/v4/edit"),
        (profiles.HY_WU_EDIT, "fal-ai/hy-wu-edit"),
    ],
)
def test_editor_endpoints_reuse_the_multi_image_strategy(client, payload, endpoint):
    """
    Запасной путь для стилизованных шаблонов. Схема у редакторов одна и та же —
    промпт плюс массив картинок, — поэтому новый эндпоинт стоит двух переменных
    окружения и ни строчки кода.
    """
    profile = replace(_multi(), payload=payload, endpoint=endpoint)
    refine.run(_request(), profile)

    args = client.arguments
    assert client.model == endpoint
    assert args["image_urls"] == ["https://cdn/1", "https://cdn/2"]
    assert args["prompt"]
    # Ключей вне схемы быть не должно: лишний заворачивает весь запрос
    assert set(args) == set(payload.sent_keys())
    assert "strength" not in args and "num_inference_steps" not in args


# --- Двухшаговый путь: причёска, затем лицо ---


@pytest.fixture
def hair(monkeypatch):
    """
    Маска волос без разметки и без mediapipe.

    Строит её сама стратегия, из шаблона, — тестам здесь проверять нечего: своя
    проверка у неё в test_hair_mask.py. Важно другое: что маска попала во
    вклейку и что вокруг неё шаблон уцелел.
    """
    built = hair_mask.HairMask(mask=_MASK, face_height=8.0, meta={"source": "parsing"})
    monkeypatch.setattr(hair_mask, "build", lambda image, **kwargs: built)
    return built


def _hair_profile() -> profiles.RefineProfile:
    """Тот же фотореализм, но с проходом причёски перед заменой лица."""
    return profiles.get("pixar_hair")


def _hair_request(**overrides) -> refine.RefineRequest:
    """
    Заказ с описанием причёски.

    Описание обязательно: фотография на первый шаг не уезжает, и без слов
    редактору нечего рисовать. Тесты, проверяющие именно это, зовут `_request`.
    """
    kwargs = {"hair": "short blonde buzz cut"}
    kwargs.update(overrides)
    return _request(**kwargs)


def test_the_hair_preset_only_adds_a_step(client, hair):
    """
    Шаг замены лица обязан остаться неотличимым от рабочего пути: тот же
    эндпоинт, та же схема, тот же отсутствующий промпт. Двухшаговая схема
    ДОБАВЛЯЕТ проход перед ним, а не переделывает его.
    """
    profile = _hair_profile()

    assert profile.strategy == "hair_swap"
    assert profile.endpoint == profiles.get("pixar_real").endpoint
    assert profile.payload.name == profiles.get("pixar_real").payload.name
    assert profile.needs_mask is False, "маску волос строит стратегия, а не пайплайн"


def test_the_hair_goes_first_and_the_face_second(client, hair, render):
    """
    Порядок шагов — главное решение этого пути. Правка волос это диффузия, и
    всё, что попадёт под неё, вернётся сглаженным; пущенная после переноса, она
    прошла бы по настоящим пикселям лица — по тому единственному, ради чего всё.
    """
    refine.run(_hair_request(), _hair_profile())

    assert len(client.calls) == 1, "на fal уходит один вызов — редактор причёски"
    assert "image_urls" in client.calls[0], "первым — редактор причёски"
    assert len(render.calls) == 1, "вторым — свой GPU-сервер"
    assert set(render.calls[0]) == {"base_image", "donor_photo", "output_format"}


def test_the_photo_never_reaches_the_editor(client, hair, render):
    """
    Дыра, стоившая двух прогонов. Получив вторым файлом портрет крупным планом,
    универсальный редактор понимает его не как «вот чья причёска», а как «вот
    что нарисовать»: и nano-banana, и seedream нарисовали лицо донора на затылке
    персонажа — в пустой области маски волос, то есть ровно там, где им
    разрешили рисовать.

    Личность переносит фейссвоп на втором шаге. На первом фотографии нет ни в
    запросе, ни в CDN — проверяется и то, и другое.
    """
    refine.run(_hair_request(), _hair_profile())

    assert client.calls[0]["image_urls"] == ["https://cdn/1"], "в массиве один шаблон"
    assert client.uploads[0][0] != b"photo-bytes"
    assert len(client.uploads) == 1, "в CDN уезжает одно окно, и больше на fal ничего"
    # Фотография попадает только на второй шаг, а он идёт на свой GPU-сервер и
    # мимо CDN вообще
    assert render.calls[0]["donor_photo"] == base64.b64encode(b"photo-bytes").decode("ascii")


def test_a_schema_carrying_the_photo_cannot_be_used_for_hair():
    """
    Тот же дефект, но пойманный до сети. Схема с фотографией в массиве не даст
    ни ошибки, ни 422: редактор ответит успехом и нарисует второе лицо. Значит,
    неверную связку нельзя собирать вовсе — только «одиночные» схемы.
    """
    with pytest.raises(InvalidImageError) as exc_info:
        replace(
            _hair_profile(), hair=replace(_hair_profile().hair, payload=profiles.NANO_BANANA)
        ).validate()

    assert "nano_banana_solo" in exc_info.value.details["solo"]


def test_hair_without_words_is_refused_before_the_network(client, hair):
    """
    Референса на этом шаге нет, и «замени причёску» без описания означает
    случайную причёску. Дефект заказа, а не модели: чинится одним полем.
    """
    with pytest.raises(InvalidImageError) as exc_info:
        refine.run(_request(), _hair_profile())

    assert exc_info.value.status_code == 400
    assert "ML_HAIR_DESCRIPTION" in exc_info.value.details["hint"]
    assert client.uploads == [], "до загрузки в CDN дойти не должно"


def test_the_face_swap_receives_the_edited_template(client, hair, render):
    """
    Смысл вклейки между шагами. Редактор перерисовал кадр целиком, но фейссвопу
    достаётся шаблон, в котором заменена ровно причёска: лицо, фон и одежда —
    исходные пиксели разворота, и лицо в том числе не тронуто ни на пиксель.
    """
    refine.run(_hair_request(), _hair_profile())

    edited = decode_image(base64.b64decode(render.calls[0]["base_image"]))
    assert tuple(edited[16, 16]) == (40, 40, 40), "под маской волос — правка редактора"
    assert tuple(edited[1, 1]) == (200, 200, 200), "вне её — шаблон побитово"
    assert render.calls[0]["donor_photo"] == base64.b64encode(b"photo-bytes").decode(
        "ascii"
    ), "фотография уезжает как есть"


def test_the_editor_gets_the_template_and_the_words(client, hair):
    """
    Схема редактора — массив, и в нём ровно один файл. Всё остальное, что он
    знает о задаче, — текст.
    """
    refine.run(_hair_request(), _hair_profile())

    arguments = client.calls[0]
    assert arguments["image_urls"] == ["https://cdn/1"]
    assert set(arguments) == set(_hair_profile().hair.payload.sent_keys())
    assert "mask_url" not in arguments, "маска работает локально, эндпоинт её не примет"


def test_the_hair_prompt_guards_the_face_and_demands_a_photograph(client, hair):
    """
    Промпт первого шага решает две задачи разом: правится одна причёска, и
    правится фотореалистично. Лицо здесь охраняется словами, потому что за ним
    придёт детектор фейссвопа, а сглаженная диффузией кожа — прямой запрет серии.
    """
    prompt = _hair_profile().hair.prompt("short blonde buzz cut").lower()

    assert "do not swap the face" in prompt
    assert "same position and angle of the head" in prompt
    # След старой причёски: длинные волосы закрывают уши, шею и одежду
    assert "rebuilt everywhere the old hair used to be" in prompt
    assert "no painting" in prompt and "no wig" in prompt
    # Описание идёт последним: инструкция им заканчивается
    assert prompt.endswith("short blonde buzz cut")


def test_the_prompt_forbids_a_second_face(client, hair):
    """
    Правка после второго прогона. Референса в запросе больше нет, но пустая
    область маски волос сама по себе приглашает модель что-нибудь туда
    нарисовать — и оба редактора нарисовали лицо. Запрет продублирован словами:
    стоит он дёшево, а стоил дорого.
    """
    prompt = _hair_profile().hair.prompt("buzz cut").lower()

    assert "no reference image" in prompt, "картинка одна, и модель должна это знать"
    assert "never draw a face" in prompt
    assert "exactly one person in this image and exactly one face" in prompt
    assert "no second face" in prompt and "no face on the back of the head" in prompt
    # Прежней адресации по порядку картинок здесь быть не должно: картинка одна
    assert "second image" not in prompt


def test_the_prompt_forbids_keeping_the_old_volume(client, hair):
    """
    Правка после первого живого прогона. Прежний текст просил сохранить «same
    head size» — фразой против перекадрирования, — и редактор прочитал её
    буквально: голова вместе с копной волос объявлена неприкосновенной по
    размеру, и вышел блондинистый шлем при объёме шаблона.

    Теперь череп с лицом и контур причёски разведены прямым текстом, а старая
    причёска названа формой на удаление, а не формой для правки.
    """
    prompt = _hair_profile().hair.prompt("buzz cut").lower()

    assert "do not keep its volume" in prompt
    assert "visibly smaller in silhouette" in prompt
    assert "shape to delete" in prompt
    # Череп не меняется — меняются только волосы вокруг него
    assert "keep the skull and the face at exactly the same size" in prompt
    # «Короткие» описываются физически: иначе для модели это «покороче прежних»
    assert "millimetres thick" in prompt and "scalp shows through" in prompt
    assert "no rounded cap" in prompt and "no helmet hair" in prompt


def test_the_order_can_describe_the_hair_in_words(client, hair):
    """
    Причёска приезжает референсом, но на фотографии её бывает не видно — шапка,
    кадр по подбородок, тёмный фон. Тогда работает описание из заказа.
    """
    refine.run(_request(hair="short blonde buzz cut"), _hair_profile())

    assert "short blonde buzz cut" in client.calls[0]["prompt"]


def test_the_words_can_come_from_the_environment_instead(client, hair, monkeypatch):
    """
    Одна причёска на весь тираж задаётся оператором один раз, а не менеджером в
    каждом заказе. Источника два, и любого из них достаточно.
    """
    profile = replace(
        _hair_profile(), hair=replace(_hair_profile().hair, description="platinum buzz cut")
    )

    refine.run(_request(), profile)

    assert "platinum buzz cut" in client.calls[0]["prompt"]


def _big_template(monkeypatch, mask_box=(24, 40)):
    """
    Разворот вчетверо больше маски волос: только на нём окно вокруг головы
    отличается от кадра целиком. На 32×32 из остальных тестов поле в высоту лица
    накрывает всё, и резать нечего.
    """
    template = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    top, bottom = mask_box
    mask[top:bottom, top:bottom] = 255

    built = hair_mask.HairMask(mask=mask, face_height=8.0, meta={"source": "parsing"})
    monkeypatch.setattr(hair_mask, "build", lambda image, **kwargs: built)
    return encode_image(template, "png")[0]


def test_the_editor_gets_a_window_around_the_head(client, monkeypatch):
    """
    Второй урок первого прогона. Правка удалась, но объём причёски остался
    шаблонным: на 4K-развороте голова занимает проценты кадра, рабочий кадр
    редактора около мегапикселя — и стричь ему физически нечего. В окно те же
    волосы приходят тысячами пикселей.
    """
    target = _big_template(monkeypatch)

    result = refine.run(_hair_request(target=target), _hair_profile())

    # Маска 24..40 плюс поле в одну высоту лица (8) с каждой стороны
    assert result.meta["hair"]["window"] == [16, 16, 32, 32]
    assert decode_image(client.uploads[0][0]).shape[:2] == (32, 32)


def test_outside_the_window_the_spread_is_untouched(client, monkeypatch, render):
    """
    Окно возвращается на место копией, а не правкой шаблона под собой. Вне окна
    и вне маски внутри него разворот обязан дойти до фейссвопа исходным.
    """
    target = _big_template(monkeypatch)

    refine.run(_hair_request(target=target), _hair_profile())

    edited = decode_image(base64.b64decode(render.calls[0]["base_image"]))
    assert edited.shape[:2] == (64, 64), "на замену лица уезжает разворот, а не окно"
    assert tuple(edited[32, 32]) == (40, 40, 40), "под маской волос — правка редактора"
    assert tuple(edited[1, 1]) == (200, 200, 200), "за окном — шаблон побитово"
    assert tuple(edited[20, 20]) == (200, 200, 200), "в окне вне маски — тоже шаблон"


def test_the_window_can_be_switched_off(client, monkeypatch):
    """
    Ноль возвращает прежнее поведение — разворот целиком, теми же байтами.
    Нужно это там, где голова и так занимает кадр: лишнее перекодирование 4K
    стоит фактуры, которую модель просят повторить.

    Стирание тоже выключено, и не для удобства теста: стёртый кадр — это новые
    пиксели, и отдать их прежними байтами невозможно. Байт в байт разворот
    уезжает ровно тогда, когда мы в нём ничего не трогали.
    """
    target = _big_template(monkeypatch)
    profile = replace(
        _hair_profile(), hair=replace(_hair_profile().hair, crop_ratio=0.0, erase_ratio=0.0)
    )

    result = refine.run(_hair_request(target=target), profile)

    assert result.meta["hair"]["window"] is None
    assert client.uploads[0] == (target, "image/png"), "шаблон уезжает как прислан"


def test_a_window_equal_to_the_frame_is_no_window(client, hair):
    """
    Поле раздулось до кадра — резать нечего. Тогда окна нет вовсе, и разворот
    уходит исходными байтами: перекодировать его ради нулевой обрезки незачем.
    Стирание при этом выключено — оно само по себе делает новые пиксели.
    """
    profile = replace(_hair_profile(), hair=replace(_hair_profile().hair, erase_ratio=0.0))

    result = refine.run(_hair_request(), profile)

    assert result.meta["hair"]["window"] is None
    assert client.uploads[0] == (_TEMPLATE_PNG, "image/png")


def _wipes(**overrides) -> profiles.RefineProfile:
    """
    Профиль с включённым стиранием — путь, который по умолчанию выключен.

    Прогон показал, что задачи он не решает: прядь лежала ВНЕ маски, и заливка до
    неё не достала. Стоил он при этом силуэта причёски и светлого ореола. Тесты
    ниже держат его рабочим — воспроизвести провал одной переменной дешевле, чем
    спорить о нём, — но включают явно, а не через умолчание.
    """
    hair = replace(_hair_profile().hair, erase_ratio=0.06, **overrides)
    return replace(_hair_profile(), hair=hair)


def _template_with_a_strand(monkeypatch) -> bytes:
    """Разворот с тёмной прядью внутри открытой маски."""
    template = np.full((64, 64, 3), 200, dtype=np.uint8)
    template[30:34, 26:38] = 20

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255

    built = hair_mask.HairMask(mask=mask, face_height=8.0, meta={"source": "parsing"})
    monkeypatch.setattr(hair_mask, "build", lambda image, **kwargs: built)
    return encode_image(template, "png")[0]


def test_the_cheek_band_reaches_the_mask(client, monkeypatch, hair):
    """
    Полоса вдоль щёк живёт в профиле, а работает в маске — между ними один
    аргумент, и молчаливо потерять его значит вернуться к локону на щеке, не
    заметив этого ни по логам, ни по тестам маски. Своя проверка у полосы в
    test_hair_mask.py; здесь проверяется только, что число до неё доезжает.
    """
    seen: dict = {}
    monkeypatch.setattr(
        hair_mask, "build", lambda image, **kwargs: (seen.update(kwargs), hair)[1]
    )

    refine.run(_hair_request(), _hair_profile())

    assert seen["cheek_ratio"] == _hair_profile().hair.cheek_ratio > 0


def test_the_wiping_is_off_by_default(client, monkeypatch):
    """
    Стирание выключено, и это вывод прогона, а не осторожность.

    Гипотеза была в том, что редактор цепляется за структуру старых волос внутри
    открытой маски. Прогон её опроверг: залитая мылом маска оставила прядь на
    щеке резкой — значит, прядь лежала ВНЕ маски, и разметка её не видит. Берётся
    она полосой вдоль щеки, а стирание к цене за себя добавило лысую голову
    (пропал силуэт причёски) и вернувшийся ореол.

    Промпт при этом обязан молчать о стирании: пообещать редактору стёртую
    область и прислать нетронутую фотографию — значит попросить его убрать
    волосы, которых он не найдёт.
    """
    target = _template_with_a_strand(monkeypatch)

    result = refine.run(_hair_request(target=target), _hair_profile())

    sent = decode_image(client.uploads[0][0])
    assert int(sent.min()) == 20, "прядь уехала редактору как есть"
    assert result.meta["hair"]["erase"] == {"erased": False, "reason": "off"}
    assert "deliberately wiped out" not in client.calls[0]["prompt"]


def test_the_wiping_still_works_when_switched_on(client, monkeypatch):
    """
    Провалившийся путь остаётся рабочим: одной переменной он воспроизводится
    целиком, вместе с припиской в промпте о том, что пятно — не содержимое.
    """
    target = _template_with_a_strand(monkeypatch)

    refine.run(_hair_request(target=target), _wipes())

    sent = decode_image(client.uploads[0][0])
    assert int(sent.min()) > 150, "под маской не осталось тёмных пикселей"
    assert "deliberately wiped out" in client.calls[0]["prompt"]


def test_the_wiping_stops_at_the_mask(client, monkeypatch, render):
    """
    Стирание — свойство ЗАПРОСА. В шаблон оно не попадает никогда: вклеивается
    ответ редактора, и вклеивается он в исходный кадр, поэтому вне маски до
    фейссвопа доезжают пиксели разворота — вместе с лицом, которое он ищет.
    """
    target = _template_with_a_strand(monkeypatch)

    refine.run(_hair_request(target=target), _wipes())

    edited = decode_image(base64.b64decode(render.calls[0]["base_image"]))
    assert tuple(edited[1, 1]) == (200, 200, 200), "вне маски — шаблон побитово"
    assert tuple(edited[32, 32]) == (40, 40, 40), "под маской — правка редактора"


def test_the_silent_editor_is_caught_against_what_was_sent(client, monkeypatch):
    """
    Стирание меняет и то, с чем сравнивается ответ. Молчаливый отказ редактора —
    это присланный кадр обратно, то есть кадр СТЁРТЫЙ, и от шаблона он отличается
    сильно: сравнение с шаблоном объявило бы такой ответ удачной правкой и
    отправило бы фейссвопу размытое пятно вместо причёски.
    """
    target = _template_with_a_strand(monkeypatch)
    monkeypatch.setattr(fal_api, "_download", lambda image: client.uploads[0][0])

    with pytest.raises(hair_swap.HairEditFailedError):
        refine.run(_hair_request(target=target), _wipes())

    assert len(client.calls) == 1, "до фейссвопа дойти не должно"


def test_an_unchanged_hair_is_refused_before_the_face_swap(client, hair, monkeypatch):
    """
    Редактор тоже умеет промолчать: 200, кадр на месте, причёска прежняя. Ловится
    это сравнением ВНУТРИ маски — среднее по кадру утонуло бы в неизменном фоне.
    Отказ обязан прийти до второго вызова: платить за него незачем.
    """
    monkeypatch.setattr(fal_api, "_download", lambda image: _TEMPLATE_PNG)

    with pytest.raises(hair_swap.HairEditFailedError) as exc_info:
        refine.run(_hair_request(), _hair_profile())

    assert exc_info.value.status_code == 422
    assert len(client.calls) == 1, "до фейссвопа дойти не должно"


def test_meta_reports_both_steps(client, hair):
    """
    Заказ прошёл двумя вызовами, и по мете это должно быть видно сразу: иначе
    странный результат ищут в фейссвопе, а испортила его правка причёски.
    """
    result = refine.run(_hair_request(), _hair_profile())

    assert result.meta["strategy"] == "hair_swap"
    assert result.meta["hair"]["source"] == "parsing"
    assert result.meta["hair"]["changed"] > 1.0
    assert result.meta["hair"]["endpoint"] == _hair_profile().hair.endpoint
    # Результат второго шага — то, что вернул GPU-сервер, и он остался прежним
    assert result.image == _GENERATED_PNG
    assert result.meta["render"]["steps"] == 8


def test_the_hair_stage_is_switched_by_one_variable(monkeypatch):
    """
    Редакторы взаимозаменяемы: схема у них одна. Новый эндпоинт обязан стоить
    переменной окружения, а не релиза, — на fal схемы меняются чаще, чем выходят
    наши версии.
    """
    monkeypatch.setattr(profiles.settings, "refine_profile", "pixar_hair")
    monkeypatch.setattr(profiles.settings, "hair_endpoint", "fal-ai/bytedance/seedream/v4/edit")
    monkeypatch.setattr(profiles.settings, "hair_payload", "seedream_solo")
    monkeypatch.setattr(profiles.settings, "hair_description", "short blonde buzz cut")

    profile = profiles.from_settings()

    assert profile.hair.endpoint == "fal-ai/bytedance/seedream/v4/edit"
    assert profile.hair.payload.name == "seedream_solo"
    assert profile.hair.payload.images_identity is False, "фотография донора туда не уезжает"
    assert "short blonde buzz cut" in profile.hair.prompt()
    # Шаг замены лица переключение причёски не задевает
    assert profile.endpoint == profiles.get("pixar_real").endpoint


def test_a_hair_schema_without_an_array_is_refused():
    """
    Редактор принимает картинки массивом. Схема с двумя именованными полями
    отправила бы ему шаблон в image_url — 422 на весь запрос после двух загрузок.
    """
    broken = replace(_hair_profile().hair, payload=profiles.FACE_SWAP)

    with pytest.raises(InvalidImageError):
        replace(_hair_profile(), hair=broken).validate()


# --- Схема запроса: диффузионные пути ---


def test_maskless_arguments_match_endpoint_schema(client):
    refine.run(_request(), _multi())

    args = client.arguments
    assert client.model == _multi().endpoint
    assert len(args["image_urls"]) == 2
    assert args["prompt"] == _multi().prompt()
    assert args["guidance_scale"] == _multi().guidance_scale
    assert args["output_format"] == "png"
    # Ровно четыре ключа — больше эндпоинт не принимает
    assert set(args) == {"image_urls", "prompt", "guidance_scale", "output_format"}


def test_the_template_goes_first_and_the_photo_second(client):
    """
    Порядок в массиве — часть контракта с промптом: инструкция говорит «первая
    картинка — сцена, вторая — личность». Перепутать их местами значит
    перерисовать фотографию заказчика по мотивам обложки, причём молча.
    """
    refine.run(_request(), _multi())

    assert client.arguments["image_urls"] == ["https://cdn/1", "https://cdn/2"]
    assert client.uploads[0] == (_TEMPLATE_PNG, "image/png")
    assert client.uploads[1] == (b"photo-bytes", "image/jpeg")


def test_the_mask_never_leaves_the_service(client):
    """
    Маска строится, но эндпоинт её не принимает: ключ mask_url означает 422 на
    весь запрос. Работает она локально, на вклейке.
    """
    refine.run(_request(), _multi())

    assert "mask_url" not in client.arguments
    assert len(client.uploads) == 2, "в CDN уезжают шаблон и фотография, и только"
    assert all(data != _MASK_PNG for data, _ in client.uploads)


@pytest.mark.parametrize(
    "key",
    ["mask_url", "strength", "num_inference_steps", "image_url", "ip_adapters", "negative_prompt"],
)
def test_arguments_carry_no_unsupported_keys(client, key):
    """
    Ключей вне схемы эндпоинта быть не должно: лишний параметр он не игнорирует,
    а заворачивает весь запрос. Отрицания идут прямо в промпт.
    """
    refine.run(_request(), _multi())

    assert key not in client.arguments


def test_masked_schema_still_sends_all_of_its_keys(client):
    """Прежняя схема не должна пострадать от того, что новая короче."""
    refine.run(_request(), _inpaint())

    args = client.arguments
    assert args["image_url"] and args["mask_url"] and args["reference_image_url"]
    assert args["strength"] == _inpaint().strength
    assert args["num_inference_steps"] == _inpaint().steps
    assert "image_urls" not in args


def test_payload_schema_switches_the_field_names(client):
    """
    Ради этого схема и вынесена в данные: другой эндпоинт — другой способ
    подмешивать личность, и стратегия об этом знать не обязана.
    """
    profile = replace(_inpaint(), payload=profiles.FLUX_GENERAL)
    refine.run(_request(), profile)

    adapters = client.arguments["ip_adapters"]
    assert adapters == [{"image_url": "https://cdn/3", "scale": profile.identity_scale}]
    assert "reference_image_url" not in client.arguments


def test_missing_mask_fails_before_network(client):
    """
    В fal маска не уезжает, но без неё нечем взять из ответа одну голову — а
    вернуть весь сгенерированный кадр значит отдать в печать перерисованные
    ткань и фон.
    """
    with pytest.raises(fal_api.MaskMissingError) as exc_info:
        refine.run(_request(mask=b""), _multi())

    assert exc_info.value.status_code == 500
    assert client.uploads == [], "до загрузки в CDN дойти не должно"
    assert client.arguments is None, "до вызова модели дойти не должно"


@pytest.mark.parametrize(
    ("profile_factory", "wrong_payload"),
    [(_multi, profiles.KONTEXT), (_profile, profiles.KONTEXT_MULTI)],
)
def test_wrong_schema_fails_before_network(client, profile_factory, wrong_payload):
    """
    Профиль, собранный из чужих частей: стратегия одна, схема от другой.
    Эндпоинт завернёт такой запрос — но уже после загрузок в CDN.
    """
    with pytest.raises(InvalidImageError):
        refine.run(_request(), replace(profile_factory(), payload=wrong_payload))

    assert client.uploads == []


def test_one_call_per_order(client, render):
    """
    Прежняя схема стоила до пяти вызовов: фон, шея, фактура, стык, сведение.
    Нынешняя обходится одним — и это половина её смысла.
    """
    refine.run(_request(), _profile())

    assert len(render.calls) == 1
    assert client.calls == [], "рабочий путь до fal не доходит вовсе"


def test_jpeg_alias(client):
    refine.run(_request(output_format="jpg"), _multi())

    # Эндпоинт знает только jpeg, но наружу принимаем и jpg
    assert client.arguments["output_format"] == "jpeg"


# --- Рабочий путь: свой GPU-сервер ---


def test_the_working_path_never_touches_fal(client, render):
    """
    Главное свойство рабочего пути, ради которого он и заводился: ни одного
    обращения к fal — ни к клиенту, ни к CDN, ни к профилю с его эндпоинтом.
    """
    refine.run(_request(), _profile())

    assert client.uploads == [] and client.calls == []
    assert render.urls == ["http://gpu-box:8300/v1/demo-render"]


def test_only_two_pictures_and_a_format_leave_the_service(render):
    """
    Контракт /v1/demo-render — шаблон и СЫРОЕ фото. Маска, кроп донора и число
    шагов считаются на той стороне тем же кодом, и слать их отсюда значило бы
    завести второй источник тех же чисел.
    """
    refine.run(_request(), _profile())

    sent = render.calls[0]
    assert set(sent) == {"base_image", "donor_photo", "output_format"}
    assert base64.b64decode(sent["base_image"]) == _TEMPLATE_PNG
    assert base64.b64decode(sent["donor_photo"]) == b"photo-bytes"


def test_the_photo_goes_as_the_customer_sent_it(render):
    """
    Ни поля вокруг, ни пережатия: сервер сам ищет лицо и строит кроп, а кайма
    его детектору только мешает — это уже стоило прогонов на прежнем пути.
    """
    refine.run(_request(identity=b"exactly-these-bytes"), _profile())

    assert base64.b64decode(render.calls[0]["donor_photo"]) == b"exactly-these-bytes"


def test_the_mask_is_not_built_and_not_sent(render):
    """Пайплайн её не строит (needs_mask=False), стратегия — не читает."""
    refine.run(_request(mask=b""), _profile())

    assert "mask" not in str(render.calls[0].keys())


def test_jpeg_alias_reaches_the_server(render):
    refine.run(_request(output_format="jpg"), _profile())

    assert render.calls[0]["output_format"] == "jpg"


def test_without_an_address_the_order_is_refused_before_the_network(monkeypatch, render):
    """
    Умолчания у адреса нет намеренно. Localhost по умолчанию означал бы, что
    ненастроенный сервис молча ходит в никуда и падает таймаутом на первом
    заказе — ровно та беда, из-за которой убрали умолчание «fal включён».
    """
    monkeypatch.setattr(local_render.settings, "render_base_url", "")

    with pytest.raises(local_render.RenderNotConfiguredError) as exc_info:
        refine.run(_request(), _profile())

    assert exc_info.value.status_code == 503
    assert "ML_RENDER_BASE_URL" in exc_info.value.message
    assert render.calls == [], "до сети дойти не должно"


def test_a_server_that_found_no_face_answers_422(render):
    """
    Тот же код, что и у молчаливого отказа фейссвопа, и это не совпадение:
    422 чинится другой фотографией, а не повтором запроса. Ответить здесь 502
    значило бы отправить заказ в очередь повторов, где он умрёт трижды.
    """
    render.response = _FakeResponse(422, None, text="на шаблоне не найден персонаж")

    with pytest.raises(NoFaceDetectedError) as exc_info:
        refine.run(_request(), _profile())

    assert exc_info.value.status_code == 422
    assert "персонаж" in exc_info.value.details["detail"]


def test_an_unreachable_server_answers_502(render):
    """Туннель упал или бокс перезагружается — это отказ шлюза, а не заказа."""
    render.error = local_render.requests.ConnectionError("нет соединения")

    with pytest.raises(local_render.RenderRequestFailedError) as exc_info:
        refine.run(_request(), _profile())

    assert exc_info.value.status_code == 502
    assert exc_info.value.details["error"] == "ConnectionError"


def test_a_server_error_carries_its_reason_but_not_its_traceback(render):
    """
    Причина нужна в деталях, а трейсбек на десятки килобайт — нет: он уедет в
    JSON ошибки Node и утонет в логах.
    """
    render.response = _FakeResponse(500, None, text="Ы" * 5000)

    with pytest.raises(local_render.RenderRequestFailedError) as exc_info:
        refine.run(_request(), _profile())

    assert exc_info.value.details["status_code"] == 500
    assert len(exc_info.value.details["detail"]) <= 300


def test_an_answer_without_a_picture_is_an_error_not_an_empty_frame(render):
    """
    Пустой ответ обязан быть отказом. Пустые байты, дошедшие до Node, стали бы
    «успешным» заказом с нечитаемым файлом — и выяснилось бы это на печати.
    """
    render.response = _FakeResponse(200, {"meta": {"steps": 8}})

    with pytest.raises(local_render.RenderRequestFailedError) as exc_info:
        refine.run(_request(), _profile())

    assert exc_info.value.details["keys"] == ["meta"]


def test_a_schema_from_another_endpoint_never_reaches_the_server(render):
    """Профиль, собранный из чужих частей, дешевле завернуть до отправки 30 МБ."""
    with pytest.raises(InvalidImageError):
        refine.run(_request(), replace(_profile(), payload=profiles.FACE_SWAP))

    assert render.calls == []


# --- Что возвращается наружу ---


def test_the_face_swap_returns_the_endpoint_bytes_untouched(client):
    """
    Локальной вклейки здесь нет: фон эндпоинт сохраняет сам, а лишнее
    перекодирование стоило бы ровно той сохранности, ради которой сюда шли.
    """
    result = refine.run(_request(), _profile())

    assert result.image == _GENERATED_PNG
    assert result.meta["mime_type"] == "image/png"


def test_the_diffusion_result_is_the_template_with_only_the_head_replaced(client):
    """
    Смысл вклейки. Модель вернула кадр целиком, но наружу уходит шаблон, в
    котором заменена ровно голова: фон и одежда — исходные пиксели разворота.
    """
    result = refine.run(_request(), _multi())
    image = decode_image(result.image)

    assert tuple(image[16, 16]) == (40, 40, 40), "под маской — генерация"
    assert tuple(image[1, 1]) == (200, 200, 200), "вне маски — шаблон"


def test_meta_reports_the_whole_profile(client):
    result = refine.run(_request(), _profile())

    assert result.meta["profile"] == "pixar_real"
    assert result.meta["strategy"] == "face_swap"
    assert result.meta["payload"] == "demo_render"
    assert result.meta["needs_mask"] is False
    assert result.meta["render_url"] == "http://gpu-box:8300/v1/demo-render"
    # Мета сервера едет целиком и под своим ключом: там числа пересадки, по
    # которым разбирают кадр, и придумывать им новые имена здесь незачем
    assert result.meta["render"]["steps"] == 8
    assert result.meta["render"]["matched"] == 1


def test_meta_separates_the_generation_from_the_result(client):
    """
    Ссылка fal ведёт на кадр ДО вклейки — с перерисованными фоном и одеждой.
    Назвать её image_url значит однажды отправить в печать не тот файл.
    """
    result = refine.run(_request(), _multi())

    assert result.meta["generated_url"] == "https://cdn/out1.png"
    assert "image_url" not in result.meta
    assert result.meta["composite"]["mask_open_px"] == 16 * 16
