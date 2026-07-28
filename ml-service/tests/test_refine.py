"""
Второй шаг: профили гиперпараметров, реестр стратегий и схема запроса.

Проверяется три разных вещи, и путать их не стоит:

  * **профиль** — числа и их связность. Несобираемая связка обязана падать до
    сети, а не после трёх загрузок в CDN;
  * **реестр** — что подход переключается именем, а не правкой pipeline.py.
    Ради этого вся конструкция и затевалась;
  * **схема запроса** — что уезжает в fal. Фиксируется тестами намеренно: на
    ней уже обжигались вживую, лишний ключ заворачивает весь запрос.
"""

from dataclasses import replace

import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.pipelines import fal_api, refine
from app.pipelines.refine import controls as control_maps
from app.pipelines.refine import profiles


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
        return {"images": [{"url": f"https://cdn/out{len(self.calls)}.png"}], "seed": 7}


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(fal_api, "client", lambda: fake)
    monkeypatch.setattr(fal_api, "_download", lambda image: b"PNGDATA")
    return fake


@pytest.fixture
def collage() -> np.ndarray:
    """Коллаж с контрастной границей — иначе Canny нечего находить."""
    image = np.full((64, 64, 3), 40, dtype=np.uint8)
    image[16:48, 16:48] = 220
    return image


def _request(collage_image=None, masks=None, **overrides) -> refine.RefineRequest:
    kwargs = {
        "collage": b"collage-bytes",
        "collage_mime": "image/png",
        "reference": b"source-bytes",
        "reference_mime": "image/jpeg",
        "masks": {"seam": b"seam-mask"} if masks is None else masks,
        "collage_image": collage_image,
    }
    kwargs.update(overrides)
    return refine.RefineRequest(**kwargs)


def _seam_only() -> profiles.RefineProfile:
    """Профиль без прохода по фону: один вызов, как было до трёх зон."""
    return replace(profiles.get("blend"), background=None)


def _controlnet_profile() -> profiles.RefineProfile:
    """Профиль с картами и эндпоинтом, который их принимает."""
    return replace(profiles.get("stylise_controlnet"), endpoint="fal-ai/with-controlnet")


# --- Профили: числа и их связность ---


def test_presets_cover_every_regime():
    """
    Консервативный режим не выброшен, а лежит рядом: если рабочий окажется
    слишком вольным, откат — это смена имени профиля, а не ревёрт.
    """
    assert {"seam", "blend", "stylise", "stylise_controlnet"} <= set(profiles.available())

    assert profiles.get("seam").strength == 0.20
    assert profiles.get("seam").mask.gradient_ratio == 0.0, "плато, как было"


def test_default_profile_blends_the_seam_without_risking_the_hair():
    """
    Связка, ради которой профиль и заведён: градиентная маска сводит шею и
    контур волос мягко, а strength остаётся в безопасном диапазоне 0.25-0.28 —
    выше контур причёски плывёт, и без карт ControlNet удержать его нечем.
    """
    blend = profiles.get("blend")

    assert 0.25 <= blend.strength <= 0.28
    assert blend.mask.gradient_ratio > 0, "градиент — половина смысла профиля"
    # Предупреждение начинается за верхней границей диапазона, а не внутри
    assert blend.safe_strength >= blend.strength
    assert blend.safe_strength <= 0.28


def test_default_profile_stays_on_the_working_endpoint():
    """
    На kontext-inpaint висит наша LoRA, и стиль обложек держится на ней. Уход с
    этого эндпоинта ради ControlNet стоил бы стиля — карт в дефолте нет
    осознанно, и вернуть их сюда молча не должно получиться.
    """
    blend = profiles.get("blend")

    assert blend.endpoint == "fal-ai/flux-kontext-lora/inpaint"
    assert blend.controls == ()


def test_stylise_sits_in_the_high_band():
    """
    Режим настоящей стилизации остался доступен: на 0.5 модель перерисовывает
    открытое маской. Дефолтом он не выбран — без карт на такой силе плывёт
    контур причёски.
    """
    stylise = profiles.get("stylise")

    assert 0.45 <= stylise.strength <= 0.55
    assert stylise.mask.gradient_ratio > 0, "на такой силе плато даёт ступеньку"
    assert stylise.safe_strength >= stylise.strength


def test_controlnet_profile_carries_canny_and_depth():
    controls = {control.kind: control for control in profiles.get("stylise_controlnet").controls}

    assert set(controls) == {"canny", "depth"}
    assert controls["canny"].source == "canny", "контур считается локально"
    assert controls["depth"].source == "image", "карту глубины считает эндпоинт"
    # Тени берутся в начале денойза, когда решается крупная форма
    assert controls["depth"].end <= controls["canny"].end


@pytest.mark.parametrize(
    "changes",
    [
        {"strength": 1.5},
        {"steps": 0},
        {"endpoint": ""},
        {"control_field": ""},
    ],
)
def test_broken_profile_is_refused_before_the_network(changes):
    """
    Профиль проверяется до сети. Иначе неверная связка стоит трёх загрузок в
    CDN и отказа fal — с текстом, по которому ничего не понять.
    """
    with pytest.raises(InvalidImageError):
        replace(_controlnet_profile(), **changes).validate()


def test_controlnet_on_an_endpoint_without_controlnet_is_refused():
    """
    Главная ловушка новой схемы: карты выглядят безобидной добавкой, но
    kontext-inpaint лишний ключ не игнорирует, а заворачивает весь запрос.
    Поймать это должен профиль, а не боевой прогон.
    """
    broken = replace(
        profiles.get("stylise_controlnet"), endpoint="fal-ai/flux-kontext-lora/inpaint"
    )

    with pytest.raises(InvalidImageError) as exc_info:
        broken.validate()

    assert "ControlNet" in exc_info.value.message


def test_default_profile_builds(monkeypatch):
    """Профиль по умолчанию обязан собираться без единой переменной окружения."""
    assert profiles.from_settings().report()["profile"] == "blend"


def test_environment_overrides_the_preset(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_profile", "seam")
    monkeypatch.setattr(profiles.settings, "refine_strength", 0.33)
    monkeypatch.setattr(profiles.settings, "refine_gradient_ratio", 0.2)

    profile = profiles.from_settings()

    assert profile.strength == 0.33
    assert profile.mask.gradient_ratio == 0.2
    assert profile.guidance_scale == profiles.get("seam").guidance_scale, "остальное из пресета"


def test_controls_can_be_switched_off_from_the_environment(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_profile", "stylise_controlnet")
    monkeypatch.setattr(profiles.settings, "refine_endpoint", "fal-ai/with-controlnet")
    monkeypatch.setattr(profiles.settings, "refine_controls", "none")

    assert profiles.from_settings().controls == ()


def test_controls_subset_is_selectable(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_profile", "stylise_controlnet")
    monkeypatch.setattr(profiles.settings, "refine_endpoint", "fal-ai/with-controlnet")
    monkeypatch.setattr(profiles.settings, "refine_controls", "canny")

    controls = profiles.from_settings().controls

    assert [control.kind for control in controls] == ["canny"]
    assert controls[0].weight == profiles.get("stylise_controlnet").controls[0].weight, (
        "вес остаётся из профиля: подбирать его строкой в окружении незачем"
    )


def test_unknown_control_in_the_environment_is_refused(monkeypatch):
    monkeypatch.setattr(profiles.settings, "refine_profile", "stylise_controlnet")
    monkeypatch.setattr(profiles.settings, "refine_controls", "canny,segmentation")

    with pytest.raises(InvalidImageError):
        profiles.from_settings()


def test_unknown_profile_name_is_refused():
    with pytest.raises(InvalidImageError):
        profiles.get("не-существует")


def test_blank_override_means_from_profile():
    """
    В .env.example ручки перечислены с пустыми значениями — так видно, что они
    есть. Пустая строка обязана означать «не задано»: иначе сервис падает на
    старте, разбирая конфиг, который оператор считает пустым.
    """
    from app.config import Settings

    blank = Settings(refine_strength="", refine_steps="", refine_gradient_ratio="")

    assert blank.refine_strength is None
    assert blank.refine_steps is None
    assert blank.refine_gradient_ratio is None


# --- Реестр стратегий: ради чего всё затевалось ---


def test_strategy_is_chosen_by_name_from_the_profile(client, collage):
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
    result = refine.run(_request(collage), replace(_seam_only(), strategy="spy"))

    assert result.image == b"SPY"
    assert seen["profile"] == "blend"
    assert client.arguments is None, "до fal дойти не должно"


def test_identity_embedding_is_declared_but_not_implemented(collage):
    """
    Проброс лицевых эмбеддингов — второй подход, под который заложен контракт.
    Пока эндпоинта нет, честнее отдать 501, чем молча отработать инпейнтингом:
    молчаливая подмена обнаружилась бы уже на печати тиража.
    """
    profile = replace(_seam_only(), strategy="identity_embedding")

    with pytest.raises(refine.RefinerNotSupportedError) as exc_info:
        refine.run(_request(collage), profile)

    assert exc_info.value.status_code == 501
    assert "identity_embedding" in refine.available()


def test_identity_reaches_the_strategy(collage):
    """
    Матрица эмбеддингов должна доезжать до стратегии, а не теряться по дороге:
    когда эндпоинт появится, первым вопросом будет именно этот.
    """
    identity = refine.Identity(embedding=np.zeros((1, 512), dtype=np.float32), model="buffalo_l")
    profile = replace(_seam_only(), strategy="identity_embedding")

    with pytest.raises(refine.RefinerNotSupportedError) as exc_info:
        refine.run(_request(collage, identity=identity), profile)

    assert exc_info.value.details["identity_present"] is True


def test_unknown_strategy_is_refused(collage):
    with pytest.raises(refine.RefinerNotSupportedError):
        refine.run(_request(collage), replace(_seam_only(), strategy="телепатия"))


# --- Схема запроса ---


def test_arguments_match_endpoint_schema(client, collage):
    refine.run(_request(collage), _seam_only())

    args = client.arguments
    assert client.model == _seam_only().endpoint
    # Три обязательные ссылки эндпоинта
    assert args["image_url"] and args["mask_url"] and args["reference_image_url"]
    assert args["prompt"] == _seam_only().prompt
    assert args["strength"] == _seam_only().strength
    assert args["guidance_scale"] == _seam_only().guidance_scale
    assert args["num_inference_steps"] == _seam_only().steps


def test_collage_goes_first_and_reference_second(client, collage):
    """
    Порядок ссылок важен: под инпейнтинг идёт коллаж, фотография — только
    референс. Перепутать их местами — значит вернуться к прежней схеме, где
    лицо рисовалось с нуля, причём молча.
    """
    refine.run(_request(collage), _seam_only())

    assert client.arguments["image_url"] == "https://cdn/1", "первым загружается коллаж"
    assert client.arguments["reference_image_url"] == "https://cdn/2", "вторым — фотография"
    assert client.uploads[0] == (b"collage-bytes", "image/png")


@pytest.mark.parametrize("key", ["ip_adapter_scale", "ip_adapters", "negative_prompt"])
def test_arguments_carry_no_unsupported_keys(client, collage, key):
    """
    Ключей вне схемы эндпоинта быть не должно: лишний параметр он не игнорирует,
    а заворачивает весь запрос. Отрицания идут прямо в промпт.
    """
    refine.run(_request(collage), _seam_only())

    assert key not in client.arguments


def test_no_control_key_without_controls(client, collage):
    """
    Пустой список карт — такой же лишний ключ, как и полный. У эндпоинта без
    ControlNet его быть не должно вовсе.
    """
    refine.run(_request(collage), _seam_only())

    assert "controlnets" not in client.arguments


def test_missing_mask_fails_before_network(client, collage):
    with pytest.raises(fal_api.MaskMissingError) as exc_info:
        refine.run(_request(collage, masks={}), _seam_only())

    assert exc_info.value.status_code == 500
    assert client.uploads == [], "до загрузки в CDN дойти не должно"
    assert client.arguments is None, "до вызова модели дойти не должно"


def test_jpeg_alias(client, collage):
    refine.run(_request(collage, output_format="jpg"), _seam_only())

    # Эндпоинт знает только jpeg, но наружу принимаем и jpg
    assert client.arguments["output_format"] == "jpeg"


# --- ControlNet ---


def test_control_maps_are_uploaded_and_described(client, collage):
    refine.run(_request(collage), _controlnet_profile())

    maps = client.arguments["controlnets"]
    assert [m["control_type"] for m in maps] == ["canny", "depth"]
    assert all(m["control_image_url"] for m in maps)
    assert maps[0]["conditioning_scale"] == 0.65
    assert maps[0]["start_percentage"] == 0.0 and maps[0]["end_percentage"] == 0.8
    # Коллаж, референс, маска и две карты
    assert len(client.uploads) == 5


def test_canny_map_is_computed_locally(client, collage):
    """
    Контур считается здесь, а не эндпоинтом: пороги — гиперпараметр, и
    подбирать их вслепую на чужой стороне невозможно.
    """
    refine.run(_request(collage), _controlnet_profile())

    from app.utils.image import decode_image

    # Порядок загрузок: коллаж, референс, карты, маски проходов
    canny_png = client.uploads[2][0]
    edges = decode_image(canny_png)[..., 0]
    assert set(np.unique(edges)) <= {0, 255}, "контурная карта бинарна"
    assert edges.any(), "на границе квадрата контур обязан найтись"


def test_depth_map_sends_the_collage_itself(client, collage):
    """
    Локального инференса глубины у сервиса нет — MiDaS потянул бы torch. Карту
    считает эндпоинт, а мы отдаём ему кадр как есть.
    """
    refine.run(_request(collage), _controlnet_profile())

    from app.utils.image import decode_image

    depth_source = decode_image(client.uploads[3][0])
    assert np.array_equal(depth_source, collage)


def test_controls_are_skipped_without_the_collage_array(client):
    """
    Карты строятся по пикселям. Без них запрос всё равно должен уехать — но с
    предупреждением, а не с молчаливым отказом от ControlNet.
    """
    refine.run(_request(collage_image=None), _controlnet_profile())

    assert "controlnets" not in client.arguments
    assert len(client.uploads) == 3


def test_canny_thresholds_are_validated(collage):
    with pytest.raises(InvalidImageError):
        control_maps.canny(collage, low=200, high=100)


def test_meta_reports_the_whole_profile(client, collage):
    result = refine.run(_request(collage), _controlnet_profile())

    assert result.meta["profile"] == "stylise_controlnet"
    assert result.meta["strategy"] == "inpaint_controlnet"
    assert result.meta["strength"] == 0.50
    assert result.meta["controls"] == ["canny:0.65", "depth:0.45"]
    assert result.meta["controls_sent"] == 2
    assert result.meta["seed"] == 7
    assert result.meta["mime_type"] == "image/png"


# --- Три зоны: два прохода с разной силой ---


def test_background_zone_runs_first_and_stronger(client, collage):
    """
    Ради этого зоны и разделили. У героя обложки грива до плеч, у заказчика
    ёжик, и вокруг вклейки остаётся кусок стёртого неба. Сводить его нечем —
    там нет содержимого, его надо сгенерировать, а это другая сила.

    Порядок обязателен: стык сводит вклейку с тем, что вокруг, и «вокруг» к
    этому моменту должно быть уже нарисовано.
    """
    profile = profiles.get("blend")
    refine.run(_request(collage, masks={"seam": b"seam", "background": b"hole"}), profile)

    first, second = client.calls
    assert first["strength"] == profile.background.strength >= 0.8
    assert second["strength"] == profile.strength <= 0.28
    assert first["prompt"] != second["prompt"], "у зон разная работа и разный промпт"


def test_second_pass_works_on_the_result_of_the_first(client, collage):
    """
    Иначе второй проход сводил бы края с тем мылом, которое первый только что
    заменил живописью, — и оба вызова были бы оплачены впустую.
    """
    refine.run(
        _request(collage, masks={"seam": b"seam", "background": b"hole"}), profiles.get("blend")
    )

    first, second = client.calls
    assert second["image_url"] == "https://cdn/out1.png"
    assert first["image_url"] != second["image_url"]


def test_zones_get_their_own_masks(client, collage):
    refine.run(
        _request(collage, masks={"seam": b"seam", "background": b"hole"}), profiles.get("blend")
    )

    first, second = client.calls
    assert first["mask_url"] != second["mask_url"], "у зон разные маски"
    assert (b"hole", "image/png") in client.uploads
    assert (b"seam", "image/png") in client.uploads


def test_without_a_hole_there_is_only_one_call(client, collage):
    """
    У персонажа со стрижкой дыры почти нет. Второй вызов стоит денег и времени,
    и платить за него не за что.
    """
    refine.run(_request(collage, masks={"seam": b"seam"}), profiles.get("blend"))

    assert len(client.calls) == 1
    assert client.calls[0]["strength"] == profiles.get("blend").strength


def test_background_weaker_than_the_seam_is_refused():
    """
    Проход по фону слабее прохода по стыку — это не настройка, а бессмыслица:
    ради генерации фона второй вызов и оплачивается.
    """
    profile = profiles.get("blend")
    broken = replace(profile, background=replace(profile.background, strength=0.1))

    with pytest.raises(InvalidImageError):
        broken.validate()


def test_meta_reports_both_zones(client, collage):
    result = refine.run(
        _request(collage, masks={"seam": b"seam", "background": b"hole"}), profiles.get("blend")
    )

    assert result.meta["zones"] == ["background", "seam"]
    assert result.meta["background_strength"] >= 0.8


# --- Зона 4: стилизация вклейки ---


def test_stylise_pass_runs_between_background_and_seam(client, collage):
    """
    Порядок из трёх проходов не произволен: фактура ложится на уже
    восстановленный фон, но до сведения стыка — иначе стык пришлось бы сводить
    дважды, второй раз поверх свежих мазков.
    """
    profile = profiles.get("blend")
    refine.run(
        _request(collage, masks={"seam": b"s", "background": b"b", "paste": b"p"}), profile
    )

    assert [a["strength"] for a in client.calls] == [
        profile.background.strength,
        profile.stylise.strength,
        profile.strength,
    ]


def test_stylise_sits_between_the_other_two_in_strength():
    """
    Компромисс: слишком слабо — фотография остаётся фотографией, слишком
    сильно — плывут черты. Между сведением стыка и генерацией фона.
    """
    profile = profiles.get("blend")

    assert profile.strength < profile.stylise.strength < profile.background.strength


def test_stylise_prompt_asks_for_paint_and_forbids_redrawing():
    """
    Промпт зоны говорит про фактуру и прямо запрещает двигать черты: на 0.35
    модель уже способна перерисовать лицо, и напоминание тут не лишнее.
    """
    prompt = profiles.get("blend").stylise.prompt.lower()

    assert "brush" in prompt and "canvas" in prompt
    assert "recognisable person" in prompt
    assert "do not redraw" in prompt


def test_without_a_paste_mask_stylisation_is_skipped(client, collage):
    """Нет маски — нет вызова: платить за проход, которому негде работать, незачем."""
    refine.run(_request(collage, masks={"seam": b"s"}), profiles.get("blend"))

    assert len(client.calls) == 1


def test_zones_must_overlap():
    """
    Отступ горячей зоны больше внешней половины кольца — значит, между ними
    полоса, которую не трогает ни один проход. Ровно она выглядела грязным
    контуром вокруг головы.
    """
    profile = profiles.get("blend")
    # До зоны фона дотягивается не сплошное кольцо, а градиент за ним —
    # проверка считает их вместе
    reach = profile.mask.edge_outer_ratio + profile.mask.gradient_ratio
    broken = replace(profile, mask=replace(profile.mask, hole_margin_ratio=reach))

    with pytest.raises(InvalidImageError):
        broken.validate()

    ok = replace(profile, mask=replace(profile.mask, hole_margin_ratio=reach * 0.9))
    assert ok.validate() is ok, "перекрытие через градиент допустимо"


def test_neck_pass_runs_right_after_the_background(client, collage):
    """
    Шея — такая же генеративная работа, как фон, и её результат должен попасть
    под последующее сведение стыка, а не наоборот.
    """
    profile = profiles.get("blend")
    refine.run(
        _request(collage, masks={"seam": b"s", "background": b"b", "neck": b"n", "paste": b"p"}),
        profile,
    )

    assert [a["strength"] for a in client.calls] == [
        profile.background.strength,
        profile.neck.strength,
        profile.stylise.strength,
        profile.strength,
    ]


def test_neck_prompt_asks_to_continue_the_chin():
    """
    Зона рисуется с нуля, и единственный ориентир по тону — подбородок сверху.
    Про него в промпте сказано прямо.
    """
    prompt = profiles.get("blend").neck.prompt.lower()

    assert "chin" in prompt and "oil" in prompt
    assert "no seam" in prompt
