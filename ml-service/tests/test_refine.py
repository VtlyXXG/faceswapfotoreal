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

    def upload(self, data, content_type):
        self.uploads.append((data, content_type))
        return f"https://cdn/{len(self.uploads)}"

    def subscribe(self, model, arguments, with_logs=False):
        self.model = model
        self.arguments = arguments
        return {"images": [{"url": "https://cdn/out.png"}], "seed": 7}


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


def _request(collage_image=None, **overrides) -> refine.RefineRequest:
    kwargs = {
        "collage": b"collage-bytes",
        "collage_mime": "image/png",
        "reference": b"source-bytes",
        "reference_mime": "image/jpeg",
        "mask": b"mask-bytes",
        "collage_image": collage_image,
    }
    kwargs.update(overrides)
    return refine.RefineRequest(**kwargs)


def _controlnet_profile() -> profiles.RefineProfile:
    """Профиль с картами и эндпоинтом, который их принимает."""
    return replace(profiles.get("stylise_controlnet"), endpoint="fal-ai/with-controlnet")


# --- Профили: числа и их связность ---


def test_presets_cover_both_regimes():
    """
    Консервативный режим не выброшен, а лежит рядом: если стилизация на 0.5
    окажется слишком вольной, откат — это смена имени профиля, а не ревёрт.
    """
    assert {"seam", "stylise", "stylise_controlnet"} <= set(profiles.available())

    assert profiles.get("seam").strength == 0.20
    assert profiles.get("seam").mask.gradient_ratio == 0.0, "плато, как было"


def test_stylise_sits_in_the_requested_band():
    """
    Диапазон 0.45-0.55 — то, ради чего второй шаг переделывали. На этой силе
    модель действительно перерисовывает открытое маской, а не подкрашивает.
    """
    stylise = profiles.get("stylise")

    assert 0.45 <= stylise.strength <= 0.55
    assert stylise.mask.gradient_ratio > 0, "на такой силе плато даёт ступеньку"
    # Предупреждение не должно срабатывать внутри рабочего диапазона
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
    assert profiles.from_settings().report()["profile"] == "stylise"


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
    result = refine.run(_request(collage), replace(profiles.get("stylise"), strategy="spy"))

    assert result.image == b"SPY"
    assert seen["profile"] == "stylise"
    assert client.arguments is None, "до fal дойти не должно"


def test_identity_embedding_is_declared_but_not_implemented(collage):
    """
    Проброс лицевых эмбеддингов — второй подход, под который заложен контракт.
    Пока эндпоинта нет, честнее отдать 501, чем молча отработать инпейнтингом:
    молчаливая подмена обнаружилась бы уже на печати тиража.
    """
    profile = replace(profiles.get("stylise"), strategy="identity_embedding")

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
    profile = replace(profiles.get("stylise"), strategy="identity_embedding")

    with pytest.raises(refine.RefinerNotSupportedError) as exc_info:
        refine.run(_request(collage, identity=identity), profile)

    assert exc_info.value.details["identity_present"] is True


def test_unknown_strategy_is_refused(collage):
    with pytest.raises(refine.RefinerNotSupportedError):
        refine.run(_request(collage), replace(profiles.get("stylise"), strategy="телепатия"))


# --- Схема запроса ---


def test_arguments_match_endpoint_schema(client, collage):
    refine.run(_request(collage), profiles.get("stylise"))

    args = client.arguments
    assert client.model == profiles.get("stylise").endpoint
    # Три обязательные ссылки эндпоинта
    assert args["image_url"] and args["mask_url"] and args["reference_image_url"]
    assert args["prompt"] == profiles.get("stylise").prompt
    assert args["strength"] == 0.50
    assert args["guidance_scale"] == profiles.get("stylise").guidance_scale
    assert args["num_inference_steps"] == profiles.get("stylise").steps


def test_collage_goes_first_and_reference_second(client, collage):
    """
    Порядок ссылок важен: под инпейнтинг идёт коллаж, фотография — только
    референс. Перепутать их местами — значит вернуться к прежней схеме, где
    лицо рисовалось с нуля, причём молча.
    """
    refine.run(_request(collage), profiles.get("stylise"))

    assert client.arguments["image_url"] == "https://cdn/1", "первым загружается коллаж"
    assert client.arguments["reference_image_url"] == "https://cdn/2", "вторым — фотография"
    assert client.uploads[0] == (b"collage-bytes", "image/png")


@pytest.mark.parametrize("key", ["ip_adapter_scale", "ip_adapters", "negative_prompt"])
def test_arguments_carry_no_unsupported_keys(client, collage, key):
    """
    Ключей вне схемы эндпоинта быть не должно: лишний параметр он не игнорирует,
    а заворачивает весь запрос. Отрицания идут прямо в промпт.
    """
    refine.run(_request(collage), profiles.get("stylise"))

    assert key not in client.arguments


def test_no_control_key_without_controls(client, collage):
    """
    Пустой список карт — такой же лишний ключ, как и полный. У эндпоинта без
    ControlNet его быть не должно вовсе.
    """
    refine.run(_request(collage), profiles.get("stylise"))

    assert "controlnets" not in client.arguments


def test_missing_mask_fails_before_network(client, collage):
    with pytest.raises(fal_api.MaskMissingError) as exc_info:
        refine.run(_request(collage, mask=None), profiles.get("stylise"))

    assert exc_info.value.status_code == 500
    assert client.uploads == [], "до загрузки в CDN дойти не должно"
    assert client.arguments is None, "до вызова модели дойти не должно"


def test_jpeg_alias(client, collage):
    refine.run(_request(collage, output_format="jpg"), profiles.get("stylise"))

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

    # Карта Canny уезжает четвёртой загрузкой, сразу после маски
    canny_png = client.uploads[3][0]
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

    depth_source = decode_image(client.uploads[4][0])
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
