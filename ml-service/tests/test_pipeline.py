"""
Оркестрация: что именно уезжает в fal и какая локальная работа этому предшествует.

Первое — в облако должны уехать ШАБЛОН и ФОТОГРАФИЯ, а не наоборот. Перепутать
их местами легко и совершенно незаметно: запрос уйдёт, картинка вернётся, и
только это будет уже не разворот книги.

Второе — локальная геометрия принадлежит диффузии и включается вместе с ней.
Рабочий фейссвоп находит область сам, а поле вокруг фотографии сбивает его
детектор; строить для него маску значит платить три секунды на 4K и получать
лишний повод отказать заказу.
"""

import numpy as np
import pytest

from app.pipelines import expression, head_mask, refine
from app.pipelines.face_swap import pipeline
from app.pipelines.refine import profiles
from app.utils.image import decode_image, encode_image

_TARGET_COLOUR = (200, 200, 200)


@pytest.fixture
def sent(monkeypatch) -> dict:
    """
    Подменяет маску и облачный вызов, возвращает то, что ушло бы в fal.

    Профиль здесь диффузионный: локальная геометрия — его работа, и проверять её
    надо на нём.
    """
    monkeypatch.setattr(profiles.settings, "refine_strategy", "kontext_multi")
    target = np.full((64, 64, 3), _TARGET_COLOUR, dtype=np.uint8)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255

    captured: dict = {}

    def _build(image, **kwargs):
        captured["mask_ratios"] = kwargs
        captured["mask_source_shape"] = image.shape
        return head_mask.HeadMask(mask=mask, face_height=20.0, meta={"source": "parsing"})

    def _run(request, profile):
        captured["request"] = request
        captured["profile"] = profile
        return refine.RefineResult(image=b"RESULT", meta={"model": "fal/test"})

    monkeypatch.setattr(head_mask, "build", _build)
    monkeypatch.setattr(pipeline.refine, "run", _run)

    target_png, _ = encode_image(target, "png")
    source_png, _ = encode_image(np.zeros((32, 32, 3), dtype=np.uint8), "png")
    result = pipeline.run(
        pipeline.SwapRequest(source=source_png, target=target_png, emotion="neutral")
    )
    return {"captured": captured, "result": result, "target": target_png, "source": source_png}


def test_the_template_goes_under_inpaint_untouched(sent):
    """
    Шаблон уезжает теми же байтами, что пришли: мы в нём ничего не меняли, а
    пережатие 4K-разворота стоило бы ровно той фактуры, которую модель просят
    повторить.
    """
    request = sent["captured"]["request"]

    assert request.target == sent["target"]
    assert tuple(decode_image(request.target)[32, 32]) == _TARGET_COLOUR


def test_the_photo_goes_as_the_identity_reference(sent):
    """
    Фотография уезжает референсом — но не теми же байтами, что пришли. Перед
    отправкой вокруг неё кладётся поле: портрет крупным планом задаёт генерации
    свой масштаб, и голова выходит больше маски. Подробности — reference.py.
    """
    request = sent["captured"]["request"]

    assert request.identity != sent["source"], "референс обязан пройти подготовку"
    assert request.identity_mime == "image/jpeg"

    # Фотография 32x32 внутри поля: холст обязан стать больше неё, а исходный
    # кадр — уцелеть целиком, потому что личность берётся именно из него
    canvas = decode_image(request.identity)
    assert canvas.shape[0] > 32 and canvas.shape[1] > 32


def test_the_reference_scale_is_reported(sent):
    """
    Масштаб референса — первое, на что смотрят, когда голова вышла не того
    размера. Значит, он обязан быть в мете рядом с мета-данными маски.
    """
    meta = sent["result"].meta

    assert "reference" in meta
    assert meta["reference"]["scale"] > 1.0


def test_mask_is_built_on_the_template(sent):
    """
    Ключевое отличие новой схемы: маска строится по ШАБЛОНУ, а не по донору.
    Фотография заказчика к геометрии кадра отношения не имеет вовсе.
    """
    assert sent["captured"]["mask_source_shape"] == (64, 64, 3)

    mask = decode_image(sent["captured"]["request"].mask)[..., 0]
    assert mask.shape == (64, 64)
    assert mask[32, 32] == 255, "голова персонажа обязана быть открыта модели"
    assert mask[2, 2] == 0, "остальной разворот трогать нельзя"


def test_mask_and_strength_come_from_one_profile(sent):
    """
    Доли маски и strength подбираются вместе. Если маску собирать по одним
    числам, а запрос по другим, разъедутся они молча — и объяснить результат
    будет нечем.
    """
    profile = sent["captured"]["profile"]
    ratios = sent["captured"]["mask_ratios"]

    assert ratios["dilate_ratio"] == profile.mask.dilate_ratio
    assert ratios["feather_ratio"] == profile.mask.feather_ratio
    assert ratios["neck_ratio"] == profile.mask.neck_ratio
    assert profile.strength >= profile.safe_strength


def test_neutral_emotion_says_nothing_about_the_expression(sent):
    """
    Умолчание новой схемы: выражение и поворот головы берутся с шаблона. Любое
    указание мимики здесь спорило бы с картинкой.
    """
    assert sent["captured"]["request"].expression == ""


def test_face_swap_skips_the_local_geometry(monkeypatch):
    """
    Рабочий путь не строит ни маску, ни поле вокруг фотографии. Маска ему не
    нужна — область эндпоинт находит своим детектором; кайма референса его
    детектору мешает. Три секунды на 4K и лишний `NoFaceDetectedError` — не та
    цена, которую стоит платить за неиспользуемое число.
    """
    captured: dict = {}

    monkeypatch.setattr(
        head_mask, "build", lambda *_, **__: pytest.fail("маска строиться не должна")
    )
    monkeypatch.setattr(
        pipeline.refine,
        "run",
        lambda request, profile: captured.setdefault("request", request)
        and refine.RefineResult(image=b"R", meta={}),
    )

    png, _ = encode_image(np.zeros((8, 8, 3), dtype=np.uint8), "png")
    result = pipeline.run(pipeline.SwapRequest(source=png, target=png))

    request = captured["request"]
    assert request.mask == b"", "маски нет — и стратегии она не нужна"
    assert request.identity == png, "фотография уезжает теми же байтами, что пришла"
    assert result.meta["mask"] is None
    assert result.meta["reference"]["reason"] == "disabled"


def test_explicit_emotion_reaches_the_prompt(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        head_mask,
        "build",
        lambda image, **_: head_mask.HeadMask(
            mask=np.full((8, 8), 255, dtype=np.uint8), face_height=4.0, meta={}
        ),
    )
    monkeypatch.setattr(
        pipeline.refine,
        "run",
        lambda request, profile: captured.setdefault("request", request)
        and refine.RefineResult(image=b"R", meta={}),
    )

    png, _ = encode_image(np.zeros((8, 8, 3), dtype=np.uint8), "png")
    pipeline.run(pipeline.SwapRequest(source=png, target=png, emotion="laugh"))

    assert "laughing" in captured["request"].expression


def test_unknown_emotion_is_refused_before_the_network(monkeypatch):
    """501 до единой загрузки в CDN: заказ с опечаткой не должен стоить денег."""
    monkeypatch.setattr(
        pipeline.refine,
        "run",
        lambda *_: pytest.fail("до fal дойти не должно"),
    )

    png, _ = encode_image(np.zeros((8, 8, 3), dtype=np.uint8), "png")

    with pytest.raises(expression.ExpressionNotSupportedError):
        pipeline.run(pipeline.SwapRequest(source=png, target=png, emotion="телепатия"))


def test_meta_carries_the_mask_report(sent):
    """
    Числа первого шага уезжают в X-Swap-Meta: чем построена маска и сколько
    кадра открыто — первое, на что смотрят, когда результат вышел странным.
    """
    meta = sent["result"].meta

    assert meta["mask"]["source"] == "parsing"
    assert meta["model"] == "fal/test"
    assert meta["emotion"] == "neutral"
    assert sent["result"].mime_type == "image/png"
