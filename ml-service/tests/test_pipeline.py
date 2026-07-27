"""
Оркестрация двух шагов: что именно уходит в fal.

Тест ровно об одном — в облако должна уезжать аппликация, а не исходная
обложка. Перепутать их местами легко и совершенно незаметно: запрос уйдёт,
картинка вернётся, и только сходство окажется прежним, никаким.
"""

import numpy as np
import pytest

from app.pipelines import collage as collage_builder
from app.pipelines import refine
from app.pipelines.face_swap import pipeline
from app.utils.image import decode_image, encode_image

_COLLAGE_COLOUR = (17, 34, 51)


@pytest.fixture
def sent(monkeypatch) -> dict:
    """Подменяет оба тяжёлых шага и возвращает то, что ушло бы в fal."""
    cover = np.full((64, 64, 3), 200, dtype=np.uint8)
    faked = np.full((64, 64, 3), _COLLAGE_COLOUR, dtype=np.uint8)

    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[10:50, 10:50] = 255
    polygon = np.array([[20, 20], [44, 20], [44, 44], [20, 44]], dtype=np.int32)

    captured: dict = {}

    def _build(*_, **kwargs):
        captured["emotion"] = kwargs.get("emotion")
        return collage_builder.Collage(
            image=faked,
            head_alpha=alpha,
            face_polygon=polygon,
            neck_line=((10, 52), (54, 52)),
            erased=np.zeros((64, 64), dtype=np.uint8),
            meta={"scale": 0.5, "emotion": "neutral", "face_height_target": 20.0},
        )

    def _run(request, profile):
        captured["request"] = request
        captured["profile"] = profile
        return refine.RefineResult(image=b"RESULT", meta={"model": "fal/test"})

    monkeypatch.setattr(collage_builder, "build", _build)
    monkeypatch.setattr(pipeline.refine, "run", _run)

    cover_png, _ = encode_image(cover, "png")
    source_png, _ = encode_image(np.zeros((32, 32, 3), dtype=np.uint8), "png")
    result = pipeline.run(
        pipeline.SwapRequest(source=source_png, target=cover_png, emotion="neutral")
    )
    return {"captured": captured, "result": result}


def test_fal_receives_the_collage_not_the_cover(sent):
    uploaded = decode_image(sent["captured"]["request"].collage)

    assert tuple(uploaded[32, 32]) == _COLLAGE_COLOUR


def test_photo_goes_as_reference_only(sent):
    request = sent["captured"]["request"]

    assert request.reference_mime == "image/png"
    assert request.masks.get("seam"), "маска стыка обязательна: без неё инпейнтинга нет"


def test_collage_goes_as_pixels_too(sent):
    """
    Карты управления строятся по пикселям коллажа. Декодировать PNG второй раз
    ради этого незачем — массив уезжает вместе с байтами.
    """
    assert sent["captured"]["request"].collage_image is not None


def test_mask_covers_the_seam_and_spares_the_face(sent):
    mask = decode_image(sent["captured"]["request"].masks["seam"])[..., 0]

    assert mask.shape == (64, 64)
    assert mask.max() == 255, "зона обработки должна быть непустой"
    assert mask[32, 32] == 0, "центр лица модели недоступен"


def test_mask_and_strength_come_from_one_profile(sent):
    """
    Ширина градиента маски и strength подбираются вместе. Если маску собирать
    по одним числам, а запрос по другим, разъедутся они молча — и объяснить
    результат будет нечем.
    """
    profile = sent["captured"]["profile"]

    assert profile.name == "blend"
    assert profile.mask.gradient_ratio > 0
    assert 0.25 <= profile.strength <= 0.28


def test_emotion_reaches_the_collage_step(sent):
    """Параметр эмоции обязан доезжать до точки расширения, а не теряться."""
    assert sent["captured"]["emotion"] == "neutral"


def test_meta_carries_the_collage_report(sent):
    """
    Числа первого шага уезжают в X-Swap-Meta: масштаб и мимика — первое, на
    что смотрят, когда результат вышел непохожим.
    """
    meta = sent["result"].meta

    assert meta["collage"]["scale"] == 0.5
    assert meta["collage"]["emotion"] == "neutral"
    assert meta["model"] == "fal/test"
    assert sent["result"].mime_type == "image/png"
