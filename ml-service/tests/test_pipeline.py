"""
Оркестрация двух шагов: что именно уходит в fal.

Тест ровно об одном — в облако должен уезжать коллаж, а не исходная обложка.
Перепутать их местами легко и совершенно незаметно: запрос уйдёт, картинка
вернётся, и только сходство окажется прежним, никаким.
"""

import numpy as np
import pytest

from app.pipelines import collage as collage_builder
from app.pipelines import fal_api
from app.pipelines.face_swap import pipeline
from app.utils.image import decode_image, encode_image

_COLLAGE_COLOUR = (17, 34, 51)


@pytest.fixture
def sent(monkeypatch) -> dict:
    """Подменяет оба тяжёлых шага и возвращает то, что ушло бы в fal."""
    cover = np.full((64, 64, 3), 200, dtype=np.uint8)
    faked = np.full((64, 64, 3), _COLLAGE_COLOUR, dtype=np.uint8)
    polygon = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)

    monkeypatch.setattr(
        collage_builder,
        "build",
        lambda *_, **__: collage_builder.Collage(
            image=faked,
            paste_polygon=polygon,
            face_polygon=polygon,
            meta={"scale": 0.5, "coverage": 0.98},
        ),
    )

    captured: dict = {}

    def _refine(**kwargs):
        captured.update(kwargs)
        return b"RESULT", {"model": "fal/test", "mime_type": "image/png"}

    monkeypatch.setattr(fal_api, "refine_collage", _refine)

    cover_png, _ = encode_image(cover, "png")
    source_png, _ = encode_image(np.zeros((32, 32, 3), dtype=np.uint8), "png")
    result = pipeline.run(pipeline.SwapRequest(source=source_png, target=cover_png))
    return {"captured": captured, "result": result}


def test_fal_receives_the_collage_not_the_cover(sent):
    uploaded = decode_image(sent["captured"]["collage"])

    assert tuple(uploaded[32, 32]) == _COLLAGE_COLOUR


def test_photo_goes_as_reference_only(sent):
    assert sent["captured"]["reference_mime"] == "image/png"
    assert sent["captured"]["mask"], "маска обязательна: без неё инпейнтинга нет"


def test_mask_matches_the_cover_frame(sent):
    mask = decode_image(sent["captured"]["mask"])

    assert mask.shape[:2] == (64, 64)
    assert mask.max() == 255, "зона обработки должна быть непустой"


def test_meta_carries_the_collage_report(sent):
    """
    Числа первого шага уезжают в X-Swap-Meta: масштаб и покрытие — первое, на
    что смотрят, когда результат вышел непохожим.
    """
    meta = sent["result"].meta

    assert meta["collage"]["coverage"] == 0.98
    assert meta["model"] == "fal/test"
    assert sent["result"].mime_type == "image/png"
