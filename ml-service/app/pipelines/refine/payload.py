"""
Сериализация пакета для собственного бэкенда Flux + PuLID.

Один JSON на кадр: очищенный шаблон, маска инпейнта, кроп лица донора, вектор
личности, промпт и веса. Отдельный модуль, потому что формат — это контракт с
чужим сервисом, и меняться он обязан осознанно, а не заодно с правкой транспорта.

**Формат на поле выбран по замеру, а не по единому правилу.**

`base_image` уезжает в JPEG: на нашем материале PNG весит 1.5 МБ против 0.21 МБ
у JPEG q95 — семикратная разница на каждом кадре, и это ровно та нагрузка на
канал, ради которой всё затевалось. Потери на q95 диффузия внутри маски всё
равно перепишет, а тон снаружи маски меряется по ИСХОДНОМУ окну (`composite`
получает нетронутую сцену), так что до готового разворота они не доезжают.

`mask_image` уезжает в PNG, и это не единообразия ради. Маска — не картинка, а
веса: по ней решается, какие пиксели переписать, а какие вернуть из шаблона
побитово. JPEG раскладывает её по блокам 8x8, и на замере той же маски q95 дал
287 пикселей ненулевого веса ТАМ, ГДЕ МАСКА БЫЛА ЧИСТЫМ НУЛЁМ, плюс просадку
255 → 254 внутри. Экономия при этом 59 КБ на фоне 210 КБ базового кадра — то
есть мы платили бы обещанием «вне маски шаблон не тронут» за полпроцента
трафика.

`donor_crop` уезжает в PNG по той же причине, только жёстче: 112x112 — это вход
лицевого энкодера, каждый пиксель там весит в векторе, а весь файл занимает
десятки килобайт. Сжимать нечего, портить есть что.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.errors import InvalidImageError
from app.utils.image import encode_image

if TYPE_CHECKING:
    from app.pipelines.refine.adapter import PuLIDWeights, SceneConditioning

# Качество базового кадра. Ниже 90 на коже проступает блочность, которую
# диффузия принимает за фактуру и honestly воспроизводит; выше 95 файл растёт
# быстрее, чем убывают артефакты.
JPEG_QUALITY = 95

# Порядок байт вектора задан явно. Numpy на нашей стороне и torch на стороне
# сервиса договориться сами не могут: '<f4' читается одинаково везде, а
# нативный порядок — только пока обе стороны на x86.
EMBEDDING_DTYPE = "<f4"
EMBEDDING_SIZE = 512


@dataclass(frozen=True)
class Encoded:
    """Закодированная картинка: base64, MIME и размер исходных байт."""

    data: str
    mime: str
    size: int


def encode_field(image: Any, fmt: str, quality: int | None = None) -> Encoded:
    """
    Кадр → base64. Размер отдаётся отдельно: по нему считается вес пакета.

    ПАМЯТЬ: на этом шаге кадр существует трижды — массив, сжатые байты и строка
    base64, которая ещё на треть длиннее байт. Для 4K это заметно, поэтому
    промежуточные байты не сохраняются нигде, кроме локальной переменной, и
    умирают на выходе из функции.
    """
    raw, mime = encode_image(image, fmt, quality)
    return Encoded(data=base64.b64encode(raw).decode("ascii"), mime=mime, size=len(raw))


def encode_embedding(embedding: Any) -> dict[str, Any]:
    """
    Вектор личности → base64 сырых байт с явными dtype и длиной.

    Длина проверяется здесь, а не на сервисе: вектор не той размерности — это
    дефект нашей экстракции, и ловить его надо до сети.
    """
    import numpy as np

    vector = np.ascontiguousarray(embedding, dtype=np.dtype(EMBEDDING_DTYPE))
    if vector.ndim != 1 or vector.size != EMBEDDING_SIZE:
        raise InvalidImageError(
            "Вектор личности не той формы",
            {"shape": list(vector.shape), "expected": [EMBEDDING_SIZE]},
        )

    norm = float(np.linalg.norm(vector))
    if not 0.99 <= norm <= 1.01:
        # Ненормированный вектор ломает косинусную близость, на которой стоит
        # PuLID, и сила влияния начинает зависеть от экспозиции донора
        raise InvalidImageError("Вектор личности не нормирован", {"norm": round(norm, 4)})

    return {
        "data": base64.b64encode(vector.tobytes()).decode("ascii"),
        "dtype": EMBEDDING_DTYPE,
        "size": EMBEDDING_SIZE,
    }


def build_packet(
    conditioning: SceneConditioning,
    *,
    quality: int = JPEG_QUALITY,
    negative_prompt: str = "",
) -> dict[str, Any]:
    """
    Собирает JSON-пакет одного кадра.

    Веса проверяются ДО кодирования: незачем жать полтора мегабайта, чтобы
    сервис отверг запрос из-за числа.
    """
    weights: PuLIDWeights = conditioning.weights
    weights.validate()

    if conditioning.sent is None:
        raise InvalidImageError(
            "Очищенный шаблон уже отпущен — пакет собирать не из чего",
            {"hint": "build_packet вызывается до release_sent"},
        )

    base = encode_field(conditioning.sent, "jpg", quality)
    mask = encode_field(conditioning.mask, "png")
    crop = encode_field(conditioning.identity.aligned, "png")

    return {
        "base_image": base.data,
        "mask_image": mask.data,
        "donor_crop": crop.data,
        "embedding": encode_embedding(conditioning.identity.embedding),
        "prompt": conditioning.prompt,
        "negative_prompt": negative_prompt,
        "fidelity": weights.fidelity,
        "steps": weights.steps,
        "start_step": weights.start_step,
        "guidance_scale": weights.guidance,
        "true_cfg": weights.true_cfg,
        # Форматы названы явно: разбирать по магическим байтам сервис, конечно,
        # умеет, но тогда наша ошибка «маска уехала JPEG-ом» станет его молчаливо
        # проглоченной проблемой вместо нашей явной
        "encoding": {
            "base_image": base.mime,
            "mask_image": mask.mime,
            "donor_crop": crop.mime,
        },
        "bytes": {"base_image": base.size, "mask_image": mask.size, "donor_crop": crop.size},
    }


def packet_weight(packet: dict[str, Any]) -> int:
    """Сколько сырых байт картинок в пакете. Для лога и для метрик канала."""
    return sum(packet.get("bytes", {}).values())
