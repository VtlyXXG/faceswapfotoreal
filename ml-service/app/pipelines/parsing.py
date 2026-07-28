"""
Семантическая разметка человека: лицо, волосы, кожа тела, одежда.

Зачем понадобилась. Две задачи упирались в один и тот же вопрос — где на
нарисованном персонаже кожа. Подкрасить его руки и грудь под тон вклейки;
перерисовать его шею так, чтобы она срослась с подбородком донора. Обе требуют
отличать кожу от ткани, и обе по цвету не решаются: на наших обложках бежевый
жилет отстоит от тона лица персонажа на |Δa|=4, шип динозавра на 8, а
собственная рука персонажа — на 17. То есть по хроме ткань и декорация ближе к
коже, чем кожа. Связность силуэта не спасает тоже: сегментатор отдаёт персонажа
и динозавра одним компонентом на 100% площади.

Модель отвечает на этот вопрос прямо. Проверено на наших разворотах: несмотря на
то, что учили её на фотографиях, масляную иллюстрацию она разбирает чисто — лицо,
причёска, шея, руки и ноги персонажа находятся отдельно друг от друга, а динозавр
в кожу не попадает.

Веса. Нового пакета не нужно: ImageSegmenter входит в mediapipe, который уже
стоит. Нужен файл модели (~16 МБ), он качается отдельно:

    python -c "import urllib.request,os; d=os.path.expanduser('~/.mediapipe');
    os.makedirs(d,exist_ok=True); urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
    'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
    d+'/selfie_multiclass_256x256.tflite')"

Без файла сервис работает: разметка возвращает None, а вызывающий код падает
обратно на прежние эвристики по цвету. Отказывать заказчику из-за отсутствия
необязательных весов было бы неправильно — до сих пор пайплайн обходился без них
вовсе.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Классы модели selfie_multiclass в порядке её выходных индексов.
_BACKGROUND, _HAIR, _SKIN, _FACE, _CLOTHES, _OTHER = range(6)


@dataclass
class Parsed:
    """
    Маски классов в координатах исходного кадра, uint8 0/255.

    :param face: кожа лица
    :param hair: волосы
    :param skin: кожа тела — шея, руки, грудь, ноги
    :param clothes: одежда
    """

    face: Any
    hair: Any
    skin: Any
    clothes: Any

    @property
    def bare_skin(self) -> Any:
        """Вся кожа персонажа: лицо и тело вместе."""
        import numpy as np

        return np.maximum(self.face, self.skin)


_SEGMENTER: Any = None
_MISSING_REPORTED = False


def model_path() -> Path:
    """Где лежит файл модели. ML_PARSING_MODEL или каталог mediapipe по умолчанию."""
    if settings.parsing_model:
        return Path(settings.parsing_model)
    home = os.environ.get("MEDIAPIPE_HOME") or Path.home() / ".mediapipe"
    return Path(home) / "selfie_multiclass_256x256.tflite"


def available() -> bool:
    return model_path().exists()


def _segmenter() -> Any:
    """
    Сегментатор на весь процесс. Кэшируется: создание графа стоит десятки
    миллисекунд, а воркер обрабатывает заказы один за другим.
    """
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER

    from mediapipe.tasks.python import BaseOptions, vision

    # Веса передаются буфером, а не путём: model_asset_path mediapipe разрешает
    # относительно каталога своего пакета, и абсолютный путь с обратными слэшами
    # превращается в site-packages плюс этот путь — файл не открывается вовсе.
    _SEGMENTER = vision.ImageSegmenter.create_from_options(
        vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=model_path().read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
    )
    log.info("разметка человека готова", extra={"model": str(model_path())})
    return _SEGMENTER


def parse(image: Any) -> Parsed | None:
    """
    Разбирает кадр на классы. None — весов нет или модель не отработала.

    Возвращать None, а не бросать, — намеренно: разметка это уточнение, а не
    условие работы. Вызывающий код обязан уметь без неё.

    :param image: BGR numpy.ndarray
    """
    global _MISSING_REPORTED

    if not available():
        if not _MISSING_REPORTED:
            # Один раз за процесс: иначе строка повторится на каждом заказе
            log.warning(
                "разметка человека недоступна — работаем по цветовым эвристикам",
                extra={"expected": str(model_path())},
            )
            _MISSING_REPORTED = True
        return None

    import cv2
    import mediapipe as mp
    import numpy as np

    try:
        frame = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        categories = _segmenter().segment(frame).category_mask.numpy_view()
    except Exception as exc:  # noqa: BLE001 — битые веса, отказ tflite на кадре
        log.warning("разметка человека не выполнена", extra={"cause": str(exc)})
        return None

    def mask(index: int) -> Any:
        return np.where(categories == index, 255, 0).astype(np.uint8)

    parsed = Parsed(
        face=mask(_FACE), hair=mask(_HAIR), skin=mask(_SKIN), clothes=mask(_CLOTHES)
    )
    log.info(
        "человек размечен",
        extra={
            "face_px": int(np.count_nonzero(parsed.face)),
            "hair_px": int(np.count_nonzero(parsed.hair)),
            "skin_px": int(np.count_nonzero(parsed.skin)),
            "clothes_px": int(np.count_nonzero(parsed.clothes)),
        },
    )
    return parsed


def reset() -> None:
    """Отпускает сегментатор. Нужен тестам: граф держит файл модели открытым."""
    global _SEGMENTER
    if _SEGMENTER is not None:
        _SEGMENTER.close()
        _SEGMENTER = None
