"""
Карты управления для ControlNet: чем модель удерживается от отсебятины.

На strength 0.5 инпейнтинг перерисовывает открытое маской по-настоящему, и без
подсказок он перерисует заодно черты лица. Карта управления — это второй канал
входа: изображение, которому модель обязана следовать по структуре, независимо
от того, сколько шума насыпали сверху.

Две карты и разные роли:

  canny — контурный рисунок. Держит линию челюсти, разрез глаз, контур
    причёски. Считается локально, cv2.Canny, пороги — в профиле.
  depth — карта объёма. Отвечает за тени: по ней видно, что голова выступает
    над плечами, и контактная тень ложится под подбородок, а не куда придётся
    по промпту. Локального инференса глубины у сервиса нет — MiDaS потянул бы
    torch, а от локальных весов уходили осознанно, — поэтому в эндпоинт уезжает
    сам коллаж, а карту считает он.

Отсюда и поле `source` у ControlSpec: `canny` — построить здесь, `image` —
отдать кадр как есть. Список карт наращивается профилем, а не кодом: новый
препроцессор появляется здесь одной функцией.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import InvalidImageError
from app.pipelines.refine.profiles import ControlSpec


def canny(image: Any, low: int, high: int) -> Any:
    """
    Контурный рисунок коллажа.

    Размытие перед детектором обязательно: фотографическая часть коллажа шумит
    на порядок сильнее живописной, и без него Canny выдаёт по коже сыпь из
    коротких штрихов, которую ControlNet честно попытается воспроизвести.

    :return: одноканальный uint8, белые контуры на чёрном
    """
    import cv2

    if not 0 <= low < high <= 255:
        raise InvalidImageError(
            "Пороги Canny должны идти по возрастанию внутри 0..255",
            {"low": low, "high": high},
        )

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), low, high)


def build(spec: ControlSpec, image: Any) -> tuple[bytes, str]:
    """
    Строит карту для одной спецификации.

    :param image: коллаж, BGR numpy.ndarray
    :return: (PNG-байты карты, mime)
    """
    from app.utils.image import encode_image

    if spec.source == "canny":
        return encode_image(canny(image, spec.low, spec.high), "png")
    if spec.source == "image":
        return encode_image(image, "png")

    raise InvalidImageError(
        "Неизвестный способ построения карты управления",
        {"kind": spec.kind, "source": spec.source},
    )
