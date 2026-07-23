"""
Контракт стилизатора — стадии согласования лица с иллюстрацией.

Тот же приём, что у BaseTextProvider в Node-сервисе: пайплайн не знает,
работает ли под ним классический CV или диффузия, и его можно заменить,
не трогая вызывающий код.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StyleOptions:
    """Сила каждой стадии, 0.0 — стадия выключена, 1.0 — максимум."""

    color: float = 0.6
    smooth: float = 0.5
    grain: float = 0.7
    sharpness: float = 0.6
    margin: float = 1.8  # во сколько раз расширяем bbox лица под рабочий кроп

    def scaled(self, factor: float) -> StyleOptions:
        """Общий множитель силы для одного запроса."""
        factor = max(0.0, min(2.0, factor))
        return StyleOptions(
            color=min(1.0, self.color * factor),
            smooth=min(1.0, self.smooth * factor),
            grain=min(1.0, self.grain * factor),
            sharpness=min(1.0, self.sharpness * factor),
            margin=self.margin,
        )


class BaseStylizer:
    name = "base"

    def stylize(self, image: Any, face: Any, options: StyleOptions) -> Any:
        """
        :param image: BGR-изображение целиком (после переноса лица)
        :param face: лицо insightface на этом изображении — bbox и лендмарки
        :param options: сила стадий
        :return: новое BGR-изображение того же размера
        """
        raise NotImplementedError(f"{self.name}: stylize() не реализован")

    def is_available(self) -> bool:
        return False
