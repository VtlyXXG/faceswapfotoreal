"""
Контракт второго шага: что стратегия получает и что обязана вернуть.

Шаг называется «генерация», и вход у него из трёх картинок:

  * **шаблон** — разворот книги как есть, с нарисованным персонажем;
  * **маска** — область его головы и шеи, то есть где модели можно рисовать;
  * **референс** — фотография заказчика, откуда берётся личность.

Пикселей заказчика в шаблоне нет и не будет: голова рисуется внутри маски
заново, в материале сцены. Отсюда и главное отличие от прежней схемы —
strength близок к единице, а не к нулю: сохранять под маской нечего.

Стратегия находится по имени из профиля, поэтому появление второго подхода
(скажем, проброса лицевых эмбеддингов, когда на fal появится эндпоинт, который
принимает их вместе с маской) не потребует править ни pipeline.py, ни
fal_api.py:

    class MyRefiner:
        name = "my_refiner"
        def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult: ...

    refine.register(MyRefiner())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.core.errors import MLServiceError
from app.core.logging import get_logger
from app.pipelines.refine.profiles import RefineProfile

log = get_logger(__name__)


class RefinerNotSupportedError(MLServiceError):
    """
    Запрошена стратегия, которой нет или которая не реализована.

    501, а не 400: запрос корректен, возможности пока нет. Разница
    существенная — по 400 вызывающая сторона чинит запрос, по 501 ждёт релиза.
    Тот же код и та же логика, что у ExpressionNotSupportedError.
    """

    status_code = 501
    code = "REFINER_NOT_SUPPORTED"


@dataclass
class RefineRequest:
    """
    Вход второго шага: сцена, рабочая область и личность.

    :param target: шаблон-разворот, PNG
    :param mask: маска головы персонажа, PNG. Белое — рисовать, чёрное — не
        трогать. Пусто у стратегий, которым локальная геометрия не нужна:
        фейссвоп находит область сам
    :param identity: фотография заказчика — референс личности
    :param expression: указание мимики для промпта. Пусто — выражение берётся с
        шаблона, и это умолчание
    :param hair: словесное описание причёски заказчика («короткий светлый
        ёжик»). Читает его одна стратегия — двухшаговая, и для неё это
        ЕДИНСТВЕННЫЙ источник того, что рисовать: фотография на шаг причёски не
        отправляется — универсальный редактор рисует по ней второе лицо.
        Пусто — либо описание задано на весь тираж (`ML_HAIR_DESCRIPTION`),
        либо заказ отклоняется до сети
    """

    target: bytes
    target_mime: str
    mask: bytes
    identity: bytes
    identity_mime: str
    expression: str = ""
    hair: str = ""
    output_format: str = "png"
    mask_mime: str = "image/png"


@dataclass
class RefineResult:
    image: bytes
    meta: dict = field(default_factory=dict)


class Refiner(Protocol):
    """Контракт стратегии генерации."""

    name: str

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        """Возвращает готовое изображение и метаданные вызова."""
        ...


_REGISTRY: dict[str, Refiner] = {}


def register(refiner: Refiner) -> None:
    """Добавляет стратегию в реестр. Повторная регистрация имени — замена."""
    _REGISTRY[refiner.name] = refiner


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Refiner:
    refiner = _REGISTRY.get((name or "").strip().lower())
    if refiner is None:
        raise RefinerNotSupportedError(
            f"Стратегия генерации «{name}» не зарегистрирована",
            {"strategy": name, "available": available()},
        )
    return refiner


def run(request: RefineRequest, profile: RefineProfile) -> RefineResult:
    """
    Выполняет второй шаг стратегией, названной в профиле.

    Единственный вход для pipeline.py: он не знает ни про fal, ни про схему
    запроса — только про профиль и про то, что на выходе будут байты.
    """
    profile.validate()
    refiner = get(profile.strategy)

    log.info("генерация головы по референсу", extra=profile.report())
    return refiner.refine(request, profile)
