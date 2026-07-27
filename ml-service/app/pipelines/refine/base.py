"""
Контракт второго шага: что стратегия получает и что обязана вернуть.

Шаг называется «стилизация», но подходов к нему принципиально разных два, и
выбор между ними — вопрос не настройки, а архитектуры:

  1. **Inpaint + ControlNet.** Личность приезжает готовыми пикселями в коллаже,
     модель работает поверх по маске, карты управления держат геометрию.
     Реализовано в `inpaint_controlnet.py`.
  2. **Проброс лицевых эмбеддингов.** Личность передаётся эндпоинту вектором
     (матрицей векторов InsightFace), и лицо рисуется с нуля по этому вектору.
     Коллаж тогда нужен только как композиция. Контракт заложен здесь,
     реализации нет — см. `identity_embedding.py`.

Разница между ними не в числах, а в том, откуда берётся личность, поэтому
переключаться между ними параметром нельзя — нужен разный код. Отсюда реестр:
профиль называет стратегию по имени, `run` её находит, и добавление третьего
подхода не требует править ни pipeline.py, ни fal_api.py.

    class MyRefiner:
        name = "my_refiner"
        def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult: ...

    refine.register(MyRefiner())

Симметрия с `expression.py` здесь намеренная: там точка расширения для мимики,
здесь — для способа стилизации, и устроены они одинаково.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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
class Identity:
    """
    Личность заказчика в виде, пригодном для проброса в эндпоинт.

    Заготовка под будущее: матрица эмбеддингов (InsightFace отдаёт по вектору
    512 float на лицо, отсюда форма Nx512) и имя модели, которой они посчитаны.
    Вектор без имени модели бессмыслен: эмбеддинги разных детекторов лежат в
    разных пространствах и несравнимы.

    Сегодня всегда None — считать их нечем и передавать некуда.
    """

    embedding: Any  # numpy.ndarray формы (N, 512) либо то, что ждёт эндпоинт
    model: str  # чем посчитано: buffalo_l, antelopev2, …


@dataclass
class RefineRequest:
    """
    Вход второго шага.

    Кроме байтов здесь лежит и коллаж массивом: карты управления строятся по
    пикселям, а декодировать PNG второй раз ради этого незачем.
    """

    collage: bytes  # шаблон с вклеенной головой, PNG
    collage_mime: str
    reference: bytes  # фотография заказчика — референс личности
    reference_mime: str
    # Маски по зонам: "seam" — стыки, "background" — дыра в фоне. Ключи те же,
    # что имена проходов в профиле: стратегия берёт маску по имени зоны, и
    # добавление четвёртой зоны не потребует нового поля.
    masks: dict[str, bytes] = field(default_factory=dict)
    collage_image: Any = None  # тот же коллаж как BGR numpy.ndarray
    identity: Identity | None = None  # эмбеддинги, когда появятся
    output_format: str = "png"


@dataclass
class RefineResult:
    image: bytes
    meta: dict = field(default_factory=dict)


class Refiner(Protocol):
    """Контракт стратегии стилизации."""

    name: str

    def refine(self, request: RefineRequest, profile: RefineProfile) -> RefineResult:
        """Возвращает стилизованное изображение и метаданные вызова."""
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
            f"Стратегия стилизации «{name}» не зарегистрирована",
            {"strategy": name, "available": available()},
        )
    return refiner


def run(request: RefineRequest, profile: RefineProfile) -> RefineResult:
    """
    Выполняет второй шаг стратегией, названной в профиле.

    Единственный вход для pipeline.py: он не знает ни про fal, ни про
    ControlNet, ни про эмбеддинги — только про профиль и про то, что на выходе
    будут байты.
    """
    profile.validate()
    refiner = get(profile.strategy)

    log.info("стилизация коллажа", extra=profile.report())
    return refiner.refine(request, profile)
