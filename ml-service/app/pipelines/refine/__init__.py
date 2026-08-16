"""
Шаг 2: генерация головы на шаблоне по референсу личности.

Пакет собран из трёх слоёв, и разделение между ними — главное, что тут есть:

  profiles.py  — числа и тексты. Strength, доли маски, схемы запросов,
                 идентификаторы эндпоинтов, промпты стилей. Ни одного вызова.
  base.py      — контракт: что стратегия получает, что возвращает, и реестр,
                 через который она находится по имени из профиля.
  стратегии    — local_render.py: РАБОЧАЯ. Свой GPU-сервер, /v1/demo-render:
                 FLUX.2 плюс пересадка головы. Ни ключей, ни облака, ни
                 локальной подготовки — сервер считает геометрию сам. В реестре
                 зовётся `face_swap`, поэтому её же берёт второй шаг hair_swap;
                 fal_face_swap.py: специализированный фейссвоп на fal, без
                 промпта и без маски. В реестре `fal_face_swap`, выключен
                 вместе со всем путём через fal;
                 hair_swap.py: два вызова — редактор правит одну причёску, затем
                 та же стратегия face_swap переносит лицо. Нужна там, где
                 причёска донора отличается от нарисованной: фейссвоп волос не
                 касается вовсе;
                 kontext_multi.py: редактор по двум картинкам плюс вклейка
                 головы. Эндпоинт у неё переключаемый: kontext провалился, но
                 схема та же у nano-banana, seedream и hy-wu — это запасной
                 путь для стилизованных шаблонов;
                 identity_inpaint.py: генерация внутри маски по фото-референсу
                 (маска лишает модель контекста, провалилась).

Снаружи нужны ровно две вещи:

    profile = profiles.from_settings()
    result = refine.run(refine.RefineRequest(...), profile)

Сменить гиперпараметр — правка profiles.py или переменная ML_REFINE_*.
Сменить подход целиком — другое значение `strategy` в профиле.
"""

from __future__ import annotations

# Импорт ради регистрации: модуль стратегии вызывает register() при загрузке.
# Имена в реестре у всех разные, поэтому порядок здесь ни на что не влияет —
# и это специально: рабочий путь не должен зависеть от того, кого импортировали
# позже.
from app.pipelines.refine import fal_face_swap as _fal_face_swap  # noqa: E402,F401
from app.pipelines.refine import hair_swap as _hair_swap  # noqa: E402,F401
from app.pipelines.refine import identity_inpaint as _identity_inpaint  # noqa: E402,F401
from app.pipelines.refine import kontext_multi as _kontext_multi  # noqa: E402,F401
from app.pipelines.refine import local_render as _local_render  # noqa: E402,F401
from app.pipelines.refine import profiles
from app.pipelines.refine.base import (
    Refiner,
    RefineRequest,
    RefineResult,
    RefinerNotSupportedError,
    available,
    get,
    register,
    run,
)

__all__ = [
    "RefineRequest",
    "RefineResult",
    "Refiner",
    "RefinerNotSupportedError",
    "available",
    "get",
    "profiles",
    "register",
    "run",
]
