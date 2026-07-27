"""
Шаг 2: стилизация готового коллажа.

Пакет собран из трёх слоёв, и разделение между ними — главное, что тут есть:

  profiles.py  — числа. Strength, пороги Canny, веса ControlNet, доли маски,
                 идентификаторы эндпоинтов, промпты. Ни одного вызова.
  base.py      — контракт: что стратегия получает, что возвращает, и реестр,
                 через который она находится по имени из профиля.
  стратегии    — inpaint_controlnet.py (работает) и identity_embedding.py
                 (контракт под проброс лицевых эмбеддингов, 501).

Снаружи нужны ровно две вещи:

    profile = profiles.from_settings()
    result = refine.run(refine.RefineRequest(...), profile)

Сменить гиперпараметр — правка profiles.py или переменная ML_REFINE_*.
Сменить подход целиком — другое значение `strategy` в профиле.
"""

from __future__ import annotations

# Импорт ради регистрации: модули стратегий вызывают register() при загрузке.
from app.pipelines.refine import identity_embedding as _identity_embedding  # noqa: E402,F401
from app.pipelines.refine import inpaint_controlnet as _inpaint_controlnet  # noqa: E402,F401
from app.pipelines.refine import profiles
from app.pipelines.refine.base import (
    Identity,
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
    "Identity",
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
