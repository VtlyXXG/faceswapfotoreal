"""
Состояние обработчика для /health/ready.

Тяжёлых весов больше нет — прогревать нечего, поэтому от прежнего реестра
моделей осталась только диагностика: доступны ли зависимости, есть ли веса
разметки и настроен ли ключ fal. Функции warmup/reset сохранены, чтобы не
менять app/main.py.
"""

from __future__ import annotations

from importlib.util import find_spec

from app.config import settings
from app.core.errors import MLServiceError
from app.pipelines import expression, fal_api, parsing, refine
from app.pipelines.refine import local_render

# Стратегии, у которых замена лица идёт на своём GPU-сервере. У двухшаговой это
# второй вызов, и без адреса она отвалится ровно так же, как одношаговая
_LOCAL_STRATEGIES = ("face_swap", "hair_swap")


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def warmup() -> None:
    """Раньше грузила веса в память. Локальных моделей нет — делать нечего."""


def reset() -> None:
    """Симметрична warmup: освобождать тоже нечего."""


def _active():
    """
    Активный профиль второго шага и причина, если собрать его не вышло.

    Профиль складывается из пресета и переопределений окружения, то есть может
    оказаться несобираемым — например, неизвестная схема запроса или стиль.
    Узнать об этом на /health/ready лучше, чем на первом живом заказе: тот
    отвалится уже после трёх загрузок в CDN.
    """
    try:
        return refine.profiles.from_settings(), None
    except MLServiceError as exc:
        return None, exc.message


def status() -> dict:
    profile, profile_error = _active()
    mask = profile.mask if profile else None

    return {
        "runtime": {
            "mediapipe": _installed("mediapipe"),
            "opencv": _installed("cv2"),
            "fal_client": _installed("fal_client"),
            # Веса семантической разметки. Работу не блокируют: без них маска
            # головы строится эллипсом по сетке лица — грубее, но рабоче
            "parsing_weights": parsing.available(),
        },
        # Профиль виден целиком намеренно: strength, вес идентичности и схема
        # запроса — главные ручки пайплайна, и подбирают их из окружения на
        # живом сервисе. Пустые значения означают, что профиль не собрался, —
        # тогда всё, что о нём известно, лежит в profile_error
        "provider": {
            "model": profile.endpoint if profile else None,
            # Первое, на что смотреть при разборе «почему сервис не готов»:
            # выключенный путь означает, что ни ключ, ни профиль второго шага
            # к готовности отношения не имеют
            "fal_enabled": fal_api.enabled(),
            "key_present": fal_api.key_present(),
            "key_env": settings.fal_key_env,
            "strength": profile.strength if profile else None,
            "profile": settings.refine_profile,
            "strategy": profile.strategy if profile else None,
            "payload": profile.payload.name if profile else None,
            "style": profile.style if profile else None,
            # Куда уезжает личность. У масочной схемы это отдельное поле, у
            # безмасочной — второй элемент массива картинок, и поля нет вовсе
            "identity_field": (
                profile.payload.identity_field or profile.payload.images_field
                if profile
                else None
            ),
            "identity_scale": profile.identity_scale if profile else None,
            "sends": profile.payload.sent_keys() if profile else [],
            # Нужна ли заказу локальная геометрия. false — маска не строится
            # вовсе, и блок mask ниже описывает только то, чем она СТРОИЛАСЬ БЫ
            "needs_mask": profile.needs_mask if profile else None,
            "profile_error": profile_error,
            "profiles": refine.profiles.available(),
            "strategies": refine.available(),
            "styles": refine.profiles.styles(),
        },
        # Рабочий путь: свой GPU-сервер. Пустой адрес — единственное, чего ему
        # не хватает для работы, и на /health/ready это должно быть сказано
        # словами, а не выясняться таймаутом на первом заказе
        "render": {
            "base_url": local_render.base_url() or None,
            "configured": local_render.configured(),
            "path": local_render.PATH,
            "timeout_s": settings.render_timeout_s,
            # Идёт ли туда текущий профиль. false означает, что выбран один из
            # путей через fal, и адрес GPU-сервера к готовности отношения не имеет
            "active": bool(profile and profile.strategy in _LOCAL_STRATEGIES),
        },
        # Единственная локальная работа: какую область шаблона отдаём модели
        "mask": {
            "detector": "mediapipe/face_mesh + selfie_multiclass",
            "parsing_model": str(parsing.model_path()),
            "dilate_ratio": mask.dilate_ratio if mask else None,
            "feather_ratio": mask.feather_ratio if mask else None,
            "neck_ratio": mask.neck_ratio if mask else None,
        },
        # Первый шаг двухшаговой стратегии. Виден всегда, работает только при
        # active=true: фейссвоп в одиночку волос не касается вовсе
        "hair": {
            "active": bool(profile and profile.strategy == "hair_swap"),
            "endpoint": profile.hair.endpoint if profile else None,
            "payload": profile.hair.payload.name if profile else None,
            "description": profile.hair.description if profile else "",
            "dilate_ratio": profile.hair.dilate_ratio if profile else None,
            "feather_ratio": profile.hair.feather_ratio if profile else None,
            "protect_ratio": profile.hair.protect_ratio if profile else None,
            "forehead_ratio": profile.hair.forehead_ratio if profile else None,
            "core_ratio": profile.hair.core_ratio if profile else None,
            "crop_ratio": profile.hair.crop_ratio if profile else None,
            "min_changed": profile.hair.min_changed if profile else None,
        },
        # Мимика: какие значения параметра emotion эндпоинт сейчас принимает
        "expressions": expression.available(),
    }
