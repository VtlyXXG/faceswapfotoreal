"""
Ленивая загрузка тяжёлых моделей.

Импорты insightface/torch намеренно отложены внутрь функций: сервис должен
подниматься и отвечать на /health даже когда веса не скачаны или ML-зависимости
ещё не установлены — Node.js API увидит degraded, а не отказ соединения.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import settings
from app.core.errors import ModelNotLoadedError
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_face_analyser = None
_face_swapper = None
_stylizer = None


def models_dir() -> Path:
    return settings.models_dir.expanduser().resolve()


def swapper_path() -> Path:
    return models_dir() / settings.face_swapper


def get_face_analyser():
    """insightface.app.FaceAnalysis — детекция лиц и извлечение эмбеддингов."""
    global _face_analyser
    if _face_analyser is not None:
        return _face_analyser

    with _lock:
        if _face_analyser is not None:
            return _face_analyser

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # noqa: BLE001
            raise ModelNotLoadedError(
                "insightface не установлен — выполните scripts/setup_venv.ps1",
                {"cause": str(exc)},
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.resolve_device() == "cuda"
            else ["CPUExecutionProvider"]
        )

        log.info("загрузка детектора %s (%s)", settings.face_detector, providers[0])
        analyser = FaceAnalysis(
            name=settings.face_detector,
            root=str(models_dir()),
            providers=providers,
        )
        analyser.prepare(ctx_id=0, det_size=(settings.det_size, settings.det_size))

        _face_analyser = analyser
        return _face_analyser


def get_face_swapper():
    """inswapper_128.onnx — модель переноса лица."""
    global _face_swapper
    if _face_swapper is not None:
        return _face_swapper

    with _lock:
        if _face_swapper is not None:
            return _face_swapper

        path = swapper_path()
        if not path.exists():
            raise ModelNotLoadedError(
                f"Веса {settings.face_swapper} не найдены",
                {"expected_path": str(path), "hint": "см. ml-service/README.md"},
            )

        try:
            import insightface
        except ImportError as exc:  # noqa: BLE001
            raise ModelNotLoadedError(
                "insightface не установлен — выполните scripts/setup_venv.ps1",
                {"cause": str(exc)},
            ) from exc

        log.info("загрузка swapper %s", path.name)
        _face_swapper = insightface.model_zoo.get_model(str(path))
        return _face_swapper


def get_stylizer():
    """Стилизатор постобработки. Весов не требует, но синглтон — для единообразия."""
    global _stylizer
    if _stylizer is not None:
        return _stylizer

    with _lock:
        if _stylizer is not None:
            return _stylizer

        from app.pipelines.style.base import StyleOptions
        from app.pipelines.style.classical import ClassicalStylizer, NoopStylizer
        from app.pipelines.style.diffusion import DiffusionStylizer

        providers = {
            "classical": ClassicalStylizer,
            "diffusion": DiffusionStylizer,
            "noop": NoopStylizer,
        }
        provider = providers.get(settings.style_provider)
        if provider is None:
            log.warning(
                "неизвестный стилизатор %s, постобработка отключена", settings.style_provider
            )
            provider = NoopStylizer

        _stylizer = provider()
        _stylizer.default_options = StyleOptions(
            color=settings.style_color,
            smooth=settings.style_smooth,
            grain=settings.style_grain,
            sharpness=settings.style_sharpness,
            margin=settings.style_margin,
        )
        log.info("стилизатор %s инициализирован", _stylizer.name)
        return _stylizer


def warmup() -> None:
    """Прогрев на старте при ML_LAZY_LOAD=false."""
    get_face_analyser()
    get_face_swapper()


def reset() -> None:
    """Сброс кэша моделей (тесты, горячая замена весов)."""
    global _face_analyser, _face_swapper, _stylizer
    with _lock:
        _face_analyser = None
        _face_swapper = None
        _stylizer = None


def runtime_status() -> dict:
    """Наличие ML-пакетов — без их импорта в память процесса."""
    from importlib.util import find_spec

    return {
        "insightface": find_spec("insightface") is not None,
        "onnxruntime": find_spec("onnxruntime") is not None,
        "torch": find_spec("torch") is not None,
    }


def _stylizer_available() -> bool:
    """Готовность стилизатора без его загрузки — для /health/ready."""
    provider = settings.style_provider
    if provider in ("classical", "noop"):
        return True
    if provider == "diffusion":
        from app.pipelines.style.diffusion import weights_present

        return weights_present()
    return False


def status() -> dict:
    """Состояние моделей для /health — без загрузки весов."""
    detector_dir = models_dir() / "models" / settings.face_detector
    return {
        "device": settings.resolve_device(),
        "models_dir": str(models_dir()),
        "runtime": runtime_status(),
        "detector": {
            "name": settings.face_detector,
            "loaded": _face_analyser is not None,
            "available": detector_dir.exists(),
        },
        "swapper": {
            "name": settings.face_swapper,
            "loaded": _face_swapper is not None,
            "available": swapper_path().exists(),
        },
        "stylizer": {
            "name": settings.style_provider,
            "loaded": _stylizer is not None,
            "available": _stylizer_available(),
        },
    }
