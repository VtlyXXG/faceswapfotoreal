"""Точка входа ML-микросервиса.

Запуск:
    uvicorn app.main:app --reload
    python -m app.main
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.middleware.request_id import RequestIdMiddleware
from app.api.routes import api_router
from app.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.pipelines import registry

setup_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    provider = registry.status()["provider"]
    log.info(
        "ml-service %s стартует: профиль=%s, провайдер=%s, ключ=%s",
        __version__,
        provider["profile"],
        provider["model"],
        "задан" if provider["key_present"] else "НЕ ЗАДАН",
    )
    if provider["profile_error"]:
        # Профиль второго шага не собрался — заказы будут падать. Ронять
        # процесс из-за этого нельзя (чинится переменной окружения, и
        # /health/ready уже отвечает degraded), но в логе старта это первое,
        # что должно броситься в глаза
        log.error("профиль обработки не собран: %s", provider["profile_error"])

    yield

    registry.reset()
    log.info("ml-service остановлен")


app = FastAPI(
    title="projectx ML Service",
    description="Face-swap и вспомогательные ML-операции для сервиса персонализации обложек",
    version=__version__,
    lifespan=lifespan,
)

# Middleware выполняются в обратном порядке добавления:
# RequestIdMiddleware зарегистрирован последним → отработает первым,
# поэтому request_id доступен уже в CORS-обработке и во всех логах запроса.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Swap-Meta", "X-Request-ID"],
)
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)
app.include_router(api_router)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.env == "development",
    )


if __name__ == "__main__":
    main()
