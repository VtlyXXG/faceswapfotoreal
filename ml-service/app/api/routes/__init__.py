from fastapi import APIRouter

from app.api.routes import face_swap, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(face_swap.router)

__all__ = ["api_router"]
