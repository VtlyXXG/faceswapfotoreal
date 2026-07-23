"""Тесты эндпоинтов состояния. Веса моделей не требуются."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ml-service"
    assert body["uptime_seconds"] >= 0


def test_readiness_reports_models():
    response = client.get("/health/ready")
    # 503 — нормальный ответ, пока веса не скачаны
    assert response.status_code in (200, 503)

    body = response.json()
    assert body["status"] in ("ready", "degraded")
    assert body["device"] in ("cpu", "cuda")
    assert "detector" in body and "swapper" in body
    assert set(body["runtime"]) == {"insightface", "onnxruntime", "torch"}

    # degraded обязан объяснять причину, ready — не имеет её
    if body["status"] == "degraded":
        assert body["reason"]
    else:
        assert body["reason"] is None


def test_openapi_lists_face_swap_routes():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths
    assert "/face-swap" in paths
    assert "/face-swap/analyse" in paths
