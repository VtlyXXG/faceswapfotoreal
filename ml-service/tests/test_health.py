"""Тесты эндпоинтов состояния. Веса моделей не требуются."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_application_starts():
    """
    Проход через lifespan, а не только по эндпоинтам.

    Обычный TestClient стартовые обработчики не выполняет, поэтому обращение к
    несуществующей настройке в них тесты не ловят — ровно так и вышло, когда
    гиперпараметры второго шага переехали в профиль: сервис падал на старте, а
    зелёный прогон об этом не знал.
    """
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ml-service"
    assert body["uptime_seconds"] >= 0


def test_readiness_reports_provider():
    response = client.get("/health/ready")
    # 503 — нормальный ответ, пока не задан FAL_KEY
    assert response.status_code in (200, 503)

    body = response.json()
    assert body["status"] in ("ready", "degraded")
    assert set(body["runtime"]) == {"mediapipe", "opencv", "fal_client", "rembg"}
    assert body["provider"]["model"]
    assert body["mask"]["detector"] == "mediapipe/face_mesh"
    # Второй шаг виден целиком: какой набор чисел активен, какой стратегией он
    # исполняется и что вообще доступно на выбор
    assert body["provider"]["profile"] and body["provider"]["strategy"]
    assert body["provider"]["profile_error"] is None, "профиль обязан собираться"
    assert "seam" in body["provider"]["profiles"]
    assert "identity_embedding" in body["provider"]["strategies"]
    assert body["mask"]["gradient_ratio"] is not None
    # Оба шага пайплайна видны снаружи: сегментатор с весами и список эмоций
    assert body["collage"]["segmenter_photo"]
    assert "neutral" in body["expressions"]

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
