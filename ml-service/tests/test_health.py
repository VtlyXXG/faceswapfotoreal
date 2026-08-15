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
    assert set(body["runtime"]) == {"mediapipe", "opencv", "fal_client", "parsing_weights"}
    assert body["provider"]["model"]
    assert "face_mesh" in body["mask"]["detector"]
    # Второй шаг виден целиком: какой набор чисел активен, какой стратегией он
    # исполняется и что вообще доступно на выбор
    assert body["provider"]["profile"] and body["provider"]["strategy"]
    assert body["provider"]["profile_error"] is None, "профиль обязан собираться"
    assert "pixar_real" in body["provider"]["profiles"]
    assert {"face_swap", "kontext_multi"} <= set(body["provider"]["strategies"])
    # Куда уезжает фотография заказчика — главный вопрос схемы
    assert body["provider"]["identity_field"]
    # И что вообще уедет в теле запроса. Эндпоинт заворачивает весь запрос из-за
    # любого лишнего ключа, поэтому список сверяется целиком, а не на вхождение
    assert body["provider"]["sends"] == ["base_image_url", "swap_image_url"]
    # Рабочий путь локальной геометрии не требует: маска в блоке ниже описывает
    # только то, чем она СТРОИЛАСЬ БЫ на диффузионных стратегиях
    assert body["provider"]["needs_mask"] is False
    assert body["mask"]["dilate_ratio"] is not None
    assert body["mask"]["neck_ratio"] is not None
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


def test_readiness_ignores_fal_when_disabled(monkeypatch):
    """
    Выключенный путь через fal не мешает готовности.

    Регрессия дорогая: пока проверка ключа стояла безусловно, свежий клон
    отвечал 503 с требованием чужого платного ключа, хотя работать собирался
    локально. Поэтому здесь проверяется не текст, а именно код ответа.
    """
    from app.pipelines import fal_api

    monkeypatch.setattr(fal_api.settings, "fal_enabled", False)
    monkeypatch.delenv("FAL_KEY", raising=False)

    response = client.get("/health/ready")
    body = response.json()

    assert body["provider"]["fal_enabled"] is False
    assert response.status_code == 200, body.get("reason")
    assert body["status"] == "ready"
    assert "FAL_KEY" not in (body.get("reason") or "")
