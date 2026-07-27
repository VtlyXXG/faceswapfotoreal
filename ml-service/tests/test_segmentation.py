"""
Вырезка головы: область головы по сетке и фильтрация силуэта.

Сам сегментатор не запускается — onnx-модель весит 176 МБ и тестируется не
здесь. Проверяется наша часть работы: правильно ли построен эллипс головы, где
проходит срез шеи и что остаётся от силуэта после ограничения областью.
"""

import cv2
import numpy as np
import pytest

from app.core.errors import MLServiceError
from app.pipelines import segmentation
from tests.conftest import face_mesh


@pytest.fixture
def whole_frame(monkeypatch):
    """Сегментатор, считающий передним планом весь кадр."""
    monkeypatch.setattr(
        segmentation,
        "silhouette",
        lambda image, model: np.full(image.shape[:2], 255, dtype=np.uint8),
    )


def _frame() -> np.ndarray:
    return np.zeros((400, 400, 3), dtype=np.uint8)


# --- Область головы ---


def test_region_reaches_above_the_hairline():
    """
    Ради причёски всё и затевалось: область обязана уходить вверх от
    подбородка настолько, чтобы вместить макушку с объёмом волос.
    """
    region, _, face_height = segmentation.head_region(face_mesh(), (400, 400))

    assert face_height == pytest.approx(80, abs=2)
    # Подбородок на y=260, канон даёт макушку на 2.3 высоты лица выше
    assert region[260 - int(2.0 * 80), 200] == 255
    # Ещё выше — уже фон, и тянуть его в аппликацию незачем
    assert region[260 - int(2.6 * 80), 200] == 0


def test_region_is_cut_flat_below_the_chin():
    """Срез шеи прямой: чем он предсказуемее, тем проще спрятать его маской."""
    region, neck_line, _ = segmentation.head_region(face_mesh(), (400, 400))

    neck_y = 260 + int(0.45 * 80)
    assert region[neck_y - 6, 200] == 255, "над срезом — голова"
    assert region[neck_y + 6, 200] == 0, "под срезом — уже не наша забота"

    # Отрезок среза горизонтален у ненаклонённой головы и лежит на той же линии
    (x1, y1), (x2, y2) = neck_line
    assert y1 == pytest.approx(y2, abs=1)
    assert y1 == pytest.approx(neck_y, abs=2)
    assert abs(x2 - x1) > 100, "отрезок перекрывает голову по ширине"


def test_region_follows_the_tilt_of_the_head():
    """
    Эллипс поворачивается вместе с головой. Иначе у наклонённого лица он срежет
    висок с одной стороны и захватит фон с другой.
    """
    straight, _, _ = segmentation.head_region(face_mesh(), (400, 400))
    tilted, _, _ = segmentation.head_region(face_mesh(angle=30), (400, 400))

    def top_x(region: np.ndarray) -> int:
        ys, xs = np.nonzero(region)
        return int(xs[ys == ys.min()].mean())

    # Наклон вправо уводит макушку вправо же
    assert top_x(tilted) > top_x(straight) + 10


def test_degenerate_mesh_is_rejected():
    flat = [(200, 200)] * 468

    with pytest.raises(MLServiceError):
        segmentation.head_region(flat, (400, 400))


# --- Ограничение силуэта областью головы ---


def test_cutout_keeps_only_the_head(whole_frame):
    """
    Сегментатор отдаёт человека целиком — с плечами, руками и всем остальным.
    В аппликацию должна попасть только голова.
    """
    head = segmentation.cutout_head(_frame(), face_mesh(), "stub")

    assert head.alpha[100, 200] == 255, "макушка на месте"
    assert head.alpha[380, 200] == 0, "плечи отрезаны"
    assert head.meta["head_px"] > 0


def test_cutout_drops_foreign_blobs(monkeypatch):
    """
    В эллипс головы попадает не только она: поднятая к лицу рука, плечо
    соседа. Это отдельные пятна альфы, и вклеивать их не нужно.
    """
    alpha = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(alpha, (200, 200), 90, 255, -1)  # сама голова
    cv2.circle(alpha, (110, 130), 25, 255, -1)  # что-то ещё рядом
    monkeypatch.setattr(segmentation, "silhouette", lambda image, model: alpha)

    head = segmentation.cutout_head(_frame(), face_mesh(), "stub")

    assert head.alpha[200, 200] == 255
    assert head.alpha[130, 110] == 0, "чужое пятно должно отвалиться"


def test_cutout_without_a_silhouette_fails_loudly(monkeypatch):
    """
    Пустая альфа — это промах сегментатора, а не «голова нулевой площади».
    Молча вклеить ничего и уехать в fal было бы хуже: платный вызов вернул бы
    нетронутую обложку, и обнаружилось бы это на печати.
    """
    monkeypatch.setattr(
        segmentation, "silhouette", lambda image, model: np.zeros(image.shape[:2], np.uint8)
    )

    with pytest.raises(segmentation.SegmenterUnavailableError) as exc_info:
        segmentation.cutout_head(_frame(), face_mesh(), "stub")

    assert exc_info.value.status_code == 503


def test_cutout_reports_how_full_the_region_is(whole_frame):
    """
    fill — доля эллипса, занятая силуэтом. По ней видно и промах сегментатора
    (около нуля), и упёршуюся в границу причёску (около единицы).
    """
    head = segmentation.cutout_head(_frame(), face_mesh(), "stub")

    assert head.meta["fill"] == pytest.approx(1.0, abs=0.01)
