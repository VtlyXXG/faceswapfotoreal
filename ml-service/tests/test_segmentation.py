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


def test_region_is_cut_along_the_jaw():
    """
    Срез идёт по линии челюсти, а не прямой через подбородок. Разница видна по
    бокам: у уха челюсть поднимается на полвысоты лица, и прямая оставила бы там
    треугольник шеи — а с ним и воротник.
    """
    region, neck_line, _ = segmentation.head_region(face_mesh(), (400, 400))

    assert region[254, 200] == 255, "над подбородком — голова"
    assert region[266, 200] == 0, "под подбородком — уже не наша забота"

    # Сбоку линия челюсти поднимается: у x=240 она проходит по y≈236, тогда как
    # прямая через подбородок оставила бы здесь ещё 24 пикселя шеи
    assert region[230, 240] == 255, "щека над челюстью на месте"
    assert region[245, 240] == 0, "шея под челюстью отрезана"

    # Отрезок среза горизонтален у ненаклонённой головы и лежит на подбородке
    (x1, y1), (x2, y2) = neck_line
    assert y1 == pytest.approx(y2, abs=1)
    assert y1 == pytest.approx(260, abs=2)
    assert abs(x2 - x1) > 100, "отрезок перекрывает голову по ширине"


def test_neck_ratio_lowers_the_cut():
    """
    Доля neck опускает срез ниже челюсти — на случай, если модели не хватит
    материала, чтобы дорисовать воротник на голом подбородке.
    """
    tight, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.0)
    loose, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.3)

    assert tight[275, 200] == 0
    assert loose[275, 200] == 255, "24 пикселя шеи под подбородком остались"


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


# --- Чистка края силуэта ---


def test_clean_alpha_drops_the_halo_and_trims_the_edge():
    """
    U²-Net отдаёт карту уверенности, а не бинарную маску: вокруг волос лежит
    широкая полутень, и в коллаже она даёт полупрозрачный ореол из фона
    фотографии. Порог убирает полутень, эрозия — последние пиксели прядей,
    которые с этим фоном уже смешаны.
    """
    alpha = np.full((100, 100), 80, dtype=np.uint8)  # полутень по всему кадру
    cv2.circle(alpha, (50, 50), 30, 255, -1)

    cleaned = segmentation.clean_alpha(alpha, erode_px=3)

    assert set(np.unique(cleaned)) <= {0, 255}, "маска должна стать бинарной"
    assert cleaned[50, 82] == 0, "полутень — это фон, а не волосы"
    assert cleaned[50, 78] == 0, "три пикселя края срезаны эрозией"
    assert cleaned[50, 74] == 255, "сама голова на месте"


def test_erosion_is_measured_from_the_face(monkeypatch):
    """
    Доля, а не пиксели: одни и те же «два пикселя» на превью съедают прядь
    целиком, а на 4K не делают ничего.
    """
    alpha = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(alpha, (200, 190), 100, 255, -1)
    monkeypatch.setattr(segmentation, "silhouette", lambda image, model: alpha)

    head = segmentation.cutout_head(_frame(), face_mesh(), "stub", erode_ratio=0.05)

    assert head.meta["erode_px"] == 4, "0.05 от лица высотой 80"
    assert head.alpha[190 - 94, 200] == 255
    assert head.alpha[190 - 98, 200] == 0, "край силуэта подрезан внутрь"


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
