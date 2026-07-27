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


def test_head_is_cut_along_the_jaw_and_the_neck_stays():
    """
    Область собирается из двух частей: голова срезается по дуге челюсти, а под
    ней остаётся колонна шеи. Дуга нужна, потому что плечи лежат на той же
    высоте, что и подбородок, — прямой срез забрал бы их вместе с воротником.
    """
    region, neck_line, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.4)

    assert region[254, 200] == 255, "над подбородком — голова"
    assert region[290, 200] == 255, "под подбородком — шея"
    assert region[330, 200] == 0, "ниже линии одежды — уже не наша забота"

    # Сбоку от шеи, на уровне плеч, области быть не должно: там футболка
    assert region[290, 300] == 0, "плечи в вырезку не попадают"

    # Отрезок стыка горизонтален у ненаклонённой головы и лежит на срезе
    (x1, y1), (x2, y2) = neck_line
    assert y1 == pytest.approx(y2, abs=1)
    assert y1 == pytest.approx(260 + 0.4 * 80, abs=3)


def test_neck_ratio_sets_how_much_neck_is_taken():
    """Доля neck — это и есть найденная линия одежды: докуда брать шею."""
    short, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.15)
    long, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.5)

    assert short[295, 200] == 0
    assert long[295, 200] == 255


def test_neck_column_narrows_towards_the_collar():
    """
    Колонна — эллипс, а не прямоугольник: прямые вертикальные бока во всю длину
    шеи читаются на живописи как наклейка, и мягким инпейнтингом их не убрать.
    """
    region, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.5)

    def width(row: int) -> int:
        return int(np.count_nonzero(region[row, :]))

    assert width(275) > width(295) > 0, "к воротнику шея сужается"


def test_erase_mode_takes_the_head_without_the_neck():
    """
    Режим стирания головы персонажа: колонны нет вовсе. Его шея остаётся на
    месте — на неё садится шея донора, и стирать её незачем.
    """
    donor, _, _ = segmentation.head_region(face_mesh(), (400, 400), neck_ratio=0.4)
    erase, _, _ = segmentation.head_region(
        face_mesh(), (400, 400), neck_ratio=0.05, follow_jaw=False, neck_column=False
    )

    assert donor[290, 200] == 255, "шея донора берётся"
    assert erase[290, 200] == 0, "шея персонажа остаётся нетронутой"
    # Прямой срез забирает ухо целиком, дуга оставила бы его кончик
    assert erase[245, 240] == 255


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


# --- Линия одежды: докуда брать шею ---


def _portrait(collar_at: int | None = None, chin_shadow: bool = False) -> np.ndarray:
    """
    Кадр под сетку-заглушку: лицо и шея телесного цвета, ниже — одежда.

    Цвета взяты не с потолка: проверка идёт по хроме LAB, и «кожа» должна
    отличаться от «ткани» именно в каналах a и b, а не яркостью.
    """
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[:, :] = (60, 40, 30)  # фон
    cv2.rectangle(frame, (140, 150), (260, 400), (170, 180, 210), -1)  # кожа
    if chin_shadow:
        # Тень под подбородком: та же кожа, но вдвое темнее
        cv2.rectangle(frame, (140, 262), (260, 280), (85, 90, 105), -1)
    if collar_at is not None:
        cv2.rectangle(frame, (100, collar_at), (300, 400), (40, 90, 220), -1)  # ткань
    return frame


def _alpha_for(frame: np.ndarray) -> np.ndarray:
    """Силуэт: всё, что не фон."""
    return np.where(frame.sum(axis=2) > 200, 255, 0).astype(np.uint8)


def test_clothing_line_stops_at_the_collar():
    """
    Ради этого всё и затевалось: срез должен встать на воротнике, а не на
    челюсти. Голова без шеи садится на обложку и висит в воздухе.
    """
    frame = _portrait(collar_at=320)

    ratio, meta = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert meta["neck_source"] == "collar"
    # Подбородок на y=260, воротник на 320: шеи 60 px при лице 80, минус отступ
    assert ratio == pytest.approx(60 / 80 - segmentation._COLLAR_MARGIN, abs=0.05)


def test_clothing_line_steps_over_the_shadow_under_the_chin():
    """
    Под подбородком всегда тень, и первые проценты пути она проверку не
    проходит. Обрывать поиск на ней — значит срезать шею целиком, ровно как
    раньше по челюсти.
    """
    frame = _portrait(collar_at=320, chin_shadow=True)

    ratio, meta = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert meta["neck_source"] == "collar"
    assert ratio > 0.5, "тень перешагнута, шея взята целиком"


def test_clothing_line_takes_what_it_can_when_the_frame_is_cropped():
    """
    Кадр обрезан под подбородком — обычное дело для присланного фото. Берём
    столько шеи, сколько есть, и не отказываем.
    """
    frame = _portrait()[:300]

    ratio, meta = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert meta["neck_source"] == "frame", "кожа упёрлась в край кадра"
    assert segmentation._NECK_MIN_RATIO <= ratio <= 0.5


def test_turtleneck_leaves_almost_no_neck():
    """
    Свитер под горло: кожа кончается сразу под подбородком. Поиск отработал
    верно — шеи в кадре действительно нет, — и срез встаёт по нижней границе.
    """
    frame = _portrait(collar_at=262)

    ratio, meta = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert meta["neck_source"] == "collar"
    assert ratio == segmentation._NECK_MIN_RATIO


def test_clothing_line_falls_back_when_there_is_nothing_to_measure():
    """
    Кадр кончается на самом подбородке — мерить нечего. Отказывать нельзя,
    фотографии присылают заказчики: берём запасной отступ и идём дальше.
    """
    frame = _portrait()[:260]

    ratio, meta = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert meta["neck_source"] == "fallback"
    assert ratio == segmentation._NECK_FALLBACK


def test_clothing_line_is_clamped():
    """
    Голая по пояс фотография не повод забирать грудь: поиск ограничен сверху,
    иначе в аппликацию поедет всё, что телесного цвета.
    """
    frame = _portrait()

    ratio, _ = segmentation.clothing_line(frame, _alpha_for(frame), face_mesh())

    assert ratio <= segmentation._NECK_MAX_RATIO


def test_cutout_reports_where_the_cut_came_from(monkeypatch):
    """
    По метаданным должно быть видно, найдена линия одежды или взят запасной
    отступ: результат на глаз одинаковый, а причина разбора полётов разная.
    """
    frame = _portrait(collar_at=320)
    monkeypatch.setattr(segmentation, "silhouette", lambda image, model: _alpha_for(frame))

    head = segmentation.cutout_head(frame, face_mesh(), "stub")

    assert head.meta["neck_source"] == "collar"
    assert head.meta["neck_ratio"] > 0.3, "шея взята, а не отрезана по челюсти"
