"""Стенд A/B: обнаружение пар, агрегаты, рекомендация, отчёт — без весов."""

import numpy as np

from bench.imaging import to_data_uri
from bench.model import (
    Cell,
    PairResult,
    Variant,
    pick_recommended,
    summarize,
)
from bench.pairs import discover_pairs
from bench.report import build_report

THRESHOLD = 0.55


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG")


# --- обнаружение пар -----------------------------------------------------


def test_discover_shared_source_many_targets(tmp_path):
    _touch(tmp_path / "source.jpg")
    _touch(tmp_path / "target_a.png")
    _touch(tmp_path / "target_b.png")

    pairs = discover_pairs(tmp_path)

    assert [p.pair_id for p in pairs] == ["target_a", "target_b"]
    assert all(p.source.name == "source.jpg" for p in pairs)


def test_discover_folder_mode(tmp_path):
    _touch(tmp_path / "case1" / "source.jpg")
    _touch(tmp_path / "case1" / "target.png")
    _touch(tmp_path / "case2" / "source.png")
    _touch(tmp_path / "case2" / "target1.png")
    _touch(tmp_path / "case2" / "target2.png")

    pairs = discover_pairs(tmp_path)
    ids = {p.pair_id for p in pairs}

    assert ids == {"case1/target", "case2/target1", "case2/target2"}


def test_discover_empty_dir(tmp_path):
    assert discover_pairs(tmp_path) == []
    assert discover_pairs(tmp_path / "missing") == []


def test_discover_ignores_target_without_source(tmp_path):
    _touch(tmp_path / "target_a.png")  # донора нет
    assert discover_pairs(tmp_path) == []


# --- агрегаты ------------------------------------------------------------


def _pair(pair_id, **identity_by_variant) -> PairResult:
    cells = [
        Cell(variant=name, identity=identity, texture=0.05)
        for name, identity in identity_by_variant.items()
    ]
    return PairResult(pair_id=pair_id, source_uri="", cells=cells)


def test_summarize_counts_below_threshold():
    variants = [Variant("plain", False), Variant("s1", True, 1.0)]
    results = [
        _pair("a", plain=0.9, s1=0.6),
        _pair("b", plain=0.8, s1=0.4),  # s1 ниже порога
    ]

    summary = {s.name: s for s in summarize(results, variants, THRESHOLD)}

    assert summary["plain"].mean_identity == 0.85
    assert summary["s1"].below_threshold == 1
    assert summary["s1"].n == 2


def test_summarize_handles_missing_identity():
    variants = [Variant("s1", True, 1.0)]
    results = [PairResult("a", "", [Cell("s1", None, None)])]

    summary = summarize(results, variants, THRESHOLD)[0]

    assert summary.mean_identity is None
    assert summary.n == 0


# --- рекомендация --------------------------------------------------------


def _pair_tex(pair_id, **texture_and_identity) -> PairResult:
    """texture_and_identity: name -> (texture, identity)."""
    cells = [
        Cell(variant=name, identity=identity, texture=texture)
        for name, (texture, identity) in texture_and_identity.items()
    ]
    return PairResult(pair_id=pair_id, source_uri="", cells=cells)


def test_recommend_excludes_variants_below_identity_threshold():
    variants = [
        Variant("plain", False),
        Variant("s0.6", True, 0.6),
        Variant("s1.0", True, 1.0),
        Variant("s1.4", True, 1.4),
    ]
    # s1.4 роняет узнаваемость ниже порога, хотя фактура у него лучше всех
    results = [
        _pair_tex(
            "a",
            plain=(0.13, 0.90),
            **{"s0.6": (0.09, 0.80), "s1.0": (0.05, 0.70), "s1.4": (0.03, 0.40)},
        )
    ]
    summaries = summarize(results, variants, THRESHOLD)

    chosen, _ = pick_recommended(variants, summaries, THRESHOLD)

    assert chosen.name == "s1.0"  # лучший по фактуре из прошедших порог


def test_recommend_prefers_best_texture_not_strongest():
    """Фактура немонотонна: сильный вариант хуже среднего — берём средний."""
    variants = [
        Variant("plain", False),
        Variant("s1.0", True, 1.0),
        Variant("s1.4", True, 1.4),
    ]
    results = [
        _pair_tex(
            "a",
            plain=(0.13, 0.90),
            **{"s1.0": (0.045, 0.85), "s1.4": (0.077, 0.82)},
        )
    ]
    summaries = summarize(results, variants, THRESHOLD)

    chosen, _ = pick_recommended(variants, summaries, THRESHOLD)

    assert chosen.name == "s1.0"


def test_recommend_rejects_texture_regression():
    variants = [Variant("plain", False), Variant("s1", True, 1.0)]
    # узнаваемость в норме, но фактура хуже, чем без обработки
    results = [PairResult("a", "", [Cell("plain", 0.9, 0.05), Cell("s1", 0.8, 0.20)])]
    summaries = summarize(results, variants, THRESHOLD)

    chosen, reason = pick_recommended(variants, summaries, THRESHOLD)

    assert chosen is None
    assert "узнаваемость" in reason


# --- отчёт ---------------------------------------------------------------


def test_build_report_is_self_contained():
    variants = [Variant("plain", False), Variant("s1", True, 1.0)]
    results = [
        PairResult(
            "cover_a",
            source_uri="data:image/jpeg;base64,AAAA",
            cells=[
                Cell("plain", 0.88, 0.13, "data:image/jpeg;base64,BBBB"),
                Cell("s1", 0.84, 0.05, "data:image/jpeg;base64,CCCC"),
            ],
        )
    ]
    summaries = summarize(results, variants, THRESHOLD)

    html = build_report(results, variants, summaries, THRESHOLD, "2026-07-24 10:00")

    assert "<!doctype html>" in html
    assert "cover_a" in html
    assert "http://" not in html  # ничего внешнего не подгружается
    assert "Рекомендация" in html


def test_report_flags_below_threshold_identity():
    variants = [Variant("s1", True, 1.0)]
    results = [PairResult("a", "", [Cell("s1", 0.40, 0.05, "data:,")])]
    summaries = summarize(results, variants, THRESHOLD)

    html = build_report(results, variants, summaries, THRESHOLD, "now")

    assert "bad" in html  # красная подсветка узнаваемости ниже порога


# --- кодирование изображений ---------------------------------------------


def test_to_data_uri_downscales_and_encodes():
    image = np.full((400, 800, 3), 120, np.uint8)

    uri = to_data_uri(image, max_width=240)

    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 100
