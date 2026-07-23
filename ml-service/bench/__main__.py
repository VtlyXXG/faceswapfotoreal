"""CLI A/B-стенда:  python -m bench [--input DIR] [--strengths a,b,c] [--out DIR]"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.core.logging import setup_logging
from app.pipelines.style.metrics import IDENTITY_THRESHOLD
from bench.model import pick_recommended
from bench.report import build_report
from bench.runner import run_bench


def _parse_strengths(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(round(float(part), 3))
    if not values:
        raise argparse.ArgumentTypeError("список strengths пуст")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench", description="A/B-стенд постобработки лица")
    parser.add_argument("--input", default="bench/samples", help="каталог с парами")
    parser.add_argument("--out", default="bench/out", help="куда писать отчёт")
    parser.add_argument(
        "--strengths",
        type=_parse_strengths,
        default=[0.6, 1.0, 1.4],
        help="значения style_strength через запятую",
    )
    args = parser.parse_args()

    setup_logging()

    data = run_bench(args.input, args.strengths)
    if data["pairs_found"] == 0:
        print(
            f"Пары не найдены в {args.input}.\n"
            "Разложите изображения одним из способов:\n"
            f"  {args.input}/source.jpg + target*.png        (один донор, много обложек)\n"
            f"  {args.input}/<кейс>/source.jpg + target*.png (папка на пару)"
        )
        return 1

    results, variants, summaries = data["results"], data["variants"], data["summaries"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    html = build_report(results, variants, summaries, IDENTITY_THRESHOLD, generated_at)
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "threshold": IDENTITY_THRESHOLD,
                "variants": [asdict(v) for v in variants],
                "summary": [asdict(s) for s in summaries],
                "pairs": [
                    {
                        "pair_id": r.pair_id,
                        "cells": [
                            {"variant": c.variant, "identity": c.identity, "texture": c.texture}
                            for c in r.cells
                        ],
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_summary(summaries, variants)
    print(f"\nОтчёт:  {report_path}")
    print(f"Данные: {json_path}")
    return 0


def _print_summary(summaries, variants) -> None:
    print(f"\n{'вариант':<16}{'узнаваемость':>14}{'фактура':>12}{'ниже порога':>14}")
    print("-" * 56)
    for s in summaries:
        identity = "—" if s.mean_identity is None else f"{s.mean_identity:.3f}"
        texture = "—" if s.mean_texture is None else f"{s.mean_texture:.4f}"
        print(f"{s.label:<16}{identity:>14}{texture:>12}{f'{s.below_threshold}/{s.n}':>14}")

    recommended, reason = pick_recommended(variants, summaries, IDENTITY_THRESHOLD)
    print()
    if recommended is not None:
        print(f"Рекомендация: {recommended.label} — {reason}")
    else:
        print(f"Рекомендация: {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
