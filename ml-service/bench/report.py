"""Сборка самодостаточного HTML-отчёта A/B-стенда.

Файл открывается локально: изображения — это персональные фото, поэтому отчёт
намеренно не публикуется и держит всё в одном файле с встроенными data: URI.
"""

from __future__ import annotations

from html import escape

from bench.model import PairResult, Variant, VariantSummary, pick_recommended

_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5;
  background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) { body { background: #16181c; color: #e8e8e8; } }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
.meta { color: #888; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc4; padding: .5rem .7rem; text-align: left; vertical-align: top; }
th { background: #0001; font-weight: 600; }
@media (prefers-color-scheme: dark) { th { background: #fff1; } }
.rec { padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0;
  background: #e6f4ea; border: 1px solid #98c9a6; }
.rec.none { background: #fdecea; border-color: #f0a9a1; }
@media (prefers-color-scheme: dark) {
  .rec { background: #16301f; border-color: #2f6b40; }
  .rec.none { background: #34191a; border-color: #7a3733; } }
.grid { display: grid; gap: .8rem; grid-auto-flow: column; justify-content: start;
  overflow-x: auto; padding-bottom: .5rem; }
.cell { text-align: center; font-size: .82rem; min-width: 120px; }
.cell img { display: block; border-radius: 6px; border: 1px solid #0002; max-width: 240px; }
.cell .caption { font-weight: 600; margin: .3rem 0 .1rem; }
.metric { font-variant-numeric: tabular-nums; }
.bad { color: #c0392b; font-weight: 700; }
.good { color: #1e8449; }
.pair-id { font-family: ui-monospace, monospace; font-size: .85rem; color: #888; }
"""


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _identity_span(value: float | None, threshold: float) -> str:
    if value is None:
        return '<span class="metric">—</span>'
    cls = "bad" if value < threshold else "good"
    return f'<span class="metric {cls}">{value:.3f}</span>'


def _summary_table(summaries: list[VariantSummary], threshold: float) -> str:
    rows = []
    for s in summaries:
        below = f'<span class="bad">{s.below_threshold}</span>' if s.below_threshold else "0"
        rows.append(
            f"<tr><td>{escape(s.label)}</td>"
            f"<td>{_identity_span(s.mean_identity, threshold)}</td>"
            f"<td class='metric'>{_fmt(s.mean_texture)}</td>"
            f"<td>{below} / {s.n}</td></tr>"
        )
    return (
        "<table><thead><tr><th>вариант</th><th>ср. узнаваемость</th>"
        "<th>ср. рассогласование фактуры</th><th>ниже порога</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _pair_block(pair: PairResult, variants: list[Variant], threshold: float) -> str:
    cells = [
        f'<div class="cell"><img src="{escape(pair.source_uri)}" alt="донор">'
        '<div class="caption">донор</div></div>'
    ]
    for variant in variants:
        cell = pair.cell(variant.name)
        if cell is None:
            continue
        note = f'<div class="pair-id">{escape(cell.note)}</div>' if cell.note else ""
        cells.append(
            f'<div class="cell"><img src="{escape(cell.crop_uri)}" alt="{escape(variant.label)}">'
            f'<div class="caption">{escape(variant.label)}</div>'
            f'<div>id {_identity_span(cell.identity, threshold)}</div>'
            f'<div class="metric">tex {_fmt(cell.texture)}</div>{note}</div>'
        )
    grid = "".join(cells)
    return f'<h2>{escape(pair.pair_id)}</h2><div class="grid">{grid}</div>'


def build_report(
    results: list[PairResult],
    variants: list[Variant],
    summaries: list[VariantSummary],
    threshold: float,
    generated_at: str,
) -> str:
    recommended, reason = pick_recommended(variants, summaries, threshold)
    if recommended is not None:
        rec_html = (
            f'<div class="rec"><strong>Рекомендация:</strong> {escape(recommended.label)}'
            f'<br><span class="meta">{escape(reason)}</span></div>'
        )
    else:
        rec_html = f'<div class="rec none"><strong>Рекомендация:</strong> {escape(reason)}</div>'

    pair_blocks = "".join(_pair_block(p, variants, threshold) for p in results)
    meta_line = f"{escape(generated_at)} · пар: {len(results)} · порог {threshold:.2f}"

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B-стенд постобработки</title><style>{_CSS}</style></head>
<body>
<h1>A/B-стенд постобработки лица</h1>
<p class="meta">{meta_line}</p>
{rec_html}
<h2>Сводка по вариантам</h2>
{_summary_table(summaries, threshold)}
<p class="meta">Узнаваемость — косинус ArcFace (больше — лучше, красное ниже порога).
Рассогласование фактуры — расхождение лица и окружения (меньше — лучше).</p>
{pair_blocks}
</body></html>"""
