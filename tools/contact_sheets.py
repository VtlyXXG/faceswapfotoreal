"""
Контактные листы: комплекты друг над другом, по семь кадров в ряд.

    python tools/contact_sheets.py --out sheets FINAL_ S1234_

Числа не заменяют глаз. Оба дефекта, которые правились в пересадке, нашли именно
глазами на готовых кадрах, а метрика к ним подбиралась потом — и один раз
подобралась неверно, показав ровные 772 «пряди» и до, и после.

Кроп берётся по силуэту стирания с запасом: в кадр попадают стык шеи, контур
причёски и щека — всё, на чём эти дефекты видны.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _bench import DONORS, TEMPLATES, frame_path, template_context

TILE = 340
_box: dict[str, tuple] = {}


def window(key):
    """Окно вокруг головы персонажа, одно на все кадры шаблона."""
    if key not in _box:
        _template, _parsed, _sil, _zone, _fh, own = template_context(key)
        x, y, w, h = cv2.boundingRect((own * 255).astype(np.uint8))
        pad = int(0.45 * max(w, h))
        _box[key] = (max(0, y - pad), y + h + pad, max(0, x - pad), x + w + pad)
    return _box[key]


def tile(path, key, label, side=TILE):
    y0, y1, x0, x1 = window(key)
    crop = cv2.imread(str(path))[y0:y1, x0:x1]
    scale = side / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 24, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    strip = np.full((26, side, 3), 24, np.uint8)
    cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
    return np.vstack([strip, canvas])


def main() -> int:
    ap = argparse.ArgumentParser(description="контактные листы по комплектам")
    ap.add_argument("--out", default="sheets", help="куда класть листы")
    ap.add_argument("prefixes", nargs="*", default=["FINAL_", "S1234_"])
    args = ap.parse_args()

    prefixes = [p if p.endswith("_") else p + "_" for p in args.prefixes]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for key in TEMPLATES:
        rows = []
        for prefix in prefixes:
            row = [tile(frame_path(prefix, f"{key}_{tag}"), key, f"{prefix.rstrip('_')} {tag}")
                   for tag in DONORS if frame_path(prefix, f"{key}_{tag}").exists()]
            if row:
                rows.append(np.hstack(row))
        if not rows:
            print(f"{key}: кадров нет, лист пропущен")
            continue
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(24, 24, 24)) for r in rows]
        sheet = np.vstack(rows)
        cv2.imwrite(str(out / f"sheet_{key}.png"), sheet)
        print(f"sheet_{key}.png  {sheet.shape[1]}x{sheet.shape[0]}", flush=True)

    print(f"листы записаны в {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
