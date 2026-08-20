"""
Пряди прежнего героя, дошедшие до печати побитово.

    python tools/strand_count.py FINAL_ S1234_

Считаются тёмные пиксели фона у самого контура стирания, НЕ изменившиеся
относительно шаблона: раз пиксель не тронут — значит вклейка прошла мимо, и
волосок прежнего персонажа доехал до готового разворота.

Колонок две, и читать надо СТРОГУЮ — там расхождение с шаблоном ровно ноль.
Колонка с допуском в восемь уровней оставлена для сравнения с прежними
прогонами, но врёт в предсказуемую сторону: подогнанная по тону вклейка
проваливается под порог и записывается в уцелевшие волосы. Разбор — в
`_bench.strand_pixels`.

Ширина полосы (16 px) — не украшение. На широком окне головы мерка давала ровные
772 px и до правки, и после, потому что считала тёмный фон вдалеке от головы и
не разделяла комплекты вовсе. В полосе у контура те же кадры дают 45 и 15 px до
правки против нуля после.

ОГОВОРКА, без которой числа врут: на dino2 и spread_08 в полосу попадает
детальная листва, неотличимая от волоска по этой мерке. Там число говорит
«столько тёмного осталось нетронутым», а не «столько прядей». Разделить их
можно только разметкой.
"""
from __future__ import annotations

import json
import sys

import cv2

from _bench import (DONORS, OUT, STRAND_BAND_PX, TEMPLATES, frame_path, parse_prefixes,
                    strand_pixels, template_context)


def main() -> int:
    prefixes = parse_prefixes(sys.argv, ("FINAL_", "S1234_"))
    result = {}

    print(f"полоса {STRAND_BAND_PX} px вдоль контура стирания")
    print("в каждой паре: строго (расхождение ровно ноль) / с допуском в 8 уровней")
    print(f"{'кадр':<16}" + "".join(f"{p.rstrip('_'):>16}" for p in prefixes))
    for key in TEMPLATES:
        for tag in DONORS:
            stem = f"{key}_{tag}"
            row, line = {}, f"{stem:<16}"
            for prefix in prefixes:
                path = frame_path(prefix, stem)
                if not path.exists():
                    row[prefix] = None
                    line += "—".rjust(16)
                    continue
                template, _parsed, _sil, strand_zone, _fh, _own = template_context(key)
                loose, strict = strand_pixels(cv2.imread(str(path)), template, strand_zone)
                row[prefix] = {"strict": strict, "loose": loose}
                line += f"{strict:>9d} /{loose:>5d}"
            result[stem] = row
            print(line, flush=True)

    (OUT / "strand_count.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    for prefix in prefixes:
        got = [v[prefix] for v in result.values() if v[prefix] is not None]
        if got:
            strict = [g["strict"] for g in got]
            print(f"{prefix.rstrip('_'):<8} строго всего {sum(strict):>7d} px, "
                  f"худший кадр {max(strict):>6d} px, чистых кадров "
                  f"{strict.count(0)}/{len(strict)} "
                  f"(с допуском всего {sum(g['loose'] for g in got)} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
