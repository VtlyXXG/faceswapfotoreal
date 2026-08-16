"""
Сводка по комплектам кадров: сходство, ступень тона, пряди, пересвет.

    python tools/measure_batch.py FINAL_ S1234_

Первая приставка считается опорной, разница показывается относительно неё.
Числа ложатся рядом в одну таблицу — иначе сравнивать нечем: у каждого прогона
своя генерация, и абсолютное значение само по себе ни о чём не говорит.

Сходство считается тем же экстрактором, что стоит в пайплайне; нужны веса
antelopev2 (см. bootstrap.py), иначе колонка выйдет пустой, а остальные мерки
посчитаются.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from _bench import (DONORS, OUT, ROOT, STEMS, TEMPLATES, changed_mask, detail_ratio,
                    frame_path, lightness, lower_zone, parse_prefixes, pose_shift,
                    strand_pixels, template_context, tone_step)

from app.pipelines import parsing  # noqa: E402  (путь добавлен в _bench)


def build_extractor():
    try:
        from app.pipelines.refine.adapter import AntelopeExtractor
        extractor = AntelopeExtractor(model_root=str(ROOT / "models" / "antelopev2"))
        extractor.load()
        extractor.warmup()
        return extractor
    except Exception as exc:
        print(f"вектор личности недоступен ({type(exc).__name__}), "
              f"сходство считаться не будет", file=sys.stderr)
        return None


def vector(extractor, image):
    if extractor is None or image is None:
        return None
    try:
        return np.asarray(extractor.extract(image).embedding, dtype=np.float64)
    except Exception:
        return None


def face_vector(extractor, frame, changed):
    """
    Кроп по области правки перед экстракцией.

    На развороте лицо занимает меньше пятой части кадра, и детектор его не берёт.
    Выравнивание идёт по опорным точкам, поэтому кроп на сам вектор не влияет —
    но применять его надо ОДИНАКОВО ко всем комплектам, иначе числа разъедутся.
    """
    x, y, w, h = cv2.boundingRect(changed.astype(np.uint8))
    pad = int(0.35 * max(w, h))
    crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    # именно `is not None`: у numpy-массива истинность неоднозначна, и `or` здесь
    # свалился бы с ValueError на первом же кадре
    found = vector(extractor, crop)
    return found if found is not None else vector(extractor, frame)


def measure(path, key, donor_vec, extractor):
    template, parsed, _silhouette, strand_zone, face_height, _own = template_context(key)
    frame = cv2.imread(str(path))
    changed = changed_mask(frame, template)
    skin_frame = np.asarray(parsing.parse(frame).bare_skin) > 127

    skin = (np.asarray(parsed.bare_skin) > 127) & skin_frame
    zone = lower_zone(changed)
    now = tone_step(frame, changed, skin, zone)
    norm = tone_step(template, changed, skin, zone)

    area = skin_frame & changed
    L = lightness(frame)
    face = face_vector(extractor, frame, changed)
    strands, strands_strict = strand_pixels(frame, template, strand_zone)

    return {
        # Единица — перепад «голова резче фона» сохранён как у шаблона. Меньше —
        # смысловой центр разворота перестал быть резче размытого фона за ним
        "detail": detail_ratio(frame, template, changed, face_height),
        # Ноль — ракурс персонажа сохранён. Мерка заведена после того, как
        # развёрнутую в фас голову приняли за рост сходства
        "pose": pose_shift(frame, key),
        "sim": None if (face is None or donor_vec is None) else float(np.dot(face, donor_vec)),
        # ступень СВЕРХ собственного перепада шаблона: у самого разворота между
        # подбородком и шеей есть законный перепад освещения, и загонять его в
        # ноль нельзя — на замере это переворачивало кадр с +0.4 на -6.3
        "resid": None if (now is None or norm is None) else now - norm,
        # Два счёта прядей, и читать надо строгий: у мерки с допуском вклейка,
        # подогнанная по тону, проваливается под порог и считается уцелевшим
        # волосом. Подробно — в `_bench.strand_pixels`
        "strands": strands,
        "strands_strict": strands_strict,
        # ровно 255, а не «ярче 250»: мягкий порог считает просто освещённую
        # щёку и показывает несуществующий регресс
        "clipped": 100.0 * float((area & (frame.max(axis=2) == 255)).sum()) / max(1, area.sum()),
        "maxL": float(np.max(L[area])) if area.sum() else None,
        "changed": 100.0 * changed.sum() / changed.size,
    }


def cell(value, spec):
    return "—".rjust(len(format(0, spec))) if value is None else format(value, spec)


def main() -> int:
    prefixes = parse_prefixes(sys.argv, ("FINAL_", "S1234_"))
    extractor = build_extractor()
    donor_vec, rows = {}, []

    for key in TEMPLATES:
        for tag in DONORS:
            stem = f"{key}_{tag}"
            if tag not in donor_vec:
                donor_vec[tag] = vector(extractor, cv2.imread(str(ROOT / DONORS[tag])))
            row = {"stem": stem}
            for prefix in prefixes:
                path = frame_path(prefix, stem)
                row[prefix] = measure(path, key, donor_vec[tag], extractor) if path.exists() else None
            rows.append(row)
            print(f"  посчитан {stem}", flush=True)

    (OUT / "measure_batch.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    for title, field, spec in (
        ("СХОДСТВО (0.30 — порог «тот же ребёнок»)", "sim", "7.3f"),
        ("ОСТАТОК СТУПЕНИ ТОНА, L* (ближе к нулю — лучше)", "resid", "+7.1f"),
        ("СМЕЩЕНИЕ РАКУРСА от шаблонного (0 — поза сохранена)", "pose", "7.2f"),
        ("РЕЗКОСТЬ ГОЛОВЫ относительно окружения, доля от шаблонной", "detail", "7.2f"),
        ("ПРЯДИ, СТРОГО — расхождение с шаблоном ровно ноль, px", "strands_strict", "7d"),
        ("ПРЯДИ С ДОПУСКОМ в 8 уровней, px (для сравнения с прежними прогонами)",
         "strands", "7d"),
        ("КОЖА, ВЫБИТАЯ В БЕЛОЕ (ровно 255), %", "clipped", "7.2f"),
    ):
        print(f"\n{title}")
        print(f"{'кадр':<16}" + "".join(f"{p.rstrip('_'):>8}" for p in prefixes))
        for row in rows:
            line = f"{row['stem']:<16}"
            for prefix in prefixes:
                value = (row[prefix] or {}).get(field) if row[prefix] else None
                line += cell(value, spec).rjust(8)
            print(line)

    print()
    for prefix in prefixes:
        got = [row[prefix] for row in rows if row[prefix]]
        if not got:
            continue
        sims = [g["sim"] for g in got if g["sim"] is not None]
        res = [abs(g["resid"]) for g in got if g["resid"] is not None]
        det = [g["detail"] for g in got if g.get("detail") is not None]
        line = f"{prefix.rstrip('_'):<8} кадров {len(got):2d}"
        if sims:
            line += f" | сходство сред {np.mean(sims):.3f} мин {min(sims):.3f}"
        if res:
            line += f" | ступень сред {np.mean(res):.1f} худшая {max(res):.1f} L*"
        if det:
            line += f" | резкость сред {np.mean(det):.2f} худшая {min(det):.2f}"
        pose = [g["pose"] for g in got if g.get("pose") is not None]
        if pose:
            line += f" | ракурс сред {np.mean(pose):.2f} худший {max(pose):.2f}"
        # Строгое число впереди и без скобок — читать надо его. С допуском стоит
        # рядом, чтобы прежние прогоны было с чем сравнить, а не чтобы усреднять
        line += (f" | пряди строго {sum(g['strands_strict'] for g in got)}px"
                 f" (с допуском {sum(g['strands'] for g in got)}px)"
                 f" | клиппинг худший {max(g['clipped'] for g in got):.2f}%")
        print(line)
    print(f"\nвсего кадров в наборе: {len(STEMS)}, таблица записана в storage/output/measure_batch.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
