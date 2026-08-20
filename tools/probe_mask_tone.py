"""
Стенд по двум находкам: переполнение маски и молчаливый отказ подгонки тона.

    python tools/probe_mask_tone.py --pages cover,spread_01,spread_10 \
                                    --donors donor6,donor2,customer --draws 2

ЧТО МЕРЯЕТСЯ И ЗАЧЕМ

1. ПЕРЕПОЛНЕНИЕ МАСКИ. Итоговая маска замены в `transplant` — объединение
   посчитанной по шаблону (постоянна) и посчитанной по ВЫРОВНЕННОЙ ГЕНЕРАЦИИ
   (каждый раз новая). Когда вторая крупнее или уезжает сдвигом, замена уходит
   с головы на туловище: на кадре появляются руки генерации поверх шаблонных.
   На разовой пробе это случилось в трёх бросках из пяти — стенд отвечает,
   насколько это системно по страницам и донорам.

2. ОТКАЗ ПОДГОНКИ ТОНА. Обе поправки — общая по сцене и отдельная по коже —
   умеют молча отказаться (низкая корреляция вне маски, мало кожи в кольце,
   гамма вне пределов). Отказ виден только в мете и наружу никак не показан,
   а именно он даёт «лицо другого оттенка, чем руки».

3. ЦЕНА ПРЕДПОЛАГАЕМОГО ЛЕЧЕНИЯ. Напрашивается ограничить маску окрестностью
   головы шаблона. Но у длинноволосой девочки на коротко стриженном персонаже
   причёска ЗАКОННО выходит за шаблонную голову, и слишком тесное ограничение
   срежет её. Поэтому на каждом кадре считается, сколько заменённых пикселей
   лежит за пределами шаблонной маски, расширенной на k высот лица, для
   нескольких k сразу: это и есть таблица размена «сколько отсечём лишнего
   против сколько отрежем нужного».

Кадры и мета кладутся в storage/output/ — он закрыт .gitignore, детские лица в
репозиторий не едут. Готовый кадр не перерисовывается: анализ можно доводить,
не тратя карту заново.

ЗЕРНО НЕ ЗАДАЁТСЯ НАМЕРЕННО. Замер должен повторять боевой путь, а там его нет;
вдобавок одинаковое зерно на оба прохода само по себе роняет сходство на ~0.19
(проверено). Повторяемость набирается числом бросков, а не фиксацией зерна.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-service"))

from app.pipelines import head_mask, transplant  # noqa: E402

PROXY = os.environ.get("GPU_PROXY", "http://195.209.216.8:8080")
BOOK = "dino_pixar_real"
TEMPLATES_DIR = Path(os.environ.get(
    "BOOK_TEMPLATES", r"D:\projectx-fallback\templates-books")) / BOOK
LOCAL_NAME = {"cover": "00_обложка.png",
              **{f"spread_{i:02d}": f"разворот_{i:02d}.png" for i in range(1, 11)}}

# Боевые доли демо-пути (GPU_DEMO_DILATE / _FEATHER / _NECK в server.py)
DILATE, FEATHER, NECK = 0.12, 0.10, 0.35

# Насколько расширять шаблонную маску, проверяя предполагаемое ограничение.
# 0 — ограничение впритык по шаблону, 1.0 — с запасом в целую высоту лица.
CLAMPS = (0.0, 0.25, 0.5, 1.0)

# Доноры: имя -> файл. `customer` — фотография от заказчика, он дал согласие
# тестировать на ней; лежит вне donors/, чтобы не смешиваться со стендовым набором
DONORS = {
    "donor2": ROOT / "donors" / "donor2.jpg",
    "donor6": ROOT / "donors" / "donor6.jpg",
    "customer": ROOT / "storage" / "input" / "customer-2026-08-20" / "donor.jpg",
}


def read_image(path: Path) -> np.ndarray | None:
    """Байтами через imdecode: имена шаблонов кириллические, imread их не берёт."""
    if not path.is_file():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def template_facts(page: str) -> dict:
    """Маска по шаблону и её геометрия. Считается один раз на страницу, на CPU."""
    image = read_image(TEMPLATES_DIR / LOCAL_NAME[page])
    if image is None:
        raise FileNotFoundError(f"нет шаблона для {page}")
    points = transplant._full_frame_landmarks(image)
    if points is None:
        raise ValueError("на шаблоне не строится сетка лица")
    geometry = head_mask.face_geometry(points)
    built = head_mask.build(image, DILATE, FEATHER, NECK)
    mask = np.asarray(built.mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    inside = mask > 127

    face_h = geometry["face_height"]
    grown = {}
    for k in CLAMPS:
        if k == 0:
            grown[k] = inside
            continue
        radius = max(1, int(round(face_h * k)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        grown[k] = cv2.dilate(inside.astype(np.uint8), kernel) > 0
    return {"image": image, "inside": inside, "grown": grown,
            "face_h": face_h, "bottom": int(np.nonzero(inside)[0].max())}


def render(page: str, donor_b64: str, out: Path) -> dict:
    if out.is_file():
        return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    req = urllib.request.Request(
        PROXY + "/v1/demo-render",
        data=json.dumps({"template_id": f"{BOOK}/{page}",
                         "donor_photo": donor_b64}).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        body = json.loads(response.read())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(body["image"]))
    meta = dict(body["meta"], spent=round(time.perf_counter() - started, 1))
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False),
                                        encoding="utf-8")
    return meta


def analyse(frame: np.ndarray, facts: dict, meta: dict) -> dict:
    """
    Что на самом деле заменено — по расхождению с шаблоном, а не по мете.

    Мета говорит, какой маской собирались заменять; расхождение показывает, что
    заменилось. Второе честнее: растушёвка и подгонка тона могут задеть пиксели
    за пределами маски.
    """
    changed = np.abs(frame.astype(np.int16) - facts["image"].astype(np.int16)).max(axis=2) > 32
    total = int(changed.sum())
    ys = np.nonzero(changed)[0]
    below = (int(ys.max()) - facts["bottom"]) if total else 0
    row = {
        "changed_px": total,
        "bottom": int(ys.max()) if total else 0,
        "below_template_mask": below,
        # Тот же заход В ДОЛЯХ ВЫСОТЫ ЛИЦА. Абсолютные пиксели тут обманывают:
        # ограничение задано долей, и на странице с крупным лицом сто пикселей
        # укладываются в допуск, а на странице с мелким — уже нет
        "below_faces": round(below / facts["face_h"], 3),
        "face_h": round(facts["face_h"], 1),
        # Сколько маски по генерации срезано ограничением. Ноль на всех кадрах
        # означал бы, что ограничение не работало вовсе
        "clipped_px": meta.get("clipped_px"),
    }
    for k in CLAMPS:
        outside = int((changed & ~facts["grown"][k]).sum())
        row[f"outside_{k}"] = outside
        row[f"outside_{k}_share"] = round(outside / total, 4) if total else 0.0
    row["tone"] = meta.get("matched")
    row["tone_reason"] = meta.get("match_reason")
    row["skin_tone"] = meta.get("skin_matched")
    row["skin_reason"] = meta.get("skin_reason")
    row["mask_old"] = meta.get("mask_old_px")
    row["mask_new"] = meta.get("mask_new_px")
    row["shift"] = meta.get("shift")
    row["scale"] = meta.get("scale")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="cover,spread_01,spread_07,spread_10")
    ap.add_argument("--donors", default="donor6,customer")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "storage" / "output" / "mask_tone"))
    ap.add_argument("--analyse-only", action="store_true",
                    help="ничего не рендерить, разобрать уже лежащее")
    args = ap.parse_args()

    out_root = Path(args.out)
    pages = args.pages.split(",")
    donors = [d for d in args.donors.split(",") if DONORS.get(d, Path("/nope")).is_file()]
    missing = set(args.donors.split(",")) - set(donors)
    if missing:
        print(f"нет фотографий: {', '.join(sorted(missing))}", file=sys.stderr)

    facts = {}
    for page in pages:
        try:
            facts[page] = template_facts(page)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{page}: {exc}", file=sys.stderr)

    rows = []
    for page in [p for p in pages if p in facts]:
        for donor in donors:
            photo = base64.b64encode(DONORS[donor].read_bytes()).decode()
            for draw in range(1, args.draws + 1):
                frame_path = out_root / page / f"{donor}_{draw}.png"
                if args.analyse_only and not frame_path.is_file():
                    continue
                try:
                    meta = render(page, photo, frame_path)
                except Exception as exc:
                    print(f"{page}/{donor}/{draw}: ОТКАЗ {exc}", file=sys.stderr)
                    continue
                row = analyse(read_image(frame_path), facts[page], meta)
                row.update(page=page, donor=donor, draw=draw)
                rows.append(row)
                print(f"{page:<11} {donor:<9} #{draw} "
                      f"ниже маски {row['below_template_mask']:>5} px  "
                      f"вне шаблона {row['outside_0.0_share']:>6.1%}  "
                      f"тон {str(row['tone']):>5} кожа {str(row['skin_tone']):>5}  "
                      f"сдвиг {row['shift']}", flush=True)

    if not rows:
        print("нечего разбирать", file=sys.stderr)
        return

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    print("\n" + "=" * 78)
    over = [r for r in rows if r["below_template_mask"] > 20]
    print(f"ПЕРЕПОЛНЕНИЕ МАСКИ: {len(over)} из {len(rows)} кадров заменили пиксели "
          f"ниже шаблонной маски более чем на 20 px")
    for page in sorted({r["page"] for r in rows}):
        page_rows = [r for r in rows if r["page"] == page]
        bad = [r for r in page_rows if r["below_template_mask"] > 20]
        worst = max(r["below_template_mask"] for r in page_rows)
        print(f"    {page:<11} {len(bad)}/{len(page_rows)}, худший заход {worst} px")

    print(f"\nОТКАЗ ПОДГОНКИ ТОНА:")
    for field, human in (("tone", "по сцене"), ("skin_tone", "по коже")):
        refused = [r for r in rows if r[field] is False]
        reasons = {r[f"{'tone' if field == 'tone' else 'skin'}_reason"] for r in refused}
        print(f"    {human:<9} отказала на {len(refused)} из {len(rows)} кадров"
              + (f", причины: {', '.join(str(x) for x in reasons)}" if refused else ""))

    print(f"\nЦЕНА ОГРАНИЧЕНИЯ МАСКИ (сколько замен пришлось бы отрезать):")
    print(f"    {'запас':>8} {'в среднем':>11} {'худший кадр':>13}")
    for k in CLAMPS:
        shares = [r[f"outside_{k}_share"] for r in rows]
        print(f"    {k:>8} {np.mean(shares):>10.1%} {max(shares):>12.1%}")
    print("    Читать так: чем больше запас, тем меньше отрежем законного —\n"
          "    в том числе длинных волос, законно выходящих за голову шаблона.")
    print(f"\nсырые числа: {out_root / 'rows.json'}")


if __name__ == "__main__":
    main()
