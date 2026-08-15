"""
Перегон 21 кадра через /v1/demo-render на GPU-сервере.

    python tools/render_batch.py --seed 1234 --prefix S1234_

Готовые кадры пропускаются, поэтому после обрыва скрипт просто запускают заново:
сервер за ночь уходил в перезагрузку дважды, и досчитывать с нуля было бы дорого
(развороты идут по полторы минуты каждый).

Отчёт пишется ПОСЛЕ КАЖДОГО кадра, а не в конце. Первый прогон оборвался на
двадцать первом кадре и унёс с собой метаданные всех двадцати посчитанных —
восстановить их можно было только пересчётом.

Сервер слушает на GPU-боксе; локально нужен туннель:

    ssh -f -N -L 8300:127.0.0.1:8300 deploy@<хост>
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time

import requests

from _bench import DONORS, OUT, ROOT, TEMPLATES


def b64(path):
    return base64.b64encode((ROOT / path).read_bytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description="перегон стендовых 21 кадра")
    ap.add_argument("--seed", type=int, required=True, help="зерно генерации")
    ap.add_argument("--prefix", required=True, help="приставка файлов, например S1234_")
    ap.add_argument("--url", default=os.environ.get("DEMO_URL", "http://127.0.0.1:8300"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.0)
    # Разбор одного дефекта не стоит сорока минут на все 21: обычно хватает трёх
    # кадров — сломанного, соседнего и заведомо целого для контроля
    ap.add_argument("--only", nargs="*", default=None,
                    help="считать только эти кадры, например dino1_id4 dino2_id6")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    meta_path = OUT / f"{args.prefix}meta.json"
    report = {}
    if meta_path.exists():
        try:
            report = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            report = {}

    done = failed = 0
    for key, template in TEMPLATES.items():
        for tag, donor in DONORS.items():
            name = f"{key}_{tag}"
            if args.only and name not in args.only:
                continue
            target = OUT / f"{args.prefix}{name}.png"
            if target.exists():
                print(f"{name}: уже есть, пропуск", flush=True)
                continue

            started = time.perf_counter()
            try:
                response = requests.post(f"{base}/v1/demo-render", timeout=1200, json={
                    "base_image": b64(template), "donor_photo": b64(donor),
                    "seed": args.seed, "steps": args.steps,
                    "guidance_scale": args.guidance,
                })
            except Exception as exc:
                failed += 1
                print(f"{name}: сеть {type(exc).__name__}: {exc}", flush=True)
                continue
            spent = time.perf_counter() - started

            if response.status_code >= 400:
                failed += 1
                print(f"{name}: сервер {response.status_code}: {response.text[:300]}", flush=True)
                continue
            body = response.json()
            image = body.get("image") or body.get("result") or body.get("base_image")
            if not image:
                failed += 1
                print(f"{name}: в ответе нет картинки, ключи: {list(body)}", flush=True)
                continue

            target.write_bytes(base64.b64decode(image))
            meta = body.get("meta", {})
            report[name] = meta
            done += 1
            meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            print(f"{name}: {spent:5.1f}s  skin_gamma={meta.get('skin_gamma', '-')}  "
                  f"matched={meta.get('matched', '-')}  erase={meta.get('erase_px', '-')}",
                  flush=True)

    print(f"готово: посчитано {done}, ошибок {failed}, всего в отчёте {len(report)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
