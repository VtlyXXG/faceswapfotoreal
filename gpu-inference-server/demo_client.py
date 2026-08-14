"""
Клиент демонстрационного эндпоинта: фотография и разворот -> готовая картинка.

Один запрос на весь путь. Ни маску, ни кроп лица готовить не нужно — сервер
считает их сам.

    python demo_client.py <разворот> <фото ребёнка> <куда сохранить.png>

Адрес сервиса берётся из переменной DEMO_URL, по умолчанию http://127.0.0.1:8300
(то есть через SSH-туннель: ssh -L 8300:127.0.0.1:8300 deploy@<хост>).
"""

import base64
import json
import os
import sys
import time

import requests

BASE = os.environ.get("DEMO_URL", "http://127.0.0.1:8300").rstrip("/")


def _b64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def run(template: str, photo: str, out: str) -> float | None:
    packet = {"base_image": _b64(template), "donor_photo": _b64(photo)}
    started = time.perf_counter()
    response = requests.post(f"{BASE}/v1/demo-render", json=packet, timeout=900)
    elapsed = time.perf_counter() - started

    if response.status_code >= 400:
        # Текст ошибки осмысленный: сервер отвечает «лицо не найдено», «нет
        # персонажа на шаблоне» и подобным, и показывать его стоит целиком
        print(f"сервер ответил {response.status_code}: {response.text[:400]}")
        return None

    answer = response.json()
    with open(out, "wb") as handle:
        handle.write(base64.b64decode(answer["image"]))

    meta = answer["meta"]
    print(f"готово за {elapsed:.1f} с (на сервере {meta['total_s']} с) -> {out}")
    print(f"  по шагам: {json.dumps(meta['timings'], ensure_ascii=False)}")
    print(f"  лицо на фото {meta['donor_face_px']} px, за краем кадра "
          f"{meta['donor_outside']:.0%}, масштаб посадки {meta['scale']}")
    return elapsed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
