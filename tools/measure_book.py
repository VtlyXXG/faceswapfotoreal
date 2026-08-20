"""
Сходство личности по страницам книги.

    python tools/measure_book.py --donors donor6,donor2 --pages all

Отвечает на вопрос, который до сих пор закрывался только глазами: насколько
похож ребёнок на КАЖДОЙ странице книги. Прежние числа проекта (0,52–0,59) сняты
на старом стендовом наборе, которого в поставке заказчика нет ни одним файлом,
и переносить их на книгу нельзя.

ПОЧЕМУ ОТДЕЛЬНЫЙ ИНСТРУМЕНТ, А НЕ `measure_batch.py`. Тот привязан к стендовому
набору жёстко — три шаблона на семь личностей, и пары «кадр — донор» там менять
запрещено, иначе числа не сравнить с прежними. Книга — другой набор и другой
вопрос, и ломать ради неё стенд значит потерять и то и другое.

КРОП НЕ ОБЯЗАТЕЛЕН, И ЭТО ПРОВЕРЕНО. Ожидалось, что на развороте 2048x2048 лицо
окажется мельче предела разметки и придётся кропить область головы. На деле
детектор antelopev2 (scrfd) находит лицо и на полном кадре: известный предел
~20% кадра — про сетку MediaPipe в пайплайне, а не про этот детектор. Проверено
сравнением векторов: полный кадр против кропа дал 0.96, то есть детектор берёт
ту же голову, а не соседнего персонажа.

Считаются ОБА числа. Кроп нормирует кадрирование: окно всегда 2,2 высоты лица,
одинаково на всех страницах, тогда как на полном кадре в поле зрения попадает
разное. Для сравнения СТРАНИЦ МЕЖДУ СОБОЙ это важнее, поэтому опорным считается
кроп, а полный кадр идёт рядом как проверка, что мерили ту же голову.

Абсолютные значения со стендовыми числами из `tools/` не сопоставимы: там другое
кадрирование. Сравнивать можно страницы между собой и прогоны между собой.
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
os.environ.setdefault("ML_ANTELOPE_ROOT", str(ROOT / "models" / "antelopev2"))

from app.pipelines import head_mask, transplant  # noqa: E402
from app.pipelines.refine.adapter import AntelopeExtractor  # noqa: E402

PROXY = os.environ.get("GPU_PROXY", "http://195.209.216.8:8080")
BOOK = "dino_pixar_real"
PAGES = ["cover"] + [f"spread_{i:02d}" for i in range(1, 11)]

# Локальные шаблоны лежат с кириллическими именами — теми, что прислал заказчик.
# На боксе они переименованы в ascii, потому что идентификатор едет в запросе.
TEMPLATES_DIR = Path(os.environ.get(
    "BOOK_TEMPLATES", r"D:\projectx-fallback\templates-books")) / BOOK
LOCAL_NAME = {"cover": "00_обложка.png",
              **{f"spread_{i:02d}": f"разворот_{i:02d}.png" for i in range(1, 11)}}

# Окно вокруг головы в высотах лица. 2,2 — чтобы в кадр попали лоб и подбородок
# с запасом на причёску, но не полезли соседние персонажи.
BOX_FACES = 2.2
UPSCALE = 2


def read_image(path: Path) -> np.ndarray | None:
    """
    Байтами через `imdecode`, а не путём через `imread`.

    `imread` берёт имя файла в кодировке системы и на не-ascii именах молча
    возвращает None — а шаблоны лежат с кириллическими именами. Ровно про это
    предупреждает `read_template` на сервере.
    """
    if not path.is_file():
        return None
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def render(page: str, donor_b64: str, out: Path, passes: int | None,
           seed: int | None = None) -> dict:
    """
    Страница через боевой прокси. Готовый файл не перерисовывается.

    ЗЕРНО ЗАДАЁТСЯ ЯВНО, И ЭТО НЕ ПЕДАНТИЗМ. Без него каждый вызов — новый
    бросок: два рендера одной страницы одним донором дали сходство 0.4053 и
    0.6681, а между собой 0.5454. Разброс между прогонами оказался больше любой
    разницы между страницами, которую замер должен был найти, — то есть по
    одному кадру на страницу мерить бессмысленно.

    С явным зерном выдача побайтово повторяется (проверено дважды, md5
    d82918d157cb1445cfb7b93d3e7d6e9b), поэтому страница меряется НЕСКОЛЬКИМИ
    зёрнами, а результат читается как разброс, а не как одно число.
    """
    if out.is_file():
        return {"cached": True}

    payload = {"template_id": f"{BOOK}/{page}", "donor_photo": donor_b64}
    if passes:
        payload["passes"] = passes
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        PROXY + "/v1/demo-render", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        body = json.loads(response.read())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(body["image"]))
    meta = body["meta"]
    return {"cached": False, "spent": time.perf_counter() - started,
            "passes": meta.get("passes"), "steps": meta.get("steps"),
            "face_px": meta.get("face_px_generated")}


def head_box(template: np.ndarray) -> tuple[int, int, int, int]:
    """Окно вокруг головы персонажа по разметке ШАБЛОНА: голова садится туда же."""
    points = transplant._full_frame_landmarks(template)
    if points is None:
        raise ValueError("на шаблоне не нашлась сетка лица")
    geometry = head_mask.face_geometry(points)
    centre = geometry["chin"] + geometry["up"] * geometry["face_height"] * 0.5
    half = geometry["face_height"] * BOX_FACES / 2
    h, w = template.shape[:2]
    return (max(0, int(centre[0] - half)), max(0, int(centre[1] - half)),
            min(w, int(centre[0] + half)), min(h, int(centre[1] + half)))


def embed(extractor, image: np.ndarray) -> np.ndarray | None:
    got = extractor.extract(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if got is None:
        return None
    return np.asarray(got.embedding, dtype=np.float64).ravel()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", default="donor6", help="через запятую, без .jpg")
    ap.add_argument("--pages", default="all", help="all либо cover,spread_01,...")
    ap.add_argument("--passes", type=int, default=None,
                    help="по умолчанию умолчание сервера — то есть книга, два прохода")
    ap.add_argument("--seeds", default="1234,777,20260820",
                    help="зёрна через запятую; на каждое считается свой кадр")
    ap.add_argument("--out", default=str(ROOT / "storage" / "output" / "book_quality"))
    args = ap.parse_args()

    pages = PAGES if args.pages == "all" else args.pages.split(",")
    donors = args.donors.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    out_root = Path(args.out)

    extractor = AntelopeExtractor(model_root=os.environ["ML_ANTELOPE_ROOT"])
    extractor.load()

    # Окна и векторы доноров считаются один раз: они от страницы не зависят
    boxes: dict[str, tuple] = {}
    for page in pages:
        template = read_image(TEMPLATES_DIR / LOCAL_NAME[page])
        if template is None:
            print(f"{page}: шаблона нет локально, пропущен", file=sys.stderr)
            continue
        try:
            boxes[page] = head_box(template)
        except ValueError as exc:
            print(f"{page}: {exc}", file=sys.stderr)

    donor_vectors: dict[str, np.ndarray] = {}
    for donor in donors:
        photo = read_image(ROOT / "donors" / f"{donor}.jpg")
        if photo is None:
            print(f"{donor}: фотографии нет, пропущен", file=sys.stderr)
            continue
        vector = embed(extractor, photo)
        if vector is None:
            print(f"{donor}: лицо на фотографии не найдено, пропущен", file=sys.stderr)
            continue
        donor_vectors[donor] = vector

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "  нет "

    print(f"\n{'страница':<12} {'донор':<8} {'зерно':>9} {'кроп':>7} {'полный':>8} "
          f"{'та же?':>7} {'шагов':>6} {'сек':>7}")
    print("-" * 72)

    rows: list[dict] = []
    for donor, donor_vector in donor_vectors.items():
        photo_b64 = base64.b64encode(
            (ROOT / "donors" / f"{donor}.jpg").read_bytes()).decode()
        for page in pages:
            if page not in boxes:
                continue
            # На каждое зерно свой кадр и своя строка: страница характеризуется
            # РАЗБРОСОМ по зёрнам, а не одним броском
            for seed in seeds:
                frame_path = out_root / donor / f"{page}_s{seed}.png"
                try:
                    info = render(page, photo_b64, frame_path, args.passes, seed)
                except Exception as exc:
                    print(f"{page:<12} {donor:<8} {seed:>9} ОТКАЗ: {exc}")
                    continue

                frame = read_image(frame_path)
                x0, y0, x1, y1 = boxes[page]
                piece = cv2.resize(frame[y0:y1, x0:x1], None, fx=UPSCALE, fy=UPSCALE,
                                   interpolation=cv2.INTER_CUBIC)

                v_crop = embed(extractor, piece)
                v_full = embed(extractor, frame)
                crop_score = cosine(v_crop, donor_vector) if v_crop is not None else None
                full_score = cosine(v_full, donor_vector) if v_full is not None else None
                # Кроп против полного кадра: низкое значение означает, что на
                # полном кадре детектор взял не ту голову, и числу веры нет
                agree = (cosine(v_crop, v_full)
                         if v_crop is not None and v_full is not None else None)

                rows.append({"page": page, "donor": donor, "seed": seed,
                             "crop": crop_score, "full": full_score,
                             "agree": agree, **info})
                print(f"{page:<12} {donor:<8} {seed:>9} {fmt(crop_score):>7} "
                      f"{fmt(full_score):>8} {fmt(agree):>7} "
                      f"{str(info.get('steps') or '-'):>6} "
                      f"{info.get('spent', 0):>7.1f}")

    print("-" * 72)
    # Сводка по страницам: у каждой свой разброс, и именно он, а не среднее,
    # отвечает на вопрос «что получит заказчик». Среднее прячет провальный бросок
    for page in pages:
        scores = [r["crop"] for r in rows if r["page"] == page and r["crop"] is not None]
        if not scores:
            continue
        print(f"{page:<12} зёрен {len(scores)}: худшее {min(scores):.4f}, "
              f"среднее {np.mean(scores):.4f}, лучшее {max(scores):.4f}, "
              f"размах {max(scores) - min(scores):.4f}")

    scored = [r["crop"] for r in rows if r["crop"] is not None]
    if scored:
        worst = min((r for r in rows if r["crop"] is not None), key=lambda r: r["crop"])
        print(f"\nвсего кадров {len(scored)}: худший {min(scored):.4f} "
              f"({worst['page']}, зерно {worst['seed']}), "
              f"среднее {np.mean(scored):.4f}, лучший {max(scored):.4f}")

    report = out_root / "measured.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсырые числа: {report}")


if __name__ == "__main__":
    main()
