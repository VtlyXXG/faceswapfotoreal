"""
Прогон GPU-сервера через НАСТОЯЩИЙ код подготовки геометрии.

Зачем отдельный скрипт. Самодельный харнесс (эллипс по bbox + кроп с отступом)
даёт шум вместо сигнала: непонятно, плохое сходство — это предел IP-Adapter или
предел маски. Здесь не написано ни строчки собственной геометрии. Вызываются
ровно те функции, что работают в живом пути:

    маска      — head_mask.build(...)        как в pipeline.py::_geometry
    donor_crop — AntelopeExtractor.align(...) как в adapter.py::extract

Второе требует пояснения, потому что выглядит пугающе. `align` — classmethod, и
InsightFace ему не нужен: подобие строится по пяти опорам из сетки MediaPipe
(`_five_points`), а ONNX-сессия glintr100 нужна только шагом позже, в `_infer`,
для 512-мерного вектора. Вектор GPU-серверу не нужен вовсе — личность он берёт
из `donor_crop` через IP-Adapter, а `embedding` только валидирует. Поэтому здесь
берётся выравнивание и не берётся распознавание: несвободные веса не грузятся,
onnxruntime не импортируется.

    python test_real_geometry.py --target SHEET_p01_pixar_real.png --source donor.jpg.jpeg
    python test_real_geometry.py ... --dry-run          # только геометрия, без сети
    python test_real_geometry.py ... --donor-side 224   # проверить гипотезу про 112px

ЧТО СМОТРЕТЬ В РЕЗУЛЬТАТЕ. Первым делом — `mask.source` в отчёте. Если там
`parsing`, форму задала семантическая разметка, то есть маска повторяет реальный
контур причёски. Если `ellipse` — разметка не сработала и код сам упал на
запасной путь, и тогда ваш эллипс и этот эллипс отличаются только качеством
исполнения, а вывод о системе делать всё ещё рано.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVICE = ROOT / "ml-service"

# Тот же приём, что в runner.py:57 — пакет `app` живёт внутри ml-service, а
# скрипт лежит в корне рядом с тестовыми файлами
sys.path.insert(0, str(SERVICE))


# --- Геометрия: только вызовы, никакой своей логики --------------------------


def build_mask(target_bgr, dilate: float, feather: float, neck: float):
    """
    Маска головы персонажа. Ровно вызов из `pipeline.py::_geometry`.

    Доли по умолчанию берутся из `MaskProfile` — тех же чисел, что стоят в
    боевом профиле. Значения вынесены в аргументы командной строки не для
    красоты: сравнивать узкую и широкую маску имеет смысл только когда обе
    построены одним кодом.
    """
    from app.pipelines import head_mask

    return head_mask.build(
        target_bgr,
        dilate_ratio=dilate,
        feather_ratio=feather,
        neck_ratio=neck,
    )


def build_donor_crop(source_bgr, side: int, margin: float):
    """
    Выровненный кроп лица заказчика — то, что уезжает в `donor_crop`.

    Обе ручки существуют потому, что живой `donor_crop` собран под ДРУГОГО
    потребителя. 112x112 в кадрировке ArcFace — это вход распознавателя
    glintr100: ему нужны глаза, нос и рот, а причёска и форма головы только
    мешают, поэтому шаблон их отрезает. IP-Adapter Plus Face — не ArcFace: это
    CLIP-vision на 224, и он смотрит на лицо целиком, вместе с контуром головы.
    То есть по умолчанию ему приходит вдвое меньшее разрешение, чем он ждёт, и
    кадрировка, из которой выброшено то, на что он обучен смотреть.

    Обе ручки меняют ТОЛЬКО холст: матрица подобия остаётся той, что построил
    живой код, поэтому выравнивание, разворот и центрирование настоящие.

    :param side: сторона кропа. 112 — ровно как в живом коде (`_ARCFACE_SIDE`)
    :param margin: во сколько раз шире поле зрения. 1.0 — кадрировка живого
        кода, 1.6-2.0 возвращают в кадр лоб, причёску и овал
    """
    import cv2
    import numpy as np

    from app.pipelines import head_mask
    from app.pipelines.refine.adapter import AntelopeExtractor

    points = head_mask.try_landmarks(source_bgr)
    if points is None:
        raise SystemExit(
            "На фотографии не найдено лицо (MediaPipe). Нужен портрет, где лицо "
            "занимает больше пятой части кадра — тот же предел, что у живого кода."
        )

    # Подобие из живого кода: источник -> холст 112x112
    matrix = np.asarray(AntelopeExtractor._matrix(points), dtype=np.float64)

    # Пересчёт под другой холст и другое поле зрения. Точка q в 112-координатах
    # переносится как (q - центр) * k + side/2, где k = side / (112 * margin).
    # При side=112 и margin=1.0 преобразование тождественно, то есть умолчания
    # дают ровно тот кроп, что уходит в живом пакете
    scale = side / (112.0 * margin)
    matrix[:, :2] *= scale
    matrix[:, 2] = scale * (matrix[:, 2] - 56.0) + side / 2.0

    aligned = cv2.warpAffine(
        source_bgr, matrix, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Сколько кропа НЕ пришло из фотографии. Широкое поле на плотно кадрированном
    # портрете вылезает за край, и там BORDER_REPLICATE размазывает крайний
    # пиксель полосами. Для энкодера это не лицо, а шум, поэтому число видно в
    # отчёте: если оно заметное, margin надо уменьшать, а не любоваться охватом
    inside = cv2.warpAffine(
        np.full(source_bgr.shape[:2], 255, dtype=np.uint8),
        matrix,
        (side, side),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    extrapolated = 1.0 - float(np.count_nonzero(inside)) / float(side * side)

    geometry = head_mask.face_geometry(points)
    face_share = float(geometry["face_height"]) / max(1, min(source_bgr.shape[:2]))
    return aligned, {
        "side": side,
        "margin": margin,
        "face_height_px": round(float(geometry["face_height"]), 1),
        "face_share": round(face_share, 3),
        # Во сколько раз CLIP-энкодеру придётся ДОРИСОВАТЬ вход интерполяцией.
        # Больше единицы — часть того, что видит адаптер, синтетическая
        "upscaled_by_clip": round(224.0 / side, 2),
        "extrapolated_share": round(extrapolated, 3),
        "source_size": [int(source_bgr.shape[1]), int(source_bgr.shape[0])],
    }


# --- Отчёт о маске -----------------------------------------------------------


def mask_report(head, target_bgr) -> dict:
    """Числа, по которым видно, эллипс это или контур головы."""
    import numpy as np

    mask = head.mask
    total = float(mask.shape[0] * mask.shape[1])
    solid = int(np.count_nonzero(mask == 255))
    soft = int(np.count_nonzero((mask > 0) & (mask < 255)))
    ys, xs = np.nonzero(mask > 0)

    box = None
    fill = None
    if ys.size:
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        box = [left, top, right - left + 1, bottom - top + 1]
        # Доля заполнения описанного прямоугольника. Само по себе это НЕ признак
        # эллипса: у настоящего контура головы с причёской она тоже выходит
        # около 0.79. Отличать форму по этому числу нельзя — смотрите
        # `build_meta.source` и mask_overlay.png
        fill = round(float(np.count_nonzero(mask > 0)) / float(box[2] * box[3]), 3)

    return {
        "face_height_px": round(float(head.face_height), 1),
        "solid_share": round(solid / total, 4),
        "feathered_px": soft,
        "bbox": box,
        "bbox_fill": fill,
        # Главное поле: чем построена маска. Его же кладёт в X-Swap-Meta живой путь
        "build_meta": head.meta,
    }


def overlay(target_bgr, mask):
    """Кадр с подсвеченной маской — чтобы форму было видно глазом, а не по числам."""
    import cv2
    import numpy as np

    tint = target_bgr.copy()
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    red = np.zeros_like(target_bgr)
    red[:, :] = (0, 0, 255)
    tint = (target_bgr * (1 - alpha * 0.45) + red * (alpha * 0.45)).astype(np.uint8)

    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(tint, contours, -1, (0, 255, 255), 2)
    return tint


# --- Запрос ------------------------------------------------------------------


def build_packet(target_bgr, mask, donor, prompt: str, seed: int | None, args) -> dict:
    """
    JSON под контракт `RenderRequest` из server.py.

    Формат полей — как в `refine/payload.py::build_packet`: базовый кадр JPEG
    (он тяжёлый, а под маской всё равно перерисовывается), маска и кроп PNG
    (маска — это веса, а не картинка, и блочность JPEG делает ненулевыми
    пиксели, которые были чистым нулём).
    """
    from app.utils.image import encode_image

    base_raw, base_mime = encode_image(target_bgr, "jpg", 95)
    mask_raw, mask_mime = encode_image(mask, "png")
    donor_raw, donor_mime = encode_image(donor, "png")

    def b64(raw: bytes) -> str:
        return base64.b64encode(raw).decode("ascii")

    packet: dict = {
        "base_image": b64(base_raw),
        "mask_image": b64(mask_raw),
        "donor_crop": b64(donor_raw),
        "prompt": prompt,
        "output_format": "png",
        "encoding": {
            "base_image": base_mime,
            "mask_image": mask_mime,
            "donor_crop": donor_mime,
        },
        "bytes": {
            "base_image": len(base_raw),
            "mask_image": len(mask_raw),
            "donor_crop": len(donor_raw),
        },
    }
    if seed is not None:
        packet["seed"] = seed
    for name in ("steps", "guidance_scale", "strength", "control_scale"):
        value = getattr(args, name)
        if value is not None:
            packet[name] = value
    return packet


def post(url: str, packet: dict, timeout: float):
    import requests

    response = requests.post(f"{url.rstrip('/')}/v1/template-render", json=packet, timeout=timeout)
    if response.status_code >= 400:
        raise SystemExit(f"сервер ответил {response.status_code}: {response.text[:400]}")
    return response.json()


# --- Оркестрация -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Прогон GPU-сервера через настоящий код подготовки маски и donor_crop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="шаблон-разворот с персонажем")
    parser.add_argument("--source", required=True, help="фотография заказчика")
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="адрес GPU-сервера")
    parser.add_argument("--out", default="test_out", help="куда складывать артефакты")
    parser.add_argument("--prompt", default="", help="пусто — сервер подставит GPU_PROMPT")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true", help="только геометрия, без сети")

    # Доли маски. Умолчания — из MaskProfile, то есть боевые
    parser.add_argument("--dilate", type=float, default=0.12)
    parser.add_argument("--feather", type=float, default=0.10)
    parser.add_argument("--neck", type=float, default=0.35)

    parser.add_argument("--donor-side", type=int, default=112, help="112 = как в живом коде")
    parser.add_argument(
        "--donor-margin",
        type=float,
        default=1.0,
        dest="donor_margin",
        help="1.0 = кадрировка живого кода; 1.6-2.0 вернут в кадр лоб и причёску",
    )

    # Необязательные ручки диффузии. None — не класть в пакет вовсе, пусть
    # сервер возьмёт свои умолчания
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--control-scale", type=float, default=None, dest="control_scale")

    args = parser.parse_args(argv)

    from app.utils.image import decode_image, encode_image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_bytes = Path(args.target).read_bytes()
    source_bytes = Path(args.source).read_bytes()
    target_bgr = decode_image(target_bytes)
    source_bgr = decode_image(source_bytes)

    print(f"шаблон: {args.target} {target_bgr.shape[1]}x{target_bgr.shape[0]}")
    print(f"фото:   {args.source} {source_bgr.shape[1]}x{source_bgr.shape[0]}")

    head = build_mask(target_bgr, args.dilate, args.feather, args.neck)
    report = mask_report(head, target_bgr)
    print("\n--- маска ---")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    donor, donor_meta = build_donor_crop(source_bgr, args.donor_side, args.donor_margin)
    print("\n--- donor_crop ---")
    print(json.dumps(donor_meta, ensure_ascii=False, indent=2))

    (out / "mask.png").write_bytes(encode_image(head.mask, "png")[0])
    (out / "mask_overlay.png").write_bytes(encode_image(overlay(target_bgr, head.mask), "png")[0])
    (out / "donor_crop.png").write_bytes(encode_image(donor, "png")[0])
    (out / "geometry.json").write_text(
        json.dumps({"mask": report, "donor_crop": donor_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nгеометрия сохранена в {out}/ — посмотрите mask_overlay.png глазами")

    if args.dry_run:
        return 0

    packet = build_packet(target_bgr, head.mask, donor, args.prompt, args.seed, args)
    weight = sum(packet["bytes"].values())
    print(f"\nпакет собран: {weight / 1024:.0f} КБ картинок, POST {args.url}")

    body = post(args.url, packet, args.timeout)
    image = base64.b64decode(body["image"])
    (out / "result.png").write_bytes(image)
    (out / "meta.json").write_text(
        json.dumps(body.get("meta", {}), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = body.get("meta", {})
    print("\n--- ответ сервера ---")
    for key in ("identity_applied", "ip_adapter", "identity_scale", "controlnet",
                "prompt_defaulted", "erase_backend", "faces_restored", "ignored", "total_s"):
        if key in meta:
            print(f"  {key}: {meta[key]}")
    print(f"\nрезультат: {out}/result.png")

    if not meta.get("identity_applied"):
        print(
            "\nВНИМАНИЕ: identity_applied=false — донор до генерации не доехал. "
            "Смотреть meta.ip_adapter и meta.ignored: сходства в этом кадре нет "
            "и быть не могло, тюнить по нему нечего."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
