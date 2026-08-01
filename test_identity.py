"""
Локальный прогон пайплайна: маска головы на шаблоне → генерация на fal.

Зачем отдельный скрипт. Пайплайн состоит из частей с совершенно разной ценой
ошибки: маска считается локально и бесплатно, генерация стоит денег и минут
ожидания. Смотреть на них надо порознь — сначала убедиться, что маска накрыла
ровно голову персонажа, и только потом платить за инференс.

    # Только маска: ключ не нужен, сеть не нужна, стоит ноль
    ml-service\\.venv\\Scripts\\python.exe test_identity.py --dry-run

    # Полный прогон (нужен FAL_KEY в окружении)
    ml-service\\.venv\\Scripts\\python.exe test_identity.py

    # Запасной путь: тот же редактор по двум картинкам, но другой моделью —
    # для шаблонов, где голова персонажа стилизована
    ml-service\\.venv\\Scripts\\python.exe test_identity.py --strategy kontext_multi ^
        --endpoint fal-ai/nano-banana/edit --payload nano_banana

    # Другой шаблон, другой донор, стиль живописи, маска пошире
    ml-service\\.venv\\Scripts\\python.exe test_identity.py ^
        --target spread_03.png --source "фото.jpg" --style impasto --dilate 0.18

Что кладётся в outputs/ на каждый прогон:

    <имя>_mask.png     сама маска. В fal она не уезжает ни на одном пути: на
                       диффузионных по ней голова возвращается в шаблон, на
                       фейссвопе она вообще не используется — там это только
                       проверка, что лицо на шаблоне различимо;
    <имя>_overlay.png  маска красным поверх шаблона — по нему и смотрят,
                       не срезана ли причёска и не залезла ли маска на плечи;
    <имя>_reference.jpg фотография в том виде, в каком её увидит модель;
    <имя>_result.png   готовый разворот (только при полном прогоне).

Эталонные листы SHEET_p01_impasto.png и SHEET_p03_pixar_real.png — это сетки
«фото → желаемый итог», а не входные данные: в них по шесть панелей в ряд.
Прогнать пайплайн на панели из такого листа можно, вырезав её: --crop x,y,w,h.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVICE = ROOT / "ml-service"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогон Generative Identity-Inpaint на локальных файлах",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="spread_01.png", help="шаблон-разворот")
    parser.add_argument("--source", default="source.jpg", help="фотография заказчика")
    parser.add_argument("--out", default=str(SERVICE / "outputs"), help="куда складывать")

    parser.add_argument("--profile", default="", help="pixar_real | impasto")
    parser.add_argument(
        "--strategy",
        default="",
        help="подход: face_swap (специализированный фейссвоп, рабочий) | kontext_multi "
        "(диффузия по кадру целиком + локальная вклейка) | identity_inpaint (инпейнт по "
        "маске). Тянет за собой эндпоинт, схему запроса, промпт и локальную работу — "
        "задавать их отдельно не нужно",
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help="эндпоинт вручную — сильнее умолчания стратегии. Вместе с --payload "
        "переключает kontext_multi на другой редактор по двум картинкам: "
        "nano_banana | seedream_edit | hy_wu_edit",
    )
    parser.add_argument("--payload", default="", help="схема запроса, см. --endpoint")
    parser.add_argument("--style", default="", help="стиль сцены в промпте")
    parser.add_argument("--scene", default="", help="дополнение к промпту стиля")
    parser.add_argument("--emotion", default="", help="мимика; пусто — как на шаблоне")
    parser.add_argument("--strength", type=float, help="сила генерации под маской")
    parser.add_argument("--steps", type=int, help="шагов инференса")
    parser.add_argument("--guidance", type=float, help="guidance scale")
    parser.add_argument(
        "--id-scale",
        type=float,
        help="вес идентичности: ниже — больше свободы повернуть голову и открыть рот. "
        "Работает только на схемах, где у референса есть свой вес (flux_general); "
        "на схемах kontext в тело запроса не попадает — скрипт предупредит",
    )

    parser.add_argument(
        "--ref-pad-max", type=float, help="предел отдаления референса; 1.0 — выключить"
    )
    parser.add_argument(
        "--ref-pad", type=float, help="слепое поле референса, когда лицо на фото не нашлось"
    )

    parser.add_argument("--dilate", type=float, help="расширение маски, доля высоты лица")
    parser.add_argument("--feather", type=float, help="растушёвка краёв маски")
    parser.add_argument("--neck", type=float, help="глубина захвата шеи под подбородком")

    parser.add_argument(
        "--crop",
        default="",
        help="вырезать область шаблона перед прогоном: x,y,w,h (для панелей SHEET-листов)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только маска и предпросмотр запроса — без обращения к fal",
    )
    return parser.parse_args()


def apply_environment(args: argparse.Namespace) -> None:
    """
    Переопределения уезжают в окружение ДО импорта app.

    Настройки читаются один раз при импорте (кэшированный Settings), поэтому
    менять их потом бессмысленно: профиль соберётся из того, что было на старте.
    """
    overrides = {
        "ML_REFINE_PROFILE": args.profile,
        # Стратегия приносит с собой эндпоинт, схему запроса и инструкцию:
        # порознь они не имеют смысла, см. profiles.StrategyDefaults
        "ML_REFINE_STRATEGY": args.strategy,
        "ML_REFINE_ENDPOINT": args.endpoint,
        "ML_REFINE_PAYLOAD": args.payload,
        "ML_REFINE_STYLE": args.style,
        "ML_REFINE_SCENE": args.scene,
        "ML_REFINE_STRENGTH": args.strength,
        "ML_REFINE_STEPS": args.steps,
        "ML_REFINE_GUIDANCE_SCALE": args.guidance,
        "ML_REFINE_IDENTITY_SCALE": args.id_scale,
        "ML_MASK_DILATE_RATIO": args.dilate,
        "ML_MASK_FEATHER_RATIO": args.feather,
        "ML_MASK_NECK_RATIO": args.neck,
        "ML_REFERENCE_PAD_MAX": args.ref_pad_max,
        "ML_REFERENCE_PAD_RATIO": args.ref_pad,
    }
    for key, value in overrides.items():
        if value not in (None, ""):
            os.environ[key] = str(value)

    # .env сервиса лежит рядом с ним, а запускаемся мы из корня проекта
    os.environ.setdefault("ML_LOG_DIR", str(SERVICE / "logs"))
    sys.path.insert(0, str(SERVICE))


def read(path: Path) -> bytes:
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    return path.read_bytes()


def crop(data: bytes, spec: str) -> bytes:
    """Вырезает область кадра: панель эталонного листа — тоже шаблон."""
    from app.utils.image import decode_image, encode_image

    try:
        x, y, width, height = (int(part) for part in spec.split(","))
    except ValueError:
        raise SystemExit("--crop ждёт четыре числа: x,y,w,h") from None

    image = decode_image(data)[y : y + height, x : x + width]
    if image.size == 0:
        raise SystemExit(f"Область {spec} лежит вне кадра")
    return encode_image(image, "png")[0]


def overlay(target: bytes, mask) -> bytes:
    """
    Маска красным поверх шаблона — единственный способ увидеть её глазами.

    Полутона краёв показаны как есть: по ним видно ширину растушёвки, а она и
    решает, будет ли граница генерации заметна на готовом развороте.
    """
    import numpy as np

    from app.utils.image import decode_image, encode_image

    image = decode_image(target).astype(np.float32)
    weight = (mask.astype(np.float32) / 255.0 * 0.55)[..., None]
    red = np.zeros_like(image)
    red[..., 2] = 255.0

    return encode_image((image * (1 - weight) + red * weight).astype(np.uint8), "png")[0]


def main() -> int:
    args = parse_args()
    apply_environment(args)

    from app.pipelines import expression, head_mask, refine
    from app.pipelines.face_swap import pipeline
    from app.utils.image import decode_image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.target).stem

    target = read(ROOT / args.target if not Path(args.target).is_absolute() else Path(args.target))
    source = read(ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source))
    if args.crop:
        target = crop(target, args.crop)

    profile = refine.profiles.from_settings()
    print("\n=== Профиль ===")
    print(json.dumps(profile.report(), ensure_ascii=False, indent=2))

    if args.id_scale is not None and profile.payload.identity_kind != "list":
        # Молча принять ручку, которая ни на что не влияет, — худшее, что тут
        # можно сделать: подбор пойдёт по прогонам, отличающимся только ценой.
        # У схемы kontext референс передаётся голой ссылкой, веса у него нет.
        print(
            f"\nВНИМАНИЕ: --id-scale={args.id_scale} не уйдёт в запрос.\n"
            f"  Схема «{profile.payload.name}» передаёт референс полем "
            f"{profile.payload.identity_field} без веса, и тело запроса при любом\n"
            f"  значении получается одно и то же. Вес живёт только у схем с\n"
            f"  identity_kind=list — переключается через ML_REFINE_PAYLOAD=flux_general.\n"
            f"  На ракурс и мимику здесь влияют --guidance, --emotion и промпт."
        )

    # --- Шаг 1: маска. Локально, бесплатно, и смотреть на неё надо первой ---
    #
    # На фейссвопе пайплайн её не строит: область эндпоинт находит сам. Здесь
    # она всё равно считается, и не зря — это единственная местная проверка
    # того, что лицо на шаблоне вообще различимо. Своё «да» она говорит за
    # СВОЙ детектор: чужой ищет лицо по-своему и на рисованных лицах слепнет.
    started = time.monotonic()
    head = head_mask.build(
        decode_image(target),
        dilate_ratio=profile.mask.dilate_ratio,
        feather_ratio=profile.mask.feather_ratio,
        neck_ratio=profile.mask.neck_ratio,
    )
    print(f"\n=== Маска ({time.monotonic() - started:.1f} с) ===")
    if not profile.needs_mask:
        print(f"(стратегия «{profile.strategy}» её не использует — это диагностика)")
    print(json.dumps(head.meta, ensure_ascii=False, indent=2))

    from app.utils.image import encode_image

    mask_png = encode_image(head.mask, "png")[0]
    (out_dir / f"{stem}_mask.png").write_bytes(mask_png)
    (out_dir / f"{stem}_overlay.png").write_bytes(overlay(target, head.mask))
    print(f"маска:   {out_dir / f'{stem}_mask.png'}")
    print(f"наложение: {out_dir / f'{stem}_overlay.png'}")

    # --- Референс: тот же расчёт, что и в пайплайне, но с файлом на диске ---
    #
    # Смотреть на него надо до оплаты по той же причине, что и на маску: если
    # голова выйдет не того размера, отвечать будет эта картинка, а не промпт.
    from app.pipelines import reference as reference_prep
    from app.pipelines.face_swap.pipeline import _sniff_mime

    target_image = decode_image(target)
    target_height, target_width = target_image.shape[:2]
    reference = reference_prep.prepare(
        source,
        _sniff_mime(source),
        target_share=head.face_height / float(target_height),
        target_aspect=target_width / float(target_height),
        pad_ratio=profile.reference_pad_ratio,
        pad_max=profile.reference_pad_max,
    )
    reference_path = out_dir / f"{stem}_reference.jpg"
    reference_path.write_bytes(reference.data)
    print("\n=== Референс ===")
    print(json.dumps(reference.meta, ensure_ascii=False, indent=2))
    print(f"референс: {reference_path}")

    # --- Что именно уедет в fal ---
    payload = profile.payload.arguments(
        image_url="<CDN шаблона>",
        mask_url="<CDN маски>",
        identity_url="<CDN фотографии>",
        prompt=profile.prompt(expression.prompt(args.emotion)),
        strength=profile.strength,
        guidance_scale=profile.guidance_scale,
        steps=profile.steps,
        output_format="png",
        identity_scale=profile.identity_scale,
    )
    print(f"\n=== Запрос к {profile.endpoint} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not profile.payload.mask_field:
        # Ключ, которого нет в теле, легко принять за ошибку сборки запроса. Он
        # и правда отсутствует — но намеренно: эндпоинт маску не принимает
        local = (
            "Она работает локально — по ней голова из ответа возвращается в шаблон,\n"
            "  и всё, что вне маски, остаётся исходными пикселями разворота."
            if profile.needs_mask
            else "Она здесь не нужна вовсе: область эндпоинт находит своим детектором,\n"
            "  а всё вне неё оставляет своим — локальной вклейки на этом пути нет."
        )
        print(
            "\nМаска в запрос НЕ входит: эндпоинт её не принимает (лишний ключ — 422)."
            f"\n  {local}"
        )

    if args.dry_run:
        print("\n--dry-run: до fal не идём.")
        return 0

    if not os.environ.get("FAL_KEY", "").strip():
        print("\nFAL_KEY не задан — генерация пропущена. Задайте ключ или ставьте --dry-run.")
        return 1

    # --- Шаг 2: перенос лица. Отсюда начинаются деньги и минуты ---
    print(f"\n=== Вызов {profile.endpoint} ===")
    started = time.monotonic()
    result = pipeline.run(
        pipeline.SwapRequest(
            source=source,
            target=target,
            emotion=args.emotion,
            output_format="png",
        )
    )

    # Расширение — по типу, который вернул эндпоинт: фейссвоп формат не
    # принимает и выбирает его сам, а имя файла обязано этому соответствовать
    suffix = "jpg" if "jpeg" in result.mime_type else "png"
    result_path = out_dir / f"{stem}_result.{suffix}"
    result_path.write_bytes(result.image)
    print(f"готово за {time.monotonic() - started:.1f} с → {result_path}")
    print(json.dumps(result.meta, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
