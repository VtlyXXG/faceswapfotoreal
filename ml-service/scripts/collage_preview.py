"""
Локальный предпросмотр первого шага: аппликация без вызова fal.ai.

Второй шаг пайплайна платный и небыстрый, а настраивается сейчас ровно первый:
насколько чисто вырезана голова. Гонять ради этого весь `/face-swap` бессмысленно —
скрипт останавливается на коллаже и раскладывает по файлам все промежуточные
стадии, чтобы было видно, какая именно из них портит результат:

    10_silhouette.png     сырая альфа rembg — весь человек, с одеждой и полутенью
    20_region.png         область головы: эллипс, срез по челюсти, колонна шеи
    30_alpha.png          итог вырезки: пересечение первых двух, порог и эрозия
    40_cutout.png         аппликация — голова с шеей на прозрачном фоне (BGRA)
    50_collage.png        коллаж: голова в шаблоне
    55_erased.png         что стёрто от головы персонажа
    56_erased_base.png    шаблон с затёртой головой, до наложения вклейки
    60_mask_seam.png      зона 2: стыки, мягкая сила
    61_mask_background.png зона 3: дыра в фоне, высокая сила
    62_zones.png          обе зоны поверх коллажа: зелёное — стык, красное — фон
    meta.json             метаданные обоих шагов вырезки

Смотреть в первую очередь на 40_cutout.png: там должны быть голова и шея до
линии одежды — и ни клочка самой одежды. Второе — 62_zones.png: зелёное должно
идти узкими полосами по контуру волос и по стыку шеи, красное — накрывать дыру
от чужой причёски и не доходить до новых волос, лицо не должно быть закрашено
вовсе.

    python scripts/collage_preview.py --source ../source.jpg --target ../target.png
    python scripts/collage_preview.py --neck-ratio 0.4   # отступ вместо поиска
    python scripts/collage_preview.py --erase-method ns --erase-pad-ratio 0.06

Модели rembg скачиваются при первом запуске (~176 МБ на модель).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.pipelines import collage as collage_builder  # noqa: E402
from app.pipelines import mask_generator, refine, segmentation  # noqa: E402


def _memoise_silhouette() -> None:
    """
    Один запуск сегментатора на кадр вместо трёх.

    Промежуточные стадии считаются здесь же, а потом `collage.build` проходит
    тот же путь заново. Для отладочного скрипта это лишние секунды на каждой
    итерации подбора долей — а результат по одному и тому же кадру побитово тот
    же самый.
    """
    original = segmentation.silhouette
    cache: dict[tuple, Any] = {}

    def cached(image: Any, model: str) -> Any:
        key = (model, image.shape, hash(image.tobytes()))
        if key not in cache:
            cache[key] = original(image, model)
        return cache[key]

    segmentation.silhouette = cached


def _save(out: Path, stages: dict[str, Any]) -> None:
    for name, image in stages.items():
        cv2.imwrite(str(out / name), image)
        print(f"  {name}")


def _zones(collage: Any, seam: Any, hole: Any) -> Any:
    """
    Три зоны одной картинкой: где что достанется модели.

    Зелёное — стыки (мягкая сила), красное — фон (высокая), нетронутое —
    защищённое лицо и остальной холст. Смотреть в первую очередь сюда: по двум
    отдельным маскам не видно, не наползают ли они друг на друга.
    """
    import numpy as np

    view = collage.astype(np.float32)
    for channel, zone in ((1, seam), (2, hole)):
        weight = (np.asarray(zone).astype(np.float32) / 255.0)[..., None]
        tint = np.zeros_like(view)
        tint[..., channel] = 255.0
        view = view * (1.0 - 0.55 * weight) + tint * (0.55 * weight)
    return np.clip(view, 0, 255).astype(np.uint8)


def _cutout(image: Any, alpha: Any) -> Any:
    """Аппликация как она есть: BGRA, всё вне головы — прозрачный ноль."""
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = alpha
    # Цвет под нулевой альфой тоже гасится: просмотрщик, игнорирующий альфу,
    # иначе покажет вырезанную голову вместе с одеждой и решит, что срез не работает
    bgra[..., :3][alpha == 0] = 0
    return bgra


def main() -> int:
    parser = argparse.ArgumentParser(description="Коллаж локально, без вызова fal.ai")
    parser.add_argument("--source", type=Path, default=_ROOT.parent / "source.jpg")
    parser.add_argument("--target", type=Path, default=_ROOT.parent / "target.png")
    parser.add_argument("--out", type=Path, default=_ROOT / "outputs" / "collage")
    parser.add_argument("--emotion", default="")
    # Две доли, которые сейчас и подбираются, — остальное берётся из настроек
    # Пусто — искать линию одежды донора по цвету
    parser.add_argument("--neck-ratio", type=float, default=settings.head_neck_ratio)
    parser.add_argument("--erode-ratio", type=float, default=settings.head_erode_ratio)
    # Масштаб: umeyama — по всем опорным точкам, остальные — по одной мерке
    parser.add_argument(
        "--scale-mark",
        choices=[*sorted(collage_builder._SCALE_MARKS), "median", "umeyama"],
        default=settings.head_scale_mark,
    )
    parser.add_argument(
        "--scale-multiplier", type=float, default=settings.head_scale_multiplier,
        help="художественный множитель размера головы: 0.85 для мультяшных серий",
    )
    parser.add_argument("--no-anchor", action="store_true", help="не опускать шею к воротнику")
    # Затирка головы персонажа на шаблоне
    parser.add_argument(
        "--erase-method",
        choices=collage_builder._ERASE_METHODS,
        default=settings.collage_erase_method,
    )
    parser.add_argument("--erase-neck-ratio", type=float, default=settings.collage_erase_neck_ratio)
    parser.add_argument("--erase-pad-ratio", type=float, default=settings.collage_erase_pad_ratio)
    # Профиль второго шага: маска рисуется по его числам, сам вызов не делается
    parser.add_argument(
        "--profile", choices=refine.profiles.available(), default=settings.refine_profile
    )
    args = parser.parse_args()

    for path in (args.source, args.target):
        if not path.exists():
            print(f"нет файла: {path}", file=sys.stderr)
            return 2

    source = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    target = cv2.imread(str(args.target), cv2.IMREAD_COLOR)
    if source is None or target is None:
        print("не удалось прочитать изображения", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    _memoise_silhouette()

    # Стадии вырезки по отдельности — ради них скрипт и написан
    points = mask_generator.face_landmarks(source)
    raw = np.asarray(segmentation.silhouette(source, settings.seg_model_photo))
    head = segmentation.cutout_head(
        source,
        points,
        settings.seg_model_photo,
        settings.head_width_ratio,
        settings.head_hair_ratio,
        args.neck_ratio,
        args.erode_ratio,
    )
    # Область строится повторно только ради картинки; отступ шеи берётся
    # найденный, иначе на 20_region.png будет не то, что реально вырезалось
    region, _, face_height = segmentation.head_region(
        points,
        source.shape[:2],
        settings.head_width_ratio,
        settings.head_hair_ratio,
        head.meta["neck_ratio"],
    )

    # Стадии вырезки пишутся до сборки коллажа: лицо ищется и на обложке тоже, и
    # если mediapipe его там не найдёт (рисованный персонаж мелко в кадре — обычное
    # дело), аппликация всё равно должна остаться на диске. Настраивается сейчас она.
    _save(
        args.out,
        {
            "10_silhouette.png": raw,
            "20_region.png": region,
            "30_alpha.png": head.alpha,
            "40_cutout.png": _cutout(source, head.alpha),
        },
    )

    collage = collage_builder.build(
        source,
        target,
        emotion=args.emotion,
        model_photo=settings.seg_model_photo,
        model_cover=settings.seg_model_cover,
        width_ratio=settings.head_width_ratio,
        hair_ratio=settings.head_hair_ratio,
        neck_ratio=args.neck_ratio,
        erode_ratio=args.erode_ratio,
        scale_mark=args.scale_mark,
        scale_multiplier=args.scale_multiplier,
        anchor_neck=not args.no_anchor,
        feather_ratio=settings.collage_feather_ratio,
        colour_match=settings.collage_colour_match,
        erase_template_head=settings.collage_erase_template_head,
        erase_method=args.erase_method,
        erase_neck_ratio=args.erase_neck_ratio,
        erase_pad_ratio=args.erase_pad_ratio,
    )

    # Маска строится по тому же профилю, что уехал бы в fal: её градиент и
    # strength подбираются вместе, и смотреть на маску от другого набора чисел
    # бессмысленно
    profile = refine.profiles.get(args.profile)
    face_height = collage.meta["face_height_paste"]
    mask = mask_generator.seam_mask(
        target.shape[:2],
        collage.head_alpha,
        collage.face_polygon,
        collage.neck_line,
        face_height,
        edge_ratio=profile.mask.edge_ratio,
        neck_ratio=profile.mask.neck_ratio,
        guard_ratio=profile.mask.guard_ratio,
        feather_ratio=profile.mask.feather_ratio,
        gradient_ratio=profile.mask.gradient_ratio,
        edge_outer_ratio=profile.mask.edge_outer_ratio,
    )
    paste = mask_generator.paste_mask(
        target.shape[:2], collage.head_alpha, collage.face_polygon, face_height,
        profile.mask.guard_ratio, profile.mask.paste_guard_strength,
        profile.mask.paste_inset_ratio,
    )
    hole = mask_generator.hole_mask(
        target.shape[:2],
        collage.head_alpha,
        collage.face_polygon,
        collage.erased,
        face_height,
        margin_ratio=profile.mask.hole_margin_ratio,
        feather_ratio=profile.mask.hole_feather_ratio,
        guard_ratio=profile.mask.guard_ratio,
    )

    # Шаблон без головы персонажа, до наложения вклейки. Именно по нему видно,
    # оставила ли затирка тёмное кольцо или пятна на месте ушей: в готовом
    # коллаже середина закрыта, и разглядеть там нечего
    target_points = mask_generator.face_landmarks(target)
    base, _, _ = collage_builder._erase_template_head(
        target,
        target_points,
        collage.head_alpha,
        collage.meta["face_height_target"],
        settings.seg_model_cover,
        args.erase_method,
        args.erase_neck_ratio,
        args.erase_pad_ratio,
    )

    _save(
        args.out,
        {
            "50_collage.png": collage.image,
            "55_erased.png": collage.erased,
            "56_erased_base.png": base,
            "60_mask_seam.png": mask,
            "61_mask_background.png": hole,
            "62_zones.png": _zones(collage.image, mask, hole),
            "63_mask_paste.png": paste,
        },
    )

    meta = {
        "source": str(args.source),
        "target": str(args.target),
        "neck_ratio": args.neck_ratio,
        "erode_ratio": args.erode_ratio,
        "refine": profile.report(),
        "face_height_photo": round(face_height, 1),
        "cutout": head.meta,
        "collage": collage.meta,
        # Сколько силуэта отрезала область головы. Близко к единице — сегментатор
        # нашёл почти одну голову, заметно меньше — в кадре было тело
        "kept_of_silhouette": round(
            int(np.count_nonzero(head.alpha > 127)) / max(1, int(np.count_nonzero(raw >= 160))), 3
        ),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nстадии сохранены в {args.out}")
    print("fal.ai не вызывался — это только первый шаг пайплайна")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
