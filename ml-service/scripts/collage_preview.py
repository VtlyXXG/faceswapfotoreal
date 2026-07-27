"""
Локальный предпросмотр первого шага: аппликация без вызова fal.ai.

Второй шаг пайплайна платный и небыстрый, а настраивается сейчас ровно первый:
насколько чисто вырезана голова. Гонять ради этого весь `/face-swap` бессмысленно —
скрипт останавливается на коллаже и раскладывает по файлам все промежуточные
стадии, чтобы было видно, какая именно из них портит результат:

    10_silhouette.png  сырая альфа rembg — весь человек, с одеждой и полутенью
    20_region.png      область головы: эллипс, срезанный снизу по линии челюсти
    30_alpha.png       итог вырезки: пересечение первых двух, порог и эрозия
    40_cutout.png      сама аппликация — голова на прозрачном фоне (BGRA)
    50_collage.png     коллаж: голова в шаблоне
    60_mask.png        маска стыка — что ушло бы в инпейнтинг (не отправляется)
    meta.json          метаданные обоих шагов вырезки

Смотреть в первую очередь на 40_cutout.png: там не должно быть ни шеи, ни
одежды, ни каймы фона фотографии вокруг волос.

    python scripts/collage_preview.py --source ../source.jpg --target ../target.png
    python scripts/collage_preview.py --erode-ratio 0.01 --neck-ratio 0.05

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
from app.pipelines import mask_generator, segmentation  # noqa: E402


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
    parser.add_argument("--neck-ratio", type=float, default=settings.head_neck_ratio)
    parser.add_argument("--erode-ratio", type=float, default=settings.head_erode_ratio)
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
    region, _, face_height = segmentation.head_region(
        points,
        source.shape[:2],
        settings.head_width_ratio,
        settings.head_hair_ratio,
        args.neck_ratio,
    )
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
        feather_ratio=settings.collage_feather_ratio,
        colour_match=settings.collage_colour_match,
        erase_template_head=settings.collage_erase_template_head,
    )

    mask = mask_generator.blend_mask(
        target.shape[:2],
        collage.head_alpha,
        collage.face_polygon,
        collage.neck_line,
        collage.erased,
        collage.meta["face_height_target"],
        edge_ratio=settings.mask_edge_ratio,
        neck_ratio=settings.mask_neck_ratio,
        guard_ratio=settings.mask_guard_ratio,
        feather_ratio=settings.mask_feather_ratio,
    )

    _save(args.out, {"50_collage.png": collage.image, "60_mask.png": mask})

    meta = {
        "source": str(args.source),
        "target": str(args.target),
        "neck_ratio": args.neck_ratio,
        "erode_ratio": args.erode_ratio,
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
