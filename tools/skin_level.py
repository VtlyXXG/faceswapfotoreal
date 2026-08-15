"""
Куда встала светлота лица: против кожи самого персонажа.

    python tools/skin_level.py FINAL_ S1234_

Зачем отдельная мерка, если есть ступень тона. Ступень мерит перепад НА СТЫКЕ.
Если поправка потемнила всю голову целиком и перелетела мимо цели, перепад на
стыке как раз уйдёт в ноль, и ступень отчитается об успехе. Промах всей головы
она не видит по построению.

Опора — кожа персонажа ВНЕ головы (шея, руки): именно к ней голову и подгоняют.
Ноль означает «лицо ребёнка встало в тон персонажа», плюс — лицо светлее сцены,
минус — темнее собственной шеи, чего быть не должно.

Так был найден перелёт на dino2: до правки отклонение там было всего +1.2…+4.3
L*, то есть подгонять было практически нечего, а после — до -11.8.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from _bench import (DONORS, TEMPLATES, changed_mask, frame_path, lightness,
                    parse_prefixes, template_context)

from app.pipelines import parsing  # noqa: E402  (путь добавлен в _bench)

_host: dict[str, float | None] = {}


def character_skin(key):
    """Медиана светлоты кожи персонажа вне головы — цель, к которой подгоняют."""
    if key not in _host:
        template, parsed, _sil, _zone, _fh, own = template_context(key)
        body = (np.asarray(parsed.bare_skin) > 127) & (own <= 0.5)
        L = lightness(template)
        _host[key] = float(np.median(L[body])) if body.sum() > 200 else None
    return _host[key]


def main() -> int:
    prefixes = parse_prefixes(sys.argv, ("FINAL_", "S1234_"))
    agg = {p: [] for p in prefixes}

    print("отклонение светлоты лица от кожи персонажа, L* (ноль — в тон)")
    print(f"{'кадр':<16}{'персонаж':>10}" + "".join(f"{p.rstrip('_'):>10}" for p in prefixes))
    for key in TEMPLATES:
        for tag in DONORS:
            stem = f"{key}_{tag}"
            host = character_skin(key)
            line = f"{stem:<16}{host:>10.1f}" if host is not None else f"{stem:<16}{'—':>10}"
            for prefix in prefixes:
                path = frame_path(prefix, stem)
                if not path.exists() or host is None:
                    line += "—".rjust(10)
                    continue
                template, _parsed, _sil, _zone, _fh, _own = template_context(key)
                frame = cv2.imread(str(path))
                changed = changed_mask(frame, template)
                skin = (np.asarray(parsing.parse(frame).bare_skin) > 127) & changed
                if skin.sum() < 200:
                    line += "—".rjust(10)
                    continue
                deviation = float(np.median(lightness(frame)[skin])) - host
                agg[prefix].append(deviation)
                line += f"{deviation:>+10.1f}"
            print(line, flush=True)

    print()
    for prefix, values in agg.items():
        if not values:
            continue
        print(f"{prefix.rstrip('_'):<8} среднее {np.mean(values):+.1f} L*, "
              f"по модулю {np.mean(np.abs(values)):.1f}, "
              f"худшее {max(values, key=abs):+.1f}, "
              f"темнее персонажа на {sum(1 for v in values if v < 0)} кадрах")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
