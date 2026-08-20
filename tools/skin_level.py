"""
Куда встала светлота лица: против ЛИЦА персонажа.

    python tools/skin_level.py FINAL_ S1234_

Зачем отдельная мерка, если есть ступень тона. Ступень мерит перепад НА СТЫКЕ.
Если поправка потемнила всю голову целиком и перелетела мимо цели, перепад на
стыке как раз уйдёт в ноль, и ступень отчитается об успехе. Промах всей головы
она не видит по построению.

ОПОРА — ЛИЦО ПЕРСОНАЖА, А НЕ ЕГО ШЕЯ, и это стоило одного неверного вывода.
Первая версия брала кожу вне головы (шея, грудь, руки): казалось, что «кожа
персонажа» это одно число. На dino1 и spread_08 лицо светлее шеи на 4.3 и 4.7
L*, разница тонет в шуме. Но на dino2 у персонажа опущена голова: лицо в тени,
шея на свету, и лицо ТЕМНЕЕ шеи на 11.8 L*. Мерка по шее объявила там перелёт
ровно на эти 11.8 — то есть показала перепад освещения самого шаблона, а не
промах поправки. По лицу все три шаблона ведут себя одинаково.

Ноль означает «лицо ребёнка встало в тон лица персонажа», плюс — светлее,
минус — темнее. Колонка «шея» оставлена для контекста: по ней видно, насколько
у шаблона вообще расходятся лицо и шея, и стоит ли доверять глазу на стыке.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from _bench import (DONORS, TEMPLATES, changed_mask, frame_path, lightness,
                    parse_prefixes, template_context)

from app.pipelines import parsing  # noqa: E402  (путь добавлен в _bench)

_reference: dict[str, tuple] = {}


def character_skin(key):
    """
    Светлота кожи персонажа: (лицо, шея).

    Лицо — кожа ПОД маской собственной головы, то есть лицо прежнего героя:
    та же поверхность, та же поза, тот же свет, что достанется ребёнку.
    Шея — кожа вне головы, для контекста.
    """
    if key not in _reference:
        template, parsed, _sil, _zone, _fh, own = template_context(key)
        skin = np.asarray(parsed.bare_skin) > 127
        head = own > 0.5
        L = lightness(template)
        face, neck = skin & head, skin & ~head
        _reference[key] = (
            float(np.median(L[face])) if face.sum() > 200 else None,
            float(np.median(L[neck])) if neck.sum() > 200 else None,
        )
    return _reference[key]


def main() -> int:
    prefixes = parse_prefixes(sys.argv, ("FINAL_", "S1234_"))
    agg = {p: [] for p in prefixes}

    print("отклонение светлоты лица ребёнка от ЛИЦА персонажа, L* (ноль — в тон)")
    print(f"{'кадр':<16}{'лицо перс.':>11}{'шея перс.':>10}"
          + "".join(f"{p.rstrip('_'):>10}" for p in prefixes))
    for key in TEMPLATES:
        for tag in DONORS:
            stem = f"{key}_{tag}"
            host_face, host_neck = character_skin(key)
            if host_face is None:
                print(f"{stem:<16} у персонажа не нашлось лица под маской")
                continue
            line = (f"{stem:<16}{host_face:>11.1f}"
                    + (f"{host_neck:>10.1f}" if host_neck is not None else f"{'—':>10}"))
            for prefix in prefixes:
                path = frame_path(prefix, stem)
                if not path.exists():
                    line += "—".rjust(10)
                    continue
                template, _parsed, _sil, _zone, _fh, _own = template_context(key)
                frame = cv2.imread(str(path))
                changed = changed_mask(frame, template)
                skin = (np.asarray(parsing.parse(frame).bare_skin) > 127) & changed
                if skin.sum() < 200:
                    line += "—".rjust(10)
                    continue
                deviation = float(np.median(lightness(frame)[skin])) - host_face
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
              f"темнее лица персонажа на {sum(1 for v in values if v < 0)} кадрах")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
