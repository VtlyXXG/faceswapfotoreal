"""
Те же стендовые мерки, но по отдельному набору из `_probe` — состав и мотив там.

    python tools/probe_batch.py render  --seed 1234 --prefix S1234_
    python tools/probe_batch.py measure S1234_
    python tools/probe_batch.py strands S1234_
    python tools/probe_batch.py skin    S1234_
    python tools/probe_batch.py sheets  --out sheets_probe S1234_

Своего кода мерок здесь нет ни строчки: команда подменяет набор и запускает тот
же самый скрипт, что считает стенд. Аргументы после команды уходят ему как есть,
так что `--only`, `--seed`, `--url` и приставки работают ровно как в стенде.

Кадры пишутся и читаются в `storage/output/probe/`, отчёты `measure_batch.json`
и `strand_count.json` ложатся туда же. Стендовая папка не затрагивается, так что
прогоны двух наборов не могут смешаться даже при совпадающих приставках.

Прогон целиком, с туннелем до GPU-бокса:

    ssh -f -N -L 8300:127.0.0.1:8300 deploy@<хост>
    python tools/probe_batch.py render  --seed 1234 --prefix S1234_
    python tools/probe_batch.py measure S1234_
    python tools/probe_batch.py sheets  --out sheets_probe S1234_
"""
from __future__ import annotations

import importlib
import sys

import _probe

COMMANDS = {
    "render": "render_batch",
    "measure": "measure_batch",
    "strands": "strand_count",
    "skin": "skin_level",
    "sheets": "contact_sheets",
}


def _prefixes_given(argv) -> bool:
    """Есть ли среди аргументов приставка комплекта, а не только значения флагов."""
    given, previous = [], None
    for item in argv:
        if not item.startswith("-") and not (previous or "").startswith("--"):
            given.append(item)
        previous = item
    return bool(given)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip())
        return 2

    command = sys.argv.pop(1)
    module_name = COMMANDS[command]
    # Приставки по умолчанию у мерок стендовые (FINAL_, S1234_). В своей папке
    # таких кадров нет, и молчаливая таблица из прочерков читается как «всё
    # сломалось», хотя спросили не тот комплект
    if module_name != "render_batch" and not _prefixes_given(sys.argv[1:]):
        print("укажите приставку комплекта, например: "
              f"python tools/probe_batch.py {command} S1234_")
        return 2

    _probe.activate()
    return importlib.import_module(module_name).main()


if __name__ == "__main__":
    raise SystemExit(main())
