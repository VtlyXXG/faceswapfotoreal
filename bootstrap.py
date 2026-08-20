"""
Провижининг весов antelopev2 для Фазы 2.

Экстрактор личности (`refine/adapter.py`) поднимает сессию onnxruntime по двум
файлам: `glintr100.onnx` — распознаватель, 512-d вектор, и `scrfd_10g_bnkps.onnx`
— детектор. Оба сняты с образа вместе с insightface и не возвращаются туда
пакетом: это сотни мегабайт, которые нужны одному шагу пайплайна.

    # Проверить окружение и что уже лежит на диске
    ml-service\\.venv\\Scripts\\python.exe bootstrap.py --check

    # Скачать по манифесту с проверкой хэшей
    ml-service\\.venv\\Scripts\\python.exe bootstrap.py --root C:\\models\\antelopev2

    # Первый раз, когда хэшей ещё нет: скачать и записать их в манифест
    ml-service\\.venv\\Scripts\\python.exe bootstrap.py --allow-unverified
    ml-service\\.venv\\Scripts\\python.exe bootstrap.py --record

**Хэши и ссылки живут в манифесте, а не в коде.** Зашитая константа тут была бы
хуже её отсутствия: неверный SHA256 удаляет каждый успешно скачанный файл, и
разбираться в этом придётся долго, потому что симптом выглядит как обрыв сети.
Манифест — JSON вида

    {"glintr100.onnx": {"url": "https://...", "sha256": "ab12..."}, ...}

и берётся он из `--manifest`, `ML_ANTELOPE_MANIFEST` либо `antelopev2.json`
рядом с корнем весов. Заполняется один раз: `--allow-unverified` скачивает и
печатает посчитанные хэши, `--record` кладёт их в манифест по уже лежащим на
диске файлам. Дальше проверка жёсткая и обязательная.

Официальный источник — пакет `antelopev2` из релизов InsightFace; отдельными
файлами он раздаётся зеркалами, и какое из них ваше, решает манифест.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Имена файлов заданы кодом: их ждёт `AntelopeExtractor`, и переименование
# сломало бы не загрузку, а инференс — то есть проявилось бы позже и дороже.
MODELS: tuple[str, ...] = ("glintr100.onnx", "scrfd_10g_bnkps.onnx")

MANIFEST_NAME = "antelopev2.json"

# Размер куска потоковой записи. Файлы — сотни мегабайт, и читать их в память
# целиком незачем: хэш считается на лету, тем же проходом, что и запись.
CHUNK = 1 << 20

# Незавершённая загрузка получает своё расширение и переименовывается в целевое
# имя только после сверки хэша. Иначе оборванная закачка оставила бы файл с
# правильным именем и неправильным содержимым, а следующий запуск счёл бы его
# готовым.
PART_SUFFIX = ".part"


class ProvisionError(RuntimeError):
    """Провижининг не выполнен. Сообщение уходит оператору как есть."""


@dataclass(frozen=True)
class Entry:
    """Строка манифеста: откуда брать файл и чем его проверять."""

    name: str
    url: str
    sha256: str

    def verified(self) -> bool:
        return bool(self.sha256)


def normalise(path: str | os.PathLike[str]) -> Path:
    """
    Строгая нормализация пути.

    `expanduser` — ради `~`, `resolve` — ради смешанных разделителей и `..`.
    На Windows это не косметика: путь, собранный из переменной окружения и
    аргумента, приходит с обоими слэшами сразу, и сравнение таких путей между
    собой даёт ложное «файла нет» при существующем файле.
    """
    return Path(path).expanduser().resolve()


def require_onnxruntime() -> str:
    """
    Проверяет, что onnxruntime в окружении есть. Иначе — выход с кодом 1.

    Проверка динамическая, а не через requirements: пакет снят с образа
    намеренно, и его отсутствие — штатное состояние Фазы 1, а не поломка. Ошибка
    поэтому обязана нести команду установки, а не трассировку ImportError.
    """
    try:
        import onnxruntime
    except ImportError:
        print(
            "onnxruntime не установлен — экстрактор личности не поднимется.\n"
            "  Установите его в то же окружение, где стоит сервис:\n"
            "      ml-service\\.venv\\Scripts\\python.exe -m pip install onnxruntime\n"
            "  Для инференса на видеокарте вместо него ставится onnxruntime-gpu.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return str(onnxruntime.__version__)


def manifest_path(root: Path, override: str) -> Path:
    if override:
        return normalise(override)
    if os.environ.get("ML_ANTELOPE_MANIFEST"):
        return normalise(os.environ["ML_ANTELOPE_MANIFEST"])
    return root / MANIFEST_NAME


def read_manifest(path: Path) -> dict[str, Entry]:
    """
    Читает манифест. Отсутствующий файл — не ошибка, а пустой манифест.

    Пустота осмысленна: с ней работает `--allow-unverified`, которым манифест и
    заполняется в первый раз.
    """
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvisionError(f"манифест не читается: {path} ({exc})") from exc

    entries: dict[str, Entry] = {}
    for name in MODELS:
        record = raw.get(name) or {}
        entries[name] = Entry(
            name=name,
            url=str(record.get("url") or os.environ.get(f"ML_ANTELOPE_URL_{name}", "")),
            sha256=str(record.get("sha256") or "").lower().strip(),
        )
    return entries


def write_manifest(path: Path, entries: dict[str, Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        entry.name: {"url": entry.url, "sha256": entry.sha256}
        for entry in entries.values()
        if entry.url or entry.sha256
    }
    path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def digest(path: Path) -> str:
    """SHA256 файла кусками: гигабайт в память ради хэша не читается."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


async def fetch(client, entry: Entry, root: Path, allow_unverified: bool) -> str:
    """
    Скачивает один файл потоком, считая хэш тем же проходом.

    Возвращает посчитанный SHA256 — он нужен и для сверки, и для `--record`.

    Файл, не прошедший проверку, удаляется НЕМЕДЛЕННО. Оставить его на диске
    значит подсунуть битые веса следующему запуску: имя правильное, размер
    правдоподобный, а сессия onnxruntime падает где-то в середине инференса.
    """
    target = root / entry.name
    part = root / (entry.name + PART_SUFFIX)

    if not entry.url:
        raise ProvisionError(
            f"{entry.name}: в манифесте нет ссылки. Впишите её в "
            f"{manifest_path(root, '')} или задайте ML_ANTELOPE_URL_{entry.name}"
        )
    if not entry.verified() and not allow_unverified:
        raise ProvisionError(
            f"{entry.name}: в манифесте нет sha256. Скачайте один раз с "
            "--allow-unverified, затем запишите хэш через --record"
        )

    sha = hashlib.sha256()
    written = 0
    part.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with client.stream("GET", entry.url, follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            with part.open("wb") as handle:
                async for chunk in response.aiter_bytes(CHUNK):
                    handle.write(chunk)
                    sha.update(chunk)
                    written += len(chunk)
                    if total:
                        print(
                            f"  {entry.name}: {written / total:6.1%} "
                            f"({written / 1024 / 1024:.0f} МБ)",
                            end="\r",
                            flush=True,
                        )
        print(" " * 60, end="\r")
    except Exception as exc:  # noqa: BLE001 — сеть, HTTP, диск: чистим одинаково
        part.unlink(missing_ok=True)
        raise ProvisionError(f"{entry.name}: загрузка не удалась ({exc})") from exc

    got = sha.hexdigest()
    if entry.verified() and got != entry.sha256:
        part.unlink(missing_ok=True)
        raise ProvisionError(
            f"{entry.name}: SHA256 не совпал, файл удалён.\n"
            f"    ожидали {entry.sha256}\n    получили {got}"
        )

    # Переименование атомарно на обеих системах и делается только теперь, когда
    # содержимое проверено
    part.replace(target)
    print(f"  {entry.name}: {written / 1024 / 1024:.0f} МБ, sha256 {got[:16]}…")
    return got


def state(root: Path, entries: dict[str, Entry]) -> dict[str, str]:
    """
    Что лежит на диске: ok, mismatch, unverified или missing.

    Разделение `unverified` и `ok` намеренное: файл на месте и файл проверенный
    — разные состояния, и второе достигается только заполненным манифестом.
    """
    report: dict[str, str] = {}
    for name in MODELS:
        path = root / name
        if not path.exists():
            report[name] = "missing"
            continue
        entry = entries.get(name)
        if entry is None or not entry.verified():
            report[name] = "unverified"
        else:
            report[name] = "ok" if digest(path) == entry.sha256 else "mismatch"
    return report


async def provision(
    root: Path, entries: dict[str, Entry], allow_unverified: bool
) -> dict[str, str]:
    """
    Догружает недостающее. Оба файла качаются параллельно: они независимы, а
    ожидание здесь сетевое и на процессор не давит.
    """
    import httpx

    current = state(root, entries)
    todo = [entries[name] for name, status in current.items() if status != "ok"]
    if not todo:
        print("Все веса на месте и проверены.")
        return {}

    for name, status in current.items():
        if status == "mismatch":
            # Битый файл сносится до загрузки: иначе при обрыве останется он же
            print(f"  {name}: хэш не сошёлся, файл удаляется")
            (root / name).unlink(missing_ok=True)

    timeout = httpx.Timeout(60.0, connect=15.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *(fetch(client, entry, root, allow_unverified) for entry in todo),
            return_exceptions=True,
        )

    computed: dict[str, str] = {}
    failures: list[str] = []
    for entry, result in zip(todo, results, strict=True):
        if isinstance(result, BaseException):
            failures.append(str(result))
        else:
            computed[entry.name] = result

    if failures:
        raise ProvisionError("\n".join(failures))
    return computed


def record(root: Path, entries: dict[str, Entry], path: Path) -> int:
    """Записывает в манифест хэши файлов, уже лежащих на диске."""
    updated = dict(entries)
    for name in MODELS:
        target = root / name
        if not target.exists():
            print(f"  {name}: на диске нет — пропущено", file=sys.stderr)
            continue
        previous = entries.get(name)
        updated[name] = Entry(
            name=name,
            url=previous.url if previous else "",
            sha256=digest(target),
        )
        print(f"  {name}: sha256 {updated[name].sha256}")

    write_manifest(path, updated)
    print(f"Манифест записан: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Провижининг весов antelopev2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("ML_ANTELOPE_ROOT", "models/antelopev2"),
        help="куда класть веса (ML_ANTELOPE_ROOT)",
    )
    parser.add_argument("--manifest", default="", help="JSON со ссылками и хэшами")
    parser.add_argument(
        "--check", action="store_true", help="только проверить, не качать"
    )
    parser.add_argument(
        "--record", action="store_true", help="записать хэши лежащих файлов"
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="разрешить загрузку без хэша в манифесте (первый раз)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = require_onnxruntime()

    root = normalise(args.root)
    path = manifest_path(root, args.manifest)
    print(f"onnxruntime {version}\nвеса: {root}\nманифест: {path}\n")

    try:
        entries = read_manifest(path)

        if args.record:
            return record(root, entries, path)

        current = state(root, entries)
        for name, status in current.items():
            print(f"  {name}: {status}")

        if args.check:
            return 0 if all(status == "ok" for status in current.values()) else 1

        print()
        computed = await_provision(root, entries, args.allow_unverified)

        if computed and args.allow_unverified:
            print(
                "\nЗагружено без проверки. Запишите хэши, чтобы дальше она была жёсткой:\n"
                f"    python bootstrap.py --root {root} --record"
            )
    except ProvisionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    return 0


def await_provision(
    root: Path, entries: dict[str, Entry], allow_unverified: bool
) -> dict[str, str]:
    """Синхронная обёртка: событийный цикл поднимается ровно на загрузку."""
    return asyncio.run(provision(root, entries, allow_unverified))


if __name__ == "__main__":
    raise SystemExit(main())
