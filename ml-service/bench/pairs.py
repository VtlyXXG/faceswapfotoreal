"""
Обнаружение пар (донор, обложка) во входном каталоге.

Два поддерживаемых расклада — оба под реальные сценарии сервиса книг:

  Режим A (папки):  input/<пара>/source.jpg + target*.png
      один ребёнок и одна обложка на папку; удобно для разнородных кейсов.

  Режим B (общий донор):  input/source.jpg + target*.png
      один ребёнок, много обложек — основной сценарий персонализации книги.

Файл-донор называется source.<ext>, обложки — target*.<ext>.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class Pair:
    pair_id: str
    source: Path
    target: Path


def _first(directory: Path, stem: str) -> Path | None:
    """Первый файл с точным именем stem.<ext>."""
    for ext in IMAGE_EXT:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _many(directory: Path, prefix: str) -> list[Path]:
    """Все изображения, чьё имя начинается с prefix, по алфавиту."""
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXT
        and path.stem.lower().startswith(prefix)
    )


def discover_pairs(root: str | Path) -> list[Pair]:
    root = Path(root)
    if not root.exists():
        return []

    pairs: list[Pair] = []

    # Режим A: подпапки с source + target*
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        source = _first(directory, "source")
        targets = _many(directory, "target")
        if source and targets:
            for target in targets:
                pairs.append(Pair(f"{directory.name}/{target.stem}", source, target))

    # Режим B: общий source в корне + target* в корне
    shared_source = _first(root, "source")
    shared_targets = _many(root, "target")
    if shared_source and shared_targets:
        for target in shared_targets:
            pairs.append(Pair(target.stem, shared_source, target))

    return pairs
