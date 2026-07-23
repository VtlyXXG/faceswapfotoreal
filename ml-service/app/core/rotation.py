"""
Хендлер ротации логов с политикой, совпадающей с Node-сервисом (pino-roll):
ротация по наступлению суток ИЛИ по превышению размера, хранение N файлов.

Стандартный TimedRotatingFileHandler умеет только время, RotatingFileHandler —
только размер, поэтому здесь первый расширен проверкой объёма.
"""

from __future__ import annotations

import os
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# app.2026-07-23.1.log — то же именование, что у pino-roll на стороне Node
_ROTATED = re.compile(r"^(?P<stem>.+)\.(?P<date>\d{4}-\d{2}-\d{2})\.(?P<index>\d+)\.log$")


class TimedSizedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(
        self,
        filename: str | os.PathLike,
        *,
        when: str = "midnight",
        interval: int = 1,
        backup_count: int = 14,
        max_bytes: int = 0,
        encoding: str = "utf-8",
        utc: bool = True,
    ) -> None:
        # backupCount=0: удаление старых файлов делаем сами — штатное опирается
        # на схему имён stdlib и не находит файлы, переименованные namer'ом
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=0,
            encoding=encoding,
            utc=utc,
        )
        self.max_bytes = max_bytes
        self.backup_limit = backup_count
        self.suffix = "%Y-%m-%d"
        self.namer = self._build_name

    # --- ротация -----------------------------------------------------------

    def shouldRollover(self, record) -> int:  # noqa: N802 — имя из stdlib
        if super().shouldRollover(record):
            return 1

        if self.max_bytes <= 0:
            return 0

        if self.stream is None:
            self.stream = self._open()

        message = f"{self.format(record)}{self.terminator}"
        size = self.stream.tell() + len(message.encode(self.encoding or "utf-8"))
        return 1 if size >= self.max_bytes else 0

    def doRollover(self) -> None:  # noqa: N802 — имя из stdlib
        super().doRollover()
        self._prune()

    # --- имена и уборка ----------------------------------------------------

    def _build_name(self, default_name: str) -> str:
        """
        `logs/app.log.2026-07-23` → `logs/app.2026-07-23.<N>.log`.

        Индекс нужен, потому что за одни сутки ротаций по размеру может быть
        несколько, а дата у них одна.
        """
        path = Path(default_name)
        date = path.suffix.lstrip(".")
        stem = Path(self.baseFilename).stem  # app.log → app

        index = 1
        while True:
            candidate = path.parent / f"{stem}.{date}.{index}.log"
            if not candidate.exists():
                return str(candidate)
            index += 1

    def _rotated_files(self) -> list[tuple[str, int, Path]]:
        """Ротированные файлы этого хендлера как (дата, индекс, путь)."""
        directory = Path(self.baseFilename).parent
        stem = Path(self.baseFilename).stem

        files = []
        for path in directory.iterdir():
            match = _ROTATED.match(path.name)
            if match and match.group("stem") == stem:
                files.append((match.group("date"), int(match.group("index")), path))
        return files

    def _prune(self) -> None:
        if self.backup_limit <= 0:
            return

        # Сортируем по дате и индексу из имени, а не по mtime: файлы одной
        # ротации по размеру создаются в пределах миллисекунды, и на Windows
        # их временные метки совпадают — порядок удаления был бы случайным
        files = [path for *_, path in sorted(self._rotated_files(), reverse=True)]
        for path in files[self.backup_limit :]:
            try:
                path.unlink()
            except OSError:  # файл уже удалён другим процессом
                pass
