"""
Путь до `server.py`: он лежит рядом с каталогом тестов, а не в пакете.

Импорт сервера тяжёлых зависимостей не тянет — torch и diffusers грузятся
внутри функций, поэтому модуль читается и на машине без карты.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
