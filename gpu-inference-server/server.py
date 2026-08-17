"""
Собственный GPU-сервер генерации: SDXL + ControlNet inpaint, LaMa, CodeFormer.

Ставится на инстанс с картой (A40) и снимает с fal.ai диффузионные пути
пайплайна. Контракт запроса намеренно совпадает с тем, что уже шлёт
`ml-service/app/pipelines/refine/adapter.py::HostedFluxPuLIDBackend`, — тот
собирает пакет в `refine/payload.py::build_packet` и постит его на `/render`,
а картинку читает из `{"image": <base64>}`. Поэтому здесь два адреса на один
обработчик: `/v1/template-render` (основной) и `/render` (алиас под уже
написанный клиент). Переезд на этот сервер стоит одной переменной с базовым
URL, а не правки пайплайна.

    ML-СЕРВИС                        ЭТОТ СЕРВЕР
    build_packet()  --- JSON --->    /v1/template-render
                                       LaMa: стереть старое содержимое маски
                                       SDXL+ControlNet: нарисовать новое
                                       CodeFormer: поднять лицо
                    <--- JSON ---    {"image": base64, "meta": {...}}


ПЕРЕНОС ЛИЧНОСТИ

Личность входит в генерацию через IP-Adapter Plus Face: `donor_crop` из пакета
идёт в пайплайн как `ip_adapter_image`, адаптер подмешивает его в кросс-внимание
UNet рядом с текстовым условием. Сила подмеса — `GPU_IP_ADAPTER_SCALE`.

Взят вариант на CLIP vision (`ip-adapter-plus-face_sdxl_vit-h`), а НЕ FaceID.
FaceID даёт заметно точнее сходство, но берёт identity из ArcFace-эмбеддинга
InsightFace, чьи официальные веса изданы под некоммерческой лицензией. Для
платного продукта это блокирует использование целиком, поэтому сходство здесь
осознанно обменяно на чистую лицензию (сам IP-Adapter — Apache 2.0).

512-мерный `embedding` из пакета по-прежнему не используется: он собран под
Flux + PuLID и адаптеру не подходит — тому нужна картинка, а не вектор. Поле
принимается и проверяется на формат, чтобы расхождение всплыло здесь, а не
тихо. Что именно доехало до генерации, видно в `meta.identity_applied`.


ГРАДАЦИЯ ОТКАЗА

Ни один из трёх компонентов не обязателен, и отсутствие весов не роняет сервис,
а понижает качество — как отсутствие весов разметки в ml-service понижает маску
до эллипса. Что именно живо, видно на `/health/ready` и в `meta.stages` каждого
ответа:

    SDXL нет        -> 503 на генерацию, сервис поднимается (для диагностики)
    ControlNet нет  -> обычный SDXL inpaint без структурного контроля
    IP-Adapter нет  -> лицо рисуется по тексту, без сходства с донором
    LaMa нет        -> стирание через cv2.inpaint (TELEA), заметно грубее
    CodeFormer нет  -> лицо не восстанавливается, кадр отдаётся как есть

Запуск:

    python download_weights.py --root ./weights
    uvicorn server:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("gpu-inference")


# --- 0. Конфигурация ---------------------------------------------------------
#
# Всё через окружение и без файла настроек: сервер живёт одним процессом на
# одной карте, и вся его конфигурация обязана читаться из `docker run -e` без
# пересборки образа.


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


WEIGHTS_ROOT = Path(_env("GPU_WEIGHTS_ROOT", "./weights")).expanduser().resolve()

# Каталоги внутри WEIGHTS_ROOT. Имена заданы кодом и совпадают с ключами
# манифеста в download_weights.py: переименование ломает обе стороны сразу.
DIR_SDXL = WEIGHTS_ROOT / "sdxl-inpaint"
DIR_CONTROLNET = WEIGHTS_ROOT / "controlnet-inpaint-sdxl"
DIR_VAE = WEIGHTS_ROOT / "sdxl-vae-fp16-fix"
DIR_LAMA = WEIGHTS_ROOT / "lama"
DIR_CODEFORMER = WEIGHTS_ROOT / "codeformer"
DIR_FACEXLIB = WEIGHTS_ROOT / "facexlib"
DIR_IPADAPTER = WEIGHTS_ROOT / "ip-adapter"

# Внутренняя раскладка репозитория h94/IP-Adapter — она же аргументы
# load_ip_adapter. Энкодер именно из models/: см. комментарий у записи
# `ip-adapter` в download_weights.py, там же почему не sdxl_models/image_encoder
IPADAPTER_SUBFOLDER = "sdxl_models"
IPADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IPADAPTER_ENCODER = "models/image_encoder"

FILE_LAMA = DIR_LAMA / "big-lama.pt"
FILE_CODEFORMER = DIR_CODEFORMER / "codeformer.pth"
FILE_CODEFORMER_JIT = DIR_CODEFORMER / "codeformer.jit.pt"

DEVICE = _env("GPU_DEVICE", "cuda")
# fp16 — единственный разумный режим для SDXL на A40: bf16 у SDXL даёт заметный
# сдвиг цвета на тёмных участках, fp32 не окупает вдвое большую память
DTYPE_NAME = _env("GPU_DTYPE", "float16")

# Рабочее разрешение диффузии. SDXL обучен на 1024 по длинной стороне, и
# отправлять ему 4K бессмысленно вдвойне: и память, и качество — модель начинает
# дублировать композицию. Разворот приходит сюда УЖЕ окном вокруг головы
# (ml-service режет его по ML_HAIR_CROP_RATIO), поэтому 1024 хватает; обратно
# кадр возвращается в исходный размер.
MAX_SIDE = _env_int("GPU_MAX_SIDE", 1024)
MIN_SIDE = _env_int("GPU_MIN_SIDE", 512)

DEFAULT_STEPS = _env_int("GPU_STEPS", 30)
DEFAULT_GUIDANCE = _env_float("GPU_GUIDANCE_SCALE", 7.0)
DEFAULT_STRENGTH = _env_float("GPU_STRENGTH", 0.92)
DEFAULT_CONTROL_SCALE = _env_float("GPU_CONTROL_SCALE", 0.55)
DEFAULT_NEGATIVE = _env(
    "GPU_NEGATIVE_PROMPT",
    "blurry, lowres, deformed face, extra limbs, watermark, text, jpeg artifacts, "
    "visible seam, pasted cut-out, collage edge",
)
# Запасной prompt на случай, когда клиент прислал пустую строку. Пустой prompt —
# не ошибка (диффузия его принимает), но это отказ от текстового руководства
# целиком: модель дорисовывает маску «чем-нибудь», и на большой маске головы это
# читается как случайный человек. Текст намеренно описывает КЛАСС объекта и
# съёмку, а не внешность: за внешность отвечает IP-Adapter, и подробное описание
# лица в prompt начинает с ним спорить.
DEFAULT_PROMPT = _env(
    "GPU_PROMPT",
    "a photo of a person, natural skin texture, sharp focus, consistent studio "
    "lighting, photorealistic, high detail",
)

# IP-Adapter. Вес — компромисс: identity тем точнее, чем он выше, но адаптер
# давит и на геометрию, и выше ~0.8 вклейка начинает игнорировать позу и ракурс
# из ControlNet, разворачивая голову к анфасу донора. 0.65 — отправная точка, а
# не измеренный оптимум: подбирать под свой набор кадров через GPU_IP_ADAPTER_SCALE.
IPADAPTER_ENABLED = _env_bool("GPU_IP_ADAPTER", True)
IPADAPTER_SCALE = _env_float("GPU_IP_ADAPTER_SCALE", 0.65)

# Восстановление лица. w — «верность»: 0 тянет к качеству картинки, 1 — к
# исходным пикселям. 0.7 держит черты и убирает мыло, ниже 0.5 CodeFormer
# начинает рисовать своё лицо, а не чинить присланное.
CODEFORMER_W = _env_float("GPU_CODEFORMER_W", 0.7)

# Стирание перед генерацией. По умолчанию включено: диффузия внутри маски
# цепляется за структуру старых волос и принимает тёмную прядь на щеке за тень
# скулы. Тот же эффект ml-service пытался получить заливкой (ML_HAIR_ERASE_RATIO)
# и провалился — LaMa рисует правдоподобный фон, а не мыло.
ERASE_ENABLED = _env_bool("GPU_ERASE", True)
RESTORE_ENABLED = _env_bool("GPU_RESTORE_FACE", True)

# --- Демонстрационный путь: FLUX.2 + пересадка головы ------------------------
#
# Отдельный путь, а не замена основному. Он принимает СЫРОЕ фото ребёнка и сам
# делает всё, что раньше делал ml-service до запроса: маска головы, кроп лица,
# стирание, пересадка. Нужен для показа «загрузили фото — получили результат»,
# без промежуточных ручных шагов.
DEMO_ENABLED = _env_bool("GPU_DEMO", True)
FLUX_MODEL = _env("GPU_FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")

# Где лежит пакет `app` из ml-service: маска головы, разметка, пересадка. Код
# намеренно не дублируется — он уже написан, покрыт тестами и измерен, а копия
# разошлась бы с оригиналом на первой же правке.
MLSERVICE_PATH = _env("GPU_MLSERVICE_PATH", "")

# Параметры генерации демо-пути. Подобраны замером и менять их без нового замера
# не следует:
#   8 шагов  — +0.05 к сходству против 4; на 16 уже хуже, чем на 8, при
#              трёхкратном времени
#   guidance 1.0 — режим дистиллированной модели, выше она ломается
FLUX_STEPS = _env_int("GPU_FLUX_STEPS", 8)
FLUX_GUIDANCE = _env_float("GPU_FLUX_GUIDANCE", 1.0)

# Шаги для кадров с МЕЛКИМ лицом, см. `demo_steps`. Замер на 21 кадре показал,
# что восьми шагов мало там, где голова занимает малую часть кадра: вклеенная
# голова выходит вдвое мягче той живописи, в которую садится.
FLUX_STEPS_SMALL_FACE = _env_int("GPU_FLUX_STEPS_SMALL_FACE", 16)

# Ниже какой высоты лица В ГЕНЕРАЦИИ кадр считается мелколицым, пиксели.
# Не доля кадра: у dino1 и spread_08 доля почти одна (17.5% против 17.6%), а
# ведут себя они противоположно — решает абсолютный размер.
FLUX_SMALL_FACE_PX = _env_float("GPU_FLUX_SMALL_FACE_PX", 200.0)

# Промпт-запрет вместо промпта-описания. Проверено на четырёх формулировках:
# описание ракурса словами («голова отвёрнута и наклонена вниз») заставляло
# модель довернуть голову ещё дальше, и лицо переставало находиться вовсе. Этот
# вариант дал лучшее удержание позы и лучшее сходство одновременно.
FLUX_PROMPT = _env(
    "GPU_FLUX_PROMPT",
    "Replace only the facial features of the child in the first image with the "
    "facial features of the child in the second image. Do not change the head "
    "orientation, the gaze direction, the hair position, the body pose or "
    "anything else in the first image.",
)

# Рабочее разрешение демо-пути. У модели потолок 4 мегапикселя, берём с запасом:
# на 8 шагах активации толще, и на 2048x2048 рядом с занятой картой уже ловился
# OutOfMemory.
FLUX_MAX_PIXELS = _env_int("GPU_FLUX_MAX_PIXELS", 3_400_000)

# Доли маски головы — те же, что у боевого профиля ml-service (`MaskProfile`).
DEMO_DILATE_RATIO = _env_float("GPU_DEMO_DILATE", 0.12)
DEMO_FEATHER_RATIO = _env_float("GPU_DEMO_FEATHER", 0.10)
DEMO_NECK_RATIO = _env_float("GPU_DEMO_NECK", 0.35)

# Кроп лица донора: сторона холста и во сколько раз шире поле зрения. 512 и 1.2
# против «боевых» 112 и 1.0 — потому что потребитель другой. Тем 112 нужен был
# вход распознавателя, где причёска мешает; генеративному редактору нужно лицо
# целиком, вместе с овалом и волосами.
DEMO_DONOR_SIDE = _env_int("GPU_DEMO_DONOR_SIDE", 512)
DEMO_DONOR_MARGIN = _env_float("GPU_DEMO_DONOR_MARGIN", 1.2)

# Согласование тона генерации с шаблоном перед вклейкой, см. `match_levels`.
# Ручка нужна затем, что поправка держится на допущении о сцене, а не на
# арифметике: если она однажды сработает не туда, выключить её надо переменной
# окружения, а не выкаткой образа.
MATCH_LEVELS = _env_bool("GPU_MATCH_LEVELS", True)

# Пределы поправки тона. Числа перенесены из ml-service вместе с логикой
# (`app/pipelines/composite.py`), потому что смысл у них тот же: поправка обязана
# чинить общий увод экспозиции и не обязана уметь ничего сверх того. Вышедшая за
# пределы прямая поправку не ограничивает, а ОТМЕНЯЕТ — прижатое к границе число
# заведомо неверно, и применить его значит добавить свой сдвиг к чужому.
MATCH_GAIN_MIN, MATCH_GAIN_MAX = 0.8, 1.25
MATCH_BIAS_MAX = 24.0

# Ниже этой дисперсии (уровни²) наклон не по чему считать: вне маски одно ровное
# небо. Тогда остаётся сдвиг — для плоской области он и есть вся поправка.
MATCH_MIN_VARIANCE = 4.0

# Поправка считается на уменьшенной копии: нужна оценка по десяткам тысяч
# пикселей, а не точность до уровня, а полный проход по трём каналам на 4K —
# сотни мегабайт трафика памяти впустую.
MATCH_PREVIEW_PX = 512

# Насколько пиксель вне маски должен разойтись, чтобы считаться следом
# восстановления, а не округлением uint8. См. `confine`.
_RESTORE_SPILL_LEVELS = 2

# Очередь. Слот один: карта одна, и две диффузии на ней не ускоряются, а делят
# память и время пополам, зато вдвое ухудшают худшую латентность. Растить это
# число имеет смысл только на нескольких картах.
QUEUE_SLOTS = _env_int("GPU_QUEUE_SLOTS", 1)
# Сколько заказов имеют право ЖДАТЬ слот. Дальше — 503 сразу, а не через десять
# минут в очереди: воркеров у ml-service три, они повторяют по таймауту, и без
# верхней границы очередь растёт быстрее, чем разбирается.
QUEUE_MAX_WAITING = _env_int("GPU_QUEUE_MAX_WAITING", 8)
QUEUE_WAIT_TIMEOUT = _env_float("GPU_QUEUE_WAIT_TIMEOUT", 240.0)

# Освобождать кэш аллокатора после каждого кадра. По умолчанию выключено: вызов
# синхронизирует устройство и стоит десятки миллисекунд, а фрагментация на 48 ГБ
# A40 при одном слоте не набирается. Включать при OOM на длинных сменах.
EMPTY_CACHE = _env_bool("GPU_EMPTY_CACHE", False)

# Оффлайн по умолчанию: сервер не имеет права ходить в HuggingFace в рантайме.
# Первый же заказ на свежем контейнере иначе упрётся в докачку 7 ГБ, а при
# масштабировании — в каждом контейнере заново.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(WEIGHTS_ROOT / ".hf"))

_EMBEDDING_DTYPE = "<f4"
_EMBEDDING_SIZE = 512


# --- 1. Очередь --------------------------------------------------------------


class QueueFull(Exception):
    """Очередь длиннее допустимого — заказ отбивается немедленно."""


class QueueTimeout(Exception):
    """Слот не освободился за отведённое время."""


class GpuQueue:
    """
    Семафор на карту плюс контроль допуска.

    Одного семафора мало. Он честно сериализует работу, но ждать на нём может
    сколько угодно заказов, и при перегрузке сервер отвечает не отказом, а
    молчанием на десять минут — худшее из возможных поведений для воркера с
    таймаутом и повторами: тот отвалится, повторит, и в очереди окажется два
    одинаковых кадра вместо одного. Поэтому очередь ограничена и по ДЛИНЕ
    (`QueueFull` -> 503), и по ВРЕМЕНИ ожидания (`QueueTimeout` -> 503).
    """

    def __init__(self, slots: int, max_waiting: int, wait_timeout: float) -> None:
        self._sem = asyncio.Semaphore(slots)
        self._max_waiting = max_waiting
        self._wait_timeout = wait_timeout
        self._waiting = 0
        self._running = 0
        self._served = 0
        self._rejected = 0
        self.slots = slots

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "running": self._running,
            "waiting": self._waiting,
            "max_waiting": self._max_waiting,
            "served": self._served,
            "rejected": self._rejected,
        }

    @asynccontextmanager
    async def slot(self):
        """Занимает слот. Отдаёт время ожидания — оно уезжает в meta ответа."""
        if self._waiting >= self._max_waiting:
            self._rejected += 1
            raise QueueFull(f"в очереди уже {self._waiting} заказов")

        self._waiting += 1
        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self._wait_timeout)
        except TimeoutError as exc:
            self._rejected += 1
            raise QueueTimeout(f"слот не освободился за {self._wait_timeout:.0f} с") from exc
        finally:
            self._waiting -= 1

        waited = time.perf_counter() - started
        self._running += 1
        try:
            yield waited
        finally:
            self._running -= 1
            self._served += 1
            self._sem.release()


# --- 2. Утилиты кадра --------------------------------------------------------


def decode_image(data: str, field_name: str, *, grayscale: bool = False) -> np.ndarray:
    """
    base64 -> BGR (или одноканальная маска).

    Формат распознаётся по содержимому, а не по заявленному в пакете MIME:
    `encoding` в пакете нужен для диагностики нашей же ошибки («маска уехала
    JPEG-ом»), но доверять ему при разборе — значит уронить кадр на опечатке.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:  # noqa: BLE001 — любой мусор в base64
        raise HTTPException(422, f"{field_name}: не base64 ({exc})") from exc

    if not raw:
        raise HTTPException(422, f"{field_name}: пустое поле")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flag)
    if image is None:
        raise HTTPException(422, f"{field_name}: не удалось разобрать как изображение")
    return image


def encode_image(image: np.ndarray, fmt: str = "png", quality: int = 95) -> str:
    """Кадр -> base64. PNG по умолчанию: результат уезжает на вклейку."""
    params: list[int] = []
    if fmt in {"jpg", "jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buffer = cv2.imencode(f".{fmt}", image, params)
    if not ok:
        raise HTTPException(500, f"не удалось закодировать результат в {fmt}")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def snap(value: int) -> int:
    """Ближайшее кратное восьми: и VAE, и ControlNet работают по сетке 8."""
    return max(8, int(round(value / 8)) * 8)


_MLSERVICE: dict[str, Any] = {}


def mlservice() -> dict[str, Any]:
    """
    Модули ml-service: маска головы, разметка, пересадка, выравнивание донора.

    Импорт отложенный и кэшированный. Отложенный — потому что путь к пакету
    приходит из окружения и на боевом образе его может не быть вовсе; кэшированный
    — потому что `parsing` держит граф tflite глобальным объектом, и второй
    импорт стоил бы второй загрузки весов.

    :raises HTTPException: 503, если пакет не найден или не импортируется. Текст
        уходит клиенту как есть: это не ошибка запроса, а незавершённая установка
    """
    if _MLSERVICE:
        return _MLSERVICE

    if MLSERVICE_PATH and MLSERVICE_PATH not in sys.path:
        sys.path.insert(0, MLSERVICE_PATH)

    # ml-service требует Python 3.11 (`requires-python = ">=3.11,<3.12"`), и
    # `app.core.logging` берёт оттуда `datetime.UTC`. На демонстрационном хосте
    # стоит 3.10, и поднимать там вторую сборку интерпретатора ради одной
    # константы дороже, чем объявить её здесь.
    #
    # Это ЗАПЛАТКА, а не решение: правильный ход — python 3.11 на хосте. Правка
    # живёт здесь, а не в копии ml-service, намеренно — копия обязана оставаться
    # побайтово равной репозиторию, иначе следующий, кто её обновит, молча
    # затрёт исправление и будет искать причину в другом месте.
    import datetime as _datetime

    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc

    try:
        from app.pipelines import head_mask, transplant
        from app.pipelines.refine.adapter import AntelopeExtractor
    except ImportError as exc:
        raise HTTPException(
            503,
            f"пакеты ml-service недоступны: {exc}. Укажите GPU_MLSERVICE_PATH "
            f"на каталог с пакетом app (сейчас: {MLSERVICE_PATH or 'не задан'})",
        ) from exc

    _MLSERVICE.update(head_mask=head_mask, transplant=transplant, extractor=AntelopeExtractor)
    return _MLSERVICE


def _mlservice_reason() -> str:
    """Почему демо-путь не готов, если дело в пакетах. Пусто — всё на месте."""
    try:
        mlservice()
    except HTTPException as exc:
        return str(exc.detail)
    return ""


def flux_size(height: int, width: int) -> tuple[int, int]:
    """
    Размер кадра для FLUX.2: кратный шестнадцати и не крупнее потолка.

    Кратность обязательна — модель делит кадр на патчи, — а потолок стоит из-за
    памяти: на 8 шагах активации толще, и полный разворот 2048x2048 рядом с
    занятой картой уже падал по OutOfMemory.
    """
    scale = min(1.0, (FLUX_MAX_PIXELS / float(height * width)) ** 0.5)
    return (max(16, int(height * scale) // 16 * 16), max(16, int(width * scale) // 16 * 16))


def demo_steps(
    face_height: float, height: int, width: int, requested: int | None
) -> tuple[int, dict[str, Any]]:
    """
    Сколько шагов диффузии просить у FLUX, если вызывающий не назвал число сам.

    ЗАЧЕМ. Замер на 21 кадре (seed 1234, попарно с восемью шагами) показал, что
    восьми мало не всем и не везде. Резкость меряется как «голова против своего
    окружения, в долях от того же у шаблона»: единица значит, что перепад
    сохранён, а у художника голова — самое резкое место кадра.

        шаблон      лицо в генерации   8 шагов        16 шагов
        dino2        78 px            0.600 / 0.431   0.691 / 0.453
        dino1       134 px            0.483 / 0.623   0.574 / 0.635
        spread_08   324 px            0.889 / 0.753   1.415 / 0.700
                                      резкость / сходство

    На двух мелколицых шестнадцать шагов дают прибыль ПО ОБЕИМ осям сразу:
    резкость +0.09, и сходство при этом не падает, а слегка растёт. Это не
    размен, платить нечем.

    А развороту они противопоказаны. Резкость там улетает за единицу — 1.415,
    то есть голова становится резче окружения, чем была у самого художника, —
    и вместе с этим монотонно валится сходство: 0.753 на восьми, 0.700 на
    шестнадцати, 0.671 на двадцати восьми. Перешарп ломает черты.

    ПОЧЕМУ ПОРОГ ПО АБСОЛЮТНОМУ РАЗМЕРУ, А НЕ ПО ДОЛЕ КАДРА. Доля лица у dino1 и
    spread_08 почти совпадает (17.5% против 17.6%), а ведут они себя
    противоположно. Абсолютный размер их разделяет: 134 px против 324 px.
    Двести — середина этого промежутка, а не измеренная точка перелома;
    ступенчатого замера между 134 и 324 не делали.

    ПОЧЕМУ МЕРЯЕТСЯ ЛИЦО В ГЕНЕРАЦИИ, А НЕ В ШАБЛОНЕ. Кадр крупнее потолка
    FLUX уезжает в генерацию уменьшенным, и лицо уменьшается вместе с ним:
    у spread_08 360 px в шаблоне превращаются в 324 px. Считать надо там, где
    работает модель.

    Двадцать восемь шагов проверены и отвергнуты: сходство ниже, чем на
    шестнадцати, перешарп сильнее, а на `dino2_id4` генерация вышла такой, что
    голову на ней не нашли вовсе — кадр не получился.

    :param face_height: высота лица на ШАБЛОНЕ, пиксели
    :param height: высота шаблона
    :param width: ширина шаблона
    :param requested: число шагов из запроса; None — решать здесь
    :return: число шагов и что об этом записать в мету
    """
    if requested is not None:
        return requested, {"steps_reason": "задано в запросе"}

    scaled_height, _ = flux_size(height, width)
    face_px = face_height * scaled_height / max(1, height)
    small = face_px < FLUX_SMALL_FACE_PX
    return (FLUX_STEPS_SMALL_FACE if small else FLUX_STEPS), {
        "steps_reason": "мелкое лицо" if small else "лицо крупное",
        "face_px_generated": round(float(face_px), 1),
    }


def working_size(height: int, width: int) -> tuple[int, int]:
    """
    Размер, в котором считает диффузия.

    Длинная сторона приводится к MAX_SIDE, пропорция сохраняется, обе стороны
    снапаются к восьмёрке. Апскейла нет намеренно: растянуть окно 400x300 до
    1024 — значит попросить модель дорисовать детали, которых в исходнике не
    было, и получить мыло на возврате в исходный размер.
    """
    longest = max(height, width)
    scale = min(1.0, MAX_SIDE / longest)
    if min(height, width) * scale < MIN_SIDE:
        scale = min(1.0, MIN_SIDE / min(height, width))
    return snap(int(height * scale)), snap(int(width * scale))


def normalise_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Маска к размеру кадра, одноканальная, 0..255.

    Интерполяция ближайшего соседа, если маска УМЕНЬШАЕТСЯ, и линейная, если
    растёт: билинейное уменьшение размывает край и делает ненулевыми пиксели,
    которые в присланной маске были чистым нулём, — ровно та беда, ради которой
    ml-service гоняет маску PNG-ом, а не JPEG-ом. Терять это на ресайзе глупо.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != shape:
        shrinking = shape[0] * shape[1] < mask.shape[0] * mask.shape[1]
        interp = cv2.INTER_NEAREST if shrinking else cv2.INTER_LINEAR
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=interp)

    return mask


def mask_bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    """Прямоугольник ненулевой части маски. None — маска пустая."""
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    top = max(0, int(ys.min()) - pad)
    left = max(0, int(xs.min()) - pad)
    bottom = min(mask.shape[0], int(ys.max()) + 1 + pad)
    right = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return top, left, bottom, right


# --- 3. Компоненты -----------------------------------------------------------


@dataclass
class LamaEraser:
    """
    Стирание содержимого маски до генерации.

    Зачем отдельным шагом, а не доверить диффузии. Inpaint-пайплайн переписывает
    присланные пиксели, а не рисует с нуля, и структура старой головы протекает
    в результат: контур причёски проступает как тень, тёмная прядь на щеке
    читается моделью как складка. ml-service уже пробовал решать это заливкой
    (ML_HAIR_ERASE_RATIO) и получил мыло внутри маски и светлый ореол по её
    границе. LaMa вместо заливки продолжает фон и одежду — ей это и делали.

    Веса — торчскриптовый `big-lama.pt`: он не тянет ни omegaconf, ни saicinpainting,
    и потому переживает обновление окружения. Без файла работает cv2.inpaint —
    хуже, но лучше, чем ничего.
    """

    path: Path
    device: str
    _model: Any = field(default=None, init=False, repr=False)
    _kind: str = field(default="none", init=False)

    @property
    def kind(self) -> str:
        return self._kind

    def load(self) -> None:
        import torch

        if self._model is not None or self._kind == "opencv":
            return

        if not self.path.exists():
            log.warning(
                "весов LaMa нет — стирание падает на cv2.inpaint",
                extra={"expected": str(self.path)},
            )
            self._kind = "opencv"
            return

        self._model = torch.jit.load(str(self.path), map_location=self.device).eval()
        self._kind = "lama"

    def erase(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._kind == "lama":
            return self._erase_lama(image, mask)
        # TELEA размазывает соседние пиксели внутрь дыры: на мелкой маске
        # приемлемо, на голове целиком даёт характерное пятно
        return cv2.inpaint(image, (mask > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)

    def _erase_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        height, width = image.shape[:2]
        # Сеть свёрточная и требует кратности восьми. Паддинг краевым пикселем,
        # а не нулём: чёрная кайма по краю втягивается в результат как настоящее
        # содержимое сцены
        pad_h, pad_w = (-height) % 8, (-width) % 8
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if pad_h or pad_w:
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        with torch.inference_mode():
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            binary = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
            result = self._model(tensor.to(self.device), binary.to(self.device))

        out = result[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)[:height, :width]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


@dataclass
class FaceRestorer:
    """
    CodeFormer поверх готового кадра.

    ПОЧЕМУ ЗАГРУЗКА ПЛУГГАБЕЛЬНАЯ. Официальные веса `codeformer.pth` — это
    state_dict, и одного файла мало: нужен код архитектуры (VQ-GAN плюс
    трансформер), а он живёт в репозитории sczhou/CodeFormer и на PyPI не
    издан. Вшивать сюда четыреста строк чужой модели — значит взять на себя её
    сопровождение. Поэтому три пути, в порядке предпочтения:

      1. `codeformer.jit.pt` — торчскрипт. Автономен и переживает обновления;
         собирается из официального репозитория одной командой (см. докстринг
         download_weights.py).
      2. `codeformer_arch.py` рядом с этим файлом — модуль с классом
         `CodeFormer`, положенный вручную из официального репозитория. Тогда
         подхватывается `codeformer.pth`.
      3. Ничего из этого — восстановление выключено, кадр отдаётся как есть, и
         `meta.stages.restore` объясняет причину.

    Детекция и вклейка лица — facexlib (те же веса, что у GFPGAN). Её модели
    тоже лежат локально: без `model_rootpath` она полезет в GitHub на первом же
    заказе и упрётся в оффлайн.
    """

    weights: Path
    jit_weights: Path
    device: str
    fidelity: float
    _net: Any = field(default=None, init=False, repr=False)
    _helper: Any = field(default=None, init=False, repr=False)
    _kind: str = field(default="none", init=False)
    _reason: str = field(default="", init=False)

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def reason(self) -> str:
        return self._reason

    def load(self) -> None:
        import torch

        if self._kind != "none":
            return

        net = self._load_net(torch)
        if net is None:
            return

        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError as exc:
            # Откат вида: веса-то загрузились, и _load_net уже проставил _kind.
            # Оставить его — значит показывать на /health/ready «backend:
            # torchscript» при выключенном восстановлении, то есть врать ровно
            # там, куда идут разбираться, почему лицо не поправлено
            self._kind = "none"
            self._reason = f"facexlib не установлен: {exc}"
            log.warning("восстановление лица выключено", extra={"reason": self._reason})
            return

        self._helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=str(DIR_FACEXLIB),
        )
        self._net = net

    def _load_net(self, torch: Any) -> Any:
        if self.jit_weights.exists():
            self._kind = "torchscript"
            return torch.jit.load(str(self.jit_weights), map_location=self.device).eval()

        if not self.weights.exists():
            self._reason = f"весов нет: {self.weights}"
            log.warning("восстановление лица выключено", extra={"reason": self._reason})
            return None

        arch = self._arch()
        if arch is None:
            self._reason = (
                "есть codeformer.pth, но нет кода архитектуры: положите рядом "
                "codeformer_arch.py либо соберите codeformer.jit.pt"
            )
            log.warning("восстановление лица выключено", extra={"reason": self._reason})
            return None

        net = arch(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        state = torch.load(str(self.weights), map_location="cpu")
        net.load_state_dict(state.get("params_ema", state))
        self._kind = "state_dict"
        return net.eval()

    @staticmethod
    def _arch() -> Any:
        """Класс архитектуры из вручную положенного модуля, если он есть."""
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from codeformer_arch import CodeFormer  # type: ignore[import-not-found]
        except ImportError:
            return None
        return CodeFormer

    def restore(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """
        :return: кадр и число обработанных лиц (0 — лицо не найдено)
        """
        import torch

        if self._net is None or self._helper is None:
            return image, 0

        self._helper.clean_all()
        self._helper.read_image(image)
        found = self._helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        if not found:
            return image, 0

        self._helper.align_warp_face()
        for cropped in self._helper.cropped_faces:
            tensor = self._to_tensor(torch, cropped)
            try:
                with torch.inference_mode():
                    output = self._net(tensor, w=self.fidelity, adain=True)[0]
                restored = self._to_image(torch, output)
            except Exception as exc:  # noqa: BLE001 — чужая модель, падать нельзя
                log.warning("CodeFormer отказал на лице", extra={"cause": str(exc)})
                restored = cropped
            self._helper.add_restored_face(restored.astype(np.uint8))

        self._helper.get_inverse_affine(None)
        return self._helper.paste_faces_to_input_image(), len(self._helper.cropped_faces)

    def _to_tensor(self, torch: Any, face: np.ndarray) -> Any:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        # Нормировка в [-1, 1] — часть контракта весов, а не вкусовщина
        return ((tensor - 0.5) / 0.5).to(self.device)

    @staticmethod
    def _to_image(torch: Any, output: Any) -> np.ndarray:
        tensor = output.squeeze(0).float().detach().cpu().clamp_(-1, 1)
        array = ((tensor + 1) / 2).permute(1, 2, 0).numpy()
        return cv2.cvtColor((array * 255.0).round(), cv2.COLOR_RGB2BGR)


@dataclass
class Renderer:
    """
    SDXL inpaint, при наличии весов — под ControlNet.

    ControlNet здесь не украшение. Голый inpaint внутри большой маски теряет
    перспективу сцены: ml-service это уже наблюдал на identity_inpaint —
    «под маской у модели нет контекста, искажённые пропорции и артефакты по
    краю». ControlNet возвращает модели структуру, отдавая ей стёртую LaMa
    подложку как управляющий кадр: внутри дыры — правдоподобное продолжение
    сцены, снаружи — точные пиксели шаблона.
    """

    device: str
    dtype_name: str
    _pipe: Any = field(default=None, init=False, repr=False)
    _controlnet: bool = field(default=False, init=False)
    _ip_adapter: bool = field(default=False, init=False)
    _ready: bool = field(default=False, init=False)
    _reason: str = field(default="", init=False)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def has_controlnet(self) -> bool:
        return self._controlnet

    @property
    def has_ip_adapter(self) -> bool:
        return self._ip_adapter

    @property
    def reason(self) -> str:
        return self._reason

    def load(self) -> None:
        import torch
        from diffusers import AutoencoderKL, ControlNetModel

        if self._ready:
            return

        if not DIR_SDXL.exists():
            self._reason = f"весов SDXL нет: {DIR_SDXL} (запустите download_weights.py)"
            log.error("генерация недоступна", extra={"reason": self._reason})
            return

        dtype = getattr(torch, self.dtype_name)
        shared: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}

        # VAE от madebyollin, а не штатный. Штатный SDXL VAE в fp16 переполняется
        # и отдаёт чёрный кадр либо NaN — известный дефект, и лечится он именно
        # подменой VAE, а не понижением до fp32 (то стоит вдвое больше памяти).
        vae = None
        if DIR_VAE.exists():
            vae = AutoencoderKL.from_pretrained(str(DIR_VAE), **shared)
        else:
            log.warning("нет sdxl-vae-fp16-fix — возможен чёрный кадр в fp16")

        # subfolder="v2" обязателен. В репозитории destitech лежат две версии:
        # старая в корне и улучшенная в v2/. Без subfolder diffusers читает
        # корень, то есть молча берёт худшую версию — и download_weights.py её
        # даже не качает (манифест тянет только v2/*), так что вместо тихой
        # деградации будет падение на отсутствующем config.json.
        # variant="fp16" здесь по той же причине, что и у SDXL ниже:
        # download_weights.py тянет только fp16-файлы, и без variant diffusers
        # ищет имена без суффикса и падает на отсутствующем файле. Строка какое-то
        # время жила только на боксе, правкой по месту, — то есть пересборка
        # образа её теряла. Здесь она затем, чтобы больше не терялась.
        controlnet = None
        if (DIR_CONTROLNET / "v2").exists():
            controlnet = ControlNetModel.from_pretrained(
                str(DIR_CONTROLNET), subfolder="v2", variant="fp16", **shared
            )
            self._controlnet = True
        else:
            log.warning("нет ControlNet v2 — работает обычный SDXL inpaint")

        self._pipe = self._build(shared, vae, controlnet)
        self._tune()
        # Строго после _tune: load_ip_adapter кладёт энкодер на self._pipe.device,
        # а на устройство пайплайн переезжает именно там. Вызов до переезда
        # оставил бы энкодер на CPU и уронил бы первый же кадр на несовпадении
        self._attach_ip_adapter()
        self._ready = True
        log.info(
            "SDXL загружен",
            extra={
                "controlnet": self._controlnet,
                "ip_adapter": self._ip_adapter,
                "vae_fix": vae is not None,
                "device": self.device,
            },
        )

    def _build(self, shared: dict[str, Any], vae: Any, controlnet: Any) -> Any:
        from diffusers import (
            StableDiffusionXLControlNetInpaintPipeline,
            StableDiffusionXLInpaintPipeline,
        )

        extra: dict[str, Any] = {"use_safetensors": True}
        if vae is not None:
            extra["vae"] = vae
        if controlnet is not None:
            extra["controlnet"] = controlnet

        cls = (
            StableDiffusionXLControlNetInpaintPipeline
            if controlnet is not None
            else StableDiffusionXLInpaintPipeline
        )

        # variant="fp16" обязателен: download_weights.py тянет только fp16-файлы,
        # и без него diffusers ищет fp32-имена и падает на отсутствующем файле.
        # Повтор без variant — на случай репозитория, где fp16 лежит без суффикса
        try:
            return cls.from_pretrained(str(DIR_SDXL), variant="fp16", **shared, **extra)
        except (OSError, ValueError) as exc:
            log.warning("fp16-вариант не найден, читаю как есть", extra={"cause": str(exc)})
            return cls.from_pretrained(str(DIR_SDXL), **shared, **extra)

    def _tune(self) -> None:
        import torch
        from diffusers import DPMSolverMultistepScheduler

        # Karras-сетка сигм на 25-30 шагах даёт то же, что DDIM на 50: шагов на
        # кадре здесь дороже всего остального вместе взятого
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
        self._pipe.to(self.device)
        # Слайсинг VAE: декодер на 1024 держит пик выше самой диффузии, а стоит
        # это единицы процентов времени
        self._pipe.enable_vae_slicing()
        self._pipe.set_progress_bar_config(disable=True)

        if torch.cuda.is_available():
            # TF32 на матмулях: A40 (Ampere) считает их вдвое быстрее fp32, а
            # точности в диффузии это не стоит ничего
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def warmup(self) -> None:
        """
        Один холостой кадр малого размера.

        Первый прогон дороже последующих на порядок: там компиляция ядер и
        аллокация арен. Без прогрева эта цена попадает в тайминг первого заказа
        и читается как «сервер медленный».
        """
        if not self._ready:
            return
        blank = np.full((512, 512, 3), 127, dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[192:320, 192:320] = 255
        # identity подаётся и на прогреве: ветка с адаптером тянет за собой ещё
        # один прогон CLIP-энкодера и свои ядра, и не прогреть её значит оставить
        # эту цену первому заказу — ровно то, ради чего прогрев и сделан
        self.render(blank, mask, blank, prompt="warmup", steps=2, identity=blank)
        log.info("прогрев завершён")

    def render(
        self,
        plate: np.ndarray,
        mask: np.ndarray,
        control: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str = "",
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        strength: float = DEFAULT_STRENGTH,
        control_scale: float = DEFAULT_CONTROL_SCALE,
        seed: int | None = None,
        identity: np.ndarray | None = None,
        identity_scale: float | None = None,
    ) -> np.ndarray:
        import torch
        from PIL import Image

        if not self._ready:
            raise HTTPException(503, self._reason or "модель не загружена")

        height, width = working_size(*plate.shape[:2])

        def to_pil(array: np.ndarray, gray: bool = False) -> Image.Image:
            resized = cv2.resize(
                array, (width, height), interpolation=cv2.INTER_AREA if gray else cv2.INTER_LANCZOS4
            )
            if gray:
                return Image.fromarray(resized, mode="L")
            return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        arguments: dict[str, Any] = {
            "prompt": prompt or DEFAULT_PROMPT,
            "negative_prompt": negative_prompt or DEFAULT_NEGATIVE,
            "image": to_pil(plate),
            "mask_image": to_pil(mask, gray=True),
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": strength,
            "generator": generator,
        }
        if self._controlnet:
            arguments["control_image"] = to_pil(control)
            arguments["controlnet_conditioning_scale"] = control_scale

        if self._ip_adapter:
            # ip_adapter_image обязателен на КАЖДОМ вызове, а не только когда
            # личность есть. load_ip_adapter переставил UNet в режим
            # `encoder_hid_dim_type="ip_image_proj"`, и тот падает с ValueError,
            # если в added_cond_kwargs нет image_embeds. Поэтому кадр без донора
            # идёт с нейтральной заглушкой и нулевым весом: вклад адаптера
            # умножается на 0, а форма аргументов остаётся прежней
            if identity is None:
                blank = np.full((224, 224, 3), 127, dtype=np.uint8)
                arguments["ip_adapter_image"] = Image.fromarray(blank)
                self._pipe.set_ip_adapter_scale(0.0)
            else:
                arguments["ip_adapter_image"] = Image.fromarray(
                    cv2.cvtColor(identity, cv2.COLOR_BGR2RGB)
                )
                self._pipe.set_ip_adapter_scale(
                    IPADAPTER_SCALE if identity_scale is None else identity_scale
                )

        with torch.inference_mode():
            result = self._pipe(**arguments).images[0]

        return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

    def _attach_ip_adapter(self) -> None:
        """
        Вход личности: IP-Adapter Plus Face поверх собранного пайплайна.

        Взят вариант на CLIP vision (`...vit-h`), а не IP-Adapter-FaceID. FaceID
        точнее по сходству, но считает identity через ArcFace-эмбеддинг
        InsightFace, а официальные веса InsightFace изданы под некоммерческой
        лицензией — для платного продукта это запрет, а не формальность. Сам
        IP-Adapter под Apache 2.0, CLIP-энкодер тоже, поэтому связка чистая.

        Отсутствие весов не роняет сервис, а понижает качество — как и всё
        остальное здесь: без адаптера генерация вернётся к «нарисуй человека по
        тексту», что видно в `/health/ready` и в `meta.identity_applied`.
        """
        if not IPADAPTER_ENABLED:
            log.info("IP-Adapter выключен переменной GPU_IP_ADAPTER")
            return

        adapter = DIR_IPADAPTER / IPADAPTER_SUBFOLDER / IPADAPTER_WEIGHT
        encoder = DIR_IPADAPTER / IPADAPTER_ENCODER
        if not adapter.exists() or not encoder.exists():
            log.warning(
                "нет IP-Adapter — личность не переносится, генерация только по тексту",
                extra={"adapter": str(adapter), "encoder": str(encoder)},
            )
            return

        try:
            # image_encoder_folder со слэшем внутри diffusers трактует как путь от
            # КОРНЯ репозитория, а без слэша — как подпапку внутри subfolder. Нам
            # нужен именно первый случай: адаптер лежит в sdxl_models/, а парный
            # ему ViT-H энкодер — в models/, это разные ветки одного репозитория
            self._pipe.load_ip_adapter(
                str(DIR_IPADAPTER),
                subfolder=IPADAPTER_SUBFOLDER,
                weight_name=IPADAPTER_WEIGHT,
                image_encoder_folder=IPADAPTER_ENCODER,
                local_files_only=True,
            )
            self._pipe.set_ip_adapter_scale(IPADAPTER_SCALE)
        except Exception as exc:  # noqa: BLE001 — причина уходит в лог целиком
            # Снять половину установки обязательно: load_ip_adapter успевает
            # переписать attention-процессоры UNet до того, как споткнётся на
            # энкодере, и пайплайн с процессорами, но без энкодера, падает уже
            # на кадре — то есть на заказе, а не на старте
            log.warning("IP-Adapter не подключился, работаю без переноса личности",
                        extra={"cause": str(exc)})
            try:
                self._pipe.unload_ip_adapter()
            except Exception:  # noqa: BLE001
                pass
            return

        self._ip_adapter = True
        log.info("IP-Adapter подключён", extra={"scale": IPADAPTER_SCALE})


@dataclass
class Flux2Renderer:
    """
    FLUX.2 [klein] 4B — генеративный редактор под демонстрационный путь.

    ЧЕМ ОН ОТЛИЧАЕТСЯ ОТ `Renderer`. Тот инпейнтит по маске: рисует внутри дыры
    и не трогает остальное. Этот так не умеет вовсе — он переписывает кадр
    целиком и возвращает СВОЮ сцену, расходясь с шаблоном на два десятка уровней
    по всему полю. Зато личность он переносит вдвое точнее: замер по семи детям
    на трёх шаблонах дал 0.56 против 0.25 у SDXL с IP-Adapter.

    Отсюда разделение обязанностей: этот класс отвечает только за голову, а
    возврат шаблона на место — за `transplant` в ml-service.

    ПОЧЕМУ ЗАГРУЗКА ОТЛОЖЕННАЯ И ПАДАЮЩАЯ МЯГКО. `Flux2KleinPipeline` появился в
    diffusers недавно и есть не во всякой сборке: в образе с torch 2.4 его нет и
    быть не может. Сервис от этого не обязан переставать работать — он обязан
    честно сказать на `/health/ready`, что демонстрационный путь недоступен, и
    продолжать обслуживать остальные.

    Веса лицензионно чистые (Apache 2.0) и не тянут за собой ни одной модели
    распознавания лиц — ради этого их и выбирали.
    """

    model_id: str
    device: str
    _pipe: Any = field(default=None, init=False, repr=False)
    _ready: bool = field(default=False, init=False)
    _reason: str = field(default="", init=False)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def reason(self) -> str:
        return self._reason

    def load(self) -> None:
        import torch

        if self._ready or self._reason:
            return

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            self._reason = f"diffusers без Flux2KleinPipeline: {exc}"
            log.warning("демонстрационный путь выключен", extra={"reason": self._reason})
            return

        try:
            pipe = Flux2KleinPipeline.from_pretrained(self.model_id, dtype=torch.bfloat16)
        except Exception as exc:  # noqa: BLE001 — нет весов, нет сети, битый кэш
            self._reason = f"веса не загрузились: {exc}"
            log.warning("демонстрационный путь выключен", extra={"reason": self._reason})
            return

        # Оффлоад, а не .to(device): модель с текстовым энкодером занимает больше,
        # чем остаётся на карте рядом с LaMa, и держать её целиком незачем — пик
        # с оффлоадом 10.6 ГБ против 24 доступных
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        self._ready = True
        log.info("FLUX.2 загружен", extra={"model": self.model_id})

    def render(self, template: np.ndarray, donor: np.ndarray, *, steps: int,
               guidance: float, seed: int | None, prompt: str | None = None) -> np.ndarray:
        """
        Кадр по двум картинкам: шаблон первым референсом, лицо донора вторым.

        Порядок референсов — часть контракта промпта: он говорит «лицо из ВТОРОЙ
        картинки на ребёнка из ПЕРВОЙ», и перестановка меняет смысл на обратный.

        :param prompt: текст запроса. None — `GPU_FLUX_PROMPT`, то есть
            умолчание сервера. Передаётся заказом ради подбора формулировки:
            она итеративная, а через переменную окружения каждый вариант стоит
            перезапуска сервиса
        """
        import torch
        from PIL import Image

        if not self._ready:
            raise HTTPException(503, self._reason or "FLUX.2 не загружен")

        height, width = flux_size(*template.shape[:2])
        def to_pil(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
            resized = cv2.resize(array, size, interpolation=cv2.INTER_LANCZOS4)
            return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        donor_side = donor.shape[0] // 16 * 16
        result = self._pipe(
            image=[to_pil(template, (width, height)),
                   to_pil(donor, (donor_side, donor_side))],
            prompt=prompt if prompt is not None else FLUX_PROMPT,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)


# --- 4. Состояние процесса ---------------------------------------------------


@dataclass
class Models:
    renderer: Renderer
    eraser: LamaEraser
    restorer: FaceRestorer
    flux: Flux2Renderer

    def load(self) -> None:
        self.renderer.load()
        self.eraser.load()
        if RESTORE_ENABLED:
            self.restorer.load()
        if DEMO_ENABLED:
            self.flux.load()

    def status(self) -> dict[str, Any]:
        return {
            "sdxl": {
                "ready": self.renderer.ready,
                "controlnet": self.renderer.has_controlnet,
                "reason": self.renderer.reason,
                "path": str(DIR_SDXL),
            },
            "identity": {
                "enabled": IPADAPTER_ENABLED,
                "backend": "ip-adapter-plus-face" if self.renderer.has_ip_adapter else "none",
                "scale": IPADAPTER_SCALE,
                "path": str(DIR_IPADAPTER),
            },
            "erase": {"enabled": ERASE_ENABLED, "backend": self.eraser.kind},
            "demo": {
                "enabled": DEMO_ENABLED,
                "ready": self.flux.ready,
                "reason": self.flux.reason,
                "model": FLUX_MODEL,
                "pipelines": _mlservice_reason() or "ok",
            },
            "restore": {
                "enabled": RESTORE_ENABLED,
                "backend": self.restorer.kind,
                "reason": self.restorer.reason,
                "fidelity": CODEFORMER_W,
            },
        }


MODELS = Models(
    renderer=Renderer(device=DEVICE, dtype_name=DTYPE_NAME),
    eraser=LamaEraser(path=FILE_LAMA, device=DEVICE),
    restorer=FaceRestorer(
        weights=FILE_CODEFORMER,
        jit_weights=FILE_CODEFORMER_JIT,
        device=DEVICE,
        fidelity=CODEFORMER_W,
    ),
    flux=Flux2Renderer(model_id=FLUX_MODEL, device=DEVICE),
)
QUEUE = GpuQueue(QUEUE_SLOTS, QUEUE_MAX_WAITING, QUEUE_WAIT_TIMEOUT)


# --- 5. Схемы ----------------------------------------------------------------


class RenderRequest(BaseModel):
    """
    Пакет одного кадра. Поля повторяют `refine/payload.py::build_packet`.

    `donor_crop`, `embedding`, `start_step` и `true_cfg` приняты ради
    совместимости с уже написанным клиентом и в генерацию не идут — см.
    докстринг модуля. Проверяются они всё равно: пусть формат разойдётся
    громко здесь, а не тихо в качестве результата.
    """

    base_image: str
    mask_image: str
    donor_crop: str | None = None
    embedding: dict[str, Any] | None = None

    prompt: str = ""
    negative_prompt: str = ""

    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=150)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE, ge=0.0, le=30.0)
    strength: float = Field(default=DEFAULT_STRENGTH, ge=0.0, le=1.0)
    control_scale: float = Field(default=DEFAULT_CONTROL_SCALE, ge=0.0, le=2.0)
    seed: int | None = None

    # Принимаются и игнорируются
    fidelity: float | None = None
    start_step: int | None = None
    true_cfg: float | None = None

    erase: bool | None = None
    restore_face: bool | None = None
    output_format: str = "png"

    encoding: dict[str, Any] | None = None
    bytes: dict[str, Any] | None = None

    @field_validator("output_format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value.lower() not in {"png", "jpg", "jpeg"}:
            raise ValueError("output_format: только png или jpg")
        return value.lower()

    @field_validator("embedding")
    @classmethod
    def _valid_embedding(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Вектор не используется, но проверяется.

        Пропустить сюда вектор чужой размерности значит согласиться, что дефект
        нашей экстракции всплывёт когда-нибудь потом — например, когда сюда
        доедет IP-Adapter и молча получит мусор.
        """
        if value is None:
            return None
        size = value.get("size")
        dtype = value.get("dtype")
        if size != _EMBEDDING_SIZE or dtype != _EMBEDDING_DTYPE:
            raise ValueError(
                f"embedding: ожидались size={_EMBEDDING_SIZE} dtype={_EMBEDDING_DTYPE}, "
                f"пришло size={size} dtype={dtype}"
            )
        return value


# --- 6. Пайплайн одного кадра ------------------------------------------------


@contextmanager
def stage(timings: dict[str, float], name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - started, 3)


def run_pipeline(request: RenderRequest) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Синхронный проход: стереть -> нарисовать -> вернуть в кадр -> восстановить.

    Порядок двух последних шагов именно такой, и это стоит держать в голове:
    CodeFormer работает не над генерацией, а над уже собранным кадром, последним
    действием. Отсюда `confine` — восстановление обязано остаться внутри маски,
    как и вклейка до него.

    Выполняется В ПОТОКЕ (см. `render_endpoint`), а не в цикле событий: торч
    отпускает GIL на счёте, и держать на нём event loop значит перестать
    отвечать на /health ровно тогда, когда это важнее всего — под нагрузкой.
    """
    timings: dict[str, float] = {}
    meta: dict[str, Any] = {}

    original = decode_image(request.base_image, "base_image")
    mask = normalise_mask(decode_image(request.mask_image, "mask_image", grayscale=True),
                          original.shape[:2])

    if mask_bbox(mask) is None:
        raise HTTPException(422, "mask_image: маска пустая — переписывать нечего")

    # Донорский кроп — вход личности для IP-Adapter. До появления адаптера он
    # только разбирался и выбрасывался; теперь это единственный источник того,
    # НА КОГО должен быть похож результат. Разбор остаётся обязательным и когда
    # адаптера нет: битый base64 здесь означает сломанную сериализацию на той
    # стороне, и узнать об этом лучше сразу, а не по внешности модели
    identity = decode_image(request.donor_crop, "donor_crop") if request.donor_crop else None

    erase = ERASE_ENABLED if request.erase is None else request.erase
    plate = original
    if erase:
        with stage(timings, "erase"):
            plate = MODELS.eraser.erase(original, mask)
        meta["erase_backend"] = MODELS.eraser.kind

    # Управляющий кадр — стёртая подложка. Она даёт ControlNet структуру сцены
    # снаружи маски и правдоподобное продолжение внутри; вариант «серая заливка»
    # проверялся у ml-service заливкой перед fal и дал светлый ореол по границе
    control = plate

    with stage(timings, "diffusion"):
        generated = MODELS.renderer.render(
            plate,
            mask,
            control,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance=request.guidance_scale,
            strength=request.strength,
            control_scale=request.control_scale,
            seed=request.seed,
            identity=identity,
        )

    with stage(timings, "composite"):
        alpha = feather_alpha(mask)
        if MATCH_LEVELS:
            generated, match_meta = match_levels(original, generated, mask)
            meta.update(match_meta)
        result = composite(original, generated, mask, alpha=alpha)

    restore = RESTORE_ENABLED if request.restore_face is None else request.restore_face
    faces = 0
    if restore:
        with stage(timings, "restore"):
            restored, faces = MODELS.restorer.restore(result)
            result, clamped, peak = confine(result, restored, alpha)
            meta["restore_clamped_px"] = clamped
            meta["restore_spill_peak"] = peak
    meta["faces_restored"] = faces

    if EMPTY_CACHE:
        import torch

        torch.cuda.empty_cache()

    # `identity_applied` — не «донор пришёл», а «донор доехал до генерации».
    # Разница видна только здесь: кроп без весов адаптера и кроп с весами дают
    # внешне одинаковый ответ и совершенно разное сходство, и вопрос «почему
    # модель не похожа» должен разбираться по этому полю, а не перебором версий
    identity_applied = identity is not None and MODELS.renderer.has_ip_adapter
    ignored = [
        name
        for name in ("fidelity", "start_step", "true_cfg", "embedding")
        if getattr(request, name, None) is not None
    ]
    if request.donor_crop and not identity_applied:
        ignored.append("donor_crop")

    meta.update(
        {
            "size": {"height": int(original.shape[0]), "width": int(original.shape[1])},
            "working_size": dict(zip(("height", "width"), working_size(*original.shape[:2]))),
            "mask_px": int(np.count_nonzero(mask)),
            "controlnet": MODELS.renderer.has_controlnet,
            "ip_adapter": MODELS.renderer.has_ip_adapter,
            "identity_applied": identity_applied,
            "identity_scale": IPADAPTER_SCALE if identity_applied else None,
            "prompt_defaulted": not request.prompt,
            "ignored": ignored,
            "timings": timings,
        }
    )
    return result, meta


def feather_alpha(mask: np.ndarray) -> np.ndarray:
    """
    Вес генерации в каждом пикселе: 1 — берём её, 0 — берём шаблон.

    Вынесено из `composite` отдельной функцией не ради красоты. Той же альфой
    ограничивается восстановление лица (см. `run_pipeline`), и посчитай её там
    заново — граница вклейки и граница восстановления разошлись бы на первом же
    расхождении формул. Один источник обязателен там, где два потребителя.

    Растушёвка края — доля от размера маски, а не константа: фиксированные
    три пикселя на окне 400 px и на окне 2000 px означают совершенно разное.
    """
    height, width = mask.shape[:2]
    box = mask_bbox(mask)
    span = max(box[2] - box[0], box[3] - box[1]) if box else max(height, width)
    radius = max(3, int(span * 0.02)) | 1

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (radius, radius), 0)
    # Размытие маски выносит ненулевую альфу ЗА исходный контур; там, где маска
    # была чистым нулём, она обязана остаться нулём
    alpha[mask == 0] = 0.0
    return alpha


def match_levels(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Приводит тон генерации к тону шаблона по пикселям ВНЕ маски.

    Перенесено из ml-service (`app/pipelines/composite.py::_match_levels`), где
    написано против светлого ореола вокруг головы, и здесь нужно ровно за тем
    же. Диффузия — правка глобальная: кадр проходит через VAE целиком и
    возвращается не в том тоне, в каком пришёл. Сам по себе этот увод незаметен,
    потому что двигает всё разом. Но мы берём из генерации не весь кадр, а
    область маски, — и её края оказываются светлее (или темнее) соседнего
    шаблона по всему периметру. Это и читается как свечение или тёмный кант.

    Растушёвка от этого не спасает и спасать не может: она размывает КРАЙ, а
    увод живёт внутри, где альфа равна единице и пиксель копируется из генерации
    как есть. Мягкий край делает пятно мягким — то есть похожим на свечение ещё
    больше.

    Чинится это тем, что у нас есть эталон. ВНЕ маски шаблон и генерация
    изображают одно и то же (LaMa рисует только под маской, снаружи подложка —
    это шаблон), и там они обязаны совпадать; расхождение там и есть увод,
    измеренный по десяткам тысяч пикселей той же сцены. По ним на канал считается
    прямая `gain * x + bias` (МНК), и ею правится генерация.

    ПОЧЕМУ ЭТО ЛОКАЛЬНАЯ ПОПРАВКА, ХОТЯ ПРЯМАЯ ПРИМЕНЯЕТСЯ КО ВСЕМУ КАДРУ.
    Поправленная генерация попадает в результат только там, где альфа больше
    нуля, — снаружи маски `composite` кладёт байт шаблона и не смотрит на
    генерацию вовсе. То есть тон за пределами маски не может измениться по
    построению, и отдельного ограничения области для этого не нужно.

    Никакого «улучшения картинки» здесь не происходит: не увела диффузия тон —
    получится единица со сдвигом в ноль, и вклейка останется прежней побитово.

    :return: поправленная генерация и отчёт о поправке
    """
    scale = MATCH_PREVIEW_PX / float(max(original.shape[:2]))
    if scale >= 1.0:
        small_original, small_generated, small_mask = original, generated, mask
    else:
        size = (max(1, int(original.shape[1] * scale)), max(1, int(original.shape[0] * scale)))
        small_original = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
        small_generated = cv2.resize(generated, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    # Порог не ноль: считать поправку по краю растушёвки нельзя — там пиксели
    # наполовину из генерации, и они утянули бы её на себя
    outside = small_mask <= 8
    if int(np.count_nonzero(outside)) < 64:
        # Маска накрыла кадр почти целиком — сравнивать не с чем. Молча
        # перемножить на единицу честнее, чем считать поправку по десятку точек
        log.info("тон генерации не поправлен: вне маски слишком мало пикселей")
        return generated, {"matched": False, "match_reason": "outside_too_small"}

    before = float(
        cv2.absdiff(small_original, small_generated).mean(axis=2)[outside].mean()
    )

    fit: list[tuple[float, float]] = []
    for channel in range(original.shape[2]):
        source = small_generated[..., channel][outside].astype(np.float64)
        target = small_original[..., channel][outside].astype(np.float64)

        source_mean, target_mean = float(source.mean()), float(target.mean())
        variance = float(source.var())

        if variance < MATCH_MIN_VARIANCE:
            gain = 1.0
        else:
            gain = float(((source - source_mean) * (target - target_mean)).mean() / variance)
        fit.append((gain, target_mean - gain * source_mean))

    # Поправка держится на одном допущении: ВНЕ маски обе картинки изображают
    # одну и ту же сцену. Вышедшая за пределы прямая означает, что допущение не
    # выполнено — генерация уехала композицией или вернулась перекрашенной, — и
    # правильного ответа у поправки тогда нет. Отказ честнее подгонки
    if any(
        not MATCH_GAIN_MIN <= gain <= MATCH_GAIN_MAX or abs(bias) > MATCH_BIAS_MAX
        for gain, bias in fit
    ):
        log.warning(
            "тон генерации не поправлен: вне маски она расходится с шаблоном "
            "сильнее, чем объясняется уводом тона",
            extra={"fit": [(round(gain, 3), round(bias, 1)) for gain, bias in fit]},
        )
        return generated, {
            "matched": False,
            "match_reason": "out_of_range",
            "match_gain": [round(gain, 3) for gain, _ in fit],
            "match_bias": [round(bias, 1) for _, bias in fit],
            "match_before": round(before, 2),
        }

    corrected = generated.astype(np.float32)
    for channel, (gain, bias) in enumerate(fit):
        corrected[..., channel] = corrected[..., channel] * gain + bias
    # Округление, а не отбрасывание дробной части: `astype` режет вниз, и на
    # единичной поправке (gain=1, bias=0) кадр от этого темнеет на пол-уровня —
    # то есть «поправка ничего не делает» перестало бы быть правдой
    corrected = np.rint(np.clip(corrected, 0, 255)).astype(np.uint8)

    after = float(
        cv2.absdiff(
            small_original,
            corrected if scale >= 1.0 else cv2.resize(corrected, small_original.shape[1::-1],
                                                      interpolation=cv2.INTER_AREA),
        ).mean(axis=2)[outside].mean()
    )

    return corrected, {
        "matched": True,
        "match_gain": [round(gain, 3) for gain, _ in fit],
        "match_bias": [round(bias, 1) for _, bias in fit],
        # Насколько генерация расходилась с шаблоном вне маски ДО и ПОСЛЕ
        # поправки. Именно первое число и превращается в кант по границе, если
        # поправку не делать; второе показывает, сколько от него осталось
        "match_before": round(before, 2),
        "match_after": round(after, 2),
    }


def composite(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """
    Возврат генерации в исходный кадр по маске.

    Главное здесь — обещание «вне маски шаблон не тронут ни на пиксель». Оно
    держится не аккуратностью, а устройством: там, где маска ноль, альфа тоже
    ноль, и в результат попадает исходный байт. Без этого диффузия, посчитанная
    в 1024 и растянутая обратно в 4K, размыла бы весь разворот ради головы.

    :param alpha: готовые веса из `feather_alpha`. Передаются, когда тот же вес
        нужен вызывающему коду; None — посчитать здесь
    """
    height, width = original.shape[:2]
    if generated.shape[:2] != (height, width):
        generated = cv2.resize(generated, (width, height), interpolation=cv2.INTER_LANCZOS4)

    if alpha is None:
        alpha = feather_alpha(mask)
    weights = alpha[..., None]

    blended = original.astype(np.float32) * (1.0 - weights) + generated.astype(np.float32) * weights
    blended = np.rint(np.clip(blended, 0, 255)).astype(np.uint8)

    # Явное восстановление шаблона там, где маска пуста. Арифметика с нулевым
    # весом даёт то же самое, но «даёт то же самое» — это про округление, а
    # обещание «фон нетронут» держаться на округлении не должно
    untouched = alpha <= 0.0
    blended[untouched] = original[untouched]
    return blended


def confine(
    before: np.ndarray, after: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """
    Возвращает правку, ограниченную областью маски. И считает, что отрезали.

    Нужно восстановлению лица. `FaceRestoreHelper.paste_faces_to_input_image`
    вклеивает поправленное лицо по СВОЕЙ аффинной матрице и по своей маске
    разбора — то есть пишет по всему кадру и про нашу маску ничего не знает. А
    вызывается оно ПОСЛЕ `composite`, последним шагом, и потому способно молча
    переписать фон, который вклейка обязалась не трогать.

    На проверенных кадрах этого не происходило: область разбора facexlib —
    кожа и волосы уже сгенерированной головы, и она лежит внутри маски. Но
    «не происходило» — это про конкретные кадры, а обещание «шаблон дойдёт до
    печати без единого изменения» держаться на везении не может: хватит лица,
    которое детектор найдёт не там, или причёски, дошедшей до края маски.

    Ограничение идёт той же альфой, что и вклейка, а не жёстким срезом по маске:
    иначе поправка обрывалась бы ступенькой ровно там, где вклейка переходит
    плавно.

    ЧТО СЧИТАЕТСЯ ВЫХОДОМ ЗА МАСКУ. Не всякий изменившийся байт: facexlib
    смешивает кадр с поправленным лицом по всему полю в float и переводит
    обратно в uint8, и от одного этого перевода вне маски набегают тысячи
    пикселей, разошедшихся на единицу. Замер на пяти шаблонах: 1000-4700
    пикселей на кадр, и КАЖДЫЙ отличается не больше чем на 4 уровня. Считать их
    выходом за маску значит показывать тревожное число там, где ничего не
    произошло, — поэтому порог, и поэтому в отчёте есть не только счётчик, но и
    величина: она отличает пыль округления от настоящей вклейки мимо маски.

    :return: кадр и (число пикселей вне маски, изменённых заметно; наибольшее
        отклонение в уровнях). Нули — восстановление уложилось в маску само
    """
    if after.shape != before.shape:
        raise HTTPException(500, "восстановление вернуло кадр другого размера")

    outside = alpha <= 0.0
    spill = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    spill[~outside] = 0
    clamped = int(np.count_nonzero(spill > _RESTORE_SPILL_LEVELS))
    peak = int(spill.max())

    weights = alpha[..., None]
    blended = before.astype(np.float32) * (1.0 - weights) + after.astype(np.float32) * weights
    blended = np.rint(np.clip(blended, 0, 255)).astype(np.uint8)
    blended[outside] = before[outside]

    if clamped:
        log.warning(
            "восстановление лица вышло за маску — правка вне маски отброшена",
            extra={"clamped_px": clamped, "clamped_peak": peak},
        )
    return blended, clamped, peak


# --- 7. Приложение -----------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=_env("GPU_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("веса", extra={"root": str(WEIGHTS_ROOT)})

    # Загрузка в потоке: она занимает десятки секунд, и держать на ней цикл
    # событий значит не отвечать на healthcheck докера всё это время
    await asyncio.to_thread(MODELS.load)
    if _env_bool("GPU_WARMUP", True):
        await asyncio.to_thread(MODELS.renderer.warmup)

    yield

    MODELS.renderer._pipe = None
    if DEVICE.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="projectx GPU inference", version="0.1.0", lifespan=lifespan)


@app.get("/health", summary="Liveness")
async def health() -> dict[str, Any]:
    """Процесс жив. Про веса ничего не утверждает — для этого /health/ready."""
    return {"status": "ok", "queue": QUEUE.stats}


@app.get("/health/ready", summary="Readiness")
async def ready() -> JSONResponse:
    """
    Готовность к работе. degraded — сервис отвечает, но качество понижено:
    нет ControlNet, нет IP-Adapter, нет LaMa, нет восстановления лица.

    Отсутствие IP-Adapter стоит в этом списке не для полноты. Без него сервер
    отдаёт технически корректный кадр с чужим лицом, и заметно это не по коду
    ошибки, а по жалобе заказчика — то есть через сутки. Пусть светится здесь.
    """
    status = MODELS.status()
    identity_ok = MODELS.renderer.has_ip_adapter or not IPADAPTER_ENABLED
    if not MODELS.renderer.ready:
        state = "unavailable"
    elif not (status["sdxl"]["controlnet"] and identity_ok and MODELS.eraser.kind == "lama"):
        state = "degraded"
    else:
        state = "ready"

    body = {
        "status": state,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "models": status,
        "queue": QUEUE.stats,
        "limits": {"max_side": MAX_SIDE, "min_side": MIN_SIDE},
    }
    return JSONResponse(body, status_code=200 if state != "unavailable" else 503)


@app.post("/v1/template-render", summary="Генерация головы на шаблоне")
@app.post("/render", summary="Алиас под HostedFluxPuLIDBackend")
async def render_endpoint(request: RenderRequest) -> dict[str, Any]:
    """
    Один кадр. Ответ — `{"image": base64, "meta": {...}}`.

    Ключ `image` выбран не произвольно: `adapter.HostedFluxPuLIDBackend._image`
    читает именно его (либо `images[0].data`), и менять его нельзя, не тронув
    клиента.
    """
    if not MODELS.renderer.ready:
        raise HTTPException(503, MODELS.renderer.reason or "модель не загружена")

    try:
        async with QUEUE.slot() as waited:
            started = time.perf_counter()
            image, meta = await asyncio.to_thread(run_pipeline, request)
            meta["queue_wait_s"] = round(waited, 3)
            meta["total_s"] = round(time.perf_counter() - started, 3)
    except QueueFull as exc:
        # 503, а не 429: у клиента 503 в списке повторяемых, а 429 он тоже
        # повторяет, но семантика «очередь занята» здесь именно про сервис
        raise HTTPException(503, f"очередь переполнена: {exc}") from exc
    except QueueTimeout as exc:
        raise HTTPException(503, f"перегрузка: {exc}") from exc

    return {
        "image": encode_image(image, request.output_format),
        "encoding": f"image/{'jpeg' if request.output_format in {'jpg', 'jpeg'} else 'png'}",
        "meta": meta,
    }


class EraseRequest(BaseModel):
    """
    Пакет стирания. Те же поля, что у генерации, но нужны только два.

    Отдельная схема, а не переиспользование `RenderRequest`: у того половина
    полей обязательна для диффузии и здесь смысла не имеет, а необязательные
    поля в запросе, который их игнорирует, — это приглашение прислать их и
    удивиться, что ничего не изменилось.
    """

    base_image: str
    mask_image: str
    output_format: str = "png"

    @field_validator("output_format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value.lower() not in {"png", "jpg", "jpeg"}:
            raise ValueError("output_format: только png или jpg")
        return value.lower()


@app.post("/v1/erase", summary="Стереть содержимое маски (LaMa)")
async def erase_endpoint(request: EraseRequest) -> dict[str, Any]:
    """
    Стирание без генерации: под маской остаётся правдоподобное продолжение сцены.

    Существует ради пересадки головы в ml-service. Там генеративный редактор
    отдаёт свою голову, но СТАРУЮ убрать нечем: вклейка накрывает только новую,
    а под шаблонной маской остаётся ободок прежней причёски. Класть туда
    выровненную генерацию нельзя — её воротник и плечи сдвинуты подобием
    относительно шаблонных, и на стыке шеи получается два воротника вместо
    одного (замерено: 3458 пикселей расходятся с шаблоном больше чем на 32
    уровня). Правильное содержимое там — продолжение шаблонной одежды и фона,
    а это ровно то, что делает LaMa.

    Очередь общая с генерацией: карта одна, и стирание на ней не бесплатное.
    """
    if MODELS.eraser.kind == "none":
        raise HTTPException(503, "стирание недоступно: веса не загружены")

    original = decode_image(request.base_image, "base_image")
    mask = normalise_mask(decode_image(request.mask_image, "mask_image", grayscale=True),
                          original.shape[:2])
    if mask_bbox(mask) is None:
        raise HTTPException(422, "mask_image: маска пустая — стирать нечего")

    timings: dict[str, float] = {}
    try:
        async with QUEUE.slot() as waited:
            with stage(timings, "erase"):
                plate = await asyncio.to_thread(MODELS.eraser.erase, original, mask)
    except QueueFull as exc:
        raise HTTPException(503, f"очередь переполнена: {exc}") from exc
    except QueueTimeout as exc:
        raise HTTPException(503, f"перегрузка: {exc}") from exc

    return {
        "image": encode_image(plate, request.output_format),
        "encoding": f"image/{'jpeg' if request.output_format in {'jpg', 'jpeg'} else 'png'}",
        "meta": {
            "erase_backend": MODELS.eraser.kind,
            "mask_px": int(np.count_nonzero(mask)),
            "timings": timings,
            "queue_wait_s": round(waited, 3),
        },
    }


class DemoRequest(BaseModel):
    """
    Пакет демонстрации: шаблон и СЫРОЕ фото ребёнка.

    Ни маски, ни кропа, ни промпта здесь нет намеренно. Всё это сервер считает
    сам — в этом и смысл: показ должен выглядеть как «загрузили фотографию,
    получили картинку», а не как последовательность из четырёх запросов.
    """

    base_image: str
    donor_photo: str

    # Текст запроса. None — берётся GPU_FLUX_PROMPT, то есть промпт-запрет,
    # подобранный под перенос ОДНОГО лица. Поле нужно затем, что подбор
    # формулировки — работа итеративная: каждый вариант через переменную
    # окружения стоит правки .env и перезапуска сервиса на боксе, а через
    # запрос — одной строки в команде. Значение по умолчанию не меняется,
    # поэтому прежние вызовы работают ровно как раньше
    prompt: str | None = Field(default=None, max_length=2000)

    # Чем меряется масштаб посадки головы: face | head | blend. None — как
    # зашито в ml-service (SCALE_MODE="head"). Ручка появилась не про запас:
    # "head" меряет силуэт «волосы плюс лицо», и как только генерация начинает
    # приносить ЧУЖУЮ причёску, эта мерка становится неверной по построению —
    # пышные волосы ужмут лицо, гладкие раздуют
    scale_mode: str | None = None

    # Доли маски головы, в высотах лица персонажа. None — умолчания сервера
    # (GPU_DEMO_DILATE / _FEATHER / _NECK), совпадающие с боевым MaskProfile.
    #
    # ЗАЧЕМ ЗАПРОСОМ. Когда генерация приносит причёску КРУПНЕЕ шаблонной, видимым
    # контуром головы становится не прядь, а кромка маски: снаружи шаблон, внутри
    # генерация, между ними геометрический спад на `feather`. На пёстром фоне это
    # читается как вырезанная накладка. Лечится подбором двух чисел, а подбор
    # через переменные окружения — это перезапуск сервиса на каждую попытку.
    dilate: float | None = Field(default=None, ge=0.0, le=1.0)
    feather: float | None = Field(default=None, ge=0.0, le=1.0)
    neck: float | None = Field(default=None, ge=0.0, le=1.0)

    seed: int | None = None
    # None, а не восьмёрка по умолчанию: иначе «не указали» и «указали восемь»
    # неразличимы, и правило из `demo_steps` не смогло бы сработать ни разу.
    # Явно названное число по-прежнему уважается и ничем не переопределяется
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance_scale: float = Field(default=FLUX_GUIDANCE, ge=0.0, le=10.0)
    output_format: str = "png"

    @field_validator("output_format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        if value.lower() not in {"png", "jpg", "jpeg"}:
            raise ValueError("output_format: только png или jpg")
        return value.lower()

    @field_validator("scale_mode")
    @classmethod
    def _known_scale_mode(cls, value: str | None) -> str | None:
        """
        Опечатка в режиме — это молча другой масштаб головы, а не отказ.

        Проверять здесь, а не в ml-service: там неизвестное значение просто
        провалится в ветку `head`, и кадр выйдет правдоподобным, но посчитанным
        не тем способом, о котором просили. Ловить такое по числам потом дорого.
        """
        if value is None:
            return None
        if value.lower() not in {"face", "head", "blend"}:
            raise ValueError("scale_mode: только face, head или blend")
        return value.lower()


def run_demo(request: DemoRequest) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Весь путь одним проходом: фото и шаблон на входе, готовый разворот на выходе.

        разметка шаблона  -> маска головы и её силуэт
        сетка лица донора -> выровненный кроп лица
        FLUX.2            -> кадр с новой головой, но со СВОЕЙ сценой
        LaMa              -> шаблон со стёртой старой головой
        transplant        -> голова из генерации в шаблон, остальное побитово

    Порядок не переставляется: стирание идёт по силуэту, а силуэт считается по
    шаблону, и обе величины нужны пересадке одновременно.
    """
    modules = mlservice()
    head_mask, transplant = modules["head_mask"], modules["transplant"]
    timings: dict[str, float] = {}
    meta: dict[str, Any] = {}

    template = decode_image(request.base_image, "base_image")
    photo = decode_image(request.donor_photo, "donor_photo")

    with stage(timings, "geometry"):
        points = transplant._full_frame_landmarks(template)
        if points is None:
            raise HTTPException(422, "base_image: на шаблоне не найден персонаж")
        geometry = head_mask.face_geometry(points)

        donor_points = head_mask.try_landmarks(photo)
        if donor_points is None:
            raise HTTPException(
                422,
                "donor_photo: лицо не найдено. Нужен портрет, где лицо занимает "
                "хотя бы пятую часть кадра — это предел детектора, а не каприз",
            )
        donor = _donor_crop(photo, donor_points, modules)
        # Доли маски разрешаются один раз и идут и в маску, и в пересадку: разъехавшись,
        # они дадут плиту, посчитанную по одной геометрии, и вклейку по другой
        dilate = DEMO_DILATE_RATIO if request.dilate is None else request.dilate
        feather = DEMO_FEATHER_RATIO if request.feather is None else request.feather
        neck = DEMO_NECK_RATIO if request.neck is None else request.neck
        head = head_mask.build(template, dilate, feather, neck)
        silhouette = transplant.head_silhouette(template, points, geometry["face_height"])

    # Шаги выбираются ПОСЛЕ геометрии: правило смотрит на высоту лица, а она
    # известна только отсюда
    steps, steps_meta = demo_steps(
        geometry["face_height"], *template.shape[:2], request.steps)
    meta.update(steps_meta)

    with stage(timings, "generate"):
        generated = MODELS.flux.render(
            template, donor,
            steps=steps, guidance=request.guidance_scale, seed=request.seed,
            prompt=request.prompt,
        )

    plate = None
    if silhouette is not None and MODELS.eraser.kind != "none":
        with stage(timings, "erase"):
            plate = MODELS.eraser.erase(template, silhouette)
    elif silhouette is None:
        # Без разметки силуэта нет, а стирать по рабочей маске нельзя: она
        # расширена на 29 пикселей за голову, и LaMa заменит целый фон догадкой
        log.warning("силуэт головы не построен — пересадка пойдёт без стирания")

    with stage(timings, "transplant"):
        pasted = transplant.transplant(
            template, generated,
            dilate, feather, neck,
            plate=plate,
            scale_mode=request.scale_mode,
        )

    meta.update(pasted.meta)
    meta.update(
        {
            "size": {"height": int(template.shape[0]), "width": int(template.shape[1])},
            "flux_size": dict(zip(("height", "width"), flux_size(*template.shape[:2]))),
            "mask_px": int(np.count_nonzero(head.mask > 127)),
            # Промпт едет в мету целиком, а не флагом «был переопределён»: при
            # подборе формулировки кадр без текста, которым он получен, —
            # бесполезен, а перебирать варианты предстоит десятками
            "prompt": request.prompt if request.prompt is not None else FLUX_PROMPT,
            "prompt_override": request.prompt is not None,
            # Доли маски — в мете всегда: с переносом причёски они перестают быть
            # умолчанием, и кадр без них не разобрать
            "dilate_ratio": dilate,
            "feather_ratio": feather,
            "neck_ratio": neck,
            "face_height": round(float(geometry["face_height"]), 1),
            "donor_face_px": round(float(head_mask.face_geometry(donor_points)["face_height"]), 1),
            "donor_outside": round(_donor_outside(photo, donor_points, modules), 3),
            "steps": steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "erased": plate is not None,
            "timings": timings,
        }
    )
    return pasted.image, meta


def _donor_crop(photo: np.ndarray, points: list, modules: dict[str, Any]) -> np.ndarray:
    """
    Выровненное лицо донора на квадратном холсте.

    Подобие берётся из живого кода экстракции (`AntelopeExtractor._matrix`) — то
    самое, что сажает глаза, нос и углы рта в канонические позиции, — и только
    пересчитывается под другой холст и более широкое поле зрения. Благодаря
    этому портрет по грудь и лицо крупным планом дают один и тот же кроп: именно
    инвариантность к кадрировке здесь и нужна, иначе результат зависел бы от
    того, как родитель держал телефон.
    """
    matrix = np.asarray(modules["extractor"]._matrix(points), dtype=np.float64)

    # Пересчёт под другой холст и более широкое поле: точка q в 112-координатах
    # переносится как (q - 56) * k + side/2. Формула и интерполяция взяты из
    # проверенного пути буква в букву — теми же кропами измерены 0.56 сходства,
    # и «улучшение» здесь означало бы, что число относится к чему-то другому
    side, margin = DEMO_DONOR_SIDE, DEMO_DONOR_MARGIN
    scale = side / (112.0 * margin)
    matrix[:, :2] *= scale
    matrix[:, 2] = scale * (matrix[:, 2] - 56.0) + side / 2.0

    return cv2.warpAffine(
        photo, matrix, (side, side),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _donor_outside(photo: np.ndarray, points: list, modules: dict[str, Any]) -> float:
    """
    Какая доля кропа НЕ пришла из фотографии, 0..1.

    Плотно кадрированный портрет при широком поле вылезает за край, и там
    `BORDER_REPLICATE` размазывает крайний пиксель полосами. Для генератора это
    не лицо, а шум. Число уходит в отчёт: заметное значение объясняет слабый
    результат лучше, чем любые догадки о модели.
    """
    matrix = np.asarray(modules["extractor"]._matrix(points), dtype=np.float64)
    side, margin = DEMO_DONOR_SIDE, DEMO_DONOR_MARGIN
    scale = side / (112.0 * margin)
    matrix[:, :2] *= scale
    matrix[:, 2] = scale * (matrix[:, 2] - 56.0) + side / 2.0

    inside = cv2.warpAffine(
        np.full(photo.shape[:2], 255, dtype=np.uint8), matrix, (side, side),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return 1.0 - float(np.count_nonzero(inside)) / float(side * side)


@app.post("/v1/demo-render", summary="Фото ребёнка + шаблон -> готовый разворот")
async def demo_endpoint(request: DemoRequest) -> dict[str, Any]:
    """
    Один запрос на весь путь. Ответ — `{"image": base64, "meta": {...}}`.

    Синхронный: показ идёт вживую, и просить заказчика опрашивать статус задачи
    ради тридцати секунд — плохой сценарий демонстрации.
    """
    if not DEMO_ENABLED:
        raise HTTPException(503, "демонстрационный путь выключен (GPU_DEMO=0)")
    if not MODELS.flux.ready:
        raise HTTPException(503, MODELS.flux.reason or "FLUX.2 не загружен")

    try:
        async with QUEUE.slot() as waited:
            started = time.perf_counter()
            image, meta = await asyncio.to_thread(run_demo, request)
            meta["queue_wait_s"] = round(waited, 3)
            meta["total_s"] = round(time.perf_counter() - started, 3)
    except QueueFull as exc:
        raise HTTPException(503, f"очередь переполнена: {exc}") from exc
    except QueueTimeout as exc:
        raise HTTPException(503, f"перегрузка: {exc}") from exc

    return {
        "image": encode_image(image, request.output_format),
        "encoding": f"image/{'jpeg' if request.output_format in {'jpg', 'jpeg'} else 'png'}",
        "meta": meta,
    }


_DEMO_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Книга с твоим лицом</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;
      background:#101418;color:#e8eaed;max-width:900px;margin:0 auto}
 h1{font-size:22px;margin:0 0 4px} p.sub{color:#9aa0a6;margin:0 0 24px;font-size:14px}
 .row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}
 label{flex:1;min-width:220px;background:#1b2026;border:1px solid #2c333b;border-radius:12px;
       padding:16px;cursor:pointer;display:block}
 label span{display:block;font-size:13px;color:#9aa0a6;margin-bottom:8px}
 input[type=file]{width:100%;color:#e8eaed;font-size:14px}
 button{width:100%;padding:16px;font-size:17px;border:0;border-radius:12px;
        background:#3b82f6;color:#fff;cursor:pointer}
 button:disabled{background:#2c333b;color:#6b7280;cursor:default}
 #out{margin-top:24px} img{width:100%;border-radius:12px;display:block}
 #status{margin-top:16px;color:#9aa0a6;font-size:14px;min-height:20px}
 .err{color:#f28b82}
</style></head><body>
<h1>Книга с твоим лицом</h1>
<p class="sub">Выберите фотографию ребёнка и разворот книги — остальное сделает сервер.</p>
<div class="row">
  <label><span>1. Фотография ребёнка</span><input type="file" id="photo" accept="image/*"></label>
  <label><span>2. Разворот книги</span><input type="file" id="template" accept="image/*"></label>
</div>
<button id="go" disabled>Сделать</button>
<div id="status"></div>
<div id="out"></div>
<script>
const photo=document.getElementById('photo'), template=document.getElementById('template'),
      go=document.getElementById('go'), status=document.getElementById('status'),
      out=document.getElementById('out');
function check(){ go.disabled=!(photo.files[0] && template.files[0]); }
photo.onchange=check; template.onchange=check;
function b64(file){ return new Promise((ok,bad)=>{ const r=new FileReader();
  r.onload=()=>ok(r.result.split(',')[1]); r.onerror=bad; r.readAsDataURL(file); }); }
go.onclick=async()=>{
  go.disabled=true; out.innerHTML=''; status.className='';
  const t0=Date.now(); let dots=0;
  const tick=setInterval(()=>{ dots=(dots+1)%4;
    status.textContent='Рисуем'+'.'.repeat(dots)+'  '+Math.round((Date.now()-t0)/1000)+' с'; },500);
  try{
    const r=await fetch('v1/demo-render',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({base_image:await b64(template.files[0]),
                           donor_photo:await b64(photo.files[0])})});
    clearInterval(tick);
    if(!r.ok){ const t=await r.text(); status.className='err';
      status.textContent='Не получилось: '+t.slice(0,300); go.disabled=false; return; }
    const d=await r.json();
    out.innerHTML='<img src="data:'+(d.encoding||'image/png')+';base64,'+d.image+'">';
    status.textContent='Готово за '+Math.round((Date.now()-t0)/1000)+' с';
  }catch(e){ clearInterval(tick); status.className='err';
    status.textContent='Ошибка сети: '+e; }
  go.disabled=false;
};
</script></body></html>"""


@app.get("/demo", summary="Страница показа: загрузить фото и получить разворот")
async def demo_page() -> Any:
    """
    Одностраничник для живого показа с телефона или ноутбука.

    Отдаётся самим сервисом, а не отдельным фронтендом: показывать надо сегодня,
    и заводить ради этого статику, сборку и второй адрес — значит потратить время
    на инфраструктуру вместо демонстрации. Ни одной внешней зависимости в
    странице нет, потому что сервер живёт оффлайн и CDN ему недоступен.

    Путь к API относительный (`v1/demo-render`), чтобы страница одинаково
    работала и через туннель, и по прямому адресу, и из-под обратного прокси.
    """
    from fastapi.responses import HTMLResponse

    return HTMLResponse(_DEMO_PAGE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=_env("GPU_HOST", "0.0.0.0"),
        port=_env_int("GPU_PORT", 8100),
        # Один воркер жёстко: веса занимают ~12 ГБ, и второй процесс не делит
        # их, а загружает свою копию. Параллелизм здесь даёт очередь, не форки
        workers=1,
        log_level=_env("GPU_LOG_LEVEL", "info").lower(),
    )
