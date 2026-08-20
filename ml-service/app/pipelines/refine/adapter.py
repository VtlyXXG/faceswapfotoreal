"""
Интеграция Flux + PuLID: перенос личности вместо фейссвопа.

Модуль готовит условия генерации, раскладывает нагрузку по двум пулам и
разговаривает с бэкендом через протокол `IdentityBackend`. Сам он ни к fal, ни к
Flux не ходит: транспорт живёт за протоколом, тип бэкенда решает конфигурация.

Ни один эндпоинт fal не принимает одновременно базовую картинку, маску и вектор
личности — проверено по openapi, см. `FalPuLIDBackend`. Бэкенд, умеющий (сцена +
маска + эмбеддинг), — тот, который разворачиваем мы, и модуль повёрнут к нему
contract-first.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeAlias

from app.core.errors import MLServiceError
from app.core.logging import get_logger
from app.pipelines.refine import payload

# `hashlib`, `cheeks`, `erase`, `hair_mask` и `composite` подтягиваются вместе с
# телами заглушек. Держать их импортированными в скелете нельзя: mediapipe и
# opencv тянутся при импорте модуля, а этот модуль обязан импортироваться
# дёшево — его читает конфигурация, чтобы узнать, какой бэкенд собран.
# `payload` этим не грешит: opencv он трогает лениво, внутри `encode_image`

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from app.pipelines.hair_mask import HairMask
    from app.pipelines.refine.profiles import RefineProfile

    # Кадр — BGR uint8 (H, W, 3); маска — uint8 (H, W); вектор — float32 (512,)
    Frame: TypeAlias = NDArray[np.uint8]
    Mask: TypeAlias = NDArray[np.uint8]
    Embedding: TypeAlias = NDArray[np.float32]
    Landmarks: TypeAlias = list[tuple[int, int]]
else:
    Frame = Mask = Embedding = Landmarks = Any
    HairMask = RefineProfile = Any

log = get_logger(__name__)

BoundBy: TypeAlias = Literal["geometry", "floor", "hair"]
Window: TypeAlias = "list[int] | None"


class IdentityBackendError(MLServiceError):
    """Бэкенд переноса личности отказал или не умеет того, что просят."""

    status_code = 502
    code = "IDENTITY_BACKEND_FAILED"


class SceneNotPreservableError(MLServiceError):
    """
    Бэкенд не умеет сохранять сцену. 500, а не 502: до сети такой запрос
    доходить не должен — разворот заказчика он не поправит, а перерисует.
    """

    status_code = 500
    code = "SCENE_NOT_PRESERVABLE"


# --- Константы экстракции ----------------------------------------------------

# Канонический шаблон ArcFace: пять опорных точек в кадре 112x112. Отсюда и
# берётся инвариантность вектора к разрешению — не из аккуратного ресайза, а из
# того, что подобие сажает глаза, нос и углы рта в ОДНИ И ТЕ ЖЕ пиксели при
# любом входе. Простой resize сохраняет кадрирование донора, и портрет по грудь
# даёт другой вектор, чем то же лицо крупным планом.
#
# Шаблон трогать нельзя: antelopev2 обучен на этой раскладке, и выравнивание по
# любой другой (в том числе «по челюсти») отдаёт валидный на вид вектор, который
# ни с чем не сходится. Челюсть поэтому идёт не в шаблон, а в проверку —
# `_jaw_consistency`.
_ARCFACE_TEMPLATE: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),  # глаз со стороны левого края кадра
    (73.5318, 51.5014),  # глаз со стороны правого края
    (56.0252, 71.7366),  # кончик носа
    (41.5493, 92.3655),  # угол рта слева
    (70.7299, 92.2041),  # угол рта справа
)
_ARCFACE_SIDE: int = 112

# Точки сетки под те же пять опор. Центр глаза — середина между его углами:
# отдельной точки центра в сетке нет, а радужка (468+) есть не всегда.
_MESH_EYE_A: tuple[int, int] = (33, 133)
_MESH_EYE_B: tuple[int, int] = (362, 263)
_MESH_NOSE: int = 1
_MESH_MOUTH: tuple[int, int] = (61, 291)

# Насколько выровненный кроп может расходиться с шаблоном по линии челюсти,
# доли стороны кропа. Проверка ловит то, чего не ловит подобие по пяти точкам:
# сильный разворот и кривую сетку, на которых глаза сядут в шаблон, а овал уедет.
_JAW_TOLERANCE: float = 0.12

# Доля кадра, ниже которой вектору не верят: на мелком лице экстрактор отдаёт
# «среднего человека». Порог тот же, что у детектора сетки.
_MIN_FACE_SHARE: float = 0.05

_EMBEDDING_SIZE: int = 512

# Сегментатор разметки — глобальный объект на процесс (`parsing._SEGMENTER`), и
# его граф tflite не реентерабелен. Два потока пула А, вызвавшие `segment()`
# одновременно, получают не исключение, а тихо испорченные маски.
#
# Следствие для планирования: препроцессинг сериализуется на разметке, поэтому
# расширять пул А бессмысленно — весь выигрыш конвейера в перекрытии с сетью.
_PARSING_LOCK = threading.Lock()


# --- 1. Конвейер экстракции идентичности -------------------------------------


@dataclass(frozen=True)
class IdentityVector:
    """
    Вектор личности донора и всё, что нужно, чтобы ему доверять.

    ПАМЯТЬ: две тысячи байт вектора плюс 37 КБ кропа против ~25 МБ исходного
    кадра. Считается один раз на ЗАКАЗ и переезжает между потоками по ссылке;
    `frozen` — чтобы поток А не мог дописать в общий объект, пока поток Б держит
    на него ссылку в теле запроса.
    """

    embedding: Embedding
    aligned: Frame
    digest: str
    quality: float
    jaw_error: float


class IdentityExtractor(Protocol):
    """Что угодно, что превращает фотографию донора в вектор личности."""

    def extract(self, image: Frame) -> IdentityVector: ...


@dataclass
class AntelopeExtractor:
    """
    antelopev2 (InsightFace): детектор scrfd + распознаватель glintr100, 512-d.

    Опоры берутся из сетки MediaPipe, уже построенной пайплайном, а не из
    детектора InsightFace: один источник геометрии на маску и на вектор.
    """

    model_root: str
    providers: Sequence[str] = ("CPUExecutionProvider",)
    recogniser: str = "glintr100.onnx"
    _session: Any = field(default=None, init=False, repr=False)
    _input: str = field(default="", init=False, repr=False)

    def load(self) -> None:
        """
        Поднимает сессию onnxruntime. Вызывать на старте, не по запросу.

        ПАМЯТЬ: веса antelopev2 — сотни мегабайт на процесс, и держать их надо
        ОДНОЙ сессией на все потоки пула А. Сессия onnxruntime потокобезопасна
        на `run()`, поэтому копия на поток здесь была бы чистой растратой.
        """
        from pathlib import Path

        if self._session is not None:
            return

        weights = Path(self.model_root) / self.recogniser
        if not weights.exists():
            raise IdentityBackendError(
                "Веса antelopev2 не найдены — вектор личности строить нечем",
                {
                    "expected": str(weights),
                    "hint": "распакуйте antelopev2 в ML_ANTELOPE_ROOT; "
                    "без него доступны шаги до экстракции (--until B)",
                },
            )

        try:
            import onnxruntime
        except ImportError as exc:
            raise IdentityBackendError(
                "onnxruntime не установлен: он снят с образа вместе с insightface",
                {"cause": str(exc), "hint": "pip install onnxruntime"},
            ) from exc

        self._session = onnxruntime.InferenceSession(str(weights), providers=list(self.providers))
        self._input = self._session.get_inputs()[0].name

    def warmup(self) -> None:
        """
        Один холостой прогон после загрузки.

        Первый `run()` у onnxruntime дороже последующих на порядок: там
        аллокация арен и выбор ядер. Без прогрева эта цена попала бы в тайминг
        первого кадра и читалась бы как «экстракция медленная».
        """
        import numpy as np

        self.load()
        self._session.run(None, {self._input: np.zeros((1, 3, 112, 112), dtype=np.float32)})

    def extract(self, image: Frame) -> IdentityVector:
        import numpy as np

        from app.pipelines import head_mask

        points = head_mask.try_landmarks(image)
        if points is None:
            raise IdentityBackendError(
                "На фотографии заказчика не найдено лицо — вектор личности не построить",
                {"hint": "нужен портрет, где лицо занимает больше пятой части кадра"},
            )

        geometry = head_mask.face_geometry(points)
        share = float(geometry["face_height"]) / max(1, min(image.shape[:2]))
        if share < _MIN_FACE_SHARE:
            log.warning(
                "лицо донора мелкое — вектор может выйти усреднённым",
                extra={"face_share": round(share, 3), "floor": _MIN_FACE_SHARE},
            )

        aligned = self.align(image, points)
        jaw_error = self._jaw_consistency(image, points)
        if jaw_error > _JAW_TOLERANCE:
            # Не отказ: портрет в три четверти — рабочий кадр, просто вектор с
            # него менее надёжен. Число уезжает наружу, чтобы «не похож» имел
            # объяснение раньше, чем начнут крутить fidelity
            log.warning(
                "лицо донора развёрнуто — овал не сел в шаблон выравнивания",
                extra={"jaw_error": round(jaw_error, 3), "tolerance": _JAW_TOLERANCE},
            )

        vector = self._infer(aligned)
        return IdentityVector(
            embedding=vector.astype(np.float32),
            aligned=aligned,
            digest=IdentityCache.key(image),
            quality=share,
            jaw_error=jaw_error,
        )

    @classmethod
    def align(cls, image: Frame, points: Landmarks) -> Frame:
        """
        Кроп 112x112, приведённый ПОДОБИЕМ к каноническим пяти точкам.

        Подобие, а не аффинное преобразование: у аффинного неравномерный
        масштаб, и оно исправило бы ракурс, растянув лицо. Снимаем разрешение и
        кадрирование, а не переписываем анатомию.
        """
        import cv2

        matrix = cls._matrix(points)
        return cv2.warpAffine(image, matrix, (_ARCFACE_SIDE, _ARCFACE_SIDE), flags=cv2.INTER_LINEAR)

    @classmethod
    def _matrix(cls, points: Landmarks) -> Frame:
        import cv2
        import numpy as np

        matrix, _ = cv2.estimateAffinePartial2D(
            cls._five_points(points),
            np.array(_ARCFACE_TEMPLATE, dtype=np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            raise IdentityBackendError("Опорные точки лица вырождены — подобие не строится")
        return matrix

    @staticmethod
    def _five_points(points: Landmarks) -> Frame:
        """
        Пять опор из сетки, упорядоченных ПО ПОЛОЖЕНИЮ В КАДРЕ.

        Нумерация сетки субъектная: у отзеркаленного донора «левый» глаз сетки
        окажется справа. Без сортировки по x зеркальный кадр даёт чужой вектор,
        и поймать это можно только сравнением двух фотографий одного человека.
        """
        import numpy as np

        def centre(pair: tuple[int, int]) -> Any:
            first, second = (np.asarray(points[index], dtype=np.float64) for index in pair)
            return (first + second) / 2.0

        eyes = sorted((centre(_MESH_EYE_A), centre(_MESH_EYE_B)), key=lambda p: p[0])
        mouth = sorted(
            (np.asarray(points[index], dtype=np.float64) for index in _MESH_MOUTH),
            key=lambda p: p[0],
        )
        nose = np.asarray(points[_MESH_NOSE], dtype=np.float64)

        return np.array([eyes[0], eyes[1], nose, mouth[0], mouth[1]], dtype=np.float32)

    @classmethod
    def _jaw_consistency(cls, image: Frame, points: Landmarks) -> float:
        """
        Насколько овал разошёлся с шаблоном после выравнивания, доли стороны.

        Выравнивание по глазам и рту ставит горизонт, но овал не проверяет: на
        развороте в три четверти пять опорных точек сядут в шаблон, а челюсть
        уедет вбок — вектор соберётся с половины лица. Здесь та же матрица
        применяется к паре точек челюсти, и меряется два расхождения: смещение
        середины подбородка от оси кропа и наклон хорды.

        Пара ЗЕРКАЛЬНАЯ (`head_mask._JAW_PAIR`): 150 и 377 зеркальными не
        являются и дают систематический перекос в 18° на ровном месте.
        """
        import numpy as np

        from app.pipelines import head_mask

        first, second = head_mask._JAW_PAIR
        matrix = cls._matrix(points)
        jaw = np.array([points[first], points[second]], dtype=np.float64)
        warped = jaw @ matrix[:, :2].T + matrix[:, 2]

        middle = float(warped[:, 0].mean())
        offset = abs(middle - _ARCFACE_SIDE / 2.0) / _ARCFACE_SIDE

        chord = warped[1] - warped[0]
        tilt = abs(float(chord[1])) / max(1e-6, abs(float(chord[0])))

        return max(offset, tilt)

    def _infer(self, aligned: Frame) -> Embedding:
        """
        Прогон кропа. BGR, (x - 127.5) / 127.5, NCHW float32, выход 512-d.

        Порядок каналов и нормировка — часть контракта весов: поданный в RGB
        кроп даёт валидный на вид вектор, который ни с чем не сходится, и
        заметить это можно только сравнением двух фотографий одного человека.
        """
        import numpy as np

        self.load()

        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        vector = np.asarray(self._session.run(None, {self._input: blob})[0]).ravel()
        if vector.size != _EMBEDDING_SIZE:
            raise IdentityBackendError(
                "Распознаватель отдал вектор не той размерности",
                {"size": int(vector.size), "expected": _EMBEDDING_SIZE},
            )

        # L2 обязательна: косинусная близость, на которой стоит PuLID,
        # определена на единичной сфере, а ненормированный вектор тянет за собой
        # яркость кропа и делает силу влияния зависимой от экспозиции донора
        return vector / max(1e-6, float(np.linalg.norm(vector)))


class IdentityCache:
    """
    Вектор на ЗАКАЗ, а не на разворот: фотография одна, разворотов десяток.

    ПАМЯТЬ: хранится вектор и кроп, ~39 КБ на заказ; словарь ограничен на
    случай долгоживущего процесса. Экстракция идёт ВНЕ замка — она секундная, и
    держать на ней потоки пула А незачем; гонка двух одинаковых ключей стоит
    одной лишней экстракции.
    """

    def __init__(self, extractor: IdentityExtractor, limit: int = 32) -> None:
        self._extractor = extractor
        self._limit = limit
        self._lock = threading.Lock()
        self._store: dict[str, IdentityVector] = {}

    def get(self, key: str, image: Frame) -> IdentityVector:
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            return cached

        # Экстракция ВНЕ замка: она секундная, и держать на ней остальные потоки
        # незачем. Гонка двух одинаковых ключей стоит одной лишней экстракции —
        # дешевле, чем сериализовать всех
        vector = self._extractor.extract(image)

        with self._lock:
            if len(self._store) >= self._limit:
                self._store.pop(next(iter(self._store)))
            self._store[key] = vector
        return vector

    @staticmethod
    def key(image: Frame) -> str:
        """Отпечаток буфера кадра. Хеш по байтам, а не по пути: файла у нас нет."""
        import hashlib

        import numpy as np

        return hashlib.sha256(np.ascontiguousarray(image)).hexdigest()[:16]


# --- 3. Вектор инъекции ------------------------------------------------------


@dataclass(frozen=True)
class PuLIDWeights:
    """
    Баланс «похож на заказчика» против «вписан в свет сцены».

    :param fidelity: сила влияния личности, 0.85..1.0. Нижняя граница жёсткая:
        ниже неё донор теряет портретное сходство под агрессивным освещением
        шаблона — сцена перебивает личность, и возвращается ровно та беда, из-за
        которой ушли с фейссвопа
    :param start_step: с какого шага денойзинга включается личность. Ноль задаёт
        личностью саму композицию; поздний старт отдаёт первым шагам свет и позу
        сцены — главная ручка против пластиковой кожи
    :param steps: шагов денойзинга. Ниже двадцати микроконтраст не успевает
        проявиться
    :param guidance: сила текста. Высокая пережигает поры в пластик
    :param true_cfg: 1.0 отключает второй проход; больше единицы включает
        честный CFG с негативным промптом и удваивает время
    """

    # ClassVar, а не поля: без этого датакласс сделал бы пределы аргументами
    # конструктора, и «нижнюю границу» можно было бы передать любую
    MIN_FIDELITY: ClassVar[float] = 0.85
    MAX_FIDELITY: ClassVar[float] = 0.90
    MAX_STEPS: ClassVar[int] = 50

    fidelity: float = 0.88
    start_step: int = 2
    steps: int = 28
    guidance: float = 3.5
    true_cfg: float = 1.0

    def validate(self) -> None:
        if not self.MIN_FIDELITY <= self.fidelity <= self.MAX_FIDELITY:
            raise IdentityBackendError(
                "Сила влияния личности вне рабочего окна",
                {
                    "fidelity": self.fidelity,
                    "window": [self.MIN_FIDELITY, self.MAX_FIDELITY],
                    "hint": "ниже окна донор теряет сходство под светом сцены, "
                    "выше — личность перебивает сцену и вклейка читается наклейкой",
                },
            )
        if not 1 <= self.steps <= self.MAX_STEPS:
            raise IdentityBackendError(
                "Шагов денойзинга вне диапазона", {"steps": self.steps, "max": self.MAX_STEPS}
            )
        if not 0 <= self.start_step <= self.steps:
            raise IdentityBackendError(
                "Личность включается позже последнего шага",
                {"start_step": self.start_step, "steps": self.steps},
            )


@dataclass
class SceneConditioning:
    """
    Полезная нагрузка на ОДИН кадр. Собирается потоком А, читается потоком Б.

    :param scene: окно вокруг головы из шаблона, НЕТРОНУТОЕ. В него потом
        вклеивается ответ, и подменять его подготовленной копией нельзя
    :param sent: очищенный шаблон — результат первого шага (`cheeks.flatten` и
        выжигание теней). Уезжает бэкенду; в готовый разворот не попадает
    :param mask: маска головы той же геометрии
    :param window: [left, top, width, height] окна в развороте либо None

    ПАМЯТЬ, узкое место номер один. Объект пересекает границу потоков, и с
    момента передачи он ЧУЖОЙ: поток А не имеет права дописывать в эти массивы,
    готовя кадр N+1. numpy между потоками не копирует (в отличие от процессов,
    где сработал бы pickle) — защиты нет никакой, только дисциплина владения.
    Отсюда `frozen` у соседних структур и запрет на in-place ниже по течению.

    ПАМЯТЬ, узкое место номер два. Разворот 4K в BGR — около 25 МБ, маска ещё 8,
    и на кадр их держится сразу несколько: окно, очищенная копия, ответ бэкенда,
    результат вклейки. Под сотню мегабайт на кадр в полёте — отсюда глубина
    конвейера единица и `release_sent()` сразу после загрузки.
    """

    scene: Frame
    sent: Frame | None
    mask: Mask
    window: Window
    prompt: str
    identity: IdentityVector
    weights: PuLIDWeights
    meta: dict[str, Any] = field(default_factory=dict)

    def release_sent(self) -> None:
        """Отпускает очищенную копию: байты уже уехали, массив держится зря."""
        self.sent = None


def hair_prompt(template: str, colour: str, description: str = "") -> str:
    """
    Промпт с параметризованным цветом волос.

    Цвет подставляется В ШАБЛОН, а не приклеивается в конец: «blonde» в хвосте
    фразы модель относит к чему придётся, вплоть до фона. Пустой цвет убирает
    упоминание цвета целиком — просить покрасить в «» значит просить случайный.
    """
    head = template.format(colour=colour.strip()) if "{colour}" in template else template
    parts = [" ".join(head.split()), description.strip()]
    return " ".join(part for part in parts if part).strip()


# --- Бэкенды -----------------------------------------------------------------


class IdentityBackend(Protocol):
    """
    Кто именно рисует. Асинхронный: ожидание здесь сетевое, не счётное.

    `preserves_scene` отделяет то, что можно поставить в конвейер, от того, что
    рисует кадр с нуля. Проверяется ДО загрузки в CDN — иначе за отказ платим
    трафиком и инференсом.
    """

    preserves_scene: bool

    async def render(self, conditioning: SceneConditioning) -> bytes: ...


@dataclass
class FalPuLIDBackend:
    """
    `fal-ai/flux-pulid`. Сцену не сохраняет, и это свойство схемы, а не настроек.

    По openapi: `prompt`, `reference_image_url`, `image_size`, `id_weight`,
    `num_inference_steps`, `guidance_scale`, `start_step`, `true_cfg`,
    `negative_prompt`, `seed`. Ни базовой картинки, ни `mask_url`, ни поля под
    готовый эмбеддинг — картинка на входе ровно одна. «Сохранить разворот и
    внедрить личность» здесь не выражается ни при каком балансе весов: не
    хватает не силы влияния, а входа.

    Класс существует как явный отказ: без него первая же попытка подставить
    эндпоинт в конфигурацию кончилась бы перерисованным разворотом.
    """

    endpoint: str = "fal-ai/flux-pulid"
    preserves_scene: bool = False

    async def render(self, conditioning: SceneConditioning) -> bytes:
        raise SceneNotPreservableError(
            "flux-pulid генерирует кадр с нуля: шаблон-разворот он не сохранит",
            {
                "endpoint": self.endpoint,
                "accepts": ["prompt", "reference_image_url", "image_size"],
                "missing": ["image_url", "mask_url", "identity_embedding"],
                "hint": "нужен бэкенд, принимающий сцену, маску и вектор",
            },
        )


# Коды, после которых имеет смысл повторить. 502/504 — отбой прокси перед GPU,
# 503 — сервис поднимается, 429 — очередь занята. Всё остальное из 4xx
# повторять нельзя: это наш неверный запрос, и три попытки сделают из одной
# ошибки три и втрое больше трафика.
_RETRY_STATUS: frozenset[int] = frozenset({429, 502, 503, 504})


@dataclass
class HostedFluxPuLIDBackend:
    """
    Собственный Flux + PuLID. Контракт сервиса:

        POST {base_url}/render   (JSON, см. `payload.build_packet`)
          base_image  — очищенный шаблон, base64 JPEG
          mask_image  — маска инпейнта, base64 PNG
          donor_crop  — лицо 112x112, base64 PNG
          embedding   — 512 float32 '<f4', base64
          prompt, negative_prompt
          fidelity, steps, start_step, guidance_scale, true_cfg
        → {"image": base64} либо image/* телом

    Эмбеддинг, а не фотография: фотография уехала бы в сеть на каждый разворот и
    на каждом заново прошла бы детектор. Рядом едет `donor_crop` — официальный
    PuLID берёт не только ArcFace-вектор, но и признаки EVA-CLIP с того же
    кропа, и сервису, которому мало вектора, не придётся запрашивать исходник.

    **Почему httpx.AsyncClient, а не блокирующий клиент в потоке.** Требование
    «зависший GPU не должен повесить пул» блокирующим клиентом НЕ выполняется:
    `asyncio.wait_for` вокруг `run_in_executor` отменяет ожидание, но не поток —
    тот остаётся висеть на `recv()` до собственного таймаута сокета, держа и
    слот в пуле, и весь кадр в памяти. Отменяемым ожидание становится только
    тогда, когда сокет принадлежит событийному циклу, то есть у асинхронного
    клиента: отмена закрывает соединение и освобождает всё немедленно.

    **Два таймаута, а не один.** `attempt_timeout` — на попытку,
    `total_timeout` — на кадр целиком. Без второго три попытки по три минуты
    дали бы девять минут ожидания там, где заказ давно пора отклонить.
    """

    base_url: str
    attempt_timeout: float = 180.0
    total_timeout: float = 420.0
    max_attempts: int = 3
    backoff_base: float = 1.5
    backoff_cap: float = 20.0
    connect_timeout: float = 10.0
    quality: int = payload.JPEG_QUALITY
    negative_prompt: str = ""
    # Принудительная сборка мусора после освобождения кадра. По умолчанию
    # выключена намеренно, см. `_release`
    collect_garbage: bool = False
    preserves_scene: bool = True
    # Подменный транспорт httpx. Существует ради тестов: отбои 502, зависание
    # GPU и отмену посреди запроса иначе не проверить ничем, кроме живого
    # сервиса, которого пока нет
    transport: Any = None

    _client: Any = field(default=None, init=False, repr=False)
    _lock: Any = field(default=None, init=False, repr=False)

    async def client(self) -> Any:
        """
        Один клиент на бэкенд: пул соединений и повторно используемый TLS.

        Клиент на запрос стоил бы рукопожатия на каждый кадр и, что хуже,
        оставлял бы дескрипторы открытыми до сборки мусора.
        """
        import httpx

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.attempt_timeout, connect=self.connect_timeout),
                    # Ограничение снизу: конвейер шлёт по одному кадру, но
                    # заказов в воркере может идти несколько
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                    transport=self.transport,
                )
        return self._client

    async def render(self, conditioning: SceneConditioning) -> bytes:
        """
        Один кадр: пакет → сеть → байты картинки.

        Очищенный шаблон отпускается в `finally` при любом исходе, включая
        отмену: держать его до сборки мусора значит держать десятки мегабайт на
        кадр, который уже никому не нужен.
        """
        conditioning.weights.validate()

        packet: dict[str, Any] | None = None
        try:
            packet = payload.build_packet(
                conditioning, quality=self.quality, negative_prompt=self.negative_prompt
            )
            log.info(
                "кадр уезжает в собственный Flux",
                extra={
                    "bytes": payload.packet_weight(packet),
                    "fidelity": conditioning.weights.fidelity,
                    "steps": conditioning.weights.steps,
                },
            )
            return await self._post(packet)
        finally:
            # Порядок важен: сначала пакет (строки base64, до трети от веса
            # массивов), потом сам очищенный кадр
            packet = None
            self._release(conditioning)

    async def _post(self, packet: dict[str, Any]) -> bytes:
        """
        Попытки с экспоненциальной задержкой в пределах общего срока.

        Срок считается один раз на весь вызов: каждая попытка получает то, что
        от него осталось, а не полный `attempt_timeout`. Иначе последняя попытка
        имела бы право пережить собственный дедлайн.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.total_timeout
        client = await self.client()
        last: dict[str, Any] = {}

        for attempt in range(1, self.max_attempts + 1):
            left = deadline - loop.time()
            if left <= 0:
                break

            try:
                async with asyncio.timeout(min(self.attempt_timeout, left)):
                    response = await client.post("/render", json=packet)
            except asyncio.CancelledError:
                # Отмена — не отказ сети. Пробрасывается немедленно и без
                # повторов: наверху либо снят заказ, либо упал весь конвейер
                raise
            except TimeoutError:
                last = {"cause": "timeout", "attempt_timeout": self.attempt_timeout}
            except Exception as exc:  # noqa: BLE001 — транспорт httpx, разбираем ниже
                if not self._is_transport_error(exc):
                    raise IdentityBackendError(
                        "Собственный Flux не ответил", {"cause": str(exc)}
                    ) from exc
                last = {"cause": type(exc).__name__, "detail": str(exc)}
            else:
                if response.status_code < 400:
                    return self._image(response)

                last = {"status": response.status_code, "body": response.text[:200]}
                if response.status_code not in _RETRY_STATUS:
                    # Наш запрос неверен: повтор даст ту же ошибку втрое дороже
                    raise IdentityBackendError("Собственный Flux отверг запрос", last)

            delay = self._backoff(attempt)
            if attempt == self.max_attempts or loop.time() + delay >= deadline:
                break
            log.warning("повтор запроса к собственному Flux", extra={"attempt": attempt, **last})
            await asyncio.sleep(delay)

        raise IdentityBackendError(
            "Собственный Flux не ответил за отведённое время",
            {"attempts": self.max_attempts, "total_timeout": self.total_timeout, **last},
        )

    def _backoff(self, attempt: int) -> float:
        """
        Экспоненциальная задержка с полным джиттером.

        Джиттер не украшение: конвейер шлёт кадры пачкой, и без него все
        повторы после отбоя GPU придут в одну и ту же миллисекунду и положат
        его повторно.
        """
        import random

        window = min(self.backoff_cap, self.backoff_base * (2 ** (attempt - 1)))
        return random.uniform(window / 2.0, window)

    @staticmethod
    def _is_transport_error(exc: BaseException) -> bool:
        import httpx

        return isinstance(exc, httpx.TransportError)

    @staticmethod
    def _image(response: Any) -> bytes:
        """Картинка из ответа: сырым телом либо base64 в JSON."""
        import base64

        if response.headers.get("content-type", "").startswith("image/"):
            return response.content

        body = response.json()
        encoded = body.get("image") or body.get("images", [{}])[0].get("data")
        if not encoded:
            raise IdentityBackendError(
                "В ответе собственного Flux нет картинки", {"keys": sorted(body)}
            )
        return base64.b64decode(encoded)

    def _release(self, conditioning: SceneConditioning) -> None:
        """
        Освобождает память кадра немедленно, а не когда дойдут руки.

        Основной механизм здесь — СБРОС ССЫЛОК, а не сборщик мусора. Массив
        numpy освобождается по счётчику ссылок в тот момент, когда исчезает
        последняя ссылка; сборщик занимается только циклами и до массивов,
        честно отпущенных, отношения не имеет.

        Где сборщик всё же нужен: трассировка исключения держит кадры стека, а
        те — локальные переменные с массивами, и такая цепочка бывает цикличной.
        Поэтому вызов оставлен ручкой `collect_garbage`, а не поведением по
        умолчанию: `gc.collect()` обходит всю кучу, и в конвейере, где кадры
        идут один за другим, он сам становится расходом.
        """
        conditioning.release_sent()
        if self.collect_garbage:
            import gc

            gc.collect()

    async def aclose(self) -> None:
        """Закрывает клиента. Незакрытый держит сокеты до конца процесса."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HostedFluxPuLIDBackend:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


# --- 2. Адаптер мультипоточности ---------------------------------------------


@dataclass
class Order:
    """Заказ: одна фотография донора и несколько разворотов."""

    donor: Frame
    frames: Sequence[Frame]
    hair_colour: str = ""
    hair_description: str = ""
    prompt_template: str = ""
    weights: PuLIDWeights = field(default_factory=PuLIDWeights)


@dataclass(frozen=True)
class RenderedFrame:
    index: int
    image: Frame
    meta: dict[str, Any]


class FramePipeline:
    """
    Два пула с жёстким разделением ролей.

    Пул А (`_cpu`) — счётный: MediaPipe, маски, выжигание теней в `cheeks`.
    Пул Б (`_io`) — сетевой: сессия к диффузии. Обёртка синхронного транспорта
    в `run_in_executor` нужна затем, что маршрут FastAPI асинхронный, а
    `fal_client.subscribe` блокирующий: без выноса он держал бы весь событийный
    цикл на всё время инференса.

    Пулы РАЗДЕЛЕНЫ намеренно. Общий пул позволил бы счётной задаче занять
    последний свободный поток и оставить сетевой ответ без обработчика — и
    наоборот; при глубине конвейера в один кадр это тупик на ровном месте.

    ПАМЯТЬ: глубина ровно один кадр. Считать на два вперёд не ускоряет ничего
    (сеть — секунды, препроцессинг — сотни миллисекунд), а держит в куче ещё
    сотню мегабайт на кадр. Потоки, а не процессы: cv2 и mediapipe отпускают GIL
    внутри своих вызовов, а процессы гоняли бы массивы через pickle — 25 МБ туда
    и столько же обратно, дороже самой работы.
    """

    def __init__(
        self,
        backend: IdentityBackend,
        identity: IdentityCache,
        profile: RefineProfile,
        *,
        cpu_workers: int = 2,
        io_workers: int = 2,
    ) -> None:
        if not backend.preserves_scene:
            raise SceneNotPreservableError(
                "Бэкенд не сохраняет сцену — в конвейер разворотов он не годится",
                {"backend": type(backend).__name__},
            )

        self._backend = backend
        self._identity = identity
        self._profile = profile
        # Явные пределы вместо умолчания ThreadPoolExecutor: по умолчанию он
        # берёт min(32, ядра + 4), и на восьмиядерной машине это дюжина потоков,
        # каждый со своим разворотом в памяти
        self._cpu = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="cpu")
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="io")

    async def run(self, order: Order) -> list[RenderedFrame]:
        """
        Прогон заказа. Кадры возвращаются в порядке разворотов.

        Порядок внутри цикла и есть всё перекрытие: препроцессинг кадра N+1
        запускается ДО того, как кадр N уйдёт в сеть.

            vector = await A(extract)              # один раз на заказ
            pending = A(preprocess, 0)
            for N:
                conditioning = await pending
                pending = A(preprocess, N + 1)     # пул А занят кадром N+1...
                image = await B(render, N)         # ...пока пул Б держит сессию N
        """
        raise NotImplementedError("цикл перекрытия по схеме выше")

    def _preprocess(self, order: Order, vector: IdentityVector, index: int) -> SceneConditioning:
        """
        Пул А. Маски, полоса щеки, окно, очистка шаблона, промпт.

        Всё локальное собрано здесь именно затем, чтобы целиком уехать с
        главного потока: `hair_mask.build` поднимает сетку и разметку,
        `cheeks.flatten` гоняет свёртки по окну — на 4K это десятки миллисекунд
        каждая, и в событийном цикле они блокировали бы весь сервис.

            with _PARSING_LOCK:
                hair = hair_mask.build(frame, **stage_ratios)
            flat, cheek_meta = cheeks.flatten(frame, hair, wipe_ratio, flat_ratio)
            scene, mask = cut(frame, hair.mask, window)
            sent, _ = cut(flat, hair.mask, window)
            sent, erase_meta = erase.wipe(sent, mask, hair.face_height, erase_ratio)
        """
        raise NotImplementedError("см. hair_swap.HairThenFaceSwapRefiner.refine")

    async def _render(self, conditioning: SceneConditioning) -> bytes:
        """Пул Б. Загрузка в CDN, инференс, скачивание; `release_sent` после."""
        raise NotImplementedError("await self._backend.render + release_sent")

    def _composite(self, conditioning: SceneConditioning, generated: bytes) -> Frame:
        """
        Вклейка ответа в НЕТРОНУТУЮ сцену: `composite.paste(..., match=True)`.

        Единственное место, где результат бэкенда встречается с исходным
        разворотом. Правило прежнее: вне маски пиксели шаблона возвращаются
        побитово.
        """
        raise NotImplementedError("composite.paste + _restore, как в hair_swap")

    @staticmethod
    def _window(shape: tuple[int, int], hair: HairMask, crop_ratio: float) -> Window:
        raise NotImplementedError("общее с hair_swap._window")

    @staticmethod
    def _cut(image: Frame, mask: Mask, window: Window) -> tuple[Frame, Mask]:
        raise NotImplementedError("общее с hair_swap._cut")

    def close(self) -> None:
        """Оба пула. Незакрытый executor держит потоки живыми до конца процесса."""
        self._cpu.shutdown(wait=True)
        self._io.shutdown(wait=True)

    async def __aenter__(self) -> FramePipeline:
        return self

    async def __aexit__(self, *_: object) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.close)


__all__ = [
    "AntelopeExtractor",
    "FalPuLIDBackend",
    "FramePipeline",
    "HostedFluxPuLIDBackend",
    "IdentityBackend",
    "IdentityBackendError",
    "IdentityCache",
    "IdentityExtractor",
    "IdentityVector",
    "Order",
    "PuLIDWeights",
    "RenderedFrame",
    "SceneConditioning",
    "SceneNotPreservableError",
    "hair_prompt",
]
