"""
Транспорт до собственного Flux + PuLID и сериализация пакета.

Сети здесь нет: httpx подменяется `MockTransport`, и проверяется то, что живым
сервисом проверить дороже всего, — поведение на отбоях, на зависшем GPU и на
отмене посреди запроса.

Отдельно проверяется формат на поле. Он выбран замером, а не единообразием:
базовый кадр уезжает JPEG-ом ради канала, маска и кроп лица — PNG, потому что
маска это веса, а кроп это вход лицевого энкодера. Тесты закрепляют оба выбора,
чтобы «привести к одному формату» не стало однажды очевидным улучшением.
"""

import asyncio
import base64
from collections.abc import Awaitable
from typing import Any

import httpx
import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.pipelines.refine import adapter, payload

_SIDE = 64


def _conditioning(fidelity: float = 0.88) -> adapter.SceneConditioning:
    scene = np.full((_SIDE, _SIDE, 3), 180, dtype=np.uint8)
    mask = np.zeros((_SIDE, _SIDE), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    vector = np.zeros(payload.EMBEDDING_SIZE, dtype=np.float32)
    vector[0] = 1.0  # уже нормирован

    identity = adapter.IdentityVector(
        embedding=vector,
        aligned=np.full((112, 112, 3), 120, dtype=np.uint8),
        digest="deadbeef",
        quality=0.4,
        jaw_error=0.01,
    )
    return adapter.SceneConditioning(
        scene=scene,
        sent=scene.copy(),
        mask=mask,
        window=None,
        prompt="short ash blonde hair",
        identity=identity,
        weights=adapter.PuLIDWeights(fidelity=fidelity),
    )


def _png(side: int = _SIDE) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", np.zeros((side, side, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def _backend(handler, **overrides) -> adapter.HostedFluxPuLIDBackend:
    return adapter.HostedFluxPuLIDBackend(
        base_url="http://flux.local",
        transport=httpx.MockTransport(handler),
        **overrides,
    )


def _run(coroutine: Awaitable[Any]) -> Any:
    """
    Событийный цикл на тест. Через `asyncio.run`, а не через pytest-asyncio:
    зависимость ради синтаксиса тянуть незачем, а цикл здесь нужен настоящий —
    отмена и таймауты на подделке не проверяются.
    """
    return asyncio.run(coroutine)


# --- Сериализация ------------------------------------------------------------


def test_the_mask_is_never_jpeg():
    """
    Маска — не картинка, а веса: по ней решается, какие пиксели переписать, а
    какие вернуть из шаблона побитово. JPEG раскладывает её по блокам 8x8, и на
    замере живой маски q95 дал 287 пикселей ненулевого веса там, где маска была
    чистым нулём. Экономия при этом 59 КБ против 210 КБ базового кадра.
    """
    packet = payload.build_packet(_conditioning())

    assert packet["encoding"]["mask_image"] == "image/png"
    assert packet["encoding"]["donor_crop"] == "image/png", "кроп лица — вход энкодера"
    assert packet["encoding"]["base_image"] == "image/jpeg", "кадр обязан ехать сжатым"


def test_the_mask_survives_the_round_trip_bit_for_bit():
    """
    Главное следствие выбора формата, и проверяется оно не форматом, а
    пикселями: маска, вернувшаяся из пакета, обязана совпасть с исходной
    ПОБИТОВО. Ноль, ставший единицей, — это подмешанный шаблон вне маски.
    """
    import cv2

    conditioning = _conditioning()
    packet = payload.build_packet(conditioning)

    raw = np.frombuffer(base64.b64decode(packet["mask_image"]), dtype=np.uint8)
    restored = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)

    assert np.array_equal(restored, conditioning.mask)


def test_the_base_image_is_actually_smaller():
    """Ради чего JPEG и заводился: пакет обязан быть легче исходных пикселей."""
    conditioning = _conditioning()

    packet = payload.build_packet(conditioning)

    assert packet["bytes"]["base_image"] < conditioning.scene.nbytes


def test_the_embedding_keeps_its_bytes_and_byte_order():
    """
    Порядок байт задан явно: numpy у нас и torch у сервиса договориться сами не
    могут, а '<f4' читается одинаково везде.
    """
    conditioning = _conditioning()

    field = payload.encode_embedding(conditioning.identity.embedding)
    restored = np.frombuffer(base64.b64decode(field["data"]), dtype=np.dtype(field["dtype"]))

    assert field["dtype"] == "<f4" and field["size"] == payload.EMBEDDING_SIZE
    assert np.array_equal(restored, conditioning.identity.embedding)


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros(256, dtype=np.float32),  # не та длина
        np.full(payload.EMBEDDING_SIZE, 0.5, dtype=np.float32),  # не нормирован
    ],
)
def test_a_broken_embedding_never_reaches_the_network(vector):
    """
    Вектор не той формы или ненормированный — дефект нашей экстракции. Ловится
    до сети: ненормированный ломает косинусную близость, на которой стоит PuLID,
    и сила влияния начинает зависеть от экспозиции донора.
    """
    with pytest.raises(InvalidImageError):
        payload.encode_embedding(vector)


def test_the_packet_is_not_built_from_a_released_frame():
    """
    `release_sent` отпускает очищенный кадр ради памяти. Собрать после этого
    пакет нельзя, и молчать об этом — значит отправить пустоту.
    """
    conditioning = _conditioning()
    conditioning.release_sent()

    with pytest.raises(InvalidImageError):
        payload.build_packet(conditioning)


@pytest.mark.parametrize("fidelity", [0.5, 0.84, 0.91, 1.0])
def test_fidelity_outside_the_window_is_refused(fidelity):
    """
    Окно 0.85..0.90 жёсткое с обеих сторон: ниже донор теряет сходство под
    светом сцены, выше личность перебивает сцену и вклейка читается наклейкой.
    """
    with pytest.raises(adapter.IdentityBackendError):
        adapter.PuLIDWeights(fidelity=fidelity).validate()


# --- Транспорт ---------------------------------------------------------------


def test_a_transient_refusal_is_retried():
    """502 — отбой прокси перед GPU, а не отказ. Три попытки, победа на третьей."""
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        if len(seen) < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    async def scenario() -> bytes:
        async with _backend(handler, backoff_base=0.01, backoff_cap=0.02) as backend:
            return await backend.render(_conditioning())

    image = _run(scenario())

    assert len(seen) == 3
    assert image[:4] == b"\x89PNG"


@pytest.mark.parametrize("status", [400, 413, 422])
def test_our_own_bad_request_is_not_retried(status):
    """
    4xx — это наш неверный запрос. Три попытки сделали бы из одной ошибки три и
    втрое больше трафика, а ответ остался бы тем же.
    """
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(status, text="nope")

    async def scenario() -> None:
        async with _backend(handler, backoff_base=0.01) as backend:
            await backend.render(_conditioning())

    with pytest.raises(adapter.IdentityBackendError):
        _run(scenario())

    assert len(seen) == 1, "неверный запрос повторять нечего"


def test_the_attempts_stop_at_the_total_deadline():
    """
    Второй таймаут существует ради этого: без него три попытки по три минуты
    дали бы девять минут ожидания там, где заказ давно пора отклонить.
    """
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(504, text="gateway timeout")

    # Задержка заведомо длиннее остатка срока: попытка первая же и последняя,
    # хотя лимит попыток разрешает три. Числа подобраны так, чтобы условие
    # выполнялось при ЛЮБОМ джиттере, — плавающий тест хуже отсутствующего
    async def scenario() -> None:
        async with _backend(
            handler, max_attempts=3, backoff_base=1.0, backoff_cap=1.0, total_timeout=0.3
        ) as backend:
            await backend.render(_conditioning())

    with pytest.raises(adapter.IdentityBackendError) as exc_info:
        _run(scenario())

    assert exc_info.value.details["total_timeout"] == 0.3
    assert len(seen) == 1, "повторы не остановлены сроком кадра"


def test_a_hung_gpu_does_not_hold_the_frame_forever():
    """
    Зависший удалённый GPU. Ожидание обязано оборваться по `attempt_timeout`, а
    не висеть до таймаута сокета: именно ради этого транспорт асинхронный, а не
    блокирующий в потоке — отмена там закрывает соединение, а не ждёт его.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, content=_png())

    async def scenario() -> None:
        async with _backend(
            handler, attempt_timeout=0.05, total_timeout=0.25, backoff_base=0.01, backoff_cap=0.01
        ) as backend:
            await backend.render(_conditioning())

    with pytest.raises(adapter.IdentityBackendError) as exc_info:
        _run(scenario())

    assert exc_info.value.details["cause"] == "timeout"


def test_the_frame_is_released_on_every_outcome():
    """
    Очищенный кадр — десятки мегабайт на 4K. Отпускается он в `finally`, то есть
    и на успехе, и на отказе: конвейер идёт кадр за кадром, и кадр, переживший
    свой запрос, копится.
    """
    good, bad = _conditioning(), _conditioning()

    async def succeed() -> None:
        handler = lambda request: httpx.Response(  # noqa: E731
            200, content=_png(), headers={"content-type": "image/png"}
        )
        async with _backend(handler) as backend:
            await backend.render(good)

    async def fail() -> None:
        async with _backend(lambda request: httpx.Response(400), backoff_base=0.01) as backend:
            await backend.render(bad)

    _run(succeed())
    assert good.sent is None, "кадр пережил успешный запрос"

    with pytest.raises(adapter.IdentityBackendError):
        _run(fail())
    assert bad.sent is None, "кадр пережил отказ"


def test_cancellation_propagates_and_frees_the_frame():
    """
    Отмена — не отказ сети: повторять нечего, наверху либо снят заказ, либо упал
    конвейер. Пробрасывается немедленно, но кадр отпускается и здесь.
    """
    conditioning = _conditioning()

    async def scenario() -> None:
        started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            started.set()
            await asyncio.sleep(10)
            return httpx.Response(200, content=_png())

        async with _backend(handler, attempt_timeout=5.0) as backend:
            task = asyncio.create_task(backend.render(conditioning))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    _run(scenario())

    assert conditioning.sent is None, "отменённый кадр остался в памяти"


def test_the_image_is_taken_from_json_too():
    """Сервис вправе ответить и телом, и base64 в JSON — берём оба."""
    encoded = base64.b64encode(_png()).decode("ascii")

    async def scenario() -> bytes:
        async with _backend(lambda request: httpx.Response(200, json={"image": encoded})) as end:
            return await end.render(_conditioning())

    assert _run(scenario())[:4] == b"\x89PNG"


def test_an_answer_without_a_picture_is_an_error():
    """Успешный код с пустым телом — молчаливый отказ, и он обязан стать явным."""

    async def scenario() -> None:
        async with _backend(lambda request: httpx.Response(200, json={"status": "ok"})) as end:
            await end.render(_conditioning())

    with pytest.raises(adapter.IdentityBackendError):
        _run(scenario())


def test_the_backoff_grows_and_is_jittered():
    """
    Задержка растёт экспоненциально и размазана джиттером: конвейер шлёт кадры
    пачкой, и без джиттера все повторы после отбоя GPU придут в одну
    миллисекунду и положат его повторно.
    """
    backend = _backend(lambda request: httpx.Response(200), backoff_base=1.0, backoff_cap=100.0)

    first = [backend._backoff(1) for _ in range(20)]
    second = [backend._backoff(2) for _ in range(20)]

    assert max(first) <= 1.0 and min(first) >= 0.5
    assert min(second) >= max(first), "вторая попытка ждёт дольше первой"
    assert len(set(first)) > 1, "джиттера нет — задержка детерминирована"
