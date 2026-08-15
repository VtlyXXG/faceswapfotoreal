"""
Сквозной прогон пайплайна: очистка шаблона → перенос личности.

Заменяет `test_identity.py`, который был линейным, синхронным и звал эндпоинт
замены лица — тот самый, с которого мы уходим из-за пластиковой кожи.

Граф из пяти шагов, и порядок в нём не свободный:

    A  чтение донора и шаблона, вектор личности           (пул CPU)
    B  маски, поза, деструкция структуры на щеках         (пул CPU)
    C  редактор снимает старую причёску                   (сеть, блокирующий SDK)
    D  сборка пакета: результат C как база                (пул CPU)
    E  ЗАГЛУШКА: пакет сериализуется на диск                (пул CPU)

**Шаги стоят по-разному, и останавливаться можно на любом.** A, B и E считаются
локально и бесплатно; платный здесь один C. Поэтому `--until` — не удобство, а
рабочий режим: сначала смотрят глазами на маску и на очищенный шаблон, и только
потом платят за инференс.

    # Маски и щёки. Ни ключа, ни сети, ни весов — стоит ноль
    ml-service\\.venv\\Scripts\\python.exe runner.py --until B

    # Плюс редактор причёски. Единственный шаг, которому нужен старый путь
    # через fal: он выключен по умолчанию, и без ML_FAL_ENABLED=true шаг
    # честно откажет с FAL_DISABLED, ничего не спрашивая и никуда не ходя
    ml-service\\.venv\\Scripts\\python.exe runner.py --until C

    # Весь граф до пакета для Flux (нужны веса antelopev2, см. bootstrap.py)
    ml-service\\.venv\\Scripts\\python.exe runner.py --until E ^
        --antelope C:\\models\\antelopev2

Что кладётся в outputs/:

    <имя>_mask.png      маска волос
    <имя>_overlay.png   маска красным поверх шаблона — по нему смотрят, не
                        срезана ли причёска и не уехала ли полоса на плечи
    <имя>_flat.png      шаблон после деструкции структуры на щеках (шаг B)
    <имя>_cleaned.png   ответ редактора, вклеенный в шаблон (шаг C) — вход шага E
    <имя>_aligned.png   кроп донора 112x112, из которого посчитан вектор
    <имя>_packet.json   тело запроса к Flux целиком (шаг E) — фикстура для
                        отладки бэкенда без пайплайна
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "ml-service"))

STEPS = ("A", "B", "C", "D", "E")


# --- Профилирование ----------------------------------------------------------


@dataclass
class Timings:
    """
    Тайминги шагов в порядке выполнения.

    Считается `perf_counter`, а не `time`: нас интересует длительность, а на
    Windows у `time.time()` разрешение хуже, чем шаг C бывает быстрым.
    """

    order: list[str] = field(default_factory=list)
    spent: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @contextlib.contextmanager
    def step(self, name: str, title: str):
        print(f"[{name}] {title} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.order.append(name)
            self.spent[name] = elapsed
            print(
                f"[{name}] {elapsed:7.2f} с   {self.notes.get(name, '')}".rstrip(),
                flush=True,
            )

    def note(self, name: str, text: str) -> None:
        self.notes[name] = text

    def report(self) -> str:
        total = sum(self.spent.values())
        lines = [
            "",
            "=== Тайминги ===",
            f"{'шаг':<4} {'секунд':>8} {'доля':>7}  примечание",
        ]
        for name in self.order:
            spent = self.spent[name]
            share = spent / total * 100 if total else 0.0
            lines.append(
                f"{name:<4} {spent:>8.2f} {share:>6.1f}%  {self.notes.get(name, '')}"
            )
        lines.append(f"{'ИТОГО':<4} {total:>8.2f}")
        return "\n".join(lines)


# --- Ресурсы -----------------------------------------------------------------


class Resources:
    """
    Всё, что поднимается один раз и закрывается один раз.

    Пулов два, и это не дублирование. Пул CPU занят MediaPipe и OpenCV, они
    отпускают GIL и работают по-настоящему. Пул I/O существует ради ОДНОГО
    места — блокирующего `fal_client.subscribe` на шаге C: положи его в общий
    пул, и трёхминутный сетевой вызов занял бы слот, нужный счётной задаче.

    Про отмену стоит знать вот что: шаг C отменить нельзя ничем. Блокирующий
    SDK живёт в потоке, и таймаут оборвёт ОЖИДАНИЕ, а сам поток продолжит висеть
    до собственного таймаута сокета. Это цена чужого SDK, и лечится она только
    заменой транспорта у fal — что уже сделано для будущего шага E, где стоит
    `httpx.AsyncClient` и отмена закрывает сокет немедленно.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")
        self.io = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io")
        self.extractor: Any = None
        self.identity: Any = None

    def warm(self) -> None:
        """
        Прогрев экстрактора: сессия onnxruntime поднимается и прогоняется
        вхолостую ДО первого кадра.

        Первый `run()` дороже последующих на порядок — там аллокация арен и
        выбор ядер. Без прогрева эта цена попала бы в тайминг шага A и читалась
        бы как «экстракция медленная».
        """
        from app.pipelines.refine import adapter

        self.extractor = adapter.AntelopeExtractor(model_root=self.args.antelope)
        self.extractor.warmup()
        self.identity = adapter.IdentityCache(self.extractor)

    async def close(self) -> None:
        """
        Закрывается всё и в любом случае.

        `shutdown(wait=True)` обязателен: без него поток шага C продолжил бы
        жить после выхода из `main`, держа и сокет, и весь кадр в памяти.

        Асинхронный метод при синхронном теле — намеренно: сюда вернётся
        закрытие httpx-клиента, когда шаг E перестанет быть заглушкой, и точка
        вызова в `finally` менять не будет.
        """
        self.extractor = None
        self.identity = None
        self.cpu.shutdown(wait=True)
        self.io.shutdown(wait=True)


# --- Шаги --------------------------------------------------------------------


async def in_pool(
    pool: ThreadPoolExecutor, fn: Callable[..., Any], *fnargs: Any
) -> Any:
    return await asyncio.get_running_loop().run_in_executor(pool, fn, *fnargs)


def step_a_identity(res: Resources, donor: Any) -> Any:
    """Вектор личности донора. Считается один раз на заказ."""
    from app.pipelines.refine import adapter

    return res.identity.get(adapter.IdentityCache.key(donor), donor)


def step_b_prepare(res: Resources, target: Any) -> dict[str, Any]:
    """
    Маски, поза, деструкция структуры старой пряди на щеках.

    Всё счётное собрано здесь: `hair_mask.build` поднимает сетку и разметку,
    `cheeks.flatten` гоняет свёртки по окну. На 4K это десятки миллисекунд
    каждая, и в событийном цикле они блокировали бы весь процесс.
    """
    from app.pipelines import cheeks, hair_mask
    from app.pipelines.refine import profiles

    profile = profiles.from_settings()
    stage = profile.hair

    hair = hair_mask.build(
        target,
        dilate_ratio=stage.dilate_ratio,
        feather_ratio=stage.feather_ratio,
        protect_ratio=stage.protect_ratio,
        forehead_ratio=stage.forehead_ratio,
        core_ratio=stage.core_ratio,
        guard_ratio=stage.guard_ratio,
        cheek_ratio=stage.cheek_ratio,
    )
    flat, cheek_meta = cheeks.flatten(
        target, hair, stage.cheek_wipe_ratio, stage.cheek_flat_ratio
    )

    return {"profile": profile, "hair": hair, "flat": flat, "cheeks": cheek_meta}


def step_c_strip_hair(
    res: Resources, prepared: dict[str, Any], target: Any
) -> dict[str, Any]:
    """
    Редактор снимает старую причёску. Блокирующий SDK — отсюда пул I/O.

    Возвращается ВКЛЕЕННЫЙ результат: из ответа берётся только область маски,
    всё остальное — включая лицо, которое ещё предстоит заменить, — остаётся
    от шаблона побитово.
    """
    from app.pipelines import composite, fal_api
    from app.utils.image import decode_image, encode_image

    profile, hair, flat = prepared["profile"], prepared["hair"], prepared["flat"]
    stage = profile.hair

    client = fal_api.client()
    scene_bytes, scene_mime = encode_image(flat, "png")
    image_url = fal_api.upload(client, scene_bytes, scene_mime)

    arguments = stage.payload.arguments(
        image_url=image_url,
        mask_url="",
        identity_url="",
        prompt=stage.prompt(res.args.hair),
        strength=profile.strength,
        guidance_scale=profile.guidance_scale,
        steps=profile.steps,
        output_format="png",
        identity_scale=profile.identity_scale,
    )
    generated, call_meta = fal_api.invoke(
        client, stage.endpoint, arguments, {"stage": "hair"}
    )

    pasted = composite.paste(target, decode_image(generated), hair.mask, match=True)
    changed = composite.difference(flat, pasted.image, hair.mask)
    return {"cleaned": pasted.image, "changed": changed, "call": call_meta}


def step_d_payload(
    res: Resources, prepared: dict[str, Any], cleaned: Any, vector: Any
) -> Any:
    """
    Пакет для собственного Flux. Базой идёт результат шага C, а не шаблон.

    Это и есть смысл порядка шагов: личность рисуется поверх кадра, с которого
    старая причёска уже снята, — иначе PuLID вписывал бы лицо в чужую копну.
    """
    from app.pipelines.refine import adapter

    return adapter.SceneConditioning(
        scene=cleaned,
        sent=cleaned.copy(),
        mask=prepared["hair"].mask,
        window=None,
        prompt=adapter.hair_prompt(res.args.prompt, res.args.colour, res.args.hair),
        identity=vector,
        weights=adapter.PuLIDWeights(fidelity=res.args.fidelity, steps=res.args.steps),
    )


def step_d_packet(res: Resources, conditioning: Any) -> dict[str, Any]:
    """
    Сериализация пакета. Считается один раз и переиспользуется шагом E.

    Раньше пакет собирался дважды — ради веса в логе и ещё раз внутри вызова
    бэкенда. На 4K это два кодирования JPEG подряд, то есть заметная доля
    тайминга шага D, потраченная ни на что.
    """
    from app.pipelines.refine import payload

    return payload.build_packet(conditioning)


def packet_weight(packet: dict[str, Any]) -> int:
    from app.pipelines.refine import payload

    return payload.packet_weight(packet)


def dump_packet(packet: dict[str, Any], path: Path, args: argparse.Namespace) -> int:
    """
    ЗАГЛУШКА ШАГА E: пакет уезжает не в сеть, а на диск.

    Боевой инстанс Flux + PuLID пока не развёрнут, и заглушка здесь честнее
    мока: на диск ложится ровно то тело запроса, которое ушло бы по сети, —
    вместе с base64 картинок, вектором личности и весами. По нему бэкенд
    поднимают и отлаживают без пайплайна, а сам файл годится как фикстура.

    Транспорт для боевого вызова уже написан и покрыт тестами
    (`adapter.HostedFluxPuLIDBackend`); замена заглушки — одна строка.

    :return: размер записанного файла в байтах
    """
    envelope = {
        # Куда это ушло бы. Пустая строка означает, что адрес не задан вовсе, —
        # для фикстуры это нормально, для боевого прогона нет
        "endpoint": f"{args.flux_url}/render" if args.flux_url else "",
        "source": Path(args.source).name,
        "target": Path(args.target).name,
        **packet,
    }

    # Пишется потоком в файл, а не через json.dumps в строку: у пакета внутри
    # base64 4K-кадра, и промежуточная строка удвоила бы его в памяти
    with path.open("w", encoding="utf-8") as handle:
        json.dump(envelope, handle, ensure_ascii=False)

    return path.stat().st_size


# --- Оркестратор -------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    from app.utils.image import decode_image, encode_image

    timings = Timings()
    res = Resources(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.target).stem

    def save(suffix: str, image: Any, fmt: str = "png") -> None:
        data, _ = encode_image(image, fmt)
        path = out / f"{stem}_{suffix}.{fmt}"
        path.write_bytes(data)
        print(f"        → {path}")

    target = donor = vector = prepared = cleaned = conditioning = packet = None
    try:
        with timings.step("A", "донор и шаблон, вектор личности"):
            target = decode_image(Path(args.target).read_bytes())
            donor = decode_image(Path(args.source).read_bytes())
            timings.note("A", f"шаблон {target.shape[1]}x{target.shape[0]}")

            if STEPS.index(args.until) >= STEPS.index("D"):
                res.warm()
                vector = await in_pool(res.cpu, step_a_identity, res, donor)
                save("aligned", vector.aligned)
                timings.note(
                    "A",
                    f"лицо донора {vector.quality:.0%} кадра, овал {vector.jaw_error:.3f}",
                )
            else:
                timings.note("A", "вектор не нужен на этом --until")

        with timings.step("B", "маски, поза, деструкция структуры на щеках"):
            prepared = await in_pool(res.cpu, step_b_prepare, res, target)
            hair = prepared["hair"]
            timings.note("B", describe_preparation(hair.meta, prepared["cheeks"]))
            save("mask", hair.mask)
            save("overlay", overlay(target, hair.mask))
            save("flat", prepared["flat"])

        if args.until == "B":
            print(timings.report())
            return 0

        with timings.step("C", "редактор снимает старую причёску"):
            stripped = await in_pool(res.io, step_c_strip_hair, res, prepared, target)
            cleaned = stripped["cleaned"]
            timings.note(
                "C", f"изменено внутри маски на {stripped['changed']:.2f} уровня"
            )
            save("cleaned", cleaned)

        if args.until == "C":
            print(timings.report())
            return 0

        with timings.step("D", "сборка пакета"):
            conditioning = await in_pool(
                res.cpu, step_d_payload, res, prepared, cleaned, vector
            )
            packet = await in_pool(res.cpu, step_d_packet, res, conditioning)
            # Пакет собран — очищенный кадр больше не нужен. На 4K это
            # десятки мегабайт, и держать их до конца прогона незачем
            conditioning.release_sent()
            timings.note("D", f"картинок {packet_weight(packet) / 1024 / 1024:.2f} МБ")

        if args.until == "D":
            print(timings.report())
            return 0

        with timings.step("E", "GPU-заглушка: пакет на диск"):
            target_path = out / f"{stem}_packet.json"
            size = await in_pool(res.cpu, dump_packet, packet, target_path, args)
            timings.note(
                "E", f"JSON {size / 1024 / 1024:.2f} МБ, боевой вызов не делался"
            )
            print(f"        → {target_path}")

        print(timings.report())
        return 0

    except KeyboardInterrupt:
        print("\nПрервано с клавиатуры.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — верхний уровень скрипта
        print(
            f"\nОШИБКА на шаге {timings.order[-1] if timings.order else '?'}: {exc}",
            file=sys.stderr,
        )
        details = getattr(exc, "details", None)
        if details:
            print(f"  подробности: {details}", file=sys.stderr)
        if timings.order:
            print(timings.report(), file=sys.stderr)
        return 1
    finally:
        # ПАМЯТЬ: разворот 4K — 25 МБ, и здесь их одновременно живёт пять штук.
        # Ссылки сбрасываются явно и до закрытия пулов: массив numpy умирает по
        # счётчику ссылок в тот момент, когда исчезает последняя, а не когда до
        # него дойдёт сборщик. Сборщик здесь занялся бы только циклами
        if conditioning is not None:
            conditioning.release_sent()
        target = donor = vector = prepared = cleaned = conditioning = packet = None
        await res.close()


def describe_preparation(mask_meta: dict[str, Any], cheek_meta: dict[str, Any]) -> str:
    """
    Строка про шаг B: поза, полосы и падение структуры.

    Это те три числа, ради которых шаг B вообще смотрят глазами. Поза говорит,
    насколько повёрнута голова; ширина дальней полосы — не схлопнулась ли она
    (за это уже платили композитом); падение структуры — сработала ли
    деструкция старой пряди.
    """
    pose = mask_meta.get("cheek_pose")
    if not pose:
        head = "позы нет (маска по разметке)"
    else:
        bands = {band["side"]: band for band in mask_meta.get("cheek_bands") or []}
        far = bands.get("far", {})
        head = (
            f"yaw {pose['yaw']:+.2f} pitch {pose['pitch']:+.2f}, "
            f"дальняя полоса {far.get('width_px', 0):.0f}px ({far.get('bound_by', '?')})"
        )

    if not cheek_meta.get("flattened"):
        return f"{head}; щёки не тронуты ({cheek_meta.get('reason')})"

    return f"{head}; структура {cheek_meta['structure_before']} → {cheek_meta['structure_after']}"


def overlay(target: Any, mask: Any) -> Any:
    """Маска красным поверх шаблона — то, на что смотрят глазами."""
    import cv2
    import numpy as np

    weight = (mask if mask.ndim == 2 else mask[..., 0]).astype(np.float32) / 255.0
    tint = np.zeros_like(target)
    tint[..., 2] = 255
    blended = target.astype(np.float32) * (1 - 0.45 * weight[..., None])
    blended += tint.astype(np.float32) * (0.45 * weight[..., None])
    return cv2.convertScaleAbs(blended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сквозной прогон: очистка шаблона → перенос личности",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", default="spread_01.png", help="шаблон-разворот")
    parser.add_argument(
        "--source", default="donor.jpg.jpeg", help="фотография заказчика"
    )
    parser.add_argument("--hair", default="short tidy hair", help="причёска словами")
    parser.add_argument(
        "--colour", default="", help="цвет волос, подставляется в промпт"
    )
    parser.add_argument(
        "--prompt",
        default="a photorealistic portrait, {colour} hair, natural skin pores, preserved microcontrast",
        help="шаблон промпта шага E; {colour} — место цвета",
    )
    parser.add_argument(
        "--fidelity", type=float, default=0.88, help="сила личности, 0.85..0.90"
    )
    parser.add_argument("--steps", type=int, default=28, help="шагов денойзинга")
    parser.add_argument(
        "--flux-url",
        default=os.environ.get("ML_FLUX_URL", ""),
        help="адрес будущего Flux. Сейчас только записывается в пакет: шаг E — заглушка",
    )
    parser.add_argument("--antelope", default=os.environ.get("ML_ANTELOPE_ROOT", ""))
    parser.add_argument("--out", default="ml-service/outputs")
    parser.add_argument(
        "--until",
        choices=STEPS,
        default="B",
        help="докуда идти. B — локально и бесплатно; C — плюс редактор; E — всё",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if STEPS.index(args.until) >= STEPS.index("D") and not args.antelope:
        print(
            "Шаг D требует весов antelopev2: --antelope или ML_ANTELOPE_ROOT.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
