"""
Провижининг весов для gpu-inference-server.

Сервер обязан работать оффлайн: в рантайме он поднят с `HF_HUB_OFFLINE=1`, и
докачивать что-либо на первом заказе ему запрещено. Причина та же, по которой
ml-service кладёт веса разметки на этапе сборки образа: иначе их качает первый
же заказ, и при масштабировании — в каждом новом контейнере заново. Только
здесь речь не о 16 МБ, а о семи гигабайтах.

    python download_weights.py --root ./weights        # всё
    python download_weights.py --check                 # что уже лежит
    python download_weights.py --only sdxl controlnet  # выборочно
    python download_weights.py --token hf_...          # если репозиторий закрыт

ИСТОЧНИКИ НЕ ПРИБИТЫ ГВОЗДЯМИ. Идентификаторы репозиториев и адреса релизов
живут в манифесте ниже и переопределяются окружением, поштучно:

    GPU_WEIGHTS_REPO_SDXL=<org>/<repo>     # для entry вида snapshot/file
    GPU_WEIGHTS_URL_LAMA=https://...       # для entry вида url

Так сделано не из любви к настройкам: репозитории на HuggingFace переезжают и
переименовываются, а файлы в релизах GitHub — двигаются между тегами. Когда
источник переедет, чинить это должно быть переменной окружения, а не правкой
кода и пересборкой образа.

ЧЕГО ЗДЕСЬ НЕТ: кода архитектуры CodeFormer. Скачивается только `codeformer.pth`
(это state_dict), а классы модели на PyPI не изданы — см. докстринг
`FaceRestorer` в server.py. Автономный путь, если восстановление лица нужно, —
собрать торчскрипт из официального репозитория и положить сюда как
`weights/codeformer/codeformer.jit.pt`:

    git clone https://github.com/sczhou/CodeFormer && cd CodeFormer
    python -c "import torch; from basicsr.utils.registry import ARCH_REGISTRY; \\
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, \\
            n_head=8, n_layers=9, connect_list=['32','64','128','256']).eval(); \\
        net.load_state_dict(torch.load('weights/CodeFormer/codeformer.pth')['params_ema']); \\
        torch.jit.save(torch.jit.script(net), 'codeformer.jit.pt')"

Без него сервер работает: восстановление выключается, причина видна на
/health/ready.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCK_NAME = "manifest.lock.json"

# Шаблоны fp16. Без них snapshot_download тянет ВСЁ: репозиторий SDXL содержит
# и fp32-веса, и .bin-дубликаты safetensors, и на диск приезжает ~27 ГБ вместо
# семи. Тексты и json обязательны отдельной строкой — там токенизаторы
# (vocab.json, merges.txt) и model_index.json, без которых пайплайн не соберётся.
FP16_PATTERNS = [
    "*.json",
    "*.txt",
    "*.fp16.safetensors",
    "**/*.fp16.safetensors",
]

# Когда fp16-варианта в репозитории нет вовсе (ControlNet и VAE обычно изданы
# одним файлом уже в половинной точности)
PLAIN_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "**/*.safetensors",
]


@dataclass(frozen=True)
class Entry:
    """
    Один источник. `kind` определяет способ доставки, а не важность.

      snapshot — каталог модели diffusers целиком (по шаблонам)
      file     — один файл из репозитория HuggingFace
      url      — прямая ссылка (релизы GitHub: LaMa, CodeFormer, facexlib)
    """

    key: str
    kind: str
    target: str
    description: str
    repo: str = ""
    filename: str = ""
    url: str = ""
    patterns: list[str] = field(default_factory=list)
    # Что обязано появиться после скачивания. Проверка нужна, потому что
    # snapshot_download с шаблонами, не совпавшими ни с чем, завершается
    # УСПЕШНО и оставляет пустой каталог — а падает это потом, на загрузке
    # пайплайна, где причина уже не видна
    required: list[str] = field(default_factory=list)
    optional: bool = False
    # Дубликат файла: facexlib ищет веса и в корне заданного каталога, и в
    # подкаталоге weights — какой именно, зависит от версии
    mirror: str = ""
    # Запрет докачки без фильтра. Фолбэк в fetch_snapshot задуман для репозитория
    # из одной модели, где не угадан суффикс имени. Там он стоит лишнего трафика,
    # здесь — катастрофы: h94/IP-Adapter это склад из полутора десятков адаптеров
    # под SD15 и SDXL плюс два энкодера, ~20 ГБ, из которых нам нужны два файла.
    # Для таких записей правильное поведение — упасть с внятной причиной
    strict: bool = False


MANIFEST: list[Entry] = [
    Entry(
        key="sdxl",
        kind="snapshot",
        repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        target="sdxl-inpaint",
        description="SDXL inpainting 0.1, fp16 (~7 ГБ)",
        patterns=FP16_PATTERNS,
        required=["model_index.json", "unet", "text_encoder", "tokenizer", "scheduler"],
    ),
    Entry(
        key="vae",
        kind="snapshot",
        repo="madebyollin/sdxl-vae-fp16-fix",
        target="sdxl-vae-fp16-fix",
        description="VAE без переполнения в fp16 (~320 МБ)",
        patterns=PLAIN_PATTERNS,
        required=["config.json"],
    ),
    Entry(
        key="controlnet",
        kind="snapshot",
        repo="destitech/controlnet-inpaint-dreamer-sdxl",
        target="controlnet-inpaint-sdxl",
        description="ControlNet inpaint для SDXL, v2 (~2.5 ГБ, fp16)",
        patterns=["v2/*.json", "v2/*.fp16.safetensors"],
        required=["v2/config.json"],
    ),
    # IP-Adapter Plus Face — единственный вход личности у этой связки.
    #
    # Одна запись, а не две: `pipe.load_ip_adapter` читает и адаптер, и энкодер
    # от ОДНОГО корня, разводя их аргументами `subfolder`/`image_encoder_folder`.
    # Разложить их по разным target значило бы потерять эту связь.
    #
    # ЭНКОДЕР БЕРЁТСЯ ИЗ models/, А НЕ ИЗ sdxl_models/. Ловушка в том, что имя
    # каталога намекает на обратное. В sdxl_models/image_encoder лежит ViT-bigG
    # (hidden_size 1664, 48 слоёв) — он парный к ip-adapter_sdxl.safetensors. Наш
    # файл называется ...vit-h и обучен на ViT-H (hidden_size 1280, 32 слоя),
    # который издан в models/image_encoder. Перепутать их значит получить не
    # ошибку, а несовпадение размерностей проекции при load_ip_adapter.
    #
    # Шаблоны перечисляют файлы поимённо: в sdxl_models/ лежат восемь адаптеров
    # (four .safetensors + столько же .bin-дубликатов), и любой шаблон вида
    # `sdxl_models/*.safetensors` утянет 3 ГБ лишнего. Отсюда же strict — фолбэк
    # без фильтра на этом репозитории означает ~20 ГБ
    Entry(
        key="ip-adapter",
        kind="snapshot",
        repo="h94/IP-Adapter",
        target="ip-adapter",
        description="IP-Adapter Plus Face SDXL + ViT-H энкодер (~3.2 ГБ)",
        patterns=[
            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ],
        required=[
            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ],
        strict=True,
    ),
    Entry(
        key="lama",
        kind="url",
        url="https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        target="lama/big-lama.pt",
        description="LaMa, торчскрипт (~200 МБ)",
    ),
    Entry(
        key="codeformer",
        kind="url",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        target="codeformer/codeformer.pth",
        description="CodeFormer, state_dict (~360 МБ)",
        optional=True,
    ),
    Entry(
        key="facexlib-detection",
        kind="url",
        url="https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        target="facexlib/detection_Resnet50_Final.pth",
        description="Детектор лица для CodeFormer (~110 МБ)",
        optional=True,
        mirror="facexlib/weights/detection_Resnet50_Final.pth",
    ),
    Entry(
        key="facexlib-parsing",
        kind="url",
        url="https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        target="facexlib/parsing_parsenet.pth",
        description="Разметка лица для обратной вклейки (~85 МБ)",
        optional=True,
        mirror="facexlib/weights/parsing_parsenet.pth",
    ),
]


# --- Переопределения ---------------------------------------------------------


def _slug(key: str) -> str:
    return key.upper().replace("-", "_")


def resolve(entry: Entry) -> Entry:
    """Применяет GPU_WEIGHTS_REPO_* / GPU_WEIGHTS_URL_* к записи манифеста."""
    repo = os.environ.get(f"GPU_WEIGHTS_REPO_{_slug(entry.key)}", "").strip()
    url = os.environ.get(f"GPU_WEIGHTS_URL_{_slug(entry.key)}", "").strip()
    if not repo and not url:
        return entry

    values = entry.__dict__.copy()
    if repo:
        values["repo"] = repo
    if url:
        values["url"] = url
        values["kind"] = "url"
    return Entry(**values)


# --- Доставка ----------------------------------------------------------------


def human(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def present(entry: Entry, root: Path) -> bool:
    """Лежит ли уже. Для snapshot проверяются `required`, для файла — он сам."""
    target = root / entry.target
    if entry.kind == "url" or entry.kind == "file":
        return target.exists() and target.stat().st_size > 0
    if not target.exists():
        return False
    return all((target / name).exists() for name in entry.required)


def fetch_snapshot(entry: Entry, root: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    target = root / entry.target
    patterns = entry.patterns or None

    snapshot_download(
        repo_id=entry.repo,
        local_dir=str(target),
        allow_patterns=patterns,
        token=token or None,
        # Симлинки на веса — беда при копировании каталога в образ докера: сам
        # файл остаётся в кэше HF, а в слой приезжает битая ссылка
        local_dir_use_symlinks=False,
    )

    missing = [name for name in entry.required if not (target / name).exists()]
    if missing and entry.strict:
        raise RuntimeError(
            f"шаблоны не дали обязательных файлов: {', '.join(missing)}. "
            f"Докачка без фильтра для {entry.key} запрещена (strict): проверьте "
            f"структуру {entry.repo} — она могла поменяться"
        )
    if missing and patterns:
        # Шаблон fp16 не совпал ни с чем — репозиторий издан без суффикса.
        # Повтор без фильтра дороже по трафику, но лучше пустого каталога,
        # который упадёт потом и в другом месте
        print(f"  fp16-вариант не найден ({', '.join(missing)}), тяну без фильтра")
        snapshot_download(
            repo_id=entry.repo,
            local_dir=str(target),
            allow_patterns=PLAIN_PATTERNS,
            token=token or None,
            local_dir_use_symlinks=False,
        )
        missing = [name for name in entry.required if not (target / name).exists()]

    if missing:
        raise RuntimeError(
            f"после скачивания нет обязательных элементов: {', '.join(missing)}. "
            f"Проверьте GPU_WEIGHTS_REPO_{_slug(entry.key)}"
        )


def fetch_file(entry: Entry, root: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    target = root / entry.target
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=entry.repo, filename=entry.filename, token=token or None)
    shutil.copyfile(downloaded, target)


def fetch_url(entry: Entry, root: Path, _: str | None) -> None:
    target = root / entry.target
    target.parent.mkdir(parents=True, exist_ok=True)

    # Скачивание во временный файл рядом с целью, а не сразу в неё: оборванная
    # закачка иначе оставляет файл нужного имени и неверного размера, а
    # следующий запуск считает его готовым и молча пропускает
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(entry.url, headers={"User-Agent": "projectx-gpu-server"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(target)

    if entry.mirror:
        mirror = root / entry.mirror
        mirror.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, mirror)


FETCH = {"snapshot": fetch_snapshot, "file": fetch_file, "url": fetch_url}


# --- Оркестрация -------------------------------------------------------------


def report(entries: list[Entry], root: Path) -> int:
    """Что лежит, что нет. Отдельный режим, потому что это первый вопрос при
    разборе «почему сервис degraded»."""
    print(f"Корень весов: {root}\n")
    missing_required = 0
    for entry in entries:
        target = root / entry.target
        ok = present(entry, root)
        if not ok and not entry.optional:
            missing_required += 1
        mark = "есть" if ok else ("нет (необязательно)" if entry.optional else "НЕТ")
        size = human(tree_size(target)) if ok else "-"
        print(f"  {entry.key:<20} {mark:<20} {size:>10}  {entry.description}")

    print(f"\nВсего на диске: {human(tree_size(root))}")
    if missing_required:
        print(f"Не хватает обязательных: {missing_required}")
    return 1 if missing_required else 0


def download(entries: list[Entry], root: Path, token: str | None, force: bool) -> int:
    root.mkdir(parents=True, exist_ok=True)
    lock: dict[str, Any] = {}
    failures: list[str] = []

    for index, entry in enumerate(entries, 1):
        head = f"[{index}/{len(entries)}] {entry.key} — {entry.description}"
        if present(entry, root) and not force:
            print(f"{head}\n  уже на месте, пропускаю")
            lock[entry.key] = _record(entry, root, "skipped")
            continue

        print(head)
        print(f"  источник: {entry.repo or entry.url}")
        try:
            FETCH[entry.kind](entry, root, token)
        except Exception as exc:  # noqa: BLE001 — сеть, права, переехавший репозиторий
            # Необязательное не роняет провижининг: сервер без CodeFormer
            # работает, и требовать его ради запуска значит блокировать
            # выкладку из-за переехавшей ссылки на GitHub
            level = "необязательное" if entry.optional else "ОБЯЗАТЕЛЬНОЕ"
            print(f"  не удалось ({level}): {exc}", file=sys.stderr)
            if not entry.optional:
                failures.append(entry.key)
            lock[entry.key] = _record(entry, root, f"failed: {exc}")
            continue

        print(f"  готово, {human(tree_size(root / entry.target))}")
        lock[entry.key] = _record(entry, root, "downloaded")

    (root / LOCK_NAME).write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nИтого на диске: {human(tree_size(root))}")
    print(f"Отчёт: {root / LOCK_NAME}")

    if failures:
        print(f"\nОбязательные не скачаны: {', '.join(failures)}", file=sys.stderr)
        print("Переопределите источник переменной GPU_WEIGHTS_REPO_<КЛЮЧ> "
              "или GPU_WEIGHTS_URL_<КЛЮЧ>", file=sys.stderr)
        return 1

    print("\nСервер готов к оффлайн-работе:")
    print("  uvicorn server:app --host 0.0.0.0 --port 8100")
    return 0


def _record(entry: Entry, root: Path, state: str) -> dict[str, Any]:
    """Строка отчёта. Нужна, чтобы «а какие веса в этом образе» отвечалось
    файлом, а не археологией по логам сборки."""
    return {
        "source": entry.repo or entry.url,
        "kind": entry.kind,
        "target": entry.target,
        "state": state,
        "bytes": tree_size(root / entry.target),
        "optional": entry.optional,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Скачивает веса SDXL, ControlNet, LaMa и CodeFormer в ./weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ключи: " + ", ".join(entry.key for entry in MANIFEST),
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("GPU_WEIGHTS_ROOT", "./weights"),
        help="куда класть веса (GPU_WEIGHTS_ROOT, по умолчанию ./weights)",
    )
    parser.add_argument("--only", nargs="+", metavar="KEY", help="скачать только эти ключи")
    parser.add_argument("--check", action="store_true", help="только показать, что уже лежит")
    parser.add_argument("--force", action="store_true", help="перекачать даже то, что на месте")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="токен HuggingFace для закрытых репозиториев (HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    # Провижининг — единственное место, которому в сеть МОЖНО. Сервер ставит
    # HF_HUB_OFFLINE=1 себе сам, и если переменная утекла сюда из окружения
    # образа, скачивание молча не сделает ничего
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    entries = [resolve(entry) for entry in MANIFEST]
    if args.only:
        known = {entry.key for entry in entries}
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f"неизвестные ключи: {', '.join(unknown)}. Есть: {', '.join(sorted(known))}")
        entries = [entry for entry in entries if entry.key in set(args.only)]

    root = Path(args.root).expanduser().resolve()
    if args.check:
        return report(entries, root)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("нет huggingface_hub: pip install -r requirements.txt", file=sys.stderr)
        return 1

    return download(entries, root, args.token, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
