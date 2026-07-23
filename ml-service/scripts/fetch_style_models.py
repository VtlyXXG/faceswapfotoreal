"""
Скачивает веса диффузионного стилизатора в models/.

    python scripts/fetch_style_models.py

Тянет ~5 ГБ:
  models/sd15/                    SD 1.5 в формате diffusers
  models/ip-adapter-faceid/       адаптер FaceID + LoRA (SD 1.5)

Идемпотентно: уже скачанное huggingface_hub не качает повторно.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Репозиторий runwayml/stable-diffusion-v1-5 снят с HF; это его действующее зеркало
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter-FaceID"
IP_ADAPTER_FILES = [
    "ip-adapter-faceid_sd15.bin",
    "ip-adapter-faceid_sd15_lora.safetensors",
]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _download_sd15(target: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"→ SD 1.5 → {target}")
    snapshot_download(
        repo_id=SD15_REPO,
        local_dir=str(target),
        # Диффузионный формат: без .ckpt/.safetensors цельных чекпойнтов и non-fp16
        allow_patterns=["*.json", "*.txt", "**/*.safetensors"],
        ignore_patterns=["*.ckpt", "*non_ema*", "*fp16*", "v1-5-pruned*"],
    )


def _download_ip_adapter(target: Path) -> None:
    from huggingface_hub import hf_hub_download

    print(f"→ IP-Adapter FaceID → {target}")
    target.mkdir(parents=True, exist_ok=True)
    for filename in IP_ADAPTER_FILES:
        hf_hub_download(repo_id=IP_ADAPTER_REPO, filename=filename, local_dir=str(target))


def main() -> int:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub не установлен — выполните scripts/setup_venv.ps1", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_sd15(MODELS_DIR / "sd15")
    _download_ip_adapter(MODELS_DIR / "ip-adapter-faceid")

    print("\nГотово. Включите стилизатор: ML_STYLE_PROVIDER=diffusion")
    print("Проверка: curl http://localhost:8000/health/ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
