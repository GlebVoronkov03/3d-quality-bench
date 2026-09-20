#!/usr/bin/env python3
"""
Установка HybridMQA и FMQM: клонирование репозиториев и загрузка checkpoint.

    python scripts/setup_ml_metrics.py
    python scripts/setup_ml_metrics.py --hybridmqa-only
    python scripts/setup_ml_metrics.py --fmqm-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
CHECKPOINTS = ROOT / "models" / "checkpoints"

HYBRIDMQA_REPO = "https://github.com/arshafiee/hybridmqa.git"
FMQM_REPO = "https://github.com/yyyykf/FMQM.git"
HF_CKPT = "arshafiee/hybridmqa-checkpoint"
CKPT_FILENAME = "ckpt_TMQA_adapted.pth"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Уже есть: {dest}")
        return
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(dest)])


def download_hybridmqa_checkpoint() -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    dst = CHECKPOINTS / CKPT_FILENAME
    if dst.exists():
        print(f"Checkpoint уже есть: {dst}")
        return dst
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=HF_CKPT, filename=CKPT_FILENAME)
        import shutil

        shutil.copy2(path, dst)
        print(f"Checkpoint сохранён: {dst}")
        return dst
    except ImportError:
        print("Установите huggingface_hub: pip install huggingface_hub")
        print(f"Или скачайте {CKPT_FILENAME} вручную в {CHECKPOINTS}")
        sys.exit(1)


def install_hybridmqa_deps() -> None:
    req = THIRD_PARTY / "hybridmqa" / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup HybridMQA / FMQM")
    parser.add_argument("--hybridmqa-only", action="store_true")
    parser.add_argument("--fmqm-only", action="store_true")
    args = parser.parse_args()

    do_hybrid = not args.fmqm_only
    do_fmqm = not args.hybridmqa_only

    if do_hybrid:
        clone_repo(HYBRIDMQA_REPO, THIRD_PARTY / "hybridmqa")
        download_hybridmqa_checkpoint()
        install_hybridmqa_deps()

    if do_fmqm:
        clone_repo(FMQM_REPO, THIRD_PARTY / "fmqm")
        print(
            "FMQM: положите текстуры PNG/JPG рядом с OBJ или в data/textures/<model_name>/"
        )

    print("\nГотово. Запустите приложение: streamlit run app.py")


if __name__ == "__main__":
    main()
