"""Заменить белый фон Blender-рендеров на cream #F4F1EA."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "docs" / "pptx_assets" / "refs"
CREAM = np.array([244, 241, 234], dtype=np.uint8)


def convert(path: Path) -> None:
    im = np.asarray(Image.open(path).convert("RGB"))
    corner = im[4, 4].astype(np.int16)
    dist = np.abs(im.astype(np.int16) - corner).sum(axis=2)
    mask = dist < 14
    out = im.copy()
    out[mask] = CREAM
    Image.fromarray(out).save(path)
    print(f"{path.name}  bg_frac={mask.mean():.3f}  corner={tuple(corner)}")


def main() -> int:
    slugs = sys.argv[1:] or ["woody", "alligator", "automaton", "well", "egorov-lidar", "relief"]
    for slug in slugs:
        convert(REFS / f"{slug}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
