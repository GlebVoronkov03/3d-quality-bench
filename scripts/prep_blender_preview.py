"""Voxel-превью огромных OBJ → docs/pptx_assets/_preview/{slug}.obj для Blender."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from render_pptx_assets import load_huge_sample  # noqa: E402

CAT = ROOT / "DataBase" / "metadata" / "catalog.csv"
OUT = ROOT / "docs" / "pptx_assets" / "_preview"


def main() -> int:
    slugs = [a for a in sys.argv[1:] if not a.startswith("-")] or ["relief"]
    rows = {r["slug"]: r for r in csv.DictReader(CAT.open(encoding="utf-8-sig"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        rec = rows[slug]
        src = ROOT / rec["path"]
        dst = OUT / f"{slug}.obj"
        print(f"preview {slug} {src.stat().st_size/1e6:.1f} MB", flush=True)
        tm = load_huge_sample(src, grid=140, max_faces=90_000)
        tm = trimesh.Trimesh(
            vertices=np.asarray(tm.vertices),
            faces=np.asarray(tm.faces),
            process=False,
        )
        tm.export(dst)
        print(f"  wrote {dst} V={len(tm.vertices)} F={len(tm.faces)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
