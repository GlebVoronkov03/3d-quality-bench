"""
Полный факториал искажений PLER-HQ: 4 типа × 10 уровней на каждый эталон из catalog.csv.
Пропускает pending_*; возобновляемый (не перезаписывает существующие OBJ).
Крупные меши (>1.5e6 граней) обрабатываются в конце.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradation import DegradationGenerator  # noqa: E402

META = ROOT / "DataBase" / "metadata"
DIST = ROOT / "DataBase" / "distorted"
LOG = META / "factorial_log.csv"
LARGE_FACE_THRESHOLD = 1_500_000


def load_ladder() -> dict:
    return json.loads((META / "distortion_ladder.json").read_text(encoding="utf-8"))


def load_catalog() -> list[dict]:
    with (META / "catalog.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def face_count(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.startswith(b"f "):
                n += 1
    return n


def ascii_source(src: Path, slug: str) -> Path:
    """pymeshlab на Windows падает на кириллических путях — копируем в ASCII-алиас."""
    try:
        src.resolve().as_posix().encode("ascii")
        return src
    except UnicodeEncodeError:
        pass
    if not src.name.isascii() or not str(src).isascii():
        alias_dir = ROOT / "DataBase" / "references" / "_ascii_aliases"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias = alias_dir / f"{slug}.obj"
        if not alias.exists() or alias.stat().st_size != src.stat().st_size:
            import shutil

            shutil.copy2(src, alias)
            mtl = src.with_suffix(".mtl")
            if mtl.exists():
                shutil.copy2(mtl, alias.with_suffix(".mtl"))
        return alias
    return src


def already(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 256


def log(msg: str) -> None:
    print(msg, flush=True)


def generate_for_model(gen: DegradationGenerator, row: dict, ladder: dict, max_mb: float) -> None:
    status = row["status"]
    src = ROOT / row["path"]
    if status == "pending_conversion":
        log(f"skip {row['slug']}: {status}")
        return
    if status == "pending_user" and not src.exists():
        log(f"skip {row['slug']}: waiting for user file")
        return
    if not src.exists():
        log(f"missing {src}")
        return
    size_mb = src.stat().st_size / 1e6
    if size_mb > max_mb:
        log(f"defer {row['slug']} {size_mb:.1f} MB > {max_mb} MB")
        return
    slug = row["slug"]
    src = ascii_source(src, slug)
    log(f"=== {slug} {size_mb:.2f} MB ===")
    jobs = [
        ("Decimation", ladder["Decimation"], None),
        ("Noise", ladder["Noise"], None),
        ("Smoothing", ladder["Smoothing"], None),
        ("Combined", ladder["Combined_decimation"], ladder["Combined_noise"]),
    ]
    for method, levels, extra in jobs:
        out_dir = DIST / slug / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, level in enumerate(levels, start=1):
            dst = out_dir / f"{slug}_{method}_{i:02d}.obj"
            if already(dst):
                continue
            if method == "Decimation":
                gen.decimate(str(src), str(dst), float(level))
            elif method == "Noise":
                gen.add_noise(str(src), str(dst), float(level))
            elif method == "Smoothing":
                gen.smooth(str(src), str(dst), int(level))
            else:
                gen.combined(str(src), str(dst), float(level), float(extra[i - 1]))
            log(f"  {dst.name}")


def main() -> int:
    max_mb = 8.0
    backend = "auto"
    only: set[str] | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--max-mb="):
            max_mb = float(arg.split("=", 1)[1])
        elif arg == "--large":
            max_mb = 1e9
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
        elif arg.startswith("--only="):
            only = {x.strip() for x in arg.split("=", 1)[1].split(",") if x.strip()}
    gen = DegradationGenerator(backend=backend)
    ladder = load_ladder()
    rows = load_catalog()
    if only:
        rows = [r for r in rows if r["slug"] in only]

    def size_of(row: dict) -> float:
        p = ROOT / row["path"]
        return p.stat().st_size if p.exists() else 1e18

    rows = sorted(rows, key=size_of)
    DIST.mkdir(parents=True, exist_ok=True)
    log(f"factorial pass max-mb={max_mb} backend={backend} n={len(rows)}")
    for row in rows:
        try:
            generate_for_model(gen, row, ladder, max_mb=max_mb)
        except Exception as exc:
            log(f"ERROR {row.get('slug')}: {exc}")
    log("factorial pass complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
