"""
MOS-подмножество под этику: 15–20 пар на сессию, эталон слева, dist справа.
5 ступеней из 10: 2, 4, 6, 8, 10. Пул стратифицирован по 6 функциональным классам.
Порядок в CSV перемешан (seed=42): не блоки «все уровни одного куба подряд».
Тяжёлые меши (WebGL / этика turntable) в пул не входят.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "DataBase" / "metadata"
OUT = META / "mos_session_design.csv"

MOS_LEVELS = [2, 4, 6, 8, 10]
METHODS = ["Decimation", "Noise", "Smoothing", "Combined"]
SHUFFLE_SEED = 42
SESSION_LEN = 18
MAX_FACES = 120_000

# Представители классов, пригодные для браузерного clay-view (два viewport).
# Тяжёлые сканы (bust, relief, sarcophagus, doll, well, peach, egorov-lidar) исключены.
SEED_SLUGS = [
    "cube",
    "torus",
    "stanford-bunny",
    "teapot",
    "suzanne",
    "nefertiti",
    "belt",
    "thingi-a",
    "aio-printer-test",
    "automaton",
    "bench",
    "fandisk",
    "axe",
    "homer",
    "cheburashka",
    "woody",
    "alligator",
    "bull-skull",
]


def load_stats() -> dict[str, dict]:
    path = META / "mesh_stats.csv"
    if not path.exists():
        return {}
    return {r["slug"]: r for r in csv.DictReader(path.open(encoding="utf-8-sig"))}


def n_faces(stats: dict, slug: str) -> int | None:
    rec = stats.get(slug) or {}
    raw = rec.get("n_faces", "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def main() -> None:
    catalog = {r["slug"]: r for r in csv.DictReader((META / "catalog.csv").open(encoding="utf-8-sig"))}
    stats = load_stats()
    raw: list[dict] = []
    skipped: list[str] = []
    for slug in SEED_SLUGS:
        rec = catalog.get(slug)
        if rec is None or rec.get("status") != "present":
            skipped.append(f"{slug}: missing")
            continue
        faces = n_faces(stats, slug)
        if faces is not None and faces > MAX_FACES:
            skipped.append(f"{slug}: {faces} faces")
            continue
        for method in METHODS:
            for lv in MOS_LEVELS:
                raw.append(
                    {
                        "slug": slug,
                        "category": rec["category"],
                        "function_ru": rec.get("function_ru", ""),
                        "method": method,
                        "level": lv,
                        "n_faces": faces if faces is not None else "",
                        "ref_path": rec["path"],
                        "dist_path": f"DataBase/distorted/{slug}/{method}/{slug}_{method}_{lv:02d}.obj",
                        "left": "reference",
                        "right": "distorted",
                        "scale": "5-point similarity MOS",
                        "presentation": "turntable 12s 360deg yaw 30deg/s, no textures",
                        "display_proxy": "0",
                    }
                )
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(raw)
    rows = []
    for i, rec in enumerate(raw, start=1):
        rec = dict(rec)
        rec["pair_id"] = i
        rec["session_block"] = (i - 1) // SESSION_LEN + 1
        rec["index_in_block"] = (i - 1) % SESSION_LEN + 1
        rows.append(rec)
    fields = [
        "pair_id",
        "slug",
        "category",
        "function_ru",
        "method",
        "level",
        "n_faces",
        "ref_path",
        "dist_path",
        "left",
        "right",
        "scale",
        "presentation",
        "display_proxy",
        "session_block",
        "index_in_block",
    ]
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_cat: dict[str, int] = {}
    for rec in rows:
        by_cat[rec["category"]] = by_cat.get(rec["category"], 0) + 1
    n_sess = (len(rows) + SESSION_LEN - 1) // SESSION_LEN
    print(f"wrote {OUT}: {len(rows)} pairs, {n_sess} blocks of {SESSION_LEN}")
    print("by class:", by_cat)
    if skipped:
        print("skipped:", "; ".join(skipped))


if __name__ == "__main__":
    main()
