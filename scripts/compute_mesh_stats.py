"""Геометрическая статистика эталонов → DataBase/metadata/mesh_stats.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "DataBase" / "metadata" / "catalog.csv"
OUT = ROOT / "DataBase" / "metadata" / "mesh_stats.csv"


def as_mesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return None


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False, skip_materials=True)
    tm = as_mesh(loaded)
    if tm is None or tm.is_empty or len(tm.faces) == 0:
        raise RuntimeError(f"empty {path}")
    return tm


def sample_mean_curvature(tm: trimesh.Trimesh, cap: int = 120_000) -> float:
    """Средний двугранный угол (рад) — быстрый прокси средней кривизны."""
    work = tm
    if len(tm.faces) > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(tm.faces), size=cap, replace=False)
        work = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces[idx], process=False)
    adj = getattr(work, "face_adjacency", None)
    if adj is None or len(adj) == 0:
        return float("nan")
    nrm = work.face_normals
    a, b = adj[:, 0], adj[:, 1]
    dots = np.clip(np.einsum("ij,ij->i", nrm[a], nrm[b]), -1.0, 1.0)
    return float(np.mean(np.arccos(dots)))


def edge_manifold(tm: trimesh.Trimesh) -> tuple[int, float]:
    edges = np.sort(np.asarray(tm.faces)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    n_nm = int(np.sum(counts != 2))
    frac = float(n_nm / max(len(counts), 1))
    return n_nm, frac


def stream_counts_aabb(path: Path) -> tuple[int, int, float]:
    nv = nf = 0
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    with path.open("rb") as handle:
        for raw in handle:
            if raw.startswith(b"v "):
                parts = raw.split()
                if len(parts) >= 4:
                    pt = np.array((float(parts[1]), float(parts[2]), float(parts[3])))
                    lo = np.minimum(lo, pt)
                    hi = np.maximum(hi, pt)
                    nv += 1
            elif raw.startswith(b"f "):
                ids = raw.split()[1:]
                if len(ids) >= 3:
                    nf += max(len(ids) - 2, 1)
    diag = float(np.linalg.norm(hi - lo)) if np.isfinite(lo).all() else float("nan")
    return nv, nf, diag


def genus_estimate(euler, n_comp) -> str:
    try:
        return f"{(2.0 * float(n_comp) - float(euler)) / 2.0:.4g}"
    except (TypeError, ValueError):
        return ""


def stats_for(path: Path) -> dict:
    if path.stat().st_size > 180_000_000:
        nv, nf, diag = stream_counts_aabb(path)
        return {
            "n_vertices": nv,
            "n_faces": nf,
            "n_edges": "",
            "euler": "",
            "genus": "",
            "genus_est": "",
            "n_components": "",
            "watertight": "",
            "nonmanifold_edges": "",
            "nonmanifold_frac": "",
            "aabb_diag": f"{diag:.6g}" if np.isfinite(diag) else "",
            "mean_abs_curv": "",
            "note": "stream V/F + AABB, file>180MB",
        }
    tm = load_mesh(path)
    v = len(tm.vertices)
    f = len(tm.faces)
    n_nm, nm_frac = edge_manifold(tm)
    e = int((3 * f) / 2) if f else 0
    aabb = tm.bounds
    diag = float(np.linalg.norm(aabb[1] - aabb[0])) if aabb is not None else float("nan")
    euler = v - e + f
    try:
        n_comp = int(tm.body_count) if hasattr(tm, "body_count") else 1
    except Exception:
        n_comp = 1
    watertight = bool(getattr(tm, "is_watertight", False))
    genus = ""
    if watertight:
        try:
            genus = int(round((2 * n_comp - euler) / 2))
        except Exception:
            genus = ""
    curv = sample_mean_curvature(tm)
    return {
        "n_vertices": v,
        "n_faces": f,
        "n_edges": e,
        "euler": euler,
        "genus": genus,
        "genus_est": genus_estimate(euler, n_comp),
        "n_components": n_comp,
        "watertight": int(watertight),
        "nonmanifold_edges": n_nm,
        "nonmanifold_frac": f"{nm_frac:.6f}",
        "aabb_diag": f"{diag:.6g}",
        "mean_abs_curv": f"{curv:.6g}" if np.isfinite(curv) else "",
        "note": "",
    }


FIELDS = [
    "slug",
    "n_vertices",
    "n_faces",
    "n_edges",
    "euler",
    "genus",
    "genus_est",
    "n_components",
    "watertight",
    "nonmanifold_edges",
    "nonmanifold_frac",
    "aabb_diag",
    "mean_abs_curv",
    "note",
]


def empty_rec(slug: str) -> dict:
    rec = {k: "" for k in FIELDS}
    rec["slug"] = slug
    return rec


def main() -> int:
    only = {a for a in sys.argv[1:] if not a.startswith("-")}
    merge = "--merge" in sys.argv or bool(only)
    catalog = list(csv.DictReader(CAT.open(encoding="utf-8-sig")))
    by_slug = {}
    if merge and OUT.exists():
        for row in csv.DictReader(OUT.open(encoding="utf-8-sig")):
            by_slug[row["slug"]] = row
    out_rows = []
    for r in catalog:
        slug = r["slug"]
        rec = by_slug.get(slug, empty_rec(slug))
        rec = {k: rec.get(k, "") for k in FIELDS}
        rec["slug"] = slug
        if only and slug not in only:
            if rec.get("genus_est") == "" and rec.get("euler") not in ("", None):
                rec["genus_est"] = genus_estimate(rec.get("euler"), rec.get("n_components"))
            out_rows.append(rec)
            continue
        src = ROOT / r["path"]
        if not src.exists():
            rec["note"] = "missing"
            out_rows.append(rec)
            print("missing", slug, flush=True)
            continue
        print(f"stat {slug} {src.stat().st_size/1e6:.1f} MB", flush=True)
        try:
            rec.update(stats_for(src))
        except Exception as exc:
            rec["note"] = str(exc)[:180]
            print("  FAIL", exc, flush=True)
        if rec.get("genus_est") == "" and rec.get("euler") not in ("", None):
            rec["genus_est"] = genus_estimate(rec.get("euler"), rec.get("n_components"))
        out_rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
