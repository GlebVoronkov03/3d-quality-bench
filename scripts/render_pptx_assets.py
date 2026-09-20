"""Рендеры и схемы для презентации PLER-HQ."""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import open3d as o3d  # noqa: E402

OUT = ROOT / "docs" / "pptx_assets"
CAT = ROOT / "DataBase" / "metadata" / "catalog.csv"

BG = "#F4F1EA"
INK = "#1C2430"
ACCENT = "#9C4A2B"
NAVY = "#243044"
MUTED = "#5C6570"
CARD = "#FFFcf7"
TEAL = "#3D6B6B"
BG_RGB = np.array([244, 241, 234], dtype=np.int16)

# Open3D silently drops n-gons; these thumbs must be rebuilt via trimesh.
FORCE_SLUGS = {
    "suzanne",
    "automaton",
    "belt",
    "boots",
    "axe",
    "murano-chair",
    "bench",
    "skeleton",
    "moncey",
    "beast",
    "relief",
    "uv-sphere-dense",
    "cube",
    "teapot",
    "well",
    "doll",
    "alligator",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "savefig.dpi": 180,
        }
    )


def occupancy(path: Path) -> float:
    if not path.exists() or path.stat().st_size < 800:
        return 0.0
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - BG_RGB).sum(axis=2)
    return float((diff > 22).mean())


def ascii_copy(src: Path) -> Path:
    try:
        src.resolve().as_posix().encode("ascii")
        return src
    except UnicodeEncodeError:
        pass
    tmp = ROOT / "docs" / "pptx_assets" / "_tmp_ascii"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / f"m_{abs(hash(src.as_posix())) % 10**8}.obj"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def _as_trimesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return None


def preview_source(slug: str, src: Path) -> Path:
    """Prefer the true reference. Distortion LODs are used only if they still look like the model."""
    if src.exists() and src.stat().st_size <= 140_000_000:
        return src
    for lvl in ("05", "06", "04", "08", "07"):
        proxy = ROOT / "DataBase" / "distorted" / slug / "Decimation" / f"{slug}_Decimation_{lvl}.obj"
        if not proxy.exists():
            continue
        sz = proxy.stat().st_size
        if sz < 800_000 or sz >= src.stat().st_size:
            continue
        try:
            tm = _as_trimesh(trimesh.load(str(proxy), process=False, skip_materials=True))
        except Exception:
            continue
        if tm is None or len(tm.faces) < 1500:
            continue
        ext = np.asarray(tm.extents, dtype=float)
        if float(np.min(ext)) / max(float(np.max(ext)), 1e-12) < 0.04:
            continue
        print(f"  proxy {slug} Decimation_{lvl}", flush=True)
        return proxy
    return src


def load_huge_sample(path: Path, grid: int = 150, max_faces: int = 120_000) -> trimesh.Trimesh:
    """Voxel-cluster a huge OBJ so a photogrammetry panel still looks like itself."""
    print(f"  stream-voxel {path.name}", flush=True)
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    n_v = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.startswith(b"v "):
                continue
            p = raw.split()
            pt = np.array((float(p[1]), float(p[2]), float(p[3])))
            lo = np.minimum(lo, pt)
            hi = np.maximum(hi, pt)
            n_v += 1
    span = np.maximum(hi - lo, 1e-12)
    inv = (grid - 1) / span
    vert_key = np.empty(n_v, dtype=np.int32)
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    key_to_new: dict[int, int] = {}
    vi = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.startswith(b"v "):
                continue
            p = raw.split()
            pt = np.array((float(p[1]), float(p[2]), float(p[3])))
            ijk = np.clip(((pt - lo) * inv).astype(np.int32), 0, grid - 1)
            key = int(ijk[0] + grid * (ijk[1] + grid * ijk[2]))
            vert_key[vi] = key
            if key not in sums:
                sums[key] = pt.astype(np.float64)
                counts[key] = 1
                key_to_new[key] = len(key_to_new)
            else:
                sums[key] += pt
                counts[key] += 1
            vi += 1
    verts = np.zeros((len(key_to_new), 3), dtype=np.float64)
    for key, idx in key_to_new.items():
        verts[idx] = sums[key] / counts[key]
    faces: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.startswith(b"f "):
                continue
            ids = [int(tok.split(b"/")[0]) - 1 for tok in raw.split()[1:] if tok not in (b"\n",)]
            if len(ids) < 3:
                continue
            mapped = [key_to_new[int(vert_key[i])] for i in ids if 0 <= i < n_v]
            if len(mapped) < 3:
                continue
            a = mapped[0]
            for k in range(1, len(mapped) - 1):
                tri = (a, mapped[k], mapped[k + 1])
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    continue
                key = tuple(sorted(tri))
                if key in seen:
                    continue
                seen.add(key)
                faces.append(tri)
                if len(faces) >= max_faces:
                    break
            if len(faces) >= max_faces:
                break
    if not faces:
        raise RuntimeError(f"no voxel faces {path.name}")
    print(f"  voxel verts={len(verts)} faces={len(faces)} from {n_v} src verts", flush=True)
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int32), process=False)


def load_trimesh(path: Path) -> trimesh.Trimesh:
    src = path if path.stat().st_size > 100_000_000 else ascii_copy(path)
    if src.stat().st_size > 180_000_000:
        return load_huge_sample(src)
    loaded = trimesh.load(str(src), process=False, skip_materials=True)
    tm = _as_trimesh(loaded)
    if tm is None or tm.is_empty or len(tm.faces) == 0:
        raise RuntimeError(f"empty {path.name}")
    return tm


def to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
    )
    return mesh


def orient_flat(pts: np.ndarray) -> np.ndarray:
    c = np.median(pts, axis=0)
    x = pts - c
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    r = vh.copy()
    if np.linalg.det(r) < 0:
        r[1] *= -1
    y = x @ r.T
    a = np.deg2rad(28.0)
    b = np.deg2rad(-16.0)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return y @ ry.T @ rx.T


def prepare_mesh(path: Path, max_faces: int = 110_000) -> o3d.geometry.TriangleMesh:
    tm = load_trimesh(path)
    mesh = to_o3d(tm)
    n = len(mesh.triangles)
    if n > max_faces:
        aabb = mesh.get_axis_aligned_bounding_box()
        ext = float(np.max(aabb.get_extent()))
        voxel = max(ext / 150.0, 1e-12)
        mesh = mesh.simplify_vertex_clustering(
            voxel, contraction=o3d.geometry.SimplificationContraction.Average
        )
    pts = np.asarray(mesh.vertices)
    if len(pts) < 3:
        raise RuntimeError(f"too few verts {path.name}")
    lo, hi = np.percentile(pts, [0.7, 99.3], axis=0)
    ext = hi - lo
    if float(np.min(ext)) / max(float(np.max(ext)), 1e-12) < 0.12:
        pts = orient_flat(pts)
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        lo, hi = np.percentile(pts, [0.7, 99.3], axis=0)
        ext = hi - lo
    center = 0.5 * (lo + hi)
    span = float(np.max(ext))
    if span <= 0:
        span = float(np.max(np.ptp(pts, axis=0))) or 1.0
    mesh.translate(-center)
    mesh.scale(0.78 / span, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.58, 0.60, 0.63])
    return mesh


def _capture(mesh, dst: Path, size: int, front, zoom: float) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=size, height=size, visible=False)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.957, 0.945, 0.918])
    opt.mesh_show_back_face = True
    opt.light_on = True
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front(list(front))
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(dst), do_render=True)
    vis.destroy_window()


def render_path(src: Path, dst: Path, size: int = 640) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh = prepare_mesh(src)
    except Exception as exc:
        print(f"  FAIL load {src.name}: {exc}", flush=True)
        return False
    views = [
        ([0.55, -0.22, 0.80], 0.78),
        ([0.55, -0.22, 0.80], 1.15),
        ([0.12, -0.08, 0.99], 0.70),
        ([0.92, -0.12, 0.38], 0.70),
        ([0.20, 0.72, 0.66], 0.75),
        ([0.55, -0.22, 0.80], 0.48),
    ]
    best_occ = -1.0
    best_score = -1.0
    tmp = dst.with_suffix(".try.png")
    for front, zoom in views:
        try:
            _capture(mesh, tmp, size, front, zoom)
        except Exception as exc:
            print(f"  FAIL capture {src.name}: {exc}", flush=True)
            continue
        occ = occupancy(tmp)
        score = float(np.exp(-((occ - 0.40) ** 2) / (2 * 0.16 ** 2)))
        if occ < 0.03:
            score *= 0.05
        if score > best_score:
            best_score = score
            best_occ = occ
            shutil.copy2(tmp, dst)
        if 0.18 <= occ <= 0.58:
            break
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    ok = dst.exists() and dst.stat().st_size > 2000 and best_occ >= 0.015
    print(f"  {'OK' if ok else 'WEAK'} {dst.name} occ={best_occ:.3f}", flush=True)
    return ok


def render_bunny_row() -> None:
    ref = ROOT / "DataBase" / "stanford-bunny.obj"
    jobs = [("ref", ref)]
    for method, lvl in (("Noise", "04"), ("Smoothing", "05"), ("Decimation", "05"), ("Combined", "04")):
        p = ROOT / "DataBase" / "distorted" / "stanford-bunny" / method / f"stanford-bunny_{method}_{lvl}.obj"
        jobs.append((method.lower(), p))
    for key, path in jobs:
        out = OUT / "bunny" / f"{key}.png"
        print(f"bunny {key}", flush=True)
        render_path(path, out, size=720)


def should_rerender(slug: str, out: Path) -> bool:
    if slug in FORCE_SLUGS:
        return True
    if not out.exists() or out.stat().st_size < 2000:
        return True
    return occupancy(out) < 0.045


def render_catalog(only: set[str] | None = None) -> None:
    with CAT.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        slug = r["slug"]
        if only and slug not in only:
            continue
        out = OUT / "refs" / f"{slug}.png"
        if not only and not should_rerender(slug, out):
            print(f"skip {slug} occ={occupancy(out):.3f}", flush=True)
            continue
        src = preview_source(slug, ROOT / r["path"])
        if not src.exists():
            print(f"missing {src}", flush=True)
            continue
        print(f"ref {slug} {src.stat().st_size/1e6:.1f} MB", flush=True)
        render_path(src, out, size=560)


def _legend(fig, text: str, y: float = 0.015) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color=MUTED)


def _box(ax, xy, w, h, text, fc=CARD, ec=NAVY, fs=10, tc=INK):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.15,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def fig_architecture() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    _box(ax, (0.25, 1.35), 2.35, 2.0, "Эталоны\n50 моделей\n6 классов", fc="#E4E8EF", fs=13)
    _box(ax, (3.05, 1.35), 2.55, 2.0, "Оператор $T_\\theta$\n4 типа", fc="#EDE3D8", fs=13)
    _box(ax, (6.05, 1.35), 2.35, 2.0, "Лестница\n$L=10$", fc="#E3EBE3", fs=13)
    _box(ax, (8.85, 2.45), 4.05, 1.35, "Корпус  $50\\times 4\\times 10=2000$", fc="#E8D5D0", fs=14)
    _box(ax, (8.85, 0.75), 4.05, 1.35, "MOS $\\subset$ факториала", fc="#E8E1C8", fs=14)
    for x0, x1, y in [(2.6, 3.05, 2.35), (5.6, 6.05, 2.35)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.annotate("", xy=(8.85, 3.1), xytext=(8.4, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("", xy=(8.85, 1.4), xytext=(8.4, 2.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: эталон — неискажённый меш; оператор T — processing, не bitstream; MOS — подмножество пар, не второй корпус.")
    fig.savefig(OUT / "fig_architecture.png")
    plt.close()


def fig_principle() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 3.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    _box(ax, (0.3, 0.9), 2.7, 2.0, "$M_{\\mathrm{ref}}$", fc="#E4E8EF", fs=22)
    _box(ax, (3.5, 0.9), 3.3, 2.0, "$M' = T_\\theta(M_{\\mathrm{ref}})$", fc="#EDE3D8", fs=18)
    _box(ax, (7.3, 0.9), 2.6, 2.0, "$(M_{\\mathrm{ref}}, M')$", fc="#E3EBE3", fs=18)
    _box(ax, (10.4, 0.9), 2.5, 2.0, "PLCC, SROCC", fc="#E8D5D0", fs=16)
    for x0, x1 in [(3.0, 3.5), (6.8, 7.3), (9.9, 10.4)]:
        ax.annotate("", xy=(x1, 1.9), xytext=(x0, 1.9), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: M_ref — эталон; T_θ — оператор с параметром θ; PLCC / SROCC — корреляция метрики q с MOS s.")
    fig.savefig(OUT / "fig_principle.png")
    plt.close()


def fig_taxonomy() -> None:
    _style()
    labels = [
        "Контроль дискретизации",
        "Канонические эталоны",
        "Артефакты оцифровки",
        "Тонкие структуры",
        "Жёсткие рёбра",
        "Перцептивное маскирование",
    ]
    sizes = [6, 9, 8, 6, 8, 13]
    colors = ["#C5D0DE", "#D9C8B4", "#C9D6B8", "#BFD0C8", "#C8C4D8", "#DCC6C6"]
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, sizes, color=colors, edgecolor=INK, linewidth=0.6, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Эталоны")
    ax.set_xlim(0, 16)
    ax.invert_yaxis()
    ax.axvline(50 / 6, color=ACCENT, ls="--", lw=1.0, label="равномерно ≈8.3")
    for yi, s in zip(y, sizes):
        ax.text(s + 0.15, yi, str(s), va="center", fontsize=13, color=NAVY, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(
        fig,
        "Легенда: столбец — число эталонов класса. Квоты сдвинуты под функциональный провал метрики, не под семантику быта/органики. Сумма = 50.",
    )
    fig.savefig(OUT / "fig_taxonomy.png")
    plt.close()


def fig_coverage() -> None:
    _style()
    datasets = [
        "LIRIS Masking",
        "LIRIS/EPFL",
        "CMDM",
        "Nehmé TOG'23",
        "SJTU-TMQA",
        "TSMD",
        "BASICS",
        "MATE-3D",
        "PLER-HQ",
    ]
    props = [
        "FR mesh",
        "Шум",
        "Smooth",
        "LOD",
        "Гибрид",
        "Дискр.",
        "Сканы",
        "10 ступ.",
        "50 этал.",
        "геом.",
    ]
    M = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0.5, 0, 0.5, 0.5, 1, 0],
            [1, 1, 0, 1, 0.5, 0, 0.5, 0.5, 0, 0],
            [1, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    ax.imshow(M, cmap="copper_r", vmin=0, vmax=1.15, aspect="auto")
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props, rotation=28, ha="right", fontsize=10)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            mark = "●" if val == 1 else ("◐" if val == 0.5 else "")
            ax.text(j, i, mark, ha="center", va="center", fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(
        fig,
        "Легенда: ● полное покрытие · ◐ частично · пусто — нет.  LOD — QEM как прокси геометрии кодека, не bitstream.  Дискр. — аналитические сетки.  геом. — без текстуры/QP.",
    )
    fig.savefig(OUT / "fig_coverage.png")
    plt.close()


def fig_nref() -> None:
    _style()
    names = ["LIRIS", "CMDM", "SJTU-TMQA", "TSMD", "Nehmé", "PLER-HQ"]
    vals = [4, 5, 21, 42, 55, 50]
    colors = [MUTED, MUTED, MUTED, MUTED, MUTED, ACCENT]
    fig, ax = plt.subplots(figsize=(12.4, 4.2))
    bars = ax.bar(names, vals, color=colors, edgecolor=INK, linewidth=0.5, width=0.62)
    ax.set_ylabel("Эталоны")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(fig, "Легенда: эталоны — число исходных мешей корпуса. Терракота — PLER-HQ.")
    fig.savefig(OUT / "fig_nref.png")
    plt.close()


def fig_operators() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    cards = [
        (0.2, "Шум", r"$\Delta\mathbf{x}\sim\mathcal{N}(0,\sigma_{\mathrm{rel}}\,D)$" + "\n" + r"ступ. 5: $\sigma=0.007$"),
        (3.45, "Сглаж.", r"Laplacian $\times k$" + "\n" + r"ступ. 5: $k=5$"),
        (6.7, "LOD", r"QEM, доля $r$" + "\n" + r"ступ. 5: $r=0.15$"),
        (9.95, "Гибрид", r"$T_{\mathrm{шум}}\!\circ T_{\mathrm{LOD}}$" + "\n" + r"та же ступень"),
    ]
    fcs = ["#E4E8EF", "#E3EBE3", "#EDE3D8", "#E8D5D0"]
    for (x, title, body), fc in zip(cards, fcs):
        _box(ax, (x, 0.45), 3.05, 3.9, "", fc=fc, fs=1)
        ax.text(x + 1.52, 3.55, title, ha="center", va="center", fontsize=16, color=NAVY, fontweight="bold")
        ax.text(x + 1.52, 1.85, body, ha="center", va="center", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(fig, "Легенда: D — диагональ AABB.  LOD / QEM — прокси геометрии кодека (не QP/bitstream).  Гибрид — композиция на одном индексе ступени.")
    fig.savefig(OUT / "fig_operators.png")
    plt.close()


def fig_scale() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    items = [
        (0.25, "50", "эталонов", "ряд Nehmé 55 / TSMD 42\n6 функциональных классов"),
        (4.55, "4", "оператора", "processing, не codec-mix\nось PLER-2.0"),
        (8.85, "10", "ступеней", "CMDM: 4 ступени грубо\nпсихометрика"),
    ]
    for x, num, sub, why in items:
        _box(ax, (x, 0.4), 4.05, 3.85, "", fc=CARD, fs=1)
        ax.text(x + 2.02, 3.15, num, ha="center", fontsize=48, color=ACCENT, fontweight="bold")
        ax.text(x + 2.02, 2.25, sub, ha="center", fontsize=16, color=NAVY)
        ax.text(x + 2.02, 1.15, why, ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, "Легенда: 50×4×10 = 2000 стимулов. Ступень — индекс 1…10 параметра оператора, не LOD-файл движка.")
    fig.savefig(OUT / "fig_scale.png")
    plt.close()


def fig_ladder() -> None:
    _style()
    cols = [str(i) for i in range(1, 11)]
    rows = ["Шум  σ", "Сглаж.  k", "LOD  r", "Гибрид"]
    data = [
        ["0.0005", "0.001", "0.002", "0.004", "0.007", "0.010", "0.015", "0.022", "0.032", "0.050"],
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        ["0.80", "0.50", "0.35", "0.25", "0.15", "0.10", "0.05", "0.025", "0.01", "0.005"],
        ["∘ той же", "", "", "", "", "", "", "", "", ""],
    ]
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    cw, rh = 0.92, 0.85
    x0, y0 = 2.15, 3.55
    ax.text(1.05, y0 + 0.55, "ступ.", ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")
    for j, c in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cw, y0 + 0.55, c, ha="center", va="center", fontsize=12, color=NAVY, fontweight="bold")
    heat = np.linspace(0.12, 0.92, 10)
    row_fc = ["#E4E8EF", "#E3EBE3", "#EDE3D8", "#E8D5D0"]
    for i, (lab, vals, fc) in enumerate(zip(rows, data, row_fc)):
        y = y0 - (i + 1) * rh
        _box(ax, (0.15, y), 1.9, rh - 0.08, lab, fc=fc, fs=10)
        for j, val in enumerate(vals):
            alpha = 0.18 + 0.72 * heat[j]
            cell = FancyBboxPatch(
                (x0 + j * cw + 0.04, y + 0.06),
                cw - 0.08,
                rh - 0.18,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=ACCENT,
                edgecolor="none",
                alpha=alpha if i < 3 else 0.12,
            )
            ax.add_patch(cell)
            if val:
                ax.text(
                    x0 + (j + 0.5) * cw,
                    y + rh * 0.45,
                    val,
                    ha="center",
                    va="center",
                    fontsize=9 if i != 0 else 8,
                    color=INK,
                    fontweight="bold",
                )
    ax.text(
        6.0,
        0.42,
        r"композиция на одном индексе ступени;  D — диагональ AABB",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, "Легенда: ступ. — индекс 1…10.  σ — доля D.  k — итерации Laplacian.  r — доля граней QEM (LOD, прокси геометрии кодека).  ∘ — композиция шум∘LOD.")
    fig.savefig(OUT / "fig_ladder.png")
    plt.close()


def fig_compare() -> None:
    _style()
    rows = [
        ["Корпус", "Эталоны", "Стимулы", "Ступени", "Ось"],
        ["LIRIS Mask.", "4", "26", "нет", "шум"],
        ["LIRIS/EPFL", "4", "88", "нет", "шум / сглаж."],
        ["CMDM", "5", "80", "4", "цвет + геом."],
        ["SJTU-TMQA", "21", "945", "смесь", "текстура / QP"],
        ["TSMD", "42", "210", "кодек", "AoM"],
        ["Nehmé '23", "55", "343k*", "кодек", "текстур. меш"],
        ["PLER-HQ", "50", "2000", "10", "только геом."],
    ]
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    widths = [2.6, 1.7, 2.0, 1.8, 3.6]
    x0, y0 = 0.7, 7.15
    for j, (w, h) in enumerate(zip(widths, rows[0])):
        x = x0 + sum(widths[:j])
        ax.text(x + w / 2, y0, h, ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")
    for i, row in enumerate(rows[1:]):
        y = y0 - (i + 1) * 0.92
        fc = "#E8D5D0" if row[0] == "PLER-HQ" else CARD
        _box(ax, (x0 - 0.15, y - 0.38), 12.0, 0.82, "", fc=fc, fs=1)
        for j, (w, val) in enumerate(zip(widths, row)):
            x = x0 + sum(widths[:j])
            weight = "bold" if row[0] == "PLER-HQ" or j == 0 else "normal"
            col = ACCENT if row[0] == "PLER-HQ" and j > 0 else INK
            ax.text(x + w / 2, y, val, ha="center", va="center", fontsize=13, color=col, fontweight=weight)
    ax.text(6.6, 0.48, "* 3000 MOS, остальное pseudo-MOS", ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, "Легенда: эталоны — исходные меши; стимулы — искажённые; ступени — градации оператора.  LOD/QEM в PLER-HQ — прокси геометрии кодека, не bitstream.")
    fig.savefig(OUT / "fig_compare.png")
    plt.close()


def fig_factorial() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.2))
    methods = ["Шум", "Сглаживание", "LOD / QEM", "Гибрид"]
    counts = [500, 500, 500, 500]
    colors = ["#C5D0DE", "#C9D6B8", "#D9C8B4", "#DCC6C6"]
    bars = ax.bar(methods, counts, color=colors, edgecolor=INK, linewidth=0.6, width=0.58)
    ax.set_ylabel("Меши")
    ax.set_ylim(0, 640)
    ax.axhline(500, color=ACCENT, lw=0.8, ls="--", label="500 на оператор")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 16, "500", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("50 × 4 × 10 = 2000", fontsize=16, color=NAVY, pad=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, "Легенда: LOD / QEM — прокси геометрии кодека.  Гибрид = шум ∘ LOD на той же ступени, не полная сетка (r × σ).")
    fig.savefig(OUT / "fig_factorial.png")
    plt.close()


def fig_proscons() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    rows = [
        ("LIRIS", "+ masking, чистый шум", "−  4 рефа, 1 тип"),
        ("CMDM", "+ DSIS, vertex color", "−  L=4,  5 рефов"),
        ("Nehmé '23", "+ N=55, CS-MOS", "−  tex / codec mix"),
        ("TSMD", "+ N=42, AoM realism", "−  только compression"),
        ("PLER-HQ", "+ geom-only, L=10, N=50", "−  нет текстуры / codec"),
    ]
    for i, (name, plus, minus) in enumerate(rows):
        y = 4.85 - i * 1.05
        fc = "#E8D5D0" if name == "PLER-HQ" else CARD
        _box(ax, (0.25, y), 2.5, 0.9, name, fc=fc, fs=14)
        _box(ax, (3.0, y), 4.7, 0.9, plus, fc="#E3EBE3", fs=13)
        _box(ax, (7.95, y), 4.95, 0.9, minus, fc="#EDE3D8", fs=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: «+» — что корпус даёт; «−» — граница. Минус PLER-HQ (нет текстуры/bitstream) — спецификация оси, не пробел.")
    fig.savefig(OUT / "fig_proscons.png")
    plt.close()


def fig_notthis() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    items = [
        (0.25, r"$\notin$ texture / QP", "Nehmé · TMQA · TSMD"),
        (3.5, r"$\notin$ point cloud", "BASICS · LS-PCQA"),
        (6.75, r"$\notin$ T23D", "MATE-3D · нет $M_{ref}$"),
        (10.0, r"$\notin$ 4D / NR", "TDMD · no-reference"),
    ]
    fcs = ["#EDE3D8", "#E4E8EF", "#E8D5D0", "#E3EBE3"]
    for (x, t, sub), fc in zip(items, fcs):
        _box(ax, (x, 0.7), 3.0, 3.2, "", fc=fc, fs=1)
        ax.text(x + 1.5, 2.55, t, ha="center", fontsize=16, color=NAVY, fontweight="bold")
        ax.text(x + 1.5, 1.45, sub, ha="center", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: ∉ — вне спецификации PLER-HQ. T23D — text-to-3D без эталона обработки. NR — no-reference.")
    fig.savefig(OUT / "fig_notthis.png")
    plt.close()


def fig_hybrid() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.0))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    _box(ax, (0.3, 1.0), 2.4, 2.0, r"$M_{\mathrm{ref}}$", fc="#E4E8EF", fs=20)
    _box(ax, (3.3, 1.0), 2.6, 2.0, r"$T_{\mathrm{d}}(\,r_l\,)$", fc="#EDE3D8", fs=18)
    _box(ax, (6.5, 1.0), 2.6, 2.0, r"$T_{\mathrm{n}}(\,\sigma_l\,)$", fc="#E3EBE3", fs=18)
    _box(ax, (9.7, 1.0), 3.15, 2.0, r"$T_{\mathrm{n}}\circ T_{\mathrm{d}}$", fc="#E8D5D0", fs=18)
    for x0, x1 in [(2.7, 3.3), (5.9, 6.5), (9.1, 9.7)]:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(6.6, 0.35, r"один индекс $l$  ·  не независимая сетка $(r,\sigma)$", ha="center", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, "Легенда: LOD (QEM, доля r) затем шум (σ той же ступени). Не декартово произведение r×σ.")
    fig.savefig(OUT / "fig_hybrid.png")
    plt.close()


def fig_metrics() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    cards = [
        (0.25, "якорь", r"$q(M_{\mathrm{ref}},M_{\mathrm{ref}})$" + "\n" + r"$=\;1$  (или $0$ err.)"),
        (4.55, "PLCC", r"$\rho_{\mathrm{P}}=\mathrm{corr}(q,s)$"),
        (8.85, "SROCC", r"$\rho_{\mathrm{S}}=\mathrm{spearman}(q,s)$"),
    ]
    fcs = ["#E4E8EF", "#E3EBE3", "#E8D5D0"]
    for (x, title, body), fc in zip(cards, fcs):
        _box(ax, (x, 0.55), 4.05, 3.5, "", fc=fc, fs=1)
        ax.text(x + 2.02, 3.15, title, ha="center", fontsize=18, color=NAVY, fontweight="bold")
        ax.text(x + 2.02, 1.7, body, ha="center", fontsize=16, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: q — объективная метрика на паре (эталон, стимул); s — MOS. Якорь: метрика на (эталон, эталон).")
    fig.savefig(OUT / "fig_metrics.png")
    plt.close()


def fig_span() -> None:
    _style()
    stats = {r["slug"]: r for r in csv.DictReader((ROOT / "DataBase" / "metadata" / "mesh_stats.csv").open(encoding="utf-8-sig"))}
    picks = [
        ("cube", "Куб"),
        ("suzanne", "Suzanne"),
        ("stanford-bunny", "Bunny"),
        ("lucy", "Lucy"),
        ("stanford-dragon", "Dragon"),
        ("bust", "Бюст"),
        ("relief", "Рельеф"),
    ]
    faces = []
    names = []
    for slug, lab in picks:
        rec = stats.get(slug, {})
        try:
            faces.append(float(rec["n_faces"]))
        except (KeyError, ValueError):
            faces.append(1.0)
        names.append(lab)
    colors = [MUTED, MUTED, MUTED, MUTED, MUTED, ACCENT, ACCENT]
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    y = np.arange(len(names))
    ax.barh(y, faces, color=colors, edgecolor=INK, linewidth=0.5, height=0.62, log=True)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Грани")
    ax.invert_yaxis()
    ax.axvline(1e2, color="#D7D0C4", lw=0.6, ls=":")
    ax.axvline(1e6, color=ACCENT, lw=0.8, ls="--", label="10^6 граней")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(fig, "Легенда: грани — |F| эталона. Пунктир 10^6 — зона, где FR-метрики дорогие. Терракота — музейные сканы постановщика.")
    fig.savefig(OUT / "fig_span.png")
    plt.close()


def fig_licenses() -> None:
    _style()
    rows = list(csv.DictReader(CAT.open(encoding="utf-8-sig")))
    order = ["personal", "project", "literature", "open"]
    labels = ["личные", "проект", "литература", "CC / синтез"]
    vals = [sum(1 for r in rows if r.get("provenance") == k) for k in order]
    colors = ["#DCC6C6", "#C8C4D8", "#C5D0DE", "#C9D6B8"]
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor=INK, linewidth=0.5, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Эталоны")
    ax.set_xlim(0, 28)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v + 0.25, yi, str(v), va="center", fontsize=13, color=NAVY, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(fig, "Легенда значков на сетках: ● личные · ◇ проект · ▲ литература · ■ CC/синтез. Сумма = 50.")
    fig.savefig(OUT / "fig_licenses.png")
    plt.close()


def fig_mos_nest() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    _box(ax, (0.4, 0.55), 7.4, 3.5, "", fc="#E4E8EF", fs=1)
    ax.text(4.1, 3.45, r"корпус  $50\times4\times10=2000$", ha="center", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(4.1, 2.85, "объективные FR-метрики", ha="center", fontsize=13, color=MUTED)
    _box(ax, (1.7, 0.85), 4.8, 1.7, r"MOS $\subset$ факториала", fc="#E8E1C8", fs=16)
    _box(ax, (8.3, 0.55), 4.5, 3.5, "", fc="#E8D5D0", fs=1)
    ax.text(10.55, 2.85, "не замена", ha="center", fontsize=18, color=NAVY, fontweight="bold")
    ax.text(10.55, 1.85, "подмножество пар\nдля наблюдателей", ha="center", fontsize=13, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: корпус 2000 — для FR-метрик; MOS — выборка пар из того же факториала, не отдельный датасет.")
    fig.savefig(OUT / "fig_mos_nest.png")
    plt.close()


def fig_roles() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    items = [
        (0.2, 3.05, "Контроль дискретизации  ·  6", "аналитический профиль,\nполюса UV, тесселяция"),
        (4.55, 3.05, "Канонические эталоны  ·  9", "якоря Stanford / AIM@SHAPE\nсопоставимость с LIRIS"),
        (8.9, 3.05, "Артефакты оцифровки  ·  8", "отверстия, плотность,\nsensor residual"),
        (0.2, 0.45, "Тонкие структуры  ·  6", "пряди, спицы, стенки;\nQEM схлопывает"),
        (4.55, 0.45, "Жёсткие рёбра  ·  8", "G0 / hard-surface;\nLaplacian стирает crease"),
        (8.9, 0.45, "Перцептивное маскирование  ·  13", "лицо, identity, saliency;\nmasking искажений"),
    ]
    fcs = ["#E4E8EF", "#EDE3D8", "#E3EBE3", "#D8EDE8", "#E4E0F0", "#E8D5D0"]
    for (x, y, title, body), fc in zip(items, fcs):
        _box(ax, (x, y), 4.05, 2.35, "", fc=fc, fs=1)
        ax.text(x + 2.02, y + 1.7, title, ha="center", fontsize=12, color=NAVY, fontweight="bold")
        ax.text(x + 2.02, y + 0.75, body, ha="center", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend(fig, "Легенда: класс задаёт, какой провал метрики/LOD/сглаживания эталон обязан ловить. Не семантика «быт / органика».")
    fig.savefig(OUT / "fig_roles.png")
    plt.close()


def fig_stats() -> None:
    _style()
    cat = {r["slug"]: r for r in csv.DictReader(CAT.open(encoding="utf-8-sig"))}
    stats = list(csv.DictReader((ROOT / "DataBase" / "metadata" / "mesh_stats.csv").open(encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    # left: nonmanifold vs class
    order = ["sampling", "canonical", "acquisition", "thin_feature", "sharp_feature", "perceptual"]
    short = ["Дискр.", "Канон.", "Оцифр.", "Тонкие", "Рёбра", "Маска"]
    colors = ["#C5D0DE", "#D9C8B4", "#C9D6B8", "#BFD0C8", "#C8C4D8", "#DCC6C6"]
    vals = []
    for key in order:
        slugs = {s for s, r in cat.items() if r.get("category") == key}
        nums = []
        for row in stats:
            if row["slug"] not in slugs:
                continue
            try:
                nums.append(float(row["nonmanifold_frac"]))
            except (KeyError, ValueError):
                pass
        vals.append(float(np.median(nums)) if nums else 0.0)
    axes[0].bar(short, vals, color=colors, edgecolor=INK, linewidth=0.5, width=0.68)
    axes[0].set_ylabel("н/м  (медиана)")
    axes[0].set_ylim(0, 1.05)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    # right: curvature vs class
    cvals = []
    for key in order:
        slugs = {s for s, r in cat.items() if r.get("category") == key}
        nums = []
        for row in stats:
            if row["slug"] not in slugs:
                continue
            try:
                nums.append(float(row["mean_abs_curv"]))
            except (KeyError, ValueError):
                pass
        cvals.append(float(np.median(nums)) if nums else 0.0)
    axes[1].bar(short, cvals, color=colors, edgecolor=INK, linewidth=0.5, width=0.68)
    axes[1].set_ylabel("крив.  (медиана, рад)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _legend(
        fig,
        "Легенда: н/м — доля рёбер, инцидентных ≠2 граням.  крив. — средний двугранный угол (прокси средней кривизны).  Медиана по классу; файлы >180 МБ без н/м и крив.",
        y=0.01,
    )
    fig.savefig(OUT / "fig_stats.png")
    plt.close()


def fig_stats_table() -> None:
    _style()
    cat = {r["slug"]: r for r in csv.DictReader(CAT.open(encoding="utf-8-sig"))}
    stats = list(csv.DictReader((ROOT / "DataBase" / "metadata" / "mesh_stats.csv").open(encoding="utf-8-sig")))
    short_cls = {
        "sampling": "дискр.",
        "canonical": "канон.",
        "acquisition": "оцифр.",
        "thin_feature": "тонк.",
        "sharp_feature": "рёбра",
        "perceptual": "маска",
    }

    def fmt_n(val) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return "—"
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        if v >= 1000:
            return f"{v/1e3:.0f}k"
        return f"{v:.0f}"

    def fmt_g(row) -> str:
        g = row.get("genus", "")
        if g not in ("", None):
            return str(g)
        return "—"

    def fmt_nm(row) -> str:
        try:
            return f"{100 * float(row['nonmanifold_frac']):.0f}%"
        except (KeyError, TypeError, ValueError):
            return "—"

    def fmt_k(row) -> str:
        try:
            v = float(row["mean_abs_curv"])
        except (KeyError, TypeError, ValueError):
            return "—"
        if v < 1e-6:
            return "≈0"
        return f"{v:.2f}"

    def fmt_d(row) -> str:
        try:
            v = float(row["aabb_diag"])
        except (KeyError, TypeError, ValueError):
            return "—"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.2g}"
        return f"{v:.2g}"

    headers = ["модель", "кл.", "верш.", "гран.", "род", "н/м", "диам.", "крив."]
    body = []
    for row in stats:
        rec = cat.get(row["slug"], {})
        name = rec.get("name_ru", row["slug"]).split("(")[0].strip()
        if len(name) > 18:
            name = name[:17] + "…"
        body.append(
            [
                name,
                short_cls.get(rec.get("category", ""), ""),
                fmt_n(row.get("n_vertices")),
                fmt_n(row.get("n_faces")),
                fmt_g(row),
                fmt_nm(row),
                fmt_d(row),
                fmt_k(row),
            ]
        )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    mid = 25
    for ax, chunk in zip(axes, (body[:mid], body[mid:])):
        ax.axis("off")
        tbl = ax.table(cellText=chunk, colLabels=headers, loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.22)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D7D0C4")
            if r == 0:
                cell.set_facecolor("#243044")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#FFFcf7" if r % 2 else "#F4F1EA")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _legend(
        fig,
        "Легенда: кл. — функциональный класс.  верш./гран. — |V|, |F|.  род — для watertight, иначе —.  "
        "н/м — доля non-manifold рёбер.  диам. — диагональ AABB (ед. модели).  крив. — средний двугранный угол, рад.",
        y=0.01,
    )
    fig.savefig(OUT / "fig_stats_table.png")
    plt.close()


def all_diagrams() -> None:
    fig_architecture()
    fig_principle()
    fig_taxonomy()
    fig_coverage()
    fig_nref()
    fig_operators()
    fig_scale()
    fig_ladder()
    fig_compare()
    fig_factorial()
    fig_proscons()
    fig_notthis()
    fig_hybrid()
    fig_metrics()
    fig_span()
    fig_licenses()
    fig_mos_nest()
    fig_roles()
    fig_stats()
    fig_stats_table()
    mapping = {
        "fig_architecture.png": "fig_pler_hq_architecture.png",
        "fig_principle.png": "fig_pler_hq_principle.png",
        "fig_taxonomy.png": "fig_pler_hq_taxonomy.png",
        "fig_coverage.png": "fig_pler_hq_coverage.png",
        "fig_factorial.png": "fig_pler_hq_factorial.png",
    }
    docs = ROOT / "docs"
    for src_name, dst_name in mapping.items():
        src = OUT / src_name
        if src.exists():
            shutil.copy2(src, docs / dst_name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bunny").mkdir(exist_ok=True)
    (OUT / "refs").mkdir(exist_ok=True)
    args = sys.argv[1:]
    if "--diagrams" in args:
        print("diagrams", flush=True)
        all_diagrams()
        print("done diagrams", flush=True)
        return 0
    only = [a for a in args if not a.startswith("-")]
    do_all = not only
    if do_all:
        print("diagrams", flush=True)
        all_diagrams()
        print("bunny", flush=True)
        render_bunny_row()
        print("catalog", flush=True)
        render_catalog()
    else:
        print("catalog", only, flush=True)
        render_catalog(set(only))
    print("done assets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
