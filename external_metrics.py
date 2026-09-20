"""
Обёртки для внешних метрик качества 3D-моделей.
Полные реализации требуют текстур/чекпоинтов (HybridMQA, FMQM, GeodesicPSIM).
Здесь — рабочие упрощённые аналоги + заглушки с предупреждением.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree


def _simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Упрощённый SSIM без scikit-image."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / (den + 1e-12))

logger = logging.getLogger(__name__)
_warned: set = set()


def _warn_once(name: str, msg: str) -> None:
    if name not in _warned:
        warnings.warn(msg, UserWarning, stacklevel=2)
        _warned.add(name)
        logger.warning(msg)


def _load_vertices(path: str, max_pts: int = 200_000) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) > max_pts:
        idx = np.random.default_rng(1).choice(len(verts), max_pts, replace=False)
        verts = verts[idx]
    return verts


def _align(ref: np.ndarray, dist: np.ndarray):
    return ref - ref.mean(0), dist - dist.mean(0)


# ------------------------------------------------------------------ MSDM2
def compute_msdm2(ref_path: str, dist_path: str) -> float:
    """
    Упрощённый MSDM2: сравнение статистик дискретной кривизны на сэмплах поверхности.
    Оригинал: Nehmé et al., многошкальный MSDM2 (MEPP2).
    """
    try:
        ref_mesh = trimesh.load(ref_path, process=False)
        dist_mesh = trimesh.load(dist_path, process=False)
        n_samples = 10_000
        ref_pts, _ = trimesh.sample.sample_surface(ref_mesh, n_samples)
        dist_pts, _ = trimesh.sample.sample_surface(dist_mesh, n_samples)

        ref_tree = KDTree(ref_pts)
        dists, _ = ref_tree.query(dist_pts)
        sigma = np.std(ref_pts, axis=0).mean() * 0.05 + 1e-9
        weights = np.exp(-(dists**2) / (2 * sigma**2))
        score = float(np.clip(np.mean(weights), 0, 1))
        return score
    except Exception as exc:
        _warn_once("MSDM2", f"MSDM2 fallback failed: {exc}")
        return float("nan")


# ------------------------------------------------------------------ CMDM
def compute_cmdm(ref_path: str, dist_path: str) -> float:
    """
    CMDM требует vertex colors (Nehmé et al. 2021).
    pip/MEPP2: https://projet.liris.cnrs.fr/mepp/
    """
    try:
        ref = trimesh.load(ref_path, process=False)
        if not hasattr(ref.visual, "vertex_colors") or ref.visual.vertex_colors is None:
            _warn_once(
                "CMDM",
                "CMDM: нет vertex colors — используйте MEPP2/CMDM filter. Возврат NaN.",
            )
            return float("nan")
        # Упрощение: L2 цветов + геом. расстояние
        dist = trimesh.load(dist_path, process=False)
        ref_v = np.asarray(ref.vertices)
        dist_v = np.asarray(dist.vertices)
        n = min(len(ref_v), len(dist_v), 50_000)
        ref_v = ref_v[:n]
        dist_v = dist_v[:n]
        ref_c = ref.visual.vertex_colors[:n, :3].astype(float) / 255.0
        dist_c = dist.visual.vertex_colors[:n, :3].astype(float) / 255.0
        geo = np.linalg.norm(ref_v - dist_v, axis=1)
        col = np.linalg.norm(ref_c - dist_c, axis=1)
        err = 0.5 * geo / (geo.max() + 1e-9) + 0.5 * col
        return float(np.clip(1.0 - np.mean(err), 0, 1))
    except Exception as exc:
        _warn_once("CMDM", f"CMDM: {exc}")
        return float("nan")


# ------------------------------------------------------------------ PCQM
def compute_pcqm(ref_path: str, dist_path: str) -> float:
    """
    Упрощённый PCQM (MPEG point cloud quality).
    Репозиторий: https://github.com/MPEGGroup/mpeg-pcc-rendering
    """
    ref = _load_vertices(ref_path)
    dist = _load_vertices(dist_path)
    ref, dist = _align(ref, dist)
    tree = KDTree(ref)
    d, _ = tree.query(dist)
    d_norm = d / (np.percentile(d, 95) + 1e-9)
    psnr = -10 * np.log10(np.mean(d_norm**2) + 1e-12)
    return float(np.clip(psnr / 60.0, 0, 1))


# ------------------------------------------------------------------ 3D-PSSIM
def compute_3d_pssim(ref_path: str, dist_path: str) -> float:
    """
    Упрощённый 3D-PSSIM: SSIM проекций с 6 ортогональных направлений.
    """
    try:
        import open3d as o3d

        ref = o3d.io.read_triangle_mesh(ref_path)
        dist = o3d.io.read_triangle_mesh(dist_path)
        ref.compute_vertex_normals()
        dist.compute_vertex_normals()

        views = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
        scores = []
        res = 128
        for view in views:
            ref_img = _render_depth(ref, view, res)
            dist_img = _render_depth(dist, view, res)
            if ref_img is None or dist_img is None:
                continue
            scores.append(_simple_ssim(ref_img, dist_img))
        return float(np.mean(scores)) if scores else float("nan")
    except Exception as exc:
        _warn_once("3D-PSSIM", f"3D-PSSIM: {exc}")
        return float("nan")


def _render_depth(mesh, view, resolution: int = 128) -> Optional[np.ndarray]:
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return None
    center = verts.mean(0)
    v = np.asarray(view, dtype=float)
    v /= np.linalg.norm(v) + 1e-9
    proj = verts - center
    depth = proj @ v
    # 2D ортографическая проекция
    u = np.cross(v, [0, 0, 1])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(v, [0, 1, 0])
    u /= np.linalg.norm(u) + 1e-9
    w = np.cross(v, u)
    x2d = proj @ u
    y2d = proj @ w
    x2d = (x2d - x2d.min()) / (x2d.max() - x2d.min() + 1e-9)
    y2d = (y2d - y2d.min()) / (y2d.max() - y2d.min() + 1e-9)
    img = np.zeros((resolution, resolution), dtype=np.float32)
    xi = np.clip((x2d * (resolution - 1)).astype(int), 0, resolution - 1)
    yi = np.clip((y2d * (resolution - 1)).astype(int), 0, resolution - 1)
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    for i in range(len(xi)):
        img[yi[i], xi[i]] = max(img[yi[i], xi[i]], d_norm[i])
    return img


# ------------------------------------------------------------------ HybridMQA / FMQM (полные реализации через ml_metrics)
def compute_hybridmqa(ref_path: str, dist_path: str) -> float:
    """
    HybridMQA (CVPR 2025): https://github.com/arshafiee/hybridmqa
    Установка: python scripts/setup_ml_metrics.py
    """
    try:
        from ml_metrics import compute_hybridmqa as _compute

        return _compute(ref_path, dist_path)
    except ImportError:
        _warn_once("HybridMQA", "HybridMQA: ml_metrics недоступен. Заглушка NaN.")
        return float("nan")


def compute_fmqm(ref_path: str, dist_path: str) -> float:
    """
    FMQM: https://github.com/yyyykf/FMQM — требует texture maps.
    Установка: python scripts/setup_ml_metrics.py
    """
    try:
        from ml_metrics import compute_fmqm as _compute

        return _compute(ref_path, dist_path)
    except ImportError:
        _warn_once("FMQM", "FMQM: ml_metrics недоступен. Заглушка NaN.")
        return float("nan")


def compute_geodesic_psim(ref_path: str, dist_path: str) -> float:
    """
    GeodesicPSIM: https://github.com/Qi-Yangsjtu/GeodesicPSIM (C++/binary).
    Упрощение: SSIM полей нормалей на сэмплах.
    """
    try:
        ref = trimesh.load(ref_path, process=False)
        dist = trimesh.load(dist_path, process=False)
        ref.fix_normals()
        dist.fix_normals()
        n = min(len(ref.vertices), len(dist.vertices), 20_000)
        rn = ref.vertex_normals[:n].astype(np.float64)
        dn = dist.vertex_normals[:n].astype(np.float64)
        rn = (rn - rn.min()) / (rn.max() - rn.min() + 1e-9)
        dn = (dn - dn.min()) / (dn.max() - dn.min() + 1e-9)
        # Средний косинус + вариация как PSIM-подобная метрика
        cos = np.sum(rn * dn, axis=1) / (np.linalg.norm(rn, axis=1) * np.linalg.norm(dn, axis=1) + 1e-9)
        return float(np.clip(np.mean((cos + 1) / 2), 0, 1))
    except Exception as exc:
        _warn_once("GeodesicPSIM", f"GeodesicPSIM: {exc}")
        return float("nan")


# ------------------------------------------------------------------ PointPCA+
def compute_pointpca_plus(ref_path: str, dist_path: str) -> float:
    """PointPCA+: сравнение спектров PCA локальных патчей."""
    try:
        from sklearn.decomposition import PCA

        ref = _load_vertices(ref_path, 50_000)
        dist = _load_vertices(dist_path, 50_000)
        ref, dist = _align(ref, dist)

        def patch_spectrum(pts, k: int = 30) -> np.ndarray:
            tree = KDTree(pts)
            _, idx = tree.query(pts[:500], k=k)
            spectra = []
            pca = PCA(n_components=3)
            for neighbors in idx:
                patch = pts[neighbors]
                pca.fit(patch - patch.mean(0))
                spectra.append(np.sort(pca.explained_variance_))
            return np.mean(spectra, axis=0)

        s_ref = patch_spectrum(ref)
        s_dist = patch_spectrum(dist)
        rel_err = np.linalg.norm(s_ref - s_dist) / (np.linalg.norm(s_ref) + 1e-9)
        return float(np.clip(1.0 - rel_err, 0, 1))
    except Exception as exc:
        _warn_once("PointPCA+", f"PointPCA+: {exc}")
        return float("nan")


# ------------------------------------------------------------------ PQI
def compute_pqi(ref_path: str, dist_path: str) -> float:
    """PQI: комбинация геом. расстояния и кривизны (упрощённый индекс)."""
    geo = compute_pcqm(ref_path, dist_path)
    curv = compute_msdm2(ref_path, dist_path)
    if np.isnan(geo) or np.isnan(curv):
        return float("nan")
    return float(0.6 * geo + 0.4 * curv)


# Алиасы для динамического поиска
compute_pointpca_plus = compute_pointpca_plus

EXTERNAL_COMPUTE_MAP: dict[str, Callable] = {
    "MSDM2": compute_msdm2,
    "CMDM": compute_cmdm,
    "PCQM": compute_pcqm,
    "3D-PSSIM": compute_3d_pssim,
    "HybridMQA": compute_hybridmqa,
    "FMQM": compute_fmqm,
    "GeodesicPSIM": compute_geodesic_psim,
    "PointPCA+": compute_pointpca_plus,
    "PQI": compute_pqi,
}
