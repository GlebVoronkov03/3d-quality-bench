"""
Встроенные метрики: PLER, геометрические и статистические (из compare_metrics.py).
"""

from __future__ import annotations

import logging
import sys
import time
import contextlib
import io
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# --- PLER modules -----------------------------------------------------------
PLER_DIR = Path(__file__).resolve().parent / "metrics" / "PLER" / "_Metric_PLER+"
if str(PLER_DIR) not in sys.path:
    sys.path.insert(0, str(PLER_DIR))

try:
    from pler_metric import PLERMetric
    from topology_analyzer import TopologyAnalyzer

    _PLER = PLERMetric()
    _TOPO = TopologyAnalyzer()
    PLER_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("PLER модули недоступны: %s", exc)
    _PLER = None
    _TOPO = None
    PLER_AVAILABLE = False

SCALE_FACTOR = 873.743993010048
F_SCORE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1]
MAX_KDTREE_POINTS = 500_000  # субсэмплинг для больших моделей


# ------------------------------------------------------------------ loaders
def load_obj_vertices(filepath: str) -> np.ndarray:
    verts = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"Нет вершин в {filepath}")
    return np.asarray(verts, dtype=np.float64)


def _subsample(points: np.ndarray, max_points: int = MAX_KDTREE_POINTS) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
    return points[idx]


def center_align(src: np.ndarray, tgt: np.ndarray):
    return src - src.mean(axis=0), tgt - tgt.mean(axis=0)


def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray):
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)
    d1, _ = tree2.query(pc1)
    d2, _ = tree1.query(pc2)
    return d1.max(), d2.max(), max(d1.max(), d2.max())


def f_score(pc1: np.ndarray, pc2: np.ndarray, threshold: float) -> float:
    tree2 = KDTree(pc2)
    d1, _ = tree2.query(pc1)
    tree1 = KDTree(pc1)
    d2, _ = tree1.query(pc2)
    precision = np.mean(d1 <= threshold)
    recall = np.mean(d2 <= threshold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def distance_stats(src: np.ndarray, tgt: np.ndarray, scale: float = SCALE_FACTOR) -> Dict[str, float]:
    tree = KDTree(tgt)
    dists, _ = tree.query(src)
    d_mm = dists * scale
    return {
        "mean_mm": float(np.mean(d_mm)),
        "rms_mm": float(np.sqrt(np.mean(d_mm**2))),
        "max_abs_mm": float(np.max(np.abs(d_mm))),
        "p95_abs_mm": float(np.percentile(np.abs(d_mm), 95)),
        "std_mm": float(np.std(d_mm)),
    }


def compute_vertex_aad(ref_path: str, dist_path: str) -> float:
    """AAD: среднее арифметическое отклонение вершин (MCM)."""
    ref = _subsample(load_obj_vertices(ref_path))
    dist = _subsample(load_obj_vertices(dist_path))
    ref_c, _ = center_align(ref, ref)
    dist_c, _ = center_align(dist, ref)
    tree = KDTree(ref_c)
    dists, _ = tree.query(dist_c)
    return float(np.mean(dists))


# ----------------------------------------------------------- PLER wrappers
def _safe_pler_compute(ref_path: str, dist_path: str):
    """Вызов PLER без вывода emoji в консоль Windows."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _PLER.compute_pler(ref_path, dist_path)


def compute_pler_geometry(ref_path: str, dist_path: str) -> float:
    if not PLER_AVAILABLE:
        return float("nan")
    t0 = time.time()
    res = _safe_pler_compute(ref_path, dist_path)
    logger.info("PLER geometry: %.2f dB за %.2f с", res.pler_db, time.time() - t0)
    return float(res.pler_db)


def compute_pler_2_complex(ref_path: str, dist_path: str) -> float:
    if not PLER_AVAILABLE:
        return float("nan")
    t0 = time.time()
    res = _safe_pler_compute(ref_path, dist_path)
    topo = _TOPO.compare_topology(ref_path, dist_path) if _TOPO else 0.0
    pler_norm = min(res.pler_db / 60.0, 1.0)
    combined = 0.7 * pler_norm + 0.3 * topo
    logger.info("PLER-2.0 complex: %.4f за %.2f с", combined, time.time() - t0)
    return float(combined)


def compute_tsi(ref_path: str, dist_path: str) -> float:
    """TSI: Topology Similarity Index."""
    if not PLER_AVAILABLE or _TOPO is None:
        return float("nan")
    return float(_TOPO.compare_topology(ref_path, dist_path))


def compute_mse(ref_path: str, dist_path: str) -> float:
    if not PLER_AVAILABLE:
        return float("nan")
    res = _safe_pler_compute(ref_path, dist_path)
    return float(res.mse)


def compute_aad(ref_path: str, dist_path: str) -> float:
    return compute_vertex_aad(ref_path, dist_path)


# ------------------------------------------------------ classic geometric
def compute_classic_metrics(ref_path: str, dist_path: str) -> Dict[str, float]:
    """Пакет стандартных метрик расстояния (мм) и F-score."""
    t0 = time.time()
    ref_pts = _subsample(load_obj_vertices(ref_path))
    dist_pts = _subsample(load_obj_vertices(dist_path))
    ref_c, _ = center_align(ref_pts, ref_pts)
    dist_c, _ = center_align(dist_pts, ref_pts)

    stats_s2r = distance_stats(dist_c, ref_c)
    tree_src = KDTree(dist_c)
    dist_r2s, _ = tree_src.query(ref_c)
    mean_r2s = float(np.mean(dist_r2s) * SCALE_FACTOR)

    chamfer = (stats_s2r["mean_mm"] + mean_r2s) / 2.0
    _, _, hausdorff = hausdorff_distance(dist_c, ref_c)
    hausdorff_mm = float(hausdorff * SCALE_FACTOR)

    out = {
        "Chamfer (mm)": chamfer,
        "Hausdorff (mm)": hausdorff_mm,
        "Mean (mm)": stats_s2r["mean_mm"],
        "Max (mm)": stats_s2r["max_abs_mm"],
        "RMS (mm)": stats_s2r["rms_mm"],
        "P95 (mm)": stats_s2r["p95_abs_mm"],
        "Std (mm)": stats_s2r["std_mm"],
    }
    for th in F_SCORE_THRESHOLDS:
        out[f"F-score {th}"] = f_score(dist_c, ref_c, th)

    logger.info("Classic metrics за %.2f с", time.time() - t0)
    return out


# ----------------------------------------------------------- metric registry
METRIC_DIRECTION = {
    "PLER-2.0 (complex)": "higher",
    "PLER geometry": "higher",
    "TSI": "higher",
    "MSE": "lower",
    "AAD": "lower",
    "Chamfer (mm)": "lower",
    "Hausdorff (mm)": "lower",
    "Mean (mm)": "lower",
    "Max (mm)": "lower",
    "RMS (mm)": "lower",
    "P95 (mm)": "lower",
    "Std (mm)": "lower",
    "F-score 0.01": "higher",
    "F-score 0.02": "higher",
    "F-score 0.05": "higher",
    "F-score 0.1": "higher",
    "MSDM2": "higher",
    "CMDM": "higher",
    "PCQM": "higher",
    "3D-PSSIM": "higher",
    "HybridMQA": "higher",
    "FMQM": "higher",
    "GeodesicPSIM": "higher",
    "PointPCA+": "higher",
    "PQI": "higher",
}

# Теоретические границы (min, max) для нормировки 0..100
METRIC_THEORETICAL_BOUNDS: Dict[str, tuple[float, float]] = {
    "PLER-2.0 (complex)": (0.0, 1.0),
    "PLER geometry": (0.0, 100.0),       # dB
    "TSI": (0.0, 1.0),
    "MSE": (0.0, 1.0),
    "AAD": (0.0, 1.0),                   # нормализованное пространство модели
    "Chamfer (mm)": (0.0, 1000.0),
    "Hausdorff (mm)": (0.0, 1000.0),
    "Mean (mm)": (0.0, 1000.0),
    "Max (mm)": (0.0, 1000.0),
    "RMS (mm)": (0.0, 1000.0),
    "P95 (mm)": (0.0, 1000.0),
    "Std (mm)": (0.0, 500.0),
    "F-score 0.01": (0.0, 1.0),
    "F-score 0.02": (0.0, 1.0),
    "F-score 0.05": (0.0, 1.0),
    "F-score 0.1": (0.0, 1.0),
    "MSDM2": (0.0, 1.0),
    "CMDM": (0.0, 1.0),
    "PCQM": (0.0, 1.0),
    "3D-PSSIM": (0.0, 1.0),
    "HybridMQA": (0.0, 1.0),
    "FMQM": (0.0, 1.0),
    "GeodesicPSIM": (0.0, 1.0),
    "PointPCA+": (0.0, 1.0),
    "PQI": (0.0, 1.0),
}

ALL_METRICS = list(METRIC_DIRECTION.keys())


# re-export для обратной совместимости
from obj_helpers import count_obj_vertices, get_model_weight_mb  # noqa: E402


def compute_selected_metrics(
    ref_path: str,
    dist_path: str,
    selected: list,
    external_module=None,
    progress_cb=None,
) -> Dict[str, float]:
    """Последовательный расчёт выбранных метрик для пары моделей."""
    results: Dict[str, float] = {}

    def _notify(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    pler_metrics = {"PLER-2.0 (complex)", "PLER geometry", "TSI", "MSE", "AAD"}
    need_pler = bool(set(selected) & pler_metrics)
    need_classic = bool(
        set(selected)
        & {
            "Chamfer (mm)",
            "Hausdorff (mm)",
            "Mean (mm)",
            "Max (mm)",
            "RMS (mm)",
            "P95 (mm)",
            "Std (mm)",
            "F-score 0.01",
            "F-score 0.02",
            "F-score 0.05",
            "F-score 0.1",
        }
    )

    if need_pler:
        pler_result = None
        topo_score = None
        if PLER_AVAILABLE and (
            {"PLER-2.0 (complex)", "PLER geometry", "MSE", "AAD"} & set(selected)
        ):
            _notify("PLER…")
            pler_result = _safe_pler_compute(ref_path, dist_path)

        if _TOPO and ({"PLER-2.0 (complex)", "TSI"} & set(selected)):
            _notify("Топология (TSI)…")
            topo_score = float(_TOPO.compare_topology(ref_path, dist_path))

        if "PLER-2.0 (complex)" in selected:
            if pler_result is not None and topo_score is not None:
                pler_norm = min(pler_result.pler_db / 60.0, 1.0)
                results["PLER-2.0 (complex)"] = float(0.7 * pler_norm + 0.3 * topo_score)
            else:
                results["PLER-2.0 (complex)"] = float("nan")
        if "PLER geometry" in selected and pler_result:
            results["PLER geometry"] = float(pler_result.pler_db)
        if "TSI" in selected:
            results["TSI"] = topo_score if topo_score is not None else compute_tsi(ref_path, dist_path)
        if "MSE" in selected and pler_result:
            results["MSE"] = float(pler_result.mse)
        if "AAD" in selected:
            _notify("AAD…")
            results["AAD"] = compute_aad(ref_path, dist_path)

    if need_classic:
        _notify("Классические метрики…")
        classic = compute_classic_metrics(ref_path, dist_path)
        for k in selected:
            if k in classic:
                results[k] = classic[k]

    external_names = {
        "MSDM2",
        "CMDM",
        "PCQM",
        "3D-PSSIM",
        "HybridMQA",
        "FMQM",
        "GeodesicPSIM",
        "PointPCA+",
        "PQI",
    }
    if external_module is not None:
        ext_map = getattr(external_module, "EXTERNAL_COMPUTE_MAP", {})
        for name in selected:
            if name in external_names:
                fn = ext_map.get(name)
                if fn:
                    _notify(f"{name}…")
                    results[name] = float(fn(ref_path, dist_path))

    return results
