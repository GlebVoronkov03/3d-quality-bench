"""
Генерация деградированных 3D-моделей (decimation, noise, smoothing, combined).
Использует pymeshlab с fallback на open3d.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pymeshlab as pml
except ImportError:  # pragma: no cover
    pml = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

ProgressCallback = Optional[Callable[[str], None]]

# 10 градаций: геометрическая прогрессия от почти незаметного до явно неприемлемого.
# Субъективный эксперимент использует подмножество; полный корпус хранит все 10.
DEFAULT_DECIMATION_LEVELS = [0.80, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.025, 0.01, 0.005]
DEFAULT_NOISE_LEVELS = [0.0005, 0.001, 0.002, 0.004, 0.007, 0.010, 0.015, 0.022, 0.032, 0.050]
DEFAULT_SMOOTH_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

MIN_FACES = 4


class DegradationGenerator:
    """Генератор деградаций OBJ-моделей."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend

    def _use_pml(self) -> bool:
        if self.backend == "open3d":
            return False
        if self.backend == "pymeshlab":
            return pml is not None
        return pml is not None

    def _log(self, op: str, t0: float, path: str) -> None:
        logger.info("%s завершено за %.2f с -> %s", op, time.time() - t0, path)

    def _notify(self, cb: ProgressCallback, msg: str) -> None:
        logger.info(msg)
        if cb:
            cb(msg)

    def _count_faces(self, path: str) -> int:
        if self._use_pml():
            ms = pml.MeshSet()
            ms.load_new_mesh(path)
            return int(ms.current_mesh().face_number())
        if o3d is not None:
            mesh = o3d.io.read_triangle_mesh(path)
            return int(len(mesh.triangles))
        return 0

    def _bbox_diagonal(self, input_path: str) -> float:
        if self._use_pml():
            ms = pml.MeshSet()
            ms.load_new_mesh(input_path)
            bbox = ms.current_mesh().bounding_box()
            diag = bbox.diagonal()
            return float(diag) if diag > 0 else 1.0
        if o3d is not None:
            mesh = o3d.io.read_triangle_mesh(input_path)
            verts = np.asarray(mesh.vertices)
            if len(verts) == 0:
                return 1.0
            return float(np.linalg.norm(verts.max(0) - verts.min(0)))
        raise RuntimeError("Нужен pymeshlab или open3d для вычисления bbox")

    def _save_meshset(self, ms, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(output_path)

    def _ratio_to_target_faces(self, n_faces: int, ratio: float) -> int:
        """ratio — доля граней, которые нужно сохранить."""
        ratio = float(max(ratio, MIN_FACES / max(n_faces, 1)))
        ratio = float(min(ratio, 1.0))
        return max(MIN_FACES, int(round(n_faces * ratio)))

    def decimate(self, input_path: str, output_path: str, ratio: float) -> str:
        """Уменьшение числа полигонов: ratio = доля сохраняемых граней (0..1)."""
        t0 = time.time()
        n_faces_in = self._count_faces(input_path)
        target_faces = self._ratio_to_target_faces(n_faces_in, ratio)

        if self._use_pml():
            ms = pml.MeshSet()
            ms.load_new_mesh(input_path)
            n_faces = ms.current_mesh().face_number()
            if n_faces > MIN_FACES and target_faces < n_faces:
                # pymeshlab 2025: targetperc часто игнорируется — используем targetfacenum
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=target_faces,
                    preservenormal=True,
                    preservetopology=False,
                    autoclean=True,
                )
            n_faces_out = ms.current_mesh().face_number()
            if n_faces_out >= n_faces and target_faces < n_faces:
                logger.warning(
                    "pymeshlab decimation не изменил mesh (%d -> %d), fallback open3d",
                    n_faces,
                    n_faces_out,
                )
                if o3d is not None:
                    return self._decimate_open3d(input_path, output_path, target_faces, t0, ratio)
            self._save_meshset(ms, output_path)
            self._log(f"decimate(ratio={ratio}, {n_faces}->{n_faces_out})", t0, output_path)
            return output_path

        if o3d is not None:
            return self._decimate_open3d(input_path, output_path, target_faces, t0, ratio)

        raise RuntimeError("pymeshlab/open3d не установлены")

    def _decimate_open3d(
        self, input_path: str, output_path: str, target_faces: int, t0: float, ratio: float
    ) -> str:
        mesh = o3d.io.read_triangle_mesh(input_path)
        n_in = len(mesh.triangles)
        if n_in <= MIN_FACES:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(output_path, mesh)
            return output_path
        target_faces = min(target_faces, n_in - 1)
        simplified = mesh.simplify_quadric_decimation(target_faces)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, simplified)
        n_out = len(simplified.triangles)
        self._log(f"decimate(ratio={ratio}, open3d {n_in}->{n_out})", t0, output_path)
        return output_path

    def add_noise(self, input_path: str, output_path: str, sigma_relative: float) -> str:
        """Гауссовский шум вершин: sigma = sigma_relative * diag(bbox)."""
        t0 = time.time()
        sigma_relative = float(np.clip(sigma_relative, 0.0005, 0.05))
        sigma = sigma_relative * self._bbox_diagonal(input_path)

        if self._use_pml():
            try:
                ms = pml.MeshSet()
                ms.load_new_mesh(input_path)
                verts = ms.current_mesh().vertex_matrix().astype(np.float64)
                noise = np.random.default_rng(42).normal(0.0, sigma, verts.shape)
                mesh_obj = ms.current_mesh()
                if hasattr(mesh_obj, "set_vertex_matrix"):
                    mesh_obj.set_vertex_matrix(verts + noise)
                else:
                    raise AttributeError("pymeshlab: set_vertex_matrix недоступен")
                self._save_meshset(ms, output_path)
                self._log(f"noise(sigma_rel={sigma_relative})", t0, output_path)
                return output_path
            except (AttributeError, Exception) as exc:
                logger.warning("pymeshlab noise failed (%s), fallback open3d", exc)

        if o3d is not None:
            mesh = o3d.io.read_triangle_mesh(input_path)
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            noise = np.random.default_rng(42).normal(0.0, sigma, verts.shape)
            mesh.vertices = o3d.utility.Vector3dVector(verts + noise)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(output_path, mesh)
            self._log(f"noise(sigma_rel={sigma_relative}, open3d)", t0, output_path)
            return output_path

        raise RuntimeError("pymeshlab/open3d не установлены")

    def smooth(self, input_path: str, output_path: str, iterations: int) -> str:
        """Лапласово сглаживание (1..10 итераций)."""
        t0 = time.time()
        iterations = int(np.clip(iterations, 1, 10))

        if self._use_pml():
            ms = pml.MeshSet()
            ms.load_new_mesh(input_path)
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=iterations, cotangentweight=True)
            self._save_meshset(ms, output_path)
            self._log(f"smooth(iter={iterations})", t0, output_path)
            return output_path

        if o3d is not None:
            mesh = o3d.io.read_triangle_mesh(input_path)
            mesh = mesh.filter_smooth_laplacian(iterations)
            mesh.compute_vertex_normals()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(output_path, mesh)
            self._log(f"smooth(iter={iterations}, open3d)", t0, output_path)
            return output_path

        raise RuntimeError("pymeshlab/open3d не установлены")

    def combined(
        self,
        input_path: str,
        output_path: str,
        dec_ratio: float,
        noise_sigma: float,
    ) -> str:
        """Сначала decimate, затем шум."""
        tmp = Path(output_path).with_suffix(".tmp.obj")
        self.decimate(input_path, str(tmp), dec_ratio)
        self.add_noise(str(tmp), output_path, noise_sigma)
        if tmp.exists():
            tmp.unlink()
        return output_path

    def generate_series(
        self,
        reference_path: str,
        output_dir: str,
        method: str,
        levels: Sequence[float],
        extra_levels: Optional[Sequence[float]] = None,
        progress_cb: ProgressCallback = None,
    ) -> List[Tuple[str, float, str]]:
        """Генерирует серию деградаций. Возвращает (path, param, method)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ref_name = Path(reference_path).stem
        results: List[Tuple[str, float, str]] = []
        ref_faces = self._count_faces(reference_path)

        for i, level in enumerate(levels, start=1):
            fname = f"{ref_name}_{method}_{i:02d}.obj"
            dst = str(out / fname)
            self._notify(
                progress_cb,
                f"Деградация {i}/{len(levels)}: {method} param={level}",
            )
            if method == "Decimation":
                self.decimate(reference_path, dst, level)
                out_faces = self._count_faces(dst)
                self._notify(
                    progress_cb,
                    f"  -> {Path(dst).name}: {ref_faces} -> {out_faces} граней",
                )
            elif method == "Noise":
                self.add_noise(reference_path, dst, level)
            elif method == "Smoothing":
                self.smooth(reference_path, dst, int(level))
            elif method == "Combined":
                dec_ratio = level
                noise_sigma = (
                    extra_levels[i - 1]
                    if extra_levels and i - 1 < len(extra_levels)
                    else DEFAULT_NOISE_LEVELS[min(i - 1, len(DEFAULT_NOISE_LEVELS) - 1)]
                )
                self.combined(reference_path, dst, dec_ratio, noise_sigma)
            else:
                raise ValueError(f"Неизвестный метод: {method}")
            results.append((dst, float(level), method))

        return results


def parse_levels(text: str, default: Iterable[float]) -> List[float]:
    """Парсит список уровней из строки '0.5, 0.3, 0.1'."""
    if not text or not text.strip():
        return list(default)
    return [float(x.strip()) for x in text.split(",") if x.strip()]
