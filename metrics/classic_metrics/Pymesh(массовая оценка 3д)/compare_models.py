#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_models.py
Переработанный скрипт сравнения эталонной 3D модели с набором искажённых моделей.
Поддерживает:
    • загрузку .obj файлов (только вершины);
    • выравнивание (центр‑в‑центр + optional ICP);
    • расчёт расстояний ближайших соседей (KD Tree);
    • набор метрик (mean, RMS, max, 95й перцентиль);
    • построение гистограмм распределения ошибок;
    • визуализацию отклонений (цветовая карта);
    • экспорт результатов в CSV.

Требуемые пакеты:
    numpy, scipy, matplotlib, open3d, pandas (для CSV), tqdm (для прогресс‑бара).
"""
import os
import glob
import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import open3d as o3d
from scipy.spatial import KDTree

# Перевод «модельных» единиц в миллиметры 
SCALE_FACTOR = 873.743993010048   # через аргументы командной строки

# Порог для ICP (в тех же единицах, что и координаты модели). Подбирайте под ваш набор.
ICP_THRESHOLD = 0.02 * SCALE_FACTOR   # 2 % от единицы измерения, обычно хватает

def load_obj_vertices(filepath: str) -> np.ndarray:
    """
    Считывает только вершины ('v') из .obj файла и возвращает массив shape (N, 3).
    Игнорируются любые строки, не начинающиеся с 'v ' (например, 'vn', 'vt', 'f' …).

    Parameters
    ----------
    filepath : str
        Путь к .obj файлу.

    Returns
    -------
    np.ndarray
        Дробные координаты всех вершин.
    """
    verts = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                # Формат: v X Y Z
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in {filepath}")
    return np.asarray(verts, dtype=np.float64)


def center_align(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выравнивает массивы точек по их центрам масс (центр‑в‑центр).

    Parameters
    ----------
    src : np.ndarray
        Массив точек «источника» (N*3).
    tgt : np.ndarray
        Массив точек «цели» (M*3).

    Returns
    -------
    src_aligned, tgt_aligned : Tuple[np.ndarray, np.ndarray]
        Точки после переноса центров в начало координат.
    """
    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)
    return src - src_center, tgt - tgt_center


def icp_align(source: np.ndarray,
              target: np.ndarray,
              threshold: float = ICP_THRESHOLD,
              max_iter: int = 50,
              verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Проводит точечную регистрацию ICP (point to point) с помощью Open3D.
    Возвращает выровненный массив source и трансформацию 4*4.

    Parameters
    ----------
    source : np.ndarray
        Точки, которые будем трансформировать (N*3).
    target : np.ndarray
        Точки‑референс (M*3).
    threshold : float
        Максимальное расстояние соответствия (по умолчанию 2% SCALE_FACTOR).
    max_iter : int
        Максимальное число итераций ICP.
    verbose : bool
        Вывести подробный лог о сходимости.

    Returns
    -------
    aligned_source : np.ndarray
        Точки source после применения найденного преобразования.
    transformation : np.ndarray
        Матрица 4*4 (R|t).
    """
    src_o3d = o3d.geometry.PointCloud()
    tgt_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(source)
    tgt_o3d.points = o3d.utility.Vector3dVector(target)

    init_trans = np.eye(4)

    reg = o3d.pipelines.registration.registration_icp(
        src_o3d,
        tgt_o3d,
        threshold,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter))

    if verbose:
        print(f"[ICP] fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.6f}")

    aligned = np.asarray(src_o3d.transform(reg.transformation).points)
    return aligned, reg.transformation


def compute_nearest_distances(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Находит для каждой точки src её ближайший сосед в tgt.
    Возвращает массив расстояний и индексы соседних точек.

    Parameters
    ----------
    src : np.ndarray
        Точки источника (N*3).
    tgt : np.ndarray
        Точки цели (M*3).

    Returns
    -------
    distances : np.ndarray
        Расстояния (N,).
    indices : np.ndarray
        Индексы ближайших точек в tgt (N,).
    """
    tree = KDTree(tgt)
    distances, indices = tree.query(src)
    return distances, indices


def distance_metrics(distances: np.ndarray,
                    scale: float = SCALE_FACTOR) -> Dict[str, float]:
    """
    Вычисляет набор статистических показателей по массиву расстояний.

    Parameters
    ----------
    distances : np.ndarray
        Сырые расстояния в «модельных» единицах.
    scale : float
        Коэффициент перевода в миллиметры (по умолчанию ваш 873.743…).

    Returns
    -------
    dict
        mean_mm, rms_mm, max_abs_mm, p95_abs_mm, std_mm.
    """
    d_mm = distances * scale
    mean = np.mean(d_mm)
    rms = np.sqrt(np.mean(d_mm ** 2))
    max_abs = np.max(np.abs(d_mm))
    p95 = np.percentile(np.abs(d_mm), 95)
    std = np.std(d_mm)

    return {
        "mean_mm": mean,
        "rms_mm": rms,
        "max_abs_mm": max_abs,
        "p95_abs_mm": p95,
        "std_mm": std,
    }


def plot_distance_histogram(distances: np.ndarray,
                           model_name: str,
                           scale: float = SCALE_FACTOR,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Строит гистограмму распределения отклонений (в мм).

    Parameters
    ----------
    distances : np.ndarray
        Сырые расстояния.
    model_name : str
        Подпись модели (будет в заголовке).
    scale : float
        Коэффициент перевода.
    ax : matplotlib.axes.Axes, optional
        Если передан используется вместо создания нового.

    Returns
    -------
    matplotlib.axes.Axes
    """
    d_mm = distances * scale
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_mm, bins=120, color="#3498db", edgecolor="#1f618d", alpha=0.75)
    ax.set_xlabel("Отклонение, мм", fontsize=12)
    ax.set_ylabel("Количество точек", fontsize=12)
    ax.set_title(f"Распределение отклонений: {model_name}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def visualize_error_cloud(reference: np.ndarray,
                         target: np.ndarray,
                         distances: np.ndarray,
                         scale: float = SCALE_FACTOR,
                         cmap_name: str = "jet") -> None:
    """
    Визуализирует две облака точек в Open3D:
        • эталон красный;
        • искажённый окрашен в соответствии с модулем отклонения (цветовая карта).

    Параметры
    ----------
    reference : np.ndarray
        Точки эталона (N*3).
    target : np.ndarray
        Точки сравниваемой модели (M*3). Должны быть уже выровнены!
    distances : np.ndarray
        Расстояния от каждой точки target к её ближайшему соседу в reference.
    scale : float
        Коэффициент перевода в мм (используется только для подписи цвета, не меняет geometry).
    cmap_name : str
        Имя цветовой карты matplotlib (по умолчанию "jet").
    """
    # Цвета по модулю отклонения
    d_mm = np.abs(distances) * scale
    norm = plt.Normalize(vmin=np.min(d_mm), vmax=np.max(d_mm))
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(d_mm))[:, :3]   # отбрасываем альфа‑канал

    # эталон (красный)
    ref_o3d = o3d.geometry.PointCloud()
    ref_o3d.points = o3d.utility.Vector3dVector(reference)
    ref_o3d.paint_uniform_color([1.0, 0.0, 0.0])   # ярко‑красный

    # модель с цветом ошибки
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(target)
    tgt_o3d.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [ref_o3d, tgt_o3d],
        window_name="Сравнение (цвет = отклонение)",
        width=800,
        height=600,
        left=50,
        top=50,
    )


class PointCloudModel:
    """
    Хранит имя и массив точек из .obj файла.
    При создании сразу переводит координаты в np.ndarray.
    """

    def __init__(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        self.filepath = Path(filepath).resolve()
        self.name = self.filepath.stem   # без расширения
        self.points = load_obj_vertices(str(self.filepath))

    def __repr__(self):
        return f"<PointCloudModel name={self.name} points={len(self.points)}>"


class ModelComparator:
    """
    Сравнивает один эталон с набором целевых моделей.
    Осуществляет:
        • (необязательно) ICP выравнивание;
        • расчёт расстояний ближайших соседей;
        • подсчёт метрик;
        • построение гистограмм;
        • визуализацию ошибок;
        • сохранение результатов в CSV.
    """

    def __init__(self,
                 reference: PointCloudModel,
                 scale_factor: float = SCALE_FACTOR,
                 icp: bool = False,
                 icp_threshold: float = ICP_THRESHOLD,
                 verbose: bool = False):
        """
        Параметры
        ----------
        reference : PointCloudModel
            Эталонная модель.
        scale_factor : float
            Коэффициент перевода в мм (по‑умолчанию ваш 873.743…).
        icp : bool
            Выполнять ли ICP регистрацию (по умолчанию False только центр‑в‑центр).
        icp_threshold : float
            Порог соответствия для ICP (в тех же единицах, что и координаты после
            масштабирования). Ставьте больше, если модели сильно различаются.
        verbose : bool
            Печатать лог‑сообщения.
        """
        self.ref = reference
        self.scale = scale_factor
        self.do_icp = icp
        self.icp_thresh = icp_threshold
        self.verbose = verbose

        # Приводим эталон к «центр‑в‑центр», а потом (опционально) к ICP‑позиции.
        self.ref_aligned = self._prepare_reference(self.ref.points)

    def _prepare_reference(self, points: np.ndarray) -> np.ndarray:
        """Центр‑в‑центр + (если включено) ICP привязка к себе (ничего не меняет)."""
        # На случай, если пользователь захочет сначала выровнять эталон к или через другие модели.
        # Пока просто центр‑в‑центр.
        aligned, _ = center_align(points, points)  # возвращает оригинал, но с нулевым центром
        return aligned

    def _align_target(self, tgt_points: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Выравнивает целевую модель относительно эталона.
        Возвращает:
            - выровненные точки,
            - (опционально) матрицу трансформации (None, если icp=False).
        """
        # 1) центр‑в‑центр (обязательно)
        src_centered, tgt_centered = center_align(tgt_points, self.ref.points)

        if self.do_icp:
            # 2) ICP (опционально)
            aligned, trans = icp_align(src_centered,
                                      tgt_centered,
                                      threshold=self.icp_thresh,
                                      max_iter=50,
                                      verbose=self.verbose)
            return aligned, trans
        else:
            return src_centered, None

    def compare(self, target: PointCloudModel) -> Dict[str, any]:
        """
        Сравнивает целевую модель с эталоном и возвращает словарь с результатами.

        Параметры
        ----------
        target : PointCloudModel
            Модель, которую сравниваем.

        Returns
        -------
        dict
            {
                'target_name': str,
                'distances': np.ndarray (raw, без масштабирования),
                'metrics': dict (mean_mm, rms_mm, max_abs_mm, p95_abs_mm, std_mm),
                'transformation': np.ndarray|None,
                'aligned_points': np.ndarray,
            }
        """
        if self.verbose:
            print(f"\n[START] Сравнение {target.name} ↔ {self.ref.name}")

        aligned_tgt, trans = self._align_target(target.points)

        # Расстояния ближайшего соседа: от каждой точки target к эталону
        dists, inds = compute_nearest_distances(aligned_tgt, self.ref.points)

        # Метрики
        metrics = distance_metrics(dists, scale=self.scale)

        if self.verbose:
            print(f"[RESULT] {target.name}: mean={metrics['mean_mm']:.3f} mm, "
                  f"RMS={metrics['rms_mm']:.3f} mm, max={metrics['max_abs_mm']:.3f} mm")

        result = {
            "target_name": target.name,
            "distances": dists,
            "metrics": metrics,
            "transformation": trans,
            "aligned_points": aligned_tgt,
        }
        return result

    def plot_all_histograms(self,
                            results: List[Dict],
                            save_path: Optional[str] = None) -> None:
        """
        Строит один общий график с гистограммами отклонений всех моделей.

        Parameters
        ----------
        results : list of dict
            Список, полученный из `compare`.
        save_path : str, optional
            Если указан – сохраняет график в файл PNG.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in results:
            name = r["target_name"]
            dists = r["distances"]
            plot_distance_histogram(dists, name, scale=self.scale, ax=ax)
        ax.set_xlabel("Отклонение, мм", fontsize=13)
        ax.set_ylabel("Количество точек", fontsize=13)
        ax.set_title("Распределения отклонений для всех моделей", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            if self.verbose:
                print(f"[INFO] Гистограмма сохранена в {save_path}")
        else:
            plt.show()

    def visualize_all(self,
                      results: List[Dict],
                      rows: int = 2,
                      cols: int = 2) -> None:
        """
        Визуализует ошибки каждой модели в отдельном окне Open3D.
        Делает это последовательно; пользователь может закрывать окно, после чего откроется следующее.

        Parameters
        ----------
        results : list of dict
            Список, полученный из `compare`.
        rows, cols : int
            Параметры сетки визуализации (незадействовано сейчас, оставлено для будущего,
            когда понадобится отрисовка в Matplotlib).
        """
        for r in results:
            name = r["target_name"]
            distances = r["distances"]
            aligned_pts = r["aligned_points"]
            print(f"\n[VISUAL] {name} – отклонения ({len(distances)} точек)")
            visualize_error_cloud(self.ref.points,
                                 aligned_pts,
                                 distances,
                                 scale=self.scale,
                                 cmap_name="jet")

    def export_to_csv(self,
                      results: List[Dict],
                      csv_path: str = "comparison_results.csv") -> None:
        """
        Сохраняет таблицу со статистикой в CSV.

        Parameters
        ----------
        results : list of dict
            Список, полученный из `compare`.
        csv_path : str
            Путь к файлу CSV.
        """
        rows = []
        for r in results:
            line = {
                "model": r["target_name"],
                **r["metrics"]
            }
            rows.append(line)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format="%.6f")
        if self.verbose:
            print(f"[INFO] Таблица результатов сохранена в {csv_path}")

def main():
    """
    Пример полного рабочего цикла:
        1. Задаём путь к эталону.
        2. Находим 10 искажённых моделей в указанных папках (pattern `distorted_*.obj`).
        3. Сравниваем каждый файл с эталоном.
        4. Строим гистограмму распределений отклонений.
        5. Визуализируем облака ошибок (поочерёдно).
        6. Сохраняем метрики в CSV.
    """

    # Путь к эталону
    reference_path = r"mesh\reference.obj"
    reference_model = PointCloudModel(reference_path)

    # Список искажённых моделей
    # Ожидается, что файлы называются, например:
    #   distorted_01.obj, distorted_02.obj, ..., distorted_10.obj
    pattern = r"mesh\distorted_*.obj"
    distorted_paths = sorted(glob.glob(pattern))

    if not distorted_paths:
        print("[ERROR] Не найдено файлов по шаблону:", pattern)
        sys.exit(1)

    # Инициализируем сравнение
    # Параметр icp=False, т.к. у вас модели уже почти выровнены.
    # При необходимости можете включить icp=True.
    comparator = ModelComparator(reference=reference_model,
                                 scale_factor=SCALE_FACTOR,
                                 icp=False,
                                 verbose=True)

    # Сравниваем всё подряд
    all_results = []
    for p in tqdm(distorted_paths, desc="Сравнение моделей"):
        target_model = PointCloudModel(p)
        res = comparator.compare(target_model)
        all_results.append(res)

    # Гистограмма распределений
    comparator.plot_all_histograms(all_results,
                                   save_path="all_histograms.png")

    # Визуализация ошибок (поочерёдно)
    # Если вам не нужна интерактивная визуализация – закомментируйте.
    comparator.visualize_all(all_results)

    # Экспорт CSV 
    comparator.export_to_csv(all_results,
                             csv_path="comparison_results.csv")

    print("\n[DONE] Все этапы завершены!")


if __name__ == "__main__":
    # Если хотите запускать сразу без дополнительных параметров:
    #   python compare_models.py
    # То просто вызовите main().
    # При желании можно добавить argparse для гибкой настройки.
    main()
