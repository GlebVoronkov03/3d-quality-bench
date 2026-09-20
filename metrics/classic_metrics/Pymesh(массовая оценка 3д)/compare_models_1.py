#!/usr/bin/env python3
"""
compare_gaussian_plot.py

Сравнение эталонной модели (mesh/reference.obj) с набором искажённых
моделей (mesh/distorted_*.obj).  Для каждой модели вычисляются подписанные
расстояния, метрики (mean, std, rms) и распределение отклонений → «% точек в
каждом бинe».  Сохраняются:

* отдельные PNG‑графики  <model>_distribution.png,
* один общий PNG          comparison_plot.png,
* CSV‑файл metrics.csv          (mean, std, rms),
* CSV‑файл distribution.csv    (по 2000 бинов для каждой модели).

Кроме того, после сохранения PNG‑файла открывается **интерактивное** окно
Matplotlib, где можно панорамировать и масштабировать график.
"""

# ------------------------------------------------------------
# Библиотеки
# ------------------------------------------------------------
from pathlib import Path
import csv
from typing import List, Dict, Tuple

import numpy as np
import open3d as o3d               # только для чтения OBJ‑файлов
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

# ------------------------------------------------------------
# Параметры (можно менять в начале скрипта)
# ------------------------------------------------------------
CONVERSION_FACTOR =  873.743993010048   # 873.743993010048 внутренние единицы → мм
NBINS = 523268                           # количество точек в KDE‑сети
MARGIN_FACTOR = 0.05                  # % от диапазона, добавляемый к краям сетки
SHOW_SCATTER = False                  # рисовать отдельные точки облака?
MODE = "percent_per_bin"               # "percent_per_bin" | "histogram" | "density_percent"

REFERENCE_PATH = Path("mesh") / "reference.obj"

OUTPUT_COMBINED = "comparison_plot.png"
METRICS_CSV = "metrics.csv"
DISTRIBUTION_CSV = "distribution.csv"


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
def load_obj_vertices(file_path: Path) -> np.ndarray:
    """Читает только строки, начинающиеся с ``v `` (координаты вершин)."""
    verts = []
    with file_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append([float(x), float(y), float(z)])
    return np.asarray(verts, dtype=np.float64)


def align_by_translation(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Транслирует src так, чтобы центр совпал с центром tgt."""
    return src + (tgt.mean(axis=0) - src.mean(axis=0))


# ------------------------------------------------------------
# Класс облака точек
# ------------------------------------------------------------
class PointCloud:
    """Простой класс облака точек, загруженный из OBJ‑файла."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.name = self.file_path.name                # напр. "distorted_3.obj"
        self.points: np.ndarray = load_obj_vertices(self.file_path)

        # После сравнения заполняются:
        self.signed_dist: np.ndarray | None = None      # подпсанные расстояния (внутр. ед.)
        self.nearest_idx: np.ndarray | None = None      # индексы ближайших точек эталона

    # ------------------------------------------------------------------
    def compare_to(self, reference: "PointCloud") -> None:
        """
        Находит ближайшую точку в эталоне и проставляет знак:
            -1 – если точка‑исскажённая ближе к центру эталона,
            +1 – иначе.
        """
        tree = KDTree(reference.points)
        dists, inds = tree.query(self.points)

        centroid = reference.points.mean(axis=0)
        rad_self = np.linalg.norm(self.points - centroid, axis=1)
        rad_ref = np.linalg.norm(reference.points[inds] - centroid, axis=1)

        signs = np.where(rad_self < rad_ref, -1.0, 1.0)
        self.signed_dist = signs * dists
        self.nearest_idx = inds

    # ------------------------------------------------------------------
    def to_mm(self) -> np.ndarray:
        """Подписанные расстояния → миллиметры."""
        if self.signed_dist is None:
            raise RuntimeError("Сначала вызовите compare_to().")
        return self.signed_dist * CONVERSION_FACTOR

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, float]:
        """Mean / Std / RMS (в мм)."""
        d = self.to_mm()
        return {
            "mean_mm": np.mean(d),
            "std_mm": np.std(d, ddof=1),
            "rms_mm": np.sqrt(np.mean(d ** 2)),
        }


# ------------------------------------------------------------
# Расчёт распределения → %‑в‑бинe (сумма ≈ 100 %)
# ------------------------------------------------------------
def distribution_percent(
    data_mm: np.ndarray,
    centers: np.ndarray,
    edges: np.ndarray,
    mode: str = "percent_per_bin",
) -> np.ndarray:
    """
    Возвращает массив «% точек в каждом бинe».

    Доступные режимы:
        * "percent_per_bin" – KDE → pdf·dx·100 (сумма ≈ 100 %);
        * "histogram"       – обычный гистограммный расчёт,
                              hist / total * 100;
        * "density_percent" – плотность в %/мм (не нормируется к 100 %).
    """
    if data_mm.size == 0:
        return np.zeros_like(centers)

    if mode == "histogram":
        hist, _ = np.histogram(data_mm, bins=edges, density=False)
        prob = hist.astype(float) / hist.sum() * 100.0
        return prob

    # ------------------- KDE -------------------
    try:
        kde = gaussian_kde(data_mm)
    except np.linalg.LinAlgError:                 # fallback, если данные «плохие»
        kde = gaussian_kde(data_mm, bw_method="silverman")

    pdf = kde(centers)                         # 1/мм
    dx = np.diff(edges).mean()
    if dx == 0:
        dx = 1.0                                 # защита от деления на 0

    if mode == "density_percent":
        # % на миллиметр, без «сумм‑= 100 %».
        prob = pdf * 100.0
        return prob

    # default – «% точек в каждом бинe» (сумма ≈ 100 %).
    prob = pdf * dx * 100.0                     # % точек в данном бине
    total = prob.sum()
    if not np.isclose(total, 100.0, rtol=1e-3):
        prob = prob / total * 100.0
    return prob


# ------------------------------------------------------------
# Построение одной кривой
# ------------------------------------------------------------
def plot_distribution(
    ax: plt.Axes,
    centers: np.ndarray,
    prob_percent: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Отрисовывает «% точек в бинe». """
    ax.plot(centers, prob_percent, color=color, linewidth=2, label=label)
    if SHOW_SCATTER:
        ax.scatter(centers, prob_percent, s=6, color=color, alpha=0.4)


# ------------------------------------------------------------
# Экспорт CSV‑файлов
# ------------------------------------------------------------
def export_metrics(metrics: List[Dict[str, float]], out_path: str = METRICS_CSV) -> None:
    """metrics.csv – model, mean_mm, std_mm, rms_mm (5 знаков)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mean_mm", "std_mm", "rms_mm"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "model": row["file"],
                    "mean_mm": f"{row['mean_mm']:.5f}",
                    "std_mm": f"{row['std_mm']:.5f}",
                    "rms_mm": f"{row['rms_mm']:.5f}",
                }
            )


def export_distribution(
    dist_dict: Dict[str, np.ndarray],
    centers: np.ndarray,
    out_path: str = DISTRIBUTION_CSV,
) -> None:
    """
    distribution.csv – model, distance_mm, probability_percent.
    Одна общая сетка `centers` используется для всех моделей.
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "distance_mm", "probability_percent"])
        for model, prob in dist_dict.items():
            for d, p in zip(centers, prob):
                writer.writerow([model, f"{d:.5f}", f"{p:.5f}"])


# ------------------------------------------------------------
# Обработка одной модели (выравнивание, сравнение, PNG, распределение)
# ------------------------------------------------------------
def process_one_model(
    distorted_path: Path,
    reference: PointCloud,
    centers: np.ndarray,
    edges: np.ndarray,
    color: str,
    mode: str = MODE,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Возвращает:
        • словарь метрик (mean, std, rms) + имя файла;
        • массив «% точек в каждом бинe» (для общего графика).

    Сохраняет отдельный PNG‑график <model>_distribution.png.
    """
    pc = PointCloud(distorted_path)

    # 1) Выравнивание (только трансляция)
    pc.points = align_by_translation(pc.points, reference.points)

    # 2) Сравнение → подпсанные расстояния
    pc.compare_to(reference)

    # 3) Метрики
    mets = pc.metrics()
    mets["file"] = pc.name

    # 4) Распределение (%‑в‑бине)
    prob_percent = distribution_percent(pc.to_mm(), centers, edges, mode=mode)

    # 5) Сохраняем отдельный PNG
    fig_i, ax_i = plt.subplots(figsize=(9, 5))
    ax_i.set_xlabel("Величина отклонения мм", fontsize=14, fontweight="bold")
    ax_i.set_ylabel("Вероятность отклонения %", fontsize=14, fontweight="bold")
    ax_i.grid(True, which="both", linestyle=":", linewidth=0.5)
    plot_distribution(ax_i, centers, prob_percent, color, pc.name)
    ax_i.set_title(
        f"Распределение отклонений: {pc.name}",
        fontsize=16,
        fontweight="bold",
    )
    ax_i.legend(loc="upper right")
    plt.tight_layout()

    out_png = f"{Path(pc.name).stem}_distribution.png"
    fig_i.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig_i)                     # экономим память

    return mets, prob_percent


# ------------------------------------------------------------
# Основная функция
# ------------------------------------------------------------
def main() -> None:
    # ------------------- 1. Эталон -------------------
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"Эталонный файл не найден: {REFERENCE_PATH}")
    reference_pc = PointCloud(REFERENCE_PATH)

    # ------------------- 2. Список всех distorted_*.obj -------------------
    distorted_paths = sorted(Path("mesh").glob("distorted_*.obj"))
    if not distorted_paths:
        raise FileNotFoundError("В папке mesh нет файлов distorted_*.obj")

    # ------------------- 3. Глобальная сетка -------------------
    # Сначала собираем все подпсанные расстояния, чтобы определить диапазон
    all_dists_mm: List[np.ndarray] = []
    for p in distorted_paths:
        pc_tmp = PointCloud(p)
        pc_tmp.points = align_by_translation(pc_tmp.points, reference_pc.points)
        pc_tmp.compare_to(reference_pc)
        all_dists_mm.append(pc_tmp.to_mm())

    global_min = min(arr.min() for arr in all_dists_mm)
    global_max = max(arr.max() for arr in all_dists_mm)
    margin = (global_max - global_min) * MARGIN_FACTOR

    # edges – границы бинов, centers – середины (для KDE)
    edges = np.linspace(global_min - margin,
                        global_max + margin,
                        NBINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # ------------------- 4. Цветовая палитра -------------------
    cmap = plt.get_cmap("tab10")               # 10 разных цветов, потом повторяются
    colors = [cmap(i % 10) for i in range(len(distorted_paths))]

    # ------------------- 5. Окно для совмённого графика -------------------
    fig_comb, ax_comb = plt.subplots(figsize=(10, 6))
    ax_comb.set_xlabel("Величина отклонения мм", fontsize=14, fontweight="bold")
    ax_comb.set_ylabel("Вероятность отклонения %", fontsize=14, fontweight="bold")
    ax_comb.grid(True, which="both", linestyle=":", linewidth=0.5)

    # ------------------- 6. Списки для CSV -------------------
    metrics_table: List[Dict[str, float]] = []        # → metrics.csv
    distribution_dict: Dict[str, np.ndarray] = {}     # → distribution.csv

    # ------------------- 7. Обрабатываем каждую модель -------------------
    for p, col in zip(distorted_paths, colors):
        mets, prob_percent = process_one_model(
            distorted_path=p,
            reference=reference_pc,
            centers=centers,
            edges=edges,
            color=col,
            mode=MODE,
        )
        # сохраняем данные
        metrics_table.append(mets)
        distribution_dict[p.name] = prob_percent

        # добавляем кривую к совместному графику
        plot_distribution(ax_comb, centers, prob_percent, col, p.name)

    # ------------------- 8. Оформляем и сохраняем общий график -------------------
    ax_comb.legend(loc="upper right")
    ax_comb.set_title(
        "Распределение отклонений всех искажённых моделей",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_comb.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches="tight")

    # ------------------- 9. **Сначала** сохраняем CSV‑файлы -------------------
    export_metrics(metrics_table, METRICS_CSV)
    export_distribution(distribution_dict, centers, DISTRIBUTION_CSV)

    # ------------------- 10. ОТКРЫВАЕМ ИНТЕРАКТИВНОЕ ОКНО -------------------
    # После этого вызова появится окно, где можно панорамировать и масштабировать
    # график.  Файлы уже созданы, поэтому их наличие не зависит от закрытия окна.
    plt.show()               # <-- интерактивное окно

    # ------------------- 11. Выводим метрики в консоль (5 знаков) -------------------
    print("\n=== Таблица метрик (5 знаков после запятой) ===")
    hdr = f"{'Модель':<30} {'Mean (mm)':>14} {'Std (mm)':>14} {'RMS (mm)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for row in metrics_table:
        print(
            f"{row['file']:<30} "
            f"{row['mean_mm']:14.5f} "
            f"{row['std_mm']:14.5f} "
            f"{row['rms_mm']:14.5f}"
        )


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
