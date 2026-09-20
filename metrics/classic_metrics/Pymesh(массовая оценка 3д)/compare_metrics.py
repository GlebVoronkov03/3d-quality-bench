#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
import open3d as o3d

SCALE_FACTOR = 873.743993010048
ICP_THRESHOLD = 0.02 * SCALE_FACTOR
F_SCORE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1]

def load_obj_vertices(filepath: str) -> np.ndarray:
    verts = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in {filepath}")
    return np.asarray(verts, dtype=np.float64)

def center_align(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)
    return src - src_center, tgt - tgt_center

def icp_align(source: np.ndarray, target: np.ndarray,
              threshold: float = ICP_THRESHOLD,
              max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    src_o3d = o3d.geometry.PointCloud()
    tgt_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(source)
    tgt_o3d.points = o3d.utility.Vector3dVector(target)
    reg = o3d.pipelines.registration.registration_icp(
        src_o3d, tgt_o3d, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return np.asarray(src_o3d.transform(reg.transformation).points), reg.transformation

def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> Tuple[float, float, float]:
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)
    d1, _ = tree2.query(pc1)   # расстояния от pc1 к pc2
    d2, _ = tree1.query(pc2)   # расстояния от pc2 к pc1
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
        'mean_mm': np.mean(d_mm),
        'rms_mm': np.sqrt(np.mean(d_mm**2)),
        'max_abs_mm': np.max(np.abs(d_mm)),
        'p95_abs_mm': np.percentile(np.abs(d_mm), 95),
        'std_mm': np.std(d_mm),
    }

def plot_metrics_vs_points(results: List[Dict], output_file: str = 'metrics_vs_points.png'):
    """
    Строит графики зависимости основных метрик от количества точек
    в логарифмическом масштабе по оси X.
    """
    # Извлекаем данные
    n_points = np.array([r['n_points'] for r in results])
    chamfer = np.array([r['chamfer_mm'] for r in results])
    hausdorff = np.array([r['hausdorff_mm'] for r in results])
    mean_ref_to_src = np.array([r['mean_ref_to_src_mm'] for r in results])
    fscore_001 = np.array([r['fscore_0.01'] for r in results])

    # Сортируем по количеству точек (для красивых линий)
    sort_idx = np.argsort(n_points)
    n_points_sorted = n_points[sort_idx]
    chamfer_sorted = chamfer[sort_idx]
    hausdorff_sorted = hausdorff[sort_idx]
    mean_ref_sorted = mean_ref_to_src[sort_idx]
    fscore_sorted = fscore_001[sort_idx]

    # Создаём фигуру с 2x2 подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Зависимость метрик от количества точек (логарифмическая шкала)', fontsize=16)

    # 1) Chamfer Distance
    ax = axes[0, 0]
    ax.plot(n_points_sorted, chamfer_sorted, 'o-', color='blue', label='Chamfer')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Количество точек (log)')
    ax.set_ylabel('Chamfer Distance, мм')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title('Chamfer Distance')

    # 2) Hausdorff Distance
    ax = axes[0, 1]
    ax.plot(n_points_sorted, hausdorff_sorted, 'o-', color='red', label='Hausdorff')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Количество точек (log)')
    ax.set_ylabel('Hausdorff Distance, мм')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title('Hausdorff Distance')

    # 3) Среднее расстояние от эталона к искажённой
    ax = axes[1, 0]
    ax.plot(n_points_sorted, mean_ref_sorted, 'o-', color='green', label='mean_ref_to_src')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Количество точек (log)')
    ax.set_ylabel('Среднее расстояние (ref→src), мм')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title('Среднее расстояние (ref → src)')

    # 4) F-score при пороге 0.01
    ax = axes[1, 1]
    ax.plot(n_points_sorted, fscore_sorted, 'o-', color='purple', label='F-score (0.01)')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel('Количество точек (log)')
    ax.set_ylabel('F-score')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title('F-score (порог 0.01)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # чтобы заголовок не перекрывался
    plt.savefig(output_file, dpi=300)
    print(f"Графики сохранены в {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Сравнение 3D моделей с эталоном.')
    parser.add_argument('reference', type=str, help='Путь к эталонному OBJ')
    parser.add_argument('--distorted_pattern', type=str, default='mesh/distorted_*.obj',
                        help='Шаблон для искажённых OBJ (glob)')
    parser.add_argument('--icp', action='store_true', default=False,
                        help='Включить ICP (по умолчанию выключен)')
    parser.add_argument('--scale', type=float, default=SCALE_FACTOR,
                        help='Коэффициент перевода в мм')
    parser.add_argument('--f_thresholds', type=float, nargs='+',
                        default=F_SCORE_THRESHOLDS,
                        help='Пороги для F-score (в модельных единицах)')
    parser.add_argument('--output_csv', type=str, default='metrics.csv',
                        help='Имя выходного CSV-файла')
    parser.add_argument('--plot_hist', action='store_true',
                        help='Построить гистограмму распределения ошибок')
    parser.add_argument('--plot_metrics', action='store_true',
                        help='Построить графики зависимости метрик от количества точек')
    args = parser.parse_args()

    # Загрузка эталона
    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Эталон не найден: {ref_path}")
    ref_points = load_obj_vertices(str(ref_path))
    ref_centered, _ = center_align(ref_points, ref_points)
    ref_n = len(ref_centered)
    print(f"Эталон: {ref_path.name}, количество вершин: {ref_n}")

    # Список искажённых
    distorted_paths = sorted(glob.glob(args.distorted_pattern))
    if not distorted_paths:
        print(f"Не найдено файлов по шаблону: {args.distorted_pattern}")
        sys.exit(1)

    results = []
    all_dists_mm = []   # для гистограммы (односторонние src->ref)

    print(f"Сравнение {len(distorted_paths)} моделей с эталоном {ref_path.name} ...")

    for dpath in tqdm(distorted_paths):
        model_name = Path(dpath).stem
        points = load_obj_vertices(dpath)
        n_points = len(points)

        # Центрирование
        src_centered, tgt_centered = center_align(points, ref_points)
        if args.icp:
            src_aligned, _ = icp_align(src_centered, tgt_centered, threshold=ICP_THRESHOLD)
        else:
            src_aligned = src_centered

        # ---- Односторонние статистики (src -> ref) ----
        stats_src_to_ref = distance_stats(src_aligned, ref_centered, scale=args.scale)
        mean_src_to_ref = stats_src_to_ref['mean_mm']
        rms_src_to_ref = stats_src_to_ref['rms_mm']
        max_src_to_ref = stats_src_to_ref['max_abs_mm']
        p95_src_to_ref = stats_src_to_ref['p95_abs_mm']
        std_src_to_ref = stats_src_to_ref['std_mm']

        # ---- Расстояния от эталона к искажённой (ref -> src) ----
        tree_src = KDTree(src_aligned)
        dist_ref_to_src, _ = tree_src.query(ref_centered)
        mean_ref_to_src = np.mean(dist_ref_to_src) * args.scale

        # ---- Симметричный Chamfer (среднее из двух средних) ----
        chamfer_avg = (mean_src_to_ref + mean_ref_to_src) / 2.0

        # ---- Hausdorff (симметричный максимум) ----
        max_src_to_ref_h, max_ref_to_src_h, hausdorff_sym = hausdorff_distance(src_aligned, ref_centered)
        max_src_to_ref_h_mm = max_src_to_ref_h * args.scale
        max_ref_to_src_h_mm = max_ref_to_src_h * args.scale
        hausdorff_sym_mm = hausdorff_sym * args.scale

        # ---- F-score для заданных порогов ----
        fscores = {}
        for th in args.f_thresholds:
            fscores[f'fscore_{th}'] = f_score(src_aligned, ref_centered, th)

        # Формируем строку результата
        row = {
            'model': model_name,
            'n_points': n_points,
            'mean_src_to_ref_mm': mean_src_to_ref,
            'mean_ref_to_src_mm': mean_ref_to_src,
            'chamfer_mm': chamfer_avg,
            'hausdorff_mm': hausdorff_sym_mm,
            'max_src_to_ref_mm': max_src_to_ref,
            'max_ref_to_src_mm': max_ref_to_src_h_mm,
            'rms_src_to_ref_mm': rms_src_to_ref,
            'p95_src_to_ref_mm': p95_src_to_ref,
            'std_src_to_ref_mm': std_src_to_ref,
            **fscores,
        }
        results.append(row)

        # Сохраняем расстояния для гистограммы (src->ref в мм)
        tree_ref = KDTree(ref_centered)
        dists, _ = tree_ref.query(src_aligned)
        all_dists_mm.append(dists * args.scale)

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, float_format='%.6f')
    print(f"Результаты сохранены в {args.output_csv}")

    # Гистограмма (опционально)
    if args.plot_hist:
        plt.figure(figsize=(10, 6))
        for dists, row in zip(all_dists_mm, results):
            plt.hist(dists, bins=80, alpha=0.5, label=f"{row['model']} (n={row['n_points']})")
        plt.xlabel('Отклонение, мм (src → ref)')
        plt.ylabel('Количество точек')
        plt.title('Распределение односторонних отклонений')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('error_histogram.png', dpi=300)
        plt.show()

    # Графики зависимости метрик от количества точек (опционально)
    if args.plot_metrics:
        plot_metrics_vs_points(results)

    # Краткая сводка
    print("\n=== Сводка метрик (средние по моделям) ===")
    print(df.describe().round(4))

if __name__ == '__main__':
    main()