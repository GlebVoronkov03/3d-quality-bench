"""Научные PNG-схемы архитектуры корпуса PLER-HQ (для docs/ и вставки в DOCX)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 200,
        }
    )


def _box(ax, xy, w, h, text, fc="#f4f4f4", ec="#222222"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.1,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", wrap=True)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Архитектура корпуса PLER-HQ", loc="left", fontsize=13, pad=8)

    _box(ax, (0.3, 1.6), 2.2, 1.8, "Эталоны\n50 моделей\n6 классов", fc="#e8eef5")
    _box(ax, (3.0, 1.6), 2.4, 1.8, "Операторы $T_\\theta$\nшум, сглаживание,\nдецимация, гибрид", fc="#efe8e0")
    _box(ax, (5.9, 1.6), 2.2, 1.8, "10 градаций\nна тип\n$L=10$", fc="#e7efe4")
    _box(ax, (8.6, 2.7), 3.0, 1.4, "Полный корпус\n$50\\times4\\times10=2000$\nискажённых мешей", fc="#f3e8e8")
    _box(ax, (8.6, 0.7), 3.0, 1.4, "Подмножество\nдля MOS\n(по протоколу этики)", fc="#f7f1d8")

    for x0, x1, y in [(2.5, 3.0, 2.5), (5.4, 5.9, 2.5), (8.1, 8.6, 3.4)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="#222", lw=1.2))
    ax.annotate("", xy=(8.6, 1.4), xytext=(8.1, 2.2),
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.2))
    ax.text(6.0, 0.25, "Объективные FR-метрики (PLER-2.0 и др.) считаются на полном факториале.", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pler_hq_architecture.png")
    plt.close()


def fig_factorial() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    methods = ["Noise", "Smoothing", "Decimation", "Combined"]
    counts = [500, 500, 500, 500]
    colors = ["#4c6a92", "#6a8f71", "#b08968", "#8c5a5a"]
    bars = ax.bar(methods, counts, color=colors, width=0.62, edgecolor="#222", linewidth=0.6)
    ax.set_ylabel("Число искажённых мешей")
    ax.set_title("Полный факториал: 50 эталонов × 4 типа × 10 уровней = 2000")
    ax.set_ylim(0, 620)
    ax.axhline(500, color="#888", lw=0.6, ls="--")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 12, str(c), ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pler_hq_factorial.png")
    plt.close()


def fig_taxonomy() -> None:
    labels = [
        "Дискретизация (6)",
        "Канон (9)",
        "Оцифровка (8)",
        "Тонкие (6)",
        "Рёбра (8)",
        "Маскирование (13)",
    ]
    sizes = [6, 9, 8, 6, 8, 13]
    colors = ["#c5d4e8", "#d9cbb8", "#d8e3c8", "#cfe0d8", "#d4d0e3", "#e6d0d0"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct=lambda p: f"{p:.0f}%",
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    )
    for t in texts:
        t.set_fontsize(8)
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Функциональная таксономия 50 эталонов PLER-HQ")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pler_hq_taxonomy.png")
    plt.close()


def fig_coverage() -> None:
    """Матрица покрытия свойств: чужие корпуса vs PLER-HQ."""
    datasets = [
        "LIRIS Masking",
        "LIRIS/EPFL",
        "CMDM",
        "Nehmé TOG'23",
        "SJTU-TMQA",
        "TSMD",
        "BASICS",
        "HY3D-Bench",
        "MATE-3D",
        "PLER-HQ",
    ]
    props = [
        "FR mesh",
        "Геом. шум",
        "Сглаживание",
        "Децимация",
        "Гибрид геом.",
        "Примитивы",
        "Фотограмметрия",
        "≥10 уровней",
        "≥50 эталонов",
        "Планируемый MOS",
    ]
    # 1 = есть, 0.5 = частично, 0 = нет
    M = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0.5, 0, 0.5, 0.5, 1, 1],
            [1, 1, 0, 1, 0.5, 0, 0.5, 0.5, 0, 1],
            [1, 0, 0, 0, 0.5, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0.5, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_title("Покрытие свойств: существующие корпуса и PLER-HQ")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            mark = "●" if val == 1 else ("◐" if val == 0.5 else "·")
            ax.text(j, i, mark, ha="center", va="center", fontsize=8,
                    color="white" if val == 1 and i == len(datasets) - 1 else "#222")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pler_hq_coverage.png")
    plt.close()


def fig_principle() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Принцип: контролируемый FR-бенчмарк геометрии", loc="left", fontsize=13)

    _box(ax, (0.3, 1.2), 2.3, 1.6, "$M_{ref}$\nизвестная топология\nи масштаб", fc="#e8eef5")
    _box(ax, (3.2, 1.2), 2.6, 1.6, "$M_{dist}=T_\\theta(M_{ref})$\nпараметр $\\theta$ задан\nвоспроизводимо", fc="#efe8e0")
    _box(ax, (6.4, 1.2), 2.3, 1.6, "Пара\n$(M_{ref}, M_{dist})$\n+ якорь ref=self", fc="#e7efe4")
    _box(ax, (9.2, 1.2), 1.6, 1.6, "q, MOS\nPLCC\nSROCC", fc="#f3e8e8")
    for x0, x1 in [(2.6, 3.2), (5.8, 6.4), (8.7, 9.2)]:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#222", lw=1.2))
    fig.tight_layout()
    fig.savefig(OUT / "fig_pler_hq_principle.png")
    plt.close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    fig_architecture()
    fig_factorial()
    fig_taxonomy()
    fig_coverage()
    fig_principle()
    print(f"figures written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
