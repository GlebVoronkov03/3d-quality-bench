"""Утилиты для OBJ: подсчёт вершин, вес файла, номер модели."""

from __future__ import annotations

import re
from pathlib import Path


def extract_model_number(name: str) -> int:
    """Числовой номер из Sphere_10 -> 10."""
    m = re.search(r"(\d+)\s*$", str(name).replace(".obj", ""))
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)$", str(name))
    return int(m.group(1)) if m else 0


def count_obj_vertices(filepath: str) -> int:
    """Быстрый подсчёт вершин в OBJ."""
    n = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                n += 1
    return n


def get_model_weight_mb(filepath: str) -> float:
    """Вес модели = размер файла, МБ."""
    p = Path(filepath)
    if not p.exists():
        return float("nan")
    return p.stat().st_size / (1024 * 1024)
