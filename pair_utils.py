"""Формирование пар (эталон, искажённая) для всех режимов."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REFERENCE_SELF_LABEL = "__REFERENCE_SELF__"


def append_reference_self_pairs(pairs: List[dict]) -> List[dict]:
    """Добавляет для каждого уникального эталона пару (ref, ref)."""
    seen: set[str] = set()
    extra: List[dict] = []
    for pair in pairs:
        ref_path = pair["reference"]
        if ref_path in seen:
            continue
        seen.add(ref_path)
        ref_name = pair.get("reference_model") or Path(ref_path).name
        extra.append(
            {
                "reference": ref_path,
                "distorted": ref_path,
                "reference_model": ref_name,
                "distorted_model": REFERENCE_SELF_LABEL,
                "distorted_path": ref_path,
                "degradation_method": "reference-self",
                "degradation_param": 1.0,
                "model_number": 10_000,
                "is_reference_self": True,
            }
        )
    return pairs + extra


def save_uploaded_obj(upload_dir: Path, uploaded_file, filename: Optional[str] = None) -> str:
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = filename or uploaded_file.name
    dst = upload_dir / name
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dst)


def parse_model_order(order_text: str, filenames: Sequence[str]) -> List[str]:
    """
    Парсит порядок моделей.
    Форматы:
      - 'file1.obj, file2.obj, file3.obj'
      - '1:file1.obj, 2:file2.obj'
      - пусто → сортировка по имени
    """
    if not order_text or not order_text.strip():
        return sorted(filenames)

    items = [x.strip() for x in re.split(r"[,;\n]+", order_text) if x.strip()]
    ordered: List[str] = []
    name_set = set(filenames)
    for item in items:
        name = item.split(":", 1)[-1].strip()
        if name in name_set:
            ordered.append(name)
    for name in sorted(filenames):
        if name not in ordered:
            ordered.append(name)
    return ordered


def build_custom_qv_pairs_from_uploads(
    ref_path: str,
    distorted_paths: List[str],
    order: List[str],
    mos_map: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """Пары для пользовательских OBJ: порядок 1..n (1 — наиболее искажённая)."""
    path_by_name = {Path(p).name: p for p in distorted_paths}
    pairs: List[dict] = []
    ref_name = Path(ref_path).name

    for rank, fname in enumerate(order, start=1):
        if fname not in path_by_name:
            continue
        dpath = path_by_name[fname]
        stem = Path(fname).stem
        pair = {
            "reference": ref_path,
            "distorted": dpath,
            "reference_model": ref_name,
            "distorted_model": stem,
            "distorted_path": dpath,
            "degradation_method": "custom-upload",
            "degradation_param": float(rank),
            "model_number": rank,
            "is_reference_self": False,
        }
        if mos_map:
            pair["MOS"] = mos_map.get(stem, mos_map.get(fname, float("nan")))
        pairs.append(pair)
    return pairs


def parse_uploaded_mos(mos_file) -> Dict[str, float]:
    """MOS CSV: model_name,mos или transposed как data/mos.csv."""
    import io

    import pandas as pd

    content = mos_file.getvalue().decode("utf-8-sig", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(content), sep=";")
        if "model_name" in df.columns and "mos" in df.columns:
            return dict(zip(df["model_name"].astype(str), df["mos"].astype(float)))
    except Exception:
        pass
    try:
        df = pd.read_csv(io.StringIO(content), sep=";", header=None)
        if df.shape[0] >= 2:
            models = df.iloc[0, 1:].tolist()
            scores = df.iloc[1, 1:].tolist()
            return {str(m).strip(): float(s) for m, s in zip(models, scores) if str(m).strip()}
    except Exception:
        pass
    return {}
