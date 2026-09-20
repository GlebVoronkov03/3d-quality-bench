"""
Анализ результатов: нормировка по эталону (ref vs self), сортировка, корреляции с MOS.
(Переименован из analysis.py — имя analysis конфликтовало с jedi.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from obj_helpers import count_obj_vertices, extract_model_number, get_model_weight_mb
from pair_utils import REFERENCE_SELF_LABEL


def _metric_bounds():
    from metrics_core import METRIC_DIRECTION, METRIC_THEORETICAL_BOUNDS
    return METRIC_DIRECTION, METRIC_THEORETICAL_BOUNDS


META_COLUMNS = [
    "reference_model",
    "distorted_model",
    "model_number",
    "n_points",
    "model_weight_mb",
    "degradation_method",
    "degradation_param",
    "is_reference_self",
    "MOS",
    "MOS_norm",
    "compute_time_s",
    "error",
]


def _is_ref_self_row(row: pd.Series) -> bool:
    if "is_reference_self" in row.index and bool(row.get("is_reference_self")):
        return True
    return str(row.get("distorted_model", "")) == REFERENCE_SELF_LABEL


def load_mos_csv(path: str = "data/mos.csv") -> Dict[str, float]:
    """Загрузка MOS из transposed CSV (model_name;Sphere_3;... / mos;1.25;...)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, sep=";", header=None)
        if df.shape[0] < 2:
            return {}
        models = df.iloc[0, 1:].tolist()
        scores = df.iloc[1, 1:].tolist()
        return {str(m).strip(): float(s) for m, s in zip(models, scores) if str(m).strip()}
    except Exception:
        df = pd.read_csv(p, sep=";")
        if "model_name" in df.columns and "mos" in df.columns:
            return dict(zip(df["model_name"], df["mos"]))
        return {}


def enrich_dataframe(
    df: pd.DataFrame,
    qv_dir: Optional[Path] = None,
    generated_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Добавляет n_points, model_weight_mb, model_number; убирает x_index."""
    out = df.copy()
    if "x_index" in out.columns:
        out = out.drop(columns=["x_index"])

    n_points_list = []
    weight_list = []
    number_list = []

    for _, row in out.iterrows():
        is_self = _is_ref_self_row(row)
        stem = str(row.get("distorted_model", "")).replace(".obj", "")
        if is_self:
            number_list.append(10_000)
        else:
            number_list.append(extract_model_number(stem))

        dist_path = row.get("distorted_path")
        if is_self:
            ref_name = str(row.get("reference_model", ""))
            ref_candidates = []
            if qv_dir:
                ref_candidates.append(qv_dir / ref_name)
                ref_candidates.append(qv_dir / f"{ref_name}.obj")
            if generated_dir:
                ref_candidates.extend(generated_dir.glob(f"**/{ref_name}"))
                ref_candidates.extend(generated_dir.glob(f"**/{Path(ref_name).stem}.obj"))
            dist_path = next((str(c) for c in ref_candidates if Path(c).exists()), dist_path)

        if not dist_path or not Path(str(dist_path)).exists():
            candidates = []
            if qv_dir:
                candidates.append(qv_dir / f"{stem}.obj")
            if generated_dir:
                candidates.extend(generated_dir.glob(f"**/{stem}.obj"))
            dist_path = next((str(c) for c in candidates if c.exists()), None)

        if dist_path and Path(dist_path).exists():
            n_points_list.append(count_obj_vertices(str(dist_path)))
            weight_list.append(get_model_weight_mb(str(dist_path)))
        elif "n_points" in row and pd.notna(row.get("n_points")):
            n_points_list.append(row["n_points"])
            weight_list.append(row.get("model_weight_mb", np.nan))
        else:
            n_points_list.append(np.nan)
            weight_list.append(np.nan)

    out["is_reference_self"] = out.apply(_is_ref_self_row, axis=1)
    out["model_number"] = number_list
    out["n_points"] = n_points_list
    out["model_weight_mb"] = weight_list
    return out


def sort_dataframe(df: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
    """Сортировка таблицы по выбранному ключу."""
    col_map = {
        "Номер модели": "model_number",
        "Количество точек": "n_points",
        "Вес модели (МБ)": "model_weight_mb",
    }
    col = col_map.get(sort_by, sort_by)
    if col not in df.columns:
        return df
    return df.sort_values(col, ascending=ascending, kind="mergesort").reset_index(drop=True)


def normalize_with_ref_anchor(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Нормировка 0..100 по якорю ref-vs-self (100) и худшей искажённой модели (0).
    Группировка по reference_model; позволяет сравнивать разные метрики на одной шкале.
    """
    direction_map, _ = _metric_bounds()
    norm = pd.DataFrame(index=df.index, columns=metric_cols, dtype=float)

    ref_groups = df.groupby("reference_model", sort=False) if "reference_model" in df.columns else [(None, df)]

    for _ref, group in ref_groups:
        self_mask = group.apply(_is_ref_self_row, axis=1)
        dist_mask = ~self_mask
        self_rows = group[self_mask]
        dist_rows = group[dist_mask]

        for col in metric_cols:
            if col not in df.columns:
                continue
            direction = direction_map.get(col, "lower")

            if self_rows.empty:
                v_best = np.nan
            else:
                v_best = self_rows[col].iloc[0]

            if dist_rows.empty or dist_rows[col].dropna().empty:
                worst = np.nan
            elif direction == "higher":
                worst = dist_rows[col].min()
            else:
                worst = dist_rows[col].max()

            for idx in group.index:
                v = df.at[idx, col]
                if pd.isna(v):
                    norm.at[idx, col] = np.nan
                    continue
                if pd.isna(v_best) or pd.isna(worst):
                    norm.at[idx, col] = np.nan
                    continue
                span = float(v_best - worst) if direction == "higher" else float(worst - v_best)
                if abs(span) < 1e-12:
                    norm.at[idx, col] = 100.0 if _is_ref_self_row(df.loc[idx]) else 0.0
                    continue
                if direction == "higher":
                    score = 100.0 * (float(v) - float(worst)) / span
                else:
                    score = 100.0 * (float(worst) - float(v)) / span
                norm.at[idx, col] = float(np.clip(score, 0.0, 100.0))

    return norm


def normalize_mos_for_plot(df: pd.DataFrame) -> pd.Series:
    """MOS → 0..100: якорь — ref-self (или max MOS в группе)."""
    if "MOS" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    mos_norm = pd.Series(np.nan, index=df.index, dtype=float)
    groups = df.groupby("reference_model", sort=False) if "reference_model" in df.columns else [(None, df)]

    for _ref, group in groups:
        self_mask = group.apply(_is_ref_self_row, axis=1)
        mos_vals = group["MOS"].dropna()
        if mos_vals.empty:
            continue
        anchor = np.nan
        if self_mask.any():
            self_mos = group.loc[self_mask, "MOS"].dropna()
            if not self_mos.empty:
                anchor = float(self_mos.iloc[0])
        if pd.isna(anchor):
            anchor = float(mos_vals.max())
        if anchor <= 0:
            continue
        for idx in group.index:
            m = df.at[idx, "MOS"]
            if pd.notna(m):
                mos_norm.at[idx] = float(np.clip(100.0 * float(m) / anchor, 0.0, 100.0))
    return mos_norm


def filter_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Исключает строки ref-vs-self из корреляций с MOS."""
    if "is_reference_self" in df.columns:
        return df[~df["is_reference_self"].fillna(False)].copy()
    return df[df["distorted_model"] != REFERENCE_SELF_LABEL].copy()


def normalize_theoretical(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Нормировка 0..100 по теоретическим min/max."""
    direction_map, bounds_map = _metric_bounds()
    norm = pd.DataFrame(index=df.index)
    for col in metric_cols:
        if col not in df.columns:
            continue
        bounds = bounds_map.get(col)
        if not bounds:
            norm[col] = np.nan
            continue
        v_min, v_max = bounds
        span = v_max - v_min
        if span <= 0:
            norm[col] = np.nan
            continue
        direction = direction_map.get(col, "lower")
        vals = []
        for v in df[col]:
            if pd.isna(v):
                vals.append(np.nan)
                continue
            v_clipped = float(np.clip(v, v_min, v_max))
            if direction == "higher":
                score = 100.0 * (v_clipped - v_min) / span
            else:
                score = 100.0 * (v_max - v_clipped) / span
            vals.append(float(np.clip(score, 0.0, 100.0)))
        norm[col] = vals
    return norm


def build_display_table(
    df_raw: pd.DataFrame,
    metric_cols: List[str],
    normalized: bool = False,
    normalization: str = "anchor",
) -> pd.DataFrame:
    """Собирает таблицу для отображения с meta-колонками в нужном порядке."""
    meta = [c for c in META_COLUMNS if c in df_raw.columns and c not in metric_cols]
    if normalized:
        if normalization == "theoretical":
            norm_part = normalize_theoretical(df_raw, metric_cols)
        else:
            norm_part = normalize_with_ref_anchor(df_raw, metric_cols)
        data = df_raw[meta].copy()
        for c in metric_cols:
            if c in norm_part.columns:
                data[c] = norm_part[c]
        if "MOS" in df_raw.columns:
            data["MOS_norm"] = normalize_mos_for_plot(df_raw)
        return data
    cols = meta + [c for c in metric_cols if c in df_raw.columns]
    return df_raw[cols].copy()


def compute_correlations(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """PLCC, SROCC, Kendall Tau каждой метрики с MOS (+ p-value)."""
    df = filter_for_correlation(df)
    if "MOS" not in df.columns or df["MOS"].notna().sum() < 3:
        return pd.DataFrame()
    rows = []
    mos = df["MOS"].values
    for col in metric_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["MOS"].notna()
        if mask.sum() < 3:
            continue
        x = df.loc[mask, col].values
        y = mos[mask]
        try:
            plcc, plcc_p = pearsonr(x, y)
            srocc, srocc_p = spearmanr(x, y)
            kt, kt_p = kendalltau(x, y)
            rows.append(
                {
                    "metric": col,
                    "PLCC": plcc,
                    "PLCC_p": plcc_p,
                    "SROCC": srocc,
                    "SROCC_p": srocc_p,
                    "Kendall_tau": kt,
                    "Kendall_p": kt_p,
                    "n": int(mask.sum()),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return buf.getvalue()
