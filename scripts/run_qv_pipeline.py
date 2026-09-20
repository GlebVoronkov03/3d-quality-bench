"""Headless QV pipeline (same as Streamlit: all metrics except CMDM/FMQM)."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pair_utils import append_reference_self_pairs
from metrics_core import ALL_METRICS, compute_selected_metrics
import external_metrics
import results_analysis as ra
from obj_helpers import count_obj_vertices, get_model_weight_mb


def get_qv_pairs():
    qv = ROOT / "data" / "qv_sphere"
    ref = qv / "Sphere_14.obj"

    def idx(name):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    pairs = []
    for p in sorted(qv.glob("Sphere_*.obj"), key=lambda x: idx(x.stem)):
        if p.name == "Sphere_14.obj":
            continue
        pairs.append(
            {
                "reference": str(ref),
                "distorted": str(p),
                "reference_model": ref.name,
                "distorted_model": p.stem,
                "distorted_path": str(p),
                "degradation_method": "QV-dataset",
                "degradation_param": idx(p.stem),
                "model_number": idx(p.stem),
                "is_reference_self": False,
            }
        )
    return pairs


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = [m for m in ALL_METRICS if m not in ("CMDM", "FMQM")]
    pairs = append_reference_self_pairs(get_qv_pairs())
    mos_map = ra.load_mos_csv(str(ROOT / "data" / "mos.csv"))
    checkpoint = ROOT / "results" / f"checkpoint_{run_id}.csv"
    rows = []

    print(f"run_id={run_id} pairs={len(pairs)} metrics={len(metrics)}")
    t_all = time.time()
    for i, pair in enumerate(pairs, 1):
        t0 = time.time()
        label = pair["distorted_model"]
        print(f"[{i}/{len(pairs)}] {label} …", flush=True)
        vals = compute_selected_metrics(
            pair["reference"],
            pair["distorted"],
            metrics,
            external_module=external_metrics,
            progress_cb=lambda m: print(f"  {m}", flush=True),
        )
        row = {
            "reference_model": pair["reference_model"],
            "distorted_model": label,
            "distorted_path": pair["distorted_path"],
            "degradation_method": pair.get("degradation_method", ""),
            "degradation_param": pair.get("degradation_param"),
            "is_reference_self": pair.get("is_reference_self", False),
            "model_number": pair.get("model_number"),
            "n_points": count_obj_vertices(pair["distorted_path"]),
            "model_weight_mb": get_model_weight_mb(pair["distorted_path"]),
            "compute_time_s": time.time() - t0,
            **vals,
        }
        stem = label.replace(".obj", "")
        if stem in mos_map:
            row["MOS"] = mos_map[stem]
        rows.append(row)
        pd.DataFrame(rows).to_csv(checkpoint, index=False, float_format="%.8f")
        print(f"  done {row['compute_time_s']:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"results_{run_id}.csv"
    df.to_csv(out, index=False, float_format="%.8f")
    print(f"TOTAL {time.time()-t_all:.1f}s ({(time.time()-t_all)/60:.1f} min)")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
