"""Benchmark full QV pipeline (all pairs, all metrics except CMDM/FMQM)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pair_utils import append_reference_self_pairs
from metrics_core import ALL_METRICS, compute_selected_metrics
import external_metrics


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
                "is_reference_self": False,
            }
        )
    return pairs


def main():
    metrics = [m for m in ALL_METRICS if m not in ("CMDM", "FMQM")]
    pairs = append_reference_self_pairs(get_qv_pairs())
    print(f"pairs={len(pairs)} metrics={len(metrics)}")
    t_all = time.time()
    times = []
    for i, p in enumerate(pairs, 1):
        t0 = time.time()
        compute_selected_metrics(
            p["reference"], p["distorted"], metrics, external_module=external_metrics
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"{i}/{len(pairs)} {p['distorted_model']}: {dt:.1f}s", flush=True)
    total = time.time() - t_all
    print(f"TOTAL {total:.1f}s ({total/60:.1f} min) avg={sum(times)/len(times):.1f}s/pair")


if __name__ == "__main__":
    main()
