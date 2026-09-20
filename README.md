# 3D Quality Bench

Streamlit research platform for **full-reference 3D mesh quality**: PLER family metrics, classic geometric metrics, controllable degradations, and correlations with MOS (PLCC / SROCC / Kendall).

> Author: **Gleb Voronkov** · Non-commercial research license.

## Why it matters
One place to generate distorted meshes (noise, smoothing, decimation, hybrid), compute 20+ metrics, normalize scores, and compare against subjective MOS — useful for codec research, XR content QA, and metric development.

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Tiny OBJ fixtures live in `data/fixtures/`. For full datasets, use your own meshes under `data/` (large binaries are intentionally not shipped).

## Components
- `app.py` — Streamlit UI
- `degradation.py` — decimate / noise / smooth / combined
- `metrics_core.py` / `external_metrics.py` — metric wrappers
- `classic_metrics/` — Chamfer / Hausdorff / F-score batch helper
- Cross-link: core PLER library → [pler-3d-quality](https://github.com/GlebVoronkov03/pler-3d-quality)

## License
**Non-Commercial Research License** (`LICENSE`). Commercial use → `mybook3@mail.ru` / `@Gleb_Voronkov`.