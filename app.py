"""
3D Model Quality Comparison — Streamlit GUI
Запуск: streamlit run app.py

Установка зависимостей:
    pip install -r requirements.txt
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Корень проекта первым в sys.path (до локальных импортов)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache_utils import clear_pycache

clear_pycache(ROOT)
# Удаляем чужой модуль analysis (jedi), если он попал в sys.modules
if "analysis" in sys.modules and not hasattr(sys.modules["analysis"], "enrich_dataframe"):
    del sys.modules["analysis"]

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tqdm import tqdm

from obj_helpers import count_obj_vertices, get_model_weight_mb
import results_analysis as ra
import external_metrics
from pair_utils import (
    REFERENCE_SELF_LABEL,
    append_reference_self_pairs,
    build_custom_qv_pairs_from_uploads,
    parse_model_order,
    parse_uploaded_mos,
    save_uploaded_obj,
)
from degradation import (
    DEFAULT_DECIMATION_LEVELS,
    DEFAULT_NOISE_LEVELS,
    DEFAULT_SMOOTH_LEVELS,
    DegradationGenerator,
    parse_levels,
)
from metrics_core import ALL_METRICS, METRIC_DIRECTION, compute_selected_metrics

# ---------------------------------------------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
QV_DIR = DATA_DIR / "qv_sphere"
HQ_DIR = DATA_DIR / "high_quality"
UPLOADS_DIR = DATA_DIR / "uploads"
GENERATED_DIR = ROOT / "generated"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"

for d in (GENERATED_DIR, RESULTS_DIR, UPLOADS_DIR, CHECKPOINTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="3D Quality Metrics",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📐 Сравнение 3D-моделей по метрикам качества")
st.caption("PLER-2.0, геометрические и перцептивные метрики | QV-сфера и пользовательские деградации")


# ================================================================ sidebar
with st.sidebar:
    st.header("Настройки")

    dataset_type = st.radio(
        "Тип датасета",
        ["QV-сфера", "Свои модели"],
        help="QV-сфера: Sphere или загрузка своих OBJ. Свои модели: генерация деградаций.",
    )

    qv_mode = "Встроенный Sphere"
    qv_ref_upload = None
    qv_dist_uploads: list = []
    qv_order_text = ""
    qv_mos_upload = None

    if dataset_type == "QV-сфера":
        st.subheader("QV-сфера")
        qv_mode = st.radio(
            "Источник моделей",
            ["Встроенный Sphere", "Загрузить свои OBJ"],
            help="1 = наиболее искажённая, n = ближе к эталону. Эталон — последний в порядке или отдельный файл.",
        )
        if qv_mode == "Загрузить свои OBJ":
            qv_ref_upload = st.file_uploader("Эталон (OBJ)", type=["obj"], key="qv_ref")
            qv_dist_uploads = st.file_uploader(
                "Искажённые модели (OBJ, несколько файлов)",
                type=["obj"],
                accept_multiple_files=True,
                key="qv_dist",
            )
            qv_order_text = st.text_area(
                "Порядок моделей (1=худшее … n=лучшее)",
                placeholder="Sphere_1.obj, Sphere_2.obj, … или 1:bad.obj, 2:mid.obj",
                help="Через запятую или с новой строки. Пусто — сортировка по имени файла.",
            )
            qv_mos_upload = st.file_uploader(
                "MOS (CSV, опционально)",
                type=["csv"],
                key="qv_mos",
                help="Формат: model_name;mos или transposed как data/mos.csv",
            )

    hybridmqa_ckpt = st.text_input(
        "HybridMQA checkpoint (.pth)",
        str(CHECKPOINTS_DIR / "ckpt_TMQA_adapted.pth"),
        help="Скачать: python scripts/setup_ml_metrics.py",
    )

    selected_metrics = []
    st.subheader("Метрики")
    select_all = st.checkbox("Выбрать все", value=True)
    for m in ALL_METRICS:
        if st.checkbox(m, value=select_all, key=f"metric_{m}"):
            selected_metrics.append(m)

    # --- Свои модели
    ref_models: list[str] = []
    method = "Decimation"
    levels_text = ""
    n_levels = 7
    noise_levels_text = ""

    if dataset_type == "Свои модели":
        st.subheader("Деградация")
        hq_files = sorted(HQ_DIR.glob("*.obj"))
        ref_options = [f.name for f in hq_files]
        ref_models = st.multiselect(
            "Эталонные модели",
            ref_options,
            default=ref_options[:1] if ref_options else [],
        )
        method = st.selectbox(
            "Метод",
            ["Decimation", "Noise", "Smoothing", "Combined"],
        )
        n_levels = st.slider("Количество уровней", 5, 20, 7)
        st.caption("Уровни (через запятую). Пусто = значения по умолчанию.")
        if method == "Decimation":
            levels_text = st.text_input("Ratios (доля сохраняемых граней)", "")
            default_hint = ", ".join(map(str, DEFAULT_DECIMATION_LEVELS))
            st.caption("Ratio = доля граней, которые останутся (0.5 → 50%). Можно < 0.01.")
        elif method == "Noise":
            levels_text = st.text_input("Sigma relative", "")
            default_hint = ", ".join(map(str, DEFAULT_NOISE_LEVELS))
        elif method == "Smoothing":
            levels_text = st.text_input("Итерации", "")
            default_hint = ", ".join(map(str, DEFAULT_SMOOTH_LEVELS))
        else:
            levels_text = st.text_input("Decimation ratios", "")
            noise_levels_text = st.text_input("Noise sigma", "")
            default_hint = "dec: " + ", ".join(map(str, DEFAULT_DECIMATION_LEVELS))
        st.caption(f"По умолчанию: {default_hint}")

    run_btn = st.button("🚀 Запустить расчёт", type="primary", use_container_width=True)


# ================================================================ helpers
def get_qv_pairs() -> list[dict]:
    ref = QV_DIR / "Sphere_14.obj"
    if not ref.exists():
        st.error(f"Эталон не найден: {ref}")
        return []
    pairs = []
    sphere_files = sorted(QV_DIR.glob("Sphere_*.obj"), key=lambda p: _sphere_index(p.stem))
    for p in sphere_files:
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
                "degradation_param": _sphere_index(p.stem),
                "model_number": _sphere_index(p.stem),
                "is_reference_self": False,
            }
        )
    return pairs


def get_qv_upload_pairs(
    run_id: str,
    ref_upload,
    dist_uploads: list,
    order_text: str,
    mos_upload=None,
) -> list[dict]:
    if not ref_upload:
        st.warning("Загрузите эталонный OBJ.")
        return []
    if not dist_uploads:
        st.warning("Загрузите хотя бы одну искажённую модель.")
        return []

    upload_dir = UPLOADS_DIR / run_id
    ref_path = save_uploaded_obj(upload_dir, ref_upload)
    dist_paths = [save_uploaded_obj(upload_dir, f) for f in dist_uploads]
    filenames = [f.name for f in dist_uploads]
    order = parse_model_order(order_text, filenames)

    mos_map = parse_uploaded_mos(mos_upload) if mos_upload else ra.load_mos_csv(str(DATA_DIR / "mos.csv"))
    pairs = build_custom_qv_pairs_from_uploads(ref_path, dist_paths, order, mos_map)
    if not pairs:
        st.error("Не удалось построить пары. Проверьте порядок и имена файлов.")
    else:
        st.info(f"Загружено {len(pairs)} пар. Эталон: `{Path(ref_path).name}`")
    return pairs


def _sphere_index(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def build_custom_pairs(
    ref_names: list[str],
    method: str,
    levels_text: str,
    n_levels: int,
    noise_text: str,
    run_id: str,
    progress_cb=None,
) -> list[dict]:
    gen = DegradationGenerator()
    pairs = []

    if method == "Decimation":
        defaults = DEFAULT_DECIMATION_LEVELS
    elif method == "Noise":
        defaults = DEFAULT_NOISE_LEVELS
    elif method == "Smoothing":
        defaults = DEFAULT_SMOOTH_LEVELS
    else:
        defaults = DEFAULT_DECIMATION_LEVELS

    levels = parse_levels(levels_text, defaults)
    # Если пользователь задал список — используем его целиком (не обрезаем)
    if not levels_text or not levels_text.strip():
        if len(levels) < n_levels:
            extra = n_levels - len(levels)
            if method == "Smoothing":
                start = int(max(levels) if levels else 1) + 1
                levels = list(levels) + list(range(start, start + extra))
            else:
                lo, hi = float(min(defaults)), float(max(defaults))
                pad = list(np.linspace(lo, hi, extra))
                levels = list(levels) + pad
        levels = levels[:n_levels]

    noise_extra = parse_levels(noise_text, DEFAULT_NOISE_LEVELS) if method == "Combined" else None

    for ref_name in ref_names:
        ref_path = str(HQ_DIR / ref_name)
        out_dir = GENERATED_DIR / run_id / Path(ref_name).stem
        generated = gen.generate_series(
            ref_path, str(out_dir), method, levels, noise_extra, progress_cb=progress_cb
        )
        for dist_path, param, m in generated:
            pairs.append(
                {
                    "reference": ref_path,
                    "distorted": dist_path,
                    "reference_model": ref_name,
                    "distorted_model": Path(dist_path).stem,
                    "distorted_path": dist_path,
                    "degradation_method": m,
                    "degradation_param": param,
                    "is_reference_self": False,
                }
            )
    return pairs


def _display_distorted_name(name: str) -> str:
    if name == REFERENCE_SELF_LABEL:
        return "эталон (ref=self)"
    return name


def run_pipeline(pairs: list[dict], metrics: list[str], run_id: str, hybridmqa_ckpt: str = "") -> pd.DataFrame:
    checkpoint = RESULTS_DIR / f"checkpoint_{run_id}.csv"
    done_keys: set[str] = set()
    rows: list[dict] = []

    if checkpoint.exists():
        prev = pd.read_csv(checkpoint)
        rows = prev.to_dict("records")
        done_keys = {f"{r['reference_model']}|{r['distorted_model']}" for r in rows}
        st.info(f"Продолжение с checkpoint: {len(rows)} пар уже обработано")

    total = len(pairs)
    progress = st.progress(0, text="Подготовка...")
    status = st.empty()
    eta_box = st.empty()

    times: list[float] = []
    mos_map = ra.load_mos_csv(str(DATA_DIR / "mos.csv"))

    import os

    if hybridmqa_ckpt:
        os.environ["HYBRIDMQA_CKPT"] = hybridmqa_ckpt

    for i, pair in enumerate(tqdm(pairs, desc="Метрики", file=sys.stdout)):
        key = f"{pair['reference_model']}|{pair['distorted_model']}"
        if key in done_keys:
            progress.progress((i + 1) / total, text=f"{i + 1}/{total} пар (из checkpoint)")
            continue

        t0 = time.time()
        is_self = pair.get("is_reference_self", False)
        label = _display_distorted_name(pair["distorted_model"])
        progress.progress(i / max(total, 1), text=f"Пара {i + 1}/{total}: {label}")
        status.text(f"Пара {i + 1}/{total}: {label} — подготовка…")

        dist_path = pair.get("distorted_path") or pair["distorted"]
        row = {
            "reference_model": pair["reference_model"],
            "distorted_model": pair["distorted_model"],
            "distorted_path": dist_path,
            "degradation_method": pair.get("degradation_method", ""),
            "degradation_param": pair.get("degradation_param", np.nan),
            "is_reference_self": is_self,
            "model_number": pair.get("model_number", _sphere_index(pair["distorted_model"])),
            "n_points": count_obj_vertices(dist_path),
            "model_weight_mb": get_model_weight_mb(dist_path),
        }

        def _metric_progress(msg: str) -> None:
            status.text(f"Пара {i + 1}/{total}: {label} — {msg}")

        try:
            metric_vals = compute_selected_metrics(
                pair["reference"],
                pair["distorted"],
                metrics,
                external_module=external_metrics,
                progress_cb=_metric_progress,
            )
            row.update(metric_vals)
        except Exception as exc:
            logger.exception("Ошибка пары %s: %s", key, exc)
            row["error"] = str(exc)

        stem = pair["distorted_model"].replace(".obj", "")
        ref_stem = Path(pair["reference_model"]).stem
        if stem in mos_map:
            row["MOS"] = mos_map[stem]
        elif pair["distorted_model"] in mos_map:
            row["MOS"] = mos_map[pair["distorted_model"]]
        elif is_self and ref_stem in mos_map:
            row["MOS"] = mos_map[ref_stem]
        elif "MOS" in pair:
            row["MOS"] = pair["MOS"]

        elapsed = time.time() - t0
        times.append(elapsed)
        row["compute_time_s"] = elapsed
        rows.append(row)

        # checkpoint после каждой пары
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(checkpoint, index=False, float_format="%.8f")

        progress.progress((i + 1) / total, text=f"{i + 1}/{total} пар")
        if times:
            avg = np.mean(times)
            remaining = avg * (total - i - 1)
            eta_box.caption(f"⏱ Среднее: {avg:.1f} с/пара | Осталось ~{remaining / 60:.1f} мин")

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"results_{run_id}.csv"
    df.to_csv(out_path, index=False, float_format="%.8f")
    logger.info("Результаты сохранены: %s", out_path)
    st.success(f"Готово! CSV: `{out_path.name}`")
    return df


# ================================================================ run
if run_btn:
    if not selected_metrics:
        st.warning("Выберите хотя бы одну метрику.")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        pairs: list[dict] = []
        if dataset_type == "QV-сфера":
            if qv_mode == "Встроенный Sphere":
                pairs = get_qv_pairs()
            else:
                pairs = get_qv_upload_pairs(
                    run_id, qv_ref_upload, qv_dist_uploads, qv_order_text, qv_mos_upload
                )
        else:
            if not ref_models:
                st.warning("Выберите эталонные модели.")
            else:
                gen_status = st.empty()
                gen_progress = st.progress(0.0, text="Генерация деградаций...")
                gen_messages: list[str] = []

                def _gen_cb(msg: str) -> None:
                    gen_messages.append(msg)
                    gen_status.text(msg)

                with st.spinner("Генерация деградированных моделей..."):
                    pairs = build_custom_pairs(
                        ref_models,
                        method,
                        levels_text,
                        n_levels,
                        noise_levels_text,
                        run_id,
                        progress_cb=_gen_cb,
                    )
                gen_progress.progress(1.0, text=f"Сгенерировано {len(pairs)} моделей")
                if pairs:
                    st.success(f"Деградации сохранены в `generated/{run_id}/`")

        if pairs:
            pairs = append_reference_self_pairs(pairs)
            with st.spinner("Расчёт метрик..."):
                df_raw = run_pipeline(pairs, selected_metrics, run_id, hybridmqa_ckpt)
                df_raw = ra.enrich_dataframe(df_raw, qv_dir=QV_DIR, generated_dir=GENERATED_DIR)
                st.session_state["df_raw"] = df_raw
                st.session_state["metric_cols"] = [
                    c for c in selected_metrics if c in df_raw.columns
                ]

# ================================================================ results
if "df_raw" in st.session_state:
    df_raw: pd.DataFrame = st.session_state["df_raw"]
    # Обогащение для старых результатов без n_points / model_weight_mb
    if "n_points" not in df_raw.columns or "x_index" in df_raw.columns:
        df_raw = ra.enrich_dataframe(df_raw, qv_dir=QV_DIR, generated_dir=GENERATED_DIR)
        st.session_state["df_raw"] = df_raw

    metric_cols: list[str] = st.session_state.get("metric_cols", [])

    st.subheader("Параметры отображения")
    sort_col, sort_dir = st.columns([2, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Сортировка строк",
            ["Номер модели", "Количество точек", "Вес модели (МБ)"],
            index=0,
        )
    with sort_dir:
        sort_asc = st.checkbox("По возрастанию", value=True)

    df_display_base = ra.sort_dataframe(df_raw, sort_by, ascending=sort_asc)

    tab_raw, tab_norm, tab_corr, tab_plot, tab_heatmap = st.tabs(
        ["Сырые метрики", "Нормированные", "Корреляции", "Графики", "Корр. матрица"]
    )

    # --- Raw
    with tab_raw:
        st.subheader("Сырые значения метрик")
        df_raw_view = ra.build_display_table(df_display_base, metric_cols, normalized=False)
        if "distorted_model" in df_raw_view.columns:
            df_raw_view = df_raw_view.copy()
            df_raw_view["distorted_model"] = df_raw_view["distorted_model"].map(_display_distorted_name)
        st.dataframe(df_raw_view, use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        csv_bytes = df_raw_view.to_csv(index=False).encode("utf-8-sig")
        c1.download_button("⬇️ CSV", csv_bytes, "metrics_raw.csv", "text/csv")
        try:
            xlsx = ra.df_to_xlsx_bytes(df_raw_view)
            c2.download_button(
                "⬇️ XLSX",
                xlsx,
                "metrics_raw.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            c2.warning(f"XLSX: {e}")

    # --- Normalized
    with tab_norm:
        df_norm_display = ra.build_display_table(df_display_base, metric_cols, normalized=True)
        if "distorted_model" in df_norm_display.columns:
            df_norm_display = df_norm_display.copy()
            df_norm_display["distorted_model"] = df_norm_display["distorted_model"].map(_display_distorted_name)
        st.subheader("Нормированные метрики (якорь ref vs self, 0–100)")
        st.caption(
            "100 = эталон сравнивается сам с собой (ref vs self), 0 = худшая искажённая модель в группе. "
            "MOS_norm — та же шкала (100 = MOS эталона). Метрики на одной шкале можно сравнивать между собой."
        )
        st.dataframe(df_norm_display, use_container_width=True, height=400)
        st.session_state["df_norm"] = df_norm_display

    # --- Correlations
    with tab_corr:
        st.subheader("Корреляции с MOS")
        st.markdown(
            """
            **Что означает `_p` (p-value)?**  
            Это статистическая значимость корреляции: вероятность получить такую же или более
            сильную связь *случайно*, если на самом деле корреляции нет.
            - **p < 0.05** — связь обычно считают значимой (5% уровень).
            - **p ≥ 0.05** — оснований считать связь надёжной мало (мало точек или слабый эффект).
            """
        )
        if "MOS" not in df_raw.columns or df_raw["MOS"].notna().sum() < 3:
            st.info("MOS недоступен или недостаточно точек (нужно ≥3).")
        else:
            df_corr = ra.compute_correlations(df_display_base, metric_cols)
            st.dataframe(df_corr, use_container_width=True)
            if not df_corr.empty:
                fig = px.bar(
                    df_corr,
                    x="metric",
                    y=["PLCC", "SROCC", "Kendall_tau"],
                    barmode="group",
                    title="Корреляции метрик с MOS",
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Plots
    with tab_plot:
        st.subheader("Интерактивные графики (Plotly)")
        df_plot = st.session_state.get("df_norm", ra.build_display_table(df_display_base, metric_cols, True))
        if "MOS_norm" not in df_plot.columns and "MOS" in df_display_base.columns:
            df_plot = df_plot.copy()
            df_plot["MOS_norm"] = ra.normalize_mos_for_plot(df_display_base)

        c_x, c_xs, c_ys = st.columns([2, 1, 1])
        x_options = {
            "Номер модели": "model_number",
            "Количество точек": "n_points",
            "Вес модели (МБ)": "model_weight_mb",
            "Параметр деградации": "degradation_param",
        }
        with c_x:
            x_label = st.selectbox("Ось X", list(x_options.keys()), index=0)
            x_col = x_options[x_label]
        with c_xs:
            x_scale = st.radio("Масштаб X", ["linear", "log"], horizontal=True, key="x_scale")
        with c_ys:
            y_scale = st.radio("Масштаб Y", ["linear", "log"], horizontal=True, key="y_scale")

        plot_options = list(metric_cols)
        if "MOS_norm" in df_plot.columns and df_plot["MOS_norm"].notna().any():
            plot_options = plot_options + ["MOS"]
        default_plot = plot_options[:3] if len(plot_options) >= 3 else plot_options
        if "MOS" in plot_options and "MOS" not in default_plot and len(default_plot) >= 2:
            default_plot = default_plot[:2] + ["MOS"]

        plot_metrics = st.multiselect(
            "Метрики на графике",
            plot_options,
            default=default_plot,
            help="MOS — нормированный MOS (0–100), для сравнения с метриками качества.",
        )

        if plot_metrics and x_col in df_plot.columns:
            fig = go.Figure()
            x_vals = df_plot[x_col]
            hover_names = df_plot.get("distorted_model", pd.Series("", index=df_plot.index)).map(
                _display_distorted_name
            )
            for m in plot_metrics:
                y_col = "MOS_norm" if m == "MOS" else m
                if y_col in df_plot.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=df_plot[y_col],
                            mode="lines+markers",
                            name=m,
                            text=hover_names,
                            hovertemplate="%{text}<br>" + x_label + ": %{x}<br>" + m + ": %{y}<extra></extra>",
                        )
                    )
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title="Нормированное значение (0–100)",
                height=520,
                legend=dict(orientation="h"),
                xaxis=dict(type=x_scale),
                yaxis=dict(type=y_scale),
            )
            st.plotly_chart(fig, use_container_width=True)

            single = st.selectbox("Одна метрика", plot_metrics)
            if single:
                y_single = "MOS_norm" if single == "MOS" else single
                fig2 = px.line(
                    df_plot,
                    x=x_col,
                    y=y_single,
                    markers=True,
                    text="distorted_model" if "distorted_model" in df_plot.columns else None,
                    title=f"{single} vs {x_label}",
                )
                if "distorted_model" in df_plot.columns:
                    fig2.update_traces(text=hover_names)
                fig2.update_layout(
                    xaxis=dict(type=x_scale),
                    yaxis=dict(type=y_scale),
                )
                st.plotly_chart(fig2, use_container_width=True)

            try:
                import plotly.io as pio

                png_buf = pio.to_image(fig, format="png", width=1200, height=600)
                st.download_button("📷 Скачать график PNG", png_buf, "metrics_plot.png", "image/png")
            except Exception:
                st.caption("Для PNG установите: pip install kaleido")

    # --- Heatmap
    with tab_heatmap:
        st.subheader("Корреляционная матрица метрик")
        if len(metric_cols) >= 2:
            import matplotlib.pyplot as plt
            import seaborn as sns

            corr_data = df_display_base[metric_cols].dropna(how="all")
            if len(corr_data) >= 2:
                cm = corr_data.corr(method="spearman")
                fig_h, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
                ax.set_title("Spearman корреляция между метриками")
                st.pyplot(fig_h)
                plt.close(fig_h)
            else:
                st.info("Недостаточно данных для матрицы.")
        else:
            st.info("Выберите ≥2 метрик.")

else:
    st.markdown(
        """
        ### Инструкция
        1. Выберите тип датасета и метрики в боковой панели.
        2. **QV-сфера**: встроенный Sphere или загрузите свои OBJ с порядком 1…n.
        3. Для **Своих моделей** задайте метод деградации и уровни.
        4. Нажмите **Запустить расчёт** — после всех пар добавляется сравнение ref vs self.
        5. Нормировка: 100 = ref vs self, 0 = худшая модель; MOS доступен на графиках.
        6. HybridMQA/FMQM: `python scripts/setup_ml_metrics.py`

        **QV-сфера**: эталон `Sphere_14.obj`, MOS — `data/mos.csv`.
        """
    )
