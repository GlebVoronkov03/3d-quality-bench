"""Интеграционные тесты для pair_utils и results_analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pair_utils import (
    REFERENCE_SELF_LABEL,
    append_reference_self_pairs,
    build_custom_qv_pairs_from_uploads,
    parse_model_order,
)
import results_analysis as ra


def test_append_reference_self_pairs():
    pairs = [
        {
            "reference": "/ref/a.obj",
            "distorted": "/dist/1.obj",
            "reference_model": "a.obj",
            "distorted_model": "m1",
        },
        {
            "reference": "/ref/a.obj",
            "distorted": "/dist/2.obj",
            "reference_model": "a.obj",
            "distorted_model": "m2",
        },
    ]
    out = append_reference_self_pairs(pairs)
    assert len(out) == 3
    self_rows = [p for p in out if p.get("is_reference_self")]
    assert len(self_rows) == 1
    assert self_rows[0]["distorted_model"] == REFERENCE_SELF_LABEL
    assert self_rows[0]["reference"] == self_rows[0]["distorted"]


def test_parse_model_order():
    files = ["c.obj", "a.obj", "b.obj"]
    assert parse_model_order("", files) == ["a.obj", "b.obj", "c.obj"]
    assert parse_model_order("b.obj, a.obj", files) == ["b.obj", "a.obj", "c.obj"]


def test_build_custom_qv_pairs():
    pairs = build_custom_qv_pairs_from_uploads(
        "/ref/ref.obj",
        ["/d/1.obj", "/d/2.obj"],
        ["1.obj", "2.obj"],
        {"1": 1.0, "2": 4.5},
    )
    assert len(pairs) == 2
    assert pairs[0]["model_number"] == 1
    assert pairs[1]["MOS"] == 4.5


def test_normalize_with_ref_anchor_higher_is_better():
    df = pd.DataFrame(
        {
            "reference_model": ["ref"] * 3,
            "distorted_model": ["bad", "mid", REFERENCE_SELF_LABEL],
            "is_reference_self": [False, False, True],
            "MSDM2": [0.2, 0.6, 1.0],
        }
    )
    norm = ra.normalize_with_ref_anchor(df, ["MSDM2"])
    assert norm.loc[0, "MSDM2"] == pytest.approx(0.0)
    assert norm.loc[1, "MSDM2"] == pytest.approx(50.0)
    assert norm.loc[2, "MSDM2"] == pytest.approx(100.0)


def test_normalize_with_ref_anchor_lower_is_better():
    df = pd.DataFrame(
        {
            "reference_model": ["ref"] * 3,
            "distorted_model": ["bad", "mid", REFERENCE_SELF_LABEL],
            "is_reference_self": [False, False, True],
            "Chamfer (mm)": [10.0, 5.0, 0.0],
        }
    )
    norm = ra.normalize_with_ref_anchor(df, ["Chamfer (mm)"])
    assert norm.loc[0, "Chamfer (mm)"] == pytest.approx(0.0)
    assert norm.loc[2, "Chamfer (mm)"] == pytest.approx(100.0)


def test_normalize_mos_for_plot():
    df = pd.DataFrame(
        {
            "reference_model": ["ref"] * 3,
            "distorted_model": ["a", "b", REFERENCE_SELF_LABEL],
            "is_reference_self": [False, False, True],
            "MOS": [2.0, 4.0, 5.0],
        }
    )
    mos_norm = ra.normalize_mos_for_plot(df)
    assert mos_norm.iloc[2] == pytest.approx(100.0)
    assert mos_norm.iloc[1] == pytest.approx(80.0)


def test_filter_for_correlation_excludes_self():
    df = pd.DataFrame(
        {
            "distorted_model": ["a", REFERENCE_SELF_LABEL],
            "is_reference_self": [False, True],
            "MOS": [1.0, 5.0],
        }
    )
    filtered = ra.filter_for_correlation(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["distorted_model"] == "a"


def test_pair_utils_no_streamlit_dependency():
    source = (ROOT / "pair_utils.py").read_text(encoding="utf-8")
    assert "import streamlit" not in source
