from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.tnm_enthalpy_time_fit import (
    SINGLE_FIG,
    TWOSTEP_FIG,
    WINDOW_TAG,
    fit_kww_enthalpy_curve,
    group_enthalpy_time_rows,
    predict_kww,
)


def test_kww_enthalpy_curve_fit_recovers_monotonic_time_trend() -> None:
    times = np.array([10.0, 30.0, 100.0, 300.0, 900.0, 1800.0])
    y = -0.2 - 1.4 * (1.0 - np.exp(-((times / 300.0) ** 0.55)))

    fit = fit_kww_enthalpy_curve(times, y)
    pred = predict_kww(times, fit)

    assert fit["n_points"] == len(times)
    assert fit["rmse"] < 0.08
    assert np.corrcoef(y, pred)[0, 1] > 0.98


def test_group_enthalpy_time_rows_builds_single_and_two_step_groups() -> None:
    rows = [
        {"mode": "single_step", "T1_C": "80", "t1_s": "10", "T2_C": "", "t2_s": "", "delta_h_total_J_g": "-0.1"},
        {"mode": "single_step", "T1_C": "80", "t1_s": "100", "T2_C": "", "t2_s": "", "delta_h_total_J_g": "-0.5"},
        {"mode": "single_step", "T1_C": "80", "t1_s": "1000", "T2_C": "", "t2_s": "", "delta_h_total_J_g": "-1.0"},
        {"mode": "two_step", "T1_C": "80", "t1_s": "50", "T2_C": "90", "t2_s": "10", "delta_h_total_J_g": "-0.2"},
        {"mode": "two_step", "T1_C": "80", "t1_s": "50", "T2_C": "90", "t2_s": "100", "delta_h_total_J_g": "-0.6"},
        {"mode": "two_step", "T1_C": "80", "t1_s": "50", "T2_C": "90", "t2_s": "1000", "delta_h_total_J_g": "-1.1"},
    ]

    groups = group_enthalpy_time_rows(rows, min_points=3)
    keys = {group["group_key"] for group in groups}

    assert "single_step|T=80" in keys
    assert "two_step|T1=80|t1=50|T2=90" in keys
    single = next(group for group in groups if group["group_key"] == "single_step|T=80")
    two = next(group for group in groups if group["group_key"] == "two_step|T1=80|t1=50|T2=90")
    assert np.allclose(single["time_s"], [10.0, 100.0, 1000.0])
    assert np.allclose(two["time_s"], [10.0, 100.0, 1000.0])


def test_enthalpy_time_figures_are_window_versioned() -> None:
    assert WINDOW_TAG == "70_105"
    assert WINDOW_TAG in SINGLE_FIG.name
    assert WINDOW_TAG in TWOSTEP_FIG.name
