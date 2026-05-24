from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.tnm_curve_fit import (
    CURVE_TARGETS,
    leave_one_out_curve_fit,
    observed_curve_parameters,
    predicted_curve,
)


def test_observed_curve_parameters_use_40_to_100_c_window() -> None:
    curve = pd.DataFrame(
        {
            "Temp_C": [40, 60, 70, 75, 80, 85, 90, 95, 100],
            "DSC_detrended_uW": [0, 0, 0, -1, -4, -9, -5, -2, 0],
        }
    )

    params = observed_curve_parameters(curve, heating_rate_c_min=10.0)

    assert set(CURVE_TARGETS) <= set(params)
    assert 70.0 <= params["tp_c"] <= 100.0
    assert params["width_c"] > 0.0


def test_predicted_curve_returns_finite_shape() -> None:
    temp = np.linspace(40.0, 100.0, 50)

    y = predicted_curve(temp, area_rel=-100.0, tp_c=85.0, height_uW=-10.0, log_width_c=np.log(8.0))

    assert y.shape == temp.shape
    assert np.isfinite(y).all()
    assert y.min() < 0.0


def test_leave_one_out_curve_fit_reports_curve_metrics() -> None:
    rows = []
    temp = np.linspace(40.0, 100.0, 61)
    for idx, time_s in enumerate([50.0, 100.0, 300.0, 500.0]):
        center = 82.0 + idx
        y = -10.0 * np.exp(-0.5 * ((temp - center) / 8.0) ** 2)
        rows.append(
            {
                "sample_id": f"s{idx}",
                "source_file": "synthetic.xlsx",
                "mode": "single_step",
                "T1_C": 80.0 + idx,
                "t1_s": time_s,
                "T2_C": np.nan,
                "t2_s": np.nan,
                "total_anneal_time_s": time_s,
                "heating_rate_C_min": 10.0,
                "curve_temp_C": temp,
                "curve_y_uW": y,
                "area_rel": float(np.trapezoid(y, temp)),
                "tp_c": center,
                "height_uW": -10.0,
                "width_c": 8.0,
                "log_width_c": float(np.log(8.0)),
                "area_J_g": -1.0,
            }
        )

    records = leave_one_out_curve_fit(rows)

    assert len(records) == 4
    assert "curve_rmse_uW" in records[0]
    assert "area_error_J_g" in records[0]
    assert records[0]["n_points_fit"] == len(temp)
