from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.eig_design import uncertainty_calibration


def test_uncertainty_calibration_reports_target_fields() -> None:
    y_true = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 5.0], [4.0, 8.0]])
    y_pred = np.array([[0.1, 0.0], [0.8, 1.0], [1.5, 4.0], [3.0, 7.5]])
    unc = np.array([[0.1, 0.1], [0.2, 0.8], [0.5, 1.0], [1.0, 0.5]])

    report = uncertainty_calibration(y_true, y_pred, unc, ["a", "b"])

    assert set(report) == {"a", "b", "summary"}
    assert "mean_abs_uncertainty_error_corr" in report["summary"]
    assert -1.0 <= report["a"]["uncertainty_error_corr"] <= 1.0
