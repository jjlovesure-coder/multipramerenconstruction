from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.kernel_regression import KernelRegressor


def test_scalar_bandwidth_matches_equivalent_ard_vector() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.5], [3.0, 3.0]])
    y = np.column_stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]])
    scalar = KernelRegressor.fit(x, y, bandwidth=1.4)
    ard = KernelRegressor.fit(x, y, bandwidth=[1.4, 1.4])

    scalar_pred, scalar_unc = scalar.predict(np.array([[1.5, 1.0], [2.5, 2.0]]))
    ard_pred, ard_unc = ard.predict(np.array([[1.5, 1.0], [2.5, 2.0]]))

    assert np.allclose(scalar_pred, ard_pred)
    assert np.allclose(scalar_unc, ard_unc)


def test_ard_bandwidth_dimension_must_match_features() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0]])
    try:
        KernelRegressor.fit(x, y, bandwidth=[1.0, 2.0, 3.0])
    except ValueError as exc:
        assert "bandwidth" in str(exc)
    else:
        raise AssertionError("Expected invalid ARD bandwidth to raise ValueError")


def test_ard_prediction_is_finite() -> None:
    x = np.array([[0.0, 0.0], [10.0, 1.0], [20.0, 2.0]])
    y = np.array([[0.0], [1.0], [0.0]])
    model = KernelRegressor.fit(x, y, bandwidth=[0.8, 2.0])
    pred, unc = model.predict(np.array([[5.0, 0.5], [15.0, 1.5]]))

    assert np.isfinite(pred).all()
    assert np.isfinite(unc).all()
