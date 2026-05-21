from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.inverse_design import MODEL_PATH, TARGETS, candidate_feature_matrix
from src.ps.inverse_design import candidate_rows
from src.ps.train_physics_informed_ps_model import predict_auxiliary_features
from src.ps.train_ps_model import FEATURES as RAW_FEATURES
import numpy as np


def test_inverse_design_defaults_to_physics_informed_model() -> None:
    assert MODEL_PATH.name == "ps_physics_informed_kernel_model.json"
    assert "path_dependence_index" in TARGETS
    assert "kovacs_peak_label" not in TARGETS


def test_candidate_feature_matrix_uses_physics_features() -> None:
    row = candidate_rows()[0]
    x = candidate_feature_matrix([row])
    assert x.shape[0] == 1
    assert x.shape[1] > 7


def test_candidate_feature_matrix_respects_payload_raw_features() -> None:
    row = candidate_rows()[0]
    payload = {
        "raw_features": RAW_FEATURES,
        "physics_features": ["phys_log_t1_s", "phys_log_total_time_s"],
    }
    x = candidate_feature_matrix([row], payload)
    assert x.shape == (1, len(RAW_FEATURES) + 2)


def test_auxiliary_process_prediction_does_not_use_test_measurements() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=float)
    z = np.array([[0.0, 0.0], [1.0, 1.0], [999.0, 999.0]], dtype=float)
    train_idx = np.array([0, 1])
    test_idx = np.array([2])

    _, test_pred = predict_auxiliary_features(x, z, train_idx, test_idx, bandwidth=1.0)

    assert test_pred.shape == (1, 2)
    assert np.all(test_pred < 2.0)
    assert not np.allclose(test_pred, z[test_idx])


if __name__ == "__main__":
    test_inverse_design_defaults_to_physics_informed_model()
    test_candidate_feature_matrix_uses_physics_features()
