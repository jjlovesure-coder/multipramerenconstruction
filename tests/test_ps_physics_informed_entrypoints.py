from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.inverse_design import MODEL_PATH, TARGETS, _target_loss, candidate_feature_matrix
from src.ps.inverse_design import candidate_rows
from src.ps.train_physics_informed_ps_model import physics_kernel_interpretation, predict_auxiliary_features
from src.ps.train_ps_model import FEATURES as RAW_FEATURES
import numpy as np


def test_inverse_design_defaults_to_best_current_model() -> None:
    assert MODEL_PATH.name == "ps_limited_time_model.json"
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


def test_candidate_feature_matrix_allows_raw_model_payload_without_physics_features() -> None:
    row = candidate_rows()[0]
    payload = {"features": RAW_FEATURES}

    x = candidate_feature_matrix([row], payload)

    assert x.shape == (1, len(RAW_FEATURES))


def test_target_loss_uses_payload_target_order() -> None:
    target_names = [
        "delta_h_total_J_g",
        "peak_area_J_g",
        "peak_temperature_Tp_C",
        "peak_height_uW",
        "recovery_index",
        "path_dependence_index",
    ]
    pred = np.array([-4.0, -3.0, 105.0, -110.0, 0.8, 0.0], dtype=float)

    assert _target_loss(pred, {"recovery_index": 0.8}, target_names) == 0.0


def test_auxiliary_process_prediction_does_not_use_test_measurements() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=float)
    z = np.array([[0.0, 0.0], [1.0, 1.0], [999.0, 999.0]], dtype=float)
    train_idx = np.array([0, 1])
    test_idx = np.array([2])

    _, test_pred = predict_auxiliary_features(x, z, train_idx, test_idx, bandwidth=1.0)

    assert test_pred.shape == (1, 2)
    assert np.all(test_pred < 2.0)
    assert not np.allclose(test_pred, z[test_idx])


def test_physics_kernel_interpretation_does_not_claim_gain_when_worse() -> None:
    summary = {
        "metrics": {
            "raw_kernel": {
                "delta_h_total_J_g": {"mae": 1.0},
                "peak_area_J_g": {"mae": 1.0},
                "recovery_index": {"mae": 1.0},
                "path_dependence_index": {"mae": 1.0},
            },
            "physics_kernel": {
                "delta_h_total_J_g": {"mae": 1.2},
                "peak_area_J_g": {"mae": 1.2},
                "recovery_index": {"mae": 1.2},
                "path_dependence_index": {"mae": 1.2},
            },
        },
        "normalized_core": {
            "raw_kernel": {"mean_mae_over_test_std": 1.0},
            "physics_kernel": {"mean_mae_over_test_std": 1.2},
        },
    }

    text = physics_kernel_interpretation(summary)

    assert "does not improve" in text
    assert "not recommended" in text
    assert "lowers" not in text


if __name__ == "__main__":
    test_inverse_design_defaults_to_physics_informed_model()
    test_candidate_feature_matrix_uses_physics_features()
