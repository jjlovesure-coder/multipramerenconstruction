from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.inverse_design import MODEL_PATH, TARGETS, candidate_feature_matrix
from src.ps.inverse_design import candidate_rows


def test_inverse_design_defaults_to_physics_informed_model() -> None:
    assert MODEL_PATH.name == "ps_physics_informed_kernel_model.json"
    assert "path_dependence_index" in TARGETS
    assert "kovacs_peak_label" not in TARGETS


def test_candidate_feature_matrix_uses_physics_features() -> None:
    row = candidate_rows()[0]
    x = candidate_feature_matrix([row])
    assert x.shape[0] == 1
    assert x.shape[1] > 7


if __name__ == "__main__":
    test_inverse_design_defaults_to_physics_informed_model()
    test_candidate_feature_matrix_uses_physics_features()
