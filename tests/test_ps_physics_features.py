from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.physics_features import physics_feature_dict


def test_physics_dose_increases_with_temperature_at_fixed_time() -> None:
    low = physics_feature_dict({"mode": "single_step", "T1_C": "70", "t1_s": "300", "T2_C": "", "t2_s": ""})
    high = physics_feature_dict({"mode": "single_step", "T1_C": "100", "t1_s": "300", "T2_C": "", "t2_s": ""})
    assert high["tnm_state_total_e160_b050"] > low["tnm_state_total_e160_b050"]


def test_two_step_state_is_sequential_and_path_sensitive() -> None:
    row = {"mode": "two_step", "T1_C": "80", "t1_s": "300", "T2_C": "100", "t2_s": "300"}
    features = physics_feature_dict(row)
    assert features["tnm_state_total_e160_b050"] >= features["tnm_state_step1_e160_b050"]
    assert abs(features["tnm_path_excess_e160_b050"]) > 1e-9


if __name__ == "__main__":
    test_physics_dose_increases_with_temperature_at_fixed_time()
    test_two_step_state_is_sequential_and_path_sensitive()
