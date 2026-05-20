from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.train_ps_model import FEATURES, TARGETS, featurize


def test_degenerate_ps_targets_are_not_trained() -> None:
    assert "kovacs_peak_label" not in TARGETS
    assert "onset_temperature_C" not in TARGETS
    assert "recovery_index" in TARGETS
    assert "path_dependence_index" in TARGETS


def test_ps_raw_features_include_time_path_interactions() -> None:
    expected = {
        "T1_logt1",
        "T2_logt2",
        "delta_T_log_time_ratio",
        "inv_T_diff",
        "log_t_ratio",
        "step_time_asymmetry",
    }
    assert expected <= set(FEATURES)

    row = {
        "mode": "two_step",
        "T1_C": "70",
        "t1_s": "100",
        "T2_C": "90",
        "t2_s": "1000",
        "total_anneal_time_s": "1100",
    }
    values = featurize(row)
    feature_map = dict(zip(FEATURES, values))

    assert len(values) == len(FEATURES)
    assert feature_map["delta_T_log_time_ratio"] > 0.0
    assert feature_map["log_t_ratio"] == 1.0
    assert feature_map["step_time_asymmetry"] > 0.0
