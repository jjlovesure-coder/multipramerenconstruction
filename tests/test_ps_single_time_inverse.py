from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.evaluate_single_time_inverse import candidate_times_by_temperature
from src.ps.evaluate_single_time_inverse import direct_fit_predict_log_time
from src.ps.evaluate_single_time_inverse import strict_single_rows


def test_single_time_inverse_candidates_use_only_strict_valid_rows() -> None:
    rows = strict_single_rows([
        {
            "mode": "single_step",
            "T1_C": "80",
            "t1_s": "50",
            "activation_enthalpy_H_star_kj_mol": "430",
            "activation_entropy_S_star_j_mol_K": "900",
            "mean_delta_h_total_J_g": "-1.0",
            "mean_peak_area_J_g": "-1.5",
            "mean_peak_temperature_Tp_C": "104",
            "mean_peak_height_uW": "-100",
            "mean_recovery_index": "0.5",
        },
        {
            "mode": "single_step",
            "T1_C": "90",
            "t1_s": "1000",
            "activation_enthalpy_H_star_kj_mol": "",
            "activation_entropy_S_star_j_mol_K": "",
            "mean_delta_h_total_J_g": "-1.0",
            "mean_peak_area_J_g": "-1.5",
            "mean_peak_temperature_Tp_C": "104",
            "mean_peak_height_uW": "-100",
            "mean_recovery_index": "0.5",
        },
    ])

    assert len(rows) == 1
    assert candidate_times_by_temperature(rows) == {"80": [50.0]}


def test_direct_fit_prediction_can_return_continuous_time() -> None:
    train_rows = [
        {"feature": "1.0", "t1_s": "10"},
        {"feature": "3.0", "t1_s": "1000"},
    ]
    pred_log = direct_fit_predict_log_time(train_rows, {"feature": "2.0"}, ["feature"], clamp_bounds=(1.0, 3.0))

    assert 1.9 < pred_log < 2.1
