from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.calculate_arrt_from_heating_rates import arrt_quality_flag
from src.ps.calculate_arrt_from_heating_rates import calculate_group_results
from src.ps.calculate_arrt_from_heating_rates import add_recovery_indices
from src.ps.calculate_arrt_from_heating_rates import scan_quality_fields
from src.ps.dsc_processing import RAW_FILES
from src.ps.strict_arrt_inverse_reconstruction import posterior_from_losses as strict_arrt_posterior_from_losses
from src.ps.train_strict_arrt_model import default_training_rows, leave_one_condition_out_predictions


def test_hs_02_is_registered_as_raw_dsc_file() -> None:
    assert "hs_02" in RAW_FILES
    assert RAW_FILES["hs_02"].name == "ps-hs-02.xlsx"


def test_single_hs_80_90_file_is_registered() -> None:
    assert "single_hs_80_90_01" in RAW_FILES
    assert RAW_FILES["single_hs_80_90_01"].name == "ps-single-hs-01.xlsx"


def test_scan_quality_marks_missing_arrt_peak_window_as_truncated() -> None:
    quality = scan_quality_fields(n_points=8867, temp_min_c=30.0, temp_max_c=104.24)

    assert quality["covers_arrt_peak_window"] is False
    assert quality["is_truncated"] is True


def test_arrt_results_include_confidence_and_mean_response_fields() -> None:
    rows = []
    for rate, tp, dh, peak in [
        (5.0, 106.0, -1.0, -1.8),
        (10.0, 108.0, -1.1, -1.9),
        (20.0, 110.0, -1.2, -2.0),
    ]:
        rows.append({
            "mode": "single_step",
            "T1_C": 90.0,
            "t1_s": 100.0,
            "T2_C": np.nan,
            "t2_s": np.nan,
            "total_anneal_time_s": 100.0,
            "heating_rate_C_min": rate,
            "arrt_peak_temperature_Tp_C": tp,
            "delta_h_total_J_g": dh,
            "peak_area_J_g": peak,
        })

    calculated, _ = calculate_group_results(rows)

    assert calculated
    row = calculated[0]
    assert "activation_enthalpy_H_star_kj_mol" in row
    assert "activation_entropy_S_star_j_mol_K" in row
    assert "kissinger_confidence" in row
    assert "mean_delta_h_total_J_g" in row
    assert "mean_peak_area_J_g" in row
    assert "mean_peak_height_uW" in row
    assert row["quality_flag"] == "ok"


def test_arrt_quality_flags_90_c_500_s_for_retest() -> None:
    flag, note = arrt_quality_flag({"mode": "single_step", "T1_C": 90.0, "t1_s": 500.0})

    assert flag == "suspect_retest_required"
    assert "retest" in note


def test_strict_arrt_default_training_rows_exclude_suspect_quality_flag() -> None:
    rows = [
        {
            "activation_enthalpy_H_star_kj_mol": "500",
            "activation_entropy_S_star_j_mol_K": "1000",
            "quality_flag": "ok",
        },
        {
            "activation_enthalpy_H_star_kj_mol": "470",
            "activation_entropy_S_star_j_mol_K": "1029",
            "quality_flag": "suspect_retest_required",
        },
    ]

    selected = default_training_rows(rows)

    assert len(selected) == 1
    assert selected[0]["quality_flag"] == "ok"


def test_incomplete_peak_window_rows_do_not_enter_kissinger_fit() -> None:
    rows = []
    for rate, tp, valid in [
        (5.0, 106.0, True),
        (10.0, 108.0, True),
        (20.0, 110.0, False),
        (40.0, 112.0, False),
    ]:
        rows.append({
            "mode": "single_step",
            "T1_C": 90.0,
            "t1_s": 1000.0,
            "T2_C": np.nan,
            "t2_s": np.nan,
            "total_anneal_time_s": 1000.0,
            "heating_rate_C_min": rate,
            "arrt_peak_temperature_Tp_C": tp if valid else np.nan,
            "covers_arrt_peak_window": valid,
            "delta_h_total_J_g": -1.0,
            "peak_area_J_g": -1.8,
            "peak_height_uW": -100.0,
            "peak_temperature_Tp_C": 104.0,
        })

    calculated, diagnostics = calculate_group_results(rows)

    assert calculated == []
    assert diagnostics[0]["can_calculate_arrt_kissinger"] is False
    assert diagnostics[0]["n_valid_peak_rates"] == 2


def test_arrt_recovery_index_is_added_before_grouping() -> None:
    rows = [
        {"mode": "single_step", "T1_C": 80.0, "delta_h_total_J_g": -1.0},
        {"mode": "single_step", "T1_C": 80.0, "delta_h_total_J_g": -2.0},
    ]

    add_recovery_indices(rows)

    assert rows[0]["recovery_index"] == 0.5
    assert rows[1]["recovery_index"] == 1.0


def test_strict_arrt_loocv_runs_without_source_file() -> None:
    rows = [
        {
            "condition_key": f"cond_{idx}",
            "mode": "single_step",
            "T1_C": str(70 + idx * 5),
            "t1_s": "100",
            "T2_C": "",
            "t2_s": "",
            "total_anneal_time_s": "100",
            "activation_enthalpy_H_star_kj_mol": str(400 + idx * 10),
            "activation_entropy_S_star_j_mol_K": str(900 + idx * 20),
            "kissinger_confidence": "high",
        }
        for idx in range(5)
    ]

    result = leave_one_condition_out_predictions(rows)

    assert result["n_samples"] == 5
    assert len(result["predictions"]) == 5
    assert "activation_enthalpy_H_star_kj_mol" in result["metrics"]


def test_strict_arrt_inverse_posterior_schema_matches_existing_inverse() -> None:
    rows = [
        {"T1_C": "50", "t1_s": "10", "T2_C": "80", "t2_s": "100", "total_anneal_time_s": "110"},
        {"T1_C": "60", "t1_s": "300", "T2_C": "90", "t2_s": "600", "total_anneal_time_s": "900"},
    ]
    posterior = strict_arrt_posterior_from_losses(rows, np.array([0.1, 0.12], dtype=float), percentile=100.0)

    assert "T1_C" in posterior
    assert "log10_t1_s" in posterior
    assert "n_valid_candidates" in posterior
    assert posterior["n_valid_candidates"] == 2
