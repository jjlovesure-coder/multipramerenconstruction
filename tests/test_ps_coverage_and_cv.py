from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.train_ps_model import coverage_summary, repeated_grouped_cv_summary


def test_coverage_summary_marks_missing_high_temperature_region() -> None:
    rows = [
        {"mode": "two_step", "source_file": "a", "T1_C": "50", "t1_s": "100", "T2_C": "70", "t2_s": "100", "total_anneal_time_s": "200", "delta_h_total_J_g": "-1", "peak_area_J_g": "-1", "peak_temperature_Tp_C": "100", "peak_height_uW": "-10", "recovery_index": "0.5", "path_dependence_index": "0.1"},
        {"mode": "two_step", "source_file": "b", "T1_C": "80", "t1_s": "100", "T2_C": "90", "t2_s": "100", "total_anneal_time_s": "200", "delta_h_total_J_g": "-1", "peak_area_J_g": "-1", "peak_temperature_Tp_C": "100", "peak_height_uW": "-10", "recovery_index": "0.5", "path_dependence_index": "0.1"},
    ]

    summary = coverage_summary(rows)

    assert summary["T1_bin"]["95-100"]["is_low_support_region"] is True
    assert summary["T1_bin"]["50-60"]["n"] == 1


def test_repeated_grouped_cv_summary_reports_mean_and_std() -> None:
    rows = []
    for idx, temp in enumerate([50, 60, 70, 80, 90, 100]):
        rows.append({
            "sample_id": str(idx),
            "source_file": f"f{idx}",
            "mode": "single_step",
            "T1_C": str(temp),
            "t1_s": "100",
            "T2_C": "",
            "t2_s": "",
            "total_anneal_time_s": "100",
            "delta_h_total_J_g": str(-idx - 1),
            "peak_area_J_g": str(-idx - 0.5),
            "peak_temperature_Tp_C": str(95 + idx),
            "peak_height_uW": str(-10 - idx),
            "recovery_index": str(0.1 * idx),
            "path_dependence_index": "0",
        })

    summary = repeated_grouped_cv_summary(rows, seeds=(1, 2), n_folds=3)

    assert summary["n_scores"] > 0
    assert "mean_metrics" in summary
    assert "std_metrics" in summary
