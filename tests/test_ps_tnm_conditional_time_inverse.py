from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.evaluate_tnm_conditional_time_inverse import candidate_rows_for_fixed_temperatures, summarize_by_pair


def test_tnm_conditional_candidates_keep_temperatures_fixed_and_use_supplied_times() -> None:
    rows = candidate_rows_for_fixed_temperatures(80.0, 90.0, [0.1, 10.0, 100.0])

    assert len(rows) == 9
    assert {row["T1_C"] for row in rows} == {"80"}
    assert {row["T2_C"] for row in rows} == {"90"}
    assert {row["t1_s"] for row in rows} == {"0.1", "10", "100"}
    assert {row["t2_s"] for row in rows} == {"0.1", "10", "100"}


def test_summarize_by_pair_reports_80_90_pairs() -> None:
    rows = [
        {
            "pair": "80->90",
            "abs_log_error_t1": 0.1,
            "abs_log_error_t2": 0.2,
            "t1_factor_error": 1.25,
            "t2_factor_error": 1.5,
            "top5_time_near_match": True,
            "posterior_width_log10_t1": 0.4,
            "posterior_width_log10_t2": 0.5,
        },
        {
            "pair": "80->90",
            "abs_log_error_t1": 0.3,
            "abs_log_error_t2": 0.4,
            "t1_factor_error": 2.0,
            "t2_factor_error": 3.0,
            "top5_time_near_match": False,
            "posterior_width_log10_t1": 0.6,
            "posterior_width_log10_t2": 0.7,
        },
    ]

    summary = summarize_by_pair(rows)

    assert summary[0]["pair"] == "80->90"
    assert summary[0]["n"] == 2
    assert summary[0]["t1_log10_MAE"] == 0.2
    assert summary[0]["top5_time_near_match"] == 0.5
