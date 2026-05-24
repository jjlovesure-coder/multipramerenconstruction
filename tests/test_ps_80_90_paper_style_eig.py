from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.eig_80_90_paper_style_design import candidate_conditions


def test_candidate_conditions_are_limited_to_80_90_pairs() -> None:
    rows = candidate_conditions()

    assert rows
    assert {(row["T1_C"], row["T2_C"]) for row in rows} == {(80.0, 90.0), (90.0, 80.0)}


def test_candidate_conditions_require_multirate_arrt_measurement() -> None:
    rows = candidate_conditions()

    assert all(row["requires_multi_rate_arrt"] for row in rows)
    assert all(row["heating_rates_C_min"] == "5,10,20,40" for row in rows)
    assert min(min(row["t1_s"], row["t2_s"]) for row in rows) >= 10.0
