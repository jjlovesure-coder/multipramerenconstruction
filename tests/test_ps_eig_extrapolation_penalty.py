from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.eig_design import extrapolation_penalty, support_from_existing_rows


def test_eig_extrapolation_penalty_discourages_unobserved_down_jump() -> None:
    existing = [
        {"mode": "two_step", "T1_C": "80", "T2_C": "90"},
        {"mode": "two_step", "T1_C": "90", "T2_C": "80"},
        {"mode": "single_step", "T1_C": "60", "T2_C": ""},
    ]
    support = support_from_existing_rows(existing)

    inside = {"mode": "two_step", "T1_C": "80", "T2_C": "90"}
    outside = {"mode": "two_step", "T1_C": "100", "T2_C": "50"}

    assert extrapolation_penalty([inside], support)[0] == 0.0
    assert extrapolation_penalty([outside], support)[0] > 0.0
