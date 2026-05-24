from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evaluate_paper_conditional_time_inverse import (
    FIG2_FIG3_TARGETS,
    candidate_rows_for_fixed_temperatures,
    time_posterior_summary,
)


def test_candidate_rows_keep_temperatures_fixed() -> None:
    rows = candidate_rows_for_fixed_temperatures(348.0, 383.0, [0.1, 1.0])

    assert len(rows) == 4
    assert {row["t1_temperature_k"] for row in rows} == {348.0}
    assert {row["t2_temperature_k"] for row in rows} == {383.0}
    assert {row["t1_s"] for row in rows} == {0.1, 1.0}
    assert {row["t2_s"] for row in rows} == {0.1, 1.0}


def test_fig2_fig3_target_set_contains_kinetic_constraints() -> None:
    assert "delta_h_kj_mol" in FIG2_FIG3_TARGETS
    assert "delta_h_peak_kj_mol" in FIG2_FIG3_TARGETS
    assert "s1_j_mol_k" in FIG2_FIG3_TARGETS
    assert "s2_j_mol_k" in FIG2_FIG3_TARGETS
    assert "h1_kj_mol_fig3" in FIG2_FIG3_TARGETS
    assert "h2_kj_mol_fig3" in FIG2_FIG3_TARGETS


def test_time_posterior_reports_only_time_dimensions() -> None:
    rows = candidate_rows_for_fixed_temperatures(348.0, 383.0, [0.1, 1.0])
    posterior = time_posterior_summary(rows, [0.0, 1.0, 1.2, 1.4])

    assert "log10_t1_s" in posterior
    assert "log10_t2_s" in posterior
    assert "T1_C" not in posterior
    assert "T2_C" not in posterior
