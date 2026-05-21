from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.conditional_time_inverse import (
    PAPER_STYLE_TARGETS,
    candidate_time_rows,
    conditional_posterior_summary,
    rank_time_candidates,
    target_loss,
)


def test_candidate_time_rows_only_change_times_and_keep_temperatures_fixed() -> None:
    rows = candidate_time_rows(50.0, 80.0)

    assert rows
    assert {row["T1_C"] for row in rows} == {"50"}
    assert {row["T2_C"] for row in rows} == {"80"}
    assert len({row["t1_s"] for row in rows}) > 1
    assert len({row["t2_s"] for row in rows}) > 1
    assert min(float(row["t1_s"]) for row in rows) >= 10.0
    assert min(float(row["t2_s"]) for row in rows) >= 10.0
    assert max(float(row["t1_s"]) for row in rows) <= 1800.0
    assert max(float(row["t2_s"]) for row in rows) <= 1800.0


def test_target_loss_uses_only_paper_style_targets() -> None:
    pred = {
        "delta_h_total_J_g": 1.0,
        "peak_area_J_g": 2.0,
        "activation_enthalpy_H_star_kj_mol": 450.0,
        "activation_entropy_S_star_j_mol_K": 900.0,
        "recovery_index": 999.0,
    }
    target = {
        "delta_h_total_J_g": 1.1,
        "peak_area_J_g": 1.8,
        "activation_enthalpy_H_star_kj_mol": 460.0,
        "activation_entropy_S_star_j_mol_K": 870.0,
        "recovery_index": -999.0,
    }

    assert PAPER_STYLE_TARGETS == [
        "delta_h_total_J_g",
        "peak_area_J_g",
        "activation_enthalpy_H_star_kj_mol",
        "activation_entropy_S_star_j_mol_K",
    ]
    assert target_loss(pred, target) == target_loss({k: pred[k] for k in PAPER_STYLE_TARGETS}, target)


def test_conditional_posterior_reports_only_time_dimensions() -> None:
    rows = [
        {"T1_C": "50", "T2_C": "80", "t1_s": "10", "t2_s": "100", "total_anneal_time_s": "110"},
        {"T1_C": "50", "T2_C": "80", "t1_s": "300", "t2_s": "600", "total_anneal_time_s": "900"},
    ]
    posterior = conditional_posterior_summary(rows, np.array([0.1, 0.11]), percentile=100.0)

    assert "log10_t1_s" in posterior
    assert "log10_t2_s" in posterior
    assert "T1_C" not in posterior
    assert "T2_C" not in posterior
    assert posterior["n_valid_candidates"] == 2


def test_rank_time_candidates_loss_components_match_final_loss() -> None:
    rows = [
        {"T1_C": "50", "T2_C": "80", "t1_s": "10", "t2_s": "100", "total_anneal_time_s": "110"},
        {"T1_C": "50", "T2_C": "80", "t1_s": "300", "t2_s": "600", "total_anneal_time_s": "900"},
    ]
    predictions = np.array(
        [
            [1.0, 2.0, 450.0, 900.0],
            [2.0, 3.0, 550.0, 1200.0],
        ],
        dtype=float,
    )
    uncertainties = np.full_like(predictions, 0.1)
    target = {
        "delta_h_total_J_g": 1.0,
        "peak_area_J_g": 2.0,
        "activation_enthalpy_H_star_kj_mol": 450.0,
        "activation_entropy_S_star_j_mol_K": 900.0,
    }

    ranked, losses = rank_time_candidates(rows, predictions, uncertainties, target, top_k=2)

    assert ranked[0]["loss"] == losses[0]
    components = ranked[0]["loss_components"]
    reconstructed = components["target_loss"] + components["time_penalty"] + components["uncertainty_penalty"]
    assert abs(ranked[0]["loss"] - reconstructed) < 1e-12
