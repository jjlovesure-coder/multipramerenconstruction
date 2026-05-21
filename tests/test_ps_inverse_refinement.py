from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.inverse_design import (
    candidate_rows,
    combined_inverse_loss,
    filter_candidates_by_target,
    interpretation_from_posterior,
    load_inverse_diagnostics,
    pareto_front,
    posterior_summary,
    t2_candidates_from_target,
)


def test_pareto_front_keeps_non_dominated_candidates() -> None:
    rows = [
        {
            "target_loss": 0.5,
            "total_anneal_time_s": "100",
            "mean_uncertainty": 0.2,
            "extrapolation_penalty": 0.1,
            "inverse_identifiability": 0.8,
        },
        {
            "target_loss": 0.4,
            "total_anneal_time_s": "200",
            "mean_uncertainty": 0.2,
            "extrapolation_penalty": 0.1,
            "inverse_identifiability": 0.8,
        },
        {
            "target_loss": 0.6,
            "total_anneal_time_s": "300",
            "mean_uncertainty": 0.5,
            "extrapolation_penalty": 0.2,
            "inverse_identifiability": 0.2,
        },
    ]

    front = pareto_front(rows)

    assert len(front) == 2
    assert rows[2] not in front


def test_candidate_rows_use_experimental_time_floor() -> None:
    rows = candidate_rows()

    times = []
    for row in rows:
        times.append(float(row["t1_s"]))
        if row["mode"] == "two_step":
            times.append(float(row["t2_s"]))

    assert min(times) >= 10.0


def test_t2_candidates_are_soft_constrained_by_tp_when_available() -> None:
    constrained = t2_candidates_from_target({"peak_temperature_Tp_C": 86.0})
    unconstrained = t2_candidates_from_target({"recovery_index": 0.8})

    assert constrained == [80, 85, 90]
    assert len(unconstrained) > len(constrained)


def test_combined_inverse_loss_prefers_identifiable_supported_candidate() -> None:
    supported = combined_inverse_loss(
        target_loss=0.25,
        total_time_s=600.0,
        mean_uncertainty=0.1,
        extrapolation_penalty=0.0,
        inverse_identifiability=0.9,
    )
    unsupported = combined_inverse_loss(
        target_loss=0.25,
        total_time_s=600.0,
        mean_uncertainty=0.1,
        extrapolation_penalty=1.0,
        inverse_identifiability=0.1,
    )

    assert supported < unsupported


def test_posterior_summary_reports_quantiles_for_flat_losses() -> None:
    rows = [
        {"T1_C": "50", "t1_s": "10", "T2_C": "70", "t2_s": "30", "total_anneal_time_s": "40"},
        {"T1_C": "60", "t1_s": "100", "T2_C": "80", "t2_s": "300", "total_anneal_time_s": "400"},
        {"T1_C": "70", "t1_s": "1000", "T2_C": "90", "t2_s": "900", "total_anneal_time_s": "1900"},
    ]
    losses = np.array([1.0, 1.01, 1.02], dtype=float)

    summary = posterior_summary(rows, losses, percentile=100.0)

    assert summary["n_valid_candidates"] == 3
    assert summary["T1_C"]["p5"] < summary["T1_C"]["p95"]
    assert summary["log10_t1_s"]["median"] == 2.0
    assert summary["loss_flatness_ratio"] > 0.0


def test_inverse_diagnostics_raise_instead_of_silent_zero_fallback(monkeypatch) -> None:
    import src.ps.eig_design as eig_design

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(eig_design, "inverse_identifiability_scores", fail)

    try:
        load_inverse_diagnostics(
            [{"mode": "single_step", "T1_C": "80", "t1_s": "100", "T2_C": "", "t2_s": "", "total_anneal_time_s": "100"}],
            {"targets": ["recovery_index"], "x_train": [[0.0]], "y_train": [[0.0]], "x_mean": [0.0], "x_std": [1.0], "bandwidth": 1.0},
            ["recovery_index"],
            allow_degraded_diagnostics=False,
        )
    except RuntimeError as exc:
        assert "diagnostics unavailable" in str(exc)
    else:
        raise AssertionError("diagnostic failures must not silently fall back to zeros")


def test_inverse_diagnostics_can_explicitly_degrade(monkeypatch) -> None:
    import src.ps.eig_design as eig_design

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(eig_design, "inverse_identifiability_scores", fail)
    diagnostics = load_inverse_diagnostics(
        [{"mode": "single_step", "T1_C": "80", "t1_s": "100", "T2_C": "", "t2_s": "", "total_anneal_time_s": "100"}],
        {"targets": ["recovery_index"], "x_train": [[0.0]], "y_train": [[0.0]], "x_mean": [0.0], "x_std": [1.0], "bandwidth": 1.0},
        ["recovery_index"],
        allow_degraded_diagnostics=True,
    )

    assert diagnostics["diagnostics_enabled"] is False
    assert np.all(diagnostics["inverse_identifiability"] == 0.0)


def test_filter_candidates_by_target_hard_filters_t2_from_tp() -> None:
    rows = candidate_rows({"peak_temperature_Tp_C": 86.0})
    filtered, mask = filter_candidates_by_target(rows, {"peak_temperature_Tp_C": 86.0})

    assert len(filtered) == int(np.sum(mask))
    assert all(row["mode"] == "single_step" or int(float(row["T2_C"])) in {80, 85, 90} for row in filtered)


def test_interpretation_marks_wide_posterior_non_identifiable() -> None:
    interpretation = interpretation_from_posterior({
        "T1_C": {"width": 50.0},
        "T2_C": {"width": 50.0},
        "log10_t1_s": {"width": 2.0},
        "log10_t2_s": {"width": 2.0},
        "loss_flatness_ratio": 3.0,
    })

    assert interpretation["status"] == "non_identifiable"
    assert "posterior" in interpretation["reason"]
