from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.tnm_calibrated_model import (
    TNM_FEATURES,
    TnmParameters,
    activation_entropy_from_preexponential,
    activation_entropy_from_tau_ref,
    design_matrix,
    fit_linear_head,
    grouped_cv_folds,
    predict_tnm_targets,
    select_parameters_repeated,
    tnm_fictive_temperature_features,
    tnm_state_features,
    tnm_tau_s,
    parameter_grid,
)
from src.ps.tnm_inverse_reconstruction import _target_loss as tnm_inverse_target_loss
from src.ps.tnm_inverse_reconstruction import interval_confidence, overall_confidence
from src.ps.tnm_inverse_reconstruction import posterior_from_losses as tnm_posterior_from_losses


def test_tnm_state_increases_with_annealing_time() -> None:
    params = TnmParameters(activation_energy_kj_mol=460.0, beta=0.5, log10_tau_ref_s=2.0)
    short = tnm_state_features({"mode": "single_step", "T1_C": "90", "t1_s": "10"}, params)
    long = tnm_state_features({"mode": "single_step", "T1_C": "90", "t1_s": "1000"}, params)

    assert long["tnm_total_state"] > short["tnm_total_state"]
    assert 0.0 <= short["tnm_total_state"] <= 1.0
    assert 0.0 <= long["tnm_total_state"] <= 1.0


def test_tnm_linear_head_predicts_expected_target_shape() -> None:
    params = TnmParameters(activation_energy_kj_mol=460.0, beta=0.5, log10_tau_ref_s=2.0)
    rows = [
        {"mode": "single_step", "T1_C": "80", "t1_s": "10", "T2_C": "", "t2_s": "", "total_anneal_time_s": "10"},
        {"mode": "single_step", "T1_C": "90", "t1_s": "100", "T2_C": "", "t2_s": "", "total_anneal_time_s": "100"},
        {"mode": "two_step", "T1_C": "80", "t1_s": "100", "T2_C": "100", "t2_s": "60", "total_anneal_time_s": "160"},
    ]
    targets = ["delta_h_total_J_g", "recovery_index"]
    y = np.array([[-0.3, 0.3], [-0.8, 0.8], [-1.0, 1.0]], dtype=float)
    x = design_matrix(rows, params)
    head = fit_linear_head(x, y, ridge=1e-6)
    pred = predict_tnm_targets(rows, params, head)

    assert pred.shape == (3, 2)
    assert head["coef"].shape == (x.shape[1] + 1, len(targets))
    assert np.isfinite(pred).all()


def test_tnm_state_exposes_path_interaction_terms() -> None:
    params = TnmParameters(activation_energy_kj_mol=460.0, beta=0.5, log10_tau_ref_s=2.0)
    row = {"mode": "two_step", "T1_C": "80", "t1_s": "100", "T2_C": "100", "t2_s": "60"}
    features = tnm_state_features(row, params)

    assert "tnm_dose_product" in features
    assert "tnm_dose_asymmetry" in features
    assert features["tnm_dose_product"] >= 0.0
    assert features["tnm_dose_asymmetry"] >= 0.0


def test_tnm_state_exposes_independent_time_resolution_features() -> None:
    expected = {
        "tnm_log_t1_s",
        "tnm_log_t2_s",
        "tnm_t1_fraction_of_total",
        "tnm_t2_fraction_of_total",
        "tnm_time_asymmetry",
    }
    assert expected <= set(TNM_FEATURES)

    params = TnmParameters(activation_energy_kj_mol=460.0, beta=0.5, log10_tau_ref_s=2.0)
    short = tnm_state_features({"mode": "two_step", "T1_C": "90", "t1_s": "10", "T2_C": "90", "t2_s": "10"}, params)
    long = tnm_state_features({"mode": "two_step", "T1_C": "90", "t1_s": "1000", "T2_C": "90", "t2_s": "10"}, params)

    assert long["tnm_log_t1_s"] > short["tnm_log_t1_s"]
    assert long["tnm_t1_fraction_of_total"] > short["tnm_t1_fraction_of_total"]


def test_grouped_cv_folds_do_not_split_source_mode_temperature_groups() -> None:
    rows = [
        {"mode": "two_step", "source_file": "a", "T1_C": "80"},
        {"mode": "two_step", "source_file": "a", "T1_C": "80"},
        {"mode": "two_step", "source_file": "b", "T1_C": "90"},
        {"mode": "single_step", "source_file": "c", "T1_C": "70"},
    ]
    train_idx = np.arange(len(rows), dtype=int)

    folds = grouped_cv_folds(rows, train_idx, n_folds=3, seed=1)

    for inner_train, val in folds:
        train_groups = {(rows[int(i)]["mode"], rows[int(i)]["source_file"], rows[int(i)]["T1_C"]) for i in inner_train}
        val_groups = {(rows[int(i)]["mode"], rows[int(i)]["source_file"], rows[int(i)]["T1_C"]) for i in val}
        assert train_groups.isdisjoint(val_groups)


def test_arrt_entropy_round_trips_tau_ref() -> None:
    params = TnmParameters(activation_energy_kj_mol=220.0, beta=0.75, log10_tau_ref_s=2.0)
    entropy = activation_entropy_from_tau_ref(params)
    tau = tnm_tau_s(373.15, 373.15, params)

    assert np.isclose(tau, params.tau_ref_s, rtol=1e-10)
    assert np.isfinite(entropy)
    assert np.isclose(entropy, activation_entropy_from_preexponential(params.preexponential_A_s), rtol=1e-10)


def test_tnm_fictive_temperature_features_track_more_relaxation() -> None:
    params = TnmParameters(
        activation_energy_kj_mol=220.0,
        beta=0.75,
        log10_tau_ref_s=2.0,
        nonlinearity_x=0.85,
        initial_fictive_temperature_k=393.15,
    )
    short = tnm_fictive_temperature_features({"mode": "single_step", "T1_C": "90", "t1_s": "10"}, params)
    long = tnm_fictive_temperature_features({"mode": "single_step", "T1_C": "90", "t1_s": "1000"}, params)

    assert long["tnm_tf_normalized_recovery"] > short["tnm_tf_normalized_recovery"]
    assert long["tnm_tf_end_k"] < short["tnm_tf_end_k"]
    assert np.isfinite(list(long.values())).all()


def test_repeated_parameter_selection_reports_stability() -> None:
    rows = [
        {"mode": "single_step", "source_file": "a", "T1_C": "80", "t1_s": "10", "T2_C": "", "t2_s": "", "total_anneal_time_s": "10"},
        {"mode": "single_step", "source_file": "a", "T1_C": "90", "t1_s": "100", "T2_C": "", "t2_s": "", "total_anneal_time_s": "100"},
        {"mode": "two_step", "source_file": "b", "T1_C": "80", "t1_s": "100", "T2_C": "100", "t2_s": "60", "total_anneal_time_s": "160"},
        {"mode": "two_step", "source_file": "c", "T1_C": "70", "t1_s": "300", "T2_C": "90", "t2_s": "300", "total_anneal_time_s": "600"},
    ]
    y = np.array([[-0.3, -0.6, 0.3, 0.0], [-0.8, -1.2, 0.8, 0.0], [-1.0, -1.5, 1.0, 0.2], [-0.9, -1.4, 0.9, 0.1]], dtype=float)
    candidates = [
        TnmParameters(300.0, 0.5, 2.0),
        TnmParameters(460.0, 0.5, 2.0),
    ]

    selected = select_parameters_repeated(rows, y, seeds=(1, 2), candidates=candidates)

    assert isinstance(selected["params"], TnmParameters)
    assert selected["score_std"] >= 0.0
    assert selected["n_repeats"] == 2
    assert selected["cv"][0]["score_mean"] <= selected["cv"][-1]["score_mean"]


def test_repeated_parameter_selection_reuses_design_matrix(monkeypatch) -> None:
    import src.ps.tnm_calibrated_model as tnm_model

    rows = [
        {"mode": "single_step", "source_file": "a", "T1_C": "80", "t1_s": "10", "T2_C": "", "t2_s": "", "total_anneal_time_s": "10"},
        {"mode": "single_step", "source_file": "b", "T1_C": "90", "t1_s": "100", "T2_C": "", "t2_s": "", "total_anneal_time_s": "100"},
        {"mode": "two_step", "source_file": "c", "T1_C": "80", "t1_s": "100", "T2_C": "100", "t2_s": "60", "total_anneal_time_s": "160"},
        {"mode": "two_step", "source_file": "d", "T1_C": "70", "t1_s": "300", "T2_C": "90", "t2_s": "300", "total_anneal_time_s": "600"},
    ]
    y = np.array([[-0.3, -0.6, 0.3, 0.0], [-0.8, -1.2, 0.8, 0.0], [-1.0, -1.5, 1.0, 0.2], [-0.9, -1.4, 0.9, 0.1]], dtype=float)
    candidates = [
        TnmParameters(300.0, 0.5, 2.0),
        TnmParameters(460.0, 0.5, 2.0),
    ]
    calls = {"count": 0}
    original = tnm_model.design_matrix

    def counted_design_matrix(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tnm_model, "design_matrix", counted_design_matrix)

    tnm_model.select_parameters_repeated(rows, y, seeds=(1, 2), candidates=candidates)

    assert calls["count"] <= len(candidates)


def test_tnm_parameter_grid_defaults_to_fast_profile() -> None:
    assert len(parameter_grid("predictive_unconstrained")) < len(parameter_grid("predictive_unconstrained", profile="exhaustive"))
    assert len(parameter_grid("arrt_constrained")) < len(parameter_grid("arrt_constrained", profile="exhaustive"))


def test_tnm_inverse_loss_prefers_closer_target() -> None:
    target = np.array([-0.5, -0.3, 0.5, 0.1], dtype=float)
    close = np.array([-0.52, -0.31, 0.51, 0.11], dtype=float)
    far = np.array([-1.0, -0.8, 0.9, -0.3], dtype=float)

    assert tnm_inverse_target_loss(close, target) < tnm_inverse_target_loss(far, target)


def test_tnm_inverse_posterior_matches_kernel_inverse_schema() -> None:
    rows = [
        {"T1_C": "50", "t1_s": "10", "T2_C": "70", "t2_s": "30", "total_anneal_time_s": "40"},
        {"T1_C": "60", "t1_s": "100", "T2_C": "80", "t2_s": "300", "total_anneal_time_s": "400"},
    ]
    posterior = tnm_posterior_from_losses(rows, np.array([0.1, 0.11], dtype=float), percentile=100.0)

    assert "T1_C" in posterior
    assert "log10_t2_s" in posterior
    assert "n_valid_candidates" in posterior
    assert posterior["n_valid_candidates"] == 2


def test_tnm_inverse_interval_confidence_uses_posterior_width() -> None:
    assert interval_confidence(4.0, high_threshold=5.0, medium_threshold=15.0) == "high"
    assert interval_confidence(10.0, high_threshold=5.0, medium_threshold=15.0) == "medium"
    assert interval_confidence(20.0, high_threshold=5.0, medium_threshold=15.0) == "low"
    assert overall_confidence(["high", "medium", "high"]) == "medium"
    assert overall_confidence(["high", "low", "high"]) == "low"
