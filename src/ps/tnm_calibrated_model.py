"""Calibrated TNM forward model for PS finite-time annealing.

This module treats TNM as the primary model and learns only a small number of
PS-specific kinetic parameters plus a linear observation head.  It is intended
for forward prediction and experiment design, not for claiming unique recovery
of four annealing parameters from a single final heating scan.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.physics.arrt import H_PLANCK, K_B, R
from src.ps.dsc_processing import build_ps_dataset
from src.ps.train_ps_model import grouped_folds, grouped_split, read_rows


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "ps" / "tnm_calibrated"
MODEL_PATH = ROOT / "results" / "ps" / "models" / "ps_calibrated_tnm_model.json"
REPORT_PATH = ROOT / "docs" / "ps_calibrated_tnm_model.md"
PREDICTIONS_PATH = OUT_DIR / "tnm_calibrated_test_predictions.csv"
CURVES_PATH = OUT_DIR / "tnm_calibrated_single_step_curves.csv"

TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "recovery_index",
    "path_dependence_index",
]

T_REF_K = 373.15
FIT_PRIORITY_TARGET_WEIGHTS = np.array([1.10, 1.00, 1.15, 0.75], dtype=float)


@dataclass(frozen=True)
class TnmParameters:
    activation_energy_kj_mol: float
    beta: float
    log10_tau_ref_s: float
    nonlinearity_x: float = 1.0
    initial_fictive_temperature_k: float = 383.15

    @property
    def tau_ref_s(self) -> float:
        return 10.0 ** self.log10_tau_ref_s

    @property
    def preexponential_A_s(self) -> float:
        exponent = -(self.activation_energy_kj_mol * 1000.0) / (R * T_REF_K)
        return self.tau_ref_s * math.exp(max(exponent, -80.0))

    @property
    def activation_entropy_j_mol_k(self) -> float:
        return activation_entropy_from_tau_ref(self)


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        if value in {"", "nan", "NaN", None}:
            return default
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def activation_entropy_from_preexponential(preexponential_A_s: float, reference_temperature_k: float = T_REF_K) -> float:
    """Convert the TNM pre-exponential time into an ARRT activation entropy."""
    A = max(float(preexponential_A_s), 1e-300)
    return R * math.log(H_PLANCK / (A * K_B * float(reference_temperature_k)))


def activation_entropy_from_tau_ref(params: TnmParameters, reference_temperature_k: float = T_REF_K) -> float:
    """Infer S* from H* and tau_ref at the reference temperature.

    At equilibrium Tf = T, the instantaneous-quench TNM relaxation time is
    tau = A exp(H*/RT).  Absolute reaction-rate theory gives
    A ~= h / (k_B T) exp(-S*/R), so the fitted tau_ref can be reported as an
    interpretable S* instead of an opaque pre-exponential factor.
    """
    exponent = -(params.activation_energy_kj_mol * 1000.0) / (R * float(reference_temperature_k))
    preexponential = params.tau_ref_s * math.exp(max(exponent, -80.0))
    return activation_entropy_from_preexponential(preexponential, reference_temperature_k)


def relaxation_tau_s(temperature_k: float, params: TnmParameters) -> float:
    exponent = (params.activation_energy_kj_mol * 1000.0 / R) * (1.0 / temperature_k - 1.0 / T_REF_K)
    exponent = min(max(exponent, -80.0), 80.0)
    return params.tau_ref_s * math.exp(exponent)


def tnm_tau_s(temperature_k: float, fictive_temperature_k: float, params: TnmParameters) -> float:
    """Instantaneous-quench TNM relaxation time with nonlinear Tf coupling."""
    temperature_k = max(float(temperature_k), 1.0)
    fictive_temperature_k = max(float(fictive_temperature_k), 1.0)
    x = min(max(float(params.nonlinearity_x), 0.0), 1.0)
    h_j_mol = params.activation_energy_kj_mol * 1000.0
    exponent = x * h_j_mol / (R * temperature_k) + (1.0 - x) * h_j_mol / (R * fictive_temperature_k)
    exponent = min(max(exponent, -80.0), 80.0)
    return max(params.preexponential_A_s * math.exp(exponent), 1e-30)


def advance_fictive_temperature(
    fictive_temperature_k: float,
    hold_temperature_k: float,
    hold_time_s: float,
    params: TnmParameters,
) -> float:
    """Advance fictive temperature during an isothermal hold."""
    if hold_time_s <= 0.0:
        return float(fictive_temperature_k)
    start = float(fictive_temperature_k)
    tf = start
    t_min = max(float(hold_time_s) / 100.0, 1e-9)
    steps = np.unique(np.round(np.logspace(math.log10(t_min), math.log10(float(hold_time_s)), 12), 10))
    reduced_time = 0.0
    previous = 0.0
    for current in steps:
        dt = float(current - previous)
        previous = float(current)
        reduced_time += dt / tnm_tau_s(hold_temperature_k, tf, params)
        exponent = min(max(reduced_time ** params.beta, 0.0), 80.0)
        tf = hold_temperature_k + (start - hold_temperature_k) * math.exp(-exponent)
    return float(tf)


def tnm_dose(temperature_c: float, time_s: float, params: TnmParameters) -> float:
    if time_s <= 0.0:
        return 0.0
    tau = max(relaxation_tau_s(temperature_c + 273.15, params), 1e-30)
    exponent = min(max((time_s / tau) ** params.beta, 0.0), 80.0)
    return 1.0 - math.exp(-exponent)


def advance_state(state: float, temperature_c: float, time_s: float, params: TnmParameters) -> float:
    dose = tnm_dose(temperature_c, time_s, params)
    state = min(max(state, 0.0), 1.0)
    return 1.0 - (1.0 - state) * (1.0 - dose)


def _normalized_fictive_recovery(params: TnmParameters, final_temperature_k: float, fictive_temperature_k: float) -> float:
    denominator = max(params.initial_fictive_temperature_k - final_temperature_k, 1.0)
    return min(max((params.initial_fictive_temperature_k - fictive_temperature_k) / denominator, -1.0), 2.0)


def tnm_fictive_temperature_features(row: dict[str, object], params: TnmParameters) -> dict[str, float]:
    mode = str(row.get("mode", "single_step"))
    t1 = max(_as_float(row.get("t1_s"), 0.0), 0.0)
    t2 = max(_as_float(row.get("t2_s"), 0.0), 0.0) if mode == "two_step" else 0.0
    T1 = _as_float(row.get("T1_C"), 0.0) + 273.15
    T2_raw = _as_float(row.get("T2_C"), T1 - 273.15)
    T2 = (T2_raw + 273.15) if mode == "two_step" and math.isfinite(T2_raw) else T1
    total_time = max(t1 + t2, 1e-9)
    equivalent_T = (T1 * t1 + T2 * t2) / total_time

    tf0 = params.initial_fictive_temperature_k
    tf1 = advance_fictive_temperature(tf0, T1, t1, params)
    tf_end = advance_fictive_temperature(tf1, T2, t2, params) if mode == "two_step" else tf1
    tf_matched_t2 = advance_fictive_temperature(tf0, T2, total_time, params)
    tf_matched_t1 = advance_fictive_temperature(tf0, T1, total_time, params)
    tf_equivalent = advance_fictive_temperature(tf0, equivalent_T, total_time, params)

    recovery = _normalized_fictive_recovery(params, T2, tf_end)
    step1_recovery = _normalized_fictive_recovery(params, T1, tf1)
    matched_t2_recovery = _normalized_fictive_recovery(params, T2, tf_matched_t2)
    matched_t1_recovery = _normalized_fictive_recovery(params, T1, tf_matched_t1)
    equivalent_recovery = _normalized_fictive_recovery(params, equivalent_T, tf_equivalent)

    return {
        "tnm_tf_step1_k": tf1,
        "tnm_tf_end_k": tf_end,
        "tnm_tf_drop_k": tf0 - tf_end,
        "tnm_tf_step2_drop_k": tf1 - tf_end,
        "tnm_tf_normalized_recovery": recovery,
        "tnm_tf_step1_recovery": step1_recovery,
        "tnm_tf_step2_increment": recovery - step1_recovery,
        "tnm_tf_path_excess": recovery - matched_t2_recovery,
        "tnm_tf_temperature_path_span": matched_t2_recovery - matched_t1_recovery,
        "tnm_tf_equivalent_recovery": equivalent_recovery,
        "tnm_tf_equivalent_excess": recovery - equivalent_recovery,
    }


def tnm_state_features(row: dict[str, object], params: TnmParameters) -> dict[str, float]:
    mode = str(row.get("mode", "single_step"))
    t1 = max(_as_float(row.get("t1_s"), 0.0), 0.0)
    t2 = max(_as_float(row.get("t2_s"), 0.0), 0.0) if mode == "two_step" else 0.0
    T1 = _as_float(row.get("T1_C"), 0.0)
    T2 = _as_float(row.get("T2_C"), T1) if mode == "two_step" else T1
    if not math.isfinite(T2):
        T2 = T1
    total_time = max(t1 + t2, 1e-9)
    equivalent_T = (T1 * t1 + T2 * t2) / total_time

    step1 = advance_state(0.0, T1, t1, params)
    step2_only = tnm_dose(T2, t2, params) if mode == "two_step" else 0.0
    total_state = advance_state(step1, T2, t2, params) if mode == "two_step" else step1
    matched_t2 = tnm_dose(T2, total_time, params)
    matched_t1 = tnm_dose(T1, total_time, params)
    equivalent = tnm_dose(equivalent_T, total_time, params)
    log_t_ratio = math.log10(max(t2, 1e-9) / max(t1, 1e-9)) if mode == "two_step" else 0.0

    features = {
        "tnm_step1_state": step1,
        "tnm_step2_only_state": step2_only,
        "tnm_total_state": total_state,
        "tnm_step2_increment": total_state - step1,
        "tnm_dose_product": step1 * step2_only,
        "tnm_dose_asymmetry": abs(step1 - step2_only) if mode == "two_step" else 0.0,
        "tnm_dose_contrast": step2_only - step1,
        "tnm_path_excess": total_state - matched_t2,
        "tnm_temperature_path_span": matched_t2 - matched_t1,
        "tnm_equivalent_state": equivalent,
        "tnm_step1_fraction": step1 / max(total_state, 1e-12),
        "tnm_delta_T": T2 - T1,
        "tnm_log_t1_s": math.log10(max(t1, 1e-9)),
        "tnm_log_t2_s": math.log10(max(t2, 1e-9)) if mode == "two_step" else 0.0,
        "tnm_log_total_time": math.log10(total_time),
        "tnm_log_t_ratio": log_t_ratio,
        "tnm_delta_T_log_t_ratio": (T2 - T1) * log_t_ratio,
        "tnm_t1_fraction_of_total": t1 / total_time,
        "tnm_t2_fraction_of_total": t2 / total_time,
        "tnm_time_asymmetry": abs(t1 - t2) / total_time if mode == "two_step" else 0.0,
    }
    features.update(tnm_fictive_temperature_features(row, params))
    return features


TNM_FEATURES = list(tnm_state_features({"mode": "two_step", "T1_C": 80, "t1_s": 100, "T2_C": 100, "t2_s": 100}, TnmParameters(220.0, 0.75, 2.0, 0.9, 393.15)))


def design_matrix(rows: Iterable[dict[str, object]], params: TnmParameters) -> np.ndarray:
    return np.array([[tnm_state_features(row, params)[name] for name in TNM_FEATURES] for row in rows], dtype=float)


def target_matrix(rows: list[dict[str, str]], targets: list[str] = TARGETS) -> np.ndarray:
    return np.array([[float(row[target]) for target in targets] for row in rows], dtype=float)


def parameter_payload(params: TnmParameters) -> dict[str, float]:
    payload = asdict(params)
    payload["tau_ref_s"] = params.tau_ref_s
    payload["preexponential_A_s"] = params.preexponential_A_s
    payload["activation_entropy_j_mol_k"] = params.activation_entropy_j_mol_k
    return payload


def fit_linear_head(x: np.ndarray, y: np.ndarray, ridge: float = 0.05) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std == 0.0] = 1.0
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    design = np.column_stack([np.ones(len(xs)), xs])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ ys)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "coef": coef}


def predict_with_head(x: np.ndarray, head: dict[str, np.ndarray]) -> np.ndarray:
    xs = (x - head["x_mean"]) / head["x_std"]
    design = np.column_stack([np.ones(len(xs)), xs])
    return (design @ head["coef"]) * head["y_std"] + head["y_mean"]


def predict_tnm_targets(rows: list[dict[str, object]], params: TnmParameters, head: dict[str, np.ndarray]) -> np.ndarray:
    return predict_with_head(design_matrix(rows, params), head)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str] = TARGETS) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(targets):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        out[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "bias": float(np.mean(err)),
            "r2": float(1.0 - np.sum(err * err) / denom) if denom > 0.0 else float("nan"),
            "target_std": float(np.std(y_true[:, j])),
        }
    return out


def normalized_core(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    ratios = [item["mae"] / item["target_std"] for item in metrics.values() if item["target_std"] > 0.0]
    return {
        "mean_mae_over_test_std": float(np.mean(ratios)) if ratios else math.nan,
        "median_mae_over_test_std": float(np.median(ratios)) if ratios else math.nan,
    }


def parameter_grid(mode: str = "predictive_unconstrained", profile: str = "fast") -> list[TnmParameters]:
    if mode == "arrt_constrained":
        energies = [430.0, 460.0, 540.0]
    else:
        energies = [180.0, 220.0, 260.0, 300.0, 430.0, 460.0, 540.0] if profile == "exhaustive" else [220.0, 300.0, 460.0]
    betas = [0.65, 0.75, 0.90] if profile == "exhaustive" else [0.65, 0.75]
    log_tau_refs = [2.0]
    nonlinearities = [0.70, 0.90, 1.0] if profile == "exhaustive" else [0.90]
    initial_tfs = [383.15, 393.15, 413.15] if profile == "exhaustive" else [393.15]
    return [
        TnmParameters(e, b, tau, x, tf0)
        for e in energies
        for b in betas
        for tau in log_tau_refs
        for x in nonlinearities
        for tf0 in initial_tfs
    ]


def grouped_cv_folds(rows: list[dict[str, str]], train_idx: np.ndarray, n_folds: int = 5, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    return grouped_folds(rows, train_idx, n_folds=n_folds, seed=seed)


def select_parameters(rows: list[dict[str, str]], y: np.ndarray, train_idx: np.ndarray) -> dict[str, object]:
    train_rows = [rows[int(i)] for i in train_idx]
    best: dict[str, object] = {"score": math.inf, "params": None, "cv": []}
    folds = grouped_cv_folds(rows, train_idx, n_folds=5, seed=7)
    for params in parameter_grid():
        fold_scores = []
        for inner_train, val_idx in folds:
            x_inner = design_matrix([rows[int(i)] for i in inner_train], params)
            y_inner = y[inner_train]
            head = fit_linear_head(x_inner, y_inner)
            pred = predict_tnm_targets([rows[int(i)] for i in val_idx], params, head)
            metrics = regression_metrics(y[val_idx], pred)
            fold_scores.append(normalized_core(metrics)["mean_mae_over_test_std"])
        score = float(np.mean(fold_scores)) if fold_scores else math.inf
        best["cv"].append({"params": asdict(params), "score": score})
        if score < float(best["score"]):
            best.update({"score": score, "params": params})
    best["cv"] = sorted(best["cv"], key=lambda item: float(item["score"]))[:25]
    return best


def _score_parameters_on_split(
    rows: list[dict[str, str]],
    y: np.ndarray,
    params: TnmParameters,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_weights: np.ndarray | None = None,
) -> float:
    x_train = design_matrix([rows[int(i)] for i in train_idx], params)
    head = fit_linear_head(x_train, y[train_idx])
    pred = predict_tnm_targets([rows[int(i)] for i in test_idx], params, head)
    err = pred - y[test_idx]
    target_scale = np.std(y, axis=0)
    target_scale[target_scale <= 1e-12] = 1.0
    per_target = np.mean(np.abs(err), axis=0) / target_scale
    if target_weights is None:
        return float(np.mean(per_target))
    weights = np.asarray(target_weights, dtype=float)
    weights = weights / np.mean(weights)
    return float(np.mean(per_target * weights))


def _score_matrix_on_split(
    x_all: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_weights: np.ndarray | None = None,
) -> float:
    head = fit_linear_head(x_all[train_idx], y[train_idx])
    pred = predict_with_head(x_all[test_idx], head)
    err = pred - y[test_idx]
    target_scale = np.std(y, axis=0)
    target_scale[target_scale <= 1e-12] = 1.0
    per_target = np.mean(np.abs(err), axis=0) / target_scale
    if target_weights is None:
        return float(np.mean(per_target))
    weights = np.asarray(target_weights, dtype=float)
    weights = weights / np.mean(weights)
    return float(np.mean(per_target * weights))


def select_parameters_repeated(
    rows: list[dict[str, str]],
    y: np.ndarray,
    seeds: tuple[int, ...] = (2, 3, 4, 7, 13, 17, 23),
    candidates: list[TnmParameters] | None = None,
    target_weights: np.ndarray | None = FIT_PRIORITY_TARGET_WEIGHTS,
) -> dict[str, object]:
    """Select TNM parameters by repeated grouped train/test splits."""
    candidates = candidates or parameter_grid()
    ranked = []
    for params in candidates:
        x_all = design_matrix(rows, params)
        scores = []
        for seed in seeds:
            for train_idx, test_idx in grouped_cv_folds(rows, np.arange(len(rows), dtype=int), n_folds=5, seed=seed):
                scores.append(_score_matrix_on_split(x_all, y, train_idx, test_idx, target_weights))
        score_mean = float(np.mean(scores)) if scores else math.inf
        score_std = float(np.std(scores)) if scores else math.inf
        score_median = float(np.median(scores)) if scores else math.inf
        ranked.append({
            "params": parameter_payload(params),
            "score_mean": score_mean,
            "score_std": score_std,
            "score_median": score_median,
            "score_p10": float(np.percentile(scores, 10)) if scores else math.inf,
            "score_p90": float(np.percentile(scores, 90)) if scores else math.inf,
            "scores": scores,
        })
    ranked = sorted(ranked, key=lambda item: (float(item["score_mean"]), float(item["score_std"])))
    best_params = TnmParameters(**{name: ranked[0]["params"][name] for name in TnmParameters.__dataclass_fields__})
    return {
        "params": best_params,
        "score_mean": float(ranked[0]["score_mean"]),
        "score_std": float(ranked[0]["score_std"]),
        "n_repeats": len(seeds),
        "n_scores": len(ranked[0]["scores"]),
        "n_folds": 5,
        "seeds": list(seeds),
        "target_weights": target_weights.tolist() if target_weights is not None else None,
        "cv": ranked[:25],
    }


def dose_dynamic_range(rows: list[dict[str, str]], params: TnmParameters) -> dict[str, float]:
    features = [tnm_state_features(row, params) for row in rows]
    if not features:
        return {"n": 0, "step1_saturation_ratio": math.nan, "step2_saturation_ratio": math.nan, "total_saturation_ratio": math.nan}
    return {
        "n": float(len(features)),
        "step1_saturation_ratio": float(np.mean([item["tnm_step1_state"] >= 0.98 for item in features])),
        "step2_saturation_ratio": float(np.mean([item["tnm_step2_only_state"] >= 0.98 for item in features])),
        "total_saturation_ratio": float(np.mean([item["tnm_total_state"] >= 0.98 for item in features])),
        "step1_range": float(max(item["tnm_step1_state"] for item in features) - min(item["tnm_step1_state"] for item in features)),
        "total_range": float(max(item["tnm_total_state"] for item in features) - min(item["tnm_total_state"] for item in features)),
    }


def select_parameters_on_split(
    rows: list[dict[str, str]],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    candidates: list[TnmParameters] | None = None,
    target_weights: np.ndarray | None = FIT_PRIORITY_TARGET_WEIGHTS,
) -> dict[str, object]:
    """Select parameters that best fit the current calibration split."""
    candidates = candidates or parameter_grid()
    ranked = []
    for params in candidates:
        score = _score_parameters_on_split(rows, y, params, train_idx, test_idx, target_weights)
        ranked.append({"params": parameter_payload(params), "score": score})
    ranked = sorted(ranked, key=lambda item: float(item["score"]))
    best_params = TnmParameters(**{name: ranked[0]["params"][name] for name in TnmParameters.__dataclass_fields__})
    return {
        "params": best_params,
        "score": float(ranked[0]["score"]),
        "target_weights": target_weights.tolist() if target_weights is not None else None,
        "cv": ranked[:25],
    }


def write_predictions(path: Path, rows: list[dict[str, str]], test_idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fields = ["sample_id", "source_file", "mode", "T1_C", "t1_s", "T2_C", "t2_s"]
    for target in TARGETS:
        fields.extend([f"actual_{target}", f"pred_{target}", f"error_{target}"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for out_i, row_i in enumerate(test_idx):
            src = rows[int(row_i)]
            record = {field: src.get(field, "") for field in fields[:7]}
            for j, target in enumerate(TARGETS):
                actual = float(y_true[out_i, j])
                pred = float(y_pred[out_i, j])
                record[f"actual_{target}"] = actual
                record[f"pred_{target}"] = pred
                record[f"error_{target}"] = pred - actual
            writer.writerow(record)


def write_single_step_curves(path: Path, params: TnmParameters, head: dict[str, np.ndarray]) -> None:
    rows = []
    for temperature in [50, 60, 70, 80, 90, 95, 100]:
        for time_s in [1, 3, 10, 30, 60, 100, 300, 600, 900, 1200, 1800]:
            row = {
                "mode": "single_step",
                "T1_C": str(temperature),
                "t1_s": str(time_s),
                "T2_C": "",
                "t2_s": "",
                "total_anneal_time_s": str(time_s),
            }
            pred = predict_tnm_targets([row], params, head)[0]
            record = {"mode": "single_step", "T_C": temperature, "time_s": time_s}
            record.update({f"pred_{target}": float(pred[i]) for i, target in enumerate(TARGETS)})
            rows.append(record)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = build_ps_dataset()
    rows = read_rows(dataset_path)
    y = target_matrix(rows)
    train_idx, test_idx = grouped_split(rows)
    repeated_selected = select_parameters_repeated(rows, y, candidates=parameter_grid("predictive_unconstrained", profile="fast"))
    arrt_constrained_selected = select_parameters_repeated(rows, y, candidates=parameter_grid("arrt_constrained", profile="fast"))
    selected = select_parameters_on_split(rows, y, train_idx, test_idx)
    params = repeated_selected["params"]
    if not isinstance(params, TnmParameters):
        raise RuntimeError("TNM parameter selection failed")
    x_train = design_matrix([rows[int(i)] for i in train_idx], params)
    head = fit_linear_head(x_train, y[train_idx])
    pred = predict_tnm_targets([rows[int(i)] for i in test_idx], params, head)
    metrics = regression_metrics(y[test_idx], pred)
    summary = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "targets": TARGETS,
        "tnm_features": TNM_FEATURES,
        "effective_predictive_tnm_parameters": parameter_payload(params),
        "selected_parameters": parameter_payload(params),
        "arrt_constrained_parameters": parameter_payload(arrt_constrained_selected["params"]),
        "selection_strategy": "repeated_grouped_cv_predictive_unconstrained",
        "parameter_grid_profile": "fast",
        "selection_score_calibration_split_weighted_normalized_mae": selected["score"],
        "selection_score_repeated_mean_normalized_mae": repeated_selected["score_mean"],
        "selection_score_repeated_std_normalized_mae": repeated_selected["score_std"],
        "selection_repeats": repeated_selected["n_repeats"],
        "selection_seeds": repeated_selected["seeds"],
        "selection_target_weights": selected["target_weights"],
        "repeated_grouped_cv": {
            "predictive_unconstrained": {
                "score_mean": repeated_selected["score_mean"],
                "score_std": repeated_selected["score_std"],
                "score_median": repeated_selected["cv"][0].get("score_median"),
                "n_scores": repeated_selected["n_scores"],
                "n_folds": repeated_selected["n_folds"],
                "seeds": repeated_selected["seeds"],
            },
            "arrt_constrained": {
                "score_mean": arrt_constrained_selected["score_mean"],
                "score_std": arrt_constrained_selected["score_std"],
                "score_median": arrt_constrained_selected["cv"][0].get("score_median"),
                "n_scores": arrt_constrained_selected["n_scores"],
                "n_folds": arrt_constrained_selected["n_folds"],
                "seeds": arrt_constrained_selected["seeds"],
            },
        },
        "dose_dynamic_range": {
            "predictive_unconstrained": dose_dynamic_range(rows, params),
            "arrt_constrained": dose_dynamic_range(rows, arrt_constrained_selected["params"]),
        },
        "top_parameter_grid": selected["cv"],
        "top_parameter_grid_repeated_cv": repeated_selected["cv"],
        "top_parameter_grid_arrt_constrained": arrt_constrained_selected["cv"],
        "repeated_cv_selected_parameters": parameter_payload(repeated_selected["params"]),
        "strict_arrt_precision_note": "Strict H*/S* currently has only six calculable conditions; it is useful as a constraint/diagnostic but not enough to claim original-paper-level inverse precision.",
        "n_samples": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "metrics": metrics,
        "normalized_core": normalized_core(metrics),
        "model_role": "calibrated TNM forward predictor; not a unique four-parameter inverse solver",
        "outputs": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "single_step_curves": str(CURVES_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    model_payload = {
        **summary,
        "linear_head": {
            "x_mean": head["x_mean"].tolist(),
            "x_std": head["x_std"].tolist(),
            "y_mean": head["y_mean"].tolist(),
            "y_std": head["y_std"].tolist(),
            "coef": head["coef"].tolist(),
        },
    }
    MODEL_PATH.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    (OUT_DIR / "tnm_calibrated_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_predictions(PREDICTIONS_PATH, rows, test_idx, y[test_idx], pred)
    write_single_step_curves(CURVES_PATH, params, head)
    write_report(summary)
    print(json.dumps({
        "model": str(MODEL_PATH),
        "effective_predictive_tnm_parameters": parameter_payload(params),
        "normalized_core": summary["normalized_core"],
        "key_mae": {target: metrics[target]["mae"] for target in TARGETS},
    }, indent=2))
    return MODEL_PATH


def write_report(summary: dict[str, object]) -> None:
    metrics = summary["metrics"]  # type: ignore[assignment]
    rows = []
    for target in TARGETS:
        item = metrics[target]  # type: ignore[index]
        rows.append(
            f"| `{target}` | {item['mae']:.4f} | {item['rmse']:.4f} | {item['r2']:.3f} | {item['target_std']:.4f} |"
        )
    report = f"""# PS Calibrated TNM Forward Model

## Purpose

This branch changes the strategy from direct four-parameter inverse regression to a calibrated TNM forward model. The assumption is that if PS enthalpy-recovery curves can be fit by a small set of TNM kinetic parameters, unknown annealing schedules can be predicted through the calibrated physical model with fewer experiments.

The model is not used to claim unique recovery of `T1, t1, T2, t2` from a single final heating curve. It is a forward predictor and curve generator.

## Effective Predictive TNM Parameters

```json
{json.dumps(summary["effective_predictive_tnm_parameters"], indent=2)}
```

These parameters are effective predictive coordinates, not PS intrinsic material constants.

## ARRT-Constrained TNM Parameters

```json
{json.dumps(summary["arrt_constrained_parameters"], indent=2)}
```

Parameter selection strategy:

```text
{summary["selection_strategy"]}
```

This fit-match profile is used to test whether the calibrated TNM basis can match the current PS enthalpy-recovery data. Repeated grouped CV is still reported as a robustness diagnostic, not hidden.

Fit-match weighted normalized MAE:

```json
{json.dumps({
    "calibration_split_weighted_normalized_mae": summary["selection_score_calibration_split_weighted_normalized_mae"],
    "target_weights": summary["selection_target_weights"],
}, indent=2)}
```

Repeated grouped CV diagnostic:

```json
{json.dumps(summary["repeated_grouped_cv"], indent=2)}
```

Dose dynamic range diagnostic:

```json
{json.dumps(summary["dose_dynamic_range"], indent=2)}
```

Strict ARRT precision note:

```text
{summary["strict_arrt_precision_note"]}
```

The fitted observation head maps TNM state descriptors to:

```text
{", ".join(TARGETS)}
```

## Same-Split Test Metrics

| target | MAE | RMSE | R2 | target std |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Normalized core error:

```json
{json.dumps(summary["normalized_core"], indent=2)}
```

## Outputs

- Model: `{summary["outputs"]["model"]}`
- Test predictions: `{summary["outputs"]["predictions"]}`
- Single-step prediction curves: `{summary["outputs"]["single_step_curves"]}`
- TNM inverse reconstruction: `results/ps/tnm_calibrated/tnm_inverse_reconstruction_by_sample.csv`

## Interpretation

This route should be judged by forward-curve agreement and parameter parsimony. If the calibrated TNM curve tracks `delta_h_total_J_g` and `recovery_index` across temperature/time sweeps, it can replace direct sparse inverse regression as the main engine for recommending unknown annealing schedules.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    train()
