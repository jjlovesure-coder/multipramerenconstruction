"""Finite-time inverse design for PS enthalpy-recovery conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.physics_features import physics_feature_dict
from src.ps.train_ps_model import TARGETS
from src.ps.train_ps_model import FEATURES as DEFAULT_RAW_FEATURES
from src.ps.train_ps_model import featurize


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results" / "ps" / "models" / "ps_limited_time_model.json"
OUT_DIR = ROOT / "results" / "ps" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_TEMPERATURES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
EXPERIMENT_TIMES_S = [10, 30, 60, 100, 300, 600, 900, 1200, 1800]

INVERSE_LOSS_WEIGHTS = {
    "time": 0.05,
    "uncertainty": 0.02,
    "extrapolation": 0.40,
    "identifiability": 0.30,
}


def load_model(path: Path = MODEL_PATH) -> KernelRegressor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return KernelRegressor(
        x_train=np.array(data["x_train"], dtype=float),
        y_train=np.array(data["y_train"], dtype=float),
        x_mean=np.array(data["x_mean"], dtype=float),
        x_std=np.array(data["x_std"], dtype=float),
        bandwidth=data["bandwidth"],
    )


def load_model_payload(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_features(row: dict[str, str], feature_names: list[str] | None = None) -> list[float]:
    return featurize(row, feature_names or DEFAULT_RAW_FEATURES)


def candidate_feature_matrix(rows: list[dict[str, str]], payload: dict | None = None) -> np.ndarray:
    payload = payload or (load_model_payload() if MODEL_PATH.exists() else {})
    raw_feature_names = list(payload.get("raw_features", payload.get("features", DEFAULT_RAW_FEATURES)))
    physics_features = list(payload.get("physics_features", []))
    matrix = []
    for row in rows:
        physics = physics_feature_dict(row)
        matrix.append(raw_features(row, raw_feature_names) + [float(physics[name]) for name in physics_features])
    return np.array(matrix, dtype=float)


def t2_candidates_from_target(target: dict[str, float] | None, window_c: float = 8.0) -> list[int]:
    temps = EXPERIMENT_TEMPERATURES
    if not target or "peak_temperature_Tp_C" not in target:
        return temps
    estimate = float(target["peak_temperature_Tp_C"])
    if not np.isfinite(estimate):
        return temps
    estimate = min(max(estimate, min(temps)), max(temps))
    candidates = [temp for temp in temps if abs(temp - estimate) <= window_c]
    return candidates or [min(temps, key=lambda temp: abs(temp - estimate))]


def candidate_rows(target: dict[str, float] | None = None) -> list[dict[str, str]]:
    rows = []
    temps = EXPERIMENT_TEMPERATURES
    times = EXPERIMENT_TIMES_S
    t2_temps = t2_candidates_from_target(target)
    for temp in temps:
        for time_s in times:
            rows.append({
                "mode": "single_step",
                "T1_C": str(temp),
                "t1_s": str(time_s),
                "T2_C": "",
                "t2_s": "",
                "total_anneal_time_s": str(time_s),
            })
    for t1 in temps:
        for t2 in t2_temps:
            for t1_s in times:
                for t2_s in times:
                    rows.append({
                        "mode": "two_step",
                        "T1_C": str(t1),
                        "t1_s": str(t1_s),
                        "T2_C": str(t2),
                        "t2_s": str(t2_s),
                        "total_anneal_time_s": str(t1_s + t2_s),
                    })
    return rows


def filter_candidates_by_target(rows: list[dict[str, str]], target: dict[str, float] | None) -> tuple[list[dict[str, str]], np.ndarray]:
    """Apply hard target-derived candidate filters before inverse loss ranking."""
    if not target or "peak_temperature_Tp_C" not in target:
        return rows, np.ones(len(rows), dtype=bool)
    allowed_t2 = set(t2_candidates_from_target(target))
    mask = np.array([
        row["mode"] == "single_step" or int(round(float(row["T2_C"]))) in allowed_t2
        for row in rows
    ], dtype=bool)
    return [row for row, keep in zip(rows, mask) if keep], mask


def _target_loss(pred: np.ndarray, target: dict[str, float], target_names: list[str] | None = None) -> float:
    target_names = target_names or TARGETS
    scale = {
        "delta_h_total_J_g": 10.0,
        "peak_area_J_g": 5.0,
        "peak_temperature_Tp_C": 5.0,
        "peak_height_uW": 500.0,
        "recovery_index": 0.15,
        "path_dependence_index": 1.0,
    }
    loss = 0.0
    for name, value in target.items():
        idx = target_names.index(name)
        loss += abs((float(pred[idx]) - float(value)) / scale[name])
    return loss / max(len(target), 1)


def combined_inverse_loss(
    target_loss: float,
    total_time_s: float,
    mean_uncertainty: float,
    extrapolation_penalty: float,
    inverse_identifiability: float,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or INVERSE_LOSS_WEIGHTS
    return float(
        target_loss
        + weights["time"] * float(total_time_s) / 1800.0
        + weights["uncertainty"] * float(mean_uncertainty)
        + weights["extrapolation"] * float(extrapolation_penalty)
        - weights["identifiability"] * float(inverse_identifiability)
    )


def _candidate_parameter_values(row: dict[str, str]) -> dict[str, float]:
    t2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else float(row["T1_C"])
    t2_s = float(row["t2_s"]) if row.get("t2_s") not in {"", "nan", "NaN"} else float(row["t1_s"])
    return {
        "T1_C": float(row["T1_C"]),
        "T2_C": t2,
        "log10_t1_s": float(np.log10(max(float(row["t1_s"]), 1e-9))),
        "log10_t2_s": float(np.log10(max(t2_s, 1e-9))),
    }


def posterior_summary(rows: list[dict[str, str]], losses: np.ndarray, percentile: float = 5.0) -> dict[str, object]:
    losses = np.asarray(losses, dtype=float)
    if len(rows) == 0:
        return {"n_valid_candidates": 0}
    threshold = float(np.percentile(losses, percentile))
    valid_indices = np.where(losses <= threshold)[0]
    if len(valid_indices) == 0:
        valid_indices = np.array([int(np.argmin(losses))])
    values = [_candidate_parameter_values(rows[int(idx)]) for idx in valid_indices]
    out: dict[str, object] = {
        "percentile": percentile,
        "n_valid_candidates": int(len(valid_indices)),
        "loss_min": float(np.min(losses)),
        "loss_threshold": threshold,
        "loss_range": [float(np.min(losses[valid_indices])), float(np.max(losses[valid_indices]))],
        "loss_flatness_ratio": float((np.max(losses[valid_indices]) - np.min(losses[valid_indices])) / max(abs(float(np.min(losses))), 1e-12)),
    }
    for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]:
        data = np.array([item[name] for item in values], dtype=float)
        out[name] = {
            "p5": float(np.percentile(data, 5)),
            "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)),
            "width": float(np.percentile(data, 95) - np.percentile(data, 5)),
        }
    return out


def interpretation_from_posterior(posterior: dict[str, object]) -> dict[str, object]:
    widths = {
        name: float(posterior.get(name, {}).get("width", np.inf))  # type: ignore[union-attr]
        for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]
    }
    temp_width = max(widths["T1_C"], widths["T2_C"])
    time_width = max(widths["log10_t1_s"], widths["log10_t2_s"])
    if temp_width <= 10.0 and time_width <= 0.5:
        status = "well_constrained"
    elif temp_width <= 25.0 and time_width <= 1.2:
        status = "partially_constrained"
    else:
        status = "non_identifiable"
    return {
        "status": status,
        "posterior_widths": widths,
        "loss_flatness_ratio": float(posterior.get("loss_flatness_ratio", np.nan)),
        "reason": f"posterior temperature width={temp_width:.3g} C, log-time width={time_width:.3g}",
    }


def load_inverse_diagnostics(
    candidates: list[dict[str, str]],
    payload: dict,
    targets: list[str],
    allow_degraded_diagnostics: bool = False,
) -> dict[str, object]:
    try:
        from src.ps.eig_design import extrapolation_penalty, inverse_identifiability_scores, noise_scales, read_existing_rows, support_from_existing_rows

        scales = noise_scales(payload, targets)
        inverse_identifiability, sensitivity_norms = inverse_identifiability_scores(candidates, payload, targets, scales)
        support = support_from_existing_rows(read_existing_rows())
        extra_penalty = extrapolation_penalty(candidates, support)
        return {
            "diagnostics_enabled": True,
            "inverse_identifiability": inverse_identifiability,
            "sensitivity_norms": sensitivity_norms,
            "support": support,
            "extrapolation_penalty": extra_penalty,
            "error": "",
        }
    except Exception as exc:
        if not allow_degraded_diagnostics:
            raise
        return {
            "diagnostics_enabled": False,
            "inverse_identifiability": np.zeros(len(candidates), dtype=float),
            "sensitivity_norms": {name: np.zeros(len(candidates), dtype=float) for name in ["T1_C", "t1_s", "T2_C", "t2_s"]},
            "support": {"lower": None, "upper": None},
            "extrapolation_penalty": np.zeros(len(candidates), dtype=float),
            "error": str(exc),
        }


def support_label(row: dict[str, str], extrapolation_value: float) -> dict[str, object]:
    if row.get("mode") == "single_step":
        coverage_bin = f"T1 {row['T1_C']}C single"
    else:
        coverage_bin = f"T1 {row['T1_C']}C -> T2 {row['T2_C']}C"
    return {
        "coverage_bin": coverage_bin,
        "is_low_support_region": bool(float(extrapolation_value) > 0.0),
    }


def _row_from_vector(values: np.ndarray) -> dict[str, str]:
    t1, log_t1, t2, log_t2 = values
    t1_s = float(np.clip(10.0 ** log_t1, 10.0, 1800.0))
    t2_s = float(np.clip(10.0 ** log_t2, 10.0, 1800.0))
    return {
        "mode": "two_step",
        "T1_C": f"{float(np.clip(t1, 50.0, 100.0)):.6g}",
        "t1_s": f"{t1_s:.6g}",
        "T2_C": f"{float(np.clip(t2, 50.0, 100.0)):.6g}",
        "t2_s": f"{t2_s:.6g}",
        "total_anneal_time_s": f"{t1_s + t2_s:.6g}",
    }


def _vector_from_row(row: dict[str, str]) -> np.ndarray:
    t2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else float(row["T1_C"])
    t2_s = float(row["t2_s"]) if row.get("t2_s") not in {"", "nan", "NaN"} else float(row["t1_s"])
    return np.array([
        float(row["T1_C"]),
        np.log10(max(float(row["t1_s"]), 10.0)),
        t2,
        np.log10(max(t2_s, 10.0)),
    ])


def refine_candidate(
    row: dict[str, str],
    target: dict[str, float],
    payload: dict,
    model: KernelRegressor,
) -> tuple[dict[str, str], bool]:
    """Continuously refine a two-step grid candidate when scipy is available."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return row, False

    target_names = list(payload.get("targets", TARGETS))

    def objective(values: np.ndarray) -> float:
        candidate = _row_from_vector(values)
        pred, unc = model.predict(candidate_feature_matrix([candidate], payload))
        total_time = float(candidate["total_anneal_time_s"])
        t2_penalty = 0.0
        if "peak_temperature_Tp_C" in target and np.isfinite(float(target["peak_temperature_Tp_C"])):
            estimate = min(max(float(target["peak_temperature_Tp_C"]), 50.0), 100.0)
            t2_penalty = max(abs(float(candidate["T2_C"]) - estimate) - 8.0, 0.0) / 8.0
        return _target_loss(pred[0], target, target_names) + 0.05 * total_time / 1800.0 + 0.02 * float(np.mean(unc[0])) + t2_penalty

    start = _vector_from_row(row)
    bounds = [(50.0, 100.0), (1.0, np.log10(1800.0)), (50.0, 100.0), (1.0, np.log10(1800.0))]
    result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-3})
    candidate = _row_from_vector(np.asarray(result.x, dtype=float))
    if objective(start) <= objective(_vector_from_row(candidate)):
        return row, True
    return candidate, True


def pareto_front(rows: list[dict]) -> list[dict]:
    """Return non-dominated rows for target match, time, uncertainty, support, and identifiability."""
    metrics = ("target_loss", "total_anneal_time_s", "mean_uncertainty", "extrapolation_penalty", "negative_inverse_identifiability")
    def metric_values(row: dict) -> np.ndarray:
        values = []
        for metric in metrics:
            if metric == "negative_inverse_identifiability" and metric not in row:
                values.append(-float(row.get("inverse_identifiability", 0.0)))
            else:
                values.append(float(row[metric]))
        return np.array(values, dtype=float)

    front = []
    for row in rows:
        dominated = False
        row_values = metric_values(row)
        for other in rows:
            if other is row:
                continue
            other_values = metric_values(other)
            if np.all(other_values <= row_values) and np.any(other_values < row_values):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def search(target: dict[str, float] | None = None, top_k: int = 10, allow_degraded_diagnostics: bool = False) -> Path:
    target = target or {"recovery_index": 0.8}
    payload = load_model_payload()
    model = load_model()
    generated_candidates = candidate_rows(target)
    candidates, target_filter_mask = filter_candidates_by_target(generated_candidates, target)
    x = candidate_feature_matrix(candidates, payload)
    pred, unc = model.predict(x)
    targets = list(payload.get("targets", TARGETS))

    target_losses = np.zeros(len(candidates))
    for i in range(len(candidates)):
        target_losses[i] = _target_loss(pred[i], target, targets)
    total_time = np.array([float(row["total_anneal_time_s"]) for row in candidates])
    mean_uncertainty = np.mean(unc, axis=1)
    diagnostics = load_inverse_diagnostics(candidates, payload, targets, allow_degraded_diagnostics=allow_degraded_diagnostics)
    inverse_identifiability = diagnostics["inverse_identifiability"]
    sensitivity_norms = diagnostics["sensitivity_norms"]
    support = diagnostics["support"]
    extra_penalty = diagnostics["extrapolation_penalty"]
    losses = np.array([
        combined_inverse_loss(
            target_loss=float(target_losses[i]),
            total_time_s=float(total_time[i]),
            mean_uncertainty=float(mean_uncertainty[i]),
            extrapolation_penalty=float(extra_penalty[i]),
            inverse_identifiability=float(inverse_identifiability[i]),
        )
        for i in range(len(candidates))
    ], dtype=float)

    order = np.argsort(losses)[:max(top_k * 3, top_k)]
    results = []
    scipy_used = False
    for idx in order:
        row = dict(candidates[idx])
        if row["mode"] == "two_step":
            row, refined = refine_candidate(row, target, payload, model)
            scipy_used = scipy_used or refined
            row["refined_continuously"] = refined
            pred_one, unc_one = model.predict(candidate_feature_matrix([row], payload))
            row_target_loss = _target_loss(pred_one[0], target, targets)
            pred_values = pred_one[0]
            unc_values = unc_one[0]
            refined_diagnostics = load_inverse_diagnostics([row], payload, targets, allow_degraded_diagnostics=allow_degraded_diagnostics)
            row_extra_penalty = float(refined_diagnostics["extrapolation_penalty"][0])
            row_identifiability = float(refined_diagnostics["inverse_identifiability"][0])
            row_loss = combined_inverse_loss(
                target_loss=row_target_loss,
                total_time_s=float(row["total_anneal_time_s"]),
                mean_uncertainty=float(np.mean(unc_one[0])),
                extrapolation_penalty=row_extra_penalty,
                inverse_identifiability=row_identifiability,
            )
        else:
            row["refined_continuously"] = False
            row_loss = float(losses[idx])
            row_target_loss = float(target_losses[idx])
            pred_values = pred[idx]
            unc_values = unc[idx]
            row_extra_penalty = float(extra_penalty[int(idx)])
            row_identifiability = float(inverse_identifiability[int(idx)])
        row["loss"] = float(row_loss)
        row["target_loss"] = float(row_target_loss)
        row["mean_uncertainty"] = float(np.mean(unc_values))
        row["extrapolation_penalty"] = row_extra_penalty
        row["inverse_identifiability"] = row_identifiability
        row["negative_inverse_identifiability"] = -row_identifiability
        row.update(support_label(row, row_extra_penalty))
        row["abs_path_dependence"] = float(abs(pred_values[targets.index("path_dependence_index")]))
        row["loss_components"] = {
            "target_loss": float(row_target_loss),
            "time_penalty": float(INVERSE_LOSS_WEIGHTS["time"] * float(row["total_anneal_time_s"]) / 1800.0),
            "uncertainty_penalty": float(INVERSE_LOSS_WEIGHTS["uncertainty"] * np.mean(unc_values)),
            "extrapolation_penalty": row_extra_penalty,
            "identifiability_bonus": float(INVERSE_LOSS_WEIGHTS["identifiability"] * row_identifiability),
        }
        for j, target_name in enumerate(targets):
            row[f"pred_{target_name}"] = float(pred_values[j])
            row[f"uncertainty_{target_name}"] = float(unc_values[j])
        for parameter, values in sensitivity_norms.items():
            row[f"sensitivity_{parameter}"] = float(values[int(idx)]) if not row.get("refined_continuously") else float("nan")
        results.append(row)

    results = sorted(results, key=lambda item: float(item["loss"]))[:top_k]
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    front = pareto_front(results)
    for row in front:
        row["pareto_optimal"] = True
    for row in results:
        row.setdefault("pareto_optimal", False)

    posterior = posterior_summary(candidates, losses)
    interpretation = interpretation_from_posterior(posterior)
    output = OUT_DIR / "ps_inverse_design_top10.json"
    output.write_text(json.dumps({
        "target": target,
        "point_estimates": results,
        "results": results,
        "deprecated_fields": {
            "results": "Alias of point_estimates kept for backward compatibility; use posterior and interpretation for inverse conclusions.",
        },
        "pareto_front": front,
        "posterior": posterior,
        "interpretation": interpretation,
        "selection_policy": "target_loss + time + uncertainty + extrapolation - identifiability; T2 soft-constrained by Tp when available",
        "candidate_space": {
            "temperature_C": EXPERIMENT_TEMPERATURES,
            "time_s": EXPERIMENT_TIMES_S,
            "T2_candidates_C": t2_candidates_from_target(target),
            "n_generated_candidates": len(generated_candidates),
            "n_candidates": len(candidates),
            "n_filtered_by_target": int(len(generated_candidates) - len(candidates)),
            "observed_support": support,
        },
        "identifiability_diagnostics": {
            "diagnostics_enabled": bool(diagnostics["diagnostics_enabled"]),
            "diagnostics_error": diagnostics.get("error", ""),
            "mean": float(np.mean(inverse_identifiability)),
            "max": float(np.max(inverse_identifiability)),
            "min": float(np.min(inverse_identifiability)),
        },
        "loss_components": {
            "weights": INVERSE_LOSS_WEIGHTS,
            "target_loss_scale": "per-target scales in _target_loss",
        },
        "continuous_refinement_used": scipy_used,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"target": target, "top_result": results[0], "output": str(output)}, indent=2))
    return output


if __name__ == "__main__":
    search({"recovery_index": 0.8})
