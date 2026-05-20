"""Expected-information-gain experiment design for finite-time PS recovery."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ps.inverse_design import MODEL_PATH, candidate_feature_matrix, candidate_rows, load_model
from src.ps.inverse_design import load_model_payload


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ps"
OUT_DIR = ROOT / "results" / "ps" / "eig"
NEXT_EXPERIMENTS_PATH = DATA_DIR / "ps_eig_next_experiments.csv"
RANKING_PATH = OUT_DIR / "ps_eig_candidate_ranking.csv"
SUMMARY_PATH = OUT_DIR / "ps_eig_selection_summary.json"


DEFAULT_TARGET_WEIGHTS = {
    "delta_h_total_J_g": 0.7,
    "peak_area_J_g": 0.6,
    "peak_temperature_Tp_C": 0.8,
    "peak_height_uW": 0.4,
    "recovery_index": 1.0,
    "path_dependence_index": 0.6,
}

DEFAULT_NOISE_FLOORS = {
    "delta_h_total_J_g": 0.20,
    "peak_area_J_g": 0.13,
    "peak_temperature_Tp_C": 5.0,
    "peak_height_uW": 50.0,
    "recovery_index": 0.15,
    "path_dependence_index": 0.20,
}

TARGET_WEIGHTS = DEFAULT_TARGET_WEIGHTS
NOISE_FLOORS = DEFAULT_NOISE_FLOORS


@dataclass
class EigConfig:
    batch_size: int = 30
    min_single_step: int = 4
    novelty_weight: float = 0.12
    inverse_identifiability_weight: float = 1.20
    time_penalty_weight: float = 0.03
    redundancy_weight: float = 0.55
    diversity_bandwidth: float = 1.0


def load_model_payload(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    targets: list[str],
) -> dict[str, object]:
    """Summarize whether predicted uncertainty tracks held-out errors."""
    out: dict[str, object] = {}
    abs_corrs = []
    for j, target in enumerate(targets):
        errors = np.abs(y_pred[:, j] - y_true[:, j])
        unc = uncertainty[:, j]
        corr = _safe_corr(errors, unc)
        abs_corrs.append(abs(corr))
        out[target] = {
            "uncertainty_error_corr": corr,
            "mean_abs_error": float(np.mean(errors)),
            "mean_uncertainty": float(np.mean(unc)),
            "uncertainty_to_error_ratio": float(np.mean(unc) / max(np.mean(errors), 1e-12)),
        }
    mean_abs = float(np.mean(abs_corrs)) if abs_corrs else 0.0
    out["summary"] = {
        "mean_abs_uncertainty_error_corr": mean_abs,
        "calibration_quality": "weak" if mean_abs < 0.25 else "moderate" if mean_abs < 0.5 else "strong",
    }
    return out


def payload_calibration_quality(payload: dict) -> dict[str, object]:
    calibration = payload.get("uncertainty_calibration") or payload.get("calibration") or {}
    summary = calibration.get("summary", {}) if isinstance(calibration, dict) else {}
    mean_abs = float(summary.get("mean_abs_uncertainty_error_corr", 0.0) or 0.0)
    quality = str(summary.get("calibration_quality", "weak" if mean_abs < 0.25 else "moderate" if mean_abs < 0.5 else "strong"))
    return {"mean_abs_uncertainty_error_corr": mean_abs, "calibration_quality": quality}


def targets_from_payload(payload: dict) -> list[str]:
    return list(payload.get("targets", DEFAULT_TARGET_WEIGHTS))


def noise_scales(payload: dict, targets: list[str]) -> np.ndarray:
    metrics = payload.get("metrics", {})
    if not metrics:
        metrics = payload.get("metrics_same_split", {})
    scales = []
    for target in targets:
        rmse = metrics.get(target, {}).get("rmse")
        floor = DEFAULT_NOISE_FLOORS[target]
        if rmse is None or not np.isfinite(float(rmse)):
            scales.append(floor)
        else:
            scales.append(max(float(rmse), floor))
    return np.array(scales, dtype=float)


def scaled_features(x: np.ndarray, payload: dict) -> np.ndarray:
    mean = np.array(payload["x_mean"], dtype=float)
    std = np.array(payload["x_std"], dtype=float)
    std[std == 0] = 1.0
    return (x - mean) / std


def nearest_distance(x_scaled: np.ndarray, reference_scaled: np.ndarray) -> np.ndarray:
    diff = x_scaled[:, None, :] - reference_scaled[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def eig_components(uncertainty: np.ndarray, scales: np.ndarray) -> np.ndarray:
    ratio2 = (uncertainty / scales[None, :]) ** 2
    return 0.5 * np.log1p(ratio2)


def _bounded(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _perturbed_row(row: dict[str, str], parameter: str, direction: float) -> dict[str, str]:
    out = dict(row)
    if parameter in {"T1_C", "T2_C"}:
        out[parameter] = str(_bounded(float(out[parameter]) + 5.0 * direction, 50.0, 100.0))
    elif parameter == "t1_s":
        factor = 3.0 if direction > 0 else 1.0 / 3.0
        out[parameter] = str(_bounded(float(out[parameter]) * factor, 1.0, 1800.0))
    elif parameter == "t2_s":
        factor = 3.0 if direction > 0 else 1.0 / 3.0
        out[parameter] = str(_bounded(float(out[parameter]) * factor, 1.0, 1800.0))
    out["total_anneal_time_s"] = str(float(out["t1_s"]) + float(out["t2_s"]))
    return out


def inverse_identifiability_scores(
    rows: list[dict[str, str]],
    payload: dict,
    targets: list[str],
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score candidates by local ability to distinguish T1/t1/T2/t2.

    The score is the log-determinant of a normalized local sensitivity Gram
    matrix.  High values mean small changes in all four inverse parameters
    produce distinguishable predicted target changes.
    """
    model = load_model()
    active = [idx for idx, target in enumerate(targets) if target in DEFAULT_TARGET_WEIGHTS]
    active_scales = scales[active]
    raw_scores = np.zeros(len(rows), dtype=float)
    sensitivity_norms = {name: np.zeros(len(rows), dtype=float) for name in ["T1_C", "t1_s", "T2_C", "t2_s"]}

    two_step_indices = [idx for idx, row in enumerate(rows) if row["mode"] == "two_step"]
    if not two_step_indices:
        return raw_scores, sensitivity_norms

    for idx in two_step_indices:
        row = rows[idx]
        columns = []
        for parameter in ["T1_C", "t1_s", "T2_C", "t2_s"]:
            low = _perturbed_row(row, parameter, -1.0)
            high = _perturbed_row(row, parameter, 1.0)
            pred_pair, _ = model.predict(candidate_feature_matrix([low, high], payload))
            delta = (pred_pair[1, active] - pred_pair[0, active]) / active_scales
            if parameter in {"T1_C", "T2_C"}:
                denom = max(float(high[parameter]) - float(low[parameter]), 1e-9)
            else:
                denom = max(np.log10(float(high[parameter])) - np.log10(float(low[parameter])), 1e-9)
            column = delta / denom
            columns.append(column)
            sensitivity_norms[parameter][idx] = float(np.linalg.norm(column))
        jacobian = np.column_stack(columns)
        gram = jacobian.T @ jacobian
        raw_scores[idx] = float(np.linalg.slogdet(gram + 1e-6 * np.eye(4))[1])

    finite = raw_scores[two_step_indices]
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    normalized = np.zeros_like(raw_scores)
    if hi > lo:
        normalized[two_step_indices] = np.clip((raw_scores[two_step_indices] - lo) / (hi - lo), 0.0, 1.0)
    return normalized, sensitivity_norms


def exact_condition_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("mode", ""),
        row.get("T1_C", ""),
        row.get("t1_s", ""),
        row.get("T2_C", ""),
        row.get("t2_s", ""),
    )


def read_existing_conditions(path: Path = DATA_DIR / "ps_preexperiment_features.csv") -> set[tuple[str, str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {exact_condition_key(row) for row in csv.DictReader(f)}


def candidate_table(mode_filter: str | None = None) -> list[dict[str, str]]:
    existing = read_existing_conditions()
    rows = []
    for row in candidate_rows():
        if mode_filter and row["mode"] != mode_filter:
            continue
        if exact_condition_key(row) not in existing:
            rows.append(row)
    return rows


def pick_best(
    remaining: set[int],
    selected: list[int],
    score: np.ndarray,
    x_scaled: np.ndarray,
    config: EigConfig,
) -> int | None:
    best_idx = None
    best_score = -np.inf
    for idx in remaining:
        adjusted = float(score[idx])
        if selected:
            diff = x_scaled[idx][None, :] - x_scaled[selected]
            min_dist2 = float(np.min(np.sum(diff * diff, axis=1)))
            redundancy = np.exp(-0.5 * min_dist2 / (config.diversity_bandwidth ** 2))
            adjusted -= config.redundancy_weight * redundancy
        if adjusted > best_score:
            best_idx = idx
            best_score = adjusted
    return best_idx


def greedy_select(score: np.ndarray, x_scaled: np.ndarray, rows: list[dict[str, str]], config: EigConfig) -> list[int]:
    selected: list[int] = []
    remaining = set(range(len(score)))

    single_step_remaining = {idx for idx in remaining if rows[idx]["mode"] == "single_step"}
    for _ in range(min(config.min_single_step, config.batch_size, len(single_step_remaining))):
        best_idx = pick_best(single_step_remaining, selected, score, x_scaled, config)
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        single_step_remaining.remove(best_idx)

    while len(selected) < min(config.batch_size, len(score)):
        best_idx = pick_best(remaining, selected, score, x_scaled, config)
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    try:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        with fallback.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return fallback


def design_next_batch(
    config: EigConfig | None = None,
    mode_filter: str | None = None,
    next_experiments_path: Path = NEXT_EXPERIMENTS_PATH,
    ranking_path: Path = RANKING_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> dict:
    config = config or EigConfig()
    payload = load_model_payload()
    calibration_quality = payload_calibration_quality(payload)
    effective_novelty_weight = config.novelty_weight
    if calibration_quality["calibration_quality"] == "weak":
        effective_novelty_weight *= 1.75
    targets = targets_from_payload(payload)
    model = load_model()
    rows = candidate_table(mode_filter)
    x = candidate_feature_matrix(rows, payload)
    x_scaled = scaled_features(x, payload)
    pred, unc = model.predict(x)

    scales = noise_scales(payload, targets)
    weights = np.array([DEFAULT_TARGET_WEIGHTS[target] for target in targets], dtype=float)
    components = eig_components(unc, scales)
    eig = components @ weights
    inverse_identifiability, sensitivity_norms = inverse_identifiability_scores(rows, payload, targets, scales)
    known_distance = nearest_distance(x_scaled, np.array(payload["x_train"], dtype=float))
    total_time = np.array([float(row["total_anneal_time_s"]) for row in rows], dtype=float)
    time_penalty = config.time_penalty_weight * total_time / 1800.0
    raw_score = (
        eig
        + effective_novelty_weight * known_distance
        + config.inverse_identifiability_weight * inverse_identifiability
        - time_penalty
    )

    order = np.argsort(-raw_score)
    selected = greedy_select(raw_score, x_scaled, rows, config)

    def decorate(idx: int, rank: int) -> dict:
        out = dict(rows[idx])
        out["rank"] = rank
        out["eig_score"] = float(raw_score[idx])
        out["expected_information_gain"] = float(eig[idx])
        out["inverse_identifiability_score"] = float(inverse_identifiability[idx])
        out["nearest_existing_distance"] = float(known_distance[idx])
        out["novelty_component"] = float(effective_novelty_weight * known_distance[idx])
        out["inverse_identifiability_component"] = float(config.inverse_identifiability_weight * inverse_identifiability[idx])
        out["time_penalty"] = float(time_penalty[idx])
        for parameter, values in sensitivity_norms.items():
            out[f"sensitivity_{parameter}"] = float(values[idx])
        for j, target in enumerate(targets):
            out[f"eig_{target}"] = float(components[idx, j])
            out[f"pred_{target}"] = float(pred[idx, j])
            out[f"uncertainty_{target}"] = float(unc[idx, j])
        return out

    ranking = [decorate(int(idx), rank) for rank, idx in enumerate(order, start=1)]
    selection = [decorate(idx, rank) for rank, idx in enumerate(selected, start=1)]
    ranking_path = write_csv(ranking_path, ranking)
    next_experiments_path = write_csv(next_experiments_path, selection)

    summary = {
        "method": "Greedy active learning with an Expected Information Gain proxy.",
        "eig_proxy": "0.5 * log(1 + predictive_variance / noise_scale^2), summed over weighted targets.",
        "inverse_identifiability_proxy": "Normalized log-det of local target sensitivities to T1, log(t1), T2, and log(t2).",
        "selected_count": len(selection),
        "candidate_count": len(rows),
        "mode_filter": mode_filter or "all",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "targets": targets,
        "target_weights": {target: DEFAULT_TARGET_WEIGHTS[target] for target in targets},
        "noise_scales": {target: float(scales[i]) for i, target in enumerate(targets)},
        "config": config.__dict__,
        "calibration": calibration_quality,
        "effective_novelty_weight": effective_novelty_weight,
        "score_formula": "EIG + novelty_component + inverse_identifiability_component - time_penalty - greedy redundancy penalty during batch selection.",
        "outputs": {
            "next_experiments": str(next_experiments_path.relative_to(ROOT)),
            "candidate_ranking": str(ranking_path.relative_to(ROOT)),
        },
        "notes": [
            "This is an information-theoretic active-learning design over the current surrogate model.",
            "It favors uncertain, novel, and non-redundant finite-time annealing conditions.",
            "It now also favors conditions that make T1/t1/T2/t2 locally distinguishable for inverse reconstruction.",
            "The default batch keeps single-step anchor experiments so the two-step path effect stays interpretable.",
            "The Kovacs label has zero weight because current PS pre-experiments do not show clear Kovacs peaks.",
        ],
        "top_selected": selection[:5],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    design_next_batch()
