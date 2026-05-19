"""Expected-information-gain experiment design for finite-time PS recovery."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ps.inverse_design import MODEL_PATH, candidate_rows, load_model
from src.ps.train_ps_model import TARGETS, featurize


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ps"
OUT_DIR = ROOT / "results" / "ps" / "eig"
NEXT_EXPERIMENTS_PATH = DATA_DIR / "ps_eig_next_experiments.csv"
RANKING_PATH = OUT_DIR / "ps_eig_candidate_ranking.csv"
SUMMARY_PATH = OUT_DIR / "ps_eig_selection_summary.json"


TARGET_WEIGHTS = {
    "delta_h_total_J_g": 0.7,
    "peak_area_J_g": 0.6,
    "peak_temperature_Tp_C": 0.8,
    "peak_height_uW": 0.4,
    "onset_temperature_C": 0.2,
    "recovery_index": 1.0,
    "path_dependence_index": 0.6,
    "kovacs_peak_label": 0.0,
}

NOISE_FLOORS = {
    "delta_h_total_J_g": 0.20,
    "peak_area_J_g": 0.13,
    "peak_temperature_Tp_C": 5.0,
    "peak_height_uW": 50.0,
    "onset_temperature_C": 5.0,
    "recovery_index": 0.15,
    "path_dependence_index": 0.20,
    "kovacs_peak_label": 1.0,
}


@dataclass
class EigConfig:
    batch_size: int = 18
    min_single_step: int = 5
    novelty_weight: float = 0.12
    time_penalty_weight: float = 0.03
    redundancy_weight: float = 0.35
    diversity_bandwidth: float = 1.0


def load_model_payload(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def noise_scales(payload: dict) -> np.ndarray:
    metrics = payload.get("metrics", {})
    scales = []
    for target in TARGETS:
        rmse = metrics.get(target, {}).get("rmse")
        floor = NOISE_FLOORS[target]
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
    model = load_model()
    rows = candidate_table(mode_filter)
    x = np.array([featurize(row) for row in rows], dtype=float)
    x_scaled = scaled_features(x, payload)
    pred, unc = model.predict(x)

    scales = noise_scales(payload)
    weights = np.array([TARGET_WEIGHTS[target] for target in TARGETS], dtype=float)
    components = eig_components(unc, scales)
    eig = components @ weights
    known_distance = nearest_distance(x_scaled, np.array(payload["x_train"], dtype=float))
    total_time = np.array([float(row["total_anneal_time_s"]) for row in rows], dtype=float)
    time_penalty = config.time_penalty_weight * total_time / 1800.0
    raw_score = eig + config.novelty_weight * known_distance - time_penalty

    order = np.argsort(-raw_score)
    selected = greedy_select(raw_score, x_scaled, rows, config)

    def decorate(idx: int, rank: int) -> dict:
        out = dict(rows[idx])
        out["rank"] = rank
        out["eig_score"] = float(raw_score[idx])
        out["expected_information_gain"] = float(eig[idx])
        out["nearest_existing_distance"] = float(known_distance[idx])
        out["time_penalty"] = float(time_penalty[idx])
        for j, target in enumerate(TARGETS):
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
        "selected_count": len(selection),
        "candidate_count": len(rows),
        "mode_filter": mode_filter or "all",
        "target_weights": TARGET_WEIGHTS,
        "noise_scales": {target: float(scales[i]) for i, target in enumerate(TARGETS)},
        "config": config.__dict__,
        "outputs": {
            "next_experiments": str(next_experiments_path.relative_to(ROOT)),
            "candidate_ranking": str(ranking_path.relative_to(ROOT)),
        },
        "notes": [
            "This is an information-theoretic active-learning design over the current surrogate model.",
            "It favors uncertain, novel, and non-redundant finite-time annealing conditions.",
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
