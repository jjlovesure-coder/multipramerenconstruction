"""Train/test split evaluation for the sparse annealing predictor."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.dataset import build_dataset
from src.models.kernel_regression import KernelRegressor
from src.models.train_forward import BASELINE_FEATURES, FEATURES, TARGETS, load_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def grouped_split(rows: list[dict[str, str]], test_fraction: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split by panel+t2 group so overlapping rows do not leak across sets."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["panel"], row["t2_s"])].append(idx)

    keys = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_test = max(1, round(len(keys) * test_fraction))
    test_keys = {tuple(keys[i]) for i in order[:n_test]}

    train_idx = []
    test_idx = []
    for key, indices in groups.items():
        if key in test_keys:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(TARGETS):
        err = y_pred[:, j] - y_true[:, j]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        r2 = float(1.0 - np.sum(err * err) / denom) if denom > 0 else float("nan")
        out[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    return out


def evaluate_feature_set(
    dataset_path: Path,
    features: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    bandwidth: float = 1.25,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    x, y, _ = load_matrix(dataset_path, features)
    model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=bandwidth)
    pred, uncertainty = model.predict(x[test_idx])
    return regression_metrics(y[test_idx], pred), pred, uncertainty, y[test_idx]


def write_predictions(path: Path, rows: list[dict[str, str]], test_idx: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray) -> None:
    fields = [
        "split",
        "panel",
        "t1_temperature_k",
        "t1_s",
        "t2_temperature_k",
        "t2_s",
    ]
    for target in TARGETS:
        fields.extend([f"actual_{target}", f"pred_{target}", f"uncertainty_{target}", f"error_{target}"])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for out_i, row_idx in enumerate(test_idx):
            src = rows[row_idx]
            row = {
                "split": "test",
                "panel": src["panel"],
                "t1_temperature_k": src["t1_temperature_k"],
                "t1_s": src["t1_s"],
                "t2_temperature_k": src["t2_temperature_k"],
                "t2_s": src["t2_s"],
            }
            for j, target in enumerate(TARGETS):
                actual = float(src[target])
                predicted = float(pred[out_i, j])
                row[f"actual_{target}"] = actual
                row[f"pred_{target}"] = predicted
                row[f"uncertainty_{target}"] = float(uncertainty[out_i, j])
                row[f"error_{target}"] = predicted - actual
            writer.writerow(row)


def main() -> Path:
    dataset_path = build_dataset()
    _, _, rows = load_matrix(dataset_path, FEATURES)
    train_idx, test_idx = grouped_split(rows, test_fraction=0.25, seed=42)

    baseline_metrics, _, _, _ = evaluate_feature_set(dataset_path, BASELINE_FEATURES, train_idx, test_idx)
    enhanced_metrics, pred, uncertainty, _ = evaluate_feature_set(dataset_path, FEATURES, train_idx, test_idx)

    split_info = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "split_strategy": "grouped_by_panel_and_t2_s",
        "seed": 42,
        "test_fraction": 0.25,
        "n_rows": len(rows),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_train_groups": int(len({(rows[i]["panel"], rows[i]["t2_s"]) for i in train_idx})),
        "n_test_groups": int(len({(rows[i]["panel"], rows[i]["t2_s"]) for i in test_idx})),
        "targets": TARGETS,
        "baseline_without_fig3_h_tp": baseline_metrics,
        "enhanced_with_fig3_h_tp": enhanced_metrics,
    }

    summary_path = OUT_DIR / "train_test_split_metrics.json"
    summary_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    write_predictions(OUT_DIR / "test_predictions.csv", rows, test_idx, pred, uncertainty)

    print(json.dumps(split_info, indent=2))
    return summary_path


if __name__ == "__main__":
    main()
