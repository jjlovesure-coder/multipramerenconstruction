"""Train and evaluate the finite-time PS enthalpy-recovery predictor."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.dsc_processing import build_ps_dataset


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "ps"
MODEL_DIR = OUT_DIR / "models"
EVAL_DIR = OUT_DIR / "evaluation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "is_two_step",
    "T1_C",
    "T2_C_filled",
    "delta_T_C",
    "log_t1_s",
    "log_t2_s",
    "log_total_time_s",
]

TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "peak_temperature_Tp_C",
    "peak_height_uW",
    "onset_temperature_C",
    "recovery_index",
    "path_dependence_index",
    "kovacs_peak_label",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def featurize(row: dict[str, str]) -> list[float]:
    is_two = 1.0 if row["mode"] == "two_step" else 0.0
    t1 = max(float(row["t1_s"]), 1e-9)
    t2_raw = row.get("t2_s", "")
    t2 = max(float(t2_raw), 1e-9) if t2_raw not in {"", "nan", "NaN"} else 1e-9
    T1 = float(row["T1_C"])
    T2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else T1
    total = max(float(row["total_anneal_time_s"]), 1e-9)
    values = {
        "is_two_step": is_two,
        "T1_C": T1,
        "T2_C_filled": T2,
        "delta_T_C": T2 - T1,
        "log_t1_s": np.log10(t1),
        "log_t2_s": np.log10(t2),
        "log_total_time_s": np.log10(total),
    }
    return [float(values[name]) for name in FEATURES]


def matrices(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    clean = []
    for row in rows:
        if all(row.get(target, "") not in {"", "nan", "NaN"} for target in TARGETS):
            clean.append(row)
    x = np.array([featurize(row) for row in clean], dtype=float)
    y = np.array([[float(row[target]) for target in TARGETS] for row in clean], dtype=float)
    return x, y


def grouped_split(rows: list[dict[str, str]], test_fraction: float = 0.25, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["mode"], row["source_file"], row["T1_C"])].append(idx)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_test = max(1, round(len(keys) * test_fraction))
    test_keys = {keys[i] for i in order[:n_test]}
    train_idx, test_idx = [], []
    for key, indices in groups.items():
        if key in test_keys:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(TARGETS):
        err = y_pred[:, j] - y_true[:, j]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        r2 = float(1.0 - np.sum(err * err) / denom) if denom > 0 else float("nan")
        out[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    return out


def write_predictions(path: Path, rows: list[dict[str, str]], test_idx: np.ndarray, pred: np.ndarray, unc: np.ndarray) -> None:
    fields = ["sample_id", "source_file", "mode", "T1_C", "t1_s", "T2_C", "t2_s"]
    for target in TARGETS:
        fields += [f"actual_{target}", f"pred_{target}", f"uncertainty_{target}", f"error_{target}"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for out_i, row_i in enumerate(test_idx):
            src = rows[row_i]
            row = {field: src.get(field, "") for field in fields[:7]}
            for j, target in enumerate(TARGETS):
                actual = float(src[target])
                predicted = float(pred[out_i, j])
                row[f"actual_{target}"] = actual
                row[f"pred_{target}"] = predicted
                row[f"uncertainty_{target}"] = float(unc[out_i, j])
                row[f"error_{target}"] = predicted - actual
            writer.writerow(row)


def train() -> Path:
    dataset_path = build_ps_dataset()
    rows = read_rows(dataset_path)
    x, y = matrices(rows)
    train_idx, test_idx = grouped_split(rows)
    model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=1.2)
    pred, unc = model.predict(x[test_idx])
    eval_metrics = metrics(y[test_idx], pred)

    payload = {
        "model_type": "numpy_rbf_kernel_regressor",
        "features": FEATURES,
        "targets": TARGETS,
        "bandwidth": model.bandwidth,
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "x_train": model.x_train.tolist(),
        "y_train": model.y_train.tolist(),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "metrics": eval_metrics,
        "sample_mass_mg": 4.7,
        "notes": [
            "DSC temperature integrals are converted to specific enthalpy using PS sample mass = 4.7 mg.",
            "Kovacs label is observational only; the finite-time PS objective is enthalpy recovery, not maximum Kovacs peak.",
        ],
    }
    model_path = MODEL_DIR / "ps_limited_time_model.json"
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (EVAL_DIR / "ps_train_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_predictions(EVAL_DIR / "ps_test_predictions.csv", rows, test_idx, pred, unc)
    print(json.dumps({"model": str(model_path), "metrics": eval_metrics}, indent=2))
    return model_path


if __name__ == "__main__":
    train()
