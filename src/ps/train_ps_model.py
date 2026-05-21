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
    "T1_logt1",
    "T2_logt2",
    "delta_T_log_time_ratio",
    "inv_T1_inv_T2_product",
    "inv_T_diff",
    "log_t_ratio",
    "step_time_asymmetry",
]

TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "peak_temperature_Tp_C",
    "peak_height_uW",
    "recovery_index",
    "path_dependence_index",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def feature_values(row: dict[str, str]) -> dict[str, float]:
    is_two = 1.0 if row["mode"] == "two_step" else 0.0
    t1 = max(float(row["t1_s"]), 1e-9)
    t2_raw = row.get("t2_s", "")
    t2 = max(float(t2_raw), 1e-9) if t2_raw not in {"", "nan", "NaN"} else 1e-9
    T1 = float(row["T1_C"])
    T2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else T1
    total = max(float(row["total_anneal_time_s"]), 1e-9)
    log_t1 = float(np.log10(t1))
    log_t2 = float(np.log10(t2))
    log_total = float(np.log10(total))
    T1_K = T1 + 273.15
    T2_K = T2 + 273.15
    log_t_ratio = float(np.log10(t2 / max(t1, 1e-9)))
    return {
        "is_two_step": is_two,
        "T1_C": T1,
        "T2_C_filled": T2,
        "delta_T_C": T2 - T1,
        "log_t1_s": log_t1,
        "log_t2_s": log_t2,
        "log_total_time_s": log_total,
        "T1_logt1": T1 * log_t1,
        "T2_logt2": T2 * log_t2,
        "delta_T_log_time_ratio": (T2 - T1) * log_t_ratio,
        "inv_T1_inv_T2_product": (1.0 / T1_K) * (1.0 / T2_K),
        "inv_T_diff": (1.0 / T1_K) - (1.0 / T2_K),
        "log_t_ratio": log_t_ratio,
        "step_time_asymmetry": abs(t1 - t2) / total,
    }


def featurize(row: dict[str, str], features: list[str] | None = None) -> list[float]:
    values = feature_values(row)
    features = features or FEATURES
    return [float(values[name]) for name in features]


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


def grouped_folds(rows: list[dict[str, str]], indices: np.ndarray | None = None, n_folds: int = 5, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(rows), dtype=int) if indices is None else np.asarray(indices, dtype=int)
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx in indices:
        row = rows[int(idx)]
        groups[(row["mode"], row.get("source_file", ""), row["T1_C"])].append(int(idx))
    keys = list(groups)
    if not keys:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    fold_count = min(max(2, n_folds), len(keys))
    fold_keys = [[] for _ in range(fold_count)]
    for pos, key_idx in enumerate(order):
        fold_keys[pos % fold_count].append(keys[int(key_idx)])
    all_indices = {int(idx) for idx in indices}
    folds = []
    for keys_for_fold in fold_keys:
        val = []
        for key in keys_for_fold:
            val.extend(groups[key])
        val_set = set(val)
        train = sorted(all_indices - val_set)
        if train and val:
            folds.append((np.array(train, dtype=int), np.array(sorted(val), dtype=int)))
    return folds


def temperature_bin(value: float, prefix: str = "") -> str:
    if 50 <= value <= 60:
        label = "50-60"
    elif 65 <= value <= 75:
        label = "65-75"
    elif 80 <= value <= 90:
        label = "80-90"
    elif 95 <= value <= 100:
        label = "95-100"
    else:
        label = "other"
    return f"{prefix}{label}" if prefix else label


def time_bin(value: float) -> str:
    if value <= 60:
        return "short_<=60s"
    if value <= 300:
        return "medium_60-300s"
    if value <= 900:
        return "long_300-900s"
    return "very_long_>900s"


def row_path_class(row: dict[str, str]) -> str:
    if row.get("mode") != "two_step" or row.get("T2_C", "") in {"", "nan", "NaN"}:
        return "single_step"
    t1 = float(row["T1_C"])
    t2 = float(row["T2_C"])
    if abs(t1 - t2) <= 1e-9:
        return "isothermal"
    return "up-jump" if t2 > t1 else "down-jump"


def _summarize_category(rows: list[dict[str, str]], labels: list[str], value_fn) -> dict[str, dict[str, float | int | bool]]:
    total = max(len(rows), 1)
    out: dict[str, dict[str, float | int | bool]] = {}
    for label in labels:
        selected = [row for row in rows if value_fn(row) == label]
        n = len(selected)
        out[label] = {
            "n": n,
            "fraction": float(n / total),
            "is_low_support_region": bool(n < 2),
        }
    return out


def coverage_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "n_samples": len(rows),
        "T1_bin": _summarize_category(rows, ["50-60", "65-75", "80-90", "95-100", "other"], lambda row: temperature_bin(float(row["T1_C"]))),
        "T2_bin": _summarize_category(
            rows,
            ["T2 50-60", "T2 65-75", "T2 80-90", "T2 95-100", "T2 other"],
            lambda row: "T2 " + temperature_bin(float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else float(row["T1_C"])),
        ),
        "path_class": _summarize_category(rows, ["single_step", "isothermal", "up-jump", "down-jump"], row_path_class),
        "time_bin": _summarize_category(rows, ["short_<=60s", "medium_60-300s", "long_300-900s", "very_long_>900s"], lambda row: time_bin(float(row["total_anneal_time_s"]))),
    }


def repeated_grouped_cv_summary(rows: list[dict[str, str]], seeds: tuple[int, ...] = (2, 3, 4, 7, 13, 17, 23), n_folds: int = 5, bandwidth: float = 1.2) -> dict[str, object]:
    x, y = matrices(rows)
    scores = []
    per_target_mae: dict[str, list[float]] = {target: [] for target in TARGETS}
    for seed in seeds:
        for train_idx, val_idx in grouped_folds(rows, np.arange(len(rows), dtype=int), n_folds=n_folds, seed=seed):
            model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=bandwidth)
            pred, _ = model.predict(x[val_idx])
            fold_metrics = metrics(y[val_idx], pred)
            ratios = []
            for target in TARGETS:
                mae = fold_metrics[target]["mae"]
                per_target_mae[target].append(mae)
                std = float(np.std(y[val_idx, TARGETS.index(target)]))
                if std > 0:
                    ratios.append(mae / std)
            if ratios:
                scores.append(float(np.mean(ratios)))
    return {
        "n_scores": len(scores),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "score_mean": float(np.mean(scores)) if scores else float("nan"),
        "score_std": float(np.std(scores)) if scores else float("nan"),
        "score_median": float(np.median(scores)) if scores else float("nan"),
        "mean_metrics": {target: {"mae": float(np.mean(values)) if values else float("nan")} for target, values in per_target_mae.items()},
        "std_metrics": {target: {"mae": float(np.std(values)) if values else float("nan")} for target, values in per_target_mae.items()},
    }


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
        "repeated_grouped_cv": repeated_grouped_cv_summary(rows),
        "coverage_summary": coverage_summary(rows),
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
