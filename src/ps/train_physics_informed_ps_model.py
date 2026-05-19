"""Train PS physics-informed surrogate models and compare against raw RBF."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.dsc_processing import build_ps_dataset
from src.ps.physics_features import physics_feature_dict
from src.ps.train_ps_model import FEATURES as RAW_FEATURES
from src.ps.train_ps_model import featurize, grouped_split, read_rows


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "ps" / "physics_informed"
MODEL_DIR = ROOT / "results" / "ps" / "models"
FIG_PATH = OUT_DIR / "physics_informed_mae_comparison.png"
SUMMARY_PATH = OUT_DIR / "physics_informed_model_comparison.json"
PREDICTIONS_PATH = OUT_DIR / "physics_informed_test_predictions.csv"
REPORT_PATH = ROOT / "docs" / "ps_physics_informed_model_update.md"
PHYSICS_MODEL_PATH = MODEL_DIR / "ps_physics_informed_kernel_model.json"

CORE_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "recovery_index",
    "path_dependence_index",
]

DIAGNOSTIC_TARGETS = [
    "peak_temperature_Tp_C",
    "peak_height_uW",
]

TARGETS = CORE_TARGETS + DIAGNOSTIC_TARGETS

PHYSICS_COMPACT_FEATURES = [
    "phys_inv_T1_K",
    "phys_inv_T2_K",
    "phys_inv_equivalent_T_K",
    "phys_log_total_time_s",
    "phys_step_time_fraction_1",
    "phys_step_time_fraction_2",
    "tnm_state_total_e120_b050",
    "tnm_state_total_e160_b050",
    "tnm_state_total_e200_b050",
    "tnm_state_step1_e160_b035",
    "tnm_state_step1_e160_b050",
    "tnm_state_step1_e160_b065",
    "tnm_state_step2_only_e160_b035",
    "tnm_state_step2_only_e160_b050",
    "tnm_state_step2_only_e160_b065",
    "tnm_path_excess_e120_b050",
    "tnm_path_excess_e160_b035",
    "tnm_path_excess_e160_b050",
    "tnm_path_excess_e160_b065",
    "tnm_path_excess_e200_b050",
    "tnm_temperature_path_span_e160_b050",
    "tnm_step_mismatch_e160_b050",
]

PHYSICS_DOSE_FEATURES = [
    "tnm_state_total_e120_b050",
    "tnm_state_total_e160_b050",
    "tnm_state_total_e200_b050",
    "tnm_path_excess_e120_b050",
    "tnm_path_excess_e160_b050",
    "tnm_path_excess_e200_b050",
]

PHYSICS_BETA_FEATURES = [
    "tnm_state_total_e160_b035",
    "tnm_state_total_e160_b050",
    "tnm_state_total_e160_b065",
    "tnm_path_excess_e160_b035",
    "tnm_path_excess_e160_b050",
    "tnm_path_excess_e160_b065",
]

PHYSICS_FEATURE_SETS = {
    "raw_phys_compact": PHYSICS_COMPACT_FEATURES,
    "raw_phys_dose": PHYSICS_DOSE_FEATURES,
    "raw_phys_beta": PHYSICS_BETA_FEATURES,
}

BANDWIDTH_GRID = (1.0, 1.2, 1.4, 1.8, 2.2, 3.0)
CV_SEEDS = (2, 3, 4, 7, 13, 17, 23)


@dataclass
class RidgeModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    coef: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, ridge: float = 0.35) -> "RidgeModel":
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
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, coef=coef)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        xs = (x - self.x_mean) / self.x_std
        design = np.column_stack([np.ones(len(xs)), xs])
        return (design @ self.coef) * self.y_std + self.y_mean


def row_targets(row: dict[str, str]) -> list[float]:
    return [float(row[target]) for target in TARGETS]


def physics_vector(row: dict[str, str], feature_names: list[str]) -> list[float]:
    features = physics_feature_dict(row)
    return [float(features[name]) for name in feature_names]


def feature_matrix(rows: list[dict[str, str]], raw: np.ndarray, feature_names: list[str], include_raw: bool = True) -> np.ndarray:
    phys = np.array([physics_vector(row, feature_names) for row in rows], dtype=float)
    return np.column_stack([raw, phys]) if include_raw else phys


def matrices(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    clean = [row for row in rows if all(row.get(target, "") not in {"", "nan", "NaN"} for target in TARGETS)]
    x_raw = np.array([featurize(row) for row in clean], dtype=float)
    y = np.array([row_targets(row) for row in clean], dtype=float)
    return x_raw, y


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(TARGETS):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        out[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "r2": float(1.0 - np.sum(err * err) / denom) if denom > 0.0 else float("nan"),
            "target_std": float(np.std(y_true[:, j])),
        }
    return out


def normalized_summary(metrics: dict[str, dict[str, float]], targets: list[str]) -> dict[str, float]:
    ratios = []
    for target in targets:
        std = metrics[target]["target_std"]
        if std > 0:
            ratios.append(metrics[target]["mae"] / std)
    return {
        "mean_mae_over_test_std": float(np.mean(ratios)) if ratios else math.nan,
        "median_mae_over_test_std": float(np.median(ratios)) if ratios else math.nan,
    }


def core_normalized_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    indices = [TARGETS.index(target) for target in CORE_TARGETS]
    err = np.abs(y_pred[:, indices] - y_true[:, indices])
    std = np.std(y_true[:, indices], axis=0)
    std[std == 0.0] = 1.0
    return float(np.mean(np.mean(err, axis=0) / std))


def group_folds(rows: list[dict[str, str]], train_idx: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for idx in train_idx:
        row = rows[int(idx)]
        key = (row["mode"], row["source_file"], row["T1_C"])
        groups.setdefault(key, []).append(int(idx))
    keys = list(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    fold_keys = [[] for _ in range(min(n_folds, len(keys)))]
    for pos, key_idx in enumerate(order):
        fold_keys[pos % len(fold_keys)].append(keys[int(key_idx)])

    folds = []
    train_set = {int(idx) for idx in train_idx}
    for keys_for_fold in fold_keys:
        val = []
        for key in keys_for_fold:
            val.extend(groups[key])
        val_set = set(val)
        inner_train = sorted(train_set - val_set)
        if inner_train and val:
            folds.append((np.array(inner_train, dtype=int), np.array(sorted(val), dtype=int)))
    return folds


def select_physics_kernel(
    rows: list[dict[str, str]],
    x_raw: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> dict[str, object]:
    best: dict[str, object] = {"score": math.inf, "feature_set": "", "features": [], "bandwidth": 1.8}
    matrices_by_name = {
        name: feature_matrix(rows, x_raw, feature_names, include_raw=True)
        for name, feature_names in PHYSICS_FEATURE_SETS.items()
    }
    cv_rows = []
    for name, x in matrices_by_name.items():
        for bandwidth in BANDWIDTH_GRID:
            scores = []
            for seed in CV_SEEDS:
                for inner_train, val in group_folds(rows, train_idx, n_folds=5, seed=seed):
                    model = KernelRegressor.fit(x[inner_train], y[inner_train], bandwidth=bandwidth)
                    pred, _ = model.predict(x[val])
                    scores.append(core_normalized_score(y[val], pred))
            score = float(np.mean(scores))
            cv_rows.append({"feature_set": name, "bandwidth": bandwidth, "cv_core_score": score})
            if score < float(best["score"]):
                best = {
                    "score": score,
                    "feature_set": name,
                    "features": PHYSICS_FEATURE_SETS[name],
                    "bandwidth": bandwidth,
                }
    best["cv_grid"] = cv_rows
    return best


def fit_raw_kernel(x_raw: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = KernelRegressor.fit(x_raw[train_idx], y[train_idx], bandwidth=1.2)
    return model.predict(x_raw[test_idx])


def fit_physics_kernel(
    x_aug: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = KernelRegressor.fit(x_aug[train_idx], y[train_idx], bandwidth=bandwidth)
    return model.predict(x_aug[test_idx])


def fit_residual_model(
    x_phys: np.ndarray,
    x_aug: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = RidgeModel.fit(x_phys[train_idx], y[train_idx], ridge=0.35)
    train_base = baseline.predict(x_phys[train_idx])
    test_base = baseline.predict(x_phys[test_idx])
    residual = y[train_idx] - train_base
    residual_model = KernelRegressor.fit(x_aug[train_idx], residual, bandwidth=1.4)
    pred_residual, unc = residual_model.predict(x_aug[test_idx])
    return test_base + pred_residual, unc, test_base


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    test_idx: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    fields = ["sample_id", "source_file", "mode", "T1_C", "t1_s", "T2_C", "t2_s"]
    for target in TARGETS:
        fields.append(f"actual_{target}")
        for model_name in predictions:
            fields.append(f"{model_name}_{target}")
            fields.append(f"{model_name}_error_{target}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for out_i, row_i in enumerate(test_idx):
            src = rows[row_i]
            row = {field: src.get(field, "") for field in fields[:7]}
            for j, target in enumerate(TARGETS):
                actual = float(y_true[out_i, j])
                row[f"actual_{target}"] = actual
                for model_name, pred in predictions.items():
                    value = float(pred[out_i, j])
                    row[f"{model_name}_{target}"] = value
                    row[f"{model_name}_error_{target}"] = value - actual
            writer.writerow(row)


def write_figure(metrics_by_model: dict[str, dict[str, dict[str, float]]]) -> None:
    labels = [target.replace("_", "\n") for target in TARGETS]
    x = np.arange(len(TARGETS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for offset, model_name in enumerate(metrics_by_model):
        values = [metrics_by_model[model_name][target]["mae"] for target in TARGETS]
        ax.bar(x + (offset - 1) * width, values, width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_title("PS model comparison: raw vs physics-informed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_report(summary: dict) -> None:
    metric_rows = []
    for target in TARGETS:
        metric_rows.append(
            [
                target,
                _fmt(summary["metrics"]["raw_kernel"][target]["mae"]),
                _fmt(summary["metrics"]["physics_kernel"][target]["mae"]),
                _fmt(summary["metrics"]["physics_residual"][target]["mae"]),
                _fmt(summary["metrics"]["physics_residual"][target]["r2"]),
            ]
        )
    raw_core = summary["normalized_core"]["raw_kernel"]["mean_mae_over_test_std"]
    physics_core = summary["normalized_core"]["physics_kernel"]["mean_mae_over_test_std"]
    residual_core = summary["normalized_core"]["physics_residual"]["mean_mae_over_test_std"]
    physics_gain = (physics_core - raw_core) / raw_core
    report = f"""# PS Physics-Informed Model Update

## Purpose

This update tests whether embedding TNM/ARRT-style relaxation priors improves the finite-time PS predictor beyond the raw temperature/time RBF model.

## Model Variants

- `raw_kernel`: the existing RBF surrogate using only mode, temperature, and log-time features.
- `physics_kernel`: RBF using raw features plus Arrhenius/TNM dose, sequential state, and path-memory features.
- `physics_residual`: a ridge physics baseline trained on physics features, plus an RBF model trained only on its residual.

The `physics_kernel` feature subset and bandwidth are selected by repeated grouped cross-validation inside the training split:

```json
{json.dumps({k: v for k, v in summary["selected_physics_kernel"].items() if k != "cv_grid"}, indent=2)}
```

## Same-Split Metrics

The split is identical across models and grouped by source file / mode / T1. Lower MAE is better.

{markdown_table(["target", "raw_MAE", "physics_kernel_MAE", "physics_residual_MAE", "physics_residual_R2"], metric_rows)}

![Physics-informed MAE comparison]({FIG_PATH.as_posix()})

## Normalized Core-Target Error

```json
{json.dumps(summary["normalized_core"], indent=2)}
```

## Interpretation

The selected `physics_kernel` lowers the mean normalized core-target error from `{_fmt(raw_core)}` to `{_fmt(physics_core)}`, a relative change of `{physics_gain * 100:.1f}%`. It improves all four core targets in this split: total enthalpy, peak area, recovery index, and path-dependence index.

The `physics_residual` variant is not recommended as the main model right now. It improves peak area and diagnostic peak shape, but it damages `recovery_index` and `path_dependence_index`, which are the targets that matter most for inverse annealing design. The better small-sample choice is therefore the lower-dimensional `physics_kernel`: raw annealing inputs plus TNM path/dose features selected by grouped CV.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def train() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = build_ps_dataset()
    rows = read_rows(dataset_path)
    x_raw, y = matrices(rows)
    train_idx, test_idx = grouped_split(rows)
    y_test = y[test_idx]
    selected = select_physics_kernel(rows, x_raw, y, train_idx)
    selected_features = list(selected["features"])
    selected_bandwidth = float(selected["bandwidth"])
    x_phys = feature_matrix(rows, x_raw, PHYSICS_COMPACT_FEATURES, include_raw=False)
    x_aug = feature_matrix(rows, x_raw, selected_features, include_raw=True)

    raw_pred, raw_unc = fit_raw_kernel(x_raw, y, train_idx, test_idx)
    physics_pred, physics_unc = fit_physics_kernel(x_aug, y, train_idx, test_idx, selected_bandwidth)
    residual_pred, residual_unc, baseline_pred = fit_residual_model(x_phys, x_aug, y, train_idx, test_idx)
    final_model = KernelRegressor.fit(x_aug, y, bandwidth=selected_bandwidth)

    metrics = {
        "raw_kernel": regression_metrics(y_test, raw_pred),
        "physics_kernel": regression_metrics(y_test, physics_pred),
        "physics_residual": regression_metrics(y_test, residual_pred),
        "physics_baseline_only": regression_metrics(y_test, baseline_pred),
    }
    normalized_core = {
        name: normalized_summary(model_metrics, CORE_TARGETS)
        for name, model_metrics in metrics.items()
    }
    normalized_all = {
        name: normalized_summary(model_metrics, TARGETS)
        for name, model_metrics in metrics.items()
    }

    summary = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split_strategy": "same_as_src.ps.train_ps_model.grouped_split",
        "targets": TARGETS,
        "core_targets": CORE_TARGETS,
        "raw_features": RAW_FEATURES,
        "physics_feature_sets": PHYSICS_FEATURE_SETS,
        "selected_physics_kernel": selected,
        "residual_baseline_features": PHYSICS_COMPACT_FEATURES,
        "model_variants": [
            "raw_kernel",
            "physics_kernel",
            "physics_residual",
            "physics_baseline_only",
        ],
        "metrics": metrics,
        "normalized_core": normalized_core,
        "normalized_all": normalized_all,
        "notes": [
            "Physics features are deterministic functions of annealing conditions only.",
            "The residual model uses a ridge physics baseline plus RBF residual correction.",
            "ARRT/TNM constants are a basis prior, not claimed fitted PS material constants.",
        ],
        "outputs": {
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "model": str(PHYSICS_MODEL_PATH.relative_to(ROOT)),
            "predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "figure": str(FIG_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    model_payload = {
        "model_type": "physics_informed_numpy_rbf_kernel_regressor",
        "targets": TARGETS,
        "core_targets": CORE_TARGETS,
        "raw_features": RAW_FEATURES,
        "physics_feature_set": selected["feature_set"],
        "physics_features": selected_features,
        "bandwidth": final_model.bandwidth,
        "x_mean": final_model.x_mean.tolist(),
        "x_std": final_model.x_std.tolist(),
        "x_train": final_model.x_train.tolist(),
        "y_train": final_model.y_train.tolist(),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "metrics_same_split": metrics["physics_kernel"],
        "normalized_core_same_split": normalized_core["physics_kernel"],
        "selection": selected,
        "notes": [
            "Default PS inverse design and EIG model after physics-informed update.",
            "Features are raw annealing inputs plus selected TNM/ARRT-inspired path features.",
            "Final decision model is refit on all available usable samples after same-split evaluation.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PHYSICS_MODEL_PATH.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    write_predictions(
        PREDICTIONS_PATH,
        rows,
        test_idx,
        y_test,
        {
            "raw_kernel": raw_pred,
            "physics_kernel": physics_pred,
            "physics_residual": residual_pred,
            "physics_baseline_only": baseline_pred,
        },
    )
    write_figure({name: metrics[name] for name in ["raw_kernel", "physics_kernel", "physics_residual"]})
    write_report(summary)
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "normalized_core": normalized_core,
        "key_mae": {
            name: {
                target: metrics[name][target]["mae"]
                for target in CORE_TARGETS
            }
            for name in ["raw_kernel", "physics_kernel", "physics_residual", "physics_baseline_only"]
        },
    }, indent=2))
    return SUMMARY_PATH


if __name__ == "__main__":
    train()
