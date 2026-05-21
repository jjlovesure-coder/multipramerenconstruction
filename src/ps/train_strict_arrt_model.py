"""Train PS predictor using strictly calculated ARRT/Kissinger S*/H* labels."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.calculate_arrt_from_heating_rates import OUT_DIR as ARRT_DIR
from src.ps.calculate_arrt_from_heating_rates import main as calculate_arrt
from src.ps.dsc_processing import ROOT
from src.ps.train_ps_model import featurize


OUT_DIR = ROOT / "results" / "ps" / "strict_arrt_model"
TARGETS = [
    "activation_enthalpy_H_star_kj_mol",
    "activation_entropy_S_star_j_mol_K",
]
BANDWIDTH = 1.2


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if all(finite(row.get(target)) for target in TARGETS)]


def condition_id(row: dict[str, str]) -> str:
    return str(row.get("condition_key") or "|".join([
        row.get("mode", ""),
        row.get("T1_C", ""),
        row.get("t1_s", ""),
        row.get("T2_C", ""),
        row.get("t2_s", ""),
    ]))


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for j, target in enumerate(TARGETS):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        metrics[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "r2": float(1.0 - np.sum(err * err) / denom) if denom > 0.0 else float("nan"),
        }
    return metrics


def leave_one_condition_out_predictions(rows: list[dict[str, str]]) -> dict[str, object]:
    """Evaluate strict ARRT labels with condition-level leave-one-out CV."""
    rows = clean_rows(rows)
    if len(rows) < 3:
        raise ValueError("leave-one-condition-out evaluation needs at least three strict ARRT rows")

    x = np.array([featurize(row) for row in rows], dtype=float)
    y = np.array([[float(row[target]) for target in TARGETS] for row in rows], dtype=float)
    predictions = []
    y_pred = np.zeros_like(y)
    for test_idx in range(len(rows)):
        train_idx = np.array([idx for idx in range(len(rows)) if idx != test_idx], dtype=int)
        model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=BANDWIDTH)
        pred, unc = model.predict(x[test_idx])
        y_pred[test_idx] = pred[0]
        row = rows[test_idx]
        record: dict[str, object] = {
            "condition_key": condition_id(row),
            "mode": row.get("mode", ""),
            "T1_C": row.get("T1_C", ""),
            "t1_s": row.get("t1_s", ""),
            "T2_C": row.get("T2_C", ""),
            "t2_s": row.get("t2_s", ""),
            "kissinger_r2": float(row["kissinger_r2"]) if finite(row.get("kissinger_r2")) else math.nan,
            "kissinger_confidence": row.get("kissinger_confidence", ""),
        }
        for j, target in enumerate(TARGETS):
            actual = float(row[target])
            predicted = float(pred[0, j])
            record[f"actual_{target}"] = actual
            record[f"pred_{target}"] = predicted
            record[f"error_{target}"] = predicted - actual
            record[f"uncertainty_{target}"] = float(unc[0, j])
        predictions.append(record)

    return {
        "n_samples": len(rows),
        "targets": TARGETS,
        "metrics": target_metrics(y, y_pred),
        "predictions": predictions,
    }


def train() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calculate_arrt()
    rows = clean_rows(read_csv(ARRT_DIR / "arrt_kissinger_results.csv"))
    if len(rows) < 5:
        payload = {
            "status": "skipped",
            "reason": "Strict ARRT/Kissinger S*/H* labels require at least five calculable annealed states for a useful train/test split.",
            "n_calculable_rows": len(rows),
            "targets": TARGETS,
            "input": str((ARRT_DIR / "arrt_kissinger_results.csv").relative_to(ROOT)),
        }
        out = OUT_DIR / "strict_arrt_training_skipped.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return out

    x = np.array([featurize(row) for row in rows], dtype=float)
    y = np.array([[float(row[target]) for target in TARGETS] for row in rows], dtype=float)
    model = KernelRegressor.fit(x, y, bandwidth=BANDWIDTH)
    loocv = leave_one_condition_out_predictions(rows)
    high_conf_rows = [row for row in rows if row.get("kissinger_confidence") != "low"]
    high_conf_loocv = leave_one_condition_out_predictions(high_conf_rows) if len(high_conf_rows) >= 3 else None
    payload = {
        "status": "trained",
        "model_type": "strict_arrt_numpy_rbf_kernel_regressor",
        "targets": TARGETS,
        "features": "src.ps.train_ps_model.FEATURES",
        "bandwidth": model.bandwidth,
        "n_samples": len(rows),
        "n_low_confidence": sum(1 for row in rows if row.get("kissinger_confidence") == "low"),
        "n_high_confidence": sum(1 for row in rows if row.get("kissinger_confidence") != "low"),
        "evaluation_policy": "leave-one-condition-out CV because strict ARRT labels are condition-level small samples",
        "metrics": loocv["metrics"],
        "loocv": loocv,
        "high_confidence_only_loocv": high_conf_loocv,
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "x_train": model.x_train.tolist(),
        "y_train": model.y_train.tolist(),
        "conditions": [
            {
                "condition_key": condition_id(row),
                "mode": row.get("mode", ""),
                "T1_C": row.get("T1_C", ""),
                "t1_s": row.get("t1_s", ""),
                "T2_C": row.get("T2_C", ""),
                "t2_s": row.get("t2_s", ""),
                "total_anneal_time_s": row.get("total_anneal_time_s", ""),
                "kissinger_r2": row.get("kissinger_r2", ""),
                "kissinger_confidence": row.get("kissinger_confidence", ""),
            }
            for row in rows
        ],
    }
    out = OUT_DIR / "strict_arrt_model.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return out


if __name__ == "__main__":
    train()
