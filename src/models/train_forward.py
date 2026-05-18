"""Train/evaluate a sparse forward predictor for annealing response."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.models.dataset import build_dataset
from src.models.kernel_regression import KernelRegressor


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "t1_temperature_k",
    "t2_temperature_k",
    "delta_t_k",
    "log_t1",
    "log_t2",
    "inv_t1_k",
    "inv_t2_k",
    "s1_j_mol_k",
    "s2_j_mol_k",
    "h1_kj_mol_arrt",
    "h2_kj_mol_arrt",
    "h1_kj_mol_fig3",
    "h2_kj_mol_fig3",
    "tp1_1000kps_k_fig3",
    "tp2_1000kps_k_fig3",
    "tnm_dose1",
    "tnm_dose2",
]

BASELINE_FEATURES = [
    "t1_temperature_k",
    "t2_temperature_k",
    "delta_t_k",
    "log_t1",
    "log_t2",
    "inv_t1_k",
    "inv_t2_k",
    "s1_j_mol_k",
    "s2_j_mol_k",
    "h1_kj_mol_arrt",
    "h2_kj_mol_arrt",
    "tnm_dose1",
    "tnm_dose2",
]

TARGETS = [
    "delta_h_kj_mol",
    "delta_h_peak_kj_mol",
    "s2_j_mol_k",
    "delta_s_j_mol_k",
    "memory_effect_label",
]


def load_matrix(path: Path, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    features = features or FEATURES
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r[c]) for c in features] for r in rows], dtype=float)
    y = np.array([[float(r[c]) for c in TARGETS] for r in rows], dtype=float)
    return x, y, rows


def loo_metrics(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.25) -> dict[str, float]:
    preds = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        model = KernelRegressor.fit(x[mask], y[mask], bandwidth=bandwidth)
        pred, _ = model.predict(x[i])
        preds.append(pred[0])
    pred = np.array(preds)
    mae = np.mean(np.abs(pred - y), axis=0)
    return {f"loo_mae_{name}": float(value) for name, value in zip(TARGETS, mae)}


def train() -> Path:
    dataset_path = build_dataset()
    x, y, rows = load_matrix(dataset_path, FEATURES)
    x_base, _, _ = load_matrix(dataset_path, BASELINE_FEATURES)
    baseline_metrics = loo_metrics(x_base, y)
    metrics = loo_metrics(x, y)
    model = KernelRegressor.fit(x, y, bandwidth=1.25)

    payload = {
        "model_type": "numpy_rbf_kernel_regressor",
        "features": FEATURES,
        "baseline_features": BASELINE_FEATURES,
        "targets": TARGETS,
        "bandwidth": model.bandwidth,
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "x_train": model.x_train.tolist(),
        "y_train": model.y_train.tolist(),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "metrics": metrics,
        "baseline_metrics_without_fig3_h_tp": baseline_metrics,
        "notes": [
            "Physics-informed features include ARRT-derived H*, Fig.3-derived H*/Tp, and simplified TNM dose.",
            "Digitized sparse data are approximate; use predictions for screening, not precision design.",
        ],
    }

    out = MODEL_DIR / "forward_kernel_model.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(out), "baseline_metrics": baseline_metrics, "enhanced_metrics": metrics}, indent=2))
    return out


if __name__ == "__main__":
    train()
