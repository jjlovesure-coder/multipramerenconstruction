"""Train PS predictor using strictly calculated ARRT/Kissinger S*/H* labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.calculate_arrt_from_heating_rates import OUT_DIR as ARRT_DIR
from src.ps.calculate_arrt_from_heating_rates import main as calculate_arrt
from src.ps.dsc_processing import ROOT
from src.ps.train_ps_model import featurize, grouped_split


OUT_DIR = ROOT / "results" / "ps" / "strict_arrt_model"
TARGETS = [
    "activation_enthalpy_H_star_kj_mol",
    "activation_entropy_S_star_j_mol_K",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def train() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calculate_arrt()
    rows = read_csv(ARRT_DIR / "arrt_kissinger_results.csv")
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
    train_idx, test_idx = grouped_split(rows, test_fraction=0.25)
    model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=1.2)
    pred, unc = model.predict(x[test_idx])
    metrics = {}
    for j, target in enumerate(TARGETS):
        err = pred[:, j] - y[test_idx, j]
        metrics[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
        }
    payload = {
        "status": "trained",
        "targets": TARGETS,
        "n_samples": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "metrics": metrics,
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "x_train": model.x_train.tolist(),
        "y_train": model.y_train.tolist(),
    }
    out = OUT_DIR / "strict_arrt_model.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return out


if __name__ == "__main__":
    train()
