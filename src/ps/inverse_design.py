"""Finite-time inverse design for PS enthalpy-recovery conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.train_ps_model import FEATURES, TARGETS, featurize


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results" / "ps" / "models" / "ps_limited_time_model.json"
OUT_DIR = ROOT / "results" / "ps" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path: Path = MODEL_PATH) -> KernelRegressor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return KernelRegressor(
        x_train=np.array(data["x_train"], dtype=float),
        y_train=np.array(data["y_train"], dtype=float),
        x_mean=np.array(data["x_mean"], dtype=float),
        x_std=np.array(data["x_std"], dtype=float),
        bandwidth=float(data["bandwidth"]),
    )


def candidate_rows() -> list[dict[str, str]]:
    rows = []
    temps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    times = [10, 30, 60, 100, 300, 600, 900, 1200, 1800]
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
        for t2 in temps:
            if t2 < t1:
                continue
            for t1_s in times:
                for t2_s in times:
                    if max(t1_s, t2_s) > 1800:
                        continue
                    rows.append({
                        "mode": "two_step",
                        "T1_C": str(t1),
                        "t1_s": str(t1_s),
                        "T2_C": str(t2),
                        "t2_s": str(t2_s),
                        "total_anneal_time_s": str(t1_s + t2_s),
                    })
    return rows


def search(target: dict[str, float] | None = None, top_k: int = 10) -> Path:
    target = target or {"recovery_index": 0.8}
    model = load_model()
    candidates = candidate_rows()
    x = np.array([featurize(row) for row in candidates], dtype=float)
    pred, unc = model.predict(x)

    scale = {
        "delta_h_total_J_g": 10.0,
        "peak_area_J_g": 5.0,
        "peak_temperature_Tp_C": 5.0,
        "peak_height_uW": 500.0,
        "onset_temperature_C": 5.0,
        "recovery_index": 0.15,
        "path_dependence_index": 1.0,
        "kovacs_peak_label": 1.0,
    }
    active = [TARGETS.index(name) for name in target]
    losses = np.zeros(len(candidates))
    for name, value in target.items():
        idx = TARGETS.index(name)
        losses += np.abs((pred[:, idx] - float(value)) / scale[name])
    losses = losses / max(len(active), 1)
    total_time = np.array([float(row["total_anneal_time_s"]) for row in candidates])
    losses += 0.05 * total_time / 1800.0

    order = np.argsort(losses)[:top_k]
    results = []
    for rank, idx in enumerate(order, start=1):
        row = dict(candidates[idx])
        row["rank"] = rank
        row["loss"] = float(losses[idx])
        for j, target_name in enumerate(TARGETS):
            row[f"pred_{target_name}"] = float(pred[idx, j])
            row[f"uncertainty_{target_name}"] = float(unc[idx, j])
        results.append(row)

    output = OUT_DIR / "ps_inverse_design_top10.json"
    output.write_text(json.dumps({"target": target, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"target": target, "top_result": results[0], "output": str(output)}, indent=2))
    return output


if __name__ == "__main__":
    search({"recovery_index": 0.8})
