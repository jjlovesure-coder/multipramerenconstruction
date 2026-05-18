"""Inverse design search over two-step annealing parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.models.dataset import _read_csv, _nearest_by_log_time, _nearest_fig3_h, S7, FIG3_H
from src.models.kernel_regression import KernelRegressor
from src.models.train_forward import FEATURES, TARGETS, MODEL_DIR
from src.physics.arrt import activation_enthalpy_from_entropy
from src.physics.tnm_features import stretched_dose


ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path: Path | None = None) -> KernelRegressor:
    path = path or MODEL_DIR / "forward_kernel_model.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return KernelRegressor(
        x_train=np.array(data["x_train"], dtype=float),
        y_train=np.array(data["y_train"], dtype=float),
        x_mean=np.array(data["x_mean"], dtype=float),
        x_std=np.array(data["x_std"], dtype=float),
        bandwidth=float(data["bandwidth"]),
    )


def feature_row(t1_temperature_k: float, t1_s: float, t2_temperature_k: float, t2_s: float) -> list[float]:
    s7 = _read_csv(S7)
    fig3_h = _read_csv(FIG3_H)
    s1 = float(_nearest_by_log_time(s7, t1_temperature_k, t1_s)["s_star_j_mol_k"])
    s2 = float(_nearest_by_log_time(s7, t2_temperature_k, t2_s)["s_star_j_mol_k"])
    h1 = activation_enthalpy_from_entropy(t1_temperature_k, t1_s, s1)
    h2 = activation_enthalpy_from_entropy(t2_temperature_k, t2_s, s2)
    h_fig3_1 = _nearest_fig3_h(fig3_h, t1_s)
    h_fig3_2 = _nearest_fig3_h(fig3_h, t2_s)
    values = {
        "t1_temperature_k": t1_temperature_k,
        "t2_temperature_k": t2_temperature_k,
        "delta_t_k": t2_temperature_k - t1_temperature_k,
        "log_t1": np.log10(t1_s),
        "log_t2": np.log10(t2_s),
        "inv_t1_k": 1.0 / t1_temperature_k,
        "inv_t2_k": 1.0 / t2_temperature_k,
        "s1_j_mol_k": s1,
        "s2_j_mol_k": s2,
        "delta_s_j_mol_k": s2 - s1,
        "h1_kj_mol_arrt": h1,
        "h2_kj_mol_arrt": h2,
        "h1_kj_mol_fig3": float(h_fig3_1["h_star_kj_mol"]),
        "h2_kj_mol_fig3": float(h_fig3_2["h_star_kj_mol"]),
        "tp1_1000kps_k_fig3": float(h_fig3_1["tp_1000kps_k"]),
        "tp2_1000kps_k_fig3": float(h_fig3_2["tp_1000kps_k"]),
        "tnm_dose1": stretched_dose(t1_temperature_k, t1_s),
        "tnm_dose2": stretched_dose(t2_temperature_k, t2_s),
    }
    return [float(values[name]) for name in FEATURES]


def candidate_grid() -> Iterable[tuple[float, float, float, float]]:
    low_temps = [348.0, 353.0, 358.0, 363.0, 368.0, 373.0]
    high_temps = [373.0, 378.0, 383.0]
    times = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    for t1_temp in low_temps:
        for t2_temp in high_temps:
            if t2_temp < t1_temp:
                continue
            for t1_s in times:
                for t2_s in times:
                    yield t1_temp, t1_s, t2_temp, t2_s


def search(target: dict[str, float], top_k: int = 10) -> Path:
    model = load_model()
    rows = []
    xs = []
    candidates = list(candidate_grid())
    for c in candidates:
        xs.append(feature_row(*c))
    pred, uncertainty = model.predict(np.array(xs, dtype=float))

    target_vec = np.array([target.get(name, np.nan) for name in TARGETS], dtype=float)
    active = ~np.isnan(target_vec)
    scale_map = {
        "delta_h_kj_mol": 0.08,
        "delta_h_peak_kj_mol": 0.04,
        "s2_j_mol_k": 30.0,
        "delta_s_j_mol_k": 40.0,
        "memory_effect_label": 1.0,
    }
    scale = np.array([scale_map[name] for name in TARGETS], dtype=float)
    losses = np.mean(np.abs((pred[:, active] - target_vec[active]) / scale[active]), axis=1)
    order = np.argsort(losses)[:top_k]
    for rank, idx in enumerate(order, start=1):
        t1_temp, t1_s, t2_temp, t2_s = candidates[idx]
        result = {
            "rank": rank,
            "loss": float(losses[idx]),
            "t1_temperature_k": t1_temp,
            "t1_s": t1_s,
            "t2_temperature_k": t2_temp,
            "t2_s": t2_s,
        }
        for j, name in enumerate(TARGETS):
            result[f"pred_{name}"] = float(pred[idx, j])
            result[f"uncertainty_{name}"] = float(uncertainty[idx, j])
        rows.append(result)

    out = PRED_DIR / "inverse_design_top10.json"
    out.write_text(json.dumps({"target": target, "results": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"target": target, "top_result": rows[0], "output": str(out)}, indent=2))
    return out


if __name__ == "__main__":
    search({"delta_h_kj_mol": 0.55, "delta_h_peak_kj_mol": 0.12, "memory_effect_label": 1.0})
