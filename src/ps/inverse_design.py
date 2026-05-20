"""Finite-time inverse design for PS enthalpy-recovery conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.physics_features import physics_feature_dict
from src.ps.train_physics_informed_ps_model import PHYSICS_BETA_FEATURES
from src.ps.train_physics_informed_ps_model import TARGETS


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results" / "ps" / "models" / "ps_physics_informed_kernel_model.json"
OUT_DIR = ROOT / "results" / "ps" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path: Path = MODEL_PATH) -> KernelRegressor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return KernelRegressor(
        x_train=np.array(data["x_train"], dtype=float),
        y_train=np.array(data["y_train"], dtype=float),
        x_mean=np.array(data["x_mean"], dtype=float),
        x_std=np.array(data["x_std"], dtype=float),
        bandwidth=data["bandwidth"],
    )


def load_model_payload(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_features(row: dict[str, str]) -> list[float]:
    is_two = 1.0 if row["mode"] == "two_step" else 0.0
    t1 = max(float(row["t1_s"]), 1e-9)
    t2_raw = row.get("t2_s", "")
    t2 = max(float(t2_raw), 1e-9) if t2_raw not in {"", "nan", "NaN"} else 1e-9
    T1 = float(row["T1_C"])
    T2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else T1
    total = max(float(row["total_anneal_time_s"]), 1e-9)
    return [
        is_two,
        T1,
        T2,
        T2 - T1,
        float(np.log10(t1)),
        float(np.log10(t2)),
        float(np.log10(total)),
    ]


def candidate_feature_matrix(rows: list[dict[str, str]], payload: dict | None = None) -> np.ndarray:
    payload = payload or (load_model_payload() if MODEL_PATH.exists() else {})
    physics_features = list(payload.get("physics_features", PHYSICS_BETA_FEATURES))
    matrix = []
    for row in rows:
        physics = physics_feature_dict(row)
        matrix.append(raw_features(row) + [float(physics[name]) for name in physics_features])
    return np.array(matrix, dtype=float)


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


def _target_loss(pred: np.ndarray, target: dict[str, float]) -> float:
    scale = {
        "delta_h_total_J_g": 10.0,
        "peak_area_J_g": 5.0,
        "peak_temperature_Tp_C": 5.0,
        "peak_height_uW": 500.0,
        "recovery_index": 0.15,
        "path_dependence_index": 1.0,
    }
    loss = 0.0
    for name, value in target.items():
        idx = TARGETS.index(name)
        loss += abs((float(pred[idx]) - float(value)) / scale[name])
    return loss / max(len(target), 1)


def _row_from_vector(values: np.ndarray) -> dict[str, str]:
    t1, log_t1, t2, log_t2 = values
    t1_s = float(np.clip(10.0 ** log_t1, 10.0, 1800.0))
    t2_s = float(np.clip(10.0 ** log_t2, 10.0, 1800.0))
    return {
        "mode": "two_step",
        "T1_C": f"{float(np.clip(t1, 50.0, 100.0)):.6g}",
        "t1_s": f"{t1_s:.6g}",
        "T2_C": f"{float(np.clip(t2, 50.0, 100.0)):.6g}",
        "t2_s": f"{t2_s:.6g}",
        "total_anneal_time_s": f"{t1_s + t2_s:.6g}",
    }


def _vector_from_row(row: dict[str, str]) -> np.ndarray:
    t2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else float(row["T1_C"])
    t2_s = float(row["t2_s"]) if row.get("t2_s") not in {"", "nan", "NaN"} else float(row["t1_s"])
    return np.array([
        float(row["T1_C"]),
        np.log10(max(float(row["t1_s"]), 10.0)),
        t2,
        np.log10(max(t2_s, 10.0)),
    ])


def refine_candidate(
    row: dict[str, str],
    target: dict[str, float],
    payload: dict,
    model: KernelRegressor,
) -> tuple[dict[str, str], bool]:
    """Continuously refine a two-step grid candidate when scipy is available."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return row, False

    def objective(values: np.ndarray) -> float:
        candidate = _row_from_vector(values)
        pred, unc = model.predict(candidate_feature_matrix([candidate], payload))
        total_time = float(candidate["total_anneal_time_s"])
        return _target_loss(pred[0], target) + 0.05 * total_time / 1800.0 + 0.02 * float(np.mean(unc[0]))

    start = _vector_from_row(row)
    bounds = [(50.0, 100.0), (1.0, np.log10(1800.0)), (50.0, 100.0), (1.0, np.log10(1800.0))]
    result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-3})
    candidate = _row_from_vector(np.asarray(result.x, dtype=float))
    if objective(start) <= objective(_vector_from_row(candidate)):
        return row, True
    return candidate, True


def pareto_front(rows: list[dict]) -> list[dict]:
    """Return non-dominated rows for loss/time/uncertainty/path-dependence."""
    metrics = ("loss", "total_anneal_time_s", "mean_uncertainty", "abs_path_dependence")
    front = []
    for row in rows:
        dominated = False
        row_values = np.array([float(row[metric]) for metric in metrics], dtype=float)
        for other in rows:
            if other is row:
                continue
            other_values = np.array([float(other[metric]) for metric in metrics], dtype=float)
            if np.all(other_values <= row_values) and np.any(other_values < row_values):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def search(target: dict[str, float] | None = None, top_k: int = 10) -> Path:
    target = target or {"recovery_index": 0.8}
    payload = load_model_payload()
    model = load_model()
    candidates = candidate_rows()
    x = candidate_feature_matrix(candidates, payload)
    pred, unc = model.predict(x)

    losses = np.zeros(len(candidates))
    for i in range(len(candidates)):
        losses[i] = _target_loss(pred[i], target)
    total_time = np.array([float(row["total_anneal_time_s"]) for row in candidates])
    losses += 0.05 * total_time / 1800.0

    order = np.argsort(losses)[:max(top_k * 3, top_k)]
    results = []
    scipy_used = False
    for idx in order:
        row = dict(candidates[idx])
        if row["mode"] == "two_step":
            row, refined = refine_candidate(row, target, payload, model)
            scipy_used = scipy_used or refined
            row["refined_continuously"] = refined
            pred_one, unc_one = model.predict(candidate_feature_matrix([row], payload))
            row_loss = _target_loss(pred_one[0], target) + 0.05 * float(row["total_anneal_time_s"]) / 1800.0
            pred_values = pred_one[0]
            unc_values = unc_one[0]
        else:
            row["refined_continuously"] = False
            row_loss = float(losses[idx])
            pred_values = pred[idx]
            unc_values = unc[idx]
        row["loss"] = float(row_loss)
        row["mean_uncertainty"] = float(np.mean(unc_values))
        row["abs_path_dependence"] = float(abs(pred_values[TARGETS.index("path_dependence_index")]))
        for j, target_name in enumerate(TARGETS):
            row[f"pred_{target_name}"] = float(pred_values[j])
            row[f"uncertainty_{target_name}"] = float(unc_values[j])
        results.append(row)

    results = sorted(results, key=lambda item: float(item["loss"]))[:top_k]
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    front = pareto_front(results)
    for row in front:
        row["pareto_optimal"] = True
    for row in results:
        row.setdefault("pareto_optimal", False)

    output = OUT_DIR / "ps_inverse_design_top10.json"
    output.write_text(json.dumps({"target": target, "results": results, "pareto_front": front, "continuous_refinement_used": scipy_used}, indent=2), encoding="utf-8")
    print(json.dumps({"target": target, "top_result": results[0], "output": str(output)}, indent=2))
    return output


if __name__ == "__main__":
    search({"recovery_index": 0.8})
