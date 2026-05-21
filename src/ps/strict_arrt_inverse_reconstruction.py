"""Inverse reconstruction using strictly measured ARRT H*/S* labels."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.calculate_arrt_from_heating_rates import OUT_DIR as ARRT_DIR
from src.ps.dsc_processing import ROOT
from src.ps.inverse_design import candidate_rows, posterior_summary
from src.ps.train_ps_model import featurize
from src.ps.train_strict_arrt_model import OUT_DIR as MODEL_DIR
from src.ps.train_strict_arrt_model import TARGETS, clean_rows, read_csv, train as train_strict_arrt


OUT_DIR = ROOT / "results" / "ps" / "strict_arrt_model"
MODEL_PATH = MODEL_DIR / "strict_arrt_model.json"
TARGET_SCALES = {
    "activation_enthalpy_H_star_kj_mol": 50.0,
    "activation_entropy_S_star_j_mol_K": 150.0,
}


def posterior_from_losses(rows: list[dict[str, str]], losses: np.ndarray, percentile: float = 5.0) -> dict[str, object]:
    return posterior_summary(rows, losses, percentile=percentile)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_model(path: Path = MODEL_PATH) -> tuple[dict[str, object], KernelRegressor]:
    if not path.exists():
        train_strict_arrt()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "trained":
        raise RuntimeError(f"Strict ARRT model is not trained: {payload.get('reason', 'unknown reason')}")
    model = KernelRegressor(
        x_train=np.array(payload["x_train"], dtype=float),
        y_train=np.array(payload["y_train"], dtype=float),
        x_mean=np.array(payload["x_mean"], dtype=float),
        x_std=np.array(payload["x_std"], dtype=float),
        bandwidth=payload.get("bandwidth", 1.2),
    )
    return payload, model


def _feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([featurize(row) for row in rows], dtype=float)


def _target_loss(pred: np.ndarray, target: dict[str, float]) -> float:
    terms = []
    for idx, name in enumerate(TARGETS):
        scale = TARGET_SCALES[name]
        terms.append(abs((float(pred[idx]) - float(target[name])) / scale))
    return float(np.mean(terms))


def _actual_parameters(row: dict[str, str]) -> dict[str, float]:
    t2 = float(row["T2_C"]) if row.get("T2_C") not in {"", "nan", "NaN"} else float(row["T1_C"])
    t2_s = float(row["t2_s"]) if row.get("t2_s") not in {"", "nan", "NaN"} else float(row["t1_s"])
    return {
        "T1_C": float(row["T1_C"]),
        "T2_C": t2,
        "log10_t1_s": float(np.log10(max(float(row["t1_s"]), 1e-9))),
        "log10_t2_s": float(np.log10(max(t2_s, 1e-9))),
    }


def _posterior_contains(posterior: dict[str, object], actual: dict[str, float], name: str) -> bool:
    stats = posterior.get(name, {})
    if not isinstance(stats, dict):
        return False
    return float(stats["p5"]) <= actual[name] <= float(stats["p95"])


def reconstruct(top_k: int = 10) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload, model = load_model()
    strict_rows = clean_rows(read_csv(ARRT_DIR / "arrt_kissinger_results.csv"))
    candidates = candidate_rows()
    x_candidates = _feature_matrix(candidates)
    pred, unc = model.predict(x_candidates)

    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for row in strict_rows:
        target = {name: float(row[name]) for name in TARGETS}
        target_losses = np.array([_target_loss(pred[i], target) for i in range(len(candidates))], dtype=float)
        mean_unc = np.mean(unc, axis=1)
        total_time = np.array([float(candidate["total_anneal_time_s"]) for candidate in candidates], dtype=float)
        losses = target_losses + 0.02 * mean_unc + 0.03 * total_time / 1800.0
        order = np.argsort(losses)[:top_k]
        posterior = posterior_from_losses(candidates, losses)
        actual = _actual_parameters(row)
        best = candidates[int(order[0])]
        best_actual = _actual_parameters(best)
        summary: dict[str, object] = {
            "condition_key": row.get("condition_key", ""),
            "mode": row.get("mode", ""),
            "kissinger_r2": float(row["kissinger_r2"]) if finite(row.get("kissinger_r2")) else math.nan,
            "kissinger_confidence": row.get("kissinger_confidence", ""),
            "posterior": posterior,
            "actual": actual,
            "best_candidate": best,
            "best_loss": float(losses[int(order[0])]),
        }
        for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]:
            stats = posterior.get(name, {})
            width = float(stats["width"]) if isinstance(stats, dict) else math.nan
            median = float(stats["median"]) if isinstance(stats, dict) else math.nan
            summary[f"posterior_contains_actual_{name}"] = _posterior_contains(posterior, actual, name)
            summary[f"posterior_width_{name}"] = width
            summary[f"posterior_center_error_{name}"] = median - actual[name]
            summary[f"best_error_{name}"] = best_actual[name] - actual[name]
        summaries.append(summary)

        for rank, idx in enumerate(order, start=1):
            candidate = dict(candidates[int(idx)])
            candidate["condition_key"] = row.get("condition_key", "")
            candidate["rank"] = rank
            candidate["loss"] = float(losses[int(idx)])
            candidate["target_loss"] = float(target_losses[int(idx)])
            candidate["mean_uncertainty"] = float(mean_unc[int(idx)])
            for j, name in enumerate(TARGETS):
                candidate[f"target_{name}"] = target[name]
                candidate[f"pred_{name}"] = float(pred[int(idx), j])
                candidate[f"error_{name}"] = float(pred[int(idx), j] - target[name])
            records.append(candidate)

    summary_metrics = {
        "n_samples": len(strict_rows),
        "targets": TARGETS,
        "model_n_samples": payload.get("n_samples"),
        "posterior_coverage": {
            name: float(np.mean([bool(item[f"posterior_contains_actual_{name}"]) for item in summaries])) if summaries else math.nan
            for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]
        },
        "best_mae": {
            name: float(np.mean([abs(float(item[f"best_error_{name}"])) for item in summaries])) if summaries else math.nan
            for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]
        },
        "posterior_width_mean": {
            name: float(np.mean([float(item[f"posterior_width_{name}"]) for item in summaries])) if summaries else math.nan
            for name in ["T1_C", "T2_C", "log10_t1_s", "log10_t2_s"]
        },
        "interpretation": "Strict H*/S* labels improve the physical target definition, but inverse uniqueness is judged by posterior width and coverage, not by top-1 alone.",
    }

    candidates_path = OUT_DIR / "strict_arrt_inverse_top_candidates.csv"
    if records:
        with candidates_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary_path = OUT_DIR / "strict_arrt_inverse_summary.json"
    summary_path.write_text(json.dumps({"summary": summary_metrics, "by_condition": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(summary_path), "summary": summary_metrics}, indent=2))
    return summary_path


if __name__ == "__main__":
    reconstruct()
