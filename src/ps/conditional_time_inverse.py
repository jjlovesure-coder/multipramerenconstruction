"""Conditional t1/t2 inverse search for fixed PS two-step temperatures.

This module implements the reduced inverse problem requested after the
four-parameter inverse search proved underdetermined: T1 and T2 are treated as
known experimental settings, and only t1/t2 are searched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.compare_strict_arrt_paper_style import prepare_rows, usable_rows
from src.ps.dsc_processing import ROOT
from src.ps.train_ps_model import featurize


OUT_DIR = ROOT / "results" / "ps" / "conditional_time_inverse"
MODEL_PATH = ROOT / "results" / "ps" / "strict_arrt_model" / "strict_paper_style_benchmark.json"

PAPER_STYLE_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "activation_enthalpy_H_star_kj_mol",
    "activation_entropy_S_star_j_mol_K",
]

TARGET_SCALES = {
    "delta_h_total_J_g": 0.20,
    "peak_area_J_g": 0.13,
    "activation_enthalpy_H_star_kj_mol": 50.0,
    "activation_entropy_S_star_j_mol_K": 150.0,
}

EXPERIMENT_TIMES_S = [10, 30, 60, 100, 300, 600, 900, 1200, 1800]
TIME_PENALTY_WEIGHT = 0.03
UNCERTAINTY_PENALTY_WEIGHT = 0.02


def _fmt_number(value: float) -> str:
    return f"{float(value):.6g}"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def strict_paper_rows() -> list[dict[str, str]]:
    """Return strict measured paper-style rows with mean delta_h/peak_area fields."""
    return usable_rows(prepare_rows(), PAPER_STYLE_TARGETS)


def candidate_time_rows(
    T1_C: float,
    T2_C: float,
    time_grid_s: Iterable[float] = EXPERIMENT_TIMES_S,
) -> list[dict[str, str]]:
    """Generate two-step candidates with fixed T1/T2 and variable t1/t2."""
    rows: list[dict[str, str]] = []
    for t1_s in time_grid_s:
        for t2_s in time_grid_s:
            t1 = float(t1_s)
            t2 = float(t2_s)
            if t1 < 10.0 or t2 < 10.0 or t1 > 1800.0 or t2 > 1800.0:
                raise ValueError("conditional inverse time grid must stay within 10-1800 s")
            rows.append({
                "mode": "two_step",
                "T1_C": _fmt_number(T1_C),
                "T2_C": _fmt_number(T2_C),
                "t1_s": _fmt_number(t1),
                "t2_s": _fmt_number(t2),
                "total_anneal_time_s": _fmt_number(t1 + t2),
            })
    return rows


def feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([featurize(row) for row in rows], dtype=float)


def target_matrix(rows: list[dict[str, str]], targets: list[str] = PAPER_STYLE_TARGETS) -> np.ndarray:
    return np.array([[float(row[target]) for target in targets] for row in rows], dtype=float)


def fit_strict_paper_model(rows: list[dict[str, str]]) -> KernelRegressor:
    if len(rows) < 3:
        raise ValueError("conditional time inverse needs at least three strict paper-style rows")
    return KernelRegressor.fit(feature_matrix(rows), target_matrix(rows), bandwidth=1.2)


def prediction_dict(values: np.ndarray, targets: list[str] = PAPER_STYLE_TARGETS) -> dict[str, float]:
    return {target: float(values[idx]) for idx, target in enumerate(targets)}


def target_loss(pred: dict[str, float], target: dict[str, float]) -> float:
    terms = []
    for name in PAPER_STYLE_TARGETS:
        if name not in pred or name not in target:
            raise KeyError(f"missing paper-style target {name}")
        terms.append(abs((float(pred[name]) - float(target[name])) / TARGET_SCALES[name]))
    return float(np.mean(terms))


def conditional_loss(
    target_loss_value: float,
    total_time_s: float,
    uncertainty_values: np.ndarray,
) -> tuple[float, dict[str, float]]:
    uncertainty_norm = float(np.mean(np.asarray(uncertainty_values, dtype=float) / np.array([TARGET_SCALES[name] for name in PAPER_STYLE_TARGETS])))
    components = {
        "target_loss": float(target_loss_value),
        "time_penalty": float(TIME_PENALTY_WEIGHT * float(total_time_s) / 3600.0),
        "uncertainty_penalty": float(UNCERTAINTY_PENALTY_WEIGHT * uncertainty_norm),
    }
    return float(sum(components.values())), components


def rank_time_candidates(
    rows: list[dict[str, str]],
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    target: dict[str, float],
    top_k: int = 10,
) -> tuple[list[dict[str, object]], np.ndarray]:
    losses = np.zeros(len(rows), dtype=float)
    components_by_row: list[dict[str, float]] = []
    target_losses = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        pred = prediction_dict(predictions[idx])
        row_target_loss = target_loss(pred, target)
        total_loss, components = conditional_loss(row_target_loss, float(row["total_anneal_time_s"]), uncertainties[idx])
        losses[idx] = total_loss
        target_losses[idx] = row_target_loss
        components_by_row.append(components)

    order = np.argsort(losses)[:top_k]
    ranked: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        row: dict[str, object] = dict(rows[int(idx)])
        row["rank"] = rank
        row["loss"] = float(losses[int(idx)])
        row["target_loss"] = float(target_losses[int(idx)])
        row["mean_uncertainty"] = float(np.mean(uncertainties[int(idx)]))
        row["loss_components"] = components_by_row[int(idx)]
        for target_idx, name in enumerate(PAPER_STYLE_TARGETS):
            row[f"pred_{name}"] = float(predictions[int(idx), target_idx])
            row[f"uncertainty_{name}"] = float(uncertainties[int(idx), target_idx])
            row[f"target_{name}"] = float(target[name])
            row[f"error_{name}"] = float(predictions[int(idx), target_idx] - float(target[name]))
        ranked.append(row)
    return ranked, losses


def conditional_posterior_summary(rows: list[dict[str, str]], losses: np.ndarray, percentile: float = 5.0) -> dict[str, object]:
    losses = np.asarray(losses, dtype=float)
    if len(rows) == 0:
        return {"n_valid_candidates": 0}
    threshold = float(np.percentile(losses, percentile))
    valid_indices = np.where(losses <= threshold)[0]
    if len(valid_indices) == 0:
        valid_indices = np.array([int(np.argmin(losses))])

    out: dict[str, object] = {
        "percentile": percentile,
        "n_valid_candidates": int(len(valid_indices)),
        "loss_min": float(np.min(losses)),
        "loss_threshold": threshold,
        "loss_range": [float(np.min(losses[valid_indices])), float(np.max(losses[valid_indices]))],
        "loss_flatness_ratio": float((np.max(losses[valid_indices]) - np.min(losses[valid_indices])) / max(abs(float(np.min(losses))), 1e-12)),
    }
    for name in ["log10_t1_s", "log10_t2_s"]:
        if name == "log10_t1_s":
            data = np.array([math.log10(float(rows[int(idx)]["t1_s"])) for idx in valid_indices], dtype=float)
        else:
            data = np.array([math.log10(float(rows[int(idx)]["t2_s"])) for idx in valid_indices], dtype=float)
        out[name] = {
            "p5": float(np.percentile(data, 5)),
            "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)),
            "width": float(np.percentile(data, 95) - np.percentile(data, 5)),
        }
    return out


def interpretation_from_time_posterior(posterior: dict[str, object]) -> dict[str, object]:
    widths = {
        name: float(posterior.get(name, {}).get("width", math.inf))  # type: ignore[union-attr]
        for name in ["log10_t1_s", "log10_t2_s"]
    }
    max_width = max(widths.values())
    if max_width <= 0.5:
        status = "well_constrained"
    elif max_width <= 1.2:
        status = "partially_constrained"
    else:
        status = "weakly_or_non_identifiable"
    return {
        "status": status,
        "posterior_widths": widths,
        "reason": f"fixed T1/T2 conditional posterior max log-time width={max_width:.3g}",
    }


def search(
    T1_C: float,
    T2_C: float,
    target: dict[str, float],
    model: KernelRegressor | None = None,
    training_rows: list[dict[str, str]] | None = None,
    top_k: int = 10,
) -> dict[str, object]:
    training_rows = training_rows or strict_paper_rows()
    model = model or fit_strict_paper_model(training_rows)
    candidates = candidate_time_rows(T1_C, T2_C)
    pred, unc = model.predict(feature_matrix(candidates))
    ranked, losses = rank_time_candidates(candidates, pred, unc, target, top_k=top_k)
    posterior = conditional_posterior_summary(candidates, losses)
    return {
        "target": {name: float(target[name]) for name in PAPER_STYLE_TARGETS},
        "fixed_temperatures": {"T1_C": float(T1_C), "T2_C": float(T2_C)},
        "point_estimates": ranked,
        "results": ranked,
        "posterior": posterior,
        "interpretation": interpretation_from_time_posterior(posterior),
        "candidate_space": {
            "time_s": EXPERIMENT_TIMES_S,
            "n_candidates": len(candidates),
            "fixed_T1_C": float(T1_C),
            "fixed_T2_C": float(T2_C),
        },
        "target_set": PAPER_STYLE_TARGETS,
        "selection_policy": "fixed T1/T2; rank t1/t2 by strict paper-style target loss + small time/uncertainty penalties",
    }


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row for row in strict_paper_rows() if row.get("mode") == "two_step" and row.get("T2_C") not in {"", "nan", "NaN"}]
    if not rows:
        raise RuntimeError("no two-step strict paper-style rows available for conditional time inverse")
    row = rows[0]
    target = {name: float(row[name]) for name in PAPER_STYLE_TARGETS}
    payload = search(float(row["T1_C"]), float(row["T2_C"]), target)
    out = OUT_DIR / "conditional_time_inverse_example.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "top_result": payload["point_estimates"][0]}, indent=2))
    return out


if __name__ == "__main__":
    main()
