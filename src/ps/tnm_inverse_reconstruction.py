"""Inverse reconstruction using the calibrated PS TNM forward model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.ps.dsc_processing import build_ps_dataset
from src.ps.inverse_design import posterior_summary
from src.ps.tnm_calibrated_model import MODEL_PATH, OUT_DIR, TARGETS, TnmParameters, design_matrix, predict_with_head, target_matrix
from src.ps.train_ps_model import read_rows


ROOT = Path(__file__).resolve().parents[2]
INVERSE_PATH = OUT_DIR / "tnm_inverse_reconstruction_by_sample.csv"
INVERSE_SUMMARY_PATH = OUT_DIR / "tnm_inverse_reconstruction_summary.json"

TARGET_SCALES = {
    "delta_h_total_J_g": 0.15,
    "peak_area_J_g": 0.10,
    "recovery_index": 0.15,
    "path_dependence_index": 0.12,
}


def load_tnm_payload(path: Path = MODEL_PATH) -> tuple[TnmParameters, dict[str, np.ndarray], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params_json = payload["selected_parameters"]
    params = TnmParameters(**{name: params_json[name] for name in TnmParameters.__dataclass_fields__})
    head = {key: np.array(value, dtype=float) for key, value in payload["linear_head"].items()}
    return params, head, payload


def candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    temps = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    times = [10, 30, 60, 100, 300, 600, 900, 1200, 1800]
    for t1 in temps:
        for t2 in temps:
            for t1_s in times:
                for t2_s in times:
                    rows.append({
                        "mode": "two_step",
                        "T1_C": str(t1),
                        "t1_s": str(t1_s),
                        "T2_C": str(t2),
                        "t2_s": str(t2_s),
                        "total_anneal_time_s": str(t1_s + t2_s),
                    })
    return rows


def _target_loss(pred: np.ndarray, target: np.ndarray) -> float:
    loss = 0.0
    for idx, name in enumerate(TARGETS):
        loss += abs(float(pred[idx] - target[idx])) / TARGET_SCALES[name]
    state_consistency = abs(float(pred[TARGETS.index("delta_h_total_J_g")] - target[TARGETS.index("delta_h_total_J_g")])) / TARGET_SCALES["delta_h_total_J_g"]
    loss += 0.5 * state_consistency
    return loss / len(TARGETS)


def posterior_from_losses(rows: list[dict[str, str]], losses: np.ndarray, percentile: float = 5.0) -> dict[str, object]:
    return posterior_summary(rows, losses, percentile=percentile)


def interval_confidence(width: float, high_threshold: float, medium_threshold: float) -> str:
    """Classify posterior interval confidence from interval width."""
    if width <= high_threshold:
        return "high"
    if width <= medium_threshold:
        return "medium"
    return "low"


def overall_confidence(labels: list[str]) -> str:
    if any(label == "low" for label in labels):
        return "low"
    if any(label == "medium" for label in labels):
        return "medium"
    return "high"


def reconstruct() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, head, payload = load_tnm_payload()
    candidates = candidate_rows()
    candidate_pred = predict_with_head(design_matrix(candidates, params), head)
    rows = [row for row in read_rows(build_ps_dataset()) if row["mode"] == "two_step"]
    y = target_matrix(rows)

    records = []
    posterior_coverages = []
    for row, target in zip(rows, y):
        losses = np.array([_target_loss(pred, target) for pred in candidate_pred], dtype=float)
        idx = int(np.argmin(losses))
        best = candidates[idx]
        posterior = posterior_from_losses(candidates, losses)
        actual_t1 = float(row["T1_C"])
        actual_t2 = float(row["T2_C"])
        actual_time1 = float(row["t1_s"])
        actual_time2 = float(row["t2_s"])
        pred_t1 = float(best["T1_C"])
        pred_t2 = float(best["T2_C"])
        pred_time1 = float(best["t1_s"])
        pred_time2 = float(best["t2_s"])
        confidence_T1 = interval_confidence(float(posterior["T1_C"]["width"]), 5.0, 15.0)
        confidence_T2 = interval_confidence(float(posterior["T2_C"]["width"]), 5.0, 15.0)
        confidence_t1 = interval_confidence(float(posterior["log10_t1_s"]["width"]), math.log10(3.0), 1.0)
        confidence_t2 = interval_confidence(float(posterior["log10_t2_s"]["width"]), math.log10(3.0), 1.0)
        record = {
            "sample_id": row.get("sample_id", ""),
            "source_file": row.get("source_file", ""),
            "actual_T1_C": actual_t1,
            "pred_T1_C": pred_t1,
            "abs_error_T1_C": abs(pred_t1 - actual_t1),
            "actual_t1_s": actual_time1,
            "pred_t1_s": pred_time1,
            "abs_log_error_t1": abs(math.log10(pred_time1) - math.log10(actual_time1)),
            "actual_T2_C": actual_t2,
            "pred_T2_C": pred_t2,
            "abs_error_T2_C": abs(pred_t2 - actual_t2),
            "actual_t2_s": actual_time2,
            "pred_t2_s": pred_time2,
            "abs_log_error_t2": abs(math.log10(pred_time2) - math.log10(actual_time2)),
            "target_loss": float(losses[idx]),
            "loss_flatness_ratio": float(posterior["loss_flatness_ratio"]),
            "posterior_contains_actual_T1": bool(posterior["T1_C"]["p5"] <= actual_t1 <= posterior["T1_C"]["p95"]),
            "posterior_contains_actual_T2": bool(posterior["T2_C"]["p5"] <= actual_t2 <= posterior["T2_C"]["p95"]),
            "posterior_contains_actual_t1": bool(posterior["log10_t1_s"]["p5"] <= math.log10(actual_time1) <= posterior["log10_t1_s"]["p95"]),
            "posterior_contains_actual_t2": bool(posterior["log10_t2_s"]["p5"] <= math.log10(actual_time2) <= posterior["log10_t2_s"]["p95"]),
            "posterior_width_T1_C": float(posterior["T1_C"]["width"]),
            "posterior_width_T2_C": float(posterior["T2_C"]["width"]),
            "posterior_width_log10_t1_s": float(posterior["log10_t1_s"]["width"]),
            "posterior_width_log10_t2_s": float(posterior["log10_t2_s"]["width"]),
            "posterior_confidence_T1": confidence_T1,
            "posterior_confidence_T2": confidence_T2,
            "posterior_confidence_t1": confidence_t1,
            "posterior_confidence_t2": confidence_t2,
            "overall_posterior_confidence": overall_confidence([confidence_T1, confidence_T2, confidence_t1, confidence_t2]),
        }
        posterior_coverages.append(record)
        for j, name in enumerate(TARGETS):
            record[f"actual_{name}"] = float(target[j])
            record[f"reconstructed_pred_{name}"] = float(candidate_pred[idx, j])
        records.append(record)

    fields = list(records[0])
    with INVERSE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "selected_parameters": payload["selected_parameters"],
        "n_two_step_samples": len(records),
        "candidate_grid_size": len(candidates),
        "mae_T1_C": float(np.mean([r["abs_error_T1_C"] for r in records])),
        "mae_T2_C": float(np.mean([r["abs_error_T2_C"] for r in records])),
        "mae_log10_t1": float(np.mean([r["abs_log_error_t1"] for r in records])),
        "mae_log10_t2": float(np.mean([r["abs_log_error_t2"] for r in records])),
        "median_target_loss": float(np.median([r["target_loss"] for r in records])),
        "posterior_coverage": {
            "T1_C": float(np.mean([r["posterior_contains_actual_T1"] for r in posterior_coverages])),
            "T2_C": float(np.mean([r["posterior_contains_actual_T2"] for r in posterior_coverages])),
            "t1_s": float(np.mean([r["posterior_contains_actual_t1"] for r in posterior_coverages])),
            "t2_s": float(np.mean([r["posterior_contains_actual_t2"] for r in posterior_coverages])),
        },
        "posterior_median_width": {
            "T1_C": float(np.median([r["posterior_width_T1_C"] for r in posterior_coverages])),
            "T2_C": float(np.median([r["posterior_width_T2_C"] for r in posterior_coverages])),
            "log10_t1_s": float(np.median([r["posterior_width_log10_t1_s"] for r in posterior_coverages])),
            "log10_t2_s": float(np.median([r["posterior_width_log10_t2_s"] for r in posterior_coverages])),
        },
        "posterior_confidence_counts": {
            "overall": {
                label: int(sum(1 for r in records if r["overall_posterior_confidence"] == label))
                for label in ("high", "medium", "low")
            },
            "T1_C": {
                label: int(sum(1 for r in records if r["posterior_confidence_T1"] == label))
                for label in ("high", "medium", "low")
            },
            "T2_C": {
                label: int(sum(1 for r in records if r["posterior_confidence_T2"] == label))
                for label in ("high", "medium", "low")
            },
            "t1_s": {
                label: int(sum(1 for r in records if r["posterior_confidence_t1"] == label))
                for label in ("high", "medium", "low")
            },
            "t2_s": {
                label: int(sum(1 for r in records if r["posterior_confidence_t2"] == label))
                for label in ("high", "medium", "low")
            },
        },
        "output": str(INVERSE_PATH.relative_to(ROOT)),
        "interpretation": (
            "This is a TNM-forward inverse search over candidate annealing conditions. "
            "It tests identifiability under the calibrated H*/S* kinetic basis; it is not a proof of unique inversion."
        ),
    }
    INVERSE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return INVERSE_PATH


if __name__ == "__main__":
    reconstruct()
