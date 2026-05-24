"""Curve-level TNM forward check for PS final-heating scans.

This module is deliberately a diagnostic forward fitter, not a claim that a
single Gaussian is the final physical TNM solution.  It tests whether compact
TNM/fictive-temperature coordinates can predict the observed 70-105 C recovery
curve shape well enough before inverse reconstruction is trusted.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ps.dsc_processing import (
    DEFAULT_HEATING_RATE_C_MIN,
    MAIN_INTEGRATION_HIGH_C,
    MAIN_INTEGRATION_LOW_C,
    MAIN_PEAK_HIGH_C,
    MAIN_PEAK_LOW_C,
    RAW_FILES,
    ROOT,
    all_recovery_scan_indices,
    baseline_scan,
    corrected_signal,
    detrend_window,
    infer_condition,
    integral_uW_C_to_J_g,
    read_protocol,
    read_signal,
    scan_for_segment,
    segment_bounds,
)
from src.ps.tnm_calibrated_model import TnmParameters, tnm_fictive_temperature_features


OUT_DIR = ROOT / "results" / "ps" / "tnm_curve_fit"
REPORT_PATH = ROOT / "docs" / "ps_tnm_curve_fit_forward_report.md"
BY_SAMPLE_PATH = OUT_DIR / "tnm_curve_fit_by_sample.csv"
SUMMARY_PATH = OUT_DIR / "tnm_curve_fit_summary.json"
FIG_DIR = OUT_DIR / "figures"
AREA_TP_FIG = FIG_DIR / "tnm_curve_fit_actual_vs_predicted.png"
ERROR_FIG = FIG_DIR / "tnm_curve_fit_error_distributions.png"
OVERLAY_FIG = FIG_DIR / "tnm_curve_fit_representative_overlays.png"
DEFAULT_PARAMS = TnmParameters(
    activation_energy_kj_mol=300.0,
    beta=0.75,
    log10_tau_ref_s=2.0,
    nonlinearity_x=0.90,
    initial_fictive_temperature_k=393.15,
)
CURVE_TARGETS = ["area_rel", "tp_c", "height_uW", "log_width_c"]


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: object, default: float = 0.0) -> float:
    return float(value) if finite(value) else default


def _condition_features(row: dict[str, object], params: TnmParameters = DEFAULT_PARAMS) -> np.ndarray:
    mode = str(row.get("mode", "single_step"))
    t1 = max(_as_float(row.get("t1_s")), 1e-9)
    t2 = max(_as_float(row.get("t2_s")), 0.0) if mode == "two_step" else 0.0
    total = max(t1 + t2, 1e-9)
    t1_c = _as_float(row.get("T1_C"))
    t2_c = _as_float(row.get("T2_C"), t1_c) if mode == "two_step" else t1_c
    tf = tnm_fictive_temperature_features(row, params)
    return np.array(
        [
            1.0,
            t1_c / 100.0,
            t2_c / 100.0,
            math.log10(t1) / 4.0,
            math.log10(max(t2, 1e-9)) / 4.0 if mode == "two_step" else 0.0,
            math.log10(total) / 4.0,
            float(tf["tnm_tf_normalized_recovery"]),
            float(tf["tnm_tf_path_excess"]),
        ],
        dtype=float,
    )


def observed_curve_parameters(curve: pd.DataFrame, heating_rate_c_min: float) -> dict[str, float]:
    temp = curve["Temp_C"].to_numpy(dtype=float)
    y = curve["DSC_detrended_uW"].to_numpy(dtype=float)
    peak_window = curve[(curve["Temp_C"] >= MAIN_PEAK_LOW_C) & (curve["Temp_C"] <= MAIN_PEAK_HIGH_C)]
    if len(peak_window) < 5:
        return {"area_rel": math.nan, "area_J_g": math.nan, "tp_c": math.nan, "height_uW": math.nan, "width_c": math.nan, "log_width_c": math.nan}
    ptemp = peak_window["Temp_C"].to_numpy(dtype=float)
    py = peak_window["DSC_detrended_uW"].to_numpy(dtype=float)
    peak_idx = int(np.argmax(np.abs(py)))
    tp = float(ptemp[peak_idx])
    height = float(py[peak_idx])
    area = float(np.trapezoid(y, temp))
    weights = np.abs(py)
    if float(np.sum(weights)) <= 1e-12:
        width = 8.0
    else:
        width = float(np.sqrt(np.average((ptemp - tp) ** 2, weights=weights)))
    width = min(max(width, 3.0), 25.0)
    return {
        "area_rel": area,
        "area_J_g": integral_uW_C_to_J_g(area, heating_rate_c_min),
        "tp_c": tp,
        "height_uW": height,
        "width_c": width,
        "log_width_c": math.log(width),
    }


def predicted_curve(temp_c: np.ndarray, area_rel: float, tp_c: float, height_uW: float, log_width_c: float) -> np.ndarray:
    width = min(max(math.exp(float(log_width_c)), 3.0), 25.0)
    peak = np.asarray(height_uW * np.exp(-0.5 * ((temp_c - tp_c) / width) ** 2), dtype=float)
    predicted_area = float(np.trapezoid(peak, temp_c))
    if abs(predicted_area) > 1e-12 and math.isfinite(area_rel):
        peak *= float(area_rel) / predicted_area
    return peak


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 0.05) -> np.ndarray:
    xtx = x.T @ x
    penalty = ridge * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def collect_curve_rows() -> list[dict[str, object]]:
    ref = baseline_scan(RAW_FILES["ref"])
    rows: list[dict[str, object]] = []
    for file_key, path in RAW_FILES.items():
        if file_key in {"ref", "empty"} or not path.exists():
            continue
        segments = read_protocol(path)
        bounds = segment_bounds(segments)
        signal = read_signal(path)
        previous_scan = -1
        cycle_id = 0
        for heat_idx in all_recovery_scan_indices(segments):
            cycle_segments = segments[previous_scan + 1: heat_idx + 1]
            previous_scan = heat_idx
            heating_rate = float(segments[heat_idx].rate_c_min)
            if abs(heating_rate - DEFAULT_HEATING_RATE_C_MIN) > 2.0:
                continue
            condition = infer_condition(file_key, cycle_segments)
            if condition["mode"] == "unknown":
                continue
            scan = scan_for_segment(signal, segments[heat_idx], bounds[heat_idx])
            if len(scan) < 20:
                continue
            corrected = corrected_signal(scan, ref)
            curve = detrend_window(corrected, MAIN_INTEGRATION_LOW_C, MAIN_INTEGRATION_HIGH_C).dropna(subset=["DSC_detrended_uW"])
            if len(curve) < 10:
                continue
            observed = observed_curve_parameters(curve, heating_rate)
            if not all(finite(observed[name]) for name in CURVE_TARGETS):
                continue
            cycle_id += 1
            total_time = float(condition["t1_s"])
            if condition["mode"] == "two_step":
                total_time += float(condition["t2_s"])
            rows.append(
                {
                    "sample_id": f"{file_key}_{cycle_id:03d}",
                    "source_file": path.name,
                    "mode": condition["mode"],
                    "T1_C": condition["T1_C"],
                    "t1_s": condition["t1_s"],
                    "T2_C": condition["T2_C"],
                    "t2_s": condition["t2_s"],
                    "total_anneal_time_s": total_time,
                    "heating_rate_C_min": heating_rate,
                    "curve_temp_C": curve["Temp_C"].to_numpy(dtype=float),
                    "curve_y_uW": curve["DSC_detrended_uW"].to_numpy(dtype=float),
                    **observed,
                }
            )
    return rows


def leave_one_out_curve_fit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(rows) < 4:
        raise ValueError("TNM curve fit needs at least four valid curves.")
    x_all = np.vstack([_condition_features(row) for row in rows])
    y_all = np.array([[float(row[target]) for target in CURVE_TARGETS] for row in rows], dtype=float)
    records: list[dict[str, object]] = []
    for test_idx, row in enumerate(rows):
        train_idx = np.array([idx for idx in range(len(rows)) if idx != test_idx], dtype=int)
        coef = fit_ridge(x_all[train_idx], y_all[train_idx])
        pred_targets = x_all[test_idx] @ coef
        temp = np.asarray(row["curve_temp_C"], dtype=float)
        actual = np.asarray(row["curve_y_uW"], dtype=float)
        pred_curve = predicted_curve(temp, *[float(v) for v in pred_targets])
        residual = pred_curve - actual
        area_error_rel = float(np.trapezoid(residual, temp))
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_file": row["source_file"],
                "mode": row["mode"],
                "T1_C": row["T1_C"],
                "t1_s": row["t1_s"],
                "T2_C": row["T2_C"],
                "t2_s": row["t2_s"],
                "total_anneal_time_s": row["total_anneal_time_s"],
                "curve_rmse_uW": float(np.sqrt(np.mean(residual * residual))),
                "curve_mae_uW": float(np.mean(np.abs(residual))),
                "area_error_J_g": integral_uW_C_to_J_g(area_error_rel, float(row["heating_rate_C_min"])),
                "tp_error_C": float(pred_targets[1] - float(row["tp_c"])),
                "peak_height_error_uW": float(pred_targets[2] - float(row["height_uW"])),
                "actual_area_J_g": row["area_J_g"],
                "pred_area_J_g": integral_uW_C_to_J_g(float(pred_targets[0]), float(row["heating_rate_C_min"])),
                "actual_tp_C": row["tp_c"],
                "pred_tp_C": float(pred_targets[1]),
                "actual_peak_height_uW": row["height_uW"],
                "pred_peak_height_uW": float(pred_targets[2]),
                "actual_width_C": row["width_c"],
                "pred_width_C": float(math.exp(pred_targets[3])),
                "n_points_fit": int(len(temp)),
                "_curve_temp_C": temp,
                "_curve_actual_uW": actual,
                "_curve_pred_uW": pred_curve,
            }
        )
    return records


def _mean(records: Iterable[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in records if finite(row.get(key))]
    return float(np.mean(values)) if values else math.nan


def _median(records: Iterable[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in records if finite(row.get(key))]
    return float(np.median(values)) if values else math.nan


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [name for name in rows[0].keys() if not name.startswith("_")]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_figures(records: list[dict[str, object]]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    actual_area = np.array([float(r["actual_area_J_g"]) for r in records], dtype=float)
    pred_area = np.array([float(r["pred_area_J_g"]) for r in records], dtype=float)
    actual_tp = np.array([float(r["actual_tp_C"]) for r in records], dtype=float)
    pred_tp = np.array([float(r["pred_tp_C"]) for r in records], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].scatter(actual_area, pred_area, s=22, alpha=0.75)
    low = float(min(actual_area.min(), pred_area.min()))
    high = float(max(actual_area.max(), pred_area.max()))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set_xlabel("Actual area (J/g)")
    axes[0].set_ylabel("Predicted area (J/g)")
    axes[0].set_title("Area")

    axes[1].scatter(actual_tp, pred_tp, s=22, alpha=0.75, color="#4c78a8")
    low = float(min(actual_tp.min(), pred_tp.min()))
    high = float(max(actual_tp.max(), pred_tp.max()))
    axes[1].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1].set_xlabel("Actual Tp (C)")
    axes[1].set_ylabel("Predicted Tp (C)")
    axes[1].set_title("Peak temperature")
    fig.suptitle("TNM Curve Fit: Actual vs Predicted")
    fig.savefig(AREA_TP_FIG, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].hist([float(r["curve_rmse_uW"]) for r in records], bins=18, color="#4c78a8", alpha=0.85)
    axes[0].set_xlabel("Curve RMSE (uW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Curve RMSE")
    axes[1].hist([float(r["area_error_J_g"]) for r in records], bins=18, color="#f58518", alpha=0.85)
    axes[1].set_xlabel("Area error (J/g)")
    axes[1].set_title("Area error")
    axes[2].hist([float(r["peak_height_error_uW"]) for r in records], bins=18, color="#54a24b", alpha=0.85)
    axes[2].set_xlabel("Peak height error (uW)")
    axes[2].set_title("Peak height error")
    fig.suptitle("TNM Curve Fit: Error Distributions")
    fig.savefig(ERROR_FIG, dpi=180)
    plt.close(fig)

    ordered = sorted(records, key=lambda r: float(r["curve_rmse_uW"]))
    selected = ordered[:2] + ordered[len(ordered) // 2: len(ordered) // 2 + 2] + ordered[-2:]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, record in zip(axes.ravel(), selected):
        temp = np.asarray(record["_curve_temp_C"], dtype=float)
        actual = np.asarray(record["_curve_actual_uW"], dtype=float)
        pred = np.asarray(record["_curve_pred_uW"], dtype=float)
        ax.plot(temp, actual, label="actual", linewidth=1.6)
        ax.plot(temp, pred, label="predicted", linewidth=1.4)
        title = f"{record['sample_id']} RMSE={float(record['curve_rmse_uW']):.1f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("T (C)")
        ax.set_ylabel("DSC detrended (uW)")
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle("TNM Curve Fit: Representative Curve Overlays")
    fig.savefig(OVERLAY_FIG, dpi=180)
    plt.close(fig)

    return {
        "actual_vs_predicted": str(AREA_TP_FIG.relative_to(ROOT)),
        "error_distributions": str(ERROR_FIG.relative_to(ROOT)),
        "representative_overlays": str(OVERLAY_FIG.relative_to(ROOT)),
    }


def write_report(summary: dict[str, object]) -> None:
    text = f"""# PS TNM Curve-Level Forward Fit

## Scope

This is the first curve-level forward diagnostic after narrowing the DSC target
window to `{MAIN_INTEGRATION_LOW_C:.0f}-{MAIN_INTEGRATION_HIGH_C:.0f} C`.  It tests whether compact TNM/fictive-temperature
coordinates can predict the full final-heating recovery curve shape before
inverse reconstruction is trusted.

This is **not yet** a full material-parameter TNM solver. It is a minimal
curve-level check that predicts peak area, peak position, peak height, and width,
then reconstructs the full curve and reports residuals.

## Summary Metrics

```json
{json.dumps(summary["metrics"], indent=2)}
```

## Figures

![Actual vs predicted]({(ROOT / summary["figures"]["actual_vs_predicted"]).as_posix()})

![Error distributions]({(ROOT / summary["figures"]["error_distributions"]).as_posix()})

![Representative overlays]({(ROOT / summary["figures"]["representative_overlays"]).as_posix()})

## Interpretation

If curve RMSE/MAE remain large relative to the observed peak heights, the inverse
posterior should be interpreted as weak even when top candidates look plausible.
The next step after this diagnostic is a stricter TNM solver that fits `Tf(t)`
and `Delta h(t)` directly.

## Outputs

- Per-sample fit errors: `{BY_SAMPLE_PATH.relative_to(ROOT)}`
- Summary: `{SUMMARY_PATH.relative_to(ROOT)}`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def run() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_curve_rows()
    records = leave_one_out_curve_fit(rows)
    write_csv(BY_SAMPLE_PATH, records)
    figures = write_figures(records)
    metrics = {
        "n_curves": len(records),
        "curve_rmse_uW_mean": _mean(records, "curve_rmse_uW"),
        "curve_mae_uW_mean": _mean(records, "curve_mae_uW"),
        "area_error_J_g_mae": _mean([{"v": abs(float(r["area_error_J_g"]))} for r in records], "v"),
        "tp_error_C_mae": _mean([{"v": abs(float(r["tp_error_C"]))} for r in records], "v"),
        "peak_height_error_uW_mae": _mean([{"v": abs(float(r["peak_height_error_uW"]))} for r in records], "v"),
        "curve_rmse_uW_median": _median(records, "curve_rmse_uW"),
        "curve_mae_uW_median": _median(records, "curve_mae_uW"),
    }
    summary = {
        "status": "complete",
        "model_role": "curve-level TNM-inspired forward diagnostic, not a full material TNM solver",
        "temperature_window_C": [MAIN_INTEGRATION_LOW_C, MAIN_INTEGRATION_HIGH_C],
        "targets": CURVE_TARGETS,
        "tnm_parameters": {
            "activation_energy_kj_mol": DEFAULT_PARAMS.activation_energy_kj_mol,
            "beta": DEFAULT_PARAMS.beta,
            "log10_tau_ref_s": DEFAULT_PARAMS.log10_tau_ref_s,
            "nonlinearity_x": DEFAULT_PARAMS.nonlinearity_x,
            "initial_fictive_temperature_k": DEFAULT_PARAMS.initial_fictive_temperature_k,
        },
        "metrics": metrics,
        "figures": figures,
        "outputs": {
            "by_sample": str(BY_SAMPLE_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))
    return SUMMARY_PATH


if __name__ == "__main__":
    run()
