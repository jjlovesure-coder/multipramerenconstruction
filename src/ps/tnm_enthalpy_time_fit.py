"""TNM/KWW enthalpy-time curve fitting for PS annealing data.

This is the Figure-2-style diagnostic: fit enthalpy recovery as a function of
annealing time for fixed temperature/path groups.  It does not fit the final
heating DSC trace shape.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ps.dsc_processing import MAIN_INTEGRATION_HIGH_C, MAIN_INTEGRATION_LOW_C, ROOT, build_ps_dataset
from src.ps.train_ps_model import read_rows


OUT_DIR = ROOT / "results" / "ps" / "tnm_enthalpy_time_fit"
FIG_DIR = OUT_DIR / "figures"
REPORT_PATH = ROOT / "docs" / "ps_tnm_enthalpy_time_fit_report.md"
SUMMARY_PATH = OUT_DIR / "tnm_enthalpy_time_fit_summary.json"
BY_GROUP_PATH = OUT_DIR / "tnm_enthalpy_time_fit_by_group.csv"
BY_POINT_PATH = OUT_DIR / "tnm_enthalpy_time_fit_by_point.csv"
WINDOW_TAG = f"{int(MAIN_INTEGRATION_LOW_C)}_{int(MAIN_INTEGRATION_HIGH_C)}"
WINDOW_LABEL = f"{MAIN_INTEGRATION_LOW_C:.0f}-{MAIN_INTEGRATION_HIGH_C:.0f} C"
SINGLE_FIG = FIG_DIR / f"single_step_enthalpy_time_fits_{WINDOW_TAG}.png"
TWOSTEP_FIG = FIG_DIR / f"two_step_enthalpy_time_fits_{WINDOW_TAG}.png"
UPJUMP_80_90_FIG = FIG_DIR / f"upjump_80_90_enthalpy_time_{WINDOW_TAG}.png"
ERROR_FIG = FIG_DIR / f"enthalpy_time_fit_errors_{WINDOW_TAG}.png"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_number(value: object) -> str:
    if not finite(value):
        return ""
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def predict_kww(time_s: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    tau = max(float(fit["tau_s"]), 1e-9)
    beta = min(max(float(fit["beta"]), 0.05), 1.0)
    progress = 1.0 - np.exp(-((np.maximum(time, 1e-9) / tau) ** beta))
    return float(fit["offset"]) + float(fit["amplitude"]) * progress


def fit_kww_enthalpy_curve(time_s: Iterable[float], enthalpy_j_g: Iterable[float]) -> dict[str, float]:
    """Grid-fit y = offset + amplitude * (1 - exp(-(t/tau)^beta))."""
    time = np.asarray(list(time_s), dtype=float)
    y = np.asarray(list(enthalpy_j_g), dtype=float)
    order = np.argsort(time)
    time = time[order]
    y = y[order]
    if len(time) < 3:
        raise ValueError("At least three time points are required for an enthalpy-time fit.")
    tau_grid = np.logspace(math.log10(max(min(time) / 3.0, 1.0)), math.log10(max(max(time) * 3.0, 3.0)), 80)
    beta_grid = np.linspace(0.25, 0.95, 29)
    best: dict[str, float] | None = None
    for tau in tau_grid:
        for beta in beta_grid:
            progress = 1.0 - np.exp(-((np.maximum(time, 1e-9) / tau) ** beta))
            x = np.column_stack([np.ones_like(progress), progress])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ coef
            residual = pred - y
            rmse = float(np.sqrt(np.mean(residual * residual)))
            if best is None or rmse < best["rmse"]:
                best = {
                    "offset": float(coef[0]),
                    "amplitude": float(coef[1]),
                    "tau_s": float(tau),
                    "beta": float(beta),
                    "rmse": rmse,
                    "mae": float(np.mean(np.abs(residual))),
                    "n_points": float(len(time)),
                }
    if best is None:
        raise RuntimeError("KWW enthalpy-time fit failed.")
    return best


def group_enthalpy_time_rows(rows: list[dict[str, str]], min_points: int = 3) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    metadata: dict[str, dict[str, object]] = {}
    for row in rows:
        if not finite(row.get("delta_h_total_J_g")):
            continue
        mode = str(row.get("mode", ""))
        if mode == "single_step" and finite(row.get("t1_s")):
            key = f"single_step|T={fmt_number(row.get('T1_C'))}"
            time = float(row["t1_s"])
            metadata[key] = {"mode": mode, "T1_C": row.get("T1_C", ""), "T2_C": "", "t1_s_fixed": "", "time_axis": "t1_s"}
        elif mode == "two_step" and finite(row.get("t2_s")):
            key = f"two_step|T1={fmt_number(row.get('T1_C'))}|t1={fmt_number(row.get('t1_s'))}|T2={fmt_number(row.get('T2_C'))}"
            time = float(row["t2_s"])
            metadata[key] = {
                "mode": mode,
                "T1_C": row.get("T1_C", ""),
                "T2_C": row.get("T2_C", ""),
                "t1_s_fixed": row.get("t1_s", ""),
                "time_axis": "t2_s",
            }
        else:
            continue
        if time <= 0.0:
            continue
        copy = dict(row)
        copy["_fit_time_s"] = str(time)
        grouped.setdefault(key, []).append(copy)

    out: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        by_time: dict[float, list[float]] = {}
        for row in values:
            by_time.setdefault(float(row["_fit_time_s"]), []).append(float(row["delta_h_total_J_g"]))
        if len(by_time) < min_points:
            continue
        times = np.array(sorted(by_time), dtype=float)
        enthalpy = np.array([float(np.mean(by_time[t])) for t in times], dtype=float)
        out.append({
            "group_key": key,
            **metadata[key],
            "time_s": times,
            "delta_h_total_J_g": enthalpy,
            "n_raw_points": len(values),
            "n_time_points": len(times),
        })
    return out


def fit_groups(groups: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    group_records: list[dict[str, object]] = []
    point_records: list[dict[str, object]] = []
    for group in groups:
        times = np.asarray(group["time_s"], dtype=float)
        y = np.asarray(group["delta_h_total_J_g"], dtype=float)
        fit = fit_kww_enthalpy_curve(times, y)
        pred = predict_kww(times, fit)
        group_record = {
            "group_key": group["group_key"],
            "mode": group["mode"],
            "T1_C": group["T1_C"],
            "T2_C": group["T2_C"],
            "t1_s_fixed": group["t1_s_fixed"],
            "time_axis": group["time_axis"],
            "n_raw_points": group["n_raw_points"],
            "n_time_points": group["n_time_points"],
            "offset_J_g": fit["offset"],
            "amplitude_J_g": fit["amplitude"],
            "tau_s": fit["tau_s"],
            "beta": fit["beta"],
            "rmse_J_g": fit["rmse"],
            "mae_J_g": fit["mae"],
        }
        group_records.append(group_record)
        for t, actual, predicted in zip(times, y, pred):
            point_records.append({
                "group_key": group["group_key"],
                "mode": group["mode"],
                "time_axis": group["time_axis"],
                "time_s": float(t),
                "actual_delta_h_total_J_g": float(actual),
                "pred_delta_h_total_J_g": float(predicted),
                "error_J_g": float(predicted - actual),
            })
    return group_records, point_records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_group_set(groups: list[dict[str, object]], group_records: list[dict[str, object]], path: Path, title: str, max_groups: int = 9) -> None:
    selected = groups[:max_groups]
    if not selected:
        return
    record_by_key = {str(row["group_key"]): row for row in group_records}
    ncols = 3
    nrows = int(math.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False, constrained_layout=True)
    for ax, group in zip(axes.ravel(), selected):
        key = str(group["group_key"])
        rec = record_by_key[key]
        times = np.asarray(group["time_s"], dtype=float)
        y = np.asarray(group["delta_h_total_J_g"], dtype=float)
        dense = np.logspace(math.log10(max(times.min(), 1e-9)), math.log10(times.max()), 160)
        fit = {
            "offset": float(rec["offset_J_g"]),
            "amplitude": float(rec["amplitude_J_g"]),
            "tau_s": float(rec["tau_s"]),
            "beta": float(rec["beta"]),
        }
        ax.scatter(times, y, label="actual", zorder=3)
        ax.plot(dense, predict_kww(dense, fit), label="TNM/KWW fit", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_title(f"{key}\nRMSE={float(rec['rmse_J_g']):.3f} J/g", fontsize=8)
        ax.set_xlabel(f"{group['time_axis']} (s)")
        ax.set_ylabel(f"delta_h_total ({WINDOW_LABEL}, J/g)")
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"{title}; integration window {WINDOW_LABEL}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_upjump_80_90(groups: list[dict[str, object]], path: Path) -> None:
    selected = [
        group
        for group in groups
        if group["mode"] == "two_step"
        and finite(group.get("T1_C"))
        and finite(group.get("T2_C"))
        and abs(float(group["T1_C"]) - 80.0) < 1.0
        and abs(float(group["T2_C"]) - 90.0) < 1.0
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for group in selected:
        times = np.asarray(group["time_s"], dtype=float)
        y = np.asarray(group["delta_h_total_J_g"], dtype=float)
        order = np.argsort(times)
        times = times[order]
        y = y[order]
        label = f"t1={fmt_number(group.get('t1_s_fixed'))} s"
        ax.plot(times, y, marker="o", linewidth=1.6, label=label)
        for t, value in zip(times[-3:], y[-3:]):
            ax.annotate(f"{value:.2f}", (t, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("t2_s (s)")
    ax.set_ylabel(f"delta_h_total ({WINDOW_LABEL}, J/g)")
    ax.set_title(f"80->90 C up-jump check; integration window {WINDOW_LABEL}")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_figures(groups: list[dict[str, object]], group_records: list[dict[str, object]], point_records: list[dict[str, object]]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    single_groups = [g for g in groups if g["mode"] == "single_step"]
    two_groups = [g for g in groups if g["mode"] == "two_step"]
    _plot_group_set(single_groups, group_records, SINGLE_FIG, "Single-step enthalpy recovery vs annealing time")
    _plot_group_set(two_groups, group_records, TWOSTEP_FIG, "Two-step enthalpy recovery vs second-step time")
    _plot_upjump_80_90(two_groups, UPJUMP_80_90_FIG)

    errors = np.array([float(row["error_J_g"]) for row in point_records], dtype=float)
    actual = np.array([float(row["actual_delta_h_total_J_g"]) for row in point_records], dtype=float)
    pred = np.array([float(row["pred_delta_h_total_J_g"]) for row in point_records], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].scatter(actual, pred, alpha=0.75)
    low = float(min(actual.min(), pred.min()))
    high = float(max(actual.max(), pred.max()))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set_xlabel(f"Actual delta_h_total ({WINDOW_LABEL}, J/g)")
    axes[0].set_ylabel(f"Predicted delta_h_total ({WINDOW_LABEL}, J/g)")
    axes[0].set_title("Actual vs fitted")
    axes[1].hist(errors, bins=18, color="#4c78a8", alpha=0.85)
    axes[1].set_xlabel("Fit error (J/g)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.suptitle(f"Enthalpy-time TNM/KWW fit errors; integration window {WINDOW_LABEL}")
    fig.savefig(ERROR_FIG, dpi=180)
    plt.close(fig)
    return {
        "single_step": str(SINGLE_FIG.relative_to(ROOT)),
        "two_step": str(TWOSTEP_FIG.relative_to(ROOT)),
        "upjump_80_90": str(UPJUMP_80_90_FIG.relative_to(ROOT)),
        "errors": str(ERROR_FIG.relative_to(ROOT)),
    }


def write_report(summary: dict[str, object]) -> None:
    figures = summary["figures"]
    text = f"""# PS TNM/KWW Enthalpy-Time Fit

## Scope

This report fits the Figure-2-style relation:

```text
annealing time -> delta_h_total_J_g, integrated over {WINDOW_LABEL}
```

It does **not** fit the final heating DSC heat-flow trace. Single-step groups are
fit as `delta_h(t1)` at fixed `T`. Two-step groups are fit as `delta_h(t2)` at
fixed `T1, t1, T2`.

## Summary

```json
{json.dumps(summary["metrics"], indent=2)}
```

## Figures

![Single-step enthalpy-time fits]({(ROOT / figures["single_step"]).as_posix()})

![Two-step enthalpy-time fits]({(ROOT / figures["two_step"]).as_posix()})

![80 to 90 C up-jump focused check]({(ROOT / figures["upjump_80_90"]).as_posix()})

![Fit errors]({(ROOT / figures["errors"]).as_posix()})

## Outputs

- Group fits: `{BY_GROUP_PATH.relative_to(ROOT)}`
- Point residuals: `{BY_POINT_PATH.relative_to(ROOT)}`
- Summary: `{SUMMARY_PATH.relative_to(ROOT)}`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def run() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(build_ps_dataset())
    groups = group_enthalpy_time_rows(rows)
    group_records, point_records = fit_groups(groups)
    write_csv(BY_GROUP_PATH, group_records)
    write_csv(BY_POINT_PATH, point_records)
    figures = write_figures(groups, group_records, point_records)
    metrics = {
        "n_groups": len(group_records),
        "n_points": len(point_records),
        "group_rmse_J_g_mean": float(np.mean([float(row["rmse_J_g"]) for row in group_records])) if group_records else math.nan,
        "group_rmse_J_g_median": float(np.median([float(row["rmse_J_g"]) for row in group_records])) if group_records else math.nan,
        "point_mae_J_g": float(np.mean([abs(float(row["error_J_g"])) for row in point_records])) if point_records else math.nan,
        "single_step_groups": sum(1 for row in group_records if row["mode"] == "single_step"),
        "two_step_groups": sum(1 for row in group_records if row["mode"] == "two_step"),
    }
    summary = {
        "status": "complete",
        "model_role": "Figure-2-style enthalpy recovery versus annealing time fit",
        "equation": "delta_h(t) = offset + amplitude * (1 - exp(-(t/tau)^beta))",
        "integration_window_C": [MAIN_INTEGRATION_LOW_C, MAIN_INTEGRATION_HIGH_C],
        "metrics": metrics,
        "figures": figures,
        "outputs": {
            "by_group": str(BY_GROUP_PATH.relative_to(ROOT)),
            "by_point": str(BY_POINT_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))
    return SUMMARY_PATH


if __name__ == "__main__":
    run()
