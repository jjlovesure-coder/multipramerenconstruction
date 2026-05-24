"""Plot multi-heating-rate DSC curves and Kissinger diagnostics for PS ARRT labels."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ps.calculate_arrt_from_heating_rates import ARRT_PEAK_HIGH_C, ARRT_PEAK_LOW_C, arrt_peak_temperature_c
from src.ps.dsc_processing import (
    RAW_FILES,
    ROOT,
    all_recovery_scan_indices,
    baseline_scan,
    corrected_signal,
    detrend_window,
    infer_condition,
    read_protocol,
    read_signal,
    scan_for_segment,
    segment_bounds,
)


OUT_DIR = ROOT / "results" / "ps" / "arrt_kissinger" / "figures"
CURVE_GRID = OUT_DIR / "arrt_multi_heating_rate_curves.png"
KISSINGER_GRID = OUT_DIR / "arrt_kissinger_fits.png"
HS_SCATTER = OUT_DIR / "arrt_hs_confidence_scatter.png"
SUMMARY_PATH = OUT_DIR / "arrt_heating_rate_figures_summary.json"
ARRT_RESULTS_PATH = ROOT / "results" / "ps" / "arrt_kissinger" / "arrt_kissinger_results.csv"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: object) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.6g}"


def condition_key(row: dict[str, object]) -> str:
    return "|".join(
        [
            str(row.get("mode", "")),
            fmt(row.get("T1_C", "")),
            fmt(row.get("t1_s", "")),
            fmt(row.get("T2_C", "")),
            fmt(row.get("t2_s", "")),
        ]
    )


def condition_label(key: str) -> str:
    mode, t1, time1, t2, time2 = (key.split("|") + ["", "", "", "", ""])[:5]
    if mode == "single_step":
        return f"{t1} C, {time1} s"
    return f"{t1} C {time1} s -> {t2} C {time2} s"


def read_arrt_results() -> list[dict[str, str]]:
    with ARRT_RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_curves(valid_keys: set[str]) -> list[dict[str, object]]:
    ref = baseline_scan(RAW_FILES["ref"])
    curves: list[dict[str, object]] = []
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
            condition = infer_condition(file_key, cycle_segments)
            key = condition_key(condition)
            if key not in valid_keys:
                continue
            scan = scan_for_segment(signal, segments[heat_idx], bounds[heat_idx])
            if len(scan) < 20:
                continue
            corrected = corrected_signal(scan, ref)
            curve = detrend_window(corrected, 70.0, 130.0).dropna(subset=["DSC_detrended_uW"])
            if len(curve) < 10:
                continue
            cycle_id += 1
            curves.append(
                {
                    "sample_id": f"{file_key}_{cycle_id:03d}",
                    "condition_key": key,
                    "heating_rate_C_min": float(segments[heat_idx].rate_c_min),
                    "temp_C": curve["Temp_C"].to_numpy(dtype=float),
                    "dsc_uW": curve["DSC_detrended_uW"].to_numpy(dtype=float),
                    "arrt_tp_C": arrt_peak_temperature_c(corrected),
                }
            )
    return curves


def plot_curve_grid(results: list[dict[str, str]], curves: list[dict[str, object]]) -> None:
    by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for curve in curves:
        by_key[str(curve["condition_key"])].append(curve)

    keys = [row["condition_key"] for row in results]
    ncols = 3
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for ax, key in zip(axes.ravel(), keys):
        rows = sorted(by_key.get(key, []), key=lambda r: float(r["heating_rate_C_min"]))
        rates = [float(r["heating_rate_C_min"]) for r in rows]
        for idx, row in enumerate(rows):
            color = cmap(idx / max(len(rows) - 1, 1))
            ax.plot(row["temp_C"], row["dsc_uW"], color=color, linewidth=1.35, label=f"{row['heating_rate_C_min']:g} C/min")
            if finite(row.get("arrt_tp_C")):
                ax.axvline(float(row["arrt_tp_C"]), color=color, alpha=0.35, linewidth=0.9)
        ax.axvspan(ARRT_PEAK_LOW_C, ARRT_PEAK_HIGH_C, color="#f2c94c", alpha=0.14)
        result = next(r for r in results if r["condition_key"] == key)
        flag = result.get("quality_flag", "ok")
        title_color = "#b00020" if flag != "ok" else "black"
        ax.set_title(
            f"{condition_label(key)}\nR2={float(result['kissinger_r2']):.3f}, H*={float(result['activation_enthalpy_H_star_kj_mol']):.1f}, S*={float(result['activation_entropy_S_star_j_mol_K']):.1f}",
            color=title_color,
            fontsize=9,
        )
        ax.set_xlim(70, 130)
        ax.set_xlabel("T (C)")
        ax.set_ylabel("DSC detrended (uW)")
        if rates:
            ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Multi-heating-rate DSC curves used for ARRT/Kissinger H*/S* labels", fontsize=14)
    fig.savefig(CURVE_GRID, dpi=180)
    plt.close(fig)


def plot_kissinger_grid(results: list[dict[str, str]], curves: list[dict[str, object]]) -> None:
    by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for curve in curves:
        if finite(curve.get("arrt_tp_C")):
            by_key[str(curve["condition_key"])].append(curve)

    keys = [row["condition_key"] for row in results]
    ncols = 3
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), squeeze=False, constrained_layout=True)
    for ax, key in zip(axes.ravel(), keys):
        rows = sorted(by_key.get(key, []), key=lambda r: float(r["heating_rate_C_min"]))
        x = np.array([1.0 / (float(r["arrt_tp_C"]) + 273.15) for r in rows], dtype=float)
        y = np.array([math.log((float(r["heating_rate_C_min"]) / 60.0) / ((float(r["arrt_tp_C"]) + 273.15) ** 3)) for r in rows], dtype=float)
        rates = np.array([float(r["heating_rate_C_min"]) for r in rows], dtype=float)
        ax.scatter(x, y, c=rates, cmap="viridis", s=42, edgecolor="black", linewidth=0.4)
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            dense = np.linspace(float(x.min()), float(x.max()), 80)
            ax.plot(dense, coef[0] * dense + coef[1], color="black", linewidth=1.1)
        for xi, yi, rate in zip(x, y, rates):
            ax.annotate(f"{rate:g}", (xi, yi), textcoords="offset points", xytext=(3, 3), fontsize=7)
        result = next(r for r in results if r["condition_key"] == key)
        flag = result.get("quality_flag", "ok")
        title_color = "#b00020" if flag != "ok" else ("#8a5a00" if float(result["kissinger_r2"]) < 0.95 else "black")
        ax.set_title(f"{condition_label(key)}\nR2={float(result['kissinger_r2']):.3f}, {flag}", color=title_color, fontsize=9)
        ax.set_xlabel("1 / Tp (1/K)")
        ax.set_ylabel("ln(beta / Tp^3)")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Kissinger fits behind ARRT H*/S* labels", fontsize=14)
    fig.savefig(KISSINGER_GRID, dpi=180)
    plt.close(fig)


def plot_hs_scatter(results: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for row in results:
        r2 = float(row["kissinger_r2"])
        flag = row.get("quality_flag", "ok")
        if flag != "ok":
            color = "#d62728"
            marker = "X"
            label = "suspect"
        elif r2 >= 0.95:
            color = "#2ca02c"
            marker = "o"
            label = "high R2"
        else:
            color = "#ffbf00"
            marker = "s"
            label = "low R2"
        ax.scatter(
            float(row["activation_enthalpy_H_star_kj_mol"]),
            float(row["activation_entropy_S_star_j_mol_K"]),
            s=80,
            c=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(condition_label(row["condition_key"]), (float(row["activation_enthalpy_H_star_kj_mol"]), float(row["activation_entropy_S_star_j_mol_K"])), textcoords="offset points", xytext=(5, 4), fontsize=7)
    handles = {}
    for color, marker, label in [("#2ca02c", "o", "high R2"), ("#ffbf00", "s", "low R2"), ("#d62728", "X", "suspect")]:
        handles[label] = ax.scatter([], [], c=color, marker=marker, edgecolor="black", label=label)
    ax.legend(handles.values(), handles.keys(), loc="best", fontsize=8)
    ax.set_xlabel("H* (kJ/mol)")
    ax.set_ylabel("S* (J/mol/K)")
    ax.set_title("ARRT H*/S* labels by Kissinger confidence")
    fig.savefig(HS_SCATTER, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = read_arrt_results()
    valid_keys = {row["condition_key"] for row in results}
    curves = collect_curves(valid_keys)
    plot_curve_grid(results, curves)
    plot_kissinger_grid(results, curves)
    plot_hs_scatter(results)
    summary = {
        "status": "complete",
        "n_conditions": len(results),
        "n_curves": len(curves),
        "figures": {
            "multi_heating_rate_curves": str(CURVE_GRID.relative_to(ROOT)),
            "kissinger_fits": str(KISSINGER_GRID.relative_to(ROOT)),
            "hs_confidence_scatter": str(HS_SCATTER.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
