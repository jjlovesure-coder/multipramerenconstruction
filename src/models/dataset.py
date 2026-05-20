"""Build processed annealing datasets from digitized figure data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from src.physics.arrt import activation_enthalpy_from_entropy
from src.physics.tnm_features import stretched_dose


ROOT = Path(__file__).resolve().parents[2]
FIG2_10 = ROOT / "results" / "extracted" / "data_points" / "overlap_enhanced" / "fig2a_c_10series_overlap_filled_points.csv"
S6 = ROOT / "results" / "extracted" / "data_points" / "fig2d_s6_experimental_points.csv"
S7 = ROOT / "results" / "extracted" / "data_points" / "s7_experimental_points.csv"
FIG3_H = ROOT / "results" / "extracted" / "data_points" / "fig3b_hstar_tp_digitized.csv"
PROCESSED = ROOT / "data" / "processed"


PANEL_T1 = {"a": 348.0, "b": 363.0, "c": 373.0}
T2_K = 383.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _series_temperature_k(row: dict[str, str]) -> float:
    return float(row["series"].split()[0])


def interpolate_by_temperature_log_time(
    rows: Iterable[dict[str, str]],
    temperature_k: float,
    time_s: float,
    value_key: str,
) -> float:
    """Bilinearly interpolate sparse paper data in T and log(time).

    Digitized S*/H* supplementary data are measured on temperature series.
    Nearest-neighbor matching creates discontinuities when a requested
    annealing condition falls between series or between log-spaced times, so
    the dataset builder uses this smoother interpolation when possible and
    naturally falls back to edge interpolation at the boundary.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot interpolate an empty row set")
    target_temp = float(temperature_k)
    target_log_time = float(np.log10(max(float(time_s), 1e-12)))
    temps = sorted({_series_temperature_k(row) for row in rows})
    lower_temps = [temp for temp in temps if temp <= target_temp]
    upper_temps = [temp for temp in temps if temp >= target_temp]
    temp_lo = lower_temps[-1] if lower_temps else temps[0]
    temp_hi = upper_temps[0] if upper_temps else temps[-1]

    def interp_time(temp: float) -> float:
        candidates = sorted(
            (float(np.log10(float(row["annealing_time_s"]))), float(row[value_key]))
            for row in rows
            if _series_temperature_k(row) == temp
        )
        if not candidates:
            raise ValueError(f"No rows at {temp} K")
        log_times = np.array([item[0] for item in candidates], dtype=float)
        values = np.array([item[1] for item in candidates], dtype=float)
        return float(np.interp(target_log_time, log_times, values))

    value_lo = interp_time(temp_lo)
    value_hi = interp_time(temp_hi)
    if temp_hi == temp_lo:
        return value_lo
    weight = (target_temp - temp_lo) / (temp_hi - temp_lo)
    return float(value_lo + weight * (value_hi - value_lo))


def interpolate_by_log_time(rows: Iterable[dict[str, str]], time_s: float, value_key: str) -> float:
    """Interpolate one-dimensional digitized relationships in log(time)."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot interpolate an empty row set")
    target_log_time = float(np.log10(max(float(time_s), 1e-12)))
    candidates = sorted(
        (float(np.log10(float(row["annealing_time_s"]))), float(row[value_key]))
        for row in rows
    )
    log_times = np.array([item[0] for item in candidates], dtype=float)
    values = np.array([item[1] for item in candidates], dtype=float)
    return float(np.interp(target_log_time, log_times, values))


def _nearest_by_log_time(rows: Iterable[dict[str, str]], temperature_k: float, time_s: float) -> dict[str, str]:
    rows = list(rows)
    available_temps = sorted({float(r["series"].split()[0]) for r in rows})
    nearest_temp = min(available_temps, key=lambda t: abs(t - float(temperature_k)))
    candidates = [r for r in rows if float(r["series"].split()[0]) == nearest_temp]
    if not candidates:
        raise ValueError(f"No S* rows for {temperature_k} K")
    return min(candidates, key=lambda r: abs(np.log10(float(r["annealing_time_s"]) / float(time_s))))


def _nearest_s6(rows: Iterable[dict[str, str]], temperature_k: float, delta_h: float) -> float:
    candidates = [r for r in rows if float(r["series"].split()[0]) == float(temperature_k)]
    if not candidates:
        return float("nan")
    best = min(candidates, key=lambda r: abs(float(r["delta_h_kj_mol"]) - float(delta_h)))
    return float(best["delta_h_peak_kj_mol"])


def _nearest_fig3_h(rows: Iterable[dict[str, str]], time_s: float) -> dict[str, str]:
    rows = list(rows)
    return min(rows, key=lambda r: abs(np.log10(float(r["annealing_time_s"]) / float(time_s))))


def interpolate_fig3_by_log_time(rows: Iterable[dict[str, str]], time_s: float) -> dict[str, float]:
    rows = list(rows)
    return {
        "h_star_kj_mol": interpolate_by_log_time(rows, time_s, "h_star_kj_mol"),
        "tp_1000kps_k": interpolate_by_log_time(rows, time_s, "tp_1000kps_k"),
    }


def build_dataset() -> Path:
    """Create data/processed/annealing_dataset.csv."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fig2_rows = _read_csv(FIG2_10)
    s6_rows = _read_csv(S6)
    s7_rows = _read_csv(S7)
    fig3_h_rows = _read_csv(FIG3_H)

    out_rows = []
    for row in fig2_rows:
        panel = row["panel"]
        t1_temperature_k = PANEL_T1[panel]
        t2_temperature_k = T2_K
        t1_s = float(row["t1_s"])
        t2_s = float(row["t2_s"])
        delta_h = float(row["delta_h_kj_mol"])

        s1 = interpolate_by_temperature_log_time(s7_rows, t1_temperature_k, t1_s, "s_star_j_mol_k")
        s2 = interpolate_by_temperature_log_time(s7_rows, t2_temperature_k, t2_s, "s_star_j_mol_k")
        h1 = activation_enthalpy_from_entropy(t1_temperature_k, t1_s, s1)
        h2 = activation_enthalpy_from_entropy(t2_temperature_k, t2_s, s2)
        fig3_h1 = interpolate_fig3_by_log_time(fig3_h_rows, t1_s)
        fig3_h2 = interpolate_fig3_by_log_time(fig3_h_rows, t2_s)
        d1 = stretched_dose(t1_temperature_k, t1_s)
        d2 = stretched_dose(t2_temperature_k, t2_s)
        delta_h_peak = _nearest_s6(s6_rows, t1_temperature_k, delta_h)

        out_rows.append({
            "panel": panel,
            "t1_temperature_k": t1_temperature_k,
            "t1_s": t1_s,
            "t2_temperature_k": t2_temperature_k,
            "t2_s": t2_s,
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
            "h1_kj_mol_fig3": float(fig3_h1["h_star_kj_mol"]),
            "h2_kj_mol_fig3": float(fig3_h2["h_star_kj_mol"]),
            "tp1_1000kps_k_fig3": float(fig3_h1["tp_1000kps_k"]),
            "tp2_1000kps_k_fig3": float(fig3_h2["tp_1000kps_k"]),
            "tnm_dose1": d1,
            "tnm_dose2": d2,
            "delta_h_kj_mol": delta_h,
            "delta_h_peak_kj_mol": delta_h_peak,
            "memory_effect_label": int(delta_h_peak >= 0.05 and delta_h >= 0.4),
            "source_assignment_method": row["assignment_method"],
        })

    fields = list(out_rows[0].keys())
    path = PROCESSED / "annealing_dataset.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return path


if __name__ == "__main__":
    print(build_dataset())
