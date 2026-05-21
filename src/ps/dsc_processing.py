"""Parse PS DSC Excel exports and extract finite-time enthalpy-recovery features.

The instrument files contain a metadata block, a temperature program, and then
columns named Time / Temp. / DSC / DDSC. The PS sample mass is known, so DSC
temperature integrals are also converted to specific enthalpy in J/g.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "dsc"
OUT_DIR = ROOT / "data" / "ps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RAW_FILES = {
    "empty": RAW_DIR / "ps-empty-01.xlsx",
    "ref": RAW_DIR / "ps-ref-01.xlsx",
    "onestep_50": RAW_DIR / "PS-onestep-01.xlsx",
    "onestep_70": RAW_DIR / "PS-onestep-02.xlsx",
    "single_eig_01": RAW_DIR / "ps-single-01.xlsx",
    "kovacs_eig_01": RAW_DIR / "ps-kovacs-01.xlsx",
    "kovacs_eig_03": RAW_DIR / "ps-kovacs-03.xlsx",
    "kovacs_eig_04": RAW_DIR / "ps-kovacs-04.xlsx",
    "hs_01": RAW_DIR / "ps-hs-01.xlsx",
    "hs_02": RAW_DIR / "ps-hs-02.xlsx",
    "kovacs_80_to_90": RAW_DIR / "pskovacs.xlsx",
    "twostep_90_to_80": RAW_DIR / "twosteps.xlsx",
}

PS_SAMPLE_MASS_MG = 4.7
PS_SAMPLE_MASS_G = PS_SAMPLE_MASS_MG / 1000.0
DEFAULT_HEATING_RATE_C_MIN = 10.0

PROCESS_FEATURES = [
    "process_total_integral_uW_min",
    "process_total_abs_integral_uW_min",
    "process_total_mean_uW",
    "process_total_std_uW",
    "process_hold_integral_uW_min",
    "process_hold_abs_integral_uW_min",
    "process_cooling_integral_uW_min",
    "process_cooling_abs_integral_uW_min",
    "process_between_step_heating_integral_uW_min",
    "process_step1_hold_mean_uW",
    "process_step1_hold_slope_uW_min",
    "process_step1_hold_delta_uW",
    "process_step2_hold_mean_uW",
    "process_step2_hold_slope_uW_min",
    "process_step2_hold_delta_uW",
    "process_final_cooling_mean_uW",
    "process_final_cooling_slope_uW_min",
    "process_pre_scan_start_dsc_uW",
    "process_pre_scan_end_dsc_uW",
    "process_pre_scan_dsc_shift_uW",
]


@dataclass(frozen=True)
class Segment:
    index: int
    start_c: float
    end_c: float
    rate_c_min: float
    hold_min: float
    sample_period_s: float

    @property
    def ramp_min(self) -> float:
        return abs(self.end_c - self.start_c) / abs(self.rate_c_min)

    @property
    def duration_min(self) -> float:
        return self.ramp_min + self.hold_min


def find_header_row(path: Path) -> int:
    preview = pd.read_excel(path, sheet_name=0, header=None, nrows=200, dtype=str)
    for idx, row in preview.iterrows():
        values = [str(v).strip() for v in row.tolist()]
        if "Time" in values and "Temp." in values and "DSC" in values:
            return int(idx)
    raise ValueError(f"Could not find Time/Temp./DSC header in {path}")


def read_protocol(path: Path) -> list[Segment]:
    header_row = find_header_row(path)
    block = pd.read_excel(path, sheet_name=0, header=None, skiprows=8, nrows=header_row - 8)
    segments: list[Segment] = []
    for _, row in block.iterrows():
        try:
            segment_id = int(float(row.iloc[1]))
            start_c = float(row.iloc[2])
            end_c = float(row.iloc[3])
            rate = float(row.iloc[4])
            hold = float(row.iloc[5])
            period = float(row.iloc[6])
        except (TypeError, ValueError):
            continue
        segments.append(Segment(segment_id, start_c, end_c, rate, hold, period))
    return segments


def read_signal(path: Path) -> pd.DataFrame:
    header_row = find_header_row(path)
    df = pd.read_excel(path, sheet_name=0, header=header_row, skiprows=[header_row + 1])
    df = df[["Time", "Temp.", "DSC", "DDSC"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time", "Temp.", "DSC"]).reset_index(drop=True)
    return df.rename(columns={"Temp.": "Temp_C"})


def segment_bounds(segments: list[Segment]) -> list[tuple[float, float]]:
    bounds = []
    cursor = 0.0
    for segment in segments:
        start = cursor
        cursor += segment.duration_min
        bounds.append((start, cursor))
    return bounds


def heating_scan_indices(segments: list[Segment]) -> list[int]:
    return [
        idx
        for idx, segment in enumerate(segments)
        if abs(segment.start_c - 30.0) <= 2.0 and segment.end_c >= 190.0 and abs(segment.rate_c_min - 10.0) <= 2.0
    ]


def all_recovery_scan_indices(segments: list[Segment]) -> list[int]:
    return [
        idx
        for idx, segment in enumerate(segments)
        if abs(segment.start_c - 30.0) <= 2.0 and segment.end_c >= 190.0 and segment.rate_c_min > 0
    ]


def scan_for_segment(df: pd.DataFrame, segment: Segment, bounds: tuple[float, float]) -> pd.DataFrame:
    start_min, _ = bounds
    ramp_end = start_min + segment.ramp_min
    scan = df[(df["Time"] >= start_min) & (df["Time"] <= ramp_end)].copy()
    scan = scan[(scan["Temp_C"] >= 30.0) & (scan["Temp_C"] <= 200.0)]
    return scan.reset_index(drop=True)


def process_feature_names() -> list[str]:
    return list(PROCESS_FEATURES)


def _empty_process_features() -> dict[str, float]:
    return {name: 0.0 for name in PROCESS_FEATURES}


def _window_for_bounds(df: pd.DataFrame, bounds: tuple[float, float]) -> pd.DataFrame:
    start_min, end_min = bounds
    return df[(df["Time"] >= start_min) & (df["Time"] <= end_min)].copy().reset_index(drop=True)


def _hold_window(df: pd.DataFrame, segment: Segment, bounds: tuple[float, float]) -> pd.DataFrame:
    if segment.hold_min <= 0.0:
        return df.iloc[0:0].copy()
    start_min, _ = bounds
    hold_start = start_min + segment.ramp_min
    hold_end = hold_start + segment.hold_min
    return df[(df["Time"] >= hold_start) & (df["Time"] <= hold_end)].copy().reset_index(drop=True)


def _integral_time_uW_min(window: pd.DataFrame, absolute: bool = False) -> float:
    if len(window) < 2:
        return 0.0
    y = window["DSC"].to_numpy(dtype=float)
    if absolute:
        y = np.abs(y)
    return float(np.trapezoid(y, window["Time"].to_numpy(dtype=float)))


def _mean(window: pd.DataFrame) -> float:
    return float(window["DSC"].mean()) if len(window) else 0.0


def _std(window: pd.DataFrame) -> float:
    return float(window["DSC"].std(ddof=0)) if len(window) else 0.0


def _slope(window: pd.DataFrame) -> float:
    if len(window) < 2:
        return 0.0
    x = window["Time"].to_numpy(dtype=float)
    y = window["DSC"].to_numpy(dtype=float)
    if float(np.ptp(x)) <= 1e-12:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _delta(window: pd.DataFrame) -> float:
    if len(window) < 2:
        return 0.0
    y = window["DSC"].to_numpy(dtype=float)
    return float(y[-1] - y[0])


def extract_pre_scan_process_features(
    df: pd.DataFrame,
    pre_scan_segments: list[Segment],
    pre_scan_bounds: list[tuple[float, float]],
) -> dict[str, float]:
    """Summarize only the annealing/cooling program before the final heating scan.

    These features can be used as auxiliary training targets. They deliberately
    exclude the final 30->200 C scan, which remains the supervised output.
    """
    features = _empty_process_features()
    if not pre_scan_segments or not pre_scan_bounds:
        return features

    windows = [_window_for_bounds(df, bounds) for bounds in pre_scan_bounds]
    total = pd.concat([window for window in windows if len(window)], ignore_index=True) if any(len(w) for w in windows) else df.iloc[0:0].copy()
    cooling_windows = [
        window
        for segment, window in zip(pre_scan_segments, windows)
        if segment.end_c < segment.start_c and len(window)
    ]
    between_heating_windows = [
        window
        for segment, window in zip(pre_scan_segments, windows)
        if segment.end_c > segment.start_c and len(window)
    ]
    hold_windows = [_hold_window(df, segment, bounds) for segment, bounds in zip(pre_scan_segments, pre_scan_bounds)]
    nonempty_holds = [window for window in hold_windows if len(window)]
    final_cooling = cooling_windows[-1] if cooling_windows else df.iloc[0:0].copy()

    features.update({
        "process_total_integral_uW_min": _integral_time_uW_min(total),
        "process_total_abs_integral_uW_min": _integral_time_uW_min(total, absolute=True),
        "process_total_mean_uW": _mean(total),
        "process_total_std_uW": _std(total),
        "process_hold_integral_uW_min": sum(_integral_time_uW_min(window) for window in nonempty_holds),
        "process_hold_abs_integral_uW_min": sum(_integral_time_uW_min(window, absolute=True) for window in nonempty_holds),
        "process_cooling_integral_uW_min": sum(_integral_time_uW_min(window) for window in cooling_windows),
        "process_cooling_abs_integral_uW_min": sum(_integral_time_uW_min(window, absolute=True) for window in cooling_windows),
        "process_between_step_heating_integral_uW_min": sum(_integral_time_uW_min(window) for window in between_heating_windows),
        "process_final_cooling_mean_uW": _mean(final_cooling),
        "process_final_cooling_slope_uW_min": _slope(final_cooling),
    })
    if len(nonempty_holds) >= 1:
        features["process_step1_hold_mean_uW"] = _mean(nonempty_holds[0])
        features["process_step1_hold_slope_uW_min"] = _slope(nonempty_holds[0])
        features["process_step1_hold_delta_uW"] = _delta(nonempty_holds[0])
    if len(nonempty_holds) >= 2:
        features["process_step2_hold_mean_uW"] = _mean(nonempty_holds[1])
        features["process_step2_hold_slope_uW_min"] = _slope(nonempty_holds[1])
        features["process_step2_hold_delta_uW"] = _delta(nonempty_holds[1])
    if len(total):
        start = float(total["DSC"].iloc[0])
        end = float(total["DSC"].iloc[-1])
        features["process_pre_scan_start_dsc_uW"] = start
        features["process_pre_scan_end_dsc_uW"] = end
        features["process_pre_scan_dsc_shift_uW"] = end - start
    return features


def infer_condition(file_key: str, cycle_segments: list[Segment]) -> dict[str, float | str]:
    if file_key in {"ref", "empty"}:
        return {"mode": file_key, "T1_C": math.nan, "t1_s": math.nan, "T2_C": math.nan, "t2_s": math.nan}
    if len(cycle_segments) == 3:
        first = cycle_segments[0]
        return {
            "mode": "single_step",
            "T1_C": first.end_c,
            "t1_s": first.hold_min * 60.0,
            "T2_C": math.nan,
            "t2_s": math.nan,
        }
    if len(cycle_segments) == 4:
        first, second = cycle_segments[0], cycle_segments[1]
        return {
            "mode": "two_step",
            "T1_C": first.end_c,
            "t1_s": first.hold_min * 60.0,
            "T2_C": second.end_c,
            "t2_s": second.hold_min * 60.0,
        }
    return {"mode": "unknown", "T1_C": math.nan, "t1_s": math.nan, "T2_C": math.nan, "t2_s": math.nan}


def baseline_scan(path: Path) -> pd.DataFrame:
    segments = read_protocol(path)
    df = read_signal(path)
    heat_indices = heating_scan_indices(segments)
    if not heat_indices:
        raise ValueError(f"No heating scan found in {path}")
    idx = heat_indices[0]
    return scan_for_segment(df, segments[idx], segment_bounds(segments)[idx])


def corrected_signal(scan: pd.DataFrame, ref_scan: pd.DataFrame) -> pd.DataFrame:
    ref = ref_scan.sort_values("Temp_C")
    scan = scan.sort_values("Temp_C").copy()
    ref_dsc = np.interp(scan["Temp_C"].to_numpy(), ref["Temp_C"].to_numpy(), ref["DSC"].to_numpy())
    scan["DSC_corrected_uW"] = scan["DSC"].to_numpy() - ref_dsc
    return scan


def detrend_window(scan: pd.DataFrame, low_c: float = 40.0, high_c: float = 160.0) -> pd.DataFrame:
    window = scan[(scan["Temp_C"] >= low_c) & (scan["Temp_C"] <= high_c)].copy()
    if len(window) < 5:
        window["DSC_detrended_uW"] = np.nan
        return window
    x = window["Temp_C"].to_numpy()
    y = window["DSC_corrected_uW"].to_numpy()
    baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    window["DSC_detrended_uW"] = y - baseline
    return window


def estimate_noise(empty_scan: pd.DataFrame, ref_scan: pd.DataFrame) -> float:
    corrected = corrected_signal(empty_scan, ref_scan)
    window = detrend_window(corrected, 40.0, 70.0)
    values = window["DSC_detrended_uW"].dropna().to_numpy()
    if len(values) == 0:
        return 1.0
    return float(np.std(values))


def integral_uW_C_to_J_g(integral_uW_C: float, heating_rate_C_min: float = DEFAULT_HEATING_RATE_C_MIN) -> float:
    seconds_per_c = 60.0 / heating_rate_C_min
    energy_j = integral_uW_C * seconds_per_c * 1e-6
    return energy_j / PS_SAMPLE_MASS_G


def extract_features(scan: pd.DataFrame, noise_sigma_uW: float, heating_rate_c_min: float = DEFAULT_HEATING_RATE_C_MIN) -> dict[str, float | int]:
    window = detrend_window(scan, 40.0, 160.0)
    peak_window = window[(window["Temp_C"] >= 70.0) & (window["Temp_C"] <= 140.0)].copy()
    if len(peak_window) < 5:
        return {
            "delta_h_total_rel": math.nan,
            "delta_h_total_J_g": math.nan,
            "peak_area_rel": math.nan,
            "peak_area_J_g": math.nan,
            "peak_temperature_Tp_C": math.nan,
            "peak_height_uW": math.nan,
            "onset_temperature_C": math.nan,
            "kovacs_peak_label": 0,
        }

    temp = window["Temp_C"].to_numpy()
    y = window["DSC_detrended_uW"].to_numpy()
    delta_h_total = float(np.trapezoid(y, temp))

    ptemp = peak_window["Temp_C"].to_numpy()
    py = peak_window["DSC_detrended_uW"].to_numpy()
    peak_idx = int(np.argmax(np.abs(py)))
    peak_height = float(py[peak_idx])
    tp = float(ptemp[peak_idx])
    threshold = 0.2 * abs(peak_height)
    local_mask = (np.abs(py) >= threshold) & (np.abs(ptemp - tp) <= 25.0)
    peak_area = float(np.trapezoid(py[local_mask], ptemp[local_mask])) if np.any(local_mask) else 0.0

    onset_mask = np.abs(py) >= max(3.0 * noise_sigma_uW, 0.1 * abs(peak_height))
    onset = float(ptemp[np.argmax(onset_mask)]) if np.any(onset_mask) else math.nan
    return {
        "delta_h_total_rel": delta_h_total,
        "delta_h_total_J_g": integral_uW_C_to_J_g(delta_h_total, heating_rate_c_min),
        "peak_area_rel": peak_area,
        "peak_area_J_g": integral_uW_C_to_J_g(peak_area, heating_rate_c_min),
        "peak_temperature_Tp_C": tp,
        "peak_height_uW": peak_height,
        "onset_temperature_C": onset,
        # A relaxation peak in the final scan is not by itself a Kovacs peak.
        # The current PS pre-experiments did not show a clear Kovacs peak, so
        # keep this conservative until manual cycle-level annotations exist.
        "kovacs_peak_label": 0,
    }


def build_ps_dataset() -> Path:
    ref = baseline_scan(RAW_FILES["ref"])
    empty = baseline_scan(RAW_FILES["empty"])
    noise = estimate_noise(empty, ref)
    rows = []

    for file_key, path in RAW_FILES.items():
        if file_key in {"ref", "empty"}:
            continue
        segments = read_protocol(path)
        bounds = segment_bounds(segments)
        df = read_signal(path)
        scan_indices = all_recovery_scan_indices(segments)
        prev_scan = -1
        cycle_id = 0
        for heat_idx in scan_indices:
            cycle_start_idx = prev_scan + 1
            cycle_segments = segments[cycle_start_idx: heat_idx + 1]
            prev_scan = heat_idx
            if abs(segments[heat_idx].rate_c_min - DEFAULT_HEATING_RATE_C_MIN) > 2.0:
                continue
            cycle_id += 1
            condition = infer_condition(file_key, cycle_segments)
            if condition["mode"] == "unknown":
                continue
            pre_segments = cycle_segments[:-1]
            cycle_bound_slice = bounds[cycle_start_idx: heat_idx + 1]
            process_features = extract_pre_scan_process_features(df, pre_segments, cycle_bound_slice[:-1])
            scan = scan_for_segment(df, segments[heat_idx], bounds[heat_idx])
            if len(scan) < 20:
                continue
            corrected = corrected_signal(scan, ref)
            features = extract_features(corrected, noise, segments[heat_idx].rate_c_min)
            total_time = float(condition["t1_s"])
            if condition["mode"] == "two_step":
                total_time += float(condition["t2_s"])
            rows.append({
                "sample_id": f"{file_key}_{cycle_id:03d}",
                "source_file": path.name,
                "cycle_id": cycle_id,
                "mode": condition["mode"],
                "T1_C": condition["T1_C"],
                "t1_s": condition["t1_s"],
                "T2_C": condition["T2_C"],
                "t2_s": condition["t2_s"],
                "total_anneal_time_s": total_time,
                "heating_rate_C_min": segments[heat_idx].rate_c_min,
                "cooling_rate_C_min": 60.0,
                "noise_sigma_uW": noise,
                **process_features,
                **features,
            })

    add_recovery_and_path_indices(rows)
    path = OUT_DIR / "ps_preexperiment_features.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def add_recovery_and_path_indices(rows: list[dict[str, float | str | int]]) -> None:
    maxima: dict[tuple[str, float], float] = {}
    for row in rows:
        key = (str(row["mode"]), float(row["T1_C"]))
        maxima[key] = max(maxima.get(key, 0.0), abs(float(row["delta_h_total_J_g"])))
    for row in rows:
        key = (str(row["mode"]), float(row["T1_C"]))
        denom = maxima.get(key, 0.0) or 1.0
        row["recovery_index"] = abs(float(row["delta_h_total_J_g"])) / denom

    single_rows = [r for r in rows if r["mode"] == "single_step"]
    for row in rows:
        if row["mode"] != "two_step" or not single_rows:
            row["path_dependence_index"] = 0.0
            continue
        t2 = float(row["T2_C"])
        time = max(float(row["t2_s"]), 1e-9)
        match = min(
            single_rows,
            key=lambda r: abs(float(r["T1_C"]) - t2) + 10.0 * abs(math.log10(max(float(r["t1_s"]), 1e-9) / time)),
        )
        row["path_dependence_index"] = float(row["delta_h_total_J_g"]) - float(match["delta_h_total_J_g"])


if __name__ == "__main__":
    print(build_ps_dataset())
