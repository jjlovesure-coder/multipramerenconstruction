"""Evaluate the new PS single-step batch against the pre-update model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.ps.dsc_processing import (
    DEFAULT_HEATING_RATE_C_MIN,
    RAW_FILES,
    add_recovery_and_path_indices,
    all_recovery_scan_indices,
    baseline_scan,
    corrected_signal,
    estimate_noise,
    extract_features,
    infer_condition,
    read_protocol,
    read_signal,
    scan_for_segment,
    segment_bounds,
)
from src.ps.inverse_design import load_model
from src.ps.train_ps_model import TARGETS, featurize, read_rows


ROOT = Path(__file__).resolve().parents[2]
BATCH_PATH = ROOT / "data" / "dsc" / "ps-single-01.xlsx"
OLD_DATASET = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"
OUT_DIR = ROOT / "results" / "ps" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_value(value: float | str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def extract_batch_rows() -> tuple[list[dict], list[dict]]:
    ref = baseline_scan(RAW_FILES["ref"])
    empty = baseline_scan(RAW_FILES["empty"])
    noise = estimate_noise(empty, ref)
    segments = read_protocol(BATCH_PATH)
    bounds = segment_bounds(segments)
    signal = read_signal(BATCH_PATH)

    rows: list[dict] = []
    excluded: list[dict] = []
    prev_scan = -1
    standard_cycle_id = 0
    for scan_idx in all_recovery_scan_indices(segments):
        cycle_segments = segments[prev_scan + 1: scan_idx + 1]
        prev_scan = scan_idx
        condition = infer_condition("single_eig_01", cycle_segments)
        heating_rate = segments[scan_idx].rate_c_min
        record = {
            "source_file": BATCH_PATH.name,
            "scan_segment_index": scan_idx + 1,
            "mode": condition["mode"],
            "T1_C": condition["T1_C"],
            "t1_s": condition["t1_s"],
            "T2_C": condition["T2_C"],
            "t2_s": condition["t2_s"],
            "total_anneal_time_s": condition["t1_s"],
            "heating_rate_C_min": heating_rate,
        }
        if condition["mode"] != "single_step":
            record["exclusion_reason"] = "could_not_infer_single_step_condition"
            excluded.append(record)
            continue
        if abs(heating_rate - DEFAULT_HEATING_RATE_C_MIN) > 2.0:
            record["exclusion_reason"] = "nonstandard_heating_rate"
            excluded.append(record)
            continue
        scan = scan_for_segment(signal, segments[scan_idx], bounds[scan_idx])
        corrected = corrected_signal(scan, ref)
        features = extract_features(corrected, noise, heating_rate)
        standard_cycle_id += 1
        row = {
            "sample_id": f"ps_single_01_{standard_cycle_id:03d}",
            "source_file": BATCH_PATH.name,
            "cycle_id": standard_cycle_id,
            "mode": "single_step",
            "T1_C": condition["T1_C"],
            "t1_s": condition["t1_s"],
            "T2_C": "",
            "t2_s": "",
            "total_anneal_time_s": condition["t1_s"],
            "heating_rate_C_min": heating_rate,
            "cooling_rate_C_min": 60.0,
            "noise_sigma_uW": noise,
            **features,
        }
        rows.append(row)
    return rows, excluded


def merge_for_indices(new_rows: list[dict]) -> list[dict]:
    old_rows = read_rows(OLD_DATASET)
    merged: list[dict] = [dict(row) for row in old_rows]
    merged.extend(new_rows)
    add_recovery_and_path_indices(merged)
    return merged[-len(new_rows):]


def compare_to_model(rows: list[dict]) -> list[dict]:
    model = load_model()
    x = np.array([featurize({k: clean_value(v) for k, v in row.items()}) for row in rows], dtype=float)
    pred, unc = model.predict(x)
    out = []
    for i, row in enumerate(rows):
        record = {
            "sample_id": row["sample_id"],
            "mode": row["mode"],
            "T1_C": float(row["T1_C"]),
            "t1_s": float(row["t1_s"]),
            "heating_rate_C_min": float(row["heating_rate_C_min"]),
        }
        for j, target in enumerate(TARGETS):
            actual = float(row[target])
            predicted = float(pred[i, j])
            record[f"actual_{target}"] = actual
            record[f"pred_{target}"] = predicted
            record[f"uncertainty_{target}"] = float(unc[i, j])
            record[f"error_{target}"] = predicted - actual
            record[f"abs_error_{target}"] = abs(predicted - actual)
        out.append(record)
    return out


def summarize(comparison: list[dict], excluded: list[dict]) -> dict:
    metrics = {}
    for target in TARGETS:
        errors = np.array([row[f"error_{target}"] for row in comparison], dtype=float)
        metrics[target] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "bias": float(np.mean(errors)),
        }
    return {
        "source_file": str(BATCH_PATH.relative_to(ROOT)),
        "standard_scans_analyzed": len(comparison),
        "excluded_scans": excluded,
        "metrics": metrics,
        "notes": [
            "Metrics compare the new single-step batch against the model before this batch is added to training.",
            "The 95 C, 100 s run is excluded from model updating because its final heating scan is 60 deg C/min, not the standard 10 deg C/min.",
        ],
    }


def main() -> None:
    rows, excluded = extract_batch_rows()
    indexed_rows = merge_for_indices(rows)
    comparison = compare_to_model(indexed_rows)
    summary = summarize(comparison, excluded)

    comparison_path = OUT_DIR / "ps_single_01_prediction_vs_observed.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    excluded_path = OUT_DIR / "ps_single_01_excluded_scans.csv"
    if excluded:
        with excluded_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(excluded[0]))
            writer.writeheader()
            writer.writerows(excluded)

    summary_path = OUT_DIR / "ps_single_01_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
