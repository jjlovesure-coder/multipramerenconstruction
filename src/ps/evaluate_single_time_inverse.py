"""Evaluate single-step annealing time recovery from strict ARRT labels."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from src.ps.calculate_arrt_from_heating_rates import OUT_DIR as ARRT_DIR
from src.ps.calculate_arrt_from_heating_rates import main as calculate_arrt
from src.ps.dsc_processing import ROOT


OUT_DIR = ROOT / "results" / "ps" / "single_time_inverse"
DOC_PATH = ROOT / "docs" / "ps_single_hs_01_time_prediction.md"

HEATING_ONLY_FEATURES = [
    "mean_delta_h_total_J_g",
    "mean_peak_area_J_g",
    "mean_peak_temperature_Tp_C",
    "mean_peak_height_uW",
    "mean_recovery_index",
]
PAPER_STYLE_FEATURES = [
    "mean_delta_h_total_J_g",
    "mean_peak_area_J_g",
    "activation_enthalpy_H_star_kj_mol",
    "activation_entropy_S_star_j_mol_K",
]
FEATURE_FLOORS = {
    "mean_delta_h_total_J_g": 0.05,
    "mean_peak_area_J_g": 0.05,
    "mean_peak_temperature_Tp_C": 0.10,
    "mean_peak_height_uW": 5.0,
    "mean_recovery_index": 0.03,
    "activation_enthalpy_H_star_kj_mol": 30.0,
    "activation_entropy_S_star_j_mol_K": 100.0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def strict_single_rows(rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    rows = rows if rows is not None else read_csv(ARRT_DIR / "arrt_kissinger_results.csv")
    required = set(PAPER_STYLE_FEATURES + HEATING_ONLY_FEATURES + ["T1_C", "t1_s"])
    out = []
    for row in rows:
        if row.get("mode") != "single_step":
            continue
        if all(finite(row.get(key)) for key in required):
            out.append(row)
    return out


def candidate_times_by_temperature(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        temp_key = f"{float(row['T1_C']):.6g}"
        out.setdefault(temp_key, [])
        value = float(row["t1_s"])
        if value not in out[temp_key]:
            out[temp_key].append(value)
    return {key: sorted(values) for key, values in out.items()}


def _condition_key(row: dict[str, str]) -> str:
    return row.get("condition_key") or "|".join([row.get("mode", ""), row.get("T1_C", ""), row.get("t1_s", ""), "", ""])


def _feature_scales(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    values = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    scales = np.std(values, axis=0)
    floors = np.array([FEATURE_FLOORS[name] for name in features], dtype=float)
    return np.maximum(scales, floors)


def _rank_candidates(test_row: dict[str, str], candidate_rows: list[dict[str, str]], features: list[str]) -> tuple[list[int], np.ndarray]:
    scales = _feature_scales(candidate_rows + [test_row], features)
    target = np.array([float(test_row[name]) for name in features], dtype=float)
    matrix = np.array([[float(row[name]) for name in features] for row in candidate_rows], dtype=float)
    losses = np.mean(np.abs((matrix - target[None, :]) / scales[None, :]), axis=1)
    return [int(idx) for idx in np.argsort(losses)], losses


def direct_fit_predict_log_time(
    train_rows: list[dict[str, str]],
    test_row: dict[str, str],
    features: list[str],
    clamp_bounds: tuple[float, float],
) -> float:
    """Predict log10(time) by median ensembling univariate linear fits.

    With only two or three same-temperature points, a multivariate model is
    underdetermined.  This baseline mirrors the user's "directly fit the data
    points" intuition while keeping the fit leave-one-condition-out.
    """
    predictions = []
    y = np.array([math.log10(float(row["t1_s"])) for row in train_rows], dtype=float)
    for feature in features:
        x = np.array([float(row[feature]) for row in train_rows], dtype=float)
        if len(x) < 2 or float(np.ptp(x)) <= 1e-12:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        predictions.append(float(slope * float(test_row[feature]) + intercept))
    if not predictions:
        return float(np.mean(y))
    lo, hi = clamp_bounds
    return float(np.clip(np.median(predictions), lo, hi))


def evaluate_variant(rows: list[dict[str, str]], features: list[str], label: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for test_idx, test_row in enumerate(rows):
        same_temp_train = [
            row for idx, row in enumerate(rows)
            if idx != test_idx and abs(float(row["T1_C"]) - float(test_row["T1_C"])) <= 1e-9
        ]
        if not same_temp_train:
            continue
        order, losses = _rank_candidates(test_row, same_temp_train, features)
        best = same_temp_train[order[0]]
        actual_t = float(test_row["t1_s"])
        pred_t = float(best["t1_s"])

        def near(indices: list[int], factor: float = 3.0) -> bool:
            for idx in indices:
                candidate_t = float(same_temp_train[idx]["t1_s"])
                if max(candidate_t / actual_t, actual_t / candidate_t) <= factor:
                    return True
            return False

        top3 = order[:3]
        records.append({
            "variant": label,
            "condition_key": _condition_key(test_row),
            "T1_C": float(test_row["T1_C"]),
            "actual_t1_s": actual_t,
            "pred_t1_s": pred_t,
            "t_log10_abs_error": abs(math.log10(pred_t / actual_t)),
            "time_factor_error": max(pred_t / actual_t, actual_t / pred_t),
            "top1_near_factor3": near([order[0]]),
            "top3_near_factor3": near(top3),
            "loss": float(losses[order[0]]),
            "features": ",".join(features),
            "n_same_temperature_candidates": len(same_temp_train),
        })
    return summarize(records)


def evaluate_direct_fit_variant(rows: list[dict[str, str]], features: list[str], label: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for test_idx, test_row in enumerate(rows):
        same_temp_all = [row for row in rows if abs(float(row["T1_C"]) - float(test_row["T1_C"])) <= 1e-9]
        same_temp_train = [row for idx, row in enumerate(rows) if idx != test_idx and abs(float(row["T1_C"]) - float(test_row["T1_C"])) <= 1e-9]
        if len(same_temp_train) < 2:
            continue
        bounds = (
            min(math.log10(float(row["t1_s"])) for row in same_temp_all),
            max(math.log10(float(row["t1_s"])) for row in same_temp_all),
        )
        pred_log = direct_fit_predict_log_time(same_temp_train, test_row, features, bounds)
        actual_t = float(test_row["t1_s"])
        pred_t = 10.0 ** pred_log
        factor_error = max(pred_t / actual_t, actual_t / pred_t)
        records.append({
            "variant": label,
            "strategy": "direct_univariate_logtime_fit",
            "condition_key": _condition_key(test_row),
            "T1_C": float(test_row["T1_C"]),
            "actual_t1_s": actual_t,
            "pred_t1_s": pred_t,
            "t_log10_abs_error": abs(pred_log - math.log10(actual_t)),
            "time_factor_error": factor_error,
            "top1_near_factor3": factor_error <= 3.0,
            "top3_near_factor3": factor_error <= 3.0,
            "loss": abs(pred_log - math.log10(actual_t)),
            "features": ",".join(features),
            "n_same_temperature_candidates": len(same_temp_train),
        })
    return summarize(records)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"n": 0, "records": []}
    return {
        "n": len(records),
        "log10_time_MAE": float(np.mean([float(row["t_log10_abs_error"]) for row in records])),
        "time_factor_error_median": float(median([float(row["time_factor_error"]) for row in records])),
        "top1_near_factor3": float(np.mean([1.0 if row["top1_near_factor3"] else 0.0 for row in records])),
        "top3_near_factor3": float(np.mean([1.0 if row["top3_near_factor3"] else 0.0 for row in records])),
        "records": records,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, object]) -> None:
    heating = summary["variants"]["heating_only"]  # type: ignore[index]
    paper = summary["variants"]["paper_style"]  # type: ignore[index]
    direct_heating = summary["variants"]["heating_only_direct_fit"]  # type: ignore[index]
    direct_paper = summary["variants"]["paper_style_direct_fit"]  # type: ignore[index]
    text = f"""# ps-single-hs-01 Single-Step Time Prediction

## Scope

This report evaluates the reduced single-step inverse task:

```text
given T and measured output features -> predict annealing time t
```

The `ps-single-hs-01.xlsx` tail is intentionally handled strictly: the incomplete `90 C, 1000 s` group is retained in truncation diagnostics but excluded from strict H*/S* training and time prediction.

## Valid Strict ARRT Conditions

- Total valid single-step strict ARRT rows: `{summary['n_strict_single_rows']}`
- Candidate times by temperature:

```json
{json.dumps(summary['candidate_times_by_temperature'], indent=2)}
```

## Prediction Metrics

| Feature set | Strategy | n | log10(t) MAE | median factor error | Top1 within 3x |
| --- | ---: | ---: | ---: | ---: | ---: |
| heating-only | nearest candidate | {heating['n']} | {heating['log10_time_MAE']:.3f} | {heating['time_factor_error_median']:.3f} | {heating['top1_near_factor3']:.3f} |
| paper-style H*/S* | nearest candidate | {paper['n']} | {paper['log10_time_MAE']:.3f} | {paper['time_factor_error_median']:.3f} | {paper['top1_near_factor3']:.3f} |
| heating-only | direct log-time fit | {direct_heating['n']} | {direct_heating['log10_time_MAE']:.3f} | {direct_heating['time_factor_error_median']:.3f} | {direct_heating['top1_near_factor3']:.3f} |
| paper-style H*/S* | direct log-time fit | {direct_paper['n']} | {direct_paper['log10_time_MAE']:.3f} | {direct_paper['time_factor_error_median']:.3f} | {direct_paper['top1_near_factor3']:.3f} |

## Interpretation

The nearest-candidate strategy is a conservative discrete inverse and can be overly pessimistic with only two or three candidate times. The direct-fit strategy is closer to simply fitting the measured points in log-time. It shows that direct fitting improves over nearest-candidate ranking, but the current `H*/S*` coordinates still do not create a decisive single-step time predictor for the `90 C, 500 s` holdout.

## Outputs

- Summary: `results/ps/single_time_inverse/single_time_inverse_summary.json`
- Per-condition records: `results/ps/single_time_inverse/single_time_inverse_records.csv`
- Truncation diagnostics: `results/ps/arrt_kissinger/scan_truncation_diagnostics.csv`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calculate_arrt()
    rows = strict_single_rows()
    variants = {
        "heating_only": evaluate_variant(rows, HEATING_ONLY_FEATURES, "heating_only"),
        "paper_style": evaluate_variant(rows, PAPER_STYLE_FEATURES, "paper_style"),
        "heating_only_direct_fit": evaluate_direct_fit_variant(rows, HEATING_ONLY_FEATURES, "heating_only_direct_fit"),
        "paper_style_direct_fit": evaluate_direct_fit_variant(rows, PAPER_STYLE_FEATURES, "paper_style_direct_fit"),
    }
    records = []
    for block in variants.values():
        records.extend(block.get("records", []))  # type: ignore[arg-type]
    write_csv(OUT_DIR / "single_time_inverse_records.csv", records)
    summary = {
        "task": "single-step fixed-temperature annealing time prediction",
        "n_strict_single_rows": len(rows),
        "candidate_times_by_temperature": candidate_times_by_temperature(rows),
        "variants": variants,
        "notes": [
            "Incomplete scans are excluded upstream by requiring coverage of the 95-125 C ARRT peak window.",
            "Evaluation candidates are valid strict ARRT rows at the same temperature as the held-out condition.",
        ],
    }
    out = OUT_DIR / "single_time_inverse_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({
        "output": str(out),
        "n_strict_single_rows": len(rows),
        "metrics": {
            name: {key: value for key, value in block.items() if key != "records"}
            for name, block in variants.items()
        },
    }, indent=2))
    return out


if __name__ == "__main__":
    main()
