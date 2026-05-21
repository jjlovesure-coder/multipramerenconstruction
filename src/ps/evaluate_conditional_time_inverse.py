"""Evaluate fixed-temperature conditional t1/t2 inverse reconstruction."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.ps.conditional_time_inverse import (
    OUT_DIR,
    PAPER_STYLE_TARGETS,
    fit_strict_paper_model,
    search,
    strict_paper_rows,
)
from src.ps.dsc_processing import ROOT
from src.ps.strict_arrt_inverse_reconstruction import reconstruct as reconstruct_strict_arrt
from src.ps.train_strict_arrt_model import train as train_strict_arrt


SUMMARY_PATH = OUT_DIR / "conditional_time_inverse_summary.json"
BY_CONDITION_PATH = OUT_DIR / "conditional_time_inverse_by_condition.csv"
TOP_CANDIDATES_PATH = OUT_DIR / "conditional_time_inverse_top_candidates.csv"
REPORT_PATH = ROOT / "docs" / "ps_conditional_time_inverse.md"
FOUR_PARAMETER_SUMMARY_PATH = ROOT / "results" / "ps" / "strict_arrt_model" / "strict_arrt_inverse_summary.json"


def time_factor(pred: float, actual: float) -> float:
    pred = max(float(pred), 1e-9)
    actual = max(float(actual), 1e-9)
    return float(max(pred / actual, actual / pred))


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def near_time_match(candidate: dict[str, object], actual_t1_s: float, actual_t2_s: float, factor: float = 3.0) -> bool:
    return time_factor(float(candidate["t1_s"]), actual_t1_s) <= factor and time_factor(float(candidate["t2_s"]), actual_t2_s) <= factor


def _contains(stats: dict[str, object], value: float) -> bool:
    return float(stats["p5"]) <= float(value) <= float(stats["p95"])


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else math.nan


def _read_four_parameter_summary() -> dict[str, object]:
    if not FOUR_PARAMETER_SUMMARY_PATH.exists():
        reconstruct_strict_arrt()
    if FOUR_PARAMETER_SUMMARY_PATH.exists():
        return json.loads(FOUR_PARAMETER_SUMMARY_PATH.read_text(encoding="utf-8")).get("summary", {})
    return {}


def evaluate(top_k: int = 10) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_strict_arrt()
    all_rows = strict_paper_rows()
    rows = [row for row in all_rows if row.get("mode") == "two_step" and row.get("T2_C") not in {"", "nan", "NaN"}]
    if len(rows) < 1:
        raise RuntimeError("conditional time inverse found no two-step strict paper-style rows")
    if len(all_rows) < 3:
        raise RuntimeError("conditional time inverse needs at least three strict paper-style rows")

    condition_records: list[dict[str, object]] = []
    top_records: list[dict[str, object]] = []
    for test_idx, row in enumerate(rows):
        train_rows = [other for other in all_rows if other.get("condition_key") != row.get("condition_key")]
        model = fit_strict_paper_model(train_rows)
        target = {name: float(row[name]) for name in PAPER_STYLE_TARGETS}
        result = search(float(row["T1_C"]), float(row["T2_C"]), target, model=model, training_rows=train_rows, top_k=top_k)
        posterior = result["posterior"]
        actual_t1_s = float(row["t1_s"])
        actual_t2_s = float(row["t2_s"])
        actual_log_t1 = math.log10(actual_t1_s)
        actual_log_t2 = math.log10(actual_t2_s)
        best = result["point_estimates"][0]
        top5 = result["point_estimates"][:5]
        top10 = result["point_estimates"][:10]
        record: dict[str, object] = {
            "condition_key": row.get("condition_key", ""),
            "mode": row.get("mode", ""),
            "kissinger_r2": float(row["kissinger_r2"]) if row.get("kissinger_r2") not in {"", "nan", "NaN"} else math.nan,
            "kissinger_confidence": row.get("kissinger_confidence", ""),
            "T1_C": float(row["T1_C"]),
            "T2_C": float(row["T2_C"]),
            "actual_t1_s": actual_t1_s,
            "actual_t2_s": actual_t2_s,
            "pred_t1_s": float(best["t1_s"]),
            "pred_t2_s": float(best["t2_s"]),
            "abs_log_error_t1": abs(math.log10(float(best["t1_s"])) - actual_log_t1),
            "abs_log_error_t2": abs(math.log10(float(best["t2_s"])) - actual_log_t2),
            "t1_factor_error": time_factor(float(best["t1_s"]), actual_t1_s),
            "t2_factor_error": time_factor(float(best["t2_s"]), actual_t2_s),
            "top1_time_near_match": near_time_match(best, actual_t1_s, actual_t2_s),
            "top5_time_near_match": any(near_time_match(candidate, actual_t1_s, actual_t2_s) for candidate in top5),
            "top10_time_near_match": any(near_time_match(candidate, actual_t1_s, actual_t2_s) for candidate in top10),
            "posterior_contains_actual_t1": _contains(posterior["log10_t1_s"], actual_log_t1),
            "posterior_contains_actual_t2": _contains(posterior["log10_t2_s"], actual_log_t2),
            "posterior_width_log10_t1": float(posterior["log10_t1_s"]["width"]),
            "posterior_width_log10_t2": float(posterior["log10_t2_s"]["width"]),
            "posterior_n_valid_candidates": int(posterior["n_valid_candidates"]),
            "loss_flatness_ratio": float(posterior["loss_flatness_ratio"]),
            "best_loss": float(best["loss"]),
            "best_target_loss": float(best["target_loss"]),
            "interpretation_status": result["interpretation"]["status"],
        }
        for name in PAPER_STYLE_TARGETS:
            record[f"target_{name}"] = target[name]
            record[f"best_pred_{name}"] = float(best[f"pred_{name}"])
            record[f"best_error_{name}"] = float(best[f"error_{name}"])
        condition_records.append(record)

        for candidate in result["point_estimates"]:
            top_record = {
                "condition_key": row.get("condition_key", ""),
                "actual_t1_s": actual_t1_s,
                "actual_t2_s": actual_t2_s,
                **candidate,
            }
            top_records.append(top_record)

    four_param = _read_four_parameter_summary()
    conditional_summary = {
        "n_conditions": len(condition_records),
        "target_set": PAPER_STYLE_TARGETS,
        "evaluation_policy": "leave-one-condition-out strict measured paper-style model; T1/T2 fixed to the actual condition",
        "near_match_definition": "t1 and t2 each within a factor of 3 of the actual time.",
        "candidate_time_grid_s": "10,30,60,100,300,600,900,1200,1800",
        "t1_log10_MAE": _safe_mean([float(row["abs_log_error_t1"]) for row in condition_records]),
        "t2_log10_MAE": _safe_mean([float(row["abs_log_error_t2"]) for row in condition_records]),
        "t1_factor_error_median": _safe_median([float(row["t1_factor_error"]) for row in condition_records]),
        "t2_factor_error_median": _safe_median([float(row["t2_factor_error"]) for row in condition_records]),
        "top1_time_near_match": _safe_mean([float(bool(row["top1_time_near_match"])) for row in condition_records]),
        "top5_time_near_match": _safe_mean([float(bool(row["top5_time_near_match"])) for row in condition_records]),
        "top10_time_near_match": _safe_mean([float(bool(row["top10_time_near_match"])) for row in condition_records]),
        "posterior_t1_coverage": _safe_mean([float(bool(row["posterior_contains_actual_t1"])) for row in condition_records]),
        "posterior_t2_coverage": _safe_mean([float(bool(row["posterior_contains_actual_t2"])) for row in condition_records]),
        "posterior_width_log10_t1_median": _safe_median([float(row["posterior_width_log10_t1"]) for row in condition_records]),
        "posterior_width_log10_t2_median": _safe_median([float(row["posterior_width_log10_t2"]) for row in condition_records]),
        "interpretation_counts": {
            status: sum(1 for row in condition_records if row["interpretation_status"] == status)
            for status in sorted({str(row["interpretation_status"]) for row in condition_records})
        },
        "four_parameter_strict_arrt_reference": four_param,
        "comparison_to_four_parameter": {
            "four_parameter_log10_t1_MAE": four_param.get("best_mae", {}).get("log10_t1_s", math.nan) if isinstance(four_param.get("best_mae", {}), dict) else math.nan,
            "four_parameter_log10_t2_MAE": four_param.get("best_mae", {}).get("log10_t2_s", math.nan) if isinstance(four_param.get("best_mae", {}), dict) else math.nan,
        },
        "outputs": {
            "by_condition": str(BY_CONDITION_PATH.relative_to(ROOT)),
            "top_candidates": str(TOP_CANDIDATES_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
        "notes": [
            "Strict measured H*/S* is the primary target set.",
            "Proxy H*/S* is not mixed into the main conditional inverse model.",
            "High posterior coverage with wide intervals should be read as weak identifiability, not high precision.",
        ],
    }
    _augment_comparison(conditional_summary)
    write_csv(BY_CONDITION_PATH, condition_records)
    write_csv(TOP_CANDIDATES_PATH, top_records)
    SUMMARY_PATH.write_text(json.dumps({"summary": conditional_summary, "by_condition": condition_records}, indent=2), encoding="utf-8")
    write_report(conditional_summary, condition_records)
    print(json.dumps({"output": str(SUMMARY_PATH), "summary": conditional_summary}, indent=2))
    return SUMMARY_PATH


def _augment_comparison(summary: dict[str, object]) -> None:
    comparison = summary["comparison_to_four_parameter"]
    if not isinstance(comparison, dict):
        return
    four_t1 = comparison.get("four_parameter_log10_t1_MAE", math.nan)
    four_t2 = comparison.get("four_parameter_log10_t2_MAE", math.nan)
    if finite(four_t1) and float(four_t1) > 0:
        comparison["t1_MAE_ratio_conditional_over_four_parameter"] = float(summary["t1_log10_MAE"]) / float(four_t1)
    if finite(four_t2) and float(four_t2) > 0:
        comparison["t2_MAE_ratio_conditional_over_four_parameter"] = float(summary["t2_log10_MAE"]) / float(four_t2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_report(summary: dict[str, object], condition_records: list[dict[str, object]]) -> None:
    comparison = summary.get("comparison_to_four_parameter", {})
    rows = [
        "| Condition | T1->T2 (C) | actual t1/t2 (s) | pred t1/t2 (s) | log errors | top5 near | posterior widths | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in condition_records:
        rows.append(
            "| "
            + " | ".join([
                str(row["condition_key"]),
                f"{_fmt(row['T1_C'], 0)}->{_fmt(row['T2_C'], 0)}",
                f"{_fmt(row['actual_t1_s'], 0)}/{_fmt(row['actual_t2_s'], 0)}",
                f"{_fmt(row['pred_t1_s'], 0)}/{_fmt(row['pred_t2_s'], 0)}",
                f"{_fmt(row['abs_log_error_t1'])}/{_fmt(row['abs_log_error_t2'])}",
                str(bool(row["top5_time_near_match"])),
                f"{_fmt(row['posterior_width_log10_t1'])}/{_fmt(row['posterior_width_log10_t2'])}",
                str(row["interpretation_status"]),
            ])
            + " |"
        )

    text = f"""# PS Conditional Time Inverse

## Purpose

This report evaluates the reduced inverse problem:

```text
given T1, T2 and paper-style targets -> reconstruct t1, t2
```

The target set is strictly measured or strict-paper-style `delta_h_total_J_g`, `peak_area_J_g`, `H*`, and `S*`. Proxy `H*/S*` is kept out of the main model.

## Summary Metrics

| Metric | Value |
| --- | ---: |
| n conditions | {_fmt(summary['n_conditions'], 0)} |
| t1 log10 MAE | {_fmt(summary['t1_log10_MAE'])} |
| t2 log10 MAE | {_fmt(summary['t2_log10_MAE'])} |
| median t1 factor error | {_fmt(summary['t1_factor_error_median'])} |
| median t2 factor error | {_fmt(summary['t2_factor_error_median'])} |
| top1 time near-match | {_fmt(summary['top1_time_near_match'])} |
| top5 time near-match | {_fmt(summary['top5_time_near_match'])} |
| top10 time near-match | {_fmt(summary['top10_time_near_match'])} |
| posterior t1 coverage | {_fmt(summary['posterior_t1_coverage'])} |
| posterior t2 coverage | {_fmt(summary['posterior_t2_coverage'])} |
| median posterior width log10 t1 | {_fmt(summary['posterior_width_log10_t1_median'])} |
| median posterior width log10 t2 | {_fmt(summary['posterior_width_log10_t2_median'])} |

## Four-Parameter Reference

| Metric | Four-parameter strict ARRT | Fixed T1/T2 conditional |
| --- | ---: | ---: |
| log10 t1 MAE | {_fmt(comparison.get('four_parameter_log10_t1_MAE'))} | {_fmt(summary['t1_log10_MAE'])} |
| log10 t2 MAE | {_fmt(comparison.get('four_parameter_log10_t2_MAE'))} | {_fmt(summary['t2_log10_MAE'])} |
| t1 conditional/four-param ratio | {_fmt(comparison.get('t1_MAE_ratio_conditional_over_four_parameter'))} |  |
| t2 conditional/four-param ratio | {_fmt(comparison.get('t2_MAE_ratio_conditional_over_four_parameter'))} |  |

## By Condition

{chr(10).join(rows)}

## Interpretation

Fixing `T1` and `T2` removes the largest source of many-to-one ambiguity, so the remaining test is whether the paper-style target vector contains enough information to locate `t1` and `t2`. The result should be judged by both point error and posterior width. If a condition has high coverage but broad posterior intervals, the model is still not uniquely identifying the time pair.

## Outputs

- Summary: `results/ps/conditional_time_inverse/conditional_time_inverse_summary.json`
- By-condition CSV: `results/ps/conditional_time_inverse/conditional_time_inverse_by_condition.csv`
- Top candidates CSV: `results/ps/conditional_time_inverse/conditional_time_inverse_top_candidates.csv`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
