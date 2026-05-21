"""Evaluate TNM conditional t1/t2 inverse reconstruction with fixed T1/T2."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.ps.conditional_time_inverse import conditional_posterior_summary
from src.ps.dsc_processing import ROOT
from src.ps.tnm_calibrated_model import MODEL_PATH, TARGETS, design_matrix, predict_with_head, target_matrix, train as train_tnm
from src.ps.tnm_inverse_reconstruction import TARGET_SCALES, load_tnm_payload
from src.ps.train_ps_model import read_rows


OUT_DIR = ROOT / "results" / "ps" / "conditional_tnm_time_inverse"
SUMMARY_PATH = OUT_DIR / "conditional_tnm_time_inverse_summary.json"
BY_SAMPLE_PATH = OUT_DIR / "conditional_tnm_time_inverse_by_sample.csv"
BY_PAIR_PATH = OUT_DIR / "conditional_tnm_time_inverse_by_pair.csv"
REPORT_PATH = ROOT / "docs" / "ps_conditional_tnm_time_inverse_comparison.md"
DATASET_PATH = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"

DEFAULT_EXPERIMENT_TIMES_S = [10, 30, 60, 100, 300, 600, 900, 1200, 1800]


def _fmt_number(value: float) -> str:
    return f"{float(value):.6g}"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def candidate_rows_for_fixed_temperatures(T1_C: float, T2_C: float, time_grid_s: Iterable[float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    times = sorted({float(value) for value in time_grid_s if finite(value) and float(value) > 0.0})
    for t1_s in times:
        for t2_s in times:
            rows.append({
                "mode": "two_step",
                "T1_C": _fmt_number(T1_C),
                "T2_C": _fmt_number(T2_C),
                "t1_s": _fmt_number(t1_s),
                "t2_s": _fmt_number(t2_s),
                "total_anneal_time_s": _fmt_number(t1_s + t2_s),
            })
    return rows


def observed_time_grid(rows: list[dict[str, str]]) -> list[float]:
    values = set(float(value) for value in DEFAULT_EXPERIMENT_TIMES_S)
    for row in rows:
        if row.get("mode") != "two_step":
            continue
        values.add(float(row["t1_s"]))
        values.add(float(row["t2_s"]))
    return sorted(value for value in values if value > 0.0)


def tnm_target_loss(pred: np.ndarray, target: np.ndarray) -> float:
    loss = 0.0
    for idx, name in enumerate(TARGETS):
        loss += abs(float(pred[idx] - target[idx])) / TARGET_SCALES[name]
    return float(loss / len(TARGETS))


def time_factor(pred: float, actual: float) -> float:
    pred = max(float(pred), 1e-12)
    actual = max(float(actual), 1e-12)
    return float(max(pred / actual, actual / pred))


def near_time_match(candidate: dict[str, object], actual_t1_s: float, actual_t2_s: float, factor: float = 3.0) -> bool:
    return time_factor(float(candidate["t1_s"]), actual_t1_s) <= factor and time_factor(float(candidate["t2_s"]), actual_t2_s) <= factor


def _contains(stats: dict[str, object], value: float) -> bool:
    return float(stats["p5"]) <= float(value) <= float(stats["p95"])


def rank_candidates(
    candidates: list[dict[str, str]],
    predictions: np.ndarray,
    target: np.ndarray,
    top_k: int = 10,
) -> tuple[list[dict[str, object]], np.ndarray]:
    losses = np.array([tnm_target_loss(pred, target) for pred in predictions], dtype=float)
    order = np.argsort(losses)[:top_k]
    ranked: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        row: dict[str, object] = dict(candidates[int(idx)])
        row["rank"] = rank
        row["loss"] = float(losses[int(idx)])
        row["target_loss"] = float(losses[int(idx)])
        for target_idx, name in enumerate(TARGETS):
            row[f"target_{name}"] = float(target[target_idx])
            row[f"pred_{name}"] = float(predictions[int(idx), target_idx])
            row[f"error_{name}"] = float(predictions[int(idx), target_idx] - target[target_idx])
        ranked.append(row)
    return ranked, losses


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else math.nan


def summarize_by_pair(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair"])].append(row)
    summary: list[dict[str, object]] = []
    for pair, values in sorted(grouped.items()):
        summary.append({
            "pair": pair,
            "n": len(values),
            "t1_log10_MAE": _mean([float(row["abs_log_error_t1"]) for row in values]),
            "t2_log10_MAE": _mean([float(row["abs_log_error_t2"]) for row in values]),
            "t1_factor_error_median": _median([float(row["t1_factor_error"]) for row in values]),
            "t2_factor_error_median": _median([float(row["t2_factor_error"]) for row in values]),
            "top1_time_near_match": _mean([float(bool(row.get("top1_time_near_match", False))) for row in values]),
            "top5_time_near_match": _mean([float(bool(row.get("top5_time_near_match", False))) for row in values]),
            "top10_time_near_match": _mean([float(bool(row.get("top10_time_near_match", False))) for row in values]),
            "posterior_t1_coverage": _mean([float(bool(row.get("posterior_contains_actual_t1", False))) for row in values]),
            "posterior_t2_coverage": _mean([float(bool(row.get("posterior_contains_actual_t2", False))) for row in values]),
            "posterior_width_log10_t1_median": _median([float(row["posterior_width_log10_t1"]) for row in values]),
            "posterior_width_log10_t2_median": _median([float(row["posterior_width_log10_t2"]) for row in values]),
        })
    return summary


def _read_conditional_paper_summary() -> dict[str, object]:
    path = ROOT / "results" / "ps" / "conditional_time_inverse" / "conditional_time_inverse_summary.json"
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT))}
    return json.loads(path.read_text(encoding="utf-8")).get("summary", {})


def _recommend_80_90_experiments(summary_by_pair: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {row["pair"]: row for row in summary_by_pair}
    recommendations: list[dict[str, object]] = []
    for pair in ["80->90", "90->80"]:
        current = lookup.get(pair, {})
        recommendations.extend([
            {
                "pair": pair,
                "T1_C": float(pair.split("->")[0]),
                "T2_C": float(pair.split("->")[1]),
                "t1_s": 10.0,
                "t2_s": 10.0,
                "purpose": "Add the missing low-low time anchor inside the 10-1800 s experimental window.",
                "current_pair_t2_log10_MAE": current.get("t2_log10_MAE", math.nan),
            },
            {
                "pair": pair,
                "T1_C": float(pair.split("->")[0]),
                "T2_C": float(pair.split("->")[1]),
                "t1_s": 300.0,
                "t2_s": 300.0,
                "purpose": "Add a balanced midpoint because current dense data mainly uses t1 near 50/500 s and many sub-10 s t2 values.",
                "current_pair_t2_log10_MAE": current.get("t2_log10_MAE", math.nan),
            },
            {
                "pair": pair,
                "T1_C": float(pair.split("->")[0]),
                "T2_C": float(pair.split("->")[1]),
                "t1_s": 900.0,
                "t2_s": 900.0,
                "purpose": "Add a long-long finite-time anchor for the practical 30 min window.",
                "current_pair_t2_log10_MAE": current.get("t2_log10_MAE", math.nan),
            },
            {
                "pair": pair,
                "T1_C": float(pair.split("->")[0]),
                "T2_C": float(pair.split("->")[1]),
                "t1_s": 100.0,
                "t2_s": 900.0,
                "purpose": "Separate step-2 dominated aging from total-time effects.",
                "current_pair_t2_log10_MAE": current.get("t2_log10_MAE", math.nan),
            },
            {
                "pair": pair,
                "T1_C": float(pair.split("->")[0]),
                "T2_C": float(pair.split("->")[1]),
                "t1_s": 900.0,
                "t2_s": 100.0,
                "purpose": "Separate step-1 dominated memory from step-2 aging.",
                "current_pair_t2_log10_MAE": current.get("t2_log10_MAE", math.nan),
            },
        ])
    return recommendations


def evaluate(top_k: int = 10) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        train_tnm()
    params, head, payload = load_tnm_payload()
    all_rows = read_rows(DATASET_PATH)
    rows = [row for row in all_rows if row.get("mode") == "two_step"]
    time_grid = observed_time_grid(rows)
    prediction_cache: dict[tuple[float, float], tuple[list[dict[str, str]], np.ndarray]] = {}
    records: list[dict[str, object]] = []
    top_records: list[dict[str, object]] = []
    for row, target in zip(rows, target_matrix(rows)):
        T1 = float(row["T1_C"])
        T2 = float(row["T2_C"])
        cache_key = (T1, T2)
        if cache_key not in prediction_cache:
            candidates = candidate_rows_for_fixed_temperatures(T1, T2, time_grid)
            prediction_cache[cache_key] = (candidates, predict_with_head(design_matrix(candidates, params), head))
        candidates, pred = prediction_cache[cache_key]
        ranked, losses = rank_candidates(candidates, pred, target, top_k=top_k)
        posterior = conditional_posterior_summary(candidates, losses)
        best = ranked[0]
        actual_t1_s = float(row["t1_s"])
        actual_t2_s = float(row["t2_s"])
        actual_log_t1 = math.log10(actual_t1_s)
        actual_log_t2 = math.log10(actual_t2_s)
        top5 = ranked[:5]
        top10 = ranked[:10]
        pair = f"{T1:.0f}->{T2:.0f}"
        record: dict[str, object] = {
            "sample_id": row.get("sample_id", ""),
            "source_file": row.get("source_file", ""),
            "pair": pair,
            "T1_C": T1,
            "T2_C": T2,
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
        }
        for name in TARGETS:
            record[f"target_{name}"] = float(best[f"target_{name}"])
            record[f"best_pred_{name}"] = float(best[f"pred_{name}"])
            record[f"best_error_{name}"] = float(best[f"error_{name}"])
        records.append(record)
        for candidate in ranked:
            top_records.append({"sample_id": row.get("sample_id", ""), "pair": pair, "actual_t1_s": actual_t1_s, "actual_t2_s": actual_t2_s, **candidate})

    by_pair = summarize_by_pair(records)
    recommendations = _recommend_80_90_experiments(by_pair)
    summary = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "selected_parameters": payload.get("selected_parameters", {}),
        "n_two_step_samples": len(records),
        "target_set": TARGETS,
        "evaluation_policy": "TNM calibrated forward model; T1/T2 fixed to actual values; t1/t2 searched on observed plus practical experiment time grid.",
        "candidate_time_grid_s": time_grid,
        "overall": {
            "t1_log10_MAE": _mean([float(row["abs_log_error_t1"]) for row in records]),
            "t2_log10_MAE": _mean([float(row["abs_log_error_t2"]) for row in records]),
            "t1_factor_error_median": _median([float(row["t1_factor_error"]) for row in records]),
            "t2_factor_error_median": _median([float(row["t2_factor_error"]) for row in records]),
            "top1_time_near_match": _mean([float(bool(row["top1_time_near_match"])) for row in records]),
            "top5_time_near_match": _mean([float(bool(row["top5_time_near_match"])) for row in records]),
            "top10_time_near_match": _mean([float(bool(row["top10_time_near_match"])) for row in records]),
            "posterior_t1_coverage": _mean([float(bool(row["posterior_contains_actual_t1"])) for row in records]),
            "posterior_t2_coverage": _mean([float(bool(row["posterior_contains_actual_t2"])) for row in records]),
            "posterior_width_log10_t1_median": _median([float(row["posterior_width_log10_t1"]) for row in records]),
            "posterior_width_log10_t2_median": _median([float(row["posterior_width_log10_t2"]) for row in records]),
        },
        "summary_by_pair": by_pair,
        "paper_style_conditional_reference": _read_conditional_paper_summary(),
        "recommended_80_90_additions": recommendations,
        "outputs": {
            "by_sample": str(BY_SAMPLE_PATH.relative_to(ROOT)),
            "by_pair": str(BY_PAIR_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    write_csv(BY_SAMPLE_PATH, records)
    write_csv(OUT_DIR / "conditional_tnm_time_inverse_top_candidates.csv", top_records)
    write_csv(BY_PAIR_PATH, by_pair)
    SUMMARY_PATH.write_text(json.dumps({"summary": summary, "by_sample": records}, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"output": str(SUMMARY_PATH), "overall": summary["overall"]}, indent=2))
    return SUMMARY_PATH


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
    return f"{number:.{digits}f}" if math.isfinite(number) else "nan"


def write_report(summary: dict[str, object]) -> None:
    overall = summary["overall"]
    paper = summary.get("paper_style_conditional_reference", {})
    pair_rows = ["| Pair | n | t1 log MAE | t2 log MAE | top5 near | posterior width t1/t2 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["summary_by_pair"]:
        pair_rows.append(
            f"| {row['pair']} | {row['n']} | {_fmt(row['t1_log10_MAE'])} | {_fmt(row['t2_log10_MAE'])} | {_fmt(row['top5_time_near_match'])} | {_fmt(row['posterior_width_log10_t1_median'])}/{_fmt(row['posterior_width_log10_t2_median'])} |"
        )
    rec_rows = ["| Pair | T1 | T2 | t1 | t2 | Purpose |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for rec in summary["recommended_80_90_additions"]:
        rec_rows.append(f"| {rec['pair']} | {_fmt(rec['T1_C'], 0)} | {_fmt(rec['T2_C'], 0)} | {_fmt(rec['t1_s'], 0)} | {_fmt(rec['t2_s'], 0)} | {rec['purpose']} |")

    text = f"""# TNM Conditional Time Inverse Comparison

## Summary

This evaluates the reduced inverse problem with the calibrated TNM forward model:

```text
given T1, T2 and final heating targets -> reconstruct t1, t2
```

Unlike the strict paper-style model, TNM uses the full two-step PS dataset and target set `{", ".join(TARGETS)}`. The dense 80-90 data include historical sub-10 s times, so this diagnostic uses the observed time grid plus the practical 10-1800 s grid.

## TNM Overall Metrics

| Metric | Value |
| --- | ---: |
| n two-step samples | {summary['n_two_step_samples']} |
| t1 log10 MAE | {_fmt(overall['t1_log10_MAE'])} |
| t2 log10 MAE | {_fmt(overall['t2_log10_MAE'])} |
| median t1 factor error | {_fmt(overall['t1_factor_error_median'])} |
| median t2 factor error | {_fmt(overall['t2_factor_error_median'])} |
| top1 time near-match | {_fmt(overall['top1_time_near_match'])} |
| top5 time near-match | {_fmt(overall['top5_time_near_match'])} |
| top10 time near-match | {_fmt(overall['top10_time_near_match'])} |
| posterior t1 coverage | {_fmt(overall['posterior_t1_coverage'])} |
| posterior t2 coverage | {_fmt(overall['posterior_t2_coverage'])} |
| posterior width log10 t1 median | {_fmt(overall['posterior_width_log10_t1_median'])} |
| posterior width log10 t2 median | {_fmt(overall['posterior_width_log10_t2_median'])} |

## Paper-Style Conditional Reference

The strict paper-style conditional inverse used only `{paper.get('n_conditions', 'missing')}` strict double-step H*/S* conditions. Its headline values were:

```json
{json.dumps({k: paper.get(k) for k in ['t1_log10_MAE', 't2_log10_MAE', 'top5_time_near_match', 'posterior_width_log10_t1_median', 'posterior_width_log10_t2_median']}, indent=2)}
```

These two evaluations are not perfectly apples-to-apples: TNM has many more two-step final-heating targets, while strict paper-style H*/S* has cleaner physics labels but only three double-step strict points.

## By Temperature Pair

{chr(10).join(pair_rows)}

## 80-90 Data Reflection

The densest pairs are `80->90` and `90->80`, but the current matrix is narrow: it mostly uses `t1` near 50 or 500 s and many `t2` values below 10 s. That is useful for fast Kovacs-style probing, but weak for the practical finite-time window `10-1800 s`.

Recommended additions:

{chr(10).join(rec_rows)}

## Outputs

- Summary: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_summary.json`
- By sample: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_by_sample.csv`
- By pair: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_by_pair.csv`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
