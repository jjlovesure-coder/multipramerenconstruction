"""Report the PS model update after adding kovacs-03/04 training batches."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "ps" / "evaluation"
FIG_DIR = ROOT / "results" / "ps" / "figures"
DOCS_DIR = ROOT / "docs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = EVAL_DIR / "ps_train_test_metrics_before_kovacs03_04.json"
UPDATED_PATH = EVAL_DIR / "ps_train_test_metrics.json"
DATASET_PATH = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"
EIG_PATH = ROOT / "data" / "ps" / "ps_eig_next_twostep_experiments.csv"
ARRT_SUMMARY_PATH = ROOT / "results" / "ps" / "arrt_kissinger" / "summary.json"
REPORT_PATH = DOCS_DIR / "ps_kovacs_03_04_training_update.md"
MAE_FIG_PATH = FIG_DIR / "ps_kovacs_03_04_mae_comparison.png"
REL_FIG_PATH = FIG_DIR / "ps_kovacs_03_04_mae_relative_change.png"
SOURCE_FIG_PATH = FIG_DIR / "ps_kovacs_03_04_sample_sources.png"


CORE_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "peak_temperature_Tp_C",
    "peak_height_uW",
    "recovery_index",
    "path_dependence_index",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:+.1f}%"


def metric_table(before: dict, after: dict) -> list[dict[str, object]]:
    rows = []
    for target in CORE_TARGETS:
        b = before["metrics"][target]["mae"]
        a = after["metrics"][target]["mae"]
        rel = (a - b) / b if b else float("nan")
        rows.append(
            {
                "target": target,
                "before_mae": b,
                "after_mae": a,
                "relative_change": rel,
                "after_rmse": after["metrics"][target]["rmse"],
                "after_r2": after["metrics"][target]["r2"],
            }
        )
    return rows


def summarize_dataset(rows: list[dict[str, str]]) -> tuple[Counter, Counter, dict[str, Counter]]:
    by_source = Counter(row["source_file"] for row in rows)
    by_mode = Counter(row["mode"] for row in rows)
    by_mode_temp: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        temp = f"{float(row['T1_C']):.0f}"
        by_mode_temp[row["mode"]][temp] += 1
    return by_source, by_mode, by_mode_temp


def write_figures(metrics: list[dict[str, object]], by_source: Counter) -> None:
    labels = [str(row["target"]).replace("_", "\n") for row in metrics]
    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - width / 2 for i in x], [float(row["before_mae"]) for row in metrics], width, label="before")
    ax.bar([i + width / 2 for i in x], [float(row["after_mae"]) for row in metrics], width, label="after")
    ax.set_ylabel("MAE")
    ax.set_title("PS predictor MAE before/after adding ps-kovacs-03/04")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(MAE_FIG_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    rel_values = [float(row["relative_change"]) * 100.0 for row in metrics]
    colors = ["#2ca02c" if value < 0 else "#d62728" for value in rel_values]
    ax.bar(labels, rel_values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE relative change (%)")
    ax.set_title("Prediction error change after adding ps-kovacs-03/04")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REL_FIG_PATH, dpi=180)
    plt.close(fig)

    top_sources = by_source.most_common()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh([name for name, _ in reversed(top_sources)], [count for _, count in reversed(top_sources)])
    ax.set_xlabel("usable feature rows")
    ax.set_title("Processed PS DSC samples by source file")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SOURCE_FIG_PATH, dpi=180)
    plt.close(fig)


def top_eig_rows(limit: int = 8) -> list[dict[str, str]]:
    rows = load_rows(EIG_PATH)
    return rows[:limit]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    before = load_json(BASELINE_PATH)
    after = load_json(UPDATED_PATH)
    dataset = load_rows(DATASET_PATH)
    metrics = metric_table(before, after)
    by_source, by_mode, by_mode_temp = summarize_dataset(dataset)
    write_figures(metrics, by_source)

    arrt = load_json(ARRT_SUMMARY_PATH) if ARRT_SUMMARY_PATH.exists() else {}
    eig_rows = top_eig_rows()

    metric_rows = [
        [
            row["target"],
            fmt(float(row["before_mae"])),
            fmt(float(row["after_mae"])),
            pct(float(row["relative_change"])),
            fmt(float(row["after_rmse"])),
            fmt(float(row["after_r2"])),
        ]
        for row in metrics
    ]
    source_rows = [[source, str(count)] for source, count in by_source.most_common()]
    eig_table_rows = [
        [
            row["rank"],
            row["T1_C"],
            row["t1_s"],
            row["T2_C"],
            row["t2_s"],
            fmt(float(row["eig_score"]), 3),
            fmt(float(row["pred_recovery_index"]), 3),
            fmt(float(row["uncertainty_recovery_index"]), 3),
        ]
        for row in eig_rows
    ]

    single_temps = ", ".join(f"{k} C: {v}" for k, v in sorted(by_mode_temp["single_step"].items(), key=lambda x: float(x[0])))
    two_temps = ", ".join(f"{k} C: {v}" for k, v in sorted(by_mode_temp["two_step"].items(), key=lambda x: float(x[0])))

    report = f"""# PS Kovacs 03/04 Training Update

## Purpose

This update adds the completed DSC training batches `ps-kovacs-03.xlsx` and `ps-kovacs-04.xlsx` to the existing finite-time PS enthalpy-recovery pipeline, retrains the surrogate predictor, and refreshes the EIG-based next-experiment ranking.

The comparison below keeps the model structure, targets, grouped split rule, sample mass (`4.7 mg`), and feature extraction logic unchanged. Therefore the change mainly reflects the extra training information from the new DSC batches.

## Dataset Growth

- Before adding `ps-kovacs-03/04`: {before["n_samples"]} usable samples, {before["n_train"]} train, {before["n_test"]} test.
- After adding `ps-kovacs-03/04`: {after["n_samples"]} usable samples, {after["n_train"]} train, {after["n_test"]} test.
- Net increase: {after["n_samples"] - before["n_samples"]} usable samples.
- Mode balance after update: single-step = {by_mode.get("single_step", 0)}, two-step = {by_mode.get("two_step", 0)}.
- Single-step T1 coverage: {single_temps}.
- Two-step T1 coverage: {two_temps}.

{markdown_table(["source_file", "usable_rows"], source_rows)}

![Processed samples by source file]({SOURCE_FIG_PATH.as_posix()})

## Prediction Improvement

Positive relative change means MAE became worse; negative means MAE improved.

{markdown_table(["target", "before_MAE", "after_MAE", "relative_change", "after_RMSE", "after_R2"], metric_rows)}

![MAE comparison]({MAE_FIG_PATH.as_posix()})

![MAE relative change]({REL_FIG_PATH.as_posix()})

## Interpretation

The added batches improved the most useful thermodynamic targets for inverse annealing design:

- `delta_h_total_J_g` MAE decreased from {fmt(before["metrics"]["delta_h_total_J_g"]["mae"])} to {fmt(after["metrics"]["delta_h_total_J_g"]["mae"])}.
- `recovery_index` MAE decreased from {fmt(before["metrics"]["recovery_index"]["mae"])} to {fmt(after["metrics"]["recovery_index"]["mae"])}.
- `peak_height_uW` also improved, which suggests the new batches help constrain signal-amplitude behavior.

The `Tp` MAE stayed essentially unchanged and slightly worsened by this split. This is not yet alarming because all extracted PS relaxation peaks remain clustered near about 105 C; the target has a narrow observed range, so small split changes can dominate the metric. The current data improve recovery-amplitude prediction more than peak-position prediction.

`path_dependence_index` changed only marginally. This means the new two-step batches add useful enthalpy-recovery information, but the model still has limited direct evidence for separating true path dependence from matched single-step behavior.

## EIG-Based Next Two-Step Recommendations

The refreshed EIG ranking still emphasizes uncertain/high-value two-step conditions, especially high-temperature or short-first-step combinations where the current model has relatively high recovery/path-dependence uncertainty.

{markdown_table(["rank", "T1_C", "t1_s", "T2_C", "t2_s", "EIG_score", "pred_recovery_index", "unc_recovery_index"], eig_table_rows)}

## ARRT / S* / H* Status

Strict ARRT/Kissinger calculation still cannot produce direct `S*` and `H*` training targets from the current PS dataset:

- total recovery scans inspected: {arrt.get("n_total_scans", "n/a")}
- annealed-state groups: {arrt.get("n_condition_groups", "n/a")}
- calculable groups: {arrt.get("n_calculable_groups", "n/a")}
- maximum distinct heating rates in one group: {arrt.get("max_distinct_heating_rates_per_group", "n/a")}
- minimum required distinct heating rates: {arrt.get("minimum_required_distinct_heating_rates", "n/a")}

So the direct paper-style target route is still blocked by insufficient repeated heating-rate coverage for the same annealed state. The next improvement should either keep using physics-informed surrogate features, or deliberately add at least three heating rates for selected identical annealing conditions.

## Assessment

The prediction gain is real but modest. The main model now estimates recovery degree better (`recovery_index` MAE about {fmt(after["metrics"]["recovery_index"]["mae"])}), which is the target most aligned with finite-time PS experiment design. However, the model is still not strong enough for precise inverse four-parameter prediction across the full 50-100 C space.

Recommended next step: use the updated EIG list to add a smaller, high-information validation batch, but include repeated conditions or controlled heating-rate triplets if the goal is to unlock direct `S* / H*` training.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "figures": [str(MAE_FIG_PATH), str(SOURCE_FIG_PATH)]}, indent=2))


if __name__ == "__main__":
    main()
