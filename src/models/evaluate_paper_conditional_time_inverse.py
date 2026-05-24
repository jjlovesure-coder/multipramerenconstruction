"""Validate paper-style conditional t1/t2 inverse on digitized paper data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable
from functools import lru_cache

import numpy as np

from src.models.dataset import PROCESSED, build_dataset
from src.models.inverse_design import feature_row
from src.models.kernel_regression import KernelRegressor
from src.models.train_forward import FEATURES


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper_conditional_time_inverse"
DOC_PATH = ROOT / "docs" / "paper_conditional_time_inverse_validation.md"

FIG2_TARGETS = [
    "delta_h_kj_mol",
    "delta_h_peak_kj_mol",
]
FIG2_FIG3_TARGETS = [
    "delta_h_kj_mol",
    "delta_h_peak_kj_mol",
    "s1_j_mol_k",
    "s2_j_mol_k",
    "delta_s_j_mol_k",
    "h1_kj_mol_fig3",
    "h2_kj_mol_fig3",
]
MODEL_PREDICTED_TARGETS = ["delta_h_kj_mol", "delta_h_peak_kj_mol"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dataset() -> list[dict[str, str]]:
    path = PROCESSED / "annealing_dataset.csv"
    if not path.exists():
        build_dataset()
    return read_rows(path)


def unique_times(rows: Iterable[dict[str, str]]) -> list[float]:
    values = set()
    for row in rows:
        values.add(float(row["t1_s"]))
        values.add(float(row["t2_s"]))
    return sorted(values)


def candidate_rows_for_fixed_temperatures(t1_temperature_k: float, t2_temperature_k: float, times: list[float]) -> list[dict[str, float]]:
    return [
        {
            "t1_temperature_k": float(t1_temperature_k),
            "t2_temperature_k": float(t2_temperature_k),
            "t1_s": float(t1_s),
            "t2_s": float(t2_s),
        }
        for t1_s in times
        for t2_s in times
    ]


@lru_cache(maxsize=None)
def _candidate_observables_cached(t1_temperature_k: float, t1_s: float, t2_temperature_k: float, t2_s: float) -> tuple[tuple[str, float], ...]:
    features = feature_row(t1_temperature_k, t1_s, t2_temperature_k, t2_s)
    values = dict(zip(FEATURES, features))
    values["delta_s_j_mol_k"] = float(values["s2_j_mol_k"]) - float(values["s1_j_mol_k"])
    values.update({
        "t1_temperature_k": float(t1_temperature_k),
        "t2_temperature_k": float(t2_temperature_k),
        "t1_s": float(t1_s),
        "t2_s": float(t2_s),
    })
    return tuple(sorted((key, float(value)) for key, value in values.items()))


def candidate_observables(row: dict[str, float]) -> dict[str, float]:
    return dict(_candidate_observables_cached(row["t1_temperature_k"], row["t1_s"], row["t2_temperature_k"], row["t2_s"]))


def fit_delta_peak_model(rows: list[dict[str, str]], train_indices: list[int]) -> KernelRegressor:
    x = np.array([[float(rows[i][feature]) for feature in FEATURES] for i in train_indices], dtype=float)
    y = np.array([[float(rows[i][target]) for target in MODEL_PREDICTED_TARGETS] for i in train_indices], dtype=float)
    return KernelRegressor.fit(x, y, bandwidth=1.25)


def scales_for_targets(rows: list[dict[str, str]], targets: list[str]) -> dict[str, float]:
    floors = {
        "delta_h_kj_mol": 0.04,
        "delta_h_peak_kj_mol": 0.02,
        "s1_j_mol_k": 10.0,
        "s2_j_mol_k": 10.0,
        "delta_s_j_mol_k": 10.0,
        "h1_kj_mol_fig3": 20.0,
        "h2_kj_mol_fig3": 20.0,
    }
    out = {}
    for target in targets:
        values = np.array([float(row[target]) for row in rows], dtype=float)
        out[target] = max(float(np.std(values)), floors[target])
    return out


def time_posterior_summary(rows: list[dict[str, float]], losses: Iterable[float], percentile: float = 5.0) -> dict[str, object]:
    losses_arr = np.asarray(list(losses), dtype=float)
    if len(rows) == 0:
        return {"n_valid_candidates": 0}
    threshold = float(np.percentile(losses_arr, percentile))
    indices = np.where(losses_arr <= threshold)[0]
    if len(indices) == 0:
        indices = np.array([int(np.argmin(losses_arr))])
    out: dict[str, object] = {
        "percentile": percentile,
        "n_valid_candidates": int(len(indices)),
        "loss_min": float(np.min(losses_arr)),
        "loss_threshold": threshold,
    }
    for name in ["t1_s", "t2_s"]:
        values = np.array([math.log10(max(float(rows[int(idx)][name]), 1e-12)) for idx in indices], dtype=float)
        key = "log10_t1_s" if name == "t1_s" else "log10_t2_s"
        out[key] = {
            "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "width": float(np.percentile(values, 95) - np.percentile(values, 5)),
        }
    return out


def evaluate_variant(rows: list[dict[str, str]], targets: list[str], label: str) -> dict[str, object]:
    times = unique_times(rows)
    scales = scales_for_targets(rows, targets)
    records = []
    for test_idx, actual in enumerate(rows):
        train_indices = [idx for idx in range(len(rows)) if idx != test_idx]
        model = fit_delta_peak_model(rows, train_indices)
        candidates = candidate_rows_for_fixed_temperatures(
            float(actual["t1_temperature_k"]),
            float(actual["t2_temperature_k"]),
            times,
        )
        candidate_features = []
        candidate_values = []
        for candidate in candidates:
            values = candidate_observables(candidate)
            candidate_values.append(values)
            candidate_features.append([values[feature] for feature in FEATURES])
        pred, unc = model.predict(np.array(candidate_features, dtype=float))
        losses = []
        per_target_losses: list[dict[str, float]] = []
        for i, values in enumerate(candidate_values):
            loss_parts = {}
            for target in targets:
                if target in MODEL_PREDICTED_TARGETS:
                    pred_value = float(pred[i, MODEL_PREDICTED_TARGETS.index(target)])
                else:
                    pred_value = float(values[target])
                loss_parts[target] = abs((pred_value - float(actual[target])) / scales[target])
            per_target_losses.append(loss_parts)
            losses.append(float(np.mean(list(loss_parts.values()))))
        losses_arr = np.array(losses, dtype=float)
        order = np.argsort(losses_arr)
        best_idx = int(order[0])
        top5 = [int(idx) for idx in order[:5]]
        top10 = [int(idx) for idx in order[:10]]
        actual_t1 = float(actual["t1_s"])
        actual_t2 = float(actual["t2_s"])

        def near(indices: list[int]) -> bool:
            for idx in indices:
                candidate = candidates[idx]
                if max(float(candidate["t1_s"]) / actual_t1, actual_t1 / float(candidate["t1_s"])) <= 3.0 and max(
                    float(candidate["t2_s"]) / actual_t2,
                    actual_t2 / float(candidate["t2_s"]),
                ) <= 3.0:
                    return True
            return False

        posterior = time_posterior_summary(candidates, losses_arr)
        best = candidates[best_idx]
        records.append({
            "variant": label,
            "sample_id": test_idx,
            "panel": actual["panel"],
            "T1_K": float(actual["t1_temperature_k"]),
            "T2_K": float(actual["t2_temperature_k"]),
            "actual_t1_s": actual_t1,
            "actual_t2_s": actual_t2,
            "pred_t1_s": float(best["t1_s"]),
            "pred_t2_s": float(best["t2_s"]),
            "t1_log10_abs_error": abs(math.log10(float(best["t1_s"]) / actual_t1)),
            "t2_log10_abs_error": abs(math.log10(float(best["t2_s"]) / actual_t2)),
            "t1_factor_error": max(float(best["t1_s"]) / actual_t1, actual_t1 / float(best["t1_s"])),
            "t2_factor_error": max(float(best["t2_s"]) / actual_t2, actual_t2 / float(best["t2_s"])),
            "top1_time_near_match": near([best_idx]),
            "top5_time_near_match": near(top5),
            "top10_time_near_match": near(top10),
            "loss": float(losses_arr[best_idx]),
            "posterior_log10_t1_width": posterior["log10_t1_s"]["width"],  # type: ignore[index]
            "posterior_log10_t2_width": posterior["log10_t2_s"]["width"],  # type: ignore[index]
            "posterior_contains_actual_t1": bool(
                posterior["log10_t1_s"]["p5"] <= math.log10(actual_t1) <= posterior["log10_t1_s"]["p95"]  # type: ignore[index]
            ),
            "posterior_contains_actual_t2": bool(
                posterior["log10_t2_s"]["p5"] <= math.log10(actual_t2) <= posterior["log10_t2_s"]["p95"]  # type: ignore[index]
            ),
            "target_loss_components": per_target_losses[best_idx],
            "mean_delta_peak_uncertainty": float(np.mean(unc[best_idx])),
        })
    return summarize_records(records, targets)


def summarize_records(records: list[dict[str, object]], targets: list[str]) -> dict[str, object]:
    def mean_bool(name: str) -> float:
        return float(np.mean([1.0 if row[name] else 0.0 for row in records]))

    panel_summary = {}
    for panel in sorted({str(row["panel"]) for row in records}):
        selected = [row for row in records if row["panel"] == panel]
        panel_summary[panel] = {
            "n": len(selected),
            "t1_log10_MAE": float(np.mean([float(row["t1_log10_abs_error"]) for row in selected])),
            "t2_log10_MAE": float(np.mean([float(row["t2_log10_abs_error"]) for row in selected])),
            "top5_time_near_match": float(np.mean([1.0 if row["top5_time_near_match"] else 0.0 for row in selected])),
            "posterior_log10_t1_width_mean": float(np.mean([float(row["posterior_log10_t1_width"]) for row in selected])),
            "posterior_log10_t2_width_mean": float(np.mean([float(row["posterior_log10_t2_width"]) for row in selected])),
        }
    return {
        "targets": targets,
        "n_samples": len(records),
        "overall": {
            "t1_log10_MAE": float(np.mean([float(row["t1_log10_abs_error"]) for row in records])),
            "t2_log10_MAE": float(np.mean([float(row["t2_log10_abs_error"]) for row in records])),
            "t1_factor_error_median": float(median([float(row["t1_factor_error"]) for row in records])),
            "t2_factor_error_median": float(median([float(row["t2_factor_error"]) for row in records])),
            "top1_time_near_match": mean_bool("top1_time_near_match"),
            "top5_time_near_match": mean_bool("top5_time_near_match"),
            "top10_time_near_match": mean_bool("top10_time_near_match"),
            "posterior_t1_coverage": mean_bool("posterior_contains_actual_t1"),
            "posterior_t2_coverage": mean_bool("posterior_contains_actual_t2"),
            "posterior_log10_t1_width_mean": float(np.mean([float(row["posterior_log10_t1_width"]) for row in records])),
            "posterior_log10_t2_width_mean": float(np.mean([float(row["posterior_log10_t2_width"]) for row in records])),
        },
        "by_panel": panel_summary,
        "records": records,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0] if key != "target_loss_components"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def write_report(summary: dict[str, object]) -> None:
    variants: dict[str, object] = summary["variants"]  # type: ignore[assignment]
    lines = [
        "# Paper-Style Conditional t1/t2 Inverse Validation",
        "",
        "## Purpose",
        "",
        "This validation fixes the original-paper temperature pair and asks whether time can be recovered from paper-style observables. `delta_h` and `delta_h_peak` are matched through a leave-one-out forward kernel, while Figure-3-like kinetic constraints are matched directly from the candidate time coordinates.",
        "",
        "## Overall Metrics",
        "",
        "| Variant | Targets | t1 log10 MAE | t2 log10 MAE | Top5 near match | Posterior width t1 | Posterior width t2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, block in variants.items():  # type: ignore[union-attr]
        overall = block["overall"]
        lines.append(
            f"| {name} | {', '.join(block['targets'])} | {overall['t1_log10_MAE']:.3f} | {overall['t2_log10_MAE']:.3f} | {overall['top5_time_near_match']:.3f} | {overall['posterior_log10_t1_width_mean']:.3f} | {overall['posterior_log10_t2_width_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The comparison is the benchmark for the PS 80/90 question. If adding H*/S* style kinetic information sharply narrows the posterior on the digitized paper data, then PS 80/90 needs the same kind of multi-rate ARRT labels rather than only more single-heating DSC curves.",
        "",
        "Outputs:",
        "",
        "- `results/paper_conditional_time_inverse/paper_conditional_time_inverse_summary.json`",
        "- `results/paper_conditional_time_inverse/paper_conditional_time_inverse_by_sample.csv`",
    ])
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = ensure_dataset()
    variants = {
        "fig2_only": evaluate_variant(rows, FIG2_TARGETS, "fig2_only"),
        "fig2_plus_fig3_kinetics": evaluate_variant(rows, FIG2_FIG3_TARGETS, "fig2_plus_fig3_kinetics"),
    }
    all_records = []
    for block in variants.values():
        all_records.extend(block["records"])  # type: ignore[index]
    summary = {
        "source_dataset": "data/processed/annealing_dataset.csv",
        "validation_mode": "leave-one-out, fixed T1/T2, search t1/t2 only",
        "near_match_definition": "both t1 and t2 within factor 3 of actual",
        "variants": variants,
        "notes": [
            "Figure 3b/c style information is represented by digitized/interpolated H* and S* fields available in the processed paper dataset.",
            "Digitized overlap-filled Figure 2 points are approximate; this is a feasibility benchmark, not an exact reproduction of the paper tables.",
        ],
    }
    out = OUT_DIR / "paper_conditional_time_inverse_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(OUT_DIR / "paper_conditional_time_inverse_by_sample.csv", all_records)
    write_report(summary)
    print(json.dumps({
        "output": str(out),
        "fig2_only": variants["fig2_only"]["overall"],
        "fig2_plus_fig3_kinetics": variants["fig2_plus_fig3_kinetics"]["overall"],
    }, indent=2))
    return out


if __name__ == "__main__":
    main()
