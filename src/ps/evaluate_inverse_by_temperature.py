"""Evaluate inverse recovery of two-step annealing parameters by temperature range."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ps.inverse_design import (
    candidate_feature_matrix,
    candidate_rows,
    combined_inverse_loss,
    load_inverse_diagnostics,
    load_model,
    load_model_payload,
    posterior_summary,
    t2_candidates_from_target,
)
from src.ps.train_ps_model import read_rows


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"
OUT_DIR = ROOT / "results" / "ps" / "inverse_temperature"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


ACTIVE_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "peak_temperature_Tp_C",
    "recovery_index",
    "path_dependence_index",
]

SCALES = {
    "delta_h_total_J_g": 0.22,
    "peak_area_J_g": 0.17,
    "peak_temperature_Tp_C": 5.0,
    "recovery_index": 0.19,
    "path_dependence_index": 0.24,
}


def as_float(value: str, default: float = math.nan) -> float:
    try:
        if value in {"", "nan", "NaN", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def t1_bin(temp: float) -> str:
    if 50 <= temp <= 60:
        return "50-60"
    if 65 <= temp <= 75:
        return "65-75"
    if 80 <= temp <= 90:
        return "80-90"
    if 95 <= temp <= 100:
        return "95-100"
    return "other"


def t2_bin(temp: float) -> str:
    if 50 <= temp <= 70:
        return "T2 50-70"
    if 75 <= temp <= 90:
        return "T2 75-90"
    if 95 <= temp <= 100:
        return "T2 95-100"
    return "T2 other"


def path_class(t1: float, t2: float) -> str:
    if abs(t1 - t2) <= 1e-6:
        return "isothermal"
    if t2 > t1:
        return "up-jump"
    return "down-jump"


def time_factor(pred: float, actual: float) -> float:
    pred = max(pred, 1e-9)
    actual = max(actual, 1e-9)
    ratio = max(pred / actual, actual / pred)
    return float(ratio)


def near_match(candidate: dict[str, str], actual: dict[str, str]) -> bool:
    t1_ok = abs(float(candidate["T1_C"]) - as_float(actual["T1_C"])) <= 5.0
    t2_ok = abs(float(candidate["T2_C"]) - as_float(actual["T2_C"])) <= 5.0
    tf1_ok = time_factor(float(candidate["t1_s"]), as_float(actual["t1_s"])) <= 3.0
    tf2_ok = time_factor(float(candidate["t2_s"]), as_float(actual["t2_s"])) <= 3.0
    return t1_ok and t2_ok and tf1_ok and tf2_ok


def build_candidate_predictions() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = load_model()
    payload = load_model_payload()
    candidates = [row for row in candidate_rows() if row["mode"] == "two_step"]
    x = candidate_feature_matrix(candidates, payload)
    pred, unc = model.predict(x)
    targets = list(payload["targets"])
    diagnostics = load_inverse_diagnostics(candidates, payload, targets)
    identifiability = diagnostics["inverse_identifiability"]
    extrapolation = diagnostics["extrapolation_penalty"]
    return candidates, pred, unc, identifiability, extrapolation


def inverse_rank(
    row: dict[str, str],
    candidates: list[dict[str, str]],
    pred: np.ndarray,
    unc: np.ndarray,
    identifiability: np.ndarray,
    extrapolation: np.ndarray,
    targets: list[str],
) -> dict:
    target_losses = np.zeros(len(candidates), dtype=float)
    for target in ACTIVE_TARGETS:
        if target not in targets:
            continue
        idx = targets.index(target)
        actual = as_float(row[target])
        target_losses += np.abs((pred[:, idx] - actual) / SCALES[target])
    target_losses /= max(sum(target in targets for target in ACTIVE_TARGETS), 1)
    allowed_t2 = set(t2_candidates_from_target({"peak_temperature_Tp_C": as_float(row["peak_temperature_Tp_C"])}))
    valid_mask = np.array([round(float(candidate["T2_C"])) in allowed_t2 for candidate in candidates], dtype=bool)
    losses = np.full(len(candidates), np.inf, dtype=float)
    for idx, candidate in enumerate(candidates):
        if not valid_mask[idx]:
            continue
        losses[idx] = combined_inverse_loss(
            target_loss=float(target_losses[idx]),
            total_time_s=float(candidate["total_anneal_time_s"]),
            mean_uncertainty=float(np.mean(unc[idx])),
            extrapolation_penalty=float(extrapolation[idx]),
            inverse_identifiability=float(identifiability[idx]),
        )
    order = np.argsort(losses)
    best_idx = int(order[0])
    best = candidates[best_idx]
    valid_candidates = [candidates[int(idx)] for idx in np.where(np.isfinite(losses))[0]]
    valid_losses = losses[np.isfinite(losses)]
    posterior = posterior_summary(valid_candidates, valid_losses)
    actual_t1 = as_float(row["T1_C"])
    actual_t2 = as_float(row["T2_C"])
    actual_time1 = as_float(row["t1_s"])
    actual_time2 = as_float(row["t2_s"])
    pred_t1 = float(best["T1_C"])
    pred_t2 = float(best["T2_C"])
    pred_time1 = float(best["t1_s"])
    pred_time2 = float(best["t2_s"])

    near_rank = None
    for rank, idx in enumerate(order, start=1):
        if near_match(candidates[int(idx)], row):
            near_rank = rank
            break

    top5 = [candidates[int(idx)] for idx in order[:5]]
    top10 = [candidates[int(idx)] for idx in order[:10]]
    return {
        "sample_id": row["sample_id"],
        "source_file": row["source_file"],
        "T1_bin": t1_bin(actual_t1),
        "T2_bin": t2_bin(actual_t2),
        "path_class": path_class(actual_t1, actual_t2),
        "actual_T1_C": actual_t1,
        "actual_t1_s": actual_time1,
        "actual_T2_C": actual_t2,
        "actual_t2_s": actual_time2,
        "pred_T1_C": pred_t1,
        "pred_t1_s": pred_time1,
        "pred_T2_C": pred_t2,
        "pred_t2_s": pred_time2,
        "loss": float(losses[best_idx]),
        "target_loss": float(target_losses[best_idx]),
        "inverse_identifiability": float(identifiability[best_idx]),
        "extrapolation_penalty": float(extrapolation[best_idx]),
        "T1_abs_error_C": abs(pred_t1 - actual_t1),
        "T2_abs_error_C": abs(pred_t2 - actual_t2),
        "t1_factor_error": time_factor(pred_time1, actual_time1),
        "t2_factor_error": time_factor(pred_time2, actual_time2),
        "top1_near_match": near_match(best, row),
        "top5_near_match": any(near_match(candidate, row) for candidate in top5),
        "top10_near_match": any(near_match(candidate, row) for candidate in top10),
        "actual_like_rank": near_rank if near_rank is not None else "",
        "posterior_contains_actual_T1": posterior["T1_C"]["p5"] <= actual_t1 <= posterior["T1_C"]["p95"],
        "posterior_contains_actual_T2": posterior["T2_C"]["p5"] <= actual_t2 <= posterior["T2_C"]["p95"],
        "posterior_contains_actual_t1": posterior["log10_t1_s"]["p5"] <= math.log10(actual_time1) <= posterior["log10_t1_s"]["p95"],
        "posterior_contains_actual_t2": posterior["log10_t2_s"]["p5"] <= math.log10(actual_time2) <= posterior["log10_t2_s"]["p95"],
        "posterior_width_T1_C": posterior["T1_C"]["width"],
        "posterior_width_T2_C": posterior["T2_C"]["width"],
        "posterior_width_log10_t1_s": posterior["log10_t1_s"]["width"],
        "posterior_width_log10_t2_s": posterior["log10_t2_s"]["width"],
        "loss_flatness_ratio": posterior["loss_flatness_ratio"],
        "posterior_n_valid_candidates": posterior["n_valid_candidates"],
    }


def summarize(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    summary = []
    for group, values in sorted(groups.items()):
        n = len(values)
        summary.append({
            "group": group,
            "n": n,
            "T1_MAE_C": float(np.mean([v["T1_abs_error_C"] for v in values])),
            "T2_MAE_C": float(np.mean([v["T2_abs_error_C"] for v in values])),
            "median_t1_factor": float(np.median([v["t1_factor_error"] for v in values])),
            "median_t2_factor": float(np.median([v["t2_factor_error"] for v in values])),
            "top1_near_rate": float(np.mean([v["top1_near_match"] for v in values])),
            "top5_near_rate": float(np.mean([v["top5_near_match"] for v in values])),
            "top10_near_rate": float(np.mean([v["top10_near_match"] for v in values])),
            "posterior_T1_coverage": float(np.mean([v["posterior_contains_actual_T1"] for v in values])),
            "posterior_T2_coverage": float(np.mean([v["posterior_contains_actual_T2"] for v in values])),
            "posterior_t1_coverage": float(np.mean([v["posterior_contains_actual_t1"] for v in values])),
            "posterior_t2_coverage": float(np.mean([v["posterior_contains_actual_t2"] for v in values])),
            "median_posterior_width_T1_C": float(np.median([v["posterior_width_T1_C"] for v in values])),
            "median_posterior_width_T2_C": float(np.median([v["posterior_width_T2_C"] for v in values])),
            "median_loss_flatness_ratio": float(np.median([v["loss_flatness_ratio"] for v in values])),
            "mean_loss": float(np.mean([v["loss"] for v in values])),
        })
    return summary


def coverage_summary(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "two_step":
            groups[t1_bin(as_float(row["T1_C"]))].append(row)
    out = []
    for group, values in sorted(groups.items()):
        t2_values = sorted({round(as_float(v["T2_C"])) for v in values})
        out.append({
            "T1_bin": group,
            "two_step_n": len(values),
            "unique_T1": ",".join(str(round(as_float(v["T1_C"]))) for v in values[:0])
            or ",".join(map(str, sorted({round(as_float(v["T1_C"])) for v in values}))),
            "unique_T2": ",".join(map(str, t2_values)),
            "unique_paths": len({(round(as_float(v["T1_C"])), round(as_float(v["T2_C"]))) for v in values}),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(by_t1: list[dict], by_t2: list[dict], evaluations: list[dict]) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []

    def bar_plot(summary: list[dict], metric: str, title: str, ylabel: str, filename: str) -> None:
        labels = [row["group"] for row in summary]
        values = [row[metric] for row in summary]
        counts = [row["n"] for row in summary]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(labels, values, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Temperature interval")
        ax.grid(axis="y", alpha=0.25)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={count}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = FIG_DIR / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(str(out.relative_to(ROOT)))

    bar_plot(by_t1, "T1_MAE_C", "Inverse T1 Error by T1 Interval", "T1 MAE (deg C)", "inverse_T1_error_by_T1_bin.png")
    bar_plot(by_t1, "T2_MAE_C", "Inverse T2 Error by T1 Interval", "T2 MAE (deg C)", "inverse_T2_error_by_T1_bin.png")
    bar_plot(by_t1, "top5_near_rate", "Top-5 Near-Match Rate by T1 Interval", "Top-5 near-match rate", "inverse_top5_rate_by_T1_bin.png")
    bar_plot(by_t2, "top5_near_rate", "Top-5 Near-Match Rate by T2 Interval", "Top-5 near-match rate", "inverse_top5_rate_by_T2_bin.png")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    colors = {"50-60": "#4c78a8", "65-75": "#f58518", "80-90": "#54a24b", "95-100": "#e45756", "other": "#999999"}
    for group in sorted({row["T1_bin"] for row in evaluations}):
        vals = [row for row in evaluations if row["T1_bin"] == group]
        ax.scatter([v["actual_T1_C"] for v in vals], [v["pred_T1_C"] for v in vals], label=group, alpha=0.75, s=40, color=colors.get(group, "#999999"))
    ax.plot([45, 105], [45, 105], "--", color="black", linewidth=1)
    ax.set_xlim(45, 105)
    ax.set_ylim(45, 105)
    ax.set_xlabel("Actual T1 (deg C)")
    ax.set_ylabel("Inverse predicted T1 (deg C)")
    ax.set_title("Inverse Reconstruction: T1")
    ax.legend(title="T1 bin", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "inverse_T1_actual_vs_pred.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    paths.append(str(out.relative_to(ROOT)))

    return paths


def main() -> None:
    payload = load_model_payload()
    targets = list(payload["targets"])
    rows = read_rows(DATASET_PATH)
    two_step_rows = [row for row in rows if row["mode"] == "two_step"]
    candidates, pred, unc, identifiability, extrapolation = build_candidate_predictions()
    evaluations = [inverse_rank(row, candidates, pred, unc, identifiability, extrapolation, targets) for row in two_step_rows]
    by_t1 = summarize(evaluations, "T1_bin")
    by_t2 = summarize(evaluations, "T2_bin")
    by_path = summarize(evaluations, "path_class")
    coverage = coverage_summary(rows)
    figures = plot_summaries(by_t1, by_t2, evaluations)

    write_csv(OUT_DIR / "inverse_reconstruction_by_sample.csv", evaluations)
    write_csv(OUT_DIR / "inverse_summary_by_T1_bin.csv", by_t1)
    write_csv(OUT_DIR / "inverse_summary_by_T2_bin.csv", by_t2)
    write_csv(OUT_DIR / "inverse_summary_by_path_class.csv", by_path)
    write_csv(OUT_DIR / "two_step_coverage_by_T1_bin.csv", coverage)

    payload = {
        "n_two_step_samples": len(two_step_rows),
        "active_targets": ACTIVE_TARGETS,
        "near_match_definition": "T1/T2 within 5 deg C and each time within a factor of 3.",
        "candidate_time_grid_s": "10,30,60,100,300,600,900,1200,1800",
        "posterior_definition": "Best 5 percent of finite regularized inverse losses after Tp-based T2 soft filtering.",
        "summary_by_T1_bin": by_t1,
        "summary_by_T2_bin": by_t2,
        "summary_by_path_class": by_path,
        "coverage_by_T1_bin": coverage,
        "figures": figures,
    }
    (OUT_DIR / "inverse_temperature_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
