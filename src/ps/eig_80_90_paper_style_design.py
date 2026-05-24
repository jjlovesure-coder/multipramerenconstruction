"""Paper-style EIG recommendations for PS 80/90 conditional time inverse."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.ps.compare_strict_arrt_paper_style import STRICT_PAPER_TARGETS, TARGET_FLOORS, finite, prepare_rows, usable_rows
from src.ps.dsc_processing import ROOT
from src.ps.train_ps_model import featurize, read_rows


OUT_DIR = ROOT / "results" / "ps" / "eig"
DOC_PATH = ROOT / "docs" / "ps_80_90_paper_style_eig_plan.md"
PLAN_PATH = ROOT / "data" / "ps" / "ps_80_90_paper_style_next_experiments.csv"
PS_FEATURE_PATH = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"

TIME_GRID_S = [10.0, 30.0, 100.0, 300.0, 900.0, 1800.0]
TEMPERATURE_PAIRS = [(80.0, 90.0), (90.0, 80.0)]
HEATING_RATES = "5,10,20,40"


def candidate_conditions() -> list[dict[str, object]]:
    rows = []
    for t1_c, t2_c in TEMPERATURE_PAIRS:
        for t1_s in TIME_GRID_S:
            for t2_s in TIME_GRID_S:
                rows.append({
                    "mode": "two_step",
                    "T1_C": t1_c,
                    "T2_C": t2_c,
                    "t1_s": t1_s,
                    "t2_s": t2_s,
                    "total_anneal_time_s": t1_s + t2_s,
                    "requires_multi_rate_arrt": True,
                    "heating_rates_C_min": HEATING_RATES,
                })
    return rows


def _model_row(row: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in row.items()}


def train_strict_model() -> tuple[KernelRegressor, np.ndarray, dict[str, float], int]:
    rows = usable_rows(prepare_rows(), STRICT_PAPER_TARGETS)
    x = np.array([featurize(row) for row in rows], dtype=float)
    y = np.array([[float(row[target]) for target in STRICT_PAPER_TARGETS] for row in rows], dtype=float)
    scales = {}
    for j, target in enumerate(STRICT_PAPER_TARGETS):
        scales[target] = max(float(np.std(y[:, j])), TARGET_FLOORS[target])
    return KernelRegressor.fit(x, y, bandwidth=1.2), y, scales, len(rows)


def current_80_90_support() -> dict[tuple[float, float], list[tuple[float, float]]]:
    support = {pair: [] for pair in TEMPERATURE_PAIRS}
    if not PS_FEATURE_PATH.exists():
        return support
    for row in read_rows(PS_FEATURE_PATH):
        try:
            pair = (float(row["T1_C"]), float(row["T2_C"]))
        except (TypeError, ValueError):
            continue
        if pair not in support:
            continue
        t1 = float(row["t1_s"])
        t2 = float(row["t2_s"])
        if t1 >= 10.0 and t2 >= 10.0:
            support[pair].append((t1, t2))
    return support


def strict_arrt_pair_support() -> set[tuple[float, float]]:
    out = set()
    for row in usable_rows(prepare_rows(), STRICT_PAPER_TARGETS):
        try:
            if row.get("mode") == "two_step":
                out.add((float(row["T1_C"]), float(row["T2_C"])))
        except (TypeError, ValueError):
            continue
    return out


def novelty_from_current(row: dict[str, object], support: dict[tuple[float, float], list[tuple[float, float]]]) -> float:
    pair = (float(row["T1_C"]), float(row["T2_C"]))
    existing = support.get(pair, [])
    if not existing:
        return 1.0
    point = np.array([math.log10(float(row["t1_s"])), math.log10(float(row["t2_s"]))], dtype=float)
    distances = [
        float(np.linalg.norm(point - np.array([math.log10(t1), math.log10(t2)], dtype=float)))
        for t1, t2 in existing
    ]
    return float(min(1.0, min(distances) / 1.0))


def sensitivity_score(model: KernelRegressor, row: dict[str, object], scales: np.ndarray) -> float:
    base = np.array(featurize(_model_row(row)), dtype=float)

    def perturbed(name: str) -> np.ndarray:
        copied = dict(row)
        copied[name] = min(max(float(copied[name]) * 1.5, 10.0), 1800.0)
        return np.array(featurize(_model_row(copied)), dtype=float)

    pred_base, _ = model.predict(base)
    pred_t1, _ = model.predict(perturbed("t1_s"))
    pred_t2, _ = model.predict(perturbed("t2_s"))
    grad = np.r_[np.abs(pred_t1[0] - pred_base[0]) / scales, np.abs(pred_t2[0] - pred_base[0]) / scales]
    return float(np.mean(grad))


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def greedy_diverse_order(rows: list[dict[str, object]], top_n: int = 18) -> list[dict[str, object]]:
    remaining = sorted(rows, key=lambda row: float(row["raw_eig_score"]), reverse=True)
    selected: list[dict[str, object]] = []
    while remaining and len(selected) < top_n:
        best_idx = 0
        best_score = -float("inf")
        for idx, row in enumerate(remaining):
            score = float(row["raw_eig_score"])
            pair = (float(row["T1_C"]), float(row["T2_C"]))
            point = np.array([math.log10(float(row["t1_s"])), math.log10(float(row["t2_s"]))], dtype=float)
            penalties = []
            for chosen in selected:
                if (float(chosen["T1_C"]), float(chosen["T2_C"])) != pair:
                    continue
                chosen_point = np.array([math.log10(float(chosen["t1_s"])), math.log10(float(chosen["t2_s"]))], dtype=float)
                penalties.append(max(0.0, 0.8 - float(np.linalg.norm(point - chosen_point))) / 0.8)
            diverse_score = score - 0.25 * max(penalties, default=0.0)
            if diverse_score > best_score:
                best_score = diverse_score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def recommended_minimum_batch(rows: list[dict[str, object]], top_n: int = 12) -> list[dict[str, object]]:
    """Combine EIG ranking with diagonal anchors needed for t1/t2 identifiability."""
    by_key = {
        (float(row["T1_C"]), float(row["T2_C"]), float(row["t1_s"]), float(row["t2_s"])): row
        for row in rows
    }
    selected: list[dict[str, object]] = []
    used: set[tuple[float, float, float, float]] = set()
    anchor_times = [30.0, 300.0, 900.0]
    for pair in TEMPERATURE_PAIRS:
        for time_s in anchor_times:
            key = (pair[0], pair[1], time_s, time_s)
            if key in by_key:
                item = dict(by_key[key])
                item["selection_reason"] = "time_diagonal_anchor"
                selected.append(item)
                used.add(key)

    for row in greedy_diverse_order(rows, top_n=len(rows)):
        key = (float(row["T1_C"]), float(row["T2_C"]), float(row["t1_s"]), float(row["t2_s"]))
        if key in used:
            continue
        item = dict(row)
        item["selection_reason"] = "high_EIG_or_identifiability"
        selected.append(item)
        used.add(key)
        if len(selected) >= top_n:
            break

    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, object]) -> None:
    top_rows: list[dict[str, object]] = summary["recommended_batch"]  # type: ignore[assignment]
    lines = [
        "# PS 80/90 Paper-Style EIG Experiment Plan",
        "",
        "## Goal",
        "",
        "The goal is not general four-parameter inverse reconstruction. It is the paper-style conditional task: with `T1,T2` fixed to the 80/90 pair, use Figure-2-like enthalpy features plus Figure-3-like `H*/S*` kinetic features to constrain `t1,t2`.",
        "",
        "## Current Gap",
        "",
        f"- Strict ARRT training samples available: `{summary['n_strict_arrt_samples']}`.",
        f"- Strict ARRT 80/90 pairs already available: `{summary['has_strict_80_90_support']}`.",
        "- Current 80/90 DSC curves are useful for enthalpy trends, but they do not provide strict multi-rate `H*/S*` labels for the 80/90 pair.",
        "",
        "## Recommended Minimum Batch",
        "",
        "| Rank | T1 -> T2 | t1 s | t2 s | Reason | EIG | Uncertainty | Sensitivity | Novelty |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows[:12]:
        lines.append(
            f"| {row['rank']} | {row['T1_C']} -> {row['T2_C']} C | {row['t1_s']} | {row['t2_s']} | {row.get('selection_reason', '')} | {float(row['eig_score']):.3f} | {float(row['uncertainty_component']):.3f} | {float(row['sensitivity_component']):.3f} | {float(row['novelty_component']):.3f} |"
        )
    lines.extend([
        "",
        "## Measurement Requirement",
        "",
        "Each selected condition should be measured at `5, 10, 20, 40 C/min` so that a strict Kissinger/ARRT `H*/S*` label can be calculated. A single 10 C/min heating curve only adds Figure-2-like enthalpy information and will not reproduce the paper-style Figure-3 constraint.",
        "",
        "## Interpretation",
        "",
        "The first batch should include both directions (`80 -> 90 C` and `90 -> 80 C`) and both balanced and asymmetric time pairs. Balanced points calibrate total relaxation dose; asymmetric points test whether the two times are separately identifiable when the temperature pair is fixed.",
        "",
        "Outputs:",
        "",
        "- full EIG ranking: `results/ps/eig/ps_80_90_paper_style_eig_candidates.csv`",
        "- recommended minimum batch: `data/ps/ps_80_90_paper_style_next_experiments.csv`",
    ])
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, _, scale_map, n_strict = train_strict_model()
    scales = np.array([scale_map[target] for target in STRICT_PAPER_TARGETS], dtype=float)
    candidates = candidate_conditions()
    x = np.array([featurize(_model_row(row)) for row in candidates], dtype=float)
    pred, unc = model.predict(x)
    current_support = current_80_90_support()
    strict_support = strict_arrt_pair_support()

    uncertainty = np.mean(np.log1p(unc / scales[None, :]), axis=1)
    sensitivity = np.array([sensitivity_score(model, row, scales) for row in candidates], dtype=float)
    novelty = np.array([novelty_from_current(row, current_support) for row in candidates], dtype=float)
    asymmetry = np.array([
        abs(math.log10(float(row["t1_s"])) - math.log10(float(row["t2_s"]))) / math.log10(180.0)
        for row in candidates
    ], dtype=float)
    time_penalty = np.array([float(row["total_anneal_time_s"]) / 3600.0 for row in candidates], dtype=float)

    uncertainty_n = normalize(uncertainty)
    sensitivity_n = normalize(sensitivity)
    novelty_n = novelty
    asymmetry_n = normalize(asymmetry)
    raw_score = 0.45 * uncertainty_n + 0.25 * sensitivity_n + 0.20 * novelty_n + 0.10 * asymmetry_n - 0.08 * time_penalty

    enriched = []
    for i, row in enumerate(candidates):
        item = dict(row)
        item["raw_eig_score"] = float(raw_score[i])
        item["eig_score"] = float(raw_score[i])
        item["uncertainty_component"] = float(uncertainty_n[i])
        item["sensitivity_component"] = float(sensitivity_n[i])
        item["novelty_component"] = float(novelty_n[i])
        item["asymmetry_component"] = float(asymmetry_n[i])
        item["time_penalty_component"] = float(time_penalty[i])
        item["has_existing_strict_arrt_pair"] = (float(row["T1_C"]), float(row["T2_C"])) in strict_support
        for j, target in enumerate(STRICT_PAPER_TARGETS):
            item[f"pred_{target}"] = float(pred[i, j])
            item[f"uncertainty_{target}"] = float(unc[i, j])
        enriched.append(item)

    full_top = greedy_diverse_order(enriched, top_n=18)
    recommended = recommended_minimum_batch(enriched, top_n=12)
    candidate_path = OUT_DIR / "ps_80_90_paper_style_eig_candidates.csv"
    write_csv(candidate_path, sorted(enriched, key=lambda row: float(row["raw_eig_score"]), reverse=True))
    write_csv(PLAN_PATH, recommended)
    summary = {
        "n_strict_arrt_samples": n_strict,
        "strict_targets": STRICT_PAPER_TARGETS,
        "candidate_temperature_pairs_C": TEMPERATURE_PAIRS,
        "time_grid_s": TIME_GRID_S,
        "heating_rates_C_min": HEATING_RATES,
        "has_strict_80_90_support": {
            "80->90": (80.0, 90.0) in strict_support,
            "90->80": (90.0, 80.0) in strict_support,
        },
        "current_80_90_support_counts_practical_time": {f"{k[0]}->{k[1]}": len(v) for k, v in current_support.items()},
        "score_definition": "0.45 uncertainty + 0.25 time sensitivity + 0.20 novelty + 0.10 asymmetry - 0.08 total-time penalty",
        "top_candidates": full_top,
        "recommended_batch": recommended,
    }
    summary_path = OUT_DIR / "ps_80_90_paper_style_eig_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"output": str(summary_path), "plan": str(PLAN_PATH), "recommended_first6": recommended[:6]}, indent=2))
    return summary_path


if __name__ == "__main__":
    main()
