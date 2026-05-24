"""Minimal t2 inverse check for the PS 80/90 two-step data.

The reduced question is:

    fixed T1/T2 and known t1 in {~50 s, ~500 s}; can final heating-curve
    outputs recover t2, and would paper-style H*/S* information be expected
    to help?
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from src.models.evaluate_paper_conditional_time_inverse import (
    FIG2_FIG3_TARGETS,
    FIG2_TARGETS,
    candidate_observables,
    candidate_rows_for_fixed_temperatures,
    ensure_dataset,
    fit_delta_peak_model,
    scales_for_targets,
    unique_times,
)
from src.models.kernel_regression import KernelRegressor
from src.models.train_forward import FEATURES as PAPER_FEATURES
from src.ps.dsc_processing import ROOT
from src.ps.train_ps_model import read_rows


OUT_DIR = ROOT / "results" / "ps" / "minimal_t2_inverse_80_90"
DOC_PATH = ROOT / "docs" / "ps_80_90_minimal_t2_inverse.md"
PS_DATASET = ROOT / "data" / "ps" / "ps_preexperiment_features.csv"
ARRT_RESULTS = ROOT / "results" / "ps" / "arrt_kissinger" / "arrt_kissinger_results.csv"

PS_TEMPERATURE_PAIRS = [(80.0, 90.0), (90.0, 80.0)]
PS_TARGET_SETS = {
    "enthalpy_only": ["delta_h_total_J_g", "peak_area_J_g"],
    "heating_curve_full": [
        "delta_h_total_J_g",
        "peak_area_J_g",
        "peak_temperature_Tp_C",
        "peak_height_uW",
        "recovery_index",
    ],
}
PS_TARGET_FLOORS = {
    "delta_h_total_J_g": 0.05,
    "peak_area_J_g": 0.05,
    "peak_temperature_Tp_C": 0.10,
    "peak_height_uW": 5.0,
    "recovery_index": 0.03,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _finite_float(row: dict[str, str], key: str) -> bool:
    try:
        return math.isfinite(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return False


def t1_group(value: float) -> str:
    return "50s" if abs(value - 50.0) <= abs(value - 500.0) else "500s"


def ps_80_90_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_rows(PS_DATASET):
        try:
            pair = (float(row["T1_C"]), float(row["T2_C"]))
            t1 = float(row["t1_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if pair not in PS_TEMPERATURE_PAIRS:
            continue
        if t1_group(t1) not in {"50s", "500s"}:
            continue
        if not all(_finite_float(row, target) for targets in PS_TARGET_SETS.values() for target in targets):
            continue
        copied = dict(row)
        copied["t1_group"] = t1_group(t1)
        rows.append(copied)
    return rows


def target_scales(rows: list[dict[str, str]], targets: list[str]) -> dict[str, float]:
    scales = {}
    for target in targets:
        values = np.array([float(row[target]) for row in rows], dtype=float)
        scales[target] = max(float(np.std(values)), PS_TARGET_FLOORS[target])
    return scales


def ps_feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([[math.log10(max(float(row["t2_s"]), 1e-12))] for row in rows], dtype=float)


def ps_target_matrix(rows: list[dict[str, str]], targets: list[str]) -> np.ndarray:
    return np.array([[float(row[target]) for target in targets] for row in rows], dtype=float)


def evaluate_ps_group(rows: list[dict[str, str]], targets: list[str], variant: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    groups = sorted({(row["T1_C"], row["T2_C"], row["t1_group"]) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if (row["T1_C"], row["T2_C"], row["t1_group"]) == group]
        if len(group_rows) < 4:
            continue
        scales = target_scales(group_rows, targets)
        candidate_rows = [dict(row) for row in group_rows]
        candidate_x = ps_feature_matrix(candidate_rows)
        for test_idx, actual in enumerate(group_rows):
            train_rows = [row for idx, row in enumerate(group_rows) if idx != test_idx]
            model = KernelRegressor.fit(ps_feature_matrix(train_rows), ps_target_matrix(train_rows, targets), bandwidth=0.55)
            pred, unc = model.predict(candidate_x)
            actual_vec = {target: float(actual[target]) for target in targets}
            losses = []
            for i in range(len(candidate_rows)):
                terms = [abs((float(pred[i, j]) - actual_vec[target]) / scales[target]) for j, target in enumerate(targets)]
                losses.append(float(np.mean(terms)))
            order = np.argsort(np.array(losses, dtype=float))
            best = candidate_rows[int(order[0])]
            actual_t2 = float(actual["t2_s"])
            pred_t2 = float(best["t2_s"])
            top3_indices = [int(idx) for idx in order[:3]]
            top5_indices = [int(idx) for idx in order[:5]]

            def near(indices: list[int], factor: float = 3.0) -> bool:
                return any(
                    max(float(candidate_rows[idx]["t2_s"]) / actual_t2, actual_t2 / float(candidate_rows[idx]["t2_s"])) <= factor
                    for idx in indices
                )

            records.append({
                "source": "ps_80_90",
                "variant": variant,
                "T1_C": float(actual["T1_C"]),
                "T2_C": float(actual["T2_C"]),
                "t1_group": actual["t1_group"],
                "actual_t1_s": float(actual["t1_s"]),
                "actual_t2_s": actual_t2,
                "pred_t2_s": pred_t2,
                "t2_log10_abs_error": abs(math.log10(pred_t2 / actual_t2)),
                "t2_factor_error": max(pred_t2 / actual_t2, actual_t2 / pred_t2),
                "top1_near_factor3": near([int(order[0])]),
                "top3_near_factor3": near(top3_indices),
                "top5_near_factor3": near(top5_indices),
                "loss": float(losses[int(order[0])]),
                "mean_uncertainty": float(np.mean(unc[int(order[0])])),
                "targets": ",".join(targets),
            })
    return records


def evaluate_ps() -> dict[str, object]:
    rows = ps_80_90_rows()
    variants = {}
    all_records: list[dict[str, object]] = []
    for name, targets in PS_TARGET_SETS.items():
        records = evaluate_ps_group(rows, targets, name)
        variants[name] = summarize(records)
        all_records.extend(records)
    return {
        "n_rows": len(rows),
        "t1_groups": sorted({row["t1_group"] for row in rows}),
        "variants": variants,
        "records": all_records,
    }


def evaluate_paper_t2_only() -> dict[str, object]:
    rows = ensure_dataset()
    times = unique_times(rows)
    variants: dict[str, object] = {}
    records_all: list[dict[str, object]] = []
    for variant, targets in {"fig2_only": FIG2_TARGETS, "fig2_plus_fig3_kinetics": FIG2_FIG3_TARGETS}.items():
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
            candidates = [row for row in candidates if abs(float(row["t1_s"]) - float(actual["t1_s"])) < 1e-12]
            candidate_values = [candidate_observables(candidate) for candidate in candidates]
            x = np.array([[values[feature] for feature in PAPER_FEATURES] for values in candidate_values], dtype=float)
            pred, _ = model.predict(x)
            losses = []
            for i, values in enumerate(candidate_values):
                terms = []
                for target in targets:
                    if target in {"delta_h_kj_mol", "delta_h_peak_kj_mol"}:
                        pred_value = float(pred[i, ["delta_h_kj_mol", "delta_h_peak_kj_mol"].index(target)])
                    else:
                        pred_value = float(values[target])
                    terms.append(abs((pred_value - float(actual[target])) / scales[target]))
                losses.append(float(np.mean(terms)))
            order = np.argsort(np.array(losses, dtype=float))
            best = candidates[int(order[0])]
            actual_t2 = float(actual["t2_s"])
            pred_t2 = float(best["t2_s"])
            records.append({
                "source": "paper",
                "variant": variant,
                "panel": actual["panel"],
                "T1_K": float(actual["t1_temperature_k"]),
                "T2_K": float(actual["t2_temperature_k"]),
                "actual_t1_s": float(actual["t1_s"]),
                "actual_t2_s": actual_t2,
                "pred_t2_s": pred_t2,
                "t2_log10_abs_error": abs(math.log10(pred_t2 / actual_t2)),
                "t2_factor_error": max(pred_t2 / actual_t2, actual_t2 / pred_t2),
                "top1_near_factor3": max(pred_t2 / actual_t2, actual_t2 / pred_t2) <= 3.0,
                "loss": float(losses[int(order[0])]),
                "targets": ",".join(targets),
            })
        variants[variant] = summarize(records)
        records_all.extend(records)
    return {"variants": variants, "records": records_all}


def has_80_90_strict_hs() -> dict[str, object]:
    rows = _read_csv(ARRT_RESULTS)
    pairs = set()
    matching = []
    for row in rows:
        try:
            if row.get("mode") != "two_step":
                continue
            pair = (float(row["T1_C"]), float(row["T2_C"]))
        except (TypeError, ValueError):
            continue
        if pair in PS_TEMPERATURE_PAIRS:
            pairs.add(pair)
            matching.append(row)
    return {
        "has_80_to_90": (80.0, 90.0) in pairs,
        "has_90_to_80": (90.0, 80.0) in pairs,
        "matching_rows": matching,
        "n_matching_rows": len(matching),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"n": 0}
    out = {
        "n": len(records),
        "t2_log10_MAE": float(np.mean([float(row["t2_log10_abs_error"]) for row in records])),
        "t2_factor_error_median": float(median([float(row["t2_factor_error"]) for row in records])),
        "top1_near_factor3": float(np.mean([1.0 if row["top1_near_factor3"] else 0.0 for row in records])),
    }
    if "top3_near_factor3" in records[0]:
        out["top3_near_factor3"] = float(np.mean([1.0 if row["top3_near_factor3"] else 0.0 for row in records]))
        out["top5_near_factor3"] = float(np.mean([1.0 if row["top5_near_factor3"] else 0.0 for row in records]))
    by_group = {}
    group_keys = sorted({tuple(str(row.get(key, "")) for key in ("T1_C", "T2_C", "t1_group")) for row in records})
    for group in group_keys:
        selected = [
            row for row in records
            if tuple(str(row.get(key, "")) for key in ("T1_C", "T2_C", "t1_group")) == group
        ]
        if not selected or group == ("", "", ""):
            continue
        by_group["|".join(group)] = {
            "n": len(selected),
            "t2_log10_MAE": float(np.mean([float(row["t2_log10_abs_error"]) for row in selected])),
            "top3_near_factor3": float(np.mean([1.0 if row.get("top3_near_factor3") else 0.0 for row in selected])),
        }
    if by_group:
        out["by_group"] = by_group
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, object]) -> None:
    paper = summary["paper_t2_only"]["variants"]  # type: ignore[index]
    ps = summary["ps_80_90_t2_only"]["variants"]  # type: ignore[index]
    hs = summary["strict_hs_80_90_availability"]  # type: ignore[index]
    lines = [
        "# PS 80/90 Minimal t2 Inverse Check",
        "",
        "## Question",
        "",
        "Fixed `T1/T2` and known `t1 ~= 50 s or 500 s`; can the model recover `t2`? This is the reduced version of the paper-style inverse problem.",
        "",
        "## Paper Sanity Check",
        "",
        "| Paper target set | t2 log10 MAE | Median factor error | Top1 within factor 3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, block in paper.items():  # type: ignore[union-attr]
        lines.append(f"| {name} | {block['t2_log10_MAE']:.3f} | {block['t2_factor_error_median']:.3f} | {block['top1_near_factor3']:.3f} |")
    lines.extend([
        "",
        "## Current PS 80/90 Check",
        "",
        "| PS target set | t2 log10 MAE | Median factor error | Top1 within factor 3 | Top3 within factor 3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, block in ps.items():  # type: ignore[union-attr]
        lines.append(
            f"| {name} | {block['t2_log10_MAE']:.3f} | {block['t2_factor_error_median']:.3f} | {block['top1_near_factor3']:.3f} | {block['top3_near_factor3']:.3f} |"
        )
    lines.extend([
        "",
        "## H*/S* Availability",
        "",
        f"- strict `80 -> 90` H*/S* available: `{hs['has_80_to_90']}`",
        f"- strict `90 -> 80` H*/S* available: `{hs['has_90_to_80']}`",
        "",
        "## Interpretation",
        "",
        "The paper sanity check tests the logic: when Figure-3-like kinetic information is available, fixed-temperature `t2` recovery improves strongly. The current PS check tests the available experimental curves: without strict 80/90 H*/S* labels, the reduced inverse can only use final heating-curve features.",
        "",
        "Therefore the minimum added experiment is not a full two-step matrix. It is enough to obtain strict H*/S* anchors for the two relevant temperatures/paths: multi-heating-rate measurements around 80 C and 90 C, preferably at the same `t1 ~= 50/500 s` protocol or at least single-step 80 C and 90 C ARRT anchors if the H*/S* is to be used as a temperature-specific prior.",
        "",
        "Outputs:",
        "",
        "- `results/ps/minimal_t2_inverse_80_90/minimal_t2_inverse_summary.json`",
        "- `results/ps/minimal_t2_inverse_80_90/ps_80_90_t2_records.csv`",
        "- `results/ps/minimal_t2_inverse_80_90/paper_t2_records.csv`",
    ])
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = evaluate_paper_t2_only()
    ps = evaluate_ps()
    hs = has_80_90_strict_hs()
    write_csv(OUT_DIR / "paper_t2_records.csv", paper["records"])  # type: ignore[arg-type]
    write_csv(OUT_DIR / "ps_80_90_t2_records.csv", ps["records"])  # type: ignore[arg-type]
    summary = {
        "question": "fixed T1/T2 and known t1 in the measured 50s/500s groups; infer t2",
        "paper_t2_only": paper,
        "ps_80_90_t2_only": ps,
        "strict_hs_80_90_availability": hs,
        "conclusion_rule": "If PS heating-curve-only t2 recovery is weak and no strict 80/90 H*/S* exists, add H*/S* anchors before expanding the two-step matrix.",
    }
    out = OUT_DIR / "minimal_t2_inverse_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({
        "output": str(out),
        "paper": {key: value["t2_log10_MAE"] for key, value in paper["variants"].items()},
        "ps": {key: value["t2_log10_MAE"] for key, value in ps["variants"].items()},
        "strict_hs_80_90_availability": hs,
    }, indent=2))
    return out


if __name__ == "__main__":
    main()
