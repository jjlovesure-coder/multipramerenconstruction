"""Compare strict measured ARRT H*/S* with proxy paper-style targets."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.physics.arrt import H_PLANCK, K_B, R
from src.ps.calculate_arrt_from_heating_rates import OUT_DIR as ARRT_DIR
from src.ps.dsc_processing import ROOT
from src.ps.inverse_design import candidate_rows
from src.ps.train_ps_model import featurize
from src.ps.train_strict_arrt_model import clean_rows, read_csv


OUT_DIR = ROOT / "results" / "ps" / "strict_arrt_model"
DOC_PATH = ROOT / "docs" / "ps_strict_arrt_paper_style_benchmark.md"
CURRENT_COMPARISON_PATH = ROOT / "results" / "ps" / "paper_style_comparison" / "summary.json"

STRICT_PAPER_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "activation_enthalpy_H_star_kj_mol",
    "activation_entropy_S_star_j_mol_K",
]
PROXY_TARGETS = [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "proxy_activation_enthalpy_H_star_kj_mol",
    "proxy_activation_entropy_S_star_j_mol_K",
]
TARGET_FLOORS = {
    "delta_h_total_J_g": 0.20,
    "peak_area_J_g": 0.13,
    "activation_enthalpy_H_star_kj_mol": 50.0,
    "activation_entropy_S_star_j_mol_K": 150.0,
    "proxy_activation_enthalpy_H_star_kj_mol": 50.0,
    "proxy_activation_entropy_S_star_j_mol_K": 150.0,
}


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def proxy_arrt_pair(row: dict[str, str]) -> tuple[float, float]:
    total_time = max(float(row["total_anneal_time_s"]), 1e-6)
    t1 = max(float(row["t1_s"]), 0.0)
    t2 = float(row["t2_s"]) if row.get("t2_s") not in {"", "nan", "NaN"} else 0.0
    t1_k = float(row["T1_C"]) + 273.15
    t2_k = float(row["T2_C"]) + 273.15 if row.get("T2_C") not in {"", "nan", "NaN"} else t1_k
    t_ann = (t1 * t1_k + t2 * t2_k) / max(t1 + t2, 1e-9)
    tp_k = float(row["mean_peak_temperature_Tp_K"])
    beta_k_s = 10.0 / 60.0

    k_ann = 1.0 / total_time
    k_peak = beta_k_s / max(tp_k, 1e-9)
    a_ann = math.log(k_ann) - math.log(K_B * t_ann / H_PLANCK)
    a_peak = math.log(k_peak) - math.log(K_B * tp_k / H_PLANCK)
    inv_ann = 1.0 / t_ann
    inv_peak = 1.0 / tp_k
    if abs(inv_peak - inv_ann) < 1e-12:
        return math.nan, math.nan
    slope = (a_peak - a_ann) / (inv_peak - inv_ann)
    h_j_mol = -R * slope
    s_j_mol_k = R * (a_ann + h_j_mol / (R * t_ann))
    return h_j_mol / 1000.0, s_j_mol_k


def prepare_rows() -> list[dict[str, str]]:
    rows = clean_rows(read_csv(ARRT_DIR / "arrt_kissinger_results.csv"))
    out = []
    for row in rows:
        copied = dict(row)
        copied["delta_h_total_J_g"] = row.get("mean_delta_h_total_J_g", "")
        copied["peak_area_J_g"] = row.get("mean_peak_area_J_g", "")
        h_proxy, s_proxy = proxy_arrt_pair(copied)
        copied["proxy_activation_enthalpy_H_star_kj_mol"] = h_proxy
        copied["proxy_activation_entropy_S_star_j_mol_K"] = s_proxy
        out.append(copied)
    return out


def usable_rows(rows: list[dict[str, str]], targets: list[str]) -> list[dict[str, str]]:
    return [row for row in rows if all(finite(row.get(target)) for target in targets)]


def metrics(targets: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(targets):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        out[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "r2": float(1.0 - np.sum(err * err) / denom) if denom > 0 else math.nan,
            "target_std": float(np.std(y_true[:, j])),
        }
    return out


def loocv(rows: list[dict[str, str]], targets: list[str]) -> dict[str, object]:
    rows = usable_rows(rows, targets)
    x = np.array([featurize(row) for row in rows], dtype=float)
    y = np.array([[float(row[target]) for target in targets] for row in rows], dtype=float)
    y_pred = np.zeros_like(y)
    predictions = []
    for test_idx in range(len(rows)):
        train_idx = np.array([idx for idx in range(len(rows)) if idx != test_idx], dtype=int)
        model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=1.2)
        pred, unc = model.predict(x[test_idx])
        y_pred[test_idx] = pred[0]
        record: dict[str, object] = {
            "condition_key": rows[test_idx].get("condition_key", ""),
            "kissinger_confidence": rows[test_idx].get("kissinger_confidence", ""),
        }
        for j, target in enumerate(targets):
            record[f"actual_{target}"] = float(y[test_idx, j])
            record[f"pred_{target}"] = float(pred[0, j])
            record[f"error_{target}"] = float(pred[0, j] - y[test_idx, j])
            record[f"uncertainty_{target}"] = float(unc[0, j])
        predictions.append(record)
    full_model = KernelRegressor.fit(x, y, bandwidth=1.2)
    return {
        "n_samples": len(rows),
        "targets": targets,
        "metrics": metrics(targets, y, y_pred),
        "predictions": predictions,
        "model": full_model,
        "x": x,
        "y": y,
    }


def normalized_mae(metrics_: dict[str, dict[str, float]]) -> float:
    ratios = []
    for value in metrics_.values():
        if value["target_std"] > 0:
            ratios.append(value["mae"] / value["target_std"])
    return float(np.mean(ratios)) if ratios else math.nan


def eig_ranking(result: dict[str, object], targets: list[str], top_n: int = 30) -> tuple[list[dict[str, object]], dict[str, float]]:
    model: KernelRegressor = result["model"]  # type: ignore[assignment]
    rows = candidate_rows()
    x = np.array([featurize(row) for row in rows], dtype=float)
    pred, unc = model.predict(x)
    scales = np.array([
        max(float(result["metrics"][target]["rmse"]), TARGET_FLOORS[target])  # type: ignore[index]
        for target in targets
    ], dtype=float)
    components = np.log1p(unc / scales[None, :])
    eig = components.mean(axis=1)
    order = np.argsort(-eig)
    ranking = []
    for rank, idx in enumerate(order[:top_n], start=1):
        row = dict(rows[int(idx)])
        row["rank"] = rank
        row["eig_score"] = float(eig[int(idx)])
        for j, target in enumerate(targets):
            row[f"eig_{target}"] = float(components[int(idx), j])
            row[f"pred_{target}"] = float(pred[int(idx), j])
            row[f"uncertainty_{target}"] = float(unc[int(idx), j])
        ranking.append(row)
    return ranking, {
        "mean": float(np.mean(eig)),
        "p50": float(np.percentile(eig, 50)),
        "p90": float(np.percentile(eig, 90)),
        "p99": float(np.percentile(eig, 99)),
        "max": float(np.max(eig)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def write_report(summary: dict[str, object]) -> None:
    metric_rows = [["Target set", "Target", "MAE", "RMSE", "R2"]]
    for label, block in [
        ("strict measured H*/S*", summary["strict_paper_style"]),
        ("old proxy H*/S*", summary["proxy_paper_style"]),
    ]:
        for target, value in block["metrics"].items():
            metric_rows.append([label, target, f"{value['mae']:.4g}", f"{value['rmse']:.4g}", f"{value['r2']:.3f}"])

    rows = [["Rank", "Strict EIG top condition", "Proxy EIG top condition"]]
    for idx in range(min(10, len(summary["strict_top30"]), len(summary["proxy_top30"]))):
        strict = summary["strict_top30"][idx]
        proxy = summary["proxy_top30"][idx]
        rows.append([
            str(idx + 1),
            f"{strict['T1_C']}C/{strict['t1_s']}s -> {strict['T2_C']}C/{strict['t2_s']}s",
            f"{proxy['T1_C']}C/{proxy['t1_s']}s -> {proxy['T2_C']}C/{proxy['t2_s']}s",
        ])

    text = f"""# Strict ARRT Paper-Style Benchmark

## Scope

This benchmark separates three quantities that were previously easy to mix:

- strict experimental `H* / S*`: calculated from multi-heating-rate Kissinger peak shifts.
- old proxy `H* / S*`: inferred from annealing time and one heating-curve `Tp`.
- current PS multi-target model: richer DSC descriptors used for forward prediction.

The strict measured set now has `{summary['n_samples']}` calculable conditions. One condition is low confidence because its Kissinger R2 is below 0.95.

Current multi-target reference from the existing PS comparison report:

```json
{json.dumps(summary['current_multi_target_reference'], indent=2)}
```

## LOOCV Prediction

{table(metric_rows)}

Mean normalized MAE:

```json
{json.dumps(summary['normalized_mae'], indent=2)}
```

## EIG Comparison

```json
{json.dumps(summary['eig_stats'], indent=2)}
```

{table(rows)}

## Interpretation

Strict `H* / S*` makes the target definition physically cleaner than the old proxy version. It does not yet make four-parameter inverse reconstruction unique, because six strict conditions cannot cover the full `T1,t1,T2,t2` space. The low-R2 `50 -> 80 C` point is useful diagnostically, but it should be down-weighted or repeated before claiming original-paper-level precision.

Minimum next additions for original-paper-style training:

```text
repeat 50 -> 80 C at 4 heating rates
add 80 -> 100 C, 90 -> 100 C, and 60 -> 100 C multi-rate ARRT points
add one low-temperature long-time point, e.g. 50 -> 70 C with t2=1800 s
```

## Outputs

- Summary: `results/ps/strict_arrt_model/strict_paper_style_benchmark.json`
- Strict EIG ranking: `results/ps/strict_arrt_model/strict_paper_style_eig_top30.csv`
- Proxy EIG ranking: `results/ps/strict_arrt_model/proxy_paper_style_eig_top30.csv`
- Strict LOOCV predictions: `results/ps/strict_arrt_model/strict_paper_style_loocv_predictions.csv`
- Proxy LOOCV predictions: `results/ps/strict_arrt_model/proxy_paper_style_loocv_predictions.csv`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows()
    strict = loocv(rows, STRICT_PAPER_TARGETS)
    proxy = loocv(rows, PROXY_TARGETS)
    strict_top, strict_eig = eig_ranking(strict, STRICT_PAPER_TARGETS)
    proxy_top, proxy_eig = eig_ranking(proxy, PROXY_TARGETS)
    current_reference: dict[str, object] = {
        "source": str(CURRENT_COMPARISON_PATH.relative_to(ROOT)),
        "status": "missing",
    }
    if CURRENT_COMPARISON_PATH.exists():
        current_payload = json.loads(CURRENT_COMPARISON_PATH.read_text(encoding="utf-8"))
        current_reference = {
            "source": str(CURRENT_COMPARISON_PATH.relative_to(ROOT)),
            "status": "loaded",
            "targets": current_payload.get("current_targets", []),
            "normalized": current_payload.get("normalized_metric_summary", {}).get("current_targets", {}),
            "eig_stats": current_payload.get("current_eig_stats", {}),
            "note": "This reference uses the full PS dataset and is not directly the same six-condition strict ARRT subset.",
        }
    write_csv(OUT_DIR / "strict_paper_style_loocv_predictions.csv", strict["predictions"])  # type: ignore[arg-type]
    write_csv(OUT_DIR / "proxy_paper_style_loocv_predictions.csv", proxy["predictions"])  # type: ignore[arg-type]
    write_csv(OUT_DIR / "strict_paper_style_eig_top30.csv", strict_top)
    write_csv(OUT_DIR / "proxy_paper_style_eig_top30.csv", proxy_top)
    summary = {
        "n_samples": len(rows),
        "strict_targets": STRICT_PAPER_TARGETS,
        "proxy_targets": PROXY_TARGETS,
        "strict_paper_style": {
            "metrics": strict["metrics"],
        },
        "proxy_paper_style": {
            "metrics": proxy["metrics"],
        },
        "normalized_mae": {
            "strict_paper_style": normalized_mae(strict["metrics"]),  # type: ignore[arg-type]
            "proxy_paper_style": normalized_mae(proxy["metrics"]),  # type: ignore[arg-type]
        },
        "eig_stats": {
            "strict_paper_style": strict_eig,
            "proxy_paper_style": proxy_eig,
        },
        "current_multi_target_reference": current_reference,
        "strict_top30": strict_top,
        "proxy_top30": proxy_top,
        "low_confidence_conditions": [
            row.get("condition_key", "") for row in rows if row.get("kissinger_confidence") == "low"
        ],
        "notes": [
            "Strict H*/S* values come from multi-heating-rate Kissinger fits.",
            "Proxy H*/S* values are kept only as a baseline and should not be mixed with strict labels.",
        ],
    }
    out = OUT_DIR / "strict_paper_style_benchmark.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"output": str(out), "normalized_mae": summary["normalized_mae"], "eig_stats": summary["eig_stats"]}, indent=2))
    return out


if __name__ == "__main__":
    main()
