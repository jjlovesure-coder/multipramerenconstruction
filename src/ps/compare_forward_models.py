"""Unified forward-model comparison on repeated grouped CV evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.ps.dsc_processing import ROOT, build_ps_dataset
from src.ps.train_ps_model import coverage_summary, read_rows, repeated_grouped_cv_summary


OUT_DIR = ROOT / "results" / "ps" / "forward_model_comparison"
SUMMARY_PATH = OUT_DIR / "forward_model_repeated_cv_summary.json"
REPORT_PATH = ROOT / "docs" / "ps_forward_model_repeated_cv_comparison.md"
PHYSICS_SUMMARY_PATH = ROOT / "results" / "ps" / "physics_informed" / "physics_informed_model_comparison.json"
TNM_MODEL_PATH = ROOT / "results" / "ps" / "models" / "ps_calibrated_tnm_model.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _extract_physics_cv(payload: dict) -> dict:
    selected = payload.get("selected_physics_kernel", {})
    if isinstance(selected, dict):
        summary = selected.get("selected_cv_summary") or selected.get("cv_summary") or {}
        return {
            "score_mean": summary.get("mean", selected.get("score")),
            "score_std": summary.get("std", selected.get("cv_core_std")),
            "score_median": summary.get("median"),
            "n_scores": summary.get("n_scores"),
            "n_folds": summary.get("n_folds"),
            "feature_set": selected.get("feature_set"),
            "kernel": selected.get("kernel"),
            "bandwidth": selected.get("bandwidth"),
        }
    return {}


def _extract_tnm_cv(payload: dict) -> dict:
    return payload.get("repeated_grouped_cv", {}).get("predictive_unconstrained", {})


def build_summary() -> dict[str, object]:
    dataset_path = build_ps_dataset()
    rows = read_rows(dataset_path)
    raw_cv = repeated_grouped_cv_summary(rows)
    physics_payload = _read_json(PHYSICS_SUMMARY_PATH)
    tnm_payload = _read_json(TNM_MODEL_PATH)
    return {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "comparison_policy": "Use repeated grouped CV as primary evidence; same-split n=6 results are diagnostic only.",
        "coverage_summary": coverage_summary(rows),
        "models": {
            "raw_kernel": raw_cv,
            "physics_informed_kernel": _extract_physics_cv(physics_payload),
            "tnm_calibrated": _extract_tnm_cv(tnm_payload),
        },
        "notes": [
            "If a model block is empty, run its training script first and re-run this comparison.",
            "Do not rank models by a single grouped split with six test samples.",
        ],
    }


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.4g}" if math.isfinite(number) else "NA"


def write_report(summary: dict[str, object]) -> None:
    models = summary["models"]  # type: ignore[assignment]
    rows = []
    for name, block in models.items():  # type: ignore[union-attr]
        rows.append(f"| `{name}` | {_fmt(block.get('score_mean', block.get('mean')))} | {_fmt(block.get('score_std', block.get('std')))} | {_fmt(block.get('n_scores'))} |")
    text = f"""# PS Forward Model Repeated-CV Comparison

## Policy

{summary['comparison_policy']}

## Repeated Grouped CV

| model | mean normalized score | std | n scores |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Coverage

```json
{json.dumps(summary['coverage_summary'], indent=2)}
```

## Interpretation

This comparison exists to prevent over-reading one small test split. Forward prediction claims should cite repeated grouped CV mean/std and then use same-split plots only as diagnostics.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"output": str(SUMMARY_PATH), "models": summary["models"]}, indent=2))
    return SUMMARY_PATH


if __name__ == "__main__":
    main()
