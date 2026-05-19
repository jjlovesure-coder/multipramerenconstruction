"""Report inverse-design and EIG changes after switching to the PS physics kernel."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "docs" / "ps_physics_kernel_entrypoint_update.md"
OLD_INVERSE = ROOT / "results" / "ps" / "predictions" / "ps_inverse_design_top10_before_physics_kernel.json"
NEW_INVERSE = ROOT / "results" / "ps" / "predictions" / "ps_inverse_design_top10.json"
OLD_EIG = ROOT / "data" / "ps" / "ps_eig_next_twostep_experiments_before_physics_kernel.csv"
NEW_EIG = ROOT / "data" / "ps" / "ps_eig_next_twostep_experiments.csv"
MODEL_SUMMARY = ROOT / "results" / "ps" / "physics_informed" / "physics_informed_model_comparison.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def condition(row: dict[str, object]) -> str:
    return f"{row['T1_C']}C {row['t1_s']}s -> {row['T2_C']}C {row['t2_s']}s"


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def inverse_rows(old: list[dict], new: list[dict], limit: int = 8) -> list[list[str]]:
    rows = []
    for idx in range(min(limit, len(old), len(new))):
        old_row = old[idx]
        new_row = new[idx]
        rows.append(
            [
                str(idx + 1),
                condition(old_row),
                fmt(old_row.get("pred_recovery_index")),
                fmt(old_row.get("loss")),
                condition(new_row),
                fmt(new_row.get("pred_recovery_index")),
                fmt(new_row.get("loss")),
            ]
        )
    return rows


def eig_rows(old: list[dict[str, str]], new: list[dict[str, str]], limit: int = 8) -> list[list[str]]:
    rows = []
    for idx in range(min(limit, len(old), len(new))):
        old_row = old[idx]
        new_row = new[idx]
        rows.append(
            [
                str(idx + 1),
                condition(old_row),
                fmt(old_row.get("eig_score")),
                fmt(old_row.get("pred_recovery_index")),
                condition(new_row),
                fmt(new_row.get("eig_score")),
                fmt(new_row.get("pred_recovery_index")),
            ]
        )
    return rows


def main() -> Path:
    old_inverse = read_json(OLD_INVERSE)["results"]
    new_inverse = read_json(NEW_INVERSE)["results"]
    old_eig = read_csv(OLD_EIG)
    new_eig = read_csv(NEW_EIG)
    model = read_json(MODEL_SUMMARY)

    selected = model["selected_physics_kernel"]
    normalized = model["normalized_core"]
    report = f"""# PS Physics Kernel Entrypoint Update

## What Changed

The default PS inverse-design and EIG entrypoints now use:

```text
results/ps/models/ps_physics_informed_kernel_model.json
```

This model uses raw annealing inputs plus selected TNM/ARRT-inspired features. The selected feature set is `{selected['feature_set']}` with bandwidth `{selected['bandwidth']}`.

Same-split core-target normalized error:

```json
{json.dumps(normalized, indent=2)}
```

## Inverse Prediction Change

Target used for comparison: `recovery_index = 0.8`.

{table(["rank", "old condition", "old pred RI", "old loss", "physics condition", "physics pred RI", "physics loss"], inverse_rows(old_inverse, new_inverse))}

Interpretation: the old inverse model could numerically hit `recovery_index ~= 0.8`, but it did so by staying near low-temperature neighboring conditions. The physics kernel is more conservative and predicts that high recovery in the current finite-time window needs a stronger temperature jump and long high-temperature second step.

## EIG Change

{table(["rank", "old EIG condition", "old score", "old pred RI", "physics EIG condition", "physics score", "physics pred RI"], eig_rows(old_eig, new_eig))}

Interpretation: EIG is no longer dominated by repeated same-temperature high-temperature points. It now gives higher priority to temperature-jump paths and path-memory contrasts, which is closer to the purpose of the two-step PS experiments.

## Practical Recommendation

Use the physics-kernel EIG list for the next round rather than the old EIG list. For inverse design, treat the new top conditions as conservative feasible candidates: they may not hit `recovery_index = 0.8` exactly, but their recommendations are physically more defensible under sparse data.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH)}, indent=2))
    return REPORT_PATH


if __name__ == "__main__":
    main()
