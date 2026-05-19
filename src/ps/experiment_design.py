"""Build the finite-time PS experiment matrix from the approved plan."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "ps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SINGLE_TEMPS_C = [50, 60, 70, 80, 90]
SINGLE_TIMES_S = [10, 60, 300, 900, 1800]

TWO_STEP_PAIRS_C = [(50, 70), (50, 80), (60, 80), (60, 90), (70, 90), (80, 100)]
TWO_STEP_TIMES_S = [(300, 60), (900, 300), (1800, 900)]

VALIDATION_ROWS = [
    ("C", "single_repeat_short", 50, 10, "", "", "repeat short-time single-step control"),
    ("C", "single_repeat_long", 90, 1800, "", "", "repeat long-time single-step control"),
    ("C", "two_step_risk_1", 50, 900, 80, 300, "placeholder for model-selected high-risk two-step point"),
    ("C", "two_step_risk_2", 70, 1800, 90, 900, "placeholder for model-selected high-risk two-step point"),
    ("C", "single_interpolation", 65, 600, "", "", "interpolation validation point"),
    ("C", "two_step_interpolation", 55, 900, 85, 300, "interpolation validation point"),
]


def build_design() -> Path:
    rows = []
    idx = 1
    for temp in SINGLE_TEMPS_C:
        for time_s in SINGLE_TIMES_S:
            rows.append({
                "experiment_id": f"A-{idx:02d}",
                "batch": "A",
                "mode": "single_step",
                "T1_C": temp,
                "t1_s": time_s,
                "T2_C": "",
                "t2_s": "",
                "max_step_time_s": 1800,
                "purpose": "single-step enthalpy recovery calibration",
            })
            idx += 1

    idx = 1
    for t1_c, t2_c in TWO_STEP_PAIRS_C:
        for t1_s, t2_s in TWO_STEP_TIMES_S:
            rows.append({
                "experiment_id": f"B-{idx:02d}",
                "batch": "B",
                "mode": "two_step",
                "T1_C": t1_c,
                "t1_s": t1_s,
                "T2_C": t2_c,
                "t2_s": t2_s,
                "max_step_time_s": 1800,
                "purpose": "finite-time path-dependence screen",
            })
            idx += 1

    for idx, (batch, label, t1_c, t1_s, t2_c, t2_s, purpose) in enumerate(VALIDATION_ROWS, start=1):
        rows.append({
            "experiment_id": f"{batch}-{idx:02d}",
            "batch": batch,
            "mode": "single_step" if t2_c == "" else "two_step",
            "T1_C": t1_c,
            "t1_s": t1_s,
            "T2_C": t2_c,
            "t2_s": t2_s,
            "max_step_time_s": 1800,
            "purpose": f"{label}: {purpose}",
        })

    path = OUT_DIR / "ps_limited_time_experiment_design.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print(build_design())

