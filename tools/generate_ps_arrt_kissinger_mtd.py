"""Generate DSC method for strict ARRT/Kissinger S*/H* training data.

Each annealed state is repeated at several final heating rates so that Tp(beta)
can be used in a Kissinger fit before calculating H* and S*.
"""

from __future__ import annotations

import csv
from pathlib import Path

from generate_ps_twostep_mtd import ROOT, method_text, single_step, two_step


OUT_PATH = ROOT / "data" / "kovactrain26111103R9-05-arrt-kissinger.mtd"
PLAN_PATH = ROOT / "data" / "ps" / "ps_arrt_kissinger_mtd_plan.csv"

HEATING_RATES_C_MIN = [5, 10, 20, 40]

# Representative states chosen from current PS model/EIG diagnostics.
CONDITIONS = [
    {
        "condition_id": "ARRT-S01",
        "mode": "single_step",
        "T1_C": 70,
        "t1_s": 300,
        "T2_C": "",
        "t2_s": "",
        "purpose": "single-step mid-temperature anchor",
    },
    {
        "condition_id": "ARRT-S02",
        "mode": "single_step",
        "T1_C": 90,
        "t1_s": 100,
        "T2_C": "",
        "t2_s": "",
        "purpose": "single-step high-temperature short-time anchor",
    },
    {
        "condition_id": "ARRT-S03",
        "mode": "single_step",
        "T1_C": 95,
        "t1_s": 100.02,
        "T2_C": "",
        "t2_s": "",
        "purpose": "complete existing 95 C 100 s multi-rate condition",
    },
    {
        "condition_id": "ARRT-D01",
        "mode": "two_step",
        "T1_C": 50,
        "t1_s": 10,
        "T2_C": 80,
        "t2_s": 1800,
        "purpose": "large old-model error two-step condition",
    },
    {
        "condition_id": "ARRT-D02",
        "mode": "two_step",
        "T1_C": 50,
        "t1_s": 10,
        "T2_C": 100,
        "t2_s": 1800,
        "purpose": "high-temperature second-step EIG condition",
    },
    {
        "condition_id": "ARRT-D03",
        "mode": "two_step",
        "T1_C": 65,
        "t1_s": 10,
        "T2_C": 100,
        "t2_s": 1200,
        "purpose": "measured EIG two-step condition with high uncertainty",
    },
]


def scan_rate_value(heating_rate_c_min: float) -> str:
    return f"{heating_rate_c_min / 600.0:.16g}"


def replace_final_scan_rate(steps: list[str], heating_rate_c_min: float) -> list[str]:
    """Patch the final 30 -> 200 scan rate while preserving local method style."""
    patched = list(steps)
    patched[-1] = patched[-1].replace(
        "Rate = 0.0166666666666667",
        f"Rate = {scan_rate_value(heating_rate_c_min)}",
    )
    return patched


def condition_steps(condition: dict, heating_rate_c_min: float) -> list[str]:
    if condition["mode"] == "single_step":
        steps = single_step(float(condition["T1_C"]), float(condition["t1_s"]))
    else:
        steps = two_step(
            float(condition["T1_C"]),
            float(condition["t1_s"]),
            float(condition["T2_C"]),
            float(condition["t2_s"]),
        )
    return replace_final_scan_rate(steps, heating_rate_c_min)


def build_plan_rows() -> list[dict[str, object]]:
    rows = []
    run_id = 1
    for condition in CONDITIONS:
        for heating_rate in HEATING_RATES_C_MIN:
            rows.append(
                {
                    "run_id": run_id,
                    **condition,
                    "final_heating_rate_C_min": heating_rate,
                    "final_scan_rate_mtd": scan_rate_value(heating_rate),
                    "calculation_use": "Tp(beta) for Kissinger fit and ARRT S*/H* calculation",
                }
            )
            run_id += 1
    return rows


def build_method() -> str:
    ramp_steps: list[str] = []
    for condition in CONDITIONS:
        for heating_rate in HEATING_RATES_C_MIN:
            ramp_steps.extend(condition_steps(condition, heating_rate))
    return method_text(ramp_steps)


def write_plan() -> None:
    rows = build_plan_rows()
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_PATH.write_text(build_method(), encoding="utf-8")
    write_plan()
    print(OUT_PATH)
    print(PLAN_PATH)


if __name__ == "__main__":
    main()
