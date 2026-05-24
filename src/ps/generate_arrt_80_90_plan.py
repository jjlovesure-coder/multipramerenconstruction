"""Generate the minimal 80/90 C multi-heating-rate ARRT experiment plan."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "ps" / "ps_80_90_arrt_heating_rate_plan.csv"
MTD_PATH = ROOT / "data" / "kovactrain26111103R9-06-80-90-arrt.mtd"
DOC_PATH = ROOT / "docs" / "ps_80_90_arrt_heating_rate_plan.md"

TEMPERATURES_C = [80, 90]
ANNEAL_TIMES_S = [50, 500, 1000]
HEATING_RATES_C_MIN = [5, 10, 20, 40]
SCAN_RATE_STRINGS = {
    5: "0.008333333333333333",
    10: "0.01666666666666667",
    20: "0.03333333333333333",
    40: "0.06666666666666667",
}


def scan_rate_c_s(rate_c_min: float) -> float:
    return float(rate_c_min) / 600.0


def scan_rate_mtd_string(rate_c_min: float) -> str:
    return SCAN_RATE_STRINGS[int(rate_c_min)]


def condition_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_id = 1
    for temp_c in TEMPERATURES_C:
        for time_s in ANNEAL_TIMES_S:
            condition_id = f"ARRT-{temp_c:02d}-T{time_s:04d}"
            for rate in HEATING_RATES_C_MIN:
                rows.append({
                    "run_id": run_id,
                    "condition_id": condition_id,
                    "mode": "single_step",
                    "T1_C": temp_c,
                    "t1_s": time_s,
                    "T2_C": "",
                    "t2_s": "",
                    "purpose": "80/90 C single-step H*/S* anchor for fixed-temperature t2 inverse",
                    "final_heating_rate_C_min": rate,
                    "final_scan_rate_mtd_C_s": scan_rate_mtd_string(rate),
                    "core_or_extension": "core" if time_s in {50, 500} else "long-time extension",
                    "calculation_use": "Tp(beta) for Kissinger fit and strict ARRT S*/H* calculation",
                })
                run_id += 1
    return rows


def mtd_ramp_step(start: float, limit: float, rate: float, hold: float, sampling: float) -> str:
    return f"""object TMUMeasTempProgRampStep
Start = {start:g}
Limit = {limit:g}
Rate = {rate}
Hold = {hold:g}
Sampling = {sampling:g}
Store = True
ExtOutOn = (
1
0
0
0
0
0
0)
ExtOutFlow = (
30
0
0
0
0
0
0)
end
"""


def build_mtd(rows: list[dict[str, object]]) -> str:
    # Existing device-readable methods use a zero-based ramp-step index:
    # RampSteps=84 -> EndStep=83. The Step object below is not counted here.
    end_step = len(rows) * 3 - 1
    parts = [
        "<Method>\n",
        "Version 4\n",
        "object TMUMeasMethod\n",
        "TempProg = TempProg\n",
        "StartCond = StartCond\n",
        "EndCond = EndCond\n",
        "object TempProg: TMUMeasTempProg\n",
        "TempProgMode = 0\n",
        "Ramp = TempProg.Ramp\n",
        "Step = TempProg.Step\n",
        "Special = 0\n",
        "object Ramp: TMUMeasTempProgRamp\n",
        f"EndStep = {end_step}\n",
    ]
    for row in rows:
        temp = float(row["T1_C"])
        time_s = float(row["t1_s"])
        rate = float(row["final_scan_rate_mtd_C_s"])
        parts.append(mtd_ramp_step(200, temp, 0.1, time_s * 10.0, 1.0))
        parts.append(mtd_ramp_step(temp, 30, 0.1, 6000.0, 1.0))
        parts.append(mtd_ramp_step(30, 200, rate, 1800.0, 0.1))
    parts.extend([
        "end\n",
        "object Step: TMUMeasTempProgStep\n",
        "Limit = 100\n",
        "Rate = 0.00833333333333333\n",
        "Hold = 6000\n",
        "Sampling = 0.5\n",
        "Store = True\n",
        "StepCount = 10\n",
        "StepTemp = 10\n",
        "Soak = 10\n",
        "ExtOutOn = (\n",
        "0\n0\n0\n0\n0\n0\n0)\n",
        "ExtOutFlow = (\n",
        "0\n0\n0\n0\n0\n0\n0)\n",
        "end\n",
        "end\n",
        "object StartCond: TMUMeasStartCond\n",
        "IsTempCond = False\n",
        "TempRange = 1\n",
        "IsYCond = False\n",
        "dYRange = 1\n",
        "IsTimeWait = True\n",
        "TimeWait = 60\n",
        "IsZeroSet = True\n",
        "IsSampleWeightStart = False\n",
        "end\n",
        "object EndCond: TMUMeasEndCond\n",
        "IsMeltProtect = False\n",
        "EnableHeatCoolAutoExten = True\n",
        "end\n",
        "end\n",
    ])
    return "".join(parts)


def write_csv(rows: list[dict[str, object]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_doc(rows: list[dict[str, object]]) -> None:
    core = [row for row in rows if row["core_or_extension"] == "core"]
    text = f"""# PS 80/90 Multi-Heating-Rate ARRT Plan

## Purpose

This is the minimum heating-rate experiment set for the reduced inverse problem:

```text
fixed T1/T2 = 80/90 or 90/80, known t1 ~= 50 s or 500 s -> infer t2
```

The previous minimal inverse check showed that final heating curves alone recover `t2` only coarsely, while the digitized paper data improve strongly when Figure-3-like `H*/S*` information is added. Therefore this plan does not expand the two-step matrix. It first builds strict single-step `H*/S*` anchors at 80 C and 90 C.

## Experiment Matrix

- Temperatures: `80 C`, `90 C`
- Annealing times: `50 s`, `500 s`, `1000 s`
- Heating rates: `5, 10, 20, 40 C/min`
- Total recommended runs: `{len(rows)}`
- Core runs if time is limited: `{len(core)}` (`50 s` and `500 s` only)

Each run uses:

```text
200 C -> annealing temperature at 0.1 C/s, hold t
annealing temperature -> 30 C at 0.1 C/s, hold 600 s
30 C -> 200 C at selected heating rate
```

## Why This Is Enough For The Next Check

The original paper used single-step heating-rate data to build the kinetic `H*/S*` coordinate. For the current PS question, the most economical analogue is to measure single-step 80 C and 90 C anchors first, then reuse those `time -> H*/S*` relationships as priors when `T1/T2` are fixed and only `t2` is unknown.

The `1000 s` points are marked as long-time extension. They stabilize the long-time end of the `H*/S*` curve and are useful because the existing 80/90 two-step data include `t2 ~= 1000 s`.

## Outputs

- CSV plan: `data/ps/ps_80_90_arrt_heating_rate_plan.csv`
- MTD method: `data/kovactrain26111103R9-06-80-90-arrt.mtd`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    rows = condition_rows()
    write_csv(rows)
    MTD_PATH.write_text(build_mtd(rows), encoding="utf-8")
    write_doc(rows)
    print({
        "csv": str(CSV_PATH),
        "mtd": str(MTD_PATH),
        "doc": str(DOC_PATH),
        "n_runs": len(rows),
    })


if __name__ == "__main__":
    main()
