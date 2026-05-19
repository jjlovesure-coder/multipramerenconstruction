"""Generate the PS DSC method with correction single-step runs plus two-step runs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "kovactrain26111103R9-01.mtd"

COOL_RATE = "0.1"
SCAN_RATE = "0.0166666666666667"


def hold_value(seconds: float) -> str:
    value = seconds * 10.0
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def temp_value(temp: float) -> str:
    if abs(temp - round(temp)) < 1e-9:
        return str(int(round(temp)))
    return str(temp)


def step(start: float, limit: float, rate: str, hold_s: float, sampling: str = "1") -> str:
    return f"""object TMUMeasTempProgRampStep
Start = {temp_value(start)}
Limit = {temp_value(limit)}
Rate = {rate}
Hold = {hold_value(hold_s)}
Sampling = {sampling}
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


def single_step(temp: float, time_s: float) -> list[str]:
    return [
        step(200, temp, COOL_RATE, time_s, "1"),
        step(temp, 30, COOL_RATE, 600, "1"),
        step(30, 200, SCAN_RATE, 180, "0.1"),
    ]


def two_step(t1: float, time1_s: float, t2: float, time2_s: float) -> list[str]:
    return [
        step(200, t1, COOL_RATE, time1_s, "1"),
        step(t1, t2, COOL_RATE, time2_s, "1"),
        step(t2, 30, COOL_RATE, 600, "1"),
        step(30, 200, SCAN_RATE, 180, "0.1"),
    ]


def build_method() -> str:
    ramp_steps: list[str] = []

    # First three runs correct the previous 95 C, 100 s nonstandard scan issue.
    for _ in range(3):
        ramp_steps.extend(single_step(95, 100.02))

    # Full optimized two-step EIG plan after the new single-step update.
    two_step_plan = [
        (65, 10, 100, 1200),
        (50, 10, 80, 1800),
        (50, 10, 100, 10),
        (50, 1800, 100, 10),
        (55, 600, 90, 300),
        (50, 10, 70, 1800),
        (50, 10, 100, 1800),
        (50, 10, 85, 60),
        (75, 10, 100, 1800),
        (55, 30, 90, 900),
        (65, 10, 95, 1800),
        (65, 100, 100, 10),
        (50, 300, 80, 100),
        (100, 10, 100, 1800),
        (65, 300, 100, 1200),
        (50, 100, 100, 100),
        (60, 10, 95, 100),
        (50, 10, 60, 1800),
    ]
    for condition in two_step_plan:
        ramp_steps.extend(two_step(*condition))

    end_step = len(ramp_steps) - 1
    body = "".join(ramp_steps)
    return f"""<Method>
Version 4
object TMUMeasMethod
TempProg = TempProg
StartCond = StartCond
EndCond = EndCond
object TempProg: TMUMeasTempProg
TempProgMode = 0
Ramp = TempProg.Ramp
Step = TempProg.Step
Special = 0
object Ramp: TMUMeasTempProgRamp
EndStep = {end_step}
{body}end
object Step: TMUMeasTempProgStep
Limit = 100
Rate = 0.00833333333333333
Hold = 6000
Sampling = 0.5
Store = True
StepCount = 10
StepTemp = 10
Soak = 10
ExtOutOn = (
0
0
0
0
0
0
0)
ExtOutFlow = (
0
0
0
0
0
0
0)
end
end
object StartCond: TMUMeasStartCond
IsTempCond = False
TempRange = 1
IsYCond = False
dYRange = 1
IsTimeWait = True
TimeWait = 60
IsZeroSet = True
IsSampleWeightStart = False
end
object EndCond: TMUMeasEndCond
IsMeltProtect = False
EnableHeatCoolAutoExten = True
end
end
"""


def main() -> None:
    OUT_PATH.write_text(build_method(), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
