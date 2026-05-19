"""Generate the next-round prioritized PS two-step DSC method."""

from __future__ import annotations

from pathlib import Path

from generate_ps_twostep_mtd import ROOT, method_text, two_step


OUT_PATH = ROOT / "data" / "kovactrain26111103R9-04-next-round.mtd"


def build_method() -> str:
    priority_plan = [
        (50, 10, 65, 1200),
        (50, 10, 75, 10),
        (65, 10, 65, 1800),
        (85, 10, 100, 1200),
        (75, 10, 100, 10),
        (50, 100, 70, 60),
        (65, 10, 85, 600),
        (50, 10, 70, 300),
    ]
    ramp_steps: list[str] = []
    for condition in priority_plan:
        ramp_steps.extend(two_step(*condition))
    return method_text(ramp_steps)


def main() -> None:
    OUT_PATH.write_text(build_method(), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
