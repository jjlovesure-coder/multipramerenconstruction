from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.generate_arrt_80_90_plan import build_mtd, condition_rows


def test_generated_mtd_endstep_matches_ramp_step_count() -> None:
    text = build_mtd(condition_rows())
    ramp_steps = len(re.findall(r"^object TMUMeasTempProgRampStep$", text, flags=re.MULTILINE))
    match = re.search(r"^EndStep = (\d+)$", text, flags=re.MULTILINE)

    assert match is not None
    assert int(match.group(1)) == ramp_steps - 1


def test_generated_mtd_uses_device_rate_precision() -> None:
    text = build_mtd(condition_rows())

    assert "Rate = 0.01666666666666667\n" in text
    assert "Rate = 0.06666666666666667\n" in text
