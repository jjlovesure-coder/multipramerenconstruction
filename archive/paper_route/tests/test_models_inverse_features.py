from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.inverse_design as inverse_design
from src.models.dataset import interpolate_by_temperature_log_time


def test_inverse_feature_row_uses_s7_interpolation() -> None:
    s7_rows = inverse_design._read_csv(inverse_design.S7)
    features = inverse_design.feature_row(365.0, 100.0, 383.0, 100.0)
    feature_map = dict(zip(inverse_design.FEATURES, features))

    expected = interpolate_by_temperature_log_time(s7_rows, 365.0, 100.0, "s_star_j_mol_k")

    assert abs(feature_map["s1_j_mol_k"] - expected) < 1e-9
