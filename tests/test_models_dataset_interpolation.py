from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dataset import interpolate_by_temperature_log_time


def test_interpolate_by_temperature_log_time_uses_neighboring_times() -> None:
    rows = [
        {"series": "350 K", "annealing_time_s": "10", "s_star_j_mol_k": "100"},
        {"series": "350 K", "annealing_time_s": "1000", "s_star_j_mol_k": "300"},
        {"series": "380 K", "annealing_time_s": "10", "s_star_j_mol_k": "200"},
        {"series": "380 K", "annealing_time_s": "1000", "s_star_j_mol_k": "400"},
    ]

    value = interpolate_by_temperature_log_time(rows, 365.0, 100.0, "s_star_j_mol_k")

    assert 230.0 < value < 270.0
