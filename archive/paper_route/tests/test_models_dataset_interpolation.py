from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dataset import interpolate_by_log_time
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


def test_interpolate_by_log_time_handles_midpoint_and_edges() -> None:
    rows = [
        {"annealing_time_s": "10", "h_star_kj_mol": "100"},
        {"annealing_time_s": "1000", "h_star_kj_mol": "300"},
    ]

    assert interpolate_by_log_time(rows, 100.0, "h_star_kj_mol") == 200.0
    assert interpolate_by_log_time(rows, 1.0, "h_star_kj_mol") == 100.0
    assert interpolate_by_log_time(rows, 10000.0, "h_star_kj_mol") == 300.0
