from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.inverse_design import pareto_front


def test_pareto_front_keeps_non_dominated_candidates() -> None:
    rows = [
        {"loss": 0.5, "total_anneal_time_s": "100", "mean_uncertainty": 0.2, "abs_path_dependence": 0.1},
        {"loss": 0.4, "total_anneal_time_s": "200", "mean_uncertainty": 0.2, "abs_path_dependence": 0.1},
        {"loss": 0.6, "total_anneal_time_s": "300", "mean_uncertainty": 0.5, "abs_path_dependence": 0.2},
    ]

    front = pareto_front(rows)

    assert len(front) == 2
    assert rows[2] not in front
