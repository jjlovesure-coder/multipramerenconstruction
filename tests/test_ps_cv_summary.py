from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.train_physics_informed_ps_model import summarize_cv_scores


def test_summarize_cv_scores_reports_distribution() -> None:
    summary = summarize_cv_scores([1.0, 2.0, 3.0], seeds=[1, 2], n_folds=5)

    assert summary["n_scores"] == 3
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert summary["p10"] < summary["p90"]
    assert summary["seeds"] == [1, 2]
    assert summary["n_folds"] == 5
