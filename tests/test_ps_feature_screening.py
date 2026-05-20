from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.train_physics_informed_ps_model import screen_feature_names


def test_screen_feature_names_keeps_signal_and_drops_redundant_features() -> None:
    base = np.linspace(0.0, 1.0, 12)
    x = np.column_stack(
        [
            base,
            base * 1.001,
            np.sin(base * np.pi),
            np.ones_like(base),
        ]
    )
    y = np.column_stack([base * 2.0, np.sin(base * np.pi)])
    result = screen_feature_names(
        x,
        y,
        ["signal", "duplicate_signal", "curved", "constant"],
        min_features=2,
        max_features=3,
        redundancy_threshold=0.995,
    )

    assert result["selected_features"]
    assert "constant" not in result["selected_features"]
    assert not ({"signal", "duplicate_signal"} <= set(result["selected_features"]))
    assert len(result["selected_features"]) >= 2
