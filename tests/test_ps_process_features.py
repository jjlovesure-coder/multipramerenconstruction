from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.dsc_processing import Segment, extract_pre_scan_process_features, process_feature_names


def test_pre_scan_process_features_exclude_final_heating_segment() -> None:
    segments = [
        Segment(1, 200.0, 80.0, 60.0, 1.0, 1.0),
        Segment(2, 80.0, 30.0, 60.0, 0.5, 1.0),
        Segment(3, 30.0, 200.0, 10.0, 0.0, 1.0),
    ]
    bounds = [(0.0, 3.0), (3.0, 4.333333333333333), (4.333333333333333, 21.333333333333332)]
    signal = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.5, 3.2, 4.2, 4.4, 5.0, 6.0],
            "Temp_C": [200.0, 140.0, 80.0, 70.0, 30.0, 31.0, 40.0, 50.0],
            "DSC": [1.0, 2.0, 3.0, 2.5, 2.0, 10000.0, 10000.0, 10000.0],
            "DDSC": [0.0] * 8,
        }
    )

    features = extract_pre_scan_process_features(signal, segments[:2], bounds[:2])

    assert set(process_feature_names()).issubset(features)
    assert features["process_pre_scan_end_dsc_uW"] == 2.0
    assert features["process_total_mean_uW"] < 4.0
    assert features["process_total_abs_integral_uW_min"] < 20.0
