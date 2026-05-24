from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ps.dsc_processing import RAW_FILES, Segment, extract_features, extract_pre_scan_process_features, process_feature_names


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


def test_extract_features_uses_70_to_105_c_main_window_and_preserves_legacy_window() -> None:
    scan = pd.DataFrame(
        {
            "Temp_C": [40.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 120.0, 140.0, 160.0],
            "DSC_corrected_uW": [1000.0, 1000.0, 0.0, -2.0, -5.0, -8.0, -10.0, -4.0, 0.0, 0.0, -100.0, -40.0, 0.0],
        }
    )

    features = extract_features(scan, noise_sigma_uW=0.1, heating_rate_c_min=10.0)

    assert features["feature_integration_low_C"] == 70.0
    assert features["feature_integration_high_C"] == 105.0
    assert "delta_h_total_40_160_J_g" in features
    assert abs(float(features["delta_h_total_40_160_J_g"])) > abs(float(features["delta_h_total_J_g"]))
    assert float(features["peak_temperature_Tp_C"]) <= 105.0


def test_extract_features_integrates_sample_minus_reference_without_extra_baseline() -> None:
    scan = pd.DataFrame(
        {
            "Temp_C": [40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 105.0],
            "DSC_corrected_uW": [1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    features = extract_features(scan, noise_sigma_uW=0.1, heating_rate_c_min=10.0)

    assert float(features["delta_h_total_rel"]) == 350.0


def test_raw_file_registry_uses_original_reference_and_90c_500s_repeat() -> None:
    assert RAW_FILES["ref"].name == "ps-refori-01.xlsx"
    assert RAW_FILES["single_90_500_repeat"].name == "ps-single-500hs-01.xlsx"
