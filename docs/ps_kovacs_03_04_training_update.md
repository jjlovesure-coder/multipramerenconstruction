# PS Kovacs 03/04 Training Update

## Purpose

This update adds the completed DSC training batches `ps-kovacs-03.xlsx` and `ps-kovacs-04.xlsx` to the existing finite-time PS enthalpy-recovery pipeline, retrains the surrogate predictor, and refreshes the EIG-based next-experiment ranking.

The comparison below keeps the model structure, targets, grouped split rule, sample mass (`4.7 mg`), and feature extraction logic unchanged. Therefore the change mainly reflects the extra training information from the new DSC batches.

## Dataset Growth

- Before adding `ps-kovacs-03/04`: 94 usable samples, 67 train, 27 test.
- After adding `ps-kovacs-03/04`: 102 usable samples, 74 train, 28 test.
- Net increase: 8 usable samples.
- Mode balance after update: single-step = 45, two-step = 57.
- Single-step T1 coverage: 50 C: 20, 70 C: 20, 90 C: 1, 95 C: 1, 100 C: 3.
- Two-step T1 coverage: 50 C: 10, 55 C: 1, 65 C: 3, 75 C: 2, 80 C: 20, 85 C: 1, 90 C: 20.

| source_file | usable_rows |
| --- | --- |
| PS-onestep-01.xlsx | 20 |
| PS-onestep-02.xlsx | 20 |
| pskovacs.xlsx | 20 |
| twosteps.xlsx | 20 |
| ps-kovacs-01.xlsx | 10 |
| ps-single-01.xlsx | 4 |
| ps-kovacs-03.xlsx | 4 |
| ps-kovacs-04.xlsx | 4 |

![Processed samples by source file](E:/knowledgepaper/multipramereconstruction/results/ps/figures/ps_kovacs_03_04_sample_sources.png)

## Prediction Improvement

Positive relative change means MAE became worse; negative means MAE improved.

| target | before_MAE | after_MAE | relative_change | after_RMSE | after_R2 |
| --- | --- | --- | --- | --- | --- |
| delta_h_total_J_g | 0.1952 | 0.1812 | -7.2% | 0.2093 | 0.2396 |
| peak_area_J_g | 0.1511 | 0.1455 | -3.7% | 0.1657 | 0.2089 |
| peak_temperature_Tp_C | 0.3444 | 0.3458 | +0.4% | 0.3933 | 0.2756 |
| peak_height_uW | 26.9250 | 24.2559 | -9.9% | 30.2044 | -0.0678 |
| recovery_index | 0.1682 | 0.1574 | -6.4% | 0.1733 | 0.1799 |
| path_dependence_index | 0.1948 | 0.1944 | -0.2% | 0.2335 | -0.0180 |

![MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/figures/ps_kovacs_03_04_mae_comparison.png)

![MAE relative change](E:/knowledgepaper/multipramereconstruction/results/ps/figures/ps_kovacs_03_04_mae_relative_change.png)

## Interpretation

The added batches improved the most useful thermodynamic targets for inverse annealing design:

- `delta_h_total_J_g` MAE decreased from 0.1952 to 0.1812.
- `recovery_index` MAE decreased from 0.1682 to 0.1574.
- `peak_height_uW` also improved, which suggests the new batches help constrain signal-amplitude behavior.

The `Tp` MAE stayed essentially unchanged and slightly worsened by this split. This is not yet alarming because all extracted PS relaxation peaks remain clustered near about 105 C; the target has a narrow observed range, so small split changes can dominate the metric. The current data improve recovery-amplitude prediction more than peak-position prediction.

`path_dependence_index` changed only marginally. This means the new two-step batches add useful enthalpy-recovery information, but the model still has limited direct evidence for separating true path dependence from matched single-step behavior.

## EIG-Based Next Two-Step Recommendations

The refreshed EIG ranking still emphasizes uncertain/high-value two-step conditions, especially high-temperature or short-first-step combinations where the current model has relatively high recovery/path-dependence uncertainty.

| rank | T1_C | t1_s | T2_C | t2_s | EIG_score | pred_recovery_index | unc_recovery_index |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100 | 10 | 100 | 1200 | 1.931 | 0.609 | 0.239 |
| 2 | 80 | 10 | 80 | 600 | 1.842 | 0.604 | 0.240 |
| 3 | 80 | 100 | 100 | 60 | 1.754 | 0.606 | 0.235 |
| 4 | 90 | 10 | 100 | 100 | 1.780 | 0.562 | 0.239 |
| 5 | 100 | 600 | 100 | 300 | 1.768 | 0.618 | 0.210 |
| 6 | 75 | 10 | 90 | 1800 | 1.762 | 0.648 | 0.245 |
| 7 | 100 | 10 | 100 | 10 | 1.766 | 0.518 | 0.224 |
| 8 | 65 | 600 | 70 | 300 | 1.687 | 0.639 | 0.211 |

## ARRT / S* / H* Status

Strict ARRT/Kissinger calculation still cannot produce direct `S*` and `H*` training targets from the current PS dataset:

- total recovery scans inspected: 103
- annealed-state groups: 82
- calculable groups: 0
- maximum distinct heating rates in one group: 2
- minimum required distinct heating rates: 3

So the direct paper-style target route is still blocked by insufficient repeated heating-rate coverage for the same annealed state. The next improvement should either keep using physics-informed surrogate features, or deliberately add at least three heating rates for selected identical annealing conditions.

## Assessment

The prediction gain is real but modest. The main model now estimates recovery degree better (`recovery_index` MAE about 0.1574), which is the target most aligned with finite-time PS experiment design. However, the model is still not strong enough for precise inverse four-parameter prediction across the full 50-100 C space.

Recommended next step: use the updated EIG list to add a smaller, high-information validation batch, but include repeated conditions or controlled heating-rate triplets if the goal is to unlock direct `S* / H*` training.
