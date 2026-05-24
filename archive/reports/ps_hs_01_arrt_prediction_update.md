# PS hs-01 ARRT / Prediction Update

## Purpose

`data/dsc/ps-hs-01.xlsx` adds multi-heating-rate DSC scans for representative
PS annealed states. This file is used to estimate PS-specific `H* / S*` by
Kissinger / absolute reaction-rate theory, then to update the physics-informed
four-parameter annealing predictor.

## Data Parsed

The file contains three single-step annealing states with four heating rates:

| annealing state | heating rates |
| --- | --- |
| 70 C, 300 s | 5, 10, 20, 40 C/min |
| 90 C, 100 s | 5, 10, 20, 40 C/min |
| 95 C, 100 s | 5, 10, 20, 40 C/min |

The 10 C/min scans are also added to the regular PS training table, increasing
the usable finite-time dataset from 102 to 105 rows.

## Tp Extraction Adjustment

For ARRT/Kissinger fitting, `Tp` is now selected from the high-temperature
relaxation peak in the 95-125 C window. This matters because the 5 C/min scans
at 70 C and 90 C contain a lower-temperature opposite-sign feature near 88 C.
Using the old absolute-peak rule selected that lower-temperature feature and
degraded the Kissinger fit.

The corrected high-temperature `Tp` values are monotonic with heating rate.

## Strict H* / S* Results

| condition | H* kJ/mol | S* J/mol/K | Kissinger R2 |
| --- | ---: | ---: | ---: |
| 70 C, 300 s | 427.25 | 910.21 | 0.9856 |
| 90 C, 100 s | 540.67 | 1213.57 | 0.9925 |
| 95 C, 100 s | 455.42 | 986.87 | 0.9889 |

These values are high compared with the earlier hand-set `160 kJ/mol` TNM basis.
The physics feature grid was therefore expanded with a PS-specific high-energy
basis:

```text
430, 460, 540 kJ/mol
```

## Model Change

The physics-informed model now searches the original TNM/ARRT basis and the new
PS hs-01 basis during grouped cross-validation. The selected model is:

```json
{
  "feature_set": "raw_phys_ps_hs",
  "features": [
    "tnm_state_total_e430_b050",
    "tnm_state_total_e460_b050",
    "tnm_state_total_e540_b050",
    "tnm_path_excess_e430_b050",
    "tnm_path_excess_e460_b050",
    "tnm_path_excess_e540_b050"
  ],
  "bandwidth": 1.4
}
```

## Same-Split Prediction Change

The new dataset changes the grouped train/test split, so the absolute numbers
should not be compared directly with the previous 102-row split. On the current
105-row split, the PS-specific physics kernel improves over the raw kernel:

| model | mean core MAE / test std |
| --- | ---: |
| raw kernel | 1.0196 |
| physics kernel with hs-01 basis | 0.8050 |
| physics residual | 0.8187 |

Core target MAE:

| target | raw kernel | physics kernel |
| --- | ---: | ---: |
| delta_h_total_J_g | 0.1744 | 0.1450 |
| peak_area_J_g | 0.1252 | 0.1104 |
| recovery_index | 0.2537 | 0.2149 |
| path_dependence_index | 0.1298 | 0.0761 |

The largest practical improvement is on `path_dependence_index`, which is the
closest current target to memory-effect behavior.

## Interpretation

The new data are useful, but they do not yet fully solve the four-parameter
prediction problem. There are only three strict `H* / S*` annealed states, all
single-step. That is enough to calibrate the activation-energy scale of the
physics features, but not enough to train `H* / S*` as independent supervised
targets over the full two-step condition space.

The recommended next measurements are the same multi-rate triplets/quartets for
two-step representative states, especially low-to-high and high-to-low path
pairs. Those will directly constrain whether `path_dependence_index` is a true
memory effect rather than only accumulated relaxation dose.

## Outputs

- `results/ps/arrt_kissinger/arrt_kissinger_results.csv`
- `results/ps/arrt_kissinger/all_recovery_scan_features.csv`
- `results/ps/physics_informed/physics_informed_model_comparison.json`
- `results/ps/models/ps_physics_informed_kernel_model.json`
- `results/ps/predictions/ps_inverse_design_top10.json`
- `data/ps/ps_eig_next_experiments.csv`

