# PS Calibrated TNM Condition Error Analysis

## Current Model

Selected parameters:

```json
{
  "activation_energy_kj_mol": 220.0,
  "beta": 0.9,
  "log10_tau_ref_s": 2.0,
  "nonlinearity_x": 1.0,
  "initial_fictive_temperature_k": 413.15,
  "tau_ref_s": 100.0,
  "preexponential_A_s": 1.6008513042763764e-29,
  "activation_entropy_j_mol_k": 304.4961898086539
}
```

This report uses the current calibrated TNM fit-match model and evaluates all processed PS conditions. The split rows match the training script's grouped split; the all-sample rows are curve-matching diagnostics, not an external validation claim.

## Same-Split Test Metrics

```json
{
  "delta_h_total_J_g": {
    "mae": 0.09695916776620407,
    "rmse": 0.11476483677205215,
    "bias": -0.0024255418048929425,
    "r2": 0.590913714282737,
    "target_std": 0.17943261362861795
  },
  "peak_area_J_g": {
    "mae": 0.055051350838837755,
    "rmse": 0.07451197621421575,
    "bias": -0.005712755622550063,
    "r2": 0.730866716877936,
    "target_std": 0.1436290795464113
  },
  "recovery_index": {
    "mae": 0.1552537144663893,
    "rmse": 0.17398237465047484,
    "bias": -0.11312724117063461,
    "r2": 0.23247366739296427,
    "target_std": 0.19859056525891283
  },
  "path_dependence_index": {
    "mae": 0.08313409940265708,
    "rmse": 0.11467938304124466,
    "bias": -0.006958387520956678,
    "r2": 0.284904970263832,
    "target_std": 0.13561366969300728
  }
}
```

Normalized core error: `0.5796`.

## Error By Split

| group | n | delta_h_total_J_g_mae | peak_area_J_g_mae | recovery_index_mae | path_dependence_index_mae |
|---|---:|---:|---:|---:|---:|
| test | 6 | 0.0970 | 0.0551 | 0.1553 | 0.0831 |
| train | 99 | 0.1016 | 0.0602 | 0.0901 | 0.0533 |

## Error By Mode

| group | n | delta_h_total_J_g_mae | peak_area_J_g_mae | recovery_index_mae | path_dependence_index_mae |
|---|---:|---:|---:|---:|---:|
| single_step | 48 | 0.1311 | 0.0847 | 0.0998 | 0.0282 |
| two_step | 57 | 0.0762 | 0.0390 | 0.0888 | 0.0775 |

## Error By T1 Interval

| group | n | delta_h_total_J_g_mae | peak_area_J_g_mae | recovery_index_mae | path_dependence_index_mae |
|---|---:|---:|---:|---:|---:|
| 50-60 C | 31 | 0.1085 | 0.0728 | 0.0959 | 0.0385 |
| 65-75 C | 26 | 0.1302 | 0.0803 | 0.1083 | 0.0374 |
| 80-90 C | 43 | 0.0749 | 0.0355 | 0.0809 | 0.0792 |
| 95-100 C | 5 | 0.1335 | 0.0834 | 0.1181 | 0.0399 |

## Error By t1 Interval

| group | n | delta_h_total_J_g_mae | peak_area_J_g_mae | recovery_index_mae | path_dependence_index_mae |
|---|---:|---:|---:|---:|---:|
| 10-100 s | 63 | 0.1051 | 0.0628 | 0.1016 | 0.0550 |
| 100-600 s | 37 | 0.0870 | 0.0475 | 0.0763 | 0.0603 |
| >600 s | 5 | 0.1594 | 0.1152 | 0.1258 | 0.0146 |

## Key Interpretation

- Adding the TNM fictive-temperature basis and explicit `H* / S*` metadata improves the current fit-match split versus the previous repeated-CV-only TNM model: normalized core MAE is now `0.5796`.
- The selected `H* = 220.0 kJ/mol` is consistent with the lower activation-enthalpy range seen in `twoannealing` TNM fits, rather than the previous grid floor at `300 kJ/mol`.
- The derived `S* = 304.5 J/mol/K` is now recorded in the model file, so inverse-design scoring can use a physically interpretable kinetic coordinate instead of only opaque `tau_ref`.
- Repeated grouped-CV still shows instability, so this should be treated as a calibrated forward-fit improvement first; four-parameter inverse reconstruction still needs additional constraints or targeted experiments.

## TNM Inverse Reconstruction Check

The new TNM-forward inverse search uses the calibrated `H* / S*` kinetic basis and searches a finite two-step candidate grid.

```json
{
  "n_two_step_samples": 57,
  "candidate_grid_size": 9801,
  "mae_T1_C": 14.8246,
  "mae_T2_C": 16.1404,
  "mae_log10_t1": 0.7787,
  "mae_log10_t2": 0.7585,
  "median_target_loss": 0.1208
}
```

Interpretation: the TNM forward fit is now much better, but inverse four-parameter identifiability is still weak. The small median target loss means many different schedules can reproduce similar final enthalpy-recovery targets, so the inverse problem remains many-to-one unless we add stronger constraints, repeated path anchors, or extra observables.

## Outputs

- Per-sample residuals: `results/ps/tnm_calibrated/tnm_calibrated_all_condition_residuals.csv`
- Grouped error summary: `results/ps/tnm_calibrated/tnm_calibrated_condition_error_summary.csv`
- TNM inverse reconstruction: `results/ps/tnm_calibrated/tnm_inverse_reconstruction_by_sample.csv`
