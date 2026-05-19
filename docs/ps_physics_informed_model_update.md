# PS Physics-Informed Model Update

## Purpose

This update tests whether embedding TNM/ARRT-style relaxation priors improves the finite-time PS predictor beyond the raw temperature/time RBF model.

## Model Variants

- `raw_kernel`: the existing RBF surrogate using only mode, temperature, and log-time features.
- `physics_kernel`: RBF using raw features plus Arrhenius/TNM dose, sequential state, and path-memory features.
- `physics_residual`: a ridge physics baseline trained on physics features, plus an RBF model trained only on its residual.

The `physics_kernel` feature subset and bandwidth are selected by repeated grouped cross-validation inside the training split:

```json
{
  "score": 0.846380242219281,
  "feature_set": "raw_phys_beta",
  "features": [
    "tnm_state_total_e160_b035",
    "tnm_state_total_e160_b050",
    "tnm_state_total_e160_b065",
    "tnm_path_excess_e160_b035",
    "tnm_path_excess_e160_b050",
    "tnm_path_excess_e160_b065"
  ],
  "bandwidth": 3.0
}
```

## Same-Split Metrics

The split is identical across models and grouped by source file / mode / T1. Lower MAE is better.

| target | raw_MAE | physics_kernel_MAE | physics_residual_MAE | physics_residual_R2 |
| --- | --- | --- | --- | --- |
| delta_h_total_J_g | 0.1812 | 0.1647 | 0.1677 | 0.2122 |
| peak_area_J_g | 0.1455 | 0.1194 | 0.0715 | 0.7660 |
| recovery_index | 0.1574 | 0.1516 | 0.1821 | -0.6010 |
| path_dependence_index | 0.1944 | 0.1900 | 0.3646 | -2.8933 |
| peak_temperature_Tp_C | 0.3458 | 0.3641 | 0.2059 | 0.6856 |
| peak_height_uW | 24.2559 | 18.1110 | 16.5666 | 0.4733 |

![Physics-informed MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/physics_informed/physics_informed_mae_comparison.png)

## Normalized Core-Target Error

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 0.7997237738809986,
    "median_mae_over_test_std": 0.8018631098353464
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 0.7352417313321563,
    "median_mae_over_test_std": 0.7394096670064233
  },
  "physics_residual": {
    "mean_mae_over_test_std": 0.9025302580639518,
    "median_mae_over_test_std": 0.8253160315061012
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 1.0418614040095204,
    "median_mae_over_test_std": 1.061212458326851
  }
}
```

## Interpretation

The selected `physics_kernel` lowers the mean normalized core-target error from `0.7997` to `0.7352`, a relative change of `-8.1%`. It improves all four core targets in this split: total enthalpy, peak area, recovery index, and path-dependence index.

The `physics_residual` variant is not recommended as the main model right now. It improves peak area and diagnostic peak shape, but it damages `recovery_index` and `path_dependence_index`, which are the targets that matter most for inverse annealing design. The better small-sample choice is therefore the lower-dimensional `physics_kernel`: raw annealing inputs plus TNM path/dose features selected by grouped CV.
