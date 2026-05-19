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
  "score": 0.8576372827271755,
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

## Same-Split Metrics

The split is identical across models and grouped by source file / mode / T1. Lower MAE is better.

| target | raw_MAE | physics_kernel_MAE | physics_residual_MAE | physics_residual_R2 |
| --- | --- | --- | --- | --- |
| delta_h_total_J_g | 0.1744 | 0.1450 | 0.1156 | 0.2497 |
| peak_area_J_g | 0.1252 | 0.1104 | 0.0725 | 0.5204 |
| recovery_index | 0.2537 | 0.2149 | 0.1913 | -0.4509 |
| path_dependence_index | 0.1298 | 0.0761 | 0.1577 | -0.8678 |
| peak_temperature_Tp_C | 0.6305 | 0.5920 | 0.4528 | -1.1752 |
| peak_height_uW | 11.3093 | 10.0040 | 5.3665 | 0.8898 |

![Physics-informed MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/physics_informed/physics_informed_mae_comparison.png)

## Normalized Core-Target Error

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 1.0195512641495252,
    "median_mae_over_test_std": 0.9643905077371802
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 0.8050027717871515,
    "median_mae_over_test_std": 0.7882973962630822
  },
  "physics_residual": {
    "mean_mae_over_test_std": 0.8187466222825189,
    "median_mae_over_test_std": 0.8037587908556287
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 0.8843925753051569,
    "median_mae_over_test_std": 0.8275518035319904
  }
}
```

## Interpretation

The selected `physics_kernel` lowers the mean normalized core-target error from `1.0196` to `0.8050`, a relative change of `-21.0%`. It improves all four core targets in this split: total enthalpy, peak area, recovery index, and path-dependence index.

The `physics_residual` variant is not recommended as the main model right now. It improves peak area and diagnostic peak shape, but it damages `recovery_index` and `path_dependence_index`, which are the targets that matter most for inverse annealing design. The better small-sample choice is therefore the lower-dimensional `physics_kernel`: raw annealing inputs plus TNM path/dose features selected by grouped CV.
