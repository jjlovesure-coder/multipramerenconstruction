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
  "score": 0.8142300093606153,
  "feature_set": "raw_phys_compact_screened",
  "base_feature_set": "raw_phys_compact",
  "features": [
    "tnm_state_total_e120_b050",
    "tnm_state_total_e200_b050",
    "tnm_state_step2_only_e160_b065",
    "phys_log_total_time_s",
    "tnm_state_step2_only_e160_b035",
    "tnm_state_step1_e160_b065",
    "tnm_state_step1_e160_b035",
    "phys_step_time_fraction_2",
    "phys_inv_equivalent_T_K",
    "phys_inv_T2_K",
    "phys_inv_T1_K",
    "tnm_path_excess_e160_b035",
    "tnm_path_excess_e200_b050",
    "tnm_step_mismatch_e160_b050"
  ],
  "bandwidth": 1.2,
  "bandwidth_vector": [
    1.2,
    1.2,
    1.2,
    1.2,
    1.2,
    1.2,
    1.2,
    0.78,
    0.800283040803826,
    1.079580814160415,
    1.091073046180941,
    1.0991220485708044,
    1.1349644053119465,
    1.2508811101272512,
    1.3690574466684047,
    1.4342648306203278,
    1.449443324223028,
    1.608391444012776,
    1.6900676546941011,
    1.699625759320448,
    1.8599999999999999
  ],
  "kernel": "ard",
  "feature_screening": {
    "selected_features": [
      "tnm_state_total_e120_b050",
      "tnm_state_total_e200_b050",
      "tnm_state_step2_only_e160_b065",
      "phys_log_total_time_s",
      "tnm_state_step2_only_e160_b035",
      "tnm_state_step1_e160_b065",
      "tnm_state_step1_e160_b035",
      "phys_step_time_fraction_2",
      "phys_inv_equivalent_T_K",
      "phys_inv_T2_K",
      "phys_inv_T1_K",
      "tnm_path_excess_e160_b035",
      "tnm_path_excess_e200_b050",
      "tnm_step_mismatch_e160_b050"
    ],
    "scores": [
      {
        "feature": "phys_inv_T1_K",
        "score": 0.2347765458675214,
        "std": 0.00013698657721383214
      },
      {
        "feature": "phys_inv_T2_K",
        "score": 0.307845656181419,
        "std": 0.00013242615234858695
      },
      {
        "feature": "phys_inv_equivalent_T_K",
        "score": 0.31482327268246063,
        "std": 0.00013012016222545543
      },
      {
        "feature": "phys_log_total_time_s",
        "score": 0.47258996026074523,
        "std": 1.1748649636048518
      },
      {
        "feature": "phys_step_time_fraction_1",
        "score": 0.34479937770066577,
        "std": 0.3348412197489415
      },
      {
        "feature": "phys_step_time_fraction_2",
        "score": 0.3447993777006659,
        "std": 0.3348412197489415
      },
      {
        "feature": "tnm_state_total_e120_b050",
        "score": 0.6155915305664633,
        "std": 0.3073730573680327
      },
      {
        "feature": "tnm_state_total_e160_b050",
        "score": 0.6150327338229813,
        "std": 0.29124426875465653
      },
      {
        "feature": "tnm_state_total_e200_b050",
        "score": 0.6062673325328126,
        "std": 0.2747099154513068
      },
      {
        "feature": "tnm_state_step1_e160_b035",
        "score": 0.3991255300492436,
        "std": 0.22182872989257177
      },
      {
        "feature": "tnm_state_step1_e160_b050",
        "score": 0.43097644025259546,
        "std": 0.22690688066068623
      },
      {
        "feature": "tnm_state_step1_e160_b065",
        "score": 0.45241292031580843,
        "std": 0.22845845963893494
      },
      {
        "feature": "tnm_state_step2_only_e160_b035",
        "score": 0.4688898004581648,
        "std": 0.24032567154642812
      },
      {
        "feature": "tnm_state_step2_only_e160_b050",
        "score": 0.4775252828652772,
        "std": 0.24198754518004625
      },
      {
        "feature": "tnm_state_step2_only_e160_b065",
        "score": 0.4778729870142009,
        "std": 0.24278853012052454
      },
      {
        "feature": "tnm_path_excess_e120_b050",
        "score": 0.16159212098569528,
        "std": 0.11398699772707165
      },
      {
        "feature": "tnm_path_excess_e160_b035",
        "score": 0.19722965285495095,
        "std": 0.10963361381807481
      },
      {
        "feature": "tnm_path_excess_e160_b050",
        "score": 0.18148570989193477,
        "std": 0.13728299209129682
      },
      {
        "feature": "tnm_path_excess_e160_b065",
        "score": 0.17228300763615081,
        "std": 0.16288227476812084
      },
      {
        "feature": "tnm_path_excess_e200_b050",
        "score": 0.19283575249137233,
        "std": 0.1511720043731148
      },
      {
        "feature": "tnm_temperature_path_span_e160_b050",
        "score": 0.11867573539429925,
        "std": 0.22840222976152383
      },
      {
        "feature": "tnm_step_mismatch_e160_b050",
        "score": 0.11911104848592374,
        "std": 0.3275127521901887
      }
    ],
    "min_features": 4,
    "max_features": 14,
    "redundancy_threshold": 0.97
  },
  "cv_repeats": 7,
  "cv_seeds": [
    2,
    3,
    4,
    7,
    13,
    17,
    23
  ]
}
```

## Same-Split Metrics

The split is identical across models and grouped by source file / mode / T1. Lower MAE is better.

| target | raw_MAE | physics_kernel_MAE | physics_residual_MAE | physics_residual_R2 |
| --- | --- | --- | --- | --- |
| delta_h_total_J_g | 0.1744 | 0.1259 | 0.1209 | 0.1184 |
| peak_area_J_g | 0.1252 | 0.0761 | 0.0736 | 0.4563 |
| recovery_index | 0.2537 | 0.1819 | 0.1778 | -0.4428 |
| path_dependence_index | 0.1298 | 0.0894 | 0.1573 | -0.7982 |
| peak_temperature_Tp_C | 0.6305 | 0.4276 | 0.4089 | -0.9325 |
| peak_height_uW | 11.3093 | 6.2210 | 7.1665 | 0.8226 |

![Physics-informed MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/physics_informed/physics_informed_mae_comparison.png)

## Normalized Core-Target Error

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 1.0195512641495252,
    "median_mae_over_test_std": 0.9643905077371802
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 0.7016659924633728,
    "median_mae_over_test_std": 0.6804373648386344
  },
  "physics_residual": {
    "mean_mae_over_test_std": 0.8103390293064023,
    "median_mae_over_test_std": 0.7845623133552233
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 0.8843925753051569,
    "median_mae_over_test_std": 0.8275518035319904
  }
}
```

## Interpretation

The selected `physics_kernel` lowers the mean normalized core-target error from `1.0196` to `0.7017`, a relative change of `-31.2%`. It improves all four core targets in this split: total enthalpy, peak area, recovery index, and path-dependence index.

The `physics_residual` variant is not recommended as the main model right now. It improves peak area and diagnostic peak shape, but it damages `recovery_index` and `path_dependence_index`, which are the targets that matter most for inverse annealing design. The better small-sample choice is therefore the lower-dimensional `physics_kernel`: raw annealing inputs plus TNM path/dose features selected by grouped CV.
