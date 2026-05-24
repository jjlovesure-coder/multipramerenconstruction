# PS Physics-Informed Model Update

## Purpose

This update tests whether embedding TNM/ARRT-style relaxation priors improves the finite-time PS predictor beyond the raw temperature/time RBF model.

## Model Variants

- `raw_kernel`: the existing RBF surrogate using only mode, temperature, and log-time features.
- `physics_kernel`: RBF using raw features plus Arrhenius/TNM dose, sequential state, and path-memory features.
- `process_auxiliary_kernel`: two-stage model. First predicts measured pre-scan annealing/cooling summaries from programmed parameters, then uses the predicted summaries to predict the final heating-scan targets.
- `physics_process_auxiliary_kernel`: same two-stage auxiliary setup, but with selected physics features in the first and second stages.
- `physics_residual`: a ridge physics baseline trained on physics features, plus an RBF model trained only on its residual.

The process-assisted variants do not use measured test-cycle annealing/cooling curves as inputs. Those curves are auxiliary labels during training only, so the validation still follows the rule that only the final heating curve provides the supervised output.

The `physics_kernel` feature subset and bandwidth are selected by repeated grouped cross-validation inside the training split:

```json
{
  "score": 1.4971863951713271,
  "feature_set": "raw_arrt_identifiability_screened",
  "base_feature_set": "raw_arrt_identifiability",
  "features": [
    "phys_inv_T_diff",
    "phys_log_t_ratio",
    "arrt_step1_fraction_of_total_hs70_b035",
    "arrt_step2_increment_hs70_b035",
    "arrt_step1_fraction_of_total_hs90_b065",
    "phys_step_time_fraction_2",
    "arrt_state_total_hs70_b035",
    "phys_delta_T_times_log_ratio",
    "arrt_state_total_hs90_b065",
    "phys_log_t2_s",
    "arrt_step2_increment_hs90_b065",
    "arrt_state_step1_hs70_b035",
    "arrt_state_step1_hs90_b065",
    "phys_log_t1_s"
  ],
  "bandwidth": 1.4,
  "bandwidth_vector": [
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    1.4,
    0.9099999999999999,
    1.0812954921997293,
    1.1767386133036315,
    1.398847335259705,
    1.5013151205371638,
    1.5519855301016363,
    1.5772738330243425,
    1.5963761260746652,
    1.639476802848581,
    1.6458063676916845,
    1.760184585795391,
    1.8145421660562742,
    2.0485752252121445,
    2.17
  ],
  "kernel": "ard",
  "feature_screening": {
    "selected_features": [
      "phys_inv_T_diff",
      "phys_log_t_ratio",
      "arrt_step1_fraction_of_total_hs70_b035",
      "arrt_step2_increment_hs70_b035",
      "arrt_step1_fraction_of_total_hs90_b065",
      "phys_step_time_fraction_2",
      "arrt_state_total_hs70_b035",
      "phys_delta_T_times_log_ratio",
      "arrt_state_total_hs90_b065",
      "phys_log_t2_s",
      "arrt_step2_increment_hs90_b065",
      "arrt_state_step1_hs70_b035",
      "arrt_state_step1_hs90_b065",
      "phys_log_t1_s"
    ],
    "scores": [
      {
        "feature": "phys_log_t1_s",
        "score": 0.11182148267653835,
        "std": 1.0747781641664236
      },
      {
        "feature": "phys_log_t2_s",
        "score": 0.22757108883521357,
        "std": 1.1986341263494988
      },
      {
        "feature": "phys_log_t_ratio",
        "score": 0.35222333083736473,
        "std": 1.2813011551919593
      },
      {
        "feature": "phys_step_time_fraction_1",
        "score": 0.24828809936566146,
        "std": 0.34543808847334695
      },
      {
        "feature": "phys_step_time_fraction_2",
        "score": 0.24828809936566154,
        "std": 0.34543808847334695
      },
      {
        "feature": "phys_time_asymmetry",
        "score": 0.09840364728443939,
        "std": 0.4488226600146978
      },
      {
        "feature": "phys_delta_T_times_log_ratio",
        "score": 0.23848600776374593,
        "std": 24.040117443390535
      },
      {
        "feature": "phys_inv_T_diff",
        "score": 0.39004787607566554,
        "std": 0.00010917461950663387
      },
      {
        "feature": "arrt_state_step1_hs70_b035",
        "score": 0.19031176136195796,
        "std": 0.3726333823671593
      },
      {
        "feature": "arrt_state_total_hs70_b035",
        "score": 0.24270407292330498,
        "std": 0.4193178467581346
      },
      {
        "feature": "arrt_step1_fraction_of_total_hs70_b035",
        "score": 0.33114809642506726,
        "std": 0.3535170819179779
      },
      {
        "feature": "arrt_step2_increment_hs70_b035",
        "score": 0.2821032482882991,
        "std": 0.2812158506078643
      },
      {
        "feature": "arrt_path_excess_hs70_b035",
        "score": 0.0943495873415795,
        "std": 0.1883952291090239
      },
      {
        "feature": "arrt_state_step1_hs95_b050",
        "score": 0.16859669253355855,
        "std": 0.3896730155380601
      },
      {
        "feature": "arrt_state_total_hs95_b050",
        "score": 0.23757023957499715,
        "std": 0.4398674877998573
      },
      {
        "feature": "arrt_step1_fraction_of_total_hs95_b050",
        "score": 0.307660736351328,
        "std": 0.3750856915431946
      },
      {
        "feature": "arrt_step2_increment_hs95_b050",
        "score": 0.25314894608835975,
        "std": 0.30419042719273043
      },
      {
        "feature": "arrt_path_excess_hs95_b050",
        "score": 0.07584481475436773,
        "std": 0.25907907041752537
      },
      {
        "feature": "arrt_state_step1_hs90_b065",
        "score": 0.13863384550076552,
        "std": 0.3713872160165311
      },
      {
        "feature": "arrt_state_total_hs90_b065",
        "score": 0.22896874915115215,
        "std": 0.4316943913544347
      },
      {
        "feature": "arrt_step1_fraction_of_total_hs90_b065",
        "score": 0.2594768654808836,
        "std": 0.4091381409347039
      },
      {
        "feature": "arrt_step2_increment_hs90_b065",
        "score": 0.20231470859202869,
        "std": 0.3239195488086562
      },
      {
        "feature": "arrt_path_excess_hs90_b065",
        "score": 0.08498869331583377,
        "std": 0.375627639787011
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
  ],
  "selected_cv_summary": {
    "n_scores": 35,
    "mean": 1.4971863951713271,
    "std": 0.8520103127656967,
    "median": 1.1886429643727752,
    "p10": 0.7339799169124006,
    "p90": 2.606120148216413,
    "seeds": [
      2,
      3,
      4,
      7,
      13,
      17,
      23
    ],
    "n_folds": 5
  }
}
```

## Same-Split Metrics

The split is identical across models and grouped by source file / mode / T1. Lower MAE is better.

| target | raw_MAE | physics_kernel_MAE | process_aux_MAE | physics_process_aux_MAE | physics_residual_MAE | physics_residual_R2 |
| --- | --- | --- | --- | --- | --- | --- |
| delta_h_total_J_g | 1.1799 | 1.3736 | 1.2502 | 1.6440 | 1.6348 | -5.6978 |
| peak_area_J_g | 1.0262 | 1.1925 | 1.0675 | 1.4573 | 1.5120 | -4.0459 |
| recovery_index | 0.3163 | 0.3490 | 0.3064 | 0.3567 | 0.3383 | -0.5007 |
| path_dependence_index | 0.8185 | 0.8949 | 1.1810 | 1.3522 | 0.8078 | -1.5685 |
| peak_temperature_Tp_C | 7.2522 | 7.4134 | 6.3757 | 6.9837 | 9.1566 | -0.7148 |
| peak_height_uW | 45.1114 | 53.6915 | 44.2010 | 58.6079 | 57.5832 | -2.6033 |

![Physics-informed MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/physics_informed/physics_informed_mae_comparison.png)

## Normalized Core-Target Error

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 1.1552472982544137,
    "median_mae_over_test_std": 1.1024038913916332
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 1.3138030581336362,
    "median_mae_over_test_std": 1.2524224586413109
  },
  "process_auxiliary_kernel": {
    "mean_mae_over_test_std": 1.2805359960571256,
    "median_mae_over_test_std": 1.2740356746726977
  },
  "physics_process_auxiliary_kernel": {
    "mean_mae_over_test_std": 1.6098901752547718,
    "median_mae_over_test_std": 1.5990323208163373
  },
  "physics_residual": {
    "mean_mae_over_test_std": 1.4622066389716228,
    "median_mae_over_test_std": 1.4264049443076685
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 1.4151436041027952,
    "median_mae_over_test_std": 1.3251766997295584
  }
}
```

## Interpretation

The selected `physics_kernel` does not improve the mean normalized core-target error in this split: `raw_kernel` is `1.1552`, while `physics_kernel` is `1.3138` (13.7% higher). It is not recommended as the default model for this window; worse core targets: delta_h_total_J_g, peak_area_J_g, recovery_index, path_dependence_index.

The process-assisted variants test whether the annealing/cooling curve contains useful training signal without being used as a test-time input. Their core normalized errors are `1.2805` for `process_auxiliary_kernel` and `1.6099` for `physics_process_auxiliary_kernel`.

The `physics_residual` variant has a core normalized error of `1.4622` in this split. The default model should be chosen from the measured same-split results rather than assumed from the physics-feature label.
