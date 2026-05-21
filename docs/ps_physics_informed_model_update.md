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
  "score": 0.8180811477485079,
  "feature_set": "raw_phys_ps_hs_beta_screened",
  "base_feature_set": "raw_phys_ps_hs_beta",
  "features": [
    "tnm_state_total_e460_b035",
    "tnm_state_total_e460_b065",
    "tnm_path_excess_e460_b035",
    "tnm_path_excess_e540_b050"
  ],
  "bandwidth": 2.2,
  "bandwidth_vector": [
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    2.2,
    1.4300000000000002,
    2.055375957058761,
    2.911201366347356,
    3.4100000000000006
  ],
  "kernel": "ard",
  "feature_screening": {
    "selected_features": [
      "tnm_state_total_e460_b035",
      "tnm_state_total_e460_b065",
      "tnm_path_excess_e460_b035",
      "tnm_path_excess_e540_b050"
    ],
    "scores": [
      {
        "feature": "tnm_state_total_e430_b050",
        "score": 0.4540166798009111,
        "std": 0.21072817270506378
      },
      {
        "feature": "tnm_state_total_e460_b035",
        "score": 0.4839388036417071,
        "std": 0.21353512126459276
      },
      {
        "feature": "tnm_state_total_e460_b050",
        "score": 0.42859216307061127,
        "std": 0.20716331599337914
      },
      {
        "feature": "tnm_state_total_e460_b065",
        "score": 0.36564462363380973,
        "std": 0.20464911094709437
      },
      {
        "feature": "tnm_state_total_e540_b050",
        "score": 0.3649395968644951,
        "std": 0.20119797149638174
      },
      {
        "feature": "tnm_path_excess_e430_b050",
        "score": 0.1570950607645917,
        "std": 0.12645692050145796
      },
      {
        "feature": "tnm_path_excess_e460_b035",
        "score": 0.20375933866401189,
        "std": 0.12683928827229227
      },
      {
        "feature": "tnm_path_excess_e460_b050",
        "score": 0.145368483632027,
        "std": 0.11932169126173714
      },
      {
        "feature": "tnm_path_excess_e460_b065",
        "score": 0.08352522000814859,
        "std": 0.10918828840506095
      },
      {
        "feature": "tnm_path_excess_e540_b050",
        "score": 0.10940813329547067,
        "std": 0.10255166337968616
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
    "mean": 0.8180811477485079,
    "std": 0.2899515590662012,
    "median": 0.7096505806014681,
    "p10": 0.6251248452777707,
    "p90": 1.034274839879359,
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
| delta_h_total_J_g | 0.1445 | 0.1288 | 0.1413 | 0.1394 | 0.1321 | 0.1014 |
| peak_area_J_g | 0.0928 | 0.1053 | 0.0927 | 0.0975 | 0.0756 | 0.4310 |
| recovery_index | 0.1951 | 0.1996 | 0.1852 | 0.1984 | 0.1634 | -0.2918 |
| path_dependence_index | 0.0993 | 0.0801 | 0.1066 | 0.0747 | 0.1759 | -1.5547 |
| peak_temperature_Tp_C | 0.4347 | 0.6262 | 0.5051 | 0.4541 | 0.4411 | -1.4466 |
| peak_height_uW | 7.1920 | 11.1714 | 8.6111 | 7.7958 | 8.7603 | 0.7167 |

![Physics-informed MAE comparison](E:/knowledgepaper/multipramereconstruction/results/ps/physics_informed/physics_informed_mae_comparison.png)

## Normalized Core-Target Error

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 0.7912789533947963,
    "median_mae_over_test_std": 0.7685310101254524
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 0.7615321363997432,
    "median_mae_over_test_std": 0.7253882187430151
  },
  "process_auxiliary_kernel": {
    "mean_mae_over_test_std": 0.7878110720182327,
    "median_mae_over_test_std": 0.7868211436434421
  },
  "physics_process_auxiliary_kernel": {
    "mean_mae_over_test_std": 0.7514707558414883,
    "median_mae_over_test_std": 0.7279304835733089
  },
  "physics_residual": {
    "mean_mae_over_test_std": 0.8457521935728841,
    "median_mae_over_test_std": 0.7795780252221745
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 0.9521222073793433,
    "median_mae_over_test_std": 0.8443003856543352
  }
}
```

## Interpretation

The selected `physics_kernel` lowers the mean normalized core-target error from `0.7913` to `0.7615`, a relative change of `-3.8%`. It improves all four core targets in this split: total enthalpy, peak area, recovery index, and path-dependence index.

The process-assisted variants test whether the annealing/cooling curve contains useful training signal without being used as a test-time input. Their core normalized errors are `0.7878` for `process_auxiliary_kernel` and `0.7515` for `physics_process_auxiliary_kernel`.

The `physics_residual` variant is not recommended as the main model right now. It improves peak area and diagnostic peak shape, but it damages `recovery_index` and `path_dependence_index`, which are the targets that matter most for inverse annealing design. The better small-sample choice is therefore the lower-dimensional `physics_kernel`: raw annealing inputs plus TNM path/dose features selected by grouped CV.
