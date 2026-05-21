# PS Calibrated TNM Forward Model

## Purpose

This branch changes the strategy from direct four-parameter inverse regression to a calibrated TNM forward model. The assumption is that if PS enthalpy-recovery curves can be fit by a small set of TNM kinetic parameters, unknown annealing schedules can be predicted through the calibrated physical model with fewer experiments.

The model is not used to claim unique recovery of `T1, t1, T2, t2` from a single final heating curve. It is a forward predictor and curve generator.

## Effective Predictive TNM Parameters

```json
{
  "activation_energy_kj_mol": 220.0,
  "beta": 0.75,
  "log10_tau_ref_s": 2.0,
  "nonlinearity_x": 0.9,
  "initial_fictive_temperature_k": 393.15,
  "tau_ref_s": 100.0,
  "preexponential_A_s": 1.6008513042763764e-29,
  "activation_entropy_j_mol_k": 304.4961898086539
}
```

These parameters are effective predictive coordinates, not PS intrinsic material constants.

## ARRT-Constrained TNM Parameters

```json
{
  "activation_energy_kj_mol": 430.0,
  "beta": 0.75,
  "log10_tau_ref_s": 2.0,
  "nonlinearity_x": 0.9,
  "initial_fictive_temperature_k": 393.15,
  "tau_ref_s": 100.0,
  "preexponential_A_s": 1.8048513878454153e-33,
  "activation_entropy_j_mol_k": 380.0779614209023
}
```

Parameter selection strategy:

```text
repeated_grouped_cv_predictive_unconstrained
```

This fit-match profile is used to test whether the calibrated TNM basis can match the current PS enthalpy-recovery data. Repeated grouped CV is still reported as a robustness diagnostic, not hidden.

Fit-match weighted normalized MAE:

```json
{
  "calibration_split_weighted_normalized_mae": 0.6316154955487943,
  "target_weights": [
    1.1,
    1.0,
    1.15,
    0.75
  ]
}
```

Repeated grouped CV diagnostic:

```json
{
  "predictive_unconstrained": {
    "score_mean": 0.803373994395016,
    "score_std": 0.4877595309470907,
    "score_median": 0.664376725638945,
    "n_scores": 35,
    "n_folds": 5,
    "seeds": [
      2,
      3,
      4,
      7,
      13,
      17,
      23
    ]
  },
  "arrt_constrained": {
    "score_mean": 0.8760410154663991,
    "score_std": 0.2642864935074167,
    "score_median": 0.8193765310566061,
    "n_scores": 35,
    "n_folds": 5,
    "seeds": [
      2,
      3,
      4,
      7,
      13,
      17,
      23
    ]
  }
}
```

Dose dynamic range diagnostic:

```json
{
  "predictive_unconstrained": {
    "n": 108.0,
    "step1_saturation_ratio": 0.0,
    "step2_saturation_ratio": 0.05555555555555555,
    "total_saturation_ratio": 0.05555555555555555,
    "step1_range": 0.8976638575501539,
    "total_range": 0.9998388835179871
  },
  "arrt_constrained": {
    "n": 108.0,
    "step1_saturation_ratio": 0.0,
    "step2_saturation_ratio": 0.05555555555555555,
    "total_saturation_ratio": 0.05555555555555555,
    "step1_range": 0.8976653599838524,
    "total_range": 0.9998397779970836
  }
}
```

Strict ARRT precision note:

```text
Strict H*/S* currently has only six calculable conditions; it is useful as a constraint/diagnostic but not enough to claim original-paper-level inverse precision.
```

The fitted observation head maps TNM state descriptors to:

```text
delta_h_total_J_g, peak_area_J_g, recovery_index, path_dependence_index
```

## Same-Split Test Metrics

| target | MAE | RMSE | R2 | target std |
|---|---:|---:|---:|---:|
| `delta_h_total_J_g` | 0.2483 | 0.2818 | -0.783 | 0.2110 |
| `peak_area_J_g` | 0.1406 | 0.1725 | -0.543 | 0.1389 |
| `recovery_index` | 0.1447 | 0.1530 | -0.207 | 0.1393 |
| `path_dependence_index` | 0.1417 | 0.1644 | -1.090 | 0.1137 |

Normalized core error:

```json
{
  "mean_mae_over_test_std": 1.118379535967137,
  "median_mae_over_test_std": 1.107676676078941
}
```

## Outputs

- Model: `results\ps\models\ps_calibrated_tnm_model.json`
- Test predictions: `results\ps\tnm_calibrated\tnm_calibrated_test_predictions.csv`
- Single-step prediction curves: `results\ps\tnm_calibrated\tnm_calibrated_single_step_curves.csv`
- TNM inverse reconstruction: `results/ps/tnm_calibrated/tnm_inverse_reconstruction_by_sample.csv`

## Interpretation

This route should be judged by forward-curve agreement and parameter parsimony. If the calibrated TNM curve tracks `delta_h_total_J_g` and `recovery_index` across temperature/time sweeps, it can replace direct sparse inverse regression as the main engine for recommending unknown annealing schedules.
