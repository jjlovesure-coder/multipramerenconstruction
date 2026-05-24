# PS 70-105 C Enthalpy, TNM Fit, And ARRT H*/S* Audit

## Scope

This audit uses the current main DSC feature window:

```text
enthalpy / curve integration window = 70-105 C
peak-search window = 70-105 C for ordinary DSC features
ARRT peak-search window = 95-125 C for multi-heating-rate Kissinger labels
```

The 70-105 C window is currently the lowest-error enthalpy-time window among the checked windows. The regenerated TNM/KWW enthalpy-time fit summary is:

```json
{
  "n_groups": 9,
  "n_points": 72,
  "group_rmse_J_g_mean": 0.23427546193205936,
  "group_rmse_J_g_median": 0.06321531796016497,
  "point_mae_J_g": 0.1713312173847774
}
```

## Enthalpy-Time TNM/KWW Diagnosis

The largest residual groups are:

| group | RMSE (J/g) | interpretation |
|---|---:|---|
| single_step, 90 C | 0.958 | Most suspicious. The 90 C time series is non-monotonic in the 70-105 C window: 50 s = 3.06 J/g, 100 s = 1.33 J/g, 300 s = 0.59 J/g, repeat 500 s = 3.21 J/g / 2.76 J/g. A monotonic KWW/TNM time law cannot fit this without large residuals. |
| single_step, 70 C | 0.839 | Also poor, but less directly tied to the H*/S* concern. It may reflect weak signal, baseline/reference subtraction sensitivity, or real non-KWW behavior at low annealing temperature. |
| two_step 80 C 49.98 s -> 90 C | 0.070 | Acceptable scalar enthalpy-time fit, but curve-level fitting below still shows early t2 points have large shape residuals. |

The 90 C, 500 s condition should remain a retest candidate. The repeated 10 C/min single-step 90 C 500 s scan does not agree with the expected monotonic trend from 90 C 50/100/300 s.

## Curve-Level TNM Forward Fit

The curve-level diagnostic was regenerated over 114 final-heating curves:

```json
{
  "curve_rmse_uW_mean": 52.66964323586883,
  "curve_mae_uW_mean": 48.083542100907486,
  "area_error_J_g_mae": 1.765981677235704,
  "tp_error_C_mae": 5.0719603086794125,
  "peak_height_error_uW_mae": 50.711593903469186,
  "curve_rmse_uW_median": 43.539156738469345,
  "curve_mae_uW_median": 37.05068525627824
}
```

This is useful as a forward diagnostic, but it is not yet good enough to treat inverse TNM curve fitting as physically strong. The highest curve residuals cluster around:

| condition | curve RMSE (uW) | area error (J/g) |
|---|---:|---:|
| two_step 50 C 1800 s -> 100 C 10 s | 153.63 | 6.54 |
| single_step 80 C 50 s | 122.25 | 5.37 |
| single_step 90 C 499.998 s repeat | 120.41 | 5.27 |
| single_step 90 C 50 s | 120.24 | 5.33 |
| two_step 80 C 49.98 s -> 90 C 0.1002 s | 114.67 | 5.08 |
| single_step 80 C 500 s | 112.86 | 4.99 |
| single_step 90 C 500 s | 109.76 | 4.80 |

Therefore the current compact curve model has a model-form limitation: it compresses each recovery peak into area, Tp, height, and width with one reconstructed peak shape. That is too weak for conditions where the recovery curve changes sign, has multiple local features, or has strong baseline/reference subtraction sensitivity.

## Multi-Heating-Rate ARRT H*/S* Diagnosis

The formula implementation is internally correct for the strict ARRT/Kissinger input:

```text
ln(beta / Tp^2) = intercept + slope * (1 / Tp)
E = -R * slope
H* = E - R * mean(Tp)
S* = R * [ln(beta / Tp^2) + H*/(R Tp) - ln(kB R / (h H*))]
```

However, correctness of H*/S* depends on whether the same physical peak is tracked across heating rates and whether the Kissinger line is linear. Current results:

| condition | H* (kJ/mol) | S* (J/mol/K) | R2 | verdict |
|---|---:|---:|---:|---|
| single_step 90 C 500 s | 204.58 | 320.52 | 0.956 | Numerically valid but flagged suspect; do not train by default before retest. |
| two_step 50 C 10 s -> 80 C 1800 s | 188.66 | 278.24 | 0.957 | Most defensible two-step H*/S* label. |
| single_step 80 C 1000 s | 171.23 | 231.40 | 0.957 | Defensible. |
| single_step 90 C 100 s | 162.84 | 207.53 | 0.951 | Defensible, but close to threshold. |
| single_step 80 C 500 s | 154.13 | 184.47 | 0.944 | Borderline. |
| single_step 70 C 300 s | 69.64 | -49.65 | 0.905 | Low-confidence; likely peak-tracking or weak-signal issue. |
| single_step 80 C 50 s | 69.74 | -49.38 | 0.905 | Low-confidence; likely peak-tracking issue. |
| single_step 90 C 50 s | 69.41 | -50.24 | 0.896 | Low-confidence; likely peak-tracking issue. |
| single_step 95 C 100 s | 186.26 | 270.44 | 0.898 | Low-confidence despite plausible magnitude. |
| two_step 50 C 10 s -> 100 C 1800 s | 153.92 | 181.74 | 0.817 | Low-confidence. |
| two_step 65 C 10 s -> 100 C 1200 s | 151.56 | 175.20 | 0.795 | Low-confidence. |

## Conclusion

1. The 70-105 C window is better for scalar enthalpy recovery and should remain the primary window for the current dataset.
2. The strongest experimental-problem candidate is single-step 90 C around 500 s, especially the repeat 10 C/min scan. It breaks the monotonic time trend and is already flagged as `suspect_retest_required`.
3. The compact curve-level TNM fit also has a model-form problem. It is good enough for screening residuals, but not yet a physically strong full-curve TNM solver.
4. H*/S* is calculated correctly from the strict ARRT equations, but only the high-R2, non-suspect rows should be treated as training-quality labels. Low-R2 rows and the flagged 90 C 500 s row should be excluded or down-weighted.
