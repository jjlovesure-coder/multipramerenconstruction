# PS TNM Curve-Level Forward Fit

## Scope

This is the first curve-level forward diagnostic after narrowing the DSC target
window to `40-100 C`.  It tests whether compact TNM/fictive-temperature
coordinates can predict the full final-heating recovery curve shape before
inverse reconstruction is trusted.

This is **not yet** a full material-parameter TNM solver. It is a minimal
curve-level check that predicts peak area, peak position, peak height, and width,
then reconstructs the full curve and reports residuals.

## Summary Metrics

```json
{
  "n_curves": 113,
  "curve_rmse_uW_mean": 45.65465669073802,
  "curve_mae_uW_mean": 39.707465792802374,
  "area_error_J_g_mae": 0.4111382505616823,
  "tp_error_C_mae": 0.4648228052402133,
  "peak_height_error_uW_mae": 9.390966760588162,
  "curve_rmse_uW_median": 46.20878589385725,
  "curve_mae_uW_median": 40.58413630934538
}
```

## Figures

![Actual vs predicted](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_curve_fit/figures/tnm_curve_fit_actual_vs_predicted.png)

![Error distributions](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_curve_fit/figures/tnm_curve_fit_error_distributions.png)

![Representative overlays](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_curve_fit/figures/tnm_curve_fit_representative_overlays.png)

## Interpretation

If curve RMSE/MAE remain large relative to the observed peak heights, the inverse
posterior should be interpreted as weak even when top candidates look plausible.
The next step after this diagnostic is a stricter TNM solver that fits `Tf(t)`
and `Delta h(t)` directly.

## Outputs

- Per-sample fit errors: `results\ps\tnm_curve_fit\tnm_curve_fit_by_sample.csv`
- Summary: `results\ps\tnm_curve_fit\tnm_curve_fit_summary.json`
