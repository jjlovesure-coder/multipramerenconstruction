# PS TNM/KWW Enthalpy-Time Fit

## Scope

This report fits the Figure-2-style relation:

```text
annealing time -> delta_h_total_J_g, integrated over 40-105 C
```

It does **not** fit the final heating DSC heat-flow trace. Single-step groups are
fit as `delta_h(t1)` at fixed `T`. Two-step groups are fit as `delta_h(t2)` at
fixed `T1, t1, T2`.

## Summary

```json
{
  "n_groups": 9,
  "n_points": 72,
  "group_rmse_J_g_mean": 0.44672741907767943,
  "group_rmse_J_g_median": 0.10477504506187096,
  "point_mae_J_g": 0.3225279345880879,
  "single_step_groups": 5,
  "two_step_groups": 4
}
```

## Figures

![Single-step enthalpy-time fits](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_enthalpy_time_fit/figures/single_step_enthalpy_time_fits_40_105.png)

![Two-step enthalpy-time fits](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_enthalpy_time_fit/figures/two_step_enthalpy_time_fits_40_105.png)

![80 to 90 C up-jump focused check](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_enthalpy_time_fit/figures/upjump_80_90_enthalpy_time_40_105.png)

![Fit errors](E:/knowledgepaper/multipramereconstruction/results/ps/tnm_enthalpy_time_fit/figures/enthalpy_time_fit_errors_40_105.png)

## Outputs

- Group fits: `results\ps\tnm_enthalpy_time_fit\tnm_enthalpy_time_fit_by_group.csv`
- Point residuals: `results\ps\tnm_enthalpy_time_fit\tnm_enthalpy_time_fit_by_point.csv`
- Summary: `results\ps\tnm_enthalpy_time_fit\tnm_enthalpy_time_fit_summary.json`
