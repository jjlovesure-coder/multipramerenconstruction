# PS Integration Window Repeatability Check

## Scope

The current DSC feature extraction uses a shared main integration window of
`70-105 C` for `delta_h_total_J_g` and `peak_area_J_g`.

This was chosen by comparing repeated scans under identical inferred conditions:
`mode, T1_C, t1_s, T2_C, t2_s`. The repeated set contains 24 condition groups
and 48 curves. Candidate windows were required to include each curve's focused
relaxation peak temperature, defined as the strongest absolute extremum inside
`70-105 C`. This avoids letting the broader glass-transition / high-temperature
baseline swing at higher temperature define the peak.

## Result

The best shared window by absolute repeatability is:

```text
70-105 C
```

Key comparisons:

| window C | repeated groups | mean abs SD J/g | median abs SD J/g | mean pair range J/g | mean relative SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40-105 | 24 | 1.488 | 1.289 | 2.104 | 0.263 |
| 60-105 | 24 | 1.038 | 0.890 | 1.469 | 0.269 |
| 65-105 | 24 | 0.925 | 0.790 | 1.308 | 0.272 |
| 70-105 | 24 | 0.811 | 0.690 | 1.146 | 0.276 |

`70-105 C` reduces the mean repeated-condition absolute standard deviation by
about 46% relative to `40-105 C`.

## Interpretation

The low-temperature part of the heating scan contributes substantial
run-to-run drift. Raising the lower bound to `70 C` improves absolute agreement
between repeated experiments. The upper bound remains `105 C` because the
focused relaxation peak temperatures in the repeated set are at or below
`104.999 C`; lower upper bounds would exclude at least one peak.

The tradeoff is that the relative SD is slightly higher because the integrated
signal magnitude is smaller. For this dataset the absolute scatter is the more
direct consistency criterion, since the goal is to reduce drift-driven target
noise before surrogate training and inverse reconstruction.

## Implementation

Updated code path:

- `src/ps/dsc_processing.py`: `MAIN_INTEGRATION_LOW_C = 70.0`, `MAIN_INTEGRATION_HIGH_C = 105.0`
- `data/ps/ps_preexperiment_features.csv`: regenerated with the new main window
- `docs/ps_tnm_enthalpy_time_fit_report.md`: regenerated with `70-105 C`
- `docs/ps_tnm_curve_fit_forward_report.md`: regenerated with `70-105 C`

