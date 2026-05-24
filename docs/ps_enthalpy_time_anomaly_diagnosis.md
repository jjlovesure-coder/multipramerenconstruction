# PS enthalpy-time anomaly diagnosis after 105 C integration update

## Current conclusion

The DSC feature extraction has been regenerated with the main integration window:

```text
40-105 C
```

The old figure filenames were reused before, so the desktop viewer could easily show a cached image. The regenerated diagnostic figures now include the window tag in the filename, for example:

```text
single_step_enthalpy_time_fits_40_105.png
two_step_enthalpy_time_fits_40_105.png
upjump_80_90_enthalpy_time_40_105.png
```

The numerical check shows two different situations:

1. The low-to-high `80 -> 90 C` last-three-point artifact is fixed by the 105 C window. The last three points are now monotonic increasing.
2. The single-step `70 C` curve and the high-to-low `90 -> 80 C` curves still contain real anomalies or weakly constrained behavior. These are not solved by only changing the integration upper bound.

## 1. Single-step 50 C and 70 C

### 50 C

The 50 C single-step curve is not strongly structured, but it is no longer a large data failure after averaging duplicate scans. It is mostly a low-signal curve:

```text
t = 0.1 s      mean delta_h_total ~= 9.017 J/g
t = 1000 s     mean delta_h_total ~= 8.913 J/g
fit RMSE       ~= 0.015 J/g
```

The total change over the whole time window is only about `0.10 J/g`, so small baseline differences and replicate scatter visibly affect the shape. This point should be treated as a weak-response low-temperature region rather than a reliable kinetics anchor.

### 70 C

The 70 C single-step curve is still problematic because one point comes from a different experiment family:

```text
sample_id: hs_01_001
source_file: ps-hs-01.xlsx
T = 70 C
t = 300 s
delta_h_total_J_g = 8.3085
```

This point is mixed into the `PS-onestep-02.xlsx` time series and creates the visual dip near 300 s. The rest of the 70 C sequence is more coherent:

```text
100 s      ~= 8.776 J/g
500 s      ~= 8.603 J/g
1000 s     ~= 8.507 J/g
```

So the main source of the 70 C disorder is not the KWW fitting curve. It is the combination of batch mixing and a single 300 s ARRT-derived point being used as if it were part of the same single-step time series.

## 2. Low-to-high two-step: 80 -> 90 C

After changing the main window to `40-105 C`, the previous last-three-point problem is removed.

For `T1=80 C, t1=49.98 s, T2=90 C`:

| t2 (s) | delta_h_total_J_g |
| ---: | ---: |
| 100.02 | 9.463 |
| 499.98 | 11.269 |
| 1000.02 | 13.434 |

For `T1=80 C, t1=499.98 s, T2=90 C`:

| t2 (s) | delta_h_total_J_g |
| ---: | ---: |
| 100.02 | 9.393 |
| 499.98 | 11.158 |
| 1000.02 | 13.314 |

This means the earlier nonmonotonic result was mainly caused by cutting the relaxation contribution too early at 100 C.

## 3. High-to-low two-step: 90 -> 80 C

The `90 -> 80 C` curves remain suspicious even after the 105 C update.

For `T1=90 C, t1=49.98 s, T2=80 C`, the last three points are:

| t2 (s) | delta_h_total_J_g |
| ---: | ---: |
| 100.02 | 9.178 |
| 499.98 | 9.173 |
| 1000.02 | 9.398 |

For `T1=90 C, t1=499.98 s, T2=80 C`, the last three points are:

| t2 (s) | delta_h_total_J_g |
| ---: | ---: |
| 100.02 | 11.280 |
| 499.98 | 11.672 |
| 1000.02 | 12.065 |

The first `t1=49.98 s` group is the more concerning one: it is nearly flat from 100 to 500 s, then jumps at 1000 s. That behavior can be physical if the 80 C step is below the effective relaxation timescale after a 90 C pre-anneal, but it can also indicate baseline/ref mismatch or an outlying 1000 s scan. It should not be used as strong evidence for a clean monotonic time law without checking the raw DSC curves.

## 4. Practical handling for the next model pass

Recommended data policy:

```text
Use 40-105 C as the current main window.
Flag single_step 70 C / 300 s from ps-hs-01.xlsx as mixed-batch.
Treat single_step 50 C as low-response / low-information.
Use 80->90 C as the cleanest two-step high-information path after the 105 C update.
Flag 90->80 C, especially t1=49.98 s, as needing raw-curve inspection before training.
```

Recommended next diagnostic:

```text
Plot raw baseline-corrected DSC traces for:
1. single_step 70 C: 100, 300, 500, 1000 s
2. two_step 90->80 C, t1=49.98 s: t2 = 100, 500, 1000 s
```

Only after this raw-curve check should these points enter TNM calibration or inverse reconstruction.
