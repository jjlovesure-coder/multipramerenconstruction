# TNM Conditional Time Inverse Comparison

## Summary

This evaluates the reduced inverse problem with the calibrated TNM forward model:

```text
given T1, T2 and final heating targets -> reconstruct t1, t2
```

Unlike the strict paper-style model, TNM uses the full two-step PS dataset and target set `delta_h_total_J_g, peak_area_J_g, recovery_index, path_dependence_index`. The dense 80-90 data include historical sub-10 s times, so this diagnostic uses the observed time grid plus the practical 10-1800 s grid.

## TNM Overall Metrics

| Metric | Value |
| --- | ---: |
| n two-step samples | 60 |
| t1 log10 MAE | 0.368 |
| t2 log10 MAE | 0.570 |
| median t1 factor error | 1.433 |
| median t2 factor error | 1.800 |
| top1 time near-match | 0.517 |
| top5 time near-match | 0.767 |
| top10 time near-match | 0.833 |
| posterior t1 coverage | 0.850 |
| posterior t2 coverage | 0.883 |
| posterior width log10 t1 median | 1.000 |
| posterior width log10 t2 median | 1.868 |

## Paper-Style Conditional Reference

The strict paper-style conditional inverse used only `3` strict double-step H*/S* conditions. Its headline values were:

```json
{
  "t1_log10_MAE": 0.0,
  "t2_log10_MAE": 0.9694950062928833,
  "top5_time_near_match": 0.6666666666666666,
  "posterior_width_log10_t1_median": 0.4771212547196624,
  "posterior_width_log10_t2_median": 0.44190300290852624
}
```

These two evaluations are not perfectly apples-to-apples: TNM has many more two-step final-heating targets, while strict paper-style H*/S* has cleaner physics labels but only three double-step strict points.

## By Temperature Pair

| Pair | n | t1 log MAE | t2 log MAE | top5 near | posterior width t1/t2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50->100 | 4 | 0.326 | 0.433 | 0.750 | 1.591/1.279 |
| 50->65 | 1 | 0.000 | 0.000 | 1.000 | 1.001/1.756 |
| 50->70 | 3 | 0.275 | 0.492 | 0.667 | 0.970/2.255 |
| 50->75 | 1 | 0.477 | 0.477 | 1.000 | 0.523/1.049 |
| 50->80 | 2 | 0.151 | 0.128 | 1.000 | 1.865/1.719 |
| 50->85 | 1 | 1.000 | 0.222 | 0.000 | 0.523/0.501 |
| 55->90 | 1 | 3.079 | 0.477 | 1.000 | 3.769/2.071 |
| 65->100 | 2 | 1.151 | 0.040 | 0.500 | 1.835/1.355 |
| 65->65 | 1 | 1.999 | 2.556 | 1.000 | 1.628/2.979 |
| 65->85 | 1 | 2.000 | 0.176 | 0.000 | 4.237/2.238 |
| 75->100 | 2 | 1.040 | 0.866 | 0.500 | 1.802/1.628 |
| 80->90 | 20 | 0.198 | 0.599 | 0.800 | 1.000/1.728 |
| 85->100 | 1 | 0.001 | 0.079 | 1.000 | 1.302/1.255 |
| 90->80 | 20 | 0.138 | 0.648 | 0.800 | 0.608/2.139 |

## 80-90 Data Reflection

The densest pairs are `80->90` and `90->80`, but the current matrix is narrow: it mostly uses `t1` near 50 or 500 s and many `t2` values below 10 s. That is useful for fast Kovacs-style probing, but weak for the practical finite-time window `10-1800 s`.

Recommended additions:

| Pair | T1 | T2 | t1 | t2 | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| 80->90 | 80 | 90 | 10 | 10 | Add the missing low-low time anchor inside the 10-1800 s experimental window. |
| 80->90 | 80 | 90 | 300 | 300 | Add a balanced midpoint because current dense data mainly uses t1 near 50/500 s and many sub-10 s t2 values. |
| 80->90 | 80 | 90 | 900 | 900 | Add a long-long finite-time anchor for the practical 30 min window. |
| 80->90 | 80 | 90 | 100 | 900 | Separate step-2 dominated aging from total-time effects. |
| 80->90 | 80 | 90 | 900 | 100 | Separate step-1 dominated memory from step-2 aging. |
| 90->80 | 90 | 80 | 10 | 10 | Add the missing low-low time anchor inside the 10-1800 s experimental window. |
| 90->80 | 90 | 80 | 300 | 300 | Add a balanced midpoint because current dense data mainly uses t1 near 50/500 s and many sub-10 s t2 values. |
| 90->80 | 90 | 80 | 900 | 900 | Add a long-long finite-time anchor for the practical 30 min window. |
| 90->80 | 90 | 80 | 100 | 900 | Separate step-2 dominated aging from total-time effects. |
| 90->80 | 90 | 80 | 900 | 100 | Separate step-1 dominated memory from step-2 aging. |

## Outputs

- Summary: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_summary.json`
- By sample: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_by_sample.csv`
- By pair: `results/ps/conditional_tnm_time_inverse/conditional_tnm_time_inverse_by_pair.csv`
