# PS Conditional Time Inverse

## Purpose

This report evaluates the reduced inverse problem:

```text
given T1, T2 and paper-style targets -> reconstruct t1, t2
```

The target set is strictly measured or strict-paper-style `delta_h_total_J_g`, `peak_area_J_g`, `H*`, and `S*`. Proxy `H*/S*` is kept out of the main model.

## Summary Metrics

| Metric | Value |
| --- | ---: |
| n conditions | 3 |
| t1 log10 MAE | 0.000 |
| t2 log10 MAE | 0.969 |
| median t1 factor error | 1.000 |
| median t2 factor error | 3.000 |
| top1 time near-match | 0.667 |
| top5 time near-match | 0.667 |
| top10 time near-match | 0.667 |
| posterior t1 coverage | 1.000 |
| posterior t2 coverage | 0.333 |
| median posterior width log10 t1 | 0.477 |
| median posterior width log10 t2 | 0.442 |

## Four-Parameter Reference

| Metric | Four-parameter strict ARRT | Fixed T1/T2 conditional |
| --- | ---: | ---: |
| log10 t1 MAE | 0.333 | 0.000 |
| log10 t2 MAE | 0.469 | 0.969 |
| t1 conditional/four-param ratio | 0.000 |  |
| t2 conditional/four-param ratio | 2.069 |  |

## By Condition

| Condition | T1->T2 (C) | actual t1/t2 (s) | pred t1/t2 (s) | log errors | top5 near | posterior widths | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_step|50|10|100|1800 | 50->100 | 10/1800 | 10/600 | 0.000/0.477 | True | 0.477/0.407 | well_constrained |
| two_step|50|10|80|1800 | 50->80 | 10/1800 | 10/10 | 0.000/2.255 | False | 0.703/0.703 | partially_constrained |
| two_step|65|10|100|1200 | 65->100 | 10/1200 | 10/1800 | 0.000/0.176 | True | 0.382/0.442 | well_constrained |

## Interpretation

Fixing `T1` and `T2` removes the largest source of many-to-one ambiguity, so the remaining test is whether the paper-style target vector contains enough information to locate `t1` and `t2`. The result should be judged by both point error and posterior width. If a condition has high coverage but broad posterior intervals, the model is still not uniquely identifying the time pair.

## Outputs

- Summary: `results/ps/conditional_time_inverse/conditional_time_inverse_summary.json`
- By-condition CSV: `results/ps/conditional_time_inverse/conditional_time_inverse_by_condition.csv`
- Top candidates CSV: `results/ps/conditional_time_inverse/conditional_time_inverse_top_candidates.csv`
