# Strict ARRT Paper-Style Benchmark

## Scope

This benchmark separates three quantities that were previously easy to mix:

- strict experimental `H* / S*`: calculated from multi-heating-rate Kissinger peak shifts.
- old proxy `H* / S*`: inferred from annealing time and one heating-curve `Tp`.
- current PS multi-target model: richer DSC descriptors used for forward prediction.

The strict measured set now has `6` calculable conditions. One condition is low confidence because its Kissinger R2 is below 0.95.

Current multi-target reference from the existing PS comparison report:

```json
{
  "source": "results\\ps\\paper_style_comparison\\summary.json",
  "status": "loaded",
  "targets": [
    "delta_h_total_J_g",
    "peak_area_J_g",
    "peak_temperature_Tp_C",
    "peak_height_uW",
    "recovery_index",
    "path_dependence_index"
  ],
  "normalized": {
    "mean_mae_over_test_std": 0.7887167759073604,
    "median_mae_over_test_std": 0.7685310101254524
  },
  "eig_stats": {
    "mean": 1.178873246611763,
    "std": 0.33471211439257403,
    "p50": 1.1949820410168872,
    "p90": 1.6086297560987273,
    "p99": 1.7588044381160517,
    "max": 1.8230265083498902
  },
  "note": "This reference uses the full PS dataset and is not directly the same six-condition strict ARRT subset."
}
```

## LOOCV Prediction

| Target set | Target | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| strict measured H*/S* | delta_h_total_J_g | 0.8112 | 1.193 | -1.692 |
| strict measured H*/S* | peak_area_J_g | 0.4874 | 0.7207 | -1.629 |
| strict measured H*/S* | activation_enthalpy_H_star_kj_mol | 75.03 | 83.39 | -1.305 |
| strict measured H*/S* | activation_entropy_S_star_j_mol_K | 200.8 | 223.2 | -1.291 |
| old proxy H*/S* | delta_h_total_J_g | 0.8112 | 1.193 | -1.692 |
| old proxy H*/S* | peak_area_J_g | 0.4874 | 0.7207 | -1.629 |
| old proxy H*/S* | proxy_activation_enthalpy_H_star_kj_mol | 102.7 | 110.7 | 0.115 |
| old proxy H*/S* | proxy_activation_entropy_S_star_j_mol_K | 271.2 | 292.4 | 0.116 |

Mean normalized MAE:

```json
{
  "strict_paper_style": 1.2349963548126157,
  "proxy_paper_style": 0.9893738045751788
}
```

## EIG Comparison

```json
{
  "strict_paper_style": {
    "mean": 0.47137566870585024,
    "p50": 0.4923763886859166,
    "p90": 0.4923763886859166,
    "p99": 0.5053220792958735,
    "max": 0.5326004825376638
  },
  "proxy_paper_style": {
    "mean": 0.568540440579273,
    "p50": 0.6011997865242837,
    "p90": 0.6011997865242837,
    "p99": 0.6011997865242837,
    "max": 0.6505603513882545
  }
}
```

| Rank | Strict EIG top condition | Proxy EIG top condition |
| --- | --- | --- |
| 1 | 90C/1800s -> C/s | 95C/1800s -> 85C/10s |
| 2 | 90C/1800s -> 90C/10s | 100C/1800s -> 80C/10s |
| 3 | 90C/1200s -> C/s | 100C/1200s -> 80C/10s |
| 4 | 100C/1800s -> 85C/10s | 100C/1800s -> 85C/10s |
| 5 | 95C/1800s -> 90C/10s | 85C/1800s -> 90C/10s |
| 6 | 100C/1200s -> 85C/10s | 100C/900s -> 80C/10s |
| 7 | 90C/900s -> C/s | 90C/1800s -> 90C/10s |
| 8 | 95C/1200s -> 85C/10s | 95C/1200s -> 85C/10s |
| 9 | 85C/1800s -> 95C/10s | 80C/1800s -> 95C/10s |
| 10 | 90C/1200s -> 90C/10s | 100C/900s -> 85C/10s |

## Interpretation

Strict `H* / S*` makes the target definition physically cleaner than the old proxy version. It does not yet make four-parameter inverse reconstruction unique, because six strict conditions cannot cover the full `T1,t1,T2,t2` space. The low-R2 `50 -> 80 C` point is useful diagnostically, but it should be down-weighted or repeated before claiming original-paper-level precision.

Minimum next additions for original-paper-style training:

```text
repeat 50 -> 80 C at 4 heating rates
add 80 -> 100 C, 90 -> 100 C, and 60 -> 100 C multi-rate ARRT points
add one low-temperature long-time point, e.g. 50 -> 70 C with t2=1800 s
```

## Outputs

- Summary: `results/ps/strict_arrt_model/strict_paper_style_benchmark.json`
- Strict EIG ranking: `results/ps/strict_arrt_model/strict_paper_style_eig_top30.csv`
- Proxy EIG ranking: `results/ps/strict_arrt_model/proxy_paper_style_eig_top30.csv`
- Strict LOOCV predictions: `results/ps/strict_arrt_model/strict_paper_style_loocv_predictions.csv`
- Proxy LOOCV predictions: `results/ps/strict_arrt_model/proxy_paper_style_loocv_predictions.csv`
