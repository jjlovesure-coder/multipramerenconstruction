# ps-single-hs-01 Single-Step Time Prediction

## Scope

This report evaluates the reduced single-step inverse task:

```text
given T and measured output features -> predict annealing time t
```

The `ps-single-hs-01.xlsx` tail is intentionally handled strictly: the incomplete `90 C, 1000 s` group is retained in truncation diagnostics but excluded from strict H*/S* training and time prediction.

## Valid Strict ARRT Conditions

- Total valid single-step strict ARRT rows: `8`
- Candidate times by temperature:

```json
{
  "70": [
    300.0
  ],
  "80": [
    50.0,
    500.0,
    1000.0
  ],
  "90": [
    50.0,
    100.0,
    500.0
  ],
  "95": [
    100.02
  ]
}
```

## Prediction Metrics

| Feature set | Strategy | n | log10(t) MAE | median factor error | Top1 within 3x |
| --- | ---: | ---: | ---: | ---: | ---: |
| heating-only | nearest candidate | 6 | 0.534 | 2.000 | 0.667 |
| paper-style H*/S* | nearest candidate | 6 | 0.534 | 2.000 | 0.667 |
| heating-only | direct log-time fit | 6 | 0.395 | 2.355 | 0.833 |
| paper-style H*/S* | direct log-time fit | 6 | 0.307 | 1.588 | 0.833 |

## Interpretation

The nearest-candidate strategy is a conservative discrete inverse and can be overly pessimistic with only two or three candidate times. The direct-fit strategy is closer to simply fitting the measured points in log-time. It shows that direct fitting improves over nearest-candidate ranking, but the current `H*/S*` coordinates still do not create a decisive single-step time predictor for the `90 C, 500 s` holdout.

## Outputs

- Summary: `results/ps/single_time_inverse/single_time_inverse_summary.json`
- Per-condition records: `results/ps/single_time_inverse/single_time_inverse_records.csv`
- Truncation diagnostics: `results/ps/arrt_kissinger/scan_truncation_diagnostics.csv`
