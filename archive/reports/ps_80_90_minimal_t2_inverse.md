# PS 80/90 Minimal t2 Inverse Check

## Question

Fixed `T1/T2` and known `t1 ~= 50 s or 500 s`; can the model recover `t2`? This is the reduced version of the paper-style inverse problem.

## Paper Sanity Check

| Paper target set | t2 log10 MAE | Median factor error | Top1 within factor 3 |
| --- | ---: | ---: | ---: |
| fig2_only | 0.359 | 2.000 | 0.746 |
| fig2_plus_fig3_kinetics | 0.112 | 1.000 | 0.936 |

## Current PS 80/90 Check

| PS target set | t2 log10 MAE | Median factor error | Top1 within factor 3 | Top3 within factor 3 |
| --- | ---: | ---: | ---: | ---: |
| enthalpy_only | 0.415 | 2.001 | 0.550 | 0.900 |
| heating_curve_full | 0.388 | 2.001 | 0.575 | 0.900 |

## H*/S* Availability

- strict `80 -> 90` H*/S* available: `False`
- strict `90 -> 80` H*/S* available: `False`

## Interpretation

The paper sanity check tests the logic: when Figure-3-like kinetic information is available, fixed-temperature `t2` recovery improves strongly. The current PS check tests the available experimental curves: without strict 80/90 H*/S* labels, the reduced inverse can only use final heating-curve features.

Therefore the minimum added experiment is not a full two-step matrix. It is enough to obtain strict H*/S* anchors for the two relevant temperatures/paths: multi-heating-rate measurements around 80 C and 90 C, preferably at the same `t1 ~= 50/500 s` protocol or at least single-step 80 C and 90 C ARRT anchors if the H*/S* is to be used as a temperature-specific prior.

Outputs:

- `results/ps/minimal_t2_inverse_80_90/minimal_t2_inverse_summary.json`
- `results/ps/minimal_t2_inverse_80_90/ps_80_90_t2_records.csv`
- `results/ps/minimal_t2_inverse_80_90/paper_t2_records.csv`
