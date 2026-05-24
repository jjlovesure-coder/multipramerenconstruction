# Paper-Style Conditional t1/t2 Inverse Validation

## Purpose

This validation fixes the original-paper temperature pair and asks whether time can be recovered from paper-style observables. `delta_h` and `delta_h_peak` are matched through a leave-one-out forward kernel, while Figure-3-like kinetic constraints are matched directly from the candidate time coordinates.

## Overall Metrics

| Variant | Targets | t1 log10 MAE | t2 log10 MAE | Top5 near match | Posterior width t1 | Posterior width t2 |
| --- | --- | --- | --- | --- | --- | --- |
| fig2_only | delta_h_kj_mol, delta_h_peak_kj_mol | 1.177 | 0.452 | 0.451 | 1.371 | 0.771 |
| fig2_plus_fig3_kinetics | delta_h_kj_mol, delta_h_peak_kj_mol, s1_j_mol_k, s2_j_mol_k, delta_s_j_mol_k, h1_kj_mol_fig3, h2_kj_mol_fig3 | 0.054 | 0.112 | 0.990 | 0.837 | 0.949 |

## Interpretation

The comparison is the benchmark for the PS 80/90 question. If adding H*/S* style kinetic information sharply narrows the posterior on the digitized paper data, then PS 80/90 needs the same kind of multi-rate ARRT labels rather than only more single-heating DSC curves.

Outputs:

- `results/paper_conditional_time_inverse/paper_conditional_time_inverse_summary.json`
- `results/paper_conditional_time_inverse/paper_conditional_time_inverse_by_sample.csv`
