# PS Inverse Reconstruction Identifiability Update

## What Changed

This update changes inverse reconstruction from a single best-condition guess into a diagnostic inverse search.

- Candidate times now match the executable DSC experiment window: `10-1800 s`.
- When `peak_temperature_Tp_C` is available, the inverse search softly limits `T2` candidates to a `+/-8 C` window around the target-derived estimate.
- The regularized inverse score now combines target matching, total time, predictive uncertainty, extrapolation penalty, and inverse-identifiability bonus.
- Outputs now include posterior summaries over the best-loss candidate set: `p5`, `median`, `p95`, width, candidate count, and loss flatness.
- TNM inverse reconstruction uses the same posterior schema and adds a delta-H state-consistency term.

## Kernel Inverse Evaluation

The updated temperature evaluation uses `57` measured two-step samples and the finite experimental time grid.

| group | n | T1 MAE C | T2 MAE C | Top-5 near | T1 posterior coverage | T2 posterior coverage | median T1 width C | median T2 width C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 50-60 | 11 | 10.91 | 15.45 | 0.09 | 0.82 | 0.27 | 30.0 | 5.0 |
| T1 65-75 | 5 | 15.00 | 10.00 | 0.00 | 0.40 | 0.60 | 20.0 | 5.0 |
| T1 80-90 | 41 | 18.54 | 11.34 | 0.05 | 0.63 | 0.02 | 30.0 | 5.0 |

The point estimate is still weak, but the posterior output is now informative: `T1` is usually broad, while `T2` is artificially narrow under the current `Tp` constraint and can miss true low/mid `T2` regions. That means this route should be treated as uncertainty-aware screening, not exact inverse reconstruction.

## TNM Inverse Evaluation

The TNM inverse run over `9801` two-step candidates gives:

| metric | value |
|---|---:|
| T1 MAE C | 14.56 |
| T2 MAE C | 15.88 |
| log10(t1) MAE | 0.752 |
| log10(t2) MAE | 0.750 |
| posterior T1 coverage | 0.772 |
| posterior T2 coverage | 0.877 |
| posterior t1 coverage | 1.000 |
| posterior t2 coverage | 0.614 |
| median posterior T1 width C | 40.0 |
| median posterior T2 width C | 35.0 |

TNM point estimates improve only slightly, but the wide posterior coverage is physically meaningful: many schedules produce similar final enthalpy-recovery observables. This supports the conclusion that the current final-heating outputs are insufficient for unique four-parameter inversion.

## Outputs

- Kernel inverse design: `results/ps/predictions/ps_inverse_design_top10.json`
- Temperature evaluation: `results/ps/inverse_temperature/inverse_temperature_summary.json`
- Per-sample inverse evaluation: `results/ps/inverse_temperature/inverse_reconstruction_by_sample.csv`
- TNM inverse summary: `results/ps/tnm_calibrated/tnm_inverse_reconstruction_summary.json`
- TNM per-sample inverse reconstruction: `results/ps/tnm_calibrated/tnm_inverse_reconstruction_by_sample.csv`
