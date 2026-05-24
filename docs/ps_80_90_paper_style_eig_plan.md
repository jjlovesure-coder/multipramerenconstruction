# PS 80/90 Paper-Style EIG Experiment Plan

## Goal

The goal is not general four-parameter inverse reconstruction. It is the paper-style conditional task: with `T1,T2` fixed to the 80/90 pair, use Figure-2-like enthalpy features plus Figure-3-like `H*/S*` kinetic features to constrain `t1,t2`.

## Current Gap

- Strict ARRT training samples available: `6`.
- Strict ARRT 80/90 pairs already available: `{'80->90': False, '90->80': False}`.
- Current 80/90 DSC curves are useful for enthalpy trends, but they do not provide strict multi-rate `H*/S*` labels for the 80/90 pair.

## Recommended Minimum Batch

| Rank | T1 -> T2 | t1 s | t2 s | Reason | EIG | Uncertainty | Sensitivity | Novelty |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 80.0 -> 90.0 C | 30.0 | 30.0 | time_diagonal_anchor | 0.504 | 0.984 | 0.000 | 0.313 |
| 2 | 80.0 -> 90.0 C | 300.0 | 300.0 | time_diagonal_anchor | 0.492 | 0.984 | 0.000 | 0.314 |
| 3 | 80.0 -> 90.0 C | 900.0 | 900.0 | time_diagonal_anchor | 0.454 | 0.984 | 0.000 | 0.259 |
| 4 | 90.0 -> 80.0 C | 30.0 | 30.0 | time_diagonal_anchor | 0.504 | 0.984 | 0.000 | 0.313 |
| 5 | 90.0 -> 80.0 C | 300.0 | 300.0 | time_diagonal_anchor | 0.492 | 0.984 | 0.000 | 0.314 |
| 6 | 90.0 -> 80.0 C | 900.0 | 900.0 | time_diagonal_anchor | 0.454 | 0.984 | 0.000 | 0.259 |
| 7 | 90.0 -> 80.0 C | 900.0 | 10.0 | high_EIG_or_identifiability | 0.807 | 0.976 | 1.000 | 0.255 |
| 8 | 80.0 -> 90.0 C | 10.0 | 300.0 | high_EIG_or_identifiability | 0.648 | 0.983 | 0.000 | 0.733 |
| 9 | 90.0 -> 80.0 C | 10.0 | 300.0 | high_EIG_or_identifiability | 0.648 | 0.984 | 0.000 | 0.733 |
| 10 | 80.0 -> 90.0 C | 900.0 | 10.0 | high_EIG_or_identifiability | 0.634 | 0.861 | 0.518 | 0.255 |
| 11 | 80.0 -> 90.0 C | 10.0 | 30.0 | high_EIG_or_identifiability | 0.609 | 0.984 | 0.000 | 0.733 |
| 12 | 90.0 -> 80.0 C | 10.0 | 30.0 | high_EIG_or_identifiability | 0.609 | 0.984 | 0.000 | 0.733 |

## Measurement Requirement

Each selected condition should be measured at `5, 10, 20, 40 C/min` so that a strict Kissinger/ARRT `H*/S*` label can be calculated. A single 10 C/min heating curve only adds Figure-2-like enthalpy information and will not reproduce the paper-style Figure-3 constraint.

## Interpretation

The first batch should include both directions (`80 -> 90 C` and `90 -> 80 C`) and both balanced and asymmetric time pairs. Balanced points calibrate total relaxation dose; asymmetric points test whether the two times are separately identifiable when the temperature pair is fixed.

Outputs:

- full EIG ranking: `results/ps/eig/ps_80_90_paper_style_eig_candidates.csv`
- recommended minimum batch: `data/ps/ps_80_90_paper_style_next_experiments.csv`
