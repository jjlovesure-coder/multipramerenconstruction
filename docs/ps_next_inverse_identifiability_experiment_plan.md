# Next PS Experiments For Inverse Identifiability

## Purpose

This batch is designed to improve inverse reconstruction diagnostics, not to maximize Kovacs peaks. The current inverse evaluation shows that single best-condition reconstruction remains weak, while posterior diagnostics reveal where the inverse problem is under-constrained.

The next round therefore targets:

- low/mid `T2` regions where posterior `T2` coverage is poor;
- down-jump paths, which currently have zero top-k near-match and weak `T2` coverage;
- the sparse `T1 = 65-75 C` interval;
- paired time-contrast experiments that separate `T1/t1` from `T2/t2` compensation.

## Exported Batch

Machine-readable file:

```text
data/ps/ps_next_inverse_identifiability_experiments.csv
```

The batch contains `24` experiments:

| batch | count | role |
|---|---:|---|
| A_downjump_T2_low | 8 | Repair low/mid `T2` and down-jump identifiability |
| B_mid_T1_gap | 6 | Fill sparse `T1 = 65-75 C` interval |
| C_time_identifiability | 4 | Separate temperature-time compensation |
| D_high_T2_identifiability | 2 | High-confidence high-`T2` anchors |
| E_single_step_controls | 4 | Single-step controls for baseline recalibration |

All annealing steps obey the current executable constraint:

```text
single step time <= 1800 s
temperature range = 50-100 C
```

Some two-step total times exceed 1800 s because each step is individually bounded by 1800 s.

## Priority

Run order:

1. `P0` rows first: ranks `1-11`. These directly target the worst inverse gaps.
2. `P1` rows next: ranks `12-20`. These improve time/path identifiability and anchor high-`T2` behavior.
3. `P2` rows last: ranks `21-24`. These are single-step controls; useful but less urgent if instrument time is limited.

If only `12` experiments are feasible, run ranks `1-12`.

If only `18` experiments are feasible, run ranks `1-18`.

## Expected Diagnostic Value

This design should not be judged only by top-1 reconstruction accuracy. The intended success criteria are:

- posterior `T2` coverage improves in the `T2 50-70` and `T2 75-90` bins;
- posterior widths shrink for `T1_C` and `log10_t2_s`;
- down-jump top-10 near-match becomes nonzero;
- TNM and kernel inverse posteriors agree more often on the same feasible parameter region.

If point-estimate MAE remains large but posterior coverage improves, the result is still useful: it means the experiment is clarifying the many-to-one inverse landscape rather than pretending there is a unique inverse solution.
