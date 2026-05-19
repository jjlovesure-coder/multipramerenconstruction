# PS Four-Step Inverse Identifiability Update

## Goal

The target is not only forward prediction. The practical goal is to make the
four annealing parameters identifiable over a wider PS temperature range:

```text
T1_C, t1_s, T2_C, t2_s
```

The workflow now follows four steps:

```text
1. Use ps-hs-01 to strengthen H*/S* physical priors.
2. Add more explicit sequential TNM/ARRT state features.
3. Upgrade EIG to include inverse-parameter identifiability.
4. Generate a broader next-experiment batch for t1/t2 reconstruction.
```

## Step 1: H*/S* Physical Priors

`ps-hs-01.xlsx` remains the strict multi-rate source for PS activation
parameters. The high-temperature relaxation peak is used for Kissinger fitting.

| condition | H* kJ/mol | S* J/mol/K | Kissinger R2 |
| --- | ---: | ---: | ---: |
| 70 C, 300 s | 427.25 | 910.21 | 0.9856 |
| 90 C, 100 s | 540.67 | 1213.57 | 0.9925 |
| 95 C, 100 s | 455.42 | 986.87 | 0.9889 |

These anchors are now available in `src/ps/physics_features.py` as
`PS_HS_ARRT_ANCHORS`.

## Step 2: Sequential State Features

The physics feature module now computes two classes of physics descriptors.

First, the previous TNM-like activation-energy basis remains available:

```text
tnm_state_step1
tnm_state_step2_only
tnm_state_total
tnm_path_excess
tnm_step1_fraction_of_total
tnm_step2_increment
```

Second, a new ARRT sequential state basis uses the measured PS `H*/S*` anchors:

```text
arrt_state_step1
arrt_state_step2_only
arrt_state_total
arrt_state_equivalent
arrt_path_excess
arrt_step_mismatch
arrt_step1_fraction_of_total
arrt_step2_increment
```

Grouped CV tested the old and new feature sets. The selected model is still:

```json
{
  "feature_set": "raw_phys_ps_hs",
  "bandwidth": 1.4
}
```

Interpretation: the new explicit ARRT sequential features are implemented and
available, but the current sparse dataset does not yet justify using them as
the default predictor. The best current sparse-data compromise is still the
compact high-energy PS basis:

```text
tnm_state_total_e430/e460/e540
tnm_path_excess_e430/e460/e540
```

Same-split core normalized error:

| model | mean core MAE / test std |
| --- | ---: |
| raw kernel | 1.0196 |
| physics kernel | 0.8050 |
| physics residual | 0.8187 |

## Step 3: EIG + Inverse Identifiability

The EIG score now has two parts:

```text
EIG = 0.5 * log(1 + predictive_variance / noise_scale^2)
```

and a new inverse-identifiability score:

```text
logdet(J^T J)
```

where `J` is the local sensitivity matrix of predicted targets with respect to:

```text
T1, log(t1), T2, log(t2)
```

High score means a small change in the four inverse parameters produces
distinguishable predicted target changes. This is aimed directly at the current
weak point: `t1/t2` ambiguity.

The two-step candidate space now allows all path directions:

```text
up-jump, down-jump, isothermal
```

instead of only `T2 >= T1`.

## Step 4: New Experiment Batch

The new default next-experiment file is:

```text
data/ps/ps_eig_next_experiments.csv
```

It contains 30 conditions:

| type | count |
| --- | ---: |
| single-step anchors | 4 |
| two-step paths | 26 |

Path coverage among two-step conditions:

| path class | count |
| --- | ---: |
| up-jump | 20 |
| down-jump | 2 |
| isothermal | 4 |

Temperature coverage:

| field | covered values |
| --- | --- |
| T1 | 60, 65, 75, 80, 85, 90, 95, 100 C |
| T2 | 50, 85, 90, 95, 100 C |

The mean inverse-identifiability score across the 30 selected points is `0.817`.
Most selected two-step points have score near `1.0`.

Top inverse-identifiability two-step points:

| rank | T1_C | t1_s | T2_C | t2_s | identifiability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 95 | 60 | 100 | 10 | 1.000 |
| 6 | 85 | 1200 | 100 | 100 | 1.000 |
| 7 | 75 | 30 | 95 | 100 | 0.997 |
| 8 | 90 | 10 | 100 | 30 | 0.997 |
| 10 | 95 | 300 | 95 | 10 | 0.998 |
| 11 | 95 | 300 | 100 | 10 | 1.000 |
| 12 | 90 | 100 | 95 | 60 | 1.000 |

## Updated Inverse Reconstruction Status

Using the current physics-informed model and all two-step PS rows:

| metric | value |
| --- | ---: |
| two-step evaluation rows | 57 |
| T1 MAE | 8.60 C |
| T2 MAE | 10.26 C |
| median t1 factor error | 49.98x |
| median t2 factor error | 5.00x |
| top-1 near-match rate | 5.3% |
| top-5 near-match rate | 15.8% |
| top-10 near-match rate | 17.5% |

The main conclusion is unchanged: forward prediction is improving, but exact
four-parameter inverse reconstruction is still limited by time-parameter
non-identifiability. The new EIG design is intentionally biased toward
experiments that should make `t1` and `t2` easier to distinguish.

## Recommended Use

Run the 30-point EIG + identifiability batch first if instrument time permits.
If only a smaller batch is feasible, prioritize the high-identifiability
two-step rows with ranks 5-12 and keep at least two single-step anchors from
ranks 1-4.

The new down-jump and isothermal points are important even if their EIG is not
always maximal, because they help distinguish path direction and total-dose
effects.

