# PS Prediction Data Requirements and Gaps

## Current Prediction Goal

The practical goal is four-parameter annealing prediction for PS:

```text
T1_C, t1_s, T2_C, t2_s -> enthalpy-recovery / memory-effect response
```

The current implementation treats the paper-style thermodynamic quantities as
physics-informed features rather than hard requirements. This is appropriate for
the present sparse dataset: the model can already run inverse prediction, while
new experiments can gradually replace approximate priors with measured PS
parameters.

## Current Model Frame

The active PS model is:

```text
raw annealing inputs + TNM/ARRT-inspired path features -> RBF kernel predictor
```

Main file:

```text
src/ps/train_physics_informed_ps_model.py
```

Default inverse-design and EIG entrypoints now use:

```text
results/ps/models/ps_physics_informed_kernel_model.json
```

The selected feature set is `raw_phys_beta`, using six compact path features:

```text
tnm_state_total_e160_b035
tnm_state_total_e160_b050
tnm_state_total_e160_b065
tnm_path_excess_e160_b035
tnm_path_excess_e160_b050
tnm_path_excess_e160_b065
```

These are deterministic features computed from annealing temperature and time.
They approximate finite-time relaxation dose and two-step path memory. They are
not yet fitted PS material constants.

## What Memory Effect Means Here

For the current PS workflow, `memory_effect` is best interpreted as a
path-dependent deviation in enthalpy recovery. In code this is represented by
`path_dependence_index`, not by the old `kovacs_peak_label`.

Operationally:

```text
memory effect ~= response(two-step path) - response(path-matched/simple reference)
```

Because the current data do not yet show strong, clean Kovacs peaks, the model
uses continuous targets such as total recovered enthalpy, peak area, recovery
index, and path-dependence index. The four-parameter prediction is therefore
evaluated by comparing predicted target values against measured DSC-derived
targets for the same annealing condition. The error is target-wise prediction
error, not only a binary "has memory peak / no memory peak" classification.

## Data Already Available

The current usable PS dataset has 102 samples after adding `ps-kovacs-03.xlsx`
and `ps-kovacs-04.xlsx`.

Available per condition:

| data type | status | current use |
| --- | --- | --- |
| `T1_C`, `t1_s`, `T2_C`, `t2_s` | available | direct model input |
| DSC recovery scan | available | source for peak/enthalpy targets |
| `delta_h_total_J_g` | available | core target |
| `peak_area_J_g` | available | core target |
| `peak_temperature_Tp_C` | available | diagnostic target and possible ARRT input |
| `peak_height_uW` | available | diagnostic target |
| `recovery_index` | available | core target |
| `path_dependence_index` | available but weakly constrained | memory-effect target |

## Missing or Weak Data

The main missing data are still `S*` and `H*` measured under PS conditions.

| data | status | why it matters |
| --- | --- | --- |
| `S_star_eff_J_mol_K` | missing | needed for direct absolute-reaction-rate features and paper-style target comparison |
| `H_star_eff_kJ_mol` | missing/approximate only | can be estimated from `Tp` and heating-rate series, but direct PS estimates need repeated heating rates |
| same annealed state measured at 3+ heating rates | missing | required for robust Kissinger/ARRT fitting |
| repeated identical annealing conditions | limited | needed to estimate experimental noise and separate model error from DSC scatter |
| low/high order control pairs for two-step paths | limited | needed to isolate true path memory from total relaxation dose |
| broader `Tp` variation | weak | current PS peaks cluster in a narrow temperature range, so `Tp` models are under-constrained |

Current ARRT/Kissinger status from the PS batches:

```text
annealed-state groups inspected: 82
calculable groups with enough heating-rate coverage: 0
maximum distinct heating rates in one group: 2
minimum recommended distinct heating rates: 3
```

So `Tp` can be extracted approximately from recovery scans, but current data are
not sufficient to turn those `Tp` values into reliable direct `H*` and `S*`
training targets. Approximate `H*` can still be used as a weak feature later,
with a clear uncertainty flag.

## Recommended Next Experimental Data

Priority 1: keep the current physics-kernel model running with the existing
targets.

Measure every new condition with:

```text
T1_C, t1_s, T2_C, t2_s
DSC heating rate
sample mass
baseline / empty reference
full recovery scan
```

Extract:

```text
delta_h_total_J_g
peak_area_J_g
peak_temperature_Tp_C
peak_height_uW
recovery_index
path_dependence_index
```

Priority 2: add direct ARRT/Kissinger triplets for a few selected annealed
states.

For each selected identical annealing condition, run at least three heating
rates, for example:

```text
5 C/min, 10 C/min, 20 C/min
```

For each heating rate, extract:

```text
Tp
peak area
total recovered enthalpy
```

These triplets are the minimum practical route to estimate `H*`. Once `H*` is
stable enough, `S*` can be inferred through the absolute reaction-rate equation
or fitted jointly, but it should be stored with an uncertainty estimate.

Priority 3: add paired path-memory controls.

Useful pairs:

```text
low -> high two-step condition
high -> low two-step condition
single-step or equivalent-dose reference
repeat of one condition
```

These pairs are more valuable for `path_dependence_index` than simply adding
more isolated conditions, because they directly test whether order matters.

## Current Accuracy Level

On the same grouped train/test split:

| model | mean core MAE / test std |
| --- | --- |
| raw kernel | 0.7997 |
| physics kernel | 0.7352 |
| physics residual | 0.9025 |

The physics-kernel model is currently the best default. It improves the finite
time recovery targets modestly, but the residual model is not recommended yet
because it hurts `recovery_index` and `path_dependence_index`.

## Practical Interpretation

The model is now good enough for conservative experiment suggestion and rough
inverse design, not for precise four-parameter reconstruction. The most useful
next data are not only more rows; they are targeted rows that reduce ambiguity:

```text
1. repeated identical conditions for noise
2. 3+ heating-rate triplets for H*/S*
3. low/high and high/low two-step pairs for memory effect
4. reference single-step or equivalent-dose controls
```

