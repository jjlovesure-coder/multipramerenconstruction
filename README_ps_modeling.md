# Polystyrene Finite-Time Enthalpy-Recovery Predictor

This workflow implements the revised PS plan: predict finite-time enthalpy recovery rather than maximizing Kovacs peaks.

## Scope

- Temperature range: 50-100 deg C.
- Maximum single annealing step: 1800 s.
- PS sample mass: 4.7 mg.
- Primary targets: `delta_h_total_J_g`, `peak_area_J_g`, `peak_temperature_Tp_C`, `peak_height_uW`, `recovery_index`.
- `kovacs_peak_label` is kept as an observational flag only.

The DSC temperature integrals are converted to specific enthalpy with:

```text
J/g = integral_uW_C * (60 / heating_rate_C_min) * 1e-6 / sample_mass_g
```

For the current PS experiments, `sample_mass_g = 0.0047` and `heating_rate_C_min = 10`.

## Commands

Build the planned experiment matrix:

```powershell
python -m src.ps.experiment_design
```

Extract features from existing PS pre-experiment Excel files:

```powershell
python -m src.ps.dsc_processing
```

Train/evaluate the finite-time predictor:

```powershell
python -m src.ps.train_ps_model
```

Train/evaluate the physics-informed PS predictor:

```powershell
python -m src.ps.train_physics_informed_ps_model
```

Search short feasible annealing conditions for a target recovery index:

```powershell
python -m src.ps.inverse_design
```

Rank and select the next finite-time PS experiments by Expected Information Gain:

```powershell
python -m src.ps.eig_design
```

## Outputs

- `data/ps/ps_limited_time_experiment_design.csv`
- `data/ps/ps_preexperiment_features.csv`
- `data/ps/ps_eig_next_experiments.csv`
- `results/ps/models/ps_limited_time_model.json`
- `results/ps/models/ps_physics_informed_kernel_model.json`
- `results/ps/evaluation/ps_train_test_metrics.json`
- `results/ps/physics_informed/physics_informed_model_comparison.json`
- `results/ps/evaluation/ps_test_predictions.csv`
- `results/ps/predictions/ps_inverse_design_top10.json`
- `results/ps/eig/ps_eig_candidate_ranking.csv`
- `results/ps/eig/ps_eig_selection_summary.json`
- `docs/ps_required_missing_data_for_prediction.md`

## Expected Information Gain Design

The EIG design is an active-learning layer over the current surrogate model. It uses
`0.5 * log(1 + predictive_variance / noise_scale^2)` as a practical information-gain
proxy, then greedily selects a constrained batch with novelty and diversity penalties.
The default selection keeps five single-step anchor experiments and uses the remaining
slots for the highest-value path-effect experiments. This gives the PS plan a sharper
information-theoretic rationale: do the finite-time experiments that are expected to
reduce uncertainty in recovery index, Tp, enthalpy area, and path dependence most
efficiently.

## Physics-Informed PS Model

The current default inverse-design and EIG entrypoints use
`results/ps/models/ps_physics_informed_kernel_model.json`. This model keeps the
raw finite-time annealing inputs and adds compact TNM/ARRT-inspired path
features. The current best feature subset is `raw_phys_beta`, which describes
total relaxation progress and two-step path excess over a small stretched
exponential basis.

For the current sparse PS dataset, the physics-informed kernel lowers the mean
core-target normalized error from `0.7997` to `0.7352` on the same grouped split.
See `docs/ps_physics_informed_model_update.md` for the metric table and
`docs/ps_required_missing_data_for_prediction.md` for the data gaps, especially
the missing direct `S*` and `H*` measurements.

`data/dsc/ps-hs-01.xlsx` adds multi-heating-rate scans for three representative
single-step annealed states. The strict ARRT/Kissinger calculation now extracts
high-temperature relaxation `Tp` values and estimates PS-specific `H* / S*`
anchors. The current physics-informed model uses the resulting high-energy
activation basis (`430/460/540 kJ/mol`) when it improves grouped CV. See
`docs/ps_hs_01_arrt_prediction_update.md`.

The EIG workflow also includes an inverse-identifiability term for the four
parameters `T1,t1,T2,t2`. It scores candidates by local sensitivity of predicted
targets to `T1`, `log(t1)`, `T2`, and `log(t2)`, then selects a 30-point batch
with single-step anchors plus diverse two-step paths. See
`docs/ps_inverse_identifiability_four_step_update.md`.

## Notes

The parser uses the instrument temperature program to split each repeated cycle into annealing and scan segments. For two-step programs it infers `T1,t1,T2,t2` from the two annealing holds preceding the 30-200 deg C scan.
