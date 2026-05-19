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
- `results/ps/evaluation/ps_train_test_metrics.json`
- `results/ps/evaluation/ps_test_predictions.csv`
- `results/ps/predictions/ps_inverse_design_top10.json`
- `results/ps/eig/ps_eig_candidate_ranking.csv`
- `results/ps/eig/ps_eig_selection_summary.json`

## Expected Information Gain Design

The EIG design is an active-learning layer over the current surrogate model. It uses
`0.5 * log(1 + predictive_variance / noise_scale^2)` as a practical information-gain
proxy, then greedily selects a constrained batch with novelty and diversity penalties.
The default selection keeps five single-step anchor experiments and uses the remaining
slots for the highest-value path-effect experiments. This gives the PS plan a sharper
information-theoretic rationale: do the finite-time experiments that are expected to
reduce uncertainty in recovery index, Tp, enthalpy area, and path dependence most
efficiently.

## Notes

The parser uses the instrument temperature program to split each repeated cycle into annealing and scan segments. For two-step programs it infers `T1,t1,T2,t2` from the two annealing holds preceding the 30-200 deg C scan.
