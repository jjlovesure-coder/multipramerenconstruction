# Multiparameter Annealing Reconstruction

This repository is now organized around the active polystyrene (PS) DSC
workflow for finite-time enthalpy-recovery prediction, inverse candidate
screening, and active-learning experiment design.

The older Song-paper digitization route is preserved as historical support
material under `archive/paper_route/`. It is not the active workflow.

## Active Scope

- Parse PS DSC Excel exports into process and recovery features.
- Train sparse forward surrogates for finite-time enthalpy recovery.
- Compare raw and physics-informed kernels with TNM-style dose/path features.
- Search feasible annealing histories for inverse candidates.
- Rank next experiments with an expected-information-gain proxy.
- Maintain the manuscript draft and the current figures used by it.

The current PS enthalpy integration window is `70-105 deg C`; legacy wider-window
fields are retained in generated feature tables only for audit comparison.

## Repository Layout

```text
src/ps/
  Active PS DSC parsing, model training, inverse search, EIG, ARRT/Kissinger,
  and TNM diagnostics.

src/models/
  Shared modeling utilities still used by the PS workflow. Historical paper
  route entrypoints remain only for compatibility.

src/physics/
  Shared ARRT/Kissinger/TNM helper functions.

data/dsc/
  Raw PS DSC Excel exports.

data/ps/
  Current processed PS feature tables and experiment plans.

results/ps/
  Current PS model payloads, metrics, EIG selections, and TNM diagnostics.

docs/
  Current PS reports plus an archive index.

manuscript/
  LaTeX working draft and manuscript figures.

archive/
  Historical paper-route assets and old stage reports.
```

## Environment

The project is plain Python and has been run from Windows PowerShell:

```powershell
python --version
python -m pip install numpy pandas matplotlib openpyxl pytest
```

## Active Commands

Extract PS features from the current DSC files:

```powershell
python -m src.ps.dsc_processing
```

Train the finite-time PS forward model:

```powershell
python -m src.ps.train_ps_model
```

Train and compare the physics-informed PS model:

```powershell
python -m src.ps.train_physics_informed_ps_model
```

Search feasible annealing histories for inverse candidates:

```powershell
python -m src.ps.inverse_design
```

Rank the next experiments by expected information gain:

```powershell
python -m src.ps.eig_design
```

Run current TNM-style curve and enthalpy-time diagnostics:

```powershell
python -m src.ps.tnm_curve_fit
python -m src.ps.tnm_enthalpy_time_fit
python -m src.ps.tnm_inverse_reconstruction
```

## Current Key Outputs

```text
data/ps/ps_preexperiment_features.csv
data/ps/ps_eig_next_experiments.csv
results/ps/models/ps_limited_time_model.json
results/ps/models/ps_physics_informed_kernel_model.json
results/ps/evaluation/ps_train_test_metrics.json
results/ps/evaluation/ps_test_predictions.csv
results/ps/physics_informed/physics_informed_model_comparison.json
results/ps/predictions/ps_inverse_design_top10.json
results/ps/eig/ps_eig_candidate_ranking.csv
results/ps/eig/ps_eig_selection_summary.json
results/ps/tnm_curve_fit/tnm_curve_fit_summary.json
results/ps/tnm_enthalpy_time_fit/tnm_enthalpy_time_fit_summary.json
```

## Current Reports

```text
README_ps_modeling.md
docs/ps_70_105_tnm_arrt_audit.md
docs/ps_arrt_kissinger_calculation.md
docs/ps_physics_informed_model_update.md
docs/ps_required_missing_data_for_prediction.md
docs/ps_tnm_curve_fit_forward_report.md
docs/ps_tnm_enthalpy_time_fit_report.md
docs/physics_forward_inverse_diagnostic.md
```

Older reports are listed in `docs/archive_index.md`.

## Verification

Focused PS checks:

```powershell
python -m pytest tests/test_ps_process_features.py tests/test_ps_physics_informed_entrypoints.py tests/test_ps_tnm_curve_fit.py tests/test_ps_tnm_enthalpy_time_fit.py -q
```

Compile active modules:

```powershell
python -m py_compile src\ps\dsc_processing.py src\ps\train_ps_model.py src\ps\train_physics_informed_ps_model.py src\ps\inverse_design.py src\ps\eig_design.py src\ps\tnm_curve_fit.py src\ps\tnm_enthalpy_time_fit.py src\ps\tnm_inverse_reconstruction.py src\models\kernel_regression.py src\physics\arrt.py src\physics\arrt_kissinger.py
```

## Notes

- The forward surrogate is the central object; inverse reconstruction is treated
  as search/screening because the four-parameter thermal history is
  underdetermined from a small set of DSC descriptors.
- The manuscript draft is a working methods and experiment-design paper. Claims
  should stay tied to the current grouped split and current validation state.
