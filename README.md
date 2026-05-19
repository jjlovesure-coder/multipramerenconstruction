# Multiprameter Annealing Reconstruction

本仓库用于从论文图像和聚苯乙烯 DSC 实验中提取焓回复数据，构建稀疏样本下的退火参数预测模型，并用 Expected Information Gain (EIG) 设计下一轮实验。

当前项目包含两条互相衔接的路线：

1. 复现论文中金属玻璃的多参数预测思路：`delta_h`、`delta_h_peak`、`S*`、`H*`、TNM 特征和绝对反应速率理论特征。
2. 将同样的思想迁移到聚苯乙烯 (PS) DSC 实验：在有限退火时间内预测焓回复、峰面积、峰位、路径依赖，并设计能计算 `S* / H*` 的多升温速率实验。

## Repository Layout

```text
paper/
  Source paper and supplementary PDF files.

tools/
  One-off extraction and method-generation scripts.

src/models/
  Paper-data sparse forward and inverse predictor.

src/physics/
  Absolute reaction rate theory, Kissinger, and TNM helper functions.

src/ps/
  PS DSC parsing, feature extraction, model training, EIG design,
  inverse evaluation, and ARRT/Kissinger diagnostics.

data/dsc/
  Raw PS DSC Excel exports.

data/ps/
  Processed PS feature tables and experiment plans.

data/processed/
  Processed digitized paper dataset.

results/
  Digitized figure data, trained model payloads, metrics, plots, and reports.

docs/
  Experiment design reports and model evaluation notes.
```

## Environment

The code is plain Python and uses common scientific packages:

```powershell
python --version
python -m pip install numpy pandas matplotlib openpyxl
```

The project has been run on Windows PowerShell from:

```text
E:\knowledgepaper\multipramereconstruction
```

## Paper-Derived Dataset Workflow

The paper and supplementary figures do not provide raw tables, so the first dataset is digitized from figures.

Main source files:

```text
paper/Song 等 - 2020 - Activation Entropy as a Key Factor Controlling the Memory Effect in Glasses.pdf
paper/Revised_Supplementary_Materials-Sep_3.pdf
```

Important extracted outputs:

```text
results/extracted/data_points/overlap_enhanced/fig2a_c_10series_overlap_filled_points.csv
results/extracted/data_points/fig2d_s6_experimental_points.csv
results/extracted/data_points/s7_experimental_points.csv
results/extracted/data_points/fig3b_hstar_tp_digitized.csv
```

Build the processed paper dataset:

```powershell
python -m src.models.dataset
```

Train the sparse forward model:

```powershell
python -m src.models.train_forward
```

Run inverse design on paper-style targets:

```powershell
python -m src.models.inverse_design
python tools/run_inverse_design_example.py
```

Key outputs:

```text
data/processed/annealing_dataset.csv
results/models/forward_kernel_model.json
results/predictions/inverse_design_top10.json
```

## PS DSC Workflow

The PS workflow predicts finite-time enthalpy recovery rather than forcing a maximum Kovacs peak. The current sample mass is:

```text
PS sample mass = 4.7 mg
```

DSC temperature integrals are converted to specific enthalpy by:

```text
J/g = integral_uW_C * (60 / heating_rate_C_min) * 1e-6 / sample_mass_g
```

Raw DSC files are read from:

```text
data/dsc/ps-empty-01.xlsx
data/dsc/ps-ref-01.xlsx
data/dsc/PS-onestep-01.xlsx
data/dsc/PS-onestep-02.xlsx
data/dsc/ps-single-01.xlsx
data/dsc/ps-kovacs-01.xlsx
data/dsc/pskovacs.xlsx
data/dsc/twosteps.xlsx
```

Extract PS features:

```powershell
python -m src.ps.dsc_processing
```

Train the current finite-time PS model:

```powershell
python -m src.ps.train_ps_model
```

Run inverse design:

```powershell
python -m src.ps.inverse_design
```

Run EIG experiment selection:

```powershell
python -m src.ps.eig_design
python -m src.ps.optimize_twostep_plan
```

Key outputs:

```text
data/ps/ps_preexperiment_features.csv
data/ps/ps_eig_next_experiments.csv
data/ps/ps_eig_next_twostep_experiments.csv
results/ps/models/ps_limited_time_model.json
results/ps/evaluation/ps_train_test_metrics.json
results/ps/eig/ps_eig_twostep_candidate_ranking.csv
```

## Current PS Targets

The current PS model predicts:

```text
delta_h_total_J_g
peak_area_J_g
peak_temperature_Tp_C
peak_height_uW
onset_temperature_C
recovery_index
path_dependence_index
kovacs_peak_label
```

`kovacs_peak_label` is kept as an observational label. Current PS pre-experiments do not show a strong finite-time Kovacs peak, so the main objective is finite-time enthalpy recovery and path dependence.

## Paper-Style Target Comparison

To test whether the current target set is too redundant, the repository also compares it with a lower-dimensional paper-style target set:

```text
delta_h
delta_h_peak
S*
H*
```

Run:

```powershell
python -m src.ps.compare_paper_style_targets
```

Important outputs:

```text
docs/ps_paper_style_target_comparison.md
results/ps/paper_style_comparison/summary.json
results/ps/paper_style_comparison/paper_style_model.json
results/ps/paper_style_comparison/paper_style_test_predictions.csv
results/ps/paper_style_comparison/figures/
```

Current conclusion: the PS target set does contain redundancy, but directly using `S* / H*` as proxy labels can bias EIG toward high-temperature short-time regions. A hybrid target set is recommended:

```text
delta_h_total_J_g
peak_area_J_g
paper_s_star_eff_J_mol_K
paper_h_star_eff_kJ_mol
path_dependence_index
```

## Strict ARRT/Kissinger S* And H*

The strict paper route requires multiple final heating rates for the same annealed state. The implemented calculation uses:

```text
ln(beta / Tp^2) = C - E / (R Tp)
E = -R * slope
H* = E - R Tp
```

Then `S*` is calculated from the absolute-rate peak equation:

```text
ln(beta / Tp^2) = -H* / (R Tp) + ln(kB R / (h H*)) + S* / R
```

Run diagnostics on existing PS data:

```powershell
python -m src.ps.calculate_arrt_from_heating_rates
```

Attempt strict `S*/H*` training:

```powershell
python -m src.ps.train_strict_arrt_model
```

Current status:

```text
Total final heating scans: 95
Condition groups: 74
Strictly calculable S*/H* groups: 0
Maximum heating-rate count for the same condition: 2
Minimum required heating-rate count: 3
```

Therefore strict `S*/H*` labels cannot yet be trained from the current PS dataset. More multi-rate DSC runs are needed.

Reports and outputs:

```text
docs/ps_arrt_kissinger_calculation.md
results/ps/arrt_kissinger/all_recovery_scan_features.csv
results/ps/arrt_kissinger/condition_rate_diagnostics.csv
results/ps/arrt_kissinger/arrt_kissinger_results.csv
results/ps/strict_arrt_model/strict_arrt_training_skipped.json
```

## Generated DSC Method Files

The repository contains generated `.mtd` method files for follow-up DSC experiments.

Full two-step EIG plan:

```text
data/kovactrain26111103R9-03.mtd
```

Next-round two-step priority plan:

```text
data/kovactrain26111103R9-04-next-round.mtd
```

Multi-heating-rate ARRT/Kissinger plan for strict `S*/H*` training:

```text
data/kovactrain26111103R9-05-arrt-kissinger.mtd
data/ps/ps_arrt_kissinger_mtd_plan.csv
```

Regenerate the ARRT/Kissinger method:

```powershell
python tools/generate_ps_arrt_kissinger_mtd.py
```

This method repeats six representative annealed states at:

```text
5, 10, 20, 40 deg C/min
```

The selected states are:

```text
single: 70 deg C 300 s
single: 90 deg C 100 s
single: 95 deg C 100.02 s
two-step: 50 deg C 10 s -> 80 deg C 1800 s
two-step: 50 deg C 10 s -> 100 deg C 1800 s
two-step: 65 deg C 10 s -> 100 deg C 1200 s
```

## Evaluation Reports

Main PS reports:

```text
docs/ps_eig_experiment_design.md
docs/ps_single_01_analysis_and_twostep_update.md
docs/ps_kovacs_01_update_analysis.md
docs/ps_inverse_temperature_interval_evaluation.md
docs/ps_paper_style_target_comparison.md
docs/ps_arrt_kissinger_calculation.md
```

Inverse temperature interval evaluation outputs:

```text
results/ps/inverse_temperature/inverse_temperature_summary.json
results/ps/inverse_temperature/figures/
```

The current inverse reconstruction conclusion is conservative: forward prediction is usable for screening, but inverse recovery of all four annealing parameters is still underdetermined in most temperature intervals.

## Typical End-To-End Commands

For PS finite-time prediction:

```powershell
python -m src.ps.dsc_processing
python -m src.ps.train_ps_model
python -m src.ps.optimize_twostep_plan
```

For paper-style comparison:

```powershell
python -m src.ps.compare_paper_style_targets
```

For strict ARRT/Kissinger diagnostics:

```powershell
python -m src.ps.calculate_arrt_from_heating_rates
python -m src.ps.train_strict_arrt_model
python tools/generate_ps_arrt_kissinger_mtd.py
```

For source-code sanity checks:

```powershell
python -m py_compile src\physics\arrt_kissinger.py src\ps\calculate_arrt_from_heating_rates.py src\ps\train_strict_arrt_model.py src\ps\compare_paper_style_targets.py tools\generate_ps_arrt_kissinger_mtd.py
```

## Modeling Notes

- The current models are sparse, physics-informed screening models, not high-precision predictive tools.
- EIG is implemented as a practical uncertainty proxy:

```text
0.5 * log(1 + predictive_variance / noise_scale^2)
```

- DSC-derived `delta_h_total_J_g` and `peak_area_J_g` depend on baseline subtraction and integration windows.
- Strict `S*/H*` calculation requires repeated final heating scans at multiple rates for the same annealed state.
- If only one heating rate exists, `S*/H*` can only be treated as a proxy or digitized/borrowed feature, not as a strict paper-equivalent experimental label.

## GitHub Remote

Current working branch:

```text
Richard
```

Remote repository:

```text
https://github.com/jjlovesure-coder/multipramerenconstruction.git
```

Push command used in this workspace:

```powershell
git -c http.proxy= -c https.proxy= push origin Richard
```

## Related Short READMEs

- `README_modeling.md`: original paper-data sparse predictor.
- `README_ps_modeling.md`: PS finite-time enthalpy-recovery predictor.

