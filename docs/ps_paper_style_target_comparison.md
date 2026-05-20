# PS Current Targets vs Paper-Style Targets

## 1. Purpose

This report compares two target definitions on the same PS dataset:

```text
current targets = delta_h_total, peak_area, Tp, peak_height, onset,
                  recovery_index, path_dependence_index, kovacs_peak_label
paper-style targets = delta_h, delta_h_peak, S*, H*
```

The paper-style set is lower dimensional and closer to the original metallic-glass workflow. For PS, `S*` and `H*` are effective ARRT proxies inferred from the annealing timescale and DSC peak position, not independently measured material constants.

## 2. Effective S* / H*

For each condition the script solves the two-rate ARRT relation:

```text
k_ann  = 1 / total_anneal_time
k_peak = beta / Tp
ln(k) - ln(kB T / h) = S*/R - H*/(R T)
```

This makes `S* / H*` a compact kinetic coordinate for comparing annealing and heating-peak constraints.

## 3. Prediction Metrics

Dataset size: `105`; train rows: `99`; test rows: `6`. Both target sets use the same input features, grouped split, and RBF bandwidth.

| Target set | Target | MAE | RMSE | R2 | MAE/test std |
| --- | --- | --- | --- | --- | --- |
| Current PS targets | delta_h_total_J_g | 0.1744 | 0.1945 | -0.175 | 0.972 |
| Current PS targets | peak_area_J_g | 0.1252 | 0.139 | 0.063 | 0.872 |
| Current PS targets | peak_temperature_Tp_C | 0.6305 | 0.7484 | -3.156 | 1.718 |
| Current PS targets | peak_height_uW | 11.31 | 13.24 | 0.503 | 0.602 |
| Current PS targets | onset_temperature_C | 0.00544 | 0.005776 | -0.019 | 0.951 |
| Current PS targets | recovery_index | 0.2537 | 0.2695 | -0.842 | 1.278 |
| Current PS targets | path_dependence_index | 0.1298 | 0.148 | -0.190 | 0.957 |
| Current PS targets | kovacs_peak_label | 0 | 0 | nan | nan |
| Paper-style targets | paper_delta_h_J_g | 0.1744 | 0.1945 | -0.175 | 0.972 |
| Paper-style targets | paper_delta_h_peak_J_g | 0.1252 | 0.139 | 0.063 | 0.872 |
| Paper-style targets | paper_s_star_eff_J_mol_K | 193.8 | 258.4 | -0.025 | 0.759 |
| Paper-style targets | paper_h_star_eff_kJ_mol | 73.27 | 97.53 | -0.026 | 0.761 |

Normalized metric summary:

```json
{
  "current_targets": {
    "mean_mae_over_test_std": 1.0498236034988404,
    "median_mae_over_test_std": 0.9570356374035932
  },
  "paper_style_targets": {
    "mean_mae_over_test_std": 0.8409754436943516,
    "median_mae_over_test_std": 0.8163879290936771
  }
}
```

## 4. EIG Distribution

| Target set | Mean EIG | P90 | P99 | Max |
| --- | --- | --- | --- | --- |
| Current PS targets | 1.200 | 1.359 | 1.444 | 1.486 |
| Paper-style targets | 1.433 | 1.614 | 1.760 | 1.907 |

![EIG distribution](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\eig_distribution_current_vs_paper_style.png)

Target-correlation summary:

```json
{
  "current_targets": {
    "mean_abs_offdiag_corr": 0.4662412530474596,
    "median_abs_offdiag_corr": 0.572555145384634,
    "max_abs_offdiag_corr": 0.9524126803463178
  },
  "paper_style_targets": {
    "mean_abs_offdiag_corr": 0.5044518328651152,
    "median_abs_offdiag_corr": 0.2856086106804316,
    "max_abs_offdiag_corr": 0.9999989608730321
  }
}
```

![Current target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\current_target_correlation.png)

![Paper-style target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\paper_style_target_correlation.png)

## 5. Top EIG Conditions

| Rank | Current target top condition | Paper-style top condition |
| --- | --- | --- |
| 1 | 100C 100s -> 50C 100s | 100C 10s -> 100C 10s |
| 2 | 100C 100s -> 50C 60s | 100C 30s -> 100C 10s |
| 3 | 100C 100s -> 50C 30s | 100C 60s -> 100C 10s |
| 4 | 100C 60s -> 50C 300s | 100C 10s -> 100C 30s |
| 5 | 100C 100s -> 50C 300s | 100C 10s -> 100C 60s |
| 6 | 100C 100s -> 50C 10s | 100C 30s -> 100C 30s |
| 7 | 100C 60s -> 50C 600s | 100C 100s -> 100C 10s |
| 8 | 100C 60s -> 50C 100s | 100C 10s -> 100C 100s |
| 9 | 100C 30s -> 50C 600s | 100C 60s -> 100C 30s |
| 10 | 100C 30s -> 50C 900s | 100C 30s -> 100C 60s |

## 6. Interpretation

The current target set contains real redundancy: several DSC peak-shape descriptors describe overlapping response modes. The paper-style target set is more compact and raises the EIG contrast, but its `S* / H*` values are proxy variables inferred from `Tp` and total annealing time, so they should be used as physics-informed coordinates rather than ground-truth constants.

Recommended hybrid target set for the next model:

```text
delta_h_total_J_g
peak_area_J_g
paper_s_star_eff_J_mol_K
paper_h_star_eff_kJ_mol
path_dependence_index
```

Keep `Tp / peak_height / onset / recovery_index` as diagnostics or derived outputs.

## 7. Outputs

- Summary: `results/ps/paper_style_comparison/summary.json`
- Paper-style model: `results/ps/paper_style_comparison/paper_style_model.json`
- Paper-style predictions: `results/ps/paper_style_comparison/paper_style_test_predictions.csv`
- Current-target EIG ranking: `results/ps/paper_style_comparison/current_target_eig_ranking_top100.csv`
- Paper-style EIG ranking: `results/ps/paper_style_comparison/paper_style_eig_ranking_top100.csv`
- Figures: `results/ps/paper_style_comparison/figures/`
