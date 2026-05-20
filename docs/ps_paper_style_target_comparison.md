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
| Current PS targets | delta_h_total_J_g | 0.1445 | 0.1719 | 0.082 | 0.805 |
| Current PS targets | peak_area_J_g | 0.09277 | 0.1179 | 0.327 | 0.646 |
| Current PS targets | peak_temperature_Tp_C | 0.4347 | 0.5966 | -1.641 | 1.184 |
| Current PS targets | peak_height_uW | 7.192 | 9.099 | 0.765 | 0.383 |
| Current PS targets | recovery_index | 0.1951 | 0.2158 | -0.181 | 0.982 |
| Current PS targets | path_dependence_index | 0.09926 | 0.124 | 0.164 | 0.732 |
| Paper-style targets | paper_delta_h_J_g | 0.1445 | 0.1719 | 0.082 | 0.805 |
| Paper-style targets | paper_delta_h_peak_J_g | 0.09277 | 0.1179 | 0.327 | 0.646 |
| Paper-style targets | paper_s_star_eff_J_mol_K | 246.3 | 329.8 | -0.669 | 0.965 |
| Paper-style targets | paper_h_star_eff_kJ_mol | 92.98 | 124.5 | -0.671 | 0.966 |

Normalized metric summary:

```json
{
  "current_targets": {
    "mean_mae_over_test_std": 0.7887167759073604,
    "median_mae_over_test_std": 0.7685310101254524
  },
  "paper_style_targets": {
    "mean_mae_over_test_std": 0.8453530127796917,
    "median_mae_over_test_std": 0.8849677829733256
  }
}
```

## 4. EIG Distribution

| Target set | Mean EIG | P90 | P99 | Max |
| --- | --- | --- | --- | --- |
| Current PS targets | 1.179 | 1.609 | 1.759 | 1.823 |
| Paper-style targets | 1.221 | 1.589 | 1.726 | 1.834 |

![EIG distribution](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\eig_distribution_current_vs_paper_style.png)

Target-correlation summary:

```json
{
  "current_targets": {
    "mean_abs_offdiag_corr": 0.6394515650887693,
    "median_abs_offdiag_corr": 0.6570493379427871,
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
| 1 | 50C 900s -> 55C 900s | 100C 10s -> 50C 1800s |
| 2 | 50C 1200s -> 55C 1200s | 100C 10s -> 50C 1200s |
| 3 | 50C 900s -> 50C 900s | 100C 10s -> 50C 900s |
| 4 | 50C 600s -> 50C 600s | 100C 10s -> 50C 600s |
| 5 | 50C 1200s -> 50C 1200s | 95C 10s -> 50C 1800s |
| 6 | 50C 1200s -> 55C 900s | 100C 30s -> 50C 1800s |
| 7 | 50C 1200s -> 60C 1200s | 100C 1800s -> 50C 10s |
| 8 | 50C 900s -> 55C 1200s | 100C 10s -> 55C 1800s |
| 9 | 50C 600s -> 55C 600s | 95C 10s -> 50C 1200s |
| 10 | 55C 600s -> 60C 600s | 100C 30s -> 50C 1200s |

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
