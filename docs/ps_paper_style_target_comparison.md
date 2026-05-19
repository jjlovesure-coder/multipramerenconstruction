# PS 论文式目标集与当前目标集对比

## 1. 问题

当前 PS 模型使用多目标输出：

```text
delta_h_total_J_g, peak_area_J_g, Tp, peak_height, onset, recovery_index,
path_dependence_index, kovacs_peak_label
```

这组目标描述更完整，但也可能引入信息冗余，尤其是 `delta_h_total`、
`peak_area`、`peak_height`、`recovery_index` 之间高度相关。为检验这一点，
本脚本把同一批 PS 数据重新整理成论文风格的低维目标：

```text
delta_h, delta_h_peak, S*, H*
```

其中 `delta_h` 和 `delta_h_peak` 分别对应 PS 的总焓回复面积与局部峰面积；
`S*` 和 `H*` 不是独立实验测量值，而是由 ARRT 两点约束反推的有效代理量。

## 2. 有效 S* / H* 的计算

对每条 PS 工况构造两个速率约束：

```text
k_ann  = 1 / total_anneal_time
k_peak = beta / Tp
ln(k) - ln(kB T / h) = S*/R - H*/(R T)
```

用等效退火温度 `T_ann` 和 DSC 峰位 `Tp` 联立求解 `S*` 与 `H*`。
这个定义的意义是：让 `S* / H*` 成为描述“退火过程和升温峰位是否一致”的
低维动力学坐标，而不是声称已经测得了真正材料本征活化熵/焓。

## 3. 同条件预测结果

数据集样本数：`94`；训练集：`67`；
测试集：`27`。两组模型使用相同输入特征、相同 grouped split、
相同 RBF kernel bandwidth。

| 目标集 | 目标 | MAE | RMSE | R2 | MAE/测试std |
| --- | --- | --- | --- | --- | --- |
| 当前PS目标 | delta_h_total_J_g | 0.1952 | 0.2171 | 0.214 | 0.797 |
| 当前PS目标 | peak_area_J_g | 0.1511 | 0.1667 | 0.224 | 0.799 |
| 当前PS目标 | peak_temperature_Tp_C | 0.3444 | 0.3737 | 0.302 | 0.770 |
| 当前PS目标 | peak_height_uW | 26.92 | 32.12 | -0.170 | 0.907 |
| 当前PS目标 | onset_temperature_C | 0.004476 | 0.005602 | -0.415 | 0.950 |
| 当前PS目标 | recovery_index | 0.1682 | 0.1938 | -0.090 | 0.906 |
| 当前PS目标 | path_dependence_index | 0.1948 | 0.2375 | -0.057 | 0.843 |
| 当前PS目标 | kovacs_peak_label | 0 | 0 | nan | nan |
| 论文式目标 | paper_delta_h_J_g | 0.1952 | 0.2171 | 0.214 | 0.797 |
| 论文式目标 | paper_delta_h_peak_J_g | 0.1511 | 0.1667 | 0.224 | 0.799 |
| 论文式目标 | paper_s_star_eff_J_mol_K | 151.4 | 189.5 | 0.501 | 0.564 |
| 论文式目标 | paper_h_star_eff_kJ_mol | 57.32 | 71.75 | 0.500 | 0.565 |

归一化综合误差：

```json
{
  "current_targets": {
    "mean_mae_over_test_std": 0.8532940379346564,
    "median_mae_over_test_std": 0.8433466771675275
  },
  "paper_style_targets": {
    "mean_mae_over_test_std": 0.6813468847288245,
    "median_mae_over_test_std": 0.6810863658880486
  }
}
```

## 4. EIG 分布差异

| 目标集 | EIG均值 | P90 | P99 | 最大值 |
| --- | --- | --- | --- | --- |
| 当前PS目标 | 1.244 | 1.542 | 1.596 | 1.617 |
| 论文式目标 | 1.299 | 1.642 | 2.117 | 2.541 |

![EIG distribution](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\eig_distribution_current_vs_paper_style.png)

目标间相关性摘要：

```json
{
  "current_targets": {
    "mean_abs_offdiag_corr": 0.4803514581836244,
    "median_abs_offdiag_corr": 0.6172844286255597,
    "max_abs_offdiag_corr": 0.9517106500866649
  },
  "paper_style_targets": {
    "mean_abs_offdiag_corr": 0.5049900823458587,
    "median_abs_offdiag_corr": 0.2826186405820037,
    "max_abs_offdiag_corr": 0.9999991461924681
  }
}
```

![Current target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\current_target_correlation.png)

![Paper-style target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\paper_style_target_correlation.png)

## 5. Top EIG 工况差异

| Rank | 当前PS目标Top工况 | 论文式目标Top工况 |
| --- | --- | --- |
| 1 | 50C 10s -> 65C 1800s | 100C 10s -> 100C 10s |
| 2 | 50C 10s -> 65C 1200s | 100C 30s -> 100C 10s |
| 3 | 50C 10s -> 65C 900s | 100C 10s -> 100C 30s |
| 4 | 50C 10s -> 65C 600s | 100C 60s -> 100C 10s |
| 5 | 50C 10s -> 60C 1800s | 100C 10s -> 100C 60s |
| 6 | 50C 10s -> 70C 300s | 100C 30s -> 100C 30s |
| 7 | 50C 30s -> 65C 1800s | 100C 100s -> 100C 10s |
| 8 | 50C 10s -> 70C 600s | 100C 10s -> 100C 100s |
| 9 | 50C 10s -> 70C 100s | 100C 60s -> 100C 30s |
| 10 | 50C 30s -> 65C 1200s | 100C 30s -> 100C 60s |

## 6. 判断

从结果看，当前目标集确实存在冗余：多个 DSC 峰形指标共同描述同一件事，
会让 EIG 在“峰面积、峰高、recovery index”之间重复计分。因此当前 EIG
更容易偏向能改变整体峰形幅度的极端路径点。

论文式目标集的优点是维度更低，EIG 更集中在动力学参数坐标上；
缺点是 `S* / H*` 在当前 PS 数据中是反推代理量，对 `Tp` 和总退火时间非常敏感。
因此它改善了“信息表达的紧凑性”，但不一定会立刻改善所有预测误差。

我的建议是下一版模型采用折中目标：

```text
delta_h_total_J_g
peak_area_J_g
paper_s_star_eff_J_mol_K
paper_h_star_eff_kJ_mol
path_dependence_index
```

并把 `Tp / peak_height / onset / recovery_index` 降级为诊断输出或派生指标。
这样既保留论文的动力学解释，又避免完全依赖由 Tp 反推出来的代理变量。

## 7. 输出文件

- 指标摘要：`results/ps/paper_style_comparison/summary.json`
- 论文式目标模型：`results/ps/paper_style_comparison/paper_style_model.json`
- 论文式目标测试预测：`results/ps/paper_style_comparison/paper_style_test_predictions.csv`
- 当前目标 EIG 排名：`results/ps/paper_style_comparison/current_target_eig_ranking_top100.csv`
- 论文式目标 EIG 排名：`results/ps/paper_style_comparison/paper_style_eig_ranking_top100.csv`
- 图表目录：`results/ps/paper_style_comparison/figures/`
