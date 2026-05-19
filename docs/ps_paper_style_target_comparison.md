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

数据集样本数：`102`；训练集：`74`；
测试集：`28`。两组模型使用相同输入特征、相同 grouped split、
相同 RBF kernel bandwidth。

| 目标集 | 目标 | MAE | RMSE | R2 | MAE/测试std |
| --- | --- | --- | --- | --- | --- |
| 当前PS目标 | delta_h_total_J_g | 0.1812 | 0.2093 | 0.240 | 0.755 |
| 当前PS目标 | peak_area_J_g | 0.1455 | 0.1657 | 0.209 | 0.781 |
| 当前PS目标 | peak_temperature_Tp_C | 0.3458 | 0.3933 | 0.276 | 0.748 |
| 当前PS目标 | peak_height_uW | 24.26 | 30.2 | -0.068 | 0.830 |
| 当前PS目标 | onset_temperature_C | 0.004495 | 0.005426 | -0.335 | 0.957 |
| 当前PS目标 | recovery_index | 0.1574 | 0.1733 | 0.180 | 0.823 |
| 当前PS目标 | path_dependence_index | 0.1944 | 0.2335 | -0.018 | 0.840 |
| 当前PS目标 | kovacs_peak_label | 0 | 0 | nan | nan |
| 论文式目标 | paper_delta_h_J_g | 0.1812 | 0.2093 | 0.240 | 0.755 |
| 论文式目标 | paper_delta_h_peak_J_g | 0.1455 | 0.1657 | 0.209 | 0.781 |
| 论文式目标 | paper_s_star_eff_J_mol_K | 157.1 | 203.8 | 0.234 | 0.675 |
| 论文式目标 | paper_h_star_eff_kJ_mol | 59.46 | 77.14 | 0.234 | 0.675 |

归一化综合误差：

```json
{
  "current_targets": {
    "mean_mae_over_test_std": 0.8192043303994533,
    "median_mae_over_test_std": 0.8227260370353497
  },
  "paper_style_targets": {
    "mean_mae_over_test_std": 0.7213503025767358,
    "median_mae_over_test_std": 0.7149048634755434
  }
}
```

## 4. EIG 分布差异

| 目标集 | EIG均值 | P90 | P99 | 最大值 |
| --- | --- | --- | --- | --- |
| 当前PS目标 | 1.337 | 1.604 | 1.691 | 1.718 |
| 论文式目标 | 1.366 | 1.672 | 2.122 | 2.539 |

![EIG distribution](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\eig_distribution_current_vs_paper_style.png)

目标间相关性摘要：

```json
{
  "current_targets": {
    "mean_abs_offdiag_corr": 0.47256635706932343,
    "median_abs_offdiag_corr": 0.605819593539345,
    "max_abs_offdiag_corr": 0.9522960923088609
  },
  "paper_style_targets": {
    "mean_abs_offdiag_corr": 0.5032176850259175,
    "median_abs_offdiag_corr": 0.28234105253504893,
    "max_abs_offdiag_corr": 0.9999990780427025
  }
}
```

![Current target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\current_target_correlation.png)

![Paper-style target correlation](E:\knowledgepaper\multipramereconstruction\results\ps\paper_style_comparison\figures\paper_style_target_correlation.png)

## 5. Top EIG 工况差异

| Rank | 当前PS目标Top工况 | 论文式目标Top工况 |
| --- | --- | --- |
| 1 | 100C 10s -> 100C 1800s | 100C 10s -> 100C 10s |
| 2 | 100C 10s -> 100C 1200s | 100C 30s -> 100C 10s |
| 3 | 100C 10s -> 100C 900s | 100C 10s -> 100C 30s |
| 4 | 100C 10s -> 100C 600s | 100C 60s -> 100C 10s |
| 5 | 95C 10s -> 95C 1800s | 100C 10s -> 100C 60s |
| 6 | 95C 10s -> 95C 1200s | 100C 30s -> 100C 30s |
| 7 | 100C 10s -> 100C 300s | 100C 100s -> 100C 10s |
| 8 | 100C 30s -> 100C 1800s | 100C 10s -> 100C 100s |
| 9 | 95C 10s -> 95C 900s | 100C 60s -> 100C 30s |
| 10 | 100C 30s -> 100C 1200s | 100C 30s -> 100C 60s |

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
