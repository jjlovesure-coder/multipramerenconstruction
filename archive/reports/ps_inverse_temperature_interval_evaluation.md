# PS 反向预测四个退火参数的温度区间训练效果评估

## 1. 评估目的

本报告评估当前模型在反向预测四个双步退火参数时的可靠性：

```text
T1, t1, T2, t2
```

评估问题是：给定实验得到的 DSC 特征，包括

```text
delta_h_total_J_g
peak_area_J_g
peak_temperature_Tp_C
recovery_index
path_dependence_index
```

模型能否反推出原始双步退火工况。

需要特别说明：正向预测准确不等于反向预测可靠。正向问题是：

```text
T1, t1, T2, t2 -> DSC 特征
```

反向问题是：

```text
DSC 特征 -> T1, t1, T2, t2
```

后者更难，因为不同退火路径可能产生相似的 DSC 曲线特征，即存在多解性和不可辨识性。

## 2. 评估方法

只评估双步退火样本，因为单步实验没有完整的 `T2,t2`。

当前双步样本数：

```text
49
```

反向搜索候选空间包括：

- `T1 = 50-100 deg C`
- `T2 = 50-100 deg C`
- 同时允许升温路径、降温路径和等温路径
- 时间候选覆盖 `0.1-1800 s`

反向预测使用当前正向 surrogate model，对所有候选工况预测 DSC 特征，再选择与目标实验特征最接近的候选工况。

损失函数使用以下目标：

| 目标 | 作用 |
|---|---|
| `delta_h_total_J_g` | 总焓回复比焓 |
| `peak_area_J_g` | relaxation peak 面积 |
| `peak_temperature_Tp_C` | 峰位 |
| `recovery_index` | 回复程度 |
| `path_dependence_index` | 路径依赖强度 |

近似命中定义为：

```text
T1 和 T2 均在真实值 ±5 deg C 内，
且 t1 和 t2 均在真实值 3 倍以内。
```

该标准比完全相等宽松，因为反向问题本身存在多解性。

## 3. 数据覆盖度

当前双步样本在 T1 温度区间上的覆盖如下：

| T1 区间 | 双步样本数 | 已覆盖 T1 | 已覆盖 T2 | 路径数 |
|---|---:|---|---|---:|
| 50-60 deg C | 7 | 50, 55 | 70, 80, 85, 90, 100 | 5 |
| 65-75 deg C | 2 | 65, 75 | 100 | 2 |
| 80-90 deg C | 40 | 80, 90 | 80, 90 | 2 |

覆盖度结论：

- `80-90 deg C` 样本最多，但路径类型很集中，主要是 `80->90` 和 `90->80`。
- `50-60 deg C` 样本较少，但 T2 覆盖较宽。
- `65-75 deg C` 样本最少，目前只有 2 个双步样本，统计结论不稳定。
- `95-100 deg C` 作为 T1 的双步数据基本不足。

## 4. 按 T1 区间的反向预测效果

| T1 区间 | 样本数 | T1 MAE | T2 MAE | t1 中位倍数误差 | t2 中位倍数误差 | Top-1 命中率 | Top-5 命中率 | Top-10 命中率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50-60 deg C | 7 | 6.43 deg C | 17.86 deg C | 60.0x | 120.0x | 0.0 | 0.0 | 0.143 |
| 65-75 deg C | 2 | 12.50 deg C | 0.00 deg C | 100.0x | 1.25x | 0.0 | 0.0 | 0.0 |
| 80-90 deg C | 40 | 18.75 deg C | 8.50 deg C | 4.30x | 14.00x | 0.025 | 0.075 | 0.100 |

解释：

- `80-90 deg C` 虽然训练样本最多，但反向恢复 `T1` 和时间参数仍然较差。这说明大量相似路径数据提高了正向拟合，却没有充分解决反向唯一性。
- `50-60 deg C` 的 `T1` 反推相对更接近，但 `T2` 和时间误差很大，说明低温起始路径的第二步信息仍不够可辨识。
- `65-75 deg C` 的 `T2` 看起来很好，但样本只有 2 个，不能说明训练充分；同时 `t1` 误差极大，说明这一区间目前不能可靠反推四参数。

## 5. 按 T2 区间的反向预测效果

| T2 区间 | 样本数 | T1 MAE | T2 MAE | t1 中位倍数误差 | t2 中位倍数误差 | Top-5 命中率 |
|---|---:|---:|---:|---:|---:|---:|
| T2 50-70 | 1 | 10.00 deg C | 20.00 deg C | 60.0x | 1.0x | 0.0 |
| T2 75-90 | 43 | 17.79 deg C | 8.37 deg C | 5.0x | 18.0x | 0.070 |
| T2 95-100 | 5 | 9.00 deg C | 17.00 deg C | 90.0x | 1.5x | 0.0 |

解释：

- `T2=75-90 deg C` 数据最多，反向恢复 `T2` 比恢复 `T1` 稍好，但时间仍然高度不稳定。
- `T2=95-100 deg C` 的时间 `t2` 相对较容易，但 `T1` 和 `T2` 本身仍混淆明显。
- `T2=50-70 deg C` 只有 1 个样本，不能用于判断训练充分。

## 6. 按路径类型的反向预测效果

| 路径类型 | 样本数 | T1 MAE | T2 MAE | t1 中位倍数误差 | t2 中位倍数误差 | Top-5 命中率 |
|---|---:|---:|---:|---:|---:|---:|
| 降温路径 T2<T1 | 20 | 25.25 deg C | 9.25 deg C | 3.60x | 36.80x | 0.0 |
| 升温路径 T2>T1 | 29 | 10.86 deg C | 9.66 deg C | 50.0x | 10.0x | 0.103 |

解释：

- 升温路径比降温路径更容易反向预测 `T1`，但时间参数仍然非常不稳定。
- 降温路径的 `T1` 误差很大，说明 `90->80 deg C` 这类路径虽然样本多，但反向时容易被模型映射到其他温度组合。
- 这说明当前特征集合对“路径方向”和“时间分配”的辨识能力不够。

## 7. 图片结果

### 7.1 T1 反向误差

![Inverse T1 error](E:/knowledgepaper/multipramereconstruction/results/ps/inverse_temperature/figures/inverse_T1_error_by_T1_bin.png)

### 7.2 T2 反向误差

![Inverse T2 error](E:/knowledgepaper/multipramereconstruction/results/ps/inverse_temperature/figures/inverse_T2_error_by_T1_bin.png)

### 7.3 Top-5 命中率

![Top5 by T1](E:/knowledgepaper/multipramereconstruction/results/ps/inverse_temperature/figures/inverse_top5_rate_by_T1_bin.png)

![Top5 by T2](E:/knowledgepaper/multipramereconstruction/results/ps/inverse_temperature/figures/inverse_top5_rate_by_T2_bin.png)

### 7.4 T1 实测-反推散点图

![T1 actual vs predicted](E:/knowledgepaper/multipramereconstruction/results/ps/inverse_temperature/figures/inverse_T1_actual_vs_pred.png)

## 8. 哪些区间训练相对充分

如果只看正向预测，`80-90 deg C` 是目前训练最充分的区间，因为样本数量最多，正向预测焓变、峰面积、Tp 的稳定性最好。

但如果目标是反向预测四个退火参数，则结论不同：

### 相对最好：80-90 deg C，但仍不达可靠反推标准

优点：

- 样本数最多，40 个双步样本。
- T2 误差相对低于部分其他区间。
- Top-10 近似命中率约 0.10，是目前几个区间里相对较高的。

问题：

- T1 MAE 仍高达 18.75 deg C。
- 时间倍数误差很大，尤其 t2 中位误差约 14 倍。
- 说明模型能学到该区间的平均 DSC 响应，但难以区分具体路径参数。

结论：`80-90 deg C` 可以认为正向训练较充分，但反向四参数预测只具备初步参考价值，不能可靠使用。

### 有潜力但样本不足：50-60 deg C

优点：

- 覆盖的 T2 较广，包括 70, 80, 85, 90, 100 deg C。
- T1 MAE 为 6.43 deg C，相比其他区间较低。

问题：

- 样本数只有 7。
- T2 MAE 高达 17.86 deg C。
- 时间参数误差极大，说明不同时间路径仍高度混淆。

结论：`50-60 deg C` 是最值得继续补数据的区间。它现在还不充分，但最有希望通过补点改善反向预测。

### 目前不足：65-75 deg C

优点：

- T2 MAE 在现有统计中为 0，但这是样本过少造成的表面结果。

问题：

- 样本数只有 2。
- Top-5 和 Top-10 命中率都是 0。
- t1 误差极大。

结论：不能认为该区间训练充分。它是下一轮必须补的关键区间之一。

### 明显不足：95-100 deg C 作为 T1 的双步路径

当前几乎没有足够的双步样本支持 `95-100 deg C` 作为第一步温度的反向预测。已有高温信息更多来自单步实验和 T2 高温点，因此只能帮助正向估计高温回复，不能支撑四参数反演。

## 9. 总体判断

当前模型已经可以做有限程度的正向预测，但还不能稳定反向预测四个退火参数。

按“反向四参数预测可用性”排序：

```text
80-90 deg C：数据最多，反向有初步参考价值，但仍不可靠
50-60 deg C：T1 有一定可辨识性，但 T2 和时间不足
65-75 deg C：样本太少，暂不可判断
95-100 deg C：作为 T1 的双步数据不足
```

更直接地说：

```text
目前没有任何温度区间已经达到“可以较可靠反推出 T1,t1,T2,t2”的程度。
```

其中最接近可用的是 `80-90 deg C`，但也只能作为候选建议，不适合作为唯一结论。

## 10. 后续实验建议

为了让反向预测真正变好，下一步实验不应只追求正向 MAE，而要提高参数可辨识性。

建议优先补：

1. `50-60 deg C` 起始路径：固定 T1，系统改变 T2 和 t2。
2. `65-75 deg C` 起始路径：补足当前样本空白。
3. 等温双步，如 `65->65`、`80->80`、`100->100`，用于分离路径效应和总时间效应。
4. 同一温度对下做时间梯度，例如固定 `50->70`，测试 `t2=60,300,1200,1800 s`。
5. 对已有 `80->90` 与 `90->80` 数据增加少量重复和中间时间点，帮助模型区分方向性。

当前最值得做的反向增强实验，不是单纯 EIG 最高的点，而是能形成局部网格的点。例如：

```text
50->65: t1=10/100 s, t2=300/1200 s
65->65: t1=10/300 s, t2=300/1800 s
65->85: t1=10/100 s, t2=300/600 s
75->100: t1=10/100 s, t2=10/1200 s
```

这些点能让模型学会“相同输出特征背后到底是温度变化造成的，还是时间变化造成的”。

## 11. 输出文件

本次评估生成：

```text
results/ps/inverse_temperature/inverse_reconstruction_by_sample.csv
results/ps/inverse_temperature/inverse_summary_by_T1_bin.csv
results/ps/inverse_temperature/inverse_summary_by_T2_bin.csv
results/ps/inverse_temperature/inverse_summary_by_path_class.csv
results/ps/inverse_temperature/two_step_coverage_by_T1_bin.csv
results/ps/inverse_temperature/inverse_temperature_summary.json
results/ps/inverse_temperature/figures/
```
