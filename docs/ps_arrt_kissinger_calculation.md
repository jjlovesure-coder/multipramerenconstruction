# PS ARRT/Kissinger 激活熵与激活焓计算诊断

## 1. 结论

可以按论文附录公式 5-9 先计算 `S*` 与 `H*`，但前提是：同一个退火态必须有多个升温速率下的 relaxation peak position `Tp`。

当前 PS 数据扫描了 `151` 条 final heating 曲线，但没有任何一个退火态同时具备至少 3 个不同升温速率。因此严格的 ARRT/Kissinger 计算目前不能生成可训练的 `S* / H*` 标签。

这和上一版“由退火时间 + Tp 反推有效代理量”不同。上一版能给每条样本赋值，但属于模型代理；本诊断要求公式 5-9 的实验输入完整，因此宁可输出不可计算，也不把代理量伪装成实验量。

## 2. 使用公式

采用 Kissinger 关系：

```text
ln(beta / Tp^2) = C - E / (R Tp)
```

由斜率得到：

```text
E = -R * slope
```

附录公式 9 给出：

```text
E = H* + R Tp
H* = E - R Tp
```

再由绝对反应速率峰位近似：

```text
ln(beta / Tp^2) = -H* / (R Tp) + ln(kB R / (h H*)) + S* / R
```

求得：

```text
S* = R * [ln(beta / Tp^2) + H*/(R Tp) - ln(kB R / (h H*))]
```

## 3. 当前数据可计算性

- 退火态分组数：`90`
- 可严格计算的分组数：`11`
- 最大升温速率种类数：`5`

## 4. 下一步实验建议

如果希望像论文一样先计算 `S* / H*` 再训练，建议为少量代表退火态增加多升温速率 DSC：

```text
beta = 5, 10, 20, 40 °C/min
```

每个退火态至少 3 个速率，最好 4 个速率以检查 Kissinger 线性。优先选择：

```text
single: 70 °C 300 s, 90 °C 100 s, 95 °C 100 s
two-step: 50 °C 10 s -> 80 °C 1800 s
two-step: 50 °C 10 s -> 100 °C 1800 s
two-step: 65 °C 10 s -> 100 °C 1200 s
```

这些点覆盖已有模型误差较高或 EIG 较高的区域。完成后，本脚本会自动输出可训练的 `activation_entropy_S_star_j_mol_K` 和 `activation_enthalpy_H_star_kj_mol`。

## 5. 输出

- 全部升温扫描特征：`results/ps/arrt_kissinger/all_recovery_scan_features.csv`
- 可计算性诊断：`results/ps/arrt_kissinger/condition_rate_diagnostics.csv`
- 严格公式计算结果：`results/ps/arrt_kissinger/arrt_kissinger_results.csv`
- 摘要：`results/ps/arrt_kissinger/summary.json`
