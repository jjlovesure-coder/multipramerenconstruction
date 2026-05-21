# 逆向重构精度改进方案

## 现状诊断

### 正向预测性能（已大幅改进）

| 模型 | normalized MAE (core targets) |
|------|------|
| raw_kernel | 0.791 |
| physics_kernel (当前最优) | 0.762 |
| physics_process_auxiliary | 0.751 |
| calibrated TNM forward | — |

正向预测已可用作实验筛选，physics-informed 特征的增益虽然绝对幅度不大（-3.8%），但在小样本下有实际意义。

### 逆向重构性能（仍然很差）

| 指标 | kernel 逆搜索 | TNM 逆搜索 |
|------|---------------|------------|
| Top-1 near-match | **0%** (全温区) | — |
| Top-5 near-match | **0%** (全温区) | — |
| Top-10 near-match | 7–14% | — |
| T1 MAE | 8–23°C | 14.8°C |
| T2 MAE | 11–15°C | 16.1°C |
| log10(t1) MAE | — | 0.78 (≈6x 因子) |
| log10(t2) MAE | — | 0.76 (≈6x 因子) |

near-match 定义（已非常宽松）：T1/T2 误差 ≤ 5°C 且每个时间误差在 3 倍以内。即使如此宽松的标准，Top-1 和 Top-5 命中率仍为 0%。

---

## 根本原因分析

### 1. 时温叠加导致的多对一映射（物理本质）

玻璃态弛豫中，相似的结构状态可以通过不同参数组合达到：

**低温 + 长时间 ≈ 高温 + 短时间**

正问题 (T₁, t₁, T₂, t₂) → 观测响应 是多对一的，逆向天然欠定。正向模型恰当地学习到了这种多对一映射，但逆向网格搜索在近似平坦的损失地形上挑"最优"，本质上是在随机选择。

### 2. 损失地形的平坦峡谷

评估数据表明，Top-1 和 Top-50 候选的损失值差异极小（很多 < 1%），意味着大量参数组合给出几乎相同的目标预测值。网格搜索在 5°C 步长和 log 尺度时间步长下，对平坦区域分辨率不足。

### 3. 当前代码的技术缺漏

| 文件 | 行号 | 问题 |
|------|------|------|
| `src/ps/inverse_design.py:85-98` | `_target_loss` 仅加权目标匹配，不考虑参数可辨识性 |
| `src/ps/inverse_design.py:54-82` | `candidate_rows` 候选网格固定 5°C 步长，精度不够 |
| `src/ps/inverse_design.py:172-224` | `search` 只返回单点 Top-K，不提供后验分布 |
| `src/ps/evaluate_inverse_by_temperature.py:96-116` | `build_candidate_predictions` 使用了不现实的 0.1s/0.5s 候选时间 |
| `src/ps/eig_design.py:159-206` | `inverse_identifiability_scores` 只在 EIG 中用，逆搜索没用到 |
| `src/ps/eig_design.py:251-267` | `extrapolation_penalty` 只在 EIG 中用，逆搜索没用到 |
| `src/ps/tnm_inverse_reconstruction.py:56-60` | TNM 逆搜索 `_target_loss` 同样缺少正则化 |
| `src/ps/inverse_design.py:126-150` | `refine_candidate` 的 Nelder-Mead 优化对平坦地形无效 |

---

## 改进方案

### 方案 1：两阶段搜索 —— 先锁 T₂ 再精搜其余参数

**原理**：`peak_temperature_Tp_C` 对 T₂ 有最强的物理约束，应优先利用。

**改动文件**：`src/ps/inverse_design.py`

**伪代码**：
```python
# Stage 1: 用 peak_temperature 和 delta_h 锁 T₂
t2_candidates = [t for t in temps if abs(t - t2_from_tp_estimate) <= 8.0]
# Stage 2: 固定 T₂ 候选，精细搜索 T₁, t₁, t₂
for t2 in t2_candidates:
    for t1 in temps:
        for logt1 in refined_log_times:
            for logt2 in refined_log_times:
                ...
```

**预期收益**：将 T₂ MAE 从 ~15°C 降至 3–5°C，并为后续时间搜索提供更强的约束。

---

### 方案 2：引入参数可辨识性正则化

**原理**：`eig_design.py:159` 已实现的 `inverse_identifiability_scores()` 计算了每个候选点的 Jacobian Gram 行列式对数。高分区域意味着 T₁、log(t₁)、T₂、log(t₂) 的小变化会在目标空间产生可区分的位移，即参数在此处"可辨识"。将这个评分作为逆搜索损失的一部分，排除不可辨识的候选。

**改动文件**：`src/ps/inverse_design.py`

**伪代码**：
```python
# 将 eig_design.py 中的 inverse_identifiability_scores 逻辑移入
# 或直接导入

def _target_loss_with_identifiability(pred, target, identifiability_score):
    target_loss = ...
    ident_penalty = 1.0 - identifiability_score  # 高分 = 低惩罚
    return target_loss + 0.3 * ident_penalty
```

**预期收益**：减少在参数不可辨识区域中的"随机选择"，提高 Top-5 命中率。

---

### 方案 3：外推惩罚 —— 约束逆搜索不偏离训练支持域

**原理**：候选网格覆盖的范围远超实验数据（目前只有 57 个两步实验分布在少数 T₁ bin 中）。在远离训练数据支持域的区域，正向模型预测不可靠，远距离外推得到的"最优"候选不应被采纳。

**改动文件**：`src/ps/inverse_design.py`

**伪代码**：
```python
# 复用 eig_design.py 中的 extrapolation_penalty 和 support_from_existing_rows
from src.ps.eig_design import (extrapolation_penalty,
                                support_from_existing_rows)

support = support_from_existing_rows(read_existing_rows_from_dataset())
extrap = extrapolation_penalty(candidates, support)
losses += 0.4 * extrap  # 外推惩罚
```

**预期收益**：阻止逆搜索在训练数据覆盖不到的区域做无意义外推。

---

### 方案 4：返回后验分布而非点估计

**原理**：既然损失地形是平坦的峡谷，最优解和次优解在物理上等价，就应该诚实地返回所有合理候选的分布，而不是假装有一个唯一解。

**改动文件**：`src/ps/inverse_design.py:search()`

**伪代码**：
```python
# 不只取 argsort[:top_k]
threshold = np.percentile(losses, 5)  # 取最优 5% 损失内的候选
valid_mask = losses <= threshold
valid_indices = np.where(valid_mask)[0]

posterior = {
    "T1_C": {"median": np.median(t1_valid), "p5": ..., "p95": ...},
    "T2_C": {"median": np.median(t2_valid), "p5": ..., "p95": ...},
    "log_t1_s": {"median": ..., "p5": ..., "p95": ...},
    "log_t2_s": {"median": ..., "p5": ..., "p95": ...},
    "n_valid_candidates": len(valid_indices),
    "loss_range": [float(np.min(losses)), float(np.max(losses[valid_mask]))],
}
```

**预期收益**：用户可以看到哪些参数被约束了、哪些参数实际上无法从数据中确定。对实验设计有直接指导意义。

---

### 方案 5：自适应网格细化

**原理**：固定 5°C 网格限制了解析精度。在粗搜索的 Top-3 候选周围做局部细搜索。

**改动文件**：`src/ps/inverse_design.py:search()`

**伪代码**：
```python
# Stage 1: 粗网格（5°C × coarse times）→ Top-3
# Stage 2: 每个 Top-3 周围细化
for best in top3_coarse:
    refined_temps = np.linspace(best_T1 - 3, best_T1 + 3, 7)  # 1°C spacing
    refined_logt1 = np.linspace(best_logt1 - 0.3, best_logt1 + 0.3, 10)
    ...
```

**预期收益**：温度精度从 5°C 提升到 ~1°C。

---

### 方案 6：TNM 逆重构增强

针对 `src/ps/tnm_inverse_reconstruction.py`：

1. **加入 TNM 状态一致性约束**：根据重构参数计算的 Tf（fictive temperature）轨迹应与观测的 ΔH 值在物理上一致：
   ```python
   # 额外约束：|pred_delta_h - target_delta_h| / target_delta_h < 容差
   ```

2. **引入模型选择惩罚**：当多个候选给出相同的损失时，偏好更简单的参数（如更短的实验时间、更低的温度）

3. **TNM 参数的后验抽样**：当前 TNM 参数选优是通过网格搜索 12 个参数组合。可以扩展到对参数做 Gibbs 抽样，得到后验参数分布，再对每组参数分别做逆重构。

---

## 实施优先级

| 优先级 | 方案 | 预期收益 | 实现难度 | 依赖 |
|--------|------|----------|----------|------|
| **P0** | 方案 1: 两阶段搜索 | 大幅降低 T₂ MAE | 低 | 无 |
| **P0** | 方案 4: 后验分布 | 诚实反映不确定性 | 低 | 无 |
| **P1** | 方案 3: 外推惩罚 | 防止无意义外推 | 低 | 复用 eig_design.py |
| **P1** | 方案 2: 可辨识性正则化 | 排除不可辨识区域 | 中 | 复用 eig_design.py |
| **P2** | 方案 5: 自适应网格细化 | 精度从 5°C → 1°C | 中 | 无 |
| **P2** | 方案 6: TNM 逆重构增强 | TNM 路径的额外提升 | 中 | 方案 3/4 |

---

## 实现检查清单

- [ ] `src/ps/inverse_design.py`: 实现两阶段 T₂-first 搜索
- [ ] `src/ps/inverse_design.py`: `search()` 返回后验分布
- [ ] `src/ps/inverse_design.py`: 加入外推惩罚
- [ ] `src/ps/inverse_design.py`: 加入可辨识性正则化
- [ ] `src/ps/inverse_design.py`: 实现自适应局部细化
- [ ] `src/ps/tnm_inverse_reconstruction.py`: 加入状态一致性约束
- [ ] `src/ps/tnm_inverse_reconstruction.py`: 返回后验分布
- [ ] `src/ps/evaluate_inverse_by_temperature.py`: 修复不现实的候选时间 (0.1s/0.5s)
- [ ] `src/ps/evaluate_inverse_by_temperature.py`: 更新评估指标反映后验分布质量
- [ ] 重新运行逆重构评估，对比改进前后的指标
