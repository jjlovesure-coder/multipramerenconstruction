# 代码审查报告：正向预测与逆向重构误差差距分析

## 一、当前状态量化

### 正向预测（可接受）

| 模型 | 核心目标归一化 MAE | 备注 |
|------|-------------------|------|
| 原始 RBF 核 | 0.79–1.02 | recovery_index 的 R² 为负 |
| 物理信息核（当前部署） | 0.70–0.76 | `results/ps/models/ps_physics_informed_kernel_model.json` |
| TNM 校准模型（最新） | **0.67** | 首次所有 4 个核心目标 R² 全部为正 |

### 逆向重构（不可接受）

near-match 定义：T₁/T₂ 误差 ≤ 5°C 且每个时间误差在 3 倍以内（已非常宽松）

| 指标 | 50-60°C | 65-75°C | 80-90°C |
|------|---------|---------|---------|
| T1 MAE (°C) | 10.9 | 15.0 | **18.5** |
| T2 MAE (°C) | 15.5 | 10.0 | 11.3 |
| t1 中位倍数误差 | **50x** | **50x** | 2x |
| t2 中位倍数误差 | 6x | 3x | **10x** |
| Top-1 near-match | 9% | **0%** | 0% |
| Top-5 near-match | 9% | **0%** | 5% |
| Top-10 near-match | 18% | **0%** | 5% |
| 后验 T2 覆盖率 | 27% | 60% | **2%** |
| 后验 T1 宽度中位数 | 30°C | 20°C | 30°C |

---

## 二、根本原因诊断（按致命程度排序）

### 根因 1：TNM dose 特征在实验时间窗口内饱和 → 时间参数不可辨识

**代码位置**: `src/ps/physics_features.py:73-81` 的 `tnm_dose()`

**机制**：

```
tau = relaxation_tau_s(T, E)  # 在 Tg 附近 τ 很小
dose = 1 - exp(-(t/τ)^β)      # t ≥ 60s 时 dose → 1，饱和！
```

在 PS 的 Tg（~373K / 100°C）附近，弛豫时间 τ 远小于实验时间窗口（10-1800s），导致几乎所有 t ≥ 60s 的退火都产生 dose ≈ 1。**不同 t₁/t₂ 组合在特征空间中坍缩为同一点**，模型完全无法区分 10 秒和 500 秒的退火。这直接解释了为什么 t₁ 的中位倍数误差高达 50 倍。

**影响链**：

```
TNM dose 饱和
  → 特征空间对 t₁/t₂ 丧失分辨率
    → 正向模型学到 dose≈1 → ΔH≈constant 的映射
      → 逆搜索时不同 t₁/t₂ 的候选给出几乎相同的目标预测
        → 损失地形平坦 → 网格搜索随机选择 → t₁/t₂ 误差巨大
```

### 根因 2：时温叠加导致的多对一映射（玻璃态物理本质）

低温 + 长时间 ≈ 高温 + 短时间。正问题 (T₁, t₁, T₂, t₂) → 观测响应天然是多对一的。

评估数据证实了这一点：Top-1 和 Top-50 候选的损失值差异极小（`loss_flatness_ratio` 中位值 0.67–1.37）。这意味着大量参数组合在物理上是等价的，逆向不存在唯一解。

这是**物理本质限制**，不是代码 bug。但可以通过以下方式缓解：
- 在后验分布中诚实反映这种多解性
- 用参数可辨识性分数排除不可辨识区域
- 加入额外约束（如严格 ARRT/Kissinger 的 S*/H*）打破简并

### 根因 3：训练数据覆盖极度不均

| T1 区间 | 双步样本数 | 占比 |
|---------|-----------|------|
| 50-60°C | 11 | 19% |
| 65-75°C | 5 | 9% |
| 80-90°C | 41 | **72%** |
| 95-100°C | 0 | 0% |

82% 的双步数据集中在 80-90°C 区间，且该区间内只有两种路径（80→90 up-jump 和 90→80 down-jump）。模型完全未见过其他温度区间的行为。

**后果链**：

```
数据覆盖不均
  → 模型学到的是"80-90°C 区域的平均 DSC 响应"
    → grouped split 下测试组 (source_file, mode, T1) 可能与训练组完全不同
      → recovery_index 的 R² 为负（预测不如用训练集均值）
        → 逆搜索在其他温度区间的外推完全不可靠
```

### 根因 4：测试集极小（n=6），所有指标统计不可靠

**代码位置**: `src/ps/train_ps_model.py:104-119` 的 `grouped_split()`

在 99/6 的 train/test 划分下，任何模型对比的 MAE 差异在统计上都可能不显著。不同文档报告了不同的数字（如 raw kernel 的 core normalized 在三个文档中分别是 0.791、1.020、0.791），因为它们可能来自不同的数据状态或 split 种子。

**更严重的是**：`src/ps/tnm_calibrated_model.py:190-191` 的 CV 折叠使用了 `np.array_split`（按位置连续切分），**不尊重分组结构**。同一 `(source_file, mode, T1_C)` 组的样本可能同时出现在训练和验证折叠中，造成信息泄露。

### 根因 5：逆搜索损失函数不包含可辨识性正则化和外推惩罚

**代码位置**: `src/ps/inverse_design.py:106-137`

`combined_inverse_loss()` 仅加权以下项：
- 目标匹配损失
- 时间惩罚（倾向短实验）
- 不确定性惩罚
- 外推惩罚（但从 `eig_design.py` 导入时，若导入失败则 fallback 为全零向量）
- 可辨识性资金（同上，fallback 为全零向量）

而 `eig_design.py` 已完整实现了：
- `inverse_identifiability_scores()`（Jacobi Gram 行列式对数，`eig_design.py:159-206`）
- `extrapolation_penalty()`（候选偏离训练支持域的程度，`eig_design.py:251-267`）

这两个组件在 EIG 实验设计中**正在使用**（权重分别为 1.20 和 0.35），但在逆搜索中**完全未生效**。

### 根因 6：TNM 校准模型中有效参数与本征参数分离

| 参数 | hs-01 Kissinger 测量 | CV 选出 | 差异 |
|------|---------------------|---------|------|
| E (kJ/mol) | 427, 455, 541 | **300** | 低 30-45% |
| β | — | **0.75** | — |
| τ_ref (s) | — | **100** | — |

CV 选择的不是物理上最正确的 TNM 参数，而是**使特征分辨率最大化的参数**：
- 较低的 E（300 vs 427）→ τ 对温度不那么敏感 → dose 在 50-100°C 范围内有更好渐变
- 较高的 β（0.75）→ KWW 曲线更陡 → 在部分弛豫区（dose 0.1-0.9）分辨率更高

这对**正向预测**是有利的（提升特征区分度），但对**逆向重构**可能有害——因为 TNM 参数不再忠实反映真实物理，逆推的参数组合可能违背物理约束。

---

## 三、代码优化方案

### P0：必须立即修复（否则所有逆重构指标不可信）

#### P0-1：修复 TNM dose 的时间不可辨识

**原理**：`log(t₁)`, `log(t₂)`, `t₁/(t₁+t₂)`, `log(t₁/t₂)` 不会随 dose 饱和，始终保留时间分辨率。

**改动文件**：`src/ps/tnm_calibrated_model.py` 的 `tnm_state_features()`

当前 TNM 校准模型的 12 维特征中已包含 `tnm_log_total_time` 和 `tnm_log_t_ratio`，但缺少独立的一阶段对数时间特征。需确保以下特征存在于特征矩阵中：

```python
"tnm_log_t1_s": math.log10(max(t1, 1e-9)),
"tnm_log_t2_s": math.log10(max(t2, 1e-9)) if mode == "two_step" else 0.0,
"tnm_t1_fraction_of_total": t1 / max(total, 1e-9),
```

同时，考虑将 `T_REF_K` 从 373.15 调低到 323.15（50°C），使 τ_ref 更大，让 dose 在 10-1800s 范围内保持分辨率。

**代码对比**：

```python
# 当前：dose 在实验条件下饱和
T_REF_K = 373.15  # 100°C, τ_ref 在此温度下很小
TAU_REF_S = 100.0

# 改进：将参考点设在低温端，使 dose 保持分辨率
T_REF_K = 323.15  # 50°C, τ_ref 在此温度下较大
# 或使用 log-t 特征绕过 dose 饱和问题
```

**预期收益**：将时间倍数误差从 50x 降至 3-5x。

---

#### P0-2：修复 CV 信息泄露

**代码位置**：`src/ps/tnm_calibrated_model.py:190-191`

```python
# 当前（有泄露：按位置连续切分，不尊重分组结构）
folds = np.array_split(train_idx, min(5, max(2, len(train_idx) // 8)))

# 修复：复用已有的分组折叠函数
from src.ps.train_physics_informed_ps_model import group_folds

# 替换为
folds = group_folds(rows, train_idx, n_folds=5, seed=seed)
```

**预期收益**：CV 分数与测试分数不再有虚假差距（当前约 2.3 倍），确保选出的 TNM 参数真正泛化。

---

#### P0-3：逆搜索损失中加入可辨识性正则化和外推惩罚

**代码位置**：`src/ps/inverse_design.py:26-31` 和 `263-298`

修改 `INVERSE_LOSS_WEIGHTS`，并确保导入不会静默失败：

```python
INVERSE_LOSS_WEIGHTS = {
    "time": 0.05,
    "uncertainty": 0.02,
    "extrapolation": 0.35,    # 原来 0.40，但仅在 try 块成功时才生效
    "identifiability": 0.30,  # 原来 0.30，同上
}
```

同时，将 `search()` 中的 try/except 改为**硬依赖**：如果无法导入 `extrapolation_penalty` 和 `inverse_identifiability_scores`，应显式报错而非静默 fallback 为零向量。

**预期收益**：排除在参数不可辨识区域和训练支持域之外的"假最优"候选。

---

#### P0-4：修复测试集过小问题

**代码位置**：`src/ps/train_ps_model.py:104-119`

将单次 75/25 划分改为重复分组 CV（如 7 seeds × 5 folds）：

```python
# 当前
train_idx, test_idx = grouped_split(rows, test_fraction=0.25, seed=7)

# 改进后
all_metrics = []
for seed in (2, 3, 7, 13, 17, 23, 42):
    for fold_train, fold_val in group_folds(rows, n_folds=5, seed=seed):
        model = KernelRegressor.fit(x[fold_train], y[fold_train], bandwidth=1.2)
        pred, _ = model.predict(x[fold_val])
        all_metrics.append(metrics(y[fold_val], pred))

# 报告均值 ± 标准差
print(f"MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
```

`train_physics_informed_ps_model.py` 已有这个框架（`select_physics_kernel` 中 7 seeds × 5 folds），但基础 PS 线和 models/ 线没有。

**预期收益**：获得有统计意义的模型对比，消除"物理核比原始核好 3.8%"这种在 n=6 下无法成立的结论。

---

### P1：高优先级（核心功能改进）

#### P1-1：两阶段逆搜索 —— 先锁 T₂ 再精搜其余参数

**原理**：`peak_temperature_Tp_C` 对 T₂ 有最强物理约束。先用 Tp 锚定 T₂ 候选窗口，大幅缩小搜索空间后再搜 T₁, t₁, t₂。

**代码位置**：`src/ps/inverse_design.py:263-298` 的 `search()`

当前 `t2_candidates_from_target()` 已实现 Tp 约束（`inverse_design.py:64-72`），但仅用于**候选生成**阶段（减少候选数），而非**损失过滤**阶段。在 `evaluate_inverse_by_temperature.py:140-141` 中有正确的硬过滤：

```python
allowed_t2 = set(t2_candidates_from_target({"peak_temperature_Tp_C": ...}))
valid_mask = np.array([round(float(c["T2_C"])) in allowed_t2 for c in candidates], dtype=bool)
```

将此过滤逻辑迁移到 `search()` 中，在计算 `combined_inverse_loss` 之前应用。

**预期收益**：T₂ MAE 从 11-16°C 降至 3-5°C。

---

#### P1-2：逆搜索返回后验分布而非单点估计

**原理**：损失地形是平坦峡谷，Top-1 和 Top-50 物理上等价。应诚实返回所有合理候选的分布。

**代码位置**：`src/ps/inverse_design.py:300-385`

当前 `posterior_summary()` 已在 `evaluate_inverse_by_temperature.py` 中用于评估，但 `search()` 的最终输出仍是单点 Top-K。修改 `search()` 的输出结构：

```python
# 当前输出
{"target": target, "results": [top_k点估计...]}

# 改进后输出
{
    "target": target,
    "point_estimates": [top_k点估计...],  # 保留，供快速参考
    "posterior": {
        "T1_C": {"p5": ..., "median": ..., "p95": ..., "width": ...},
        "T2_C": {"p5": ..., "median": ..., "p95": ..., "width": ...},
        "log10_t1_s": {"p5": ..., "median": ..., "p95": ..., "width": ...},
        "log10_t2_s": {"p5": ..., "median": ..., "p95": ..., "width": ...},
        "n_valid_candidates": ...,  # 损失在前 5% 的候选数
        "loss_flatness_ratio": ...,  # 损失地形的平坦程度
    },
    "interpretation": "T1和T2的后验宽度在X°C以内，说明被约束；t1的后验宽度跨2个数量级，说明无法从数据中确定"
}
```

**预期收益**：用户可以看到哪些参数被约束了、哪些参数无法从数据中确定。对实验设计有直接指导意义。

---

#### P1-3：增大 Ridge 正则化，稳定 TNM 线性头系数

**代码位置**：`src/ps/tnm_calibrated_model.py` 中 Ridge 回归的 λ

当前 ridge λ = 1e-4 几乎为零，`ps_calibrated_tnm_model.json` 中系数幅度达到 10+：

```
tnm_total_state      →  coef = [-9.74, -11.38, 3.34, -10.84]
tnm_step2_increment  →  coef = [3.25, 4.01, -0.70, 3.67]
```

大系数互相抵消的格局在加入新数据后可能剧烈变化。

```python
# 将 ridge 从 1e-4 增大
ridge = 0.05

# 或从 12 个特征中移除完全共线的冗余特征：
# tnm_step2_increment = tnm_total_state - tnm_step1_state（完全共线）
```

---

#### P1-4：统一所有模型的评估基线

**问题**：三个文档报告了不同的 raw kernel 性能数字（0.791 vs 1.020 vs 0.791），无法直接对比"TNM 校准模型比物理核好多少"。

**方案**：在同一个脚本中，用完全相同的 split，依次评估 raw / physics / TNM calibrated 三个模型，输出统一对比表。可新增 `tools/compare_all_models.py`。

---

### P2：中优先级（精度提升，需要适度工作量）

#### P2-1：自适应网格细化

**代码位置**：`src/ps/inverse_design.py:263-298` 的 `search()`

在粗搜索（5°C 步长）的 Top-3 候选周围，以 1°C 步长和更密集的对数时间网格做局部二次搜索。

```python
# Stage 1: 粗网格 → Top-3
# Stage 2: 每个 Top-3 周围细化
for best in top3:
    refined_t1 = np.linspace(best_T1 - 3, best_T1 + 3, 7)   # 1°C spacing
    refined_t2 = np.linspace(best_T2 - 3, best_T2 + 3, 7)
    refined_logt1 = np.linspace(best_logt1 - 0.3, best_logt1 + 0.3, 10)
    refined_logt2 = np.linspace(best_logt2 - 0.3, best_logt2 + 0.3, 10)
```

**预期收益**：温度精度从 5°C 提升到 ~1°C。

---

#### P2-2：TNM 连续优化替代纯网格搜索

**代码位置**：`src/ps/tnm_inverse_reconstruction.py:70-93` 的 `reconstruct()`

当前 TNM 逆重构使用全网格枚举（11×11×9×9 ≈ 9801 个候选）。TNM 的前向模型对输入是完全可微的（KWW 函数有解析导数），可以直接做连续优化：

```python
from scipy.optimize import minimize

def loss(params):
    T1, T2, log_t1, log_t2 = params
    # 构建特征 → TNM 前向预测 → 比较目标
    ...

# 多点启动避免局部最优
for init in grid_top3:
    result = minimize(loss, init, method='L-BFGS-B', bounds=[...])
```

保留网格搜索作为初值生成器。

---

#### P2-3：加入 recovery_index 的温度相关 bias 修正

**代码位置**：`src/ps/tnm_calibrated_model.py` 的预测后处理

当前 recovery_index 存在系统偏差：低温低估（50-60°C bias=-0.041）、高温高估（95-100°C bias=+0.059）。在输出层加一个简单的温度相关修正项：

```python
bias_correction = a * (T1_C - T_ref) + b
pred_recovery_corrected = pred_recovery + bias_correction
```

其中 a, b 从训练残差拟合。

---

### P3：低优先级（需要更多数据或较大改动）

#### P3-1：链式多输出预测

先预测最容易的目标（ΔH_total），然后将其预测值作为额外特征去预测其他目标。改善弱信号目标（如 path_dependence_index）的预测。

#### P3-2：不确定性校准

对模型输出的 uncertainty 做 calibration plot（预测不确定性 vs 实际绝对误差）。如果 Pearson r < 0.3，说明 EIG 代理基本无效，需要保序回归校准或共形预测。

#### P3-3：循环实验验证闭环

执行 EIG 推荐的下一轮实验 → 将新数据加入训练集 → 重新训练 → 对比预测误差变化趋势。经过 2-3 轮循环后检验模型预测能力是否收敛。

---

## 四、下一步实验设计方案

### 核心策略转变

当前数据量（57 个双步样本，82% 集中在 80-90°C）下，**对四参数做精确点估计在物理上就不可能**。应将目标从"找到唯一最优 (T₁, t₁, T₂, t₂)"转为"确定哪些参数区间与观测一致、哪些被排除"。

### 实验矩阵

#### 批次 A：修复低温区间覆盖缺失（P0，8 个实验）

| # | T1 (°C) | t1 (s) | T2 (°C) | t2 (s) | 目的 |
|---|---------|--------|---------|--------|------|
| A1 | 50 | 100 | 70 | 600 | 低温 up-jump 锚点 |
| A2 | 50 | 600 | 70 | 100 | 低温时间不对称 |
| A3 | 55 | 100 | 75 | 600 | 中低温 up-jump |
| A4 | 55 | 600 | 75 | 100 | 中低温时间不对称 |
| A5 | 65 | 100 | 85 | 600 | 65-75 区间补点 |
| A6 | 65 | 600 | 85 | 100 | 65-75 时间不对称 |
| A7 | 75 | 100 | 100 | 600 | 高跨度 up-jump |
| A8 | 75 | 600 | 100 | 100 | 极端不对称 |

#### 批次 B：时间对比度实验（P1，6 个实验）

固定温度路径，只改变时间分配以打破时温简并：

| # | T1 | t1 | T2 | t2 | 目的 |
|---|----|----|----|----|------|
| B1 | 80 | 10 | 90 | 600 | 极端 t1 ≪ t2 |
| B2 | 80 | 600 | 90 | 10 | 极端 t1 ≫ t2 |
| B3 | 80 | 60 | 90 | 300 | 中等不对称 A |
| B4 | 80 | 300 | 90 | 60 | 中等不对称 B |
| B5 | 80 | 100 | 80 | 600 | 等温不对称 A |
| B6 | 80 | 600 | 80 | 100 | 等温不对称 B |

#### 批次 C：多升温速率 ARRT/Kissinger（P1，4 状态 × 4 速率 = 16 次扫描）

在同一个退火状态下，用 5/10/20/40 °C/min 四种升温速率做最终扫描：

| 状态 | 类型 | T1 | t1 | T2 | t2 | 目的 |
|------|------|----|----|----|----|------|
| S1 | 单步 | 70 | 300 | — | — | 已有 hs-01 锚点 |
| S2 | 单步 | 90 | 100 | — | — | 已有 hs-01 锚点 |
| S3 | 单步 | 95 | 100 | — | — | 已有 hs-01 锚点 |
| S4 | 双步 | 65 | 600 | 85 | 100 | **新增**：覆盖 65-75 缺口的代表性双步 |

### 评估标准转变

不要只用 Top-K 命中率评估逆重构。加入以下指标：

1. **后验覆盖率**：真实参数是否落在后验 P5-P95 区间内
2. **后验宽度收缩**：新数据加入后，后验区间是否变窄（量化信息增益）
3. **损失平坦度改善**：`loss_flatness_ratio` 是否下降
4. **参数间 trade-off 可视化**：T₁ vs log(t₁) 的后验散点图，展示时温补偿方向

---

## 五、立即行动清单

| # | 文件 | 行号 | 改动 | 预期收益 |
|---|------|------|------|----------|
| 1 | `src/ps/tnm_calibrated_model.py` | 190-191 | `np.array_split` → 分组折叠 `group_folds()` | 消除 CV 信息泄露 |
| 2 | `src/ps/tnm_calibrated_model.py` | TNM 特征 | 确保 `log_t1`, `log_t2` 在特征矩阵中 | 恢复时间分辨率 |
| 3 | `src/ps/tnm_calibrated_model.py` | Ridge λ | 1e-4 → 0.05 | 稳定线性头系数 |
| 4 | `src/ps/inverse_design.py` | 26-31 | `extrapolation` 和 `identifiability` 权重生效 | 排除不可辨识区域 |
| 5 | `src/ps/inverse_design.py` | 263-298 | 从 try/except fallback 改为硬导入 | 防止静默失效 |
| 6 | `src/ps/inverse_design.py` | 76-103 | 损失计算前强制过滤 T₂ 不合规候选 | T₂ MAE 降 3-5°C |
| 7 | `src/ps/inverse_design.py` | 300-385 | `search()` 输出包含后验分布 | 诚实反映不确定性 |
| 8 | `src/ps/inverse_design.py` | 263-298 | 对 Top-3 候选做 1°C 局部细网格 | 温度精度 5°C→1°C |
| 9 | `src/ps/train_ps_model.py` | 41-48 | 从 `TARGETS` 移除 `kovacs_peak_label` 和 `onset_temperature_C` | 消除退化目标 |
| 10 | `src/ps/train_ps_model.py` | 104-119 | 单次 split → 重复分组 CV，报告均值 ± 标准差 | 统计严谨性 |
| 11 | 新建 `tools/compare_all_models.py` | — | 同一 split 下统一对比 raw / physics / TNM | 统一评估基线 |

---

## 六、总结

代码架构和方法论方向正确（物理先验 → 核回归 / TNM → 正向预测 → 逆向搜索），但逆重构失败的根因不在模型架构，而在三个根本性问题：

1. **TNM dose 特征饱和**导致时间参数在特征空间坍缩，t₁ 完全不可辨识（50x 误差）
2. **逆搜索损失函数缺少可辨识性正则化**，在平坦地形上随机选择"最优"解
3. **训练数据 82% 集中在 80-90°C 单一温度区间**，模型在其他区间纯外推

修复 (1)(2) 可立即改善逆重构诊断质量（让真实误差暴露出来），但要实质性缩小重构误差，必须通过 (3) 系统性补数据后才能实现。

**最关键的原则转变**：不要追求四参数精确点估计。在玻璃态弛豫的物理框架下，时温叠加意味着逆向天然欠定。正确的做法是返回后验分布，诚实地告诉实验者"T₂ 被约束在 ±3°C，但 t₁ 在 10-1000s 范围内都与观测一致"——这正是实验设计决策所需的信息。
