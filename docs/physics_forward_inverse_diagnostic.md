# PS 正向预测与反向重构的物理底层诊断

## 0. 先给结论

目前预测和反向重构效果不足，不能简单归因于“回归算法不够强”。更合理的判断是：

1. **四参数逆向重构本身高度多解**。最终升温曲线只能看到退火后的结构态投影，不能唯一还原 `T1, t1, T2, t2` 的完整路径。即使正向模型很好，逆向也更适合输出 posterior 区间，而不是单点答案。
2. **当前 TNM 嵌入仍是有效预测特征，不是完整 TNM 曲线拟合器**。代码中已有 fictive-temperature 版本，但主线里大量特征仍是 dose/state + 线性头，和论文附录中直接拟合 `Tf(t)` 与 `Δh(t)` 的 TNM 模型不是同一个强度。
3. **严格 `H* / S*` 标签现在太少，而且 90 °C 单步数据出现非单调信号**。这会让“给定输出反推时间”的问题变得不稳定，尤其 `90 °C, 500 s` 在 `H* / S*` 上没有表现成更长时间的简单延拓。
4. **焓变计算可能贡献系统误差，但目前不像唯一主因**。需要做 baseline/window/sign/mass/rate 的敏感性审计；不过反演失败的形态更像“观测量不足 + 多对一映射 + 动力学标签稀疏”，而不是单一积分公式错误。

我的当前判断：**首先应验证测量与特征提取的物理一致性，然后把 TNM 从“特征先验”推进到“曲线级 forward 模型校核”，最后再做反演。** 反演应默认是条件 posterior 问题，而不是四参数唯一预测问题。

---

## 1. DSC 升温曲线到焓变的定义

### 1.1 测量量

DSC 给出热流信号：

```text
q(T) : heat flow, 当前代码单位为 uW
β = dT/dt : heating rate, 当前常用 °C/min
m : sample mass = 4.7 mg = 0.0047 g
```

升温时：

```text
dt = dT / β
```

因此温度积分换算为比焓：

```text
d h = q(t) dt / m
    = q(T) dT / (β m)
```

若 `q` 用 `uW`，`β` 用 `°C/min`，则：

```text
integral_uW_C = ∫ q(T) dT
energy_J = integral_uW_C * (60 / β) * 1e-6
specific_enthalpy_J_g = energy_J / m_g
```

这正对应当前 `src/ps/dsc_processing.py` 中的换算：

```text
delta_h_total_J_g = ∫ DSC_detrended_uW dT * (60 / heating_rate_C_min) * 1e-6 / 0.0047
```

### 1.2 基线与积分窗口

当前处理流程是：

1. 读取 `ref` 作为基线；
2. 对目标 scan 按温度插值扣除 `ref`；
3. 在 `40-160 °C` 做线性 detrend；
4. `delta_h_total` 对 `40-160 °C` 的 detrended 曲线积分；
5. `peak_area` 在 `70-140 °C` 中找主峰，再取峰高 20% 阈值和 `±25 °C` 局部区域积分；
6. `Tp` 是该局部峰的温度位置。

这些定义本身可运行，但有几个物理风险：

- PS 的 relaxation peak 与玻璃化转变附近的基线变化耦合，线性 detrend 可能过度或不足扣除 `Cp` 漂移。
- 当前 `peak_idx = argmax(abs(signal))`，而 ARRT/Kissinger 严格计算改用 `95-125 °C` 的负峰 `argmin`。这意味着普通特征 `Tp` 和 ARRT `Tp` 不是完全同一个峰定义。
- `delta_h_total` 的符号取决于仪器热流方向和扣除方式；如果不同批次 ref/empty 漂移，绝对值和符号都可能受影响。
- 缺尾端数据会直接破坏峰位和积分。`ps-single-hs-01.xlsx` 中 `90 °C, 1000 s` 已被标记为 incomplete，不能用于严格 `H* / S*`。

### 1.3 需要确认的焓变审计

为了排除“焓变算错”这个根因，应做以下审计：

```text
同一曲线对 40-160, 50-150, 60-145 °C 积分窗口做敏感性
同一曲线对 ref-only, empty-ref, endpoint-linear, polynomial baseline 做敏感性
检查 4.7 mg 质量、升温速率 β、热流单位 uW 的换算
画出每个条件扣基线前/后曲线，确认主峰方向和符号一致
对重复点估计 delta_h_total 与 peak_area 的实验标准差
```

如果这些敏感性已经达到或超过模型误差，那么预测差首先是测量/特征不确定性问题；如果敏感性远小于模型误差，则主要矛盾在动力学建模和反演可辨识性。

---

## 2. 绝对反应速率理论与 Kissinger/ARRT

### 2.0 PS 与 Song 2020 体系的边界

这里借用 Song 2020 的 `H* / S*` 思路时，必须先划清边界。原论文是 Au 基金属玻璃、flash DSC 高速量热、原子/局域协同重排体系；当前对象是聚苯乙烯，主导过程更接近链段 `α` 松弛和 sub-Tg physical aging，并且强烈受 `Tg` 附近 `Cp` step、分子量/热史、样品接触、温度校准和基线选择影响。

因此，`S*` 在当前 PS 项目中应先被理解为：

```text
kinetic-coordinate analogy
```

也就是一个动力学坐标或约束特征，而不是可以直接照搬为“PS memory effect 判据”的材料常数。只有当 PS 自己的 `Δh(t)`、`H*_Kissinger(t)` 和 `S*_Kissinger(t)` 标定曲线稳定后，才能讨论它是否真的控制 PS 的路径记忆。

### 2.1 绝对反应速率理论

论文附录使用的核心思想是：结构松弛可近似看作跨越势垒的热激活过程。速率常数：

```text
k(T) = (kB T / h) exp(S* / R) exp(-H* / R T)
```

等价地：

```text
τ(T) = 1 / k(T)
ln τ = ln(h / kB T) - S*/R + H*/RT
```

其中：

- `H*`：迁移/松弛激活焓；
- `S*`：迁移/松弛激活熵；
- `R`：气体常数；
- `kB`：Boltzmann 常数；
- `h`：Planck 常数。

**物理含义**：`H*` 控制温度敏感性，`S*` 控制可达构型/协同运动通道数量。论文把 memory effect 与 `S*` 的跃迁方向联系起来：从低 `S*` 状态跳到高 `S*` 状态更容易出现记忆峰。

为避免混淆，后文统一区分三类符号：

```text
H*_Kissinger / S*_Kissinger: 多升温速率 Tp 由 Kissinger/ARRT 严格计算的实验量
H*_TNM_eff / S*_TNM_eff: TNM forward 模型参数或 tau_ref 推导出的有效预测量
H*_proxy / S*_proxy: 由单条曲线、退火时间或近似公式反推的代理量
```

### 2.2 Kissinger 峰位关系

多升温速率实验中，取同一退火态在不同 `β` 下的 relaxation peak position `Tp`。Kissinger 关系：

```text
ln(β / Tp^2) = C - E / (R Tp)
```

令：

```text
x = 1 / Tp
y = ln(β / Tp^2)
```

线性拟合：

```text
y = intercept + slope * x
E = -R * slope
```

这里 `β` 的单位必须特别小心：

```text
DSC 焓变换算: β_heat_C_min, 单位 °C/min
Kissinger/ARRT 熵公式: β_heat_K_s, 单位 K/s
```

斜率求 `E` 时，把所有升温速率同时乘以常数只改变截距，不改变斜率；但 `S*_Kissinger` 由截距/绝对峰位公式得到，若误把 `°C/min` 当成 `K/s`，会整体偏移：

```text
R ln(60) ≈ 34 J/mol/K
```

这个偏移已经接近当前 PS `S*` 条件差异的可观部分，因此必须固定单位。

附录中给出：

```text
E ≈ H* + R Tp
```

所以代码中使用：

```text
H* = E - R * mean(Tp)
```

严格来说，每个升温速率点都有自己的 `R Tp` 修正；当前用 `mean(Tp)` 给出组水平报告值。因为 `R Tp` 约为 `3 kJ/mol`，相对于当前 `400-600 kJ/mol` 的 `H*_Kissinger` 很小，但在报告里仍应说明这是均值化近似。

再由峰位方程求 `S*`：

```text
ln(β / Tp^2)
  = -H*/(R Tp) + ln(kB R / (h H*)) + S*/R
```

整理得：

```text
S* = R [ ln(β / Tp^2) + H*/(R Tp) - ln(kB R / (h H*)) ]
```

当前 `src/physics/arrt_kissinger.py` 正是这个实现，并且输入升温速率为 `heating_rate_C_min / 60.0`，即 `K/s`。

### 2.3 使用条件

严格 `H* / S*` 不是任意一条 DSC 曲线都能算，需要：

```text
同一个退火态
至少 3 个有效升温速率
每条曲线都完整覆盖 relaxation peak 窗口
峰位来自同一物理峰
Kissinger 拟合近似线性
```

当前项目使用 `95-125 °C` 作为 ARRT 峰位窗口，并且截断曲线不进入拟合。这是正确的保守策略。但仅有 3 个速率时 `R²` 可能显得过于乐观，所以报告还应同时给出：

```text
n_rates
各升温速率 Tp shift
Kissinger 残差图或残差表
是否覆盖完整 95-125 °C 峰位窗口
低置信度标记
```

### 2.4 当前严格 ARRT 数据的物理信号

当前诊断基于 `results/ps/arrt_kissinger/arrt_kissinger_results.csv`，并已排除 `ps-single-hs-01.xlsx` 中 incomplete 的 `90 °C, 1000 s` 多升温速率组。结果中，单步 80 °C 近似随时间增加：

```text
80 °C, 50 s:   H*_Kissinger ≈ 467 kJ/mol, S*_Kissinger ≈ 1017 J/mol/K
80 °C, 500 s:  H*_Kissinger ≈ 603 kJ/mol, S*_Kissinger ≈ 1379 J/mol/K
80 °C, 1000 s: H*_Kissinger ≈ 634 kJ/mol, S*_Kissinger ≈ 1464 J/mol/K
```

但 90 °C 不单调：

```text
90 °C, 50 s:  H*_Kissinger ≈ 529 kJ/mol, S*_Kissinger ≈ 1184 J/mol/K
90 °C, 100 s: H*_Kissinger ≈ 541 kJ/mol, S*_Kissinger ≈ 1214 J/mol/K
90 °C, 500 s: H*_Kissinger ≈ 470 kJ/mol, S*_Kissinger ≈ 1029 J/mol/K
```

这个现象非常关键。若它是真实物理，说明 90 °C 下更长退火可能已经进入另一种结构态/峰形选择区间，不能用简单单调时间坐标反推。若它是测量或峰位选择误差，则 `H* / S*` 标签本身会误导模型。

因此，`90 °C, 500 s` 预测失败不能简单说算法差；它可能暴露了 `H* / S*` 当前标签的物理一致性问题。

---

## 3. TNM 模型的物理定义

### 3.1 fictive temperature

玻璃结构态常用 fictive temperature `Tf` 描述。直观上：

```text
Tf 高：结构更接近高温液体态，焓较高，未充分老化
Tf 低：结构更接近平衡玻璃态，焓较低，老化更充分
```

在等温退火温度 `Ta` 下，`Tf(t)` 向 `Ta` 松弛，但松弛不是单指数，而是非指数、非线性的。

### 3.2 TNM 松弛时间

TNM 的关键是松弛时间同时依赖实际温度 `T` 和结构态 `Tf`：

```text
τ(T, Tf) = A exp[ H*/R * ( x/T + (1-x)/Tf ) ]
```

或写成：

```text
ln τ = ln A + H*/R * ( x/T + (1-x)/Tf )
```

参数含义：

- `A`：前因子；
- `H*`：激活焓；
- `x`：非线性参数，`0 < x <= 1`；
- `β`：KWW 非指数参数，`0 < β <= 1`。

若 `x=1`，松弛时间只依赖温度，退化为简单 Arrhenius。若 `x<1`，当前结构态也会影响后续松弛速度，这正是玻璃记忆/路径依赖的来源之一。

### 3.3 reduced time 与 KWW 响应

定义 reduced time：

```text
ξ(t) = ∫ dt / τ(T(t), Tf(t))
```

KWW 响应函数：

```text
φ(t) = exp[-ξ(t)^β]
```

对单步等温退火，可近似理解为：

```text
Tf(t) = Ta + [Tf(0) - Ta] exp[-(t/τ)^β]
```

这只是帮助理解的示意。严格 TNM 中 `τ(T,Tf)` 会随着 `Tf` 的演化而变化，因此一般不是一个固定 `τ` 的闭式 KWW 解，需要用 reduced time 或数值推进来更新 `Tf`。

对双步退火，需要分段叠加：

```text
step 1: T1, t1 使 Tf 从初始态向 T1 松弛
step 2: T2, t2 从 step 1 后的 Tf 继续向 T2 松弛
```

这正对应论文附录中单步和双步 TNM 公式的物理含义：不是简单用 `t_total`，而是通过 `Tf` 的历史状态进行路径传播。

### 3.4 焓变与 `Tf`

结构过剩焓可近似理解为相对于平衡态的构型焓差：

```text
h_excess(T, Tf) ≈ ∫_T^Tf ΔCp(T') dT'
```

退火使 `Tf` 向退火温度移动，从而降低结构过剩焓：

```text
aging: h_excess_before - h_excess_after
```

实验 final heating scan 的 recovery area 不是直接观测退火过程中的 `dTf`，而是在升温过程中，样品相对参考态释放/恢复结构焓后的投影积分。它还会受到 `Cp` step、基线选择和升温过程中继续松弛的影响。论文中 TNM 能拟合 `Δh(t)`，核心不是因为它直接“预测四参数”，而是因为它用少量物理参数描述了从时间温度路径到结构态演化的 forward map。

---

## 4. 当前代码里的 TNM 与论文 TNM 的差别

### 4.1 简化 dose/state 特征

早期和部分 physics-informed 特征使用：

```text
τ(T) = τ_ref exp[ E/R * (1/T - 1/T_ref) ]
dose = 1 - exp[-(t/τ)^β]
state_after_step = 1 - (1 - state_before) * (1 - dose)
```

这个形式有物理启发，但它不是完整 TNM：

- `τ` 主要依赖 `T`，不完整依赖 `Tf`；
- `state` 是归一化进度，不是真实 fictive temperature；
- 后面接线性/核回归头，参数更像“预测坐标”，不是材料本征参数。

### 4.2 calibrated TNM 版本

`src/ps/tnm_calibrated_model.py` 已经加入更接近 TNM 的函数：

```text
τ(T, Tf) = A exp[ H*/R * (x/T + (1-x)/Tf) ]
```

并用 `advance_fictive_temperature()` 分段推进 `Tf`。这是正确方向。但最终仍然通过有限 TNM 特征矩阵和线性 observation head 去预测多个 DSC 输出。因此它仍应被解释为：

```text
calibrated TNM forward predictor
```

而不是：

```text
唯一确定 PS 本征 TNM 参数的物理拟合
```

### 4.3 为什么 TNM 参数可能“不准”

参数不准有三种不同含义：

1. **材料本征不准**：`H*, A, x, β` 与真实 PS 不符；
2. **预测坐标不准**：参数让特征区分度最大，但不是物理本征；
3. **观测头吸收了物理误差**：线性/核模型把 baseline、峰形、积分误差也一起拟合了。

当前更可能是第 2 和第 3 种混合。也就是说，TNM 的引入帮助了 forward prediction，但还没有强到能支撑物理唯一反演。

当我们说 predictive TNM 参数远离 strict ARRT 锚点时，只能说它们代表的温度敏感性或有效时间尺度不同，不能直接说二者测量的是同一个本征 `H* / S*`。

---

## 5. 正向预测的物理路径

正向问题是：

```text
T1, t1, T2, t2 -> final heating DSC features
```

更物理的分解应为：

```text
thermal history
  -> structural state evolution, mainly Tf(t) or distribution of relaxation modes
  -> enthalpy recovery curve during final heating
  -> extracted features: Δh, peak_area, Tp, H*, S*
```

当前模型常把它压缩成：

```text
thermal history
  -> physics features / TNM dose / ARRT dose
  -> kernel or linear head
  -> extracted scalar targets
```

这个压缩会带来两个限制：

- 如果 scalar targets 丢掉了峰形信息，反演更不可辨识；
- 如果 `H* / S*` 来自单步或少数多速率点，它们未必能代表双步路径后的真实结构态。

因此，正向预测优先级应是：

```text
先拟合完整 DSC relaxation curve 或 Δh(t) 曲线
再从曲线提取指标
最后才做反向搜索
```

这与论文路径一致：TNM 先拟合焓变随退火时间的曲线，再解释 memory effect 区间。

---

## 6. 反向重构的物理限制

反向问题是：

```text
observed final heating features -> T1, t1, T2, t2
```

这在物理上不是严格一一映射。原因包括：

### 6.1 时温等效

低温长时间和高温短时间可能产生相近的 reduced time：

```text
ξ = ∫ dt / τ(T, Tf)
```

如果两个工况给出相近 `ξ` 和相近最终 `Tf`，最终升温曲线就可能非常接近。

### 6.2 路径信息被最终升温投影压缩

DSC final scan 主要看退火后的结构态。不同路径可能收敛到近似相同的 `Tf` 或相同的松弛模式分布，因此：

```text
(T1, t1, T2, t2)_A ≠ (T1, t1, T2, t2)_B
but features_A ≈ features_B
```

### 6.3 标量特征不足

如果只用：

```text
Δh, peak_area, Tp, H*, S*
```

这仍然是少数标量。四个输入参数加上路径非线性，反演会很容易出现平坦 loss 地形。固定 `T1/T2` 后只反演 `t1/t2` 是正确降阶，但若 `H* / S*` 本身不稳定，时间仍然不唯一。

### 6.4 后验分布比 top-1 更重要

因此反演输出应是：

```text
top candidates
posterior p5/median/p95
loss flatness
identifiability diagnostics
coverage / extrapolation label
```

如果 posterior 很宽，结论应是“不可辨识”，不是“模型预测错了一个点”。

---

## 7. 三类可能根因的判别

### 7.1 TNM 参数不准

**症状**

- forward curve shape 拟合不好；
- 同一温度下 `Δh(t)` 曲线斜率明显错；
- dose/state 特征很快饱和，无法区分 50 s 和 500 s；
- predictive TNM 参数远离 strict ARRT `H* / S*` 锚点。

**当前证据**

- 代码中已有 `dose_dynamic_range` 诊断，说明这类风险已经被注意到。
- 当前参数网格中的 `τ_ref` 很窄，很多特征可能更像“区分度调参”而不是物理标定。
- TNM conditional time inverse 在全双步数据上改善了 `t1/t2`，但 `t2` posterior 仍宽，说明 forward 信息有用但不够唯一。

**验证方法**

```text
对每个温度单独拟合 Δh(t) 曲线
检查 TNM 参数是否能同时拟合 80 °C 和 90 °C
比较 predictive_unconstrained 与 arrt_constrained 的 forward error
画 dose/state 随 log(t) 的曲线，确认 10-1800 s 内没有全部饱和
```

### 7.2 玻璃弛豫理解/状态变量不足

**症状**

- `H* / S*` 与时间或温度不呈简单单调；
- 双步路径中相同 `T2,t2` 受 `T1,t1` 明显影响；
- 单个 `Tf` 无法解释峰形变化；
- memory effect 与 `Δh` 或 `S*` 只在特定路径方向出现。

**当前证据**

- 90 °C 的 `H* / S*` 严格结果在 50/100/500 s 上不单调。
- 原论文强调 memory effect 出现需要 `Δh` 达到一定程度，并与从低 `S*` 到高 `S*` 的跃迁相关。这不是单纯的总时间问题。
- PS 预实验中没有明显 Kovacs 峰，说明当前时间窗可能主要是弱路径效应/焓回复，而非强 memory peak。

**验证方法**

```text
重测/补测 90 °C, 1000 s 多升温速率
在 80/90 °C 做单步 Δh(t) 密集曲线，确认 H*/S* 是否单调
比较 up-jump 80->90 与 down-jump 90->80 的同 t1/t2 对称性
从完整 relaxation curve 拟合 Tf 或 relaxation-mode distribution，而不是只用峰值标量
```

### 7.3 焓变计算有误

**症状**

- 重复点的 `Δh` 方差接近或超过条件间差异；
- `delta_h_total` 与 `peak_area` 变化方向互相矛盾；
- 换 baseline/window 后模型结论反转；
- 截断曲线或漂移曲线进入训练。

**当前证据**

- 已经排除了明显截断的 `90 °C, 1000 s` 严格 ARRT。
- 但普通 DSC 特征仍依赖 `40-160 °C` 线性 detrend 和局部峰阈值，确实需要敏感性检查。
- 如果 ref/empty 每批漂移没有被充分控制，`Δh_total` 会系统偏移。

**验证方法**

```text
baseline/window 敏感性矩阵
批次内 ref/empty 漂移图
同一条件重复实验误差条
积分前后曲线人工抽查
与已知 PS 文献量级比较 J/g 是否合理
```

---

## 8. 为什么“直接拟合数据点”有时会更好

你说“直接拟合数据点也不会差到这种地步”，这个判断是有道理的。

如果任务降成：

```text
固定 T，只在同一温度下由输出特征拟合 log10(t)
```

简单单变量/局部拟合可能会优于复杂 inverse search。原因是：

- 它不试图解决多温度、多路径的全局等效问题；
- 它避免把 `H* / S*` 的物理噪声扩大到四参数搜索；
- 它直接利用局部单调关系。

当前报告中 direct log-time fit 确实优于 nearest-candidate inverse，这说明“模型搜索形式”也会放大误差。但这并不意味着物理模型没用，而是说明物理模型应该先作为 forward curve constraint，而不是一开始就承担全局逆映射。

---

## 9. 建议的确认顺序

### Step 1: 测量与焓变定义审计

目标：确认 `Δh / peak_area / Tp` 没有被 baseline 或窗口主导。

输出应包括：

```text
每条代表曲线的 raw/ref-corrected/detrended 图
不同积分窗口下 Δh 的变化
不同 baseline 下 Δh 的变化
重复点误差条
```

### Step 2: 严格 `H* / S*` 标签审计

目标：确认 `H* / S*` 是稳定动力学坐标，不是峰位选择噪声。

重点检查：

```text
80 °C: 50/500/1000 s 是否单调可靠
90 °C: 50/100/500 s 非单调是否可重复
90 °C, 1000 s 是否补测完整
Kissinger R2 与 Tp shift 是否物理合理
```

### Step 3: 先做 TNM forward 曲线校核

目标：不先反演，而是问：

```text
给定 T 和 t，TNM 能不能拟合 Δh(t) 或完整 relaxation curve？
```

如果不能，反演没有意义。如果可以，再做条件反演。

### Step 4: 分层反演

建议顺序：

```text
固定 T，反推单步 t
固定 T1/T2，反推 t1/t2
固定 T2 或 Tp 窗口，反推 T1/t1/t2
最后才尝试完整 T1/t1/T2/t2 posterior
```

每一层都要报告 posterior 宽度。如果上一层都宽，下一层一定不应声称可唯一反推。

---

## 10. 下一版建模应坚持的物理边界

1. **`H* / S*` 分三类写清楚**：
   - `H*_Kissinger / S*_Kissinger`：多升温速率 Kissinger/ARRT 得到的严格实验量；
   - `H*_TNM_eff / S*_TNM_eff`：由 TNM 参数或 `τ_ref` 推导出的有效预测量；
   - `H*_proxy / S*_proxy`：由单条曲线、退火时间或近似公式反推的辅助量。

2. **TNM 参数不能直接叫 PS 本征常数**，除非它能跨温度、跨时间、跨批次拟合完整曲线。

3. **正向预测先于反向重构**。只有 forward curve fit 过关，inverse posterior 才有物理意义。

4. **反演默认多解**。固定温度和缩小候选空间是合理的物理约束，不是退步。

5. **PS 与金属玻璃不能机械等同**。原论文的 `S*` 逻辑可以作为 kinetic-coordinate 思路，但 PS 的链段松弛、玻璃化转变窗口、DSC 基线和峰形都不同，需要先建立 PS 自己的 `Δh(t)` 与 `H*/S*(t)` 标定曲线。

---

## 11. 我认为最可能的根因排序

### 11.1 Forward prediction 根因排序

这里的 forward prediction 指：

```text
给定退火工况 -> 预测 final heating 输出特征或曲线
```

| 排名 | 根因 | 判断 |
| --- | --- | --- |
| 1 | 焓变/峰面积/峰位特征稳定性尚未完成敏感性审计 | 如果 baseline/window/sign 的不确定性接近条件差异，任何 forward 模型都会被误导。 |
| 2 | strict `H*_Kissinger / S*_Kissinger` 样本少且 90 °C 非单调 | 直接影响 paper-style 坐标是否能作为稳定动力学标签。 |
| 3 | 当前 TNM 还不是完整曲线级校核模型 | dose/state 特征有用，但不能替代跨温度 `Tf(t)` 与 `Δh(t)` 曲线拟合。 |
| 4 | PS 与 Song 金属玻璃物理机制不同 | `S*` 可作为动力学坐标类比，但不能直接照搬 memory 判据。 |
| 5 | 回归器选择不够强 | 不是第一矛盾。小样本下更复杂算法可能只是更好地过拟合。 |

### 11.2 Inverse reconstruction 根因排序

这里的 inverse reconstruction 指：

```text
给定 final heating 输出特征 -> 反推 T1, t1, T2, t2 或 t1, t2
```

| 排名 | 根因 | 判断 |
| --- | --- | --- |
| 1 | 反向问题多解、最终升温标量信息不足 | 最核心。即使 forward 好，四参数也很难唯一反推。 |
| 2 | `T1/T2` 未固定时存在严重时温等效 | 低温长时与高温短时可能给出相似 reduced time 和相似 final scan。 |
| 3 | 当前 TNM forward 还没有被曲线级验证 | 反演依赖 forward map；forward map 不稳时 posterior 会虚假收窄或错误排序。 |
| 4 | `H*_Kissinger / S*_Kissinger` 对时间的单调性和可重复性不足 | 固定温度后反推时间仍可能失败。 |
| 5 | 搜索/损失函数形式放大误差 | nearest-candidate 或全局网格搜索可能比局部直接拟合更悲观。 |

一句话：**现在最该做的不是继续换模型，而是把“测量量是否可靠”、“动力学坐标是否稳定”、“TNM forward 是否拟合曲线”、“inverse 是否可辨识”四件事拆开验证。**

---

## 12. 额外实验风险清单

PS 比金属玻璃更容易受到实验流程细节影响，后续报告应显式记录：

```text
erase / rejuvenation 温度和保持时间是否一致
cooling rate 是否一致
退火结束到 final heating 之间是否存在等待 aging
样品盘接触状态和热阻
样品质量误差与称量记录
温度校准和不同升温速率下的温度滞后
氧化、残余溶剂、含水或样品老化
实验顺序是否随机化，是否存在批次漂移
同一条件是否有重复测量
```

这些因素不一定是当前主因，但如果不记录，会让后续把 PS 数据解释成 `H*_Kissinger / S*_Kissinger` 或 TNM 参数时缺少误差边界。

---

## 13. 方案 A 实施后的诊断入口

本轮按你的确认执行了最小物理修正闭环：

```text
主焓变/峰面积窗口: 40-100 °C
legacy 对照窗口: 40-160 °C
90 °C, 500 s strict ARRT: suspect_retest_required
TNM forward: 新增曲线级 40-100 °C 拟合诊断
TNM inverse: 新增 posterior confidence interval labels
```

新增/更新输出：

```text
results/ps/arrt_kissinger/arrt_kissinger_results.csv
results/ps/strict_arrt_model/strict_arrt_model.json
results/ps/tnm_curve_fit/tnm_curve_fit_summary.json
results/ps/tnm_curve_fit/tnm_curve_fit_by_sample.csv
docs/ps_tnm_curve_fit_forward_report.md
results/ps/tnm_calibrated/tnm_inverse_reconstruction_summary.json
results/ps/tnm_calibrated/tnm_inverse_reconstruction_by_sample.csv
```

当前关键结果：

```text
strict ARRT 可计算组: 11
默认严格训练组: 10
排除组: single_step|90|500||, quality_flag=suspect_retest_required

TNM curve-level forward:
n_curves = 113
curve_rmse_uW_mean = 45.65
curve_mae_uW_mean = 39.71
area_error_J_g_mae = 0.411
tp_error_C_mae = 0.465
peak_height_error_uW_mae = 9.39

TNM inverse after retraining on 40-100 °C targets:
mae_T1_C = 9.75 °C
mae_T2_C = 11.67 °C
mae_log10_t1 = 0.807
mae_log10_t2 = 1.144
posterior median width T1/T2 = 45/30 °C
posterior median width log10_t1/log10_t2 = 2.26/2.26
overall posterior confidence = low for all 60 two-step samples
```

解释：窄窗口和重新训练使 top-1 inverse 误差比旧模型/旧目标匹配时好很多，但 posterior 仍然极宽，所以当前结论仍是“forward 可继续校核，inverse 不可声称唯一”。这正支持后续必须做完整 TNM 曲线拟合器和复测 `90 °C, 500 s` 的判断。
