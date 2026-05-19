# PS Physics-Informed Residual Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a PS surrogate model that embeds TNM/ARRT-style finite-time relaxation priors instead of relying only on raw temperature/time interpolation.

**Architecture:** Add a small PS physics-feature module that converts single-step and two-step annealing conditions into Arrhenius/TNM dose, sequential structural-state, and path-memory descriptors. Train a residual surrogate: a constrained physics baseline predicts core enthalpy-recovery targets, then an RBF kernel model learns the residual using both raw and physics-informed features. Keep the existing model unchanged for fair comparison.

**Tech Stack:** Python, numpy, csv/json, matplotlib, existing `KernelRegressor`, existing DSC feature table from `src.ps.dsc_processing`.

---

### Task 1: Physics Feature Unit

**Files:**
- Create: `tests/test_ps_physics_features.py`
- Create: `src/ps/physics_features.py`

- [ ] **Step 1: Write the failing test**

```python
from src.ps.physics_features import physics_feature_dict


def test_physics_dose_increases_with_temperature_at_fixed_time():
    low = physics_feature_dict({"mode": "single_step", "T1_C": "70", "t1_s": "300", "T2_C": "", "t2_s": ""})
    high = physics_feature_dict({"mode": "single_step", "T1_C": "100", "t1_s": "300", "T2_C": "", "t2_s": ""})
    assert high["tnm_state_total_e160_b050"] > low["tnm_state_total_e160_b050"]


def test_two_step_state_is_sequential_and_path_sensitive():
    row = {"mode": "two_step", "T1_C": "80", "t1_s": "300", "T2_C": "100", "t2_s": "300"}
    features = physics_feature_dict(row)
    assert features["tnm_state_total_e160_b050"] >= features["tnm_state_step1_e160_b050"]
    assert abs(features["tnm_path_excess_e160_b050"]) > 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_ps_physics_features.py`

Expected: FAIL with `ModuleNotFoundError` or missing `physics_feature_dict`.

- [ ] **Step 3: Implement the feature module**

Create `src/ps/physics_features.py` with `physics_feature_dict(row)` returning deterministic raw+physics descriptors from only annealing conditions.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_ps_physics_features.py`

Expected: PASS.

### Task 2: Residual Training Model

**Files:**
- Create: `src/ps/train_physics_informed_ps_model.py`
- Output: `results/ps/physics_informed/physics_informed_model_comparison.json`
- Output: `results/ps/physics_informed/physics_informed_test_predictions.csv`
- Output: `results/ps/physics_informed/physics_informed_mae_comparison.png`
- Output: `docs/ps_physics_informed_model_update.md`

- [ ] **Step 1: Implement model comparison**

Train three models on the same grouped split:

```text
raw_kernel: existing raw feature RBF
physics_kernel: RBF using raw + physics features
physics_residual: linear physics baseline + residual RBF
```

- [ ] **Step 2: Evaluate core targets**

Use these targets:

```text
delta_h_total_J_g
peak_area_J_g
recovery_index
path_dependence_index
```

Also report diagnostic targets:

```text
peak_temperature_Tp_C
peak_height_uW
```

- [ ] **Step 3: Save comparison artifacts**

Write JSON, CSV predictions, PNG MAE comparison, and Markdown interpretation.

### Task 3: Verification

**Files:**
- Verify: `tests/test_ps_physics_features.py`
- Verify: `src/ps/physics_features.py`
- Verify: `src/ps/train_physics_informed_ps_model.py`

- [ ] **Step 1: Run behavior test**

Run: `python tests/test_ps_physics_features.py`

- [ ] **Step 2: Run training comparison**

Run: `python -m src.ps.train_physics_informed_ps_model`

- [ ] **Step 3: Run syntax check**

Run: `python -m py_compile src\ps\physics_features.py src\ps\train_physics_informed_ps_model.py tests\test_ps_physics_features.py`

- [ ] **Step 4: Inspect output metrics**

Open `results/ps/physics_informed/physics_informed_model_comparison.json` and verify whether the physics-informed model improves the core targets. If it does not improve, report that directly and keep the result as evidence.
