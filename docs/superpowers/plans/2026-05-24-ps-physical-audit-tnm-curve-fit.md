# PS Physical Audit And TNM Curve Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved minimal physical-correction path: narrower DSC enthalpy window, suspect 90 °C/500 s strict ARRT flag, curve-level TNM forward fitting, and posterior confidence intervals for inverse outputs.

**Architecture:** Keep existing scalar model entrypoints compatible while adding physically explicit diagnostics. `dsc_processing.py` becomes the single source of truth for the 40-100 °C main enthalpy features and legacy 40-160 °C comparison fields. A new curve-fitting module reads processed final-heating curves, fits a compact TNM-shaped curve model, and writes per-condition curve residuals and plots/tables. Existing inverse outputs gain posterior confidence labels without changing old fields.

**Tech Stack:** Python, NumPy, pandas, existing DSC parsing utilities, pytest, Markdown/CSV/JSON outputs.

---

## Files

- Modify: `src/ps/dsc_processing.py`
  - Add named feature windows.
  - Make `delta_h_total_J_g` and `peak_area_J_g` use the 40-100 °C physical window.
  - Preserve legacy 40-160 °C results as comparison fields.
- Modify: `src/ps/calculate_arrt_from_heating_rates.py`
  - Flag `single_step, 90 °C, 500 s` as suspect pending retest.
  - Keep it visible in diagnostics, but allow training code to exclude suspect rows.
- Modify: `src/ps/train_strict_arrt_model.py`
  - Exclude `quality_flag != ok` from default training.
  - Report included/excluded rows explicitly.
- Create: `src/ps/tnm_curve_fit.py`
  - Fit curve-level TNM-inspired relaxation shapes against final-heating DSC curves.
  - Export per-sample curve metrics and a Markdown report.
- Modify: `src/ps/tnm_inverse_reconstruction.py`
  - Add posterior interval confidence labels to TNM inverse summary rows.
- Modify/Add tests:
  - `tests/test_ps_model_targets_and_features.py`
  - `tests/test_ps_strict_arrt.py`
  - `tests/test_ps_tnm_curve_fit.py`
  - `tests/test_ps_tnm_calibration.py`

---

## Task 1: Narrow DSC enthalpy window while preserving legacy comparison

- [ ] Add constants in `src/ps/dsc_processing.py`:

```python
MAIN_INTEGRATION_LOW_C = 40.0
MAIN_INTEGRATION_HIGH_C = 100.0
LEGACY_INTEGRATION_LOW_C = 40.0
LEGACY_INTEGRATION_HIGH_C = 160.0
MAIN_PEAK_LOW_C = 70.0
MAIN_PEAK_HIGH_C = 100.0
LEGACY_PEAK_LOW_C = 70.0
LEGACY_PEAK_HIGH_C = 140.0
```

- [ ] Refactor `extract_features()` so the primary fields are calculated from the main window:

```text
delta_h_total_J_g
peak_area_J_g
peak_temperature_Tp_C
peak_height_uW
onset_temperature_C
```

- [ ] Add legacy comparison fields:

```text
delta_h_total_40_160_J_g
peak_area_70_140_J_g
peak_temperature_Tp_70_140_C
peak_height_70_140_uW
```

- [ ] Run:

```powershell
python -m pytest tests/test_ps_model_targets_and_features.py tests/test_ps_process_features.py -q
```

Expected: tests pass after expected assertion updates.

---

## Task 2: Flag suspect 90 °C, 500 s strict ARRT condition

- [ ] Add a helper in `src/ps/calculate_arrt_from_heating_rates.py`:

```python
def arrt_quality_flag(row: dict[str, object]) -> tuple[str, str]:
    if (
        str(row.get("mode")) == "single_step"
        and abs(float(row.get("T1_C", "nan")) - 90.0) <= 1e-6
        and abs(float(row.get("t1_s", "nan")) - 500.0) <= 1e-3
    ):
        return "suspect_retest_required", "90 C 500 s H*/S* decreased unexpectedly; user plans retest"
    return "ok", ""
```

- [ ] Include `quality_flag` and `quality_note` in `arrt_kissinger_results.csv`.
- [ ] Update `src/ps/train_strict_arrt_model.py` to use only `quality_flag == ok` by default.
- [ ] Run:

```powershell
python -m pytest tests/test_ps_strict_arrt.py -q
```

Expected: test confirms suspect row is present in diagnostics/results and excluded from default strict ARRT training rows.

---

## Task 3: Add curve-level TNM forward fitter

- [ ] Create `src/ps/tnm_curve_fit.py`.
- [ ] Reuse existing DSC parsing to collect final heating curves.
- [ ] Fit a compact curve-level model:

```text
y_hat(T) = baseline0 + baseline1*T + amplitude * exp(-0.5 * ((T - Tp_pred) / width)^2)
Tp_pred = a0 + a1 * normalized_TNM_recovery + a2 * T1 + a3 * T2
amplitude_pred = b0 + b1 * normalized_TNM_recovery
```

This is not claimed as full material TNM. It is the minimal curve-level forward check before a full TNM solver.

- [ ] Export:

```text
results/ps/tnm_curve_fit/tnm_curve_fit_by_sample.csv
results/ps/tnm_curve_fit/tnm_curve_fit_summary.json
docs/ps_tnm_curve_fit_forward_report.md
```

- [ ] Metrics per sample:

```text
curve_rmse_uW
curve_mae_uW
area_error_J_g
tp_error_C
peak_height_error_uW
n_points_fit
```

- [ ] Run:

```powershell
python -m src.ps.tnm_curve_fit
python -m pytest tests/test_ps_tnm_curve_fit.py -q
```

Expected: outputs exist, contain per-sample curve metrics, and report whether forward curve fit is good enough to trust inverse work.

---

## Task 4: Add posterior confidence interval labels

- [ ] In `src/ps/tnm_inverse_reconstruction.py`, add:

```python
def interval_confidence(width: float, high_threshold: float, medium_threshold: float) -> str:
    if width <= high_threshold:
        return "high"
    if width <= medium_threshold:
        return "medium"
    return "low"
```

- [ ] Add fields:

```text
posterior_confidence_T1
posterior_confidence_T2
posterior_confidence_t1
posterior_confidence_t2
overall_posterior_confidence
```

- [ ] Run:

```powershell
python -m pytest tests/test_ps_tnm_calibration.py -q
```

Expected: posterior confidence labels appear and are based on interval width, not top-1 error.

---

## Task 5: Regenerate reports and verify

- [ ] Run:

```powershell
python -m src.ps.calculate_arrt_from_heating_rates
python -m src.ps.train_strict_arrt_model
python -m src.ps.tnm_curve_fit
python -m src.ps.tnm_inverse_reconstruction
```

- [ ] Run focused tests:

```powershell
python -m pytest tests/test_ps_model_targets_and_features.py tests/test_ps_process_features.py tests/test_ps_strict_arrt.py tests/test_ps_tnm_curve_fit.py tests/test_ps_tnm_calibration.py -q
```

- [ ] Update `docs/physics_forward_inverse_diagnostic.md` with one short addendum linking the new outputs.

Expected final state:

```text
Main enthalpy uses <=100 °C.
Legacy 40-160 °C fields remain available for comparison.
90 °C, 500 s strict ARRT row is flagged as suspect and excluded from default strict training.
TNM forward has curve-level residual outputs.
Inverse reports posterior confidence intervals.
```

