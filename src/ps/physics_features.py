"""Physics-informed PS annealing features.

The current PS dataset is small, so these features intentionally encode a
simple prior: finite-time enthalpy recovery should mostly follow accumulated
structural-relaxation progress.  The parameters are not claimed as fitted PS
material constants; they form a compact TNM/ARRT-inspired basis for sparse
learning and residual correction.
"""

from __future__ import annotations

import math

from src.physics.arrt import R


T_REF_K = 373.15
TAU_REF_S = 100.0
ENERGY_GRID_KJ_MOL = (120.0, 160.0, 200.0)
BETA_GRID = (0.35, 0.50, 0.65)


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_time(value: object) -> float:
    return max(_as_float(value, 0.0), 0.0)


def _temperature_k(value: object) -> float:
    return _as_float(value, 0.0) + 273.15


def relaxation_tau_s(temperature_k: float, activation_energy_kj_mol: float) -> float:
    """Arrhenius relaxation time anchored near the PS Tg region."""
    exponent = (activation_energy_kj_mol * 1000.0 / R) * (1.0 / temperature_k - 1.0 / T_REF_K)
    exponent = min(max(exponent, -80.0), 80.0)
    return TAU_REF_S * math.exp(exponent)


def tnm_dose(temperature_k: float, time_s: float, activation_energy_kj_mol: float, beta: float) -> float:
    """Dimensionless stretched-exponential relaxation progress."""
    if time_s <= 0.0:
        return 0.0
    tau = max(relaxation_tau_s(temperature_k, activation_energy_kj_mol), 1e-30)
    ratio = max(time_s / tau, 0.0)
    exponent = min(max(ratio**beta, 0.0), 80.0)
    return 1.0 - math.exp(-exponent)


def advance_state(state: float, temperature_k: float, time_s: float, activation_energy_kj_mol: float, beta: float) -> float:
    """Sequentially advance a normalized structural recovery state."""
    dose = tnm_dose(temperature_k, time_s, activation_energy_kj_mol, beta)
    return 1.0 - (1.0 - max(min(state, 1.0), 0.0)) * (1.0 - dose)


def _suffix(energy_kj_mol: float, beta: float) -> str:
    return f"e{int(round(energy_kj_mol)):03d}_b{int(round(beta * 100)):03d}"


def physics_feature_dict(row: dict[str, object]) -> dict[str, float]:
    """Return deterministic physics-informed features from annealing inputs."""
    mode = str(row.get("mode", "single_step"))
    t1 = _safe_time(row.get("t1_s"))
    t2 = _safe_time(row.get("t2_s")) if mode == "two_step" else 0.0
    T1 = _temperature_k(row.get("T1_C"))
    T2_raw = _as_float(row.get("T2_C"), math.nan)
    T2 = (T2_raw + 273.15) if mode == "two_step" and math.isfinite(T2_raw) else T1
    total = max(t1 + t2, 1e-9)
    equivalent_T = (t1 * T1 + t2 * T2) / total

    out = {
        "phys_inv_T1_K": 1.0 / T1,
        "phys_inv_T2_K": 1.0 / T2,
        "phys_equivalent_T_K": equivalent_T,
        "phys_inv_equivalent_T_K": 1.0 / equivalent_T,
        "phys_log_total_time_s": math.log10(total),
        "phys_step_time_fraction_1": t1 / total,
        "phys_step_time_fraction_2": t2 / total,
    }

    for energy in ENERGY_GRID_KJ_MOL:
        for beta in BETA_GRID:
            suffix = _suffix(energy, beta)
            step1 = advance_state(0.0, T1, t1, energy, beta)
            total_state = advance_state(step1, T2, t2, energy, beta) if mode == "two_step" else step1
            step2_only = tnm_dose(T2, t2, energy, beta) if mode == "two_step" else 0.0
            equivalent = tnm_dose(equivalent_T, total, energy, beta)
            matched_t2 = tnm_dose(T2, total, energy, beta)
            matched_t1 = tnm_dose(T1, total, energy, beta)

            out[f"tnm_state_step1_{suffix}"] = step1
            out[f"tnm_state_step2_only_{suffix}"] = step2_only
            out[f"tnm_state_total_{suffix}"] = total_state
            out[f"tnm_state_equivalent_{suffix}"] = equivalent
            out[f"tnm_path_excess_{suffix}"] = total_state - matched_t2
            out[f"tnm_temperature_path_span_{suffix}"] = matched_t2 - matched_t1
            out[f"tnm_step_mismatch_{suffix}"] = step2_only - step1

    return out


PHYSICS_FEATURES = list(physics_feature_dict({"mode": "two_step", "T1_C": 80, "t1_s": 300, "T2_C": 100, "t2_s": 300}))
