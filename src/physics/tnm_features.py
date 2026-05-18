"""Small TNM-inspired feature helpers.

This is not a full Tool-Narayanaswamy-Moynihan solver. The goal is to encode
the paper's fitted parameters as physics-informed features for sparse learning.
"""

from __future__ import annotations

import math

from .arrt import R


PAPER_H_KJ_MOL = 164.0
PAPER_A_S = 6e-22
PAPER_X = 0.6
PAPER_BETA = 0.43


def tnm_relaxation_time_s(
    temperature_k: float,
    activation_enthalpy_kj_mol: float = PAPER_H_KJ_MOL,
    prefactor_s: float = PAPER_A_S,
) -> float:
    """Arrhenius TNM relaxation timescale using the paper's fitted constants."""
    return prefactor_s * math.exp((activation_enthalpy_kj_mol * 1000.0) / (R * float(temperature_k)))


def stretched_dose(
    temperature_k: float,
    time_s: float,
    activation_enthalpy_kj_mol: float = PAPER_H_KJ_MOL,
    beta: float = PAPER_BETA,
) -> float:
    """Dimensionless TNM-like dose, 1 - exp(-(t/tau)^beta)."""
    tau = max(tnm_relaxation_time_s(temperature_k, activation_enthalpy_kj_mol), 1e-30)
    ratio = max(float(time_s), 0.0) / tau
    return 1.0 - math.exp(-(ratio ** beta))
