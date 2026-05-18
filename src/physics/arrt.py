"""Absolute reaction rate theory helpers.

The paper relates relaxation kinetics to

    k = (k_B T / h) exp(S*/R) exp(-H*/RT)

For the MVP predictor we use k ~= 1 / t as a coarse timescale estimate.
That makes H* a derived feature instead of a primary fitted quantity.
"""

from __future__ import annotations

import math


R = 8.31446261815324
K_B = 1.380649e-23
H_PLANCK = 6.62607015e-34


def rate_from_timescale(time_s: float, floor_s: float = 1e-9) -> float:
    """Approximate reaction rate from an annealing timescale."""
    return 1.0 / max(float(time_s), floor_s)


def activation_enthalpy_from_entropy(
    temperature_k: float,
    time_s: float,
    entropy_j_mol_k: float,
) -> float:
    """Estimate H* in kJ/mol from T, t, and S* using ARRT."""
    t = float(temperature_k)
    k = rate_from_timescale(time_s)
    entropy = float(entropy_j_mol_k)
    h_j_mol = R * t * (math.log(K_B * t / H_PLANCK) + entropy / R - math.log(k))
    return h_j_mol / 1000.0


def arrt_rate(
    temperature_k: float,
    entropy_j_mol_k: float,
    enthalpy_kj_mol: float,
) -> float:
    """Reaction rate from ARRT."""
    t = float(temperature_k)
    return (
        K_B * t / H_PLANCK
        * math.exp(float(entropy_j_mol_k) / R)
        * math.exp(-(float(enthalpy_kj_mol) * 1000.0) / (R * t))
    )
