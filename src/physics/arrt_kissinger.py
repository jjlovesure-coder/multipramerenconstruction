"""ARRT/Kissinger calculations for activation entropy and enthalpy.

The supplementary equations use peak positions measured at multiple heating
rates.  A Kissinger slope gives the apparent activation energy, and the
Starkweather/absolute-rate relation separates it into H* and S*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.physics.arrt import H_PLANCK, K_B, R


@dataclass(frozen=True)
class PeakRatePoint:
    heating_rate_k_s: float
    peak_temperature_k: float


@dataclass(frozen=True)
class ArrtKissingerResult:
    n_rates: int
    activation_energy_kj_mol: float
    activation_enthalpy_kj_mol: float
    activation_entropy_j_mol_k: float
    mean_peak_temperature_k: float
    slope: float
    intercept: float
    r2: float


def kissinger_fit(points: Iterable[PeakRatePoint]) -> tuple[float, float, float]:
    """Fit ln(beta / Tp^3) = intercept + slope * (1 / Tp)."""
    pts = list(points)
    if len(pts) < 3:
        raise ValueError("At least three heating rates are needed for a Kissinger fit.")
    x = np.array([1.0 / p.peak_temperature_k for p in pts], dtype=float)
    y = np.array([math.log(p.heating_rate_k_s / (p.peak_temperature_k ** 3)) for p in pts], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(intercept), float(r2)


def arrt_entropy_from_peak(
    activation_enthalpy_kj_mol: float,
    peak_temperature_k: float,
    heating_rate_k_s: float,
) -> float:
    """Compute S* from the absolute-rate peak equation.

    Uses the common approximation:

        ln(beta / Tp^3) = -H*/(R Tp) + ln(kB R / (h H*)) + S*/R

    where beta is K/s and H* is J/mol.
    """
    h_j_mol = activation_enthalpy_kj_mol * 1000.0
    if h_j_mol <= 0:
        return math.nan
    lhs = math.log(heating_rate_k_s / (peak_temperature_k ** 3))
    prefactor = math.log(K_B * R / (H_PLANCK * h_j_mol))
    return R * (lhs + h_j_mol / (R * peak_temperature_k) - prefactor)


def activation_from_peak_rates(points: Iterable[PeakRatePoint]) -> ArrtKissingerResult:
    """Calculate E, H*, and S* from multi-rate peak data."""
    pts = list(points)
    slope, intercept, r2 = kissinger_fit(pts)
    activation_energy_j_mol = -R * slope
    mean_tp = float(np.mean([p.peak_temperature_k for p in pts]))
    enthalpy_j_mol = activation_energy_j_mol - R * mean_tp
    enthalpy_kj_mol = enthalpy_j_mol / 1000.0
    entropies = [
        arrt_entropy_from_peak(enthalpy_kj_mol, p.peak_temperature_k, p.heating_rate_k_s)
        for p in pts
    ]
    valid_entropies = [v for v in entropies if math.isfinite(v)]
    entropy = float(np.mean(valid_entropies)) if valid_entropies else math.nan
    return ArrtKissingerResult(
        n_rates=len(pts),
        activation_energy_kj_mol=activation_energy_j_mol / 1000.0,
        activation_enthalpy_kj_mol=enthalpy_kj_mol,
        activation_entropy_j_mol_k=entropy,
        mean_peak_temperature_k=mean_tp,
        slope=slope,
        intercept=intercept,
        r2=r2,
    )
