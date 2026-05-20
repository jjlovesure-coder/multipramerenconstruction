"""Tiny RBF-kernel regressor implemented with numpy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class KernelRegressor:
    x_train: np.ndarray
    y_train: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    bandwidth: float | list[float] = 1.0
    ridge: float = 1e-6

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        bandwidth: float | Sequence[float] = 1.0,
        ridge: float = 1e-6,
    ) -> "KernelRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        cls._bandwidth_array(bandwidth, x.shape[1])
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        stored_bandwidth: float | list[float]
        if np.asarray(bandwidth, dtype=float).ndim == 0:
            stored_bandwidth = float(bandwidth)
        else:
            stored_bandwidth = [float(value) for value in bandwidth]
        return cls((x - mean) / std, y, mean, std, stored_bandwidth, ridge)

    @staticmethod
    def _bandwidth_array(bandwidth: float | Sequence[float], n_features: int) -> np.ndarray:
        values = np.asarray(bandwidth, dtype=float)
        if values.ndim == 0:
            if float(values) <= 0.0:
                raise ValueError("bandwidth must be positive")
            return np.full(n_features, float(values), dtype=float)
        values = values.reshape(-1)
        if len(values) != n_features:
            raise ValueError(f"ARD bandwidth has {len(values)} values but feature matrix has {n_features} columns")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("bandwidth values must be finite and positive")
        return values

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.x_mean.shape[0]:
            raise ValueError(f"x has {x.shape[1]} features but model expects {self.x_mean.shape[0]}")
        bandwidth = self._bandwidth_array(self.bandwidth, self.x_mean.shape[0])
        x_scaled = (x - self.x_mean) / self.x_std
        diff = x_scaled[:, None, :] - self.x_train[None, :, :]
        dist2 = np.sum((diff / bandwidth[None, None, :]) ** 2, axis=2)
        weights = np.exp(-0.5 * dist2) + self.ridge
        weights = weights / weights.sum(axis=1, keepdims=True)
        pred = weights @ self.y_train
        variance = np.sum(weights[:, :, None] * ((self.y_train[None, :, :] - pred[:, None, :]) ** 2), axis=1)
        if not np.isfinite(pred).all() or not np.isfinite(variance).all():
            raise FloatingPointError("KernelRegressor produced non-finite predictions")
        return pred, np.sqrt(np.maximum(variance, 0.0))
