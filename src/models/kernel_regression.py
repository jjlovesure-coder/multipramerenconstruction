"""Tiny RBF-kernel regressor implemented with numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KernelRegressor:
    x_train: np.ndarray
    y_train: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    bandwidth: float = 1.0
    ridge: float = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0, ridge: float = 1e-6) -> "KernelRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return cls((x - mean) / std, y, mean, std, bandwidth, ridge)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        x_scaled = (x - self.x_mean) / self.x_std
        diff = x_scaled[:, None, :] - self.x_train[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        weights = np.exp(-0.5 * dist2 / (self.bandwidth ** 2)) + self.ridge
        weights = weights / weights.sum(axis=1, keepdims=True)
        pred = weights @ self.y_train
        variance = np.sum(weights[:, :, None] * ((self.y_train[None, :, :] - pred[:, None, :]) ** 2), axis=1)
        return pred, np.sqrt(np.maximum(variance, 0.0))
