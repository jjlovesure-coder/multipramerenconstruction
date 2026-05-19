# PS Physics Kernel Entrypoint Update

## What Changed

The default PS inverse-design and EIG entrypoints now use:

```text
results/ps/models/ps_physics_informed_kernel_model.json
```

This model uses raw annealing inputs plus selected TNM/ARRT-inspired features. The selected feature set is `raw_phys_beta` with bandwidth `3.0`.

Same-split core-target normalized error:

```json
{
  "raw_kernel": {
    "mean_mae_over_test_std": 0.7997237738809986,
    "median_mae_over_test_std": 0.8018631098353464
  },
  "physics_kernel": {
    "mean_mae_over_test_std": 0.7352417313321563,
    "median_mae_over_test_std": 0.7394096670064233
  },
  "physics_residual": {
    "mean_mae_over_test_std": 0.9025302580639518,
    "median_mae_over_test_std": 0.8253160315061012
  },
  "physics_baseline_only": {
    "mean_mae_over_test_std": 1.0418614040095204,
    "median_mae_over_test_std": 1.061212458326851
  }
}
```

## Inverse Prediction Change

Target used for comparison: `recovery_index = 0.8`.

| rank | old condition | old pred RI | old loss | physics condition | physics pred RI | physics loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50C 900s -> 55C 10s | 0.800 | 0.025 | 50C 10s -> 100C 1800s | 0.755 | 0.348 |
| 2 | 50C 900s -> 55C 30s | 0.800 | 0.029 | 50C 30s -> 100C 1800s | 0.755 | 0.350 |
| 3 | 50C 900s -> 55C 100s | 0.800 | 0.030 | 50C 60s -> 100C 1800s | 0.755 | 0.352 |
| 4 | 50C 900s -> 55C 60s | 0.800 | 0.030 | 50C 10s -> 100C 1200s | 0.752 | 0.353 |
| 5 | 55C 900s -> 55C 10s | 0.799 | 0.030 | 50C 100s -> 100C 1800s | 0.755 | 0.353 |
| 6 | 55C 900s -> 55C 30s | 0.799 | 0.034 | 50C 30s -> 100C 1200s | 0.752 | 0.355 |
| 7 | 50C 900s -> 50C 30s | 0.801 | 0.035 | 50C 60s -> 100C 1200s | 0.752 | 0.356 |
| 8 | 55C 900s -> 55C 60s | 0.799 | 0.036 | 50C 100s -> 100C 1200s | 0.752 | 0.357 |

Interpretation: the old inverse model could numerically hit `recovery_index ~= 0.8`, but it did so by staying near low-temperature neighboring conditions. The physics kernel is more conservative and predicts that high recovery in the current finite-time window needs a stronger temperature jump and long high-temperature second step.

## EIG Change

| rank | old EIG condition | old score | old pred RI | physics EIG condition | physics score | physics pred RI |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 100C 10s -> 100C 1200s | 1.931 | 0.609 | 100C 100s -> 100C 100s | 1.692 | 0.711 |
| 2 | 80C 10s -> 80C 600s | 1.842 | 0.604 | 70C 900s -> 100C 300s | 1.677 | 0.723 |
| 3 | 80C 100s -> 100C 60s | 1.754 | 0.606 | 50C 100s -> 100C 10s | 1.650 | 0.649 |
| 4 | 90C 10s -> 100C 100s | 1.780 | 0.562 | 50C 30s -> 100C 100s | 1.654 | 0.702 |
| 5 | 100C 600s -> 100C 300s | 1.768 | 0.618 | 70C 1800s -> 70C 300s | 1.633 | 0.681 |
| 6 | 75C 10s -> 90C 1800s | 1.762 | 0.648 | 100C 10s -> 100C 30s | 1.660 | 0.668 |
| 7 | 100C 10s -> 100C 10s | 1.766 | 0.518 | 100C 1800s -> 100C 10s | 1.638 | 0.732 |
| 8 | 65C 600s -> 70C 300s | 1.687 | 0.639 | 70C 60s -> 100C 60s | 1.626 | 0.678 |

Interpretation: EIG is no longer dominated by repeated same-temperature high-temperature points. It now gives higher priority to temperature-jump paths and path-memory contrasts, which is closer to the purpose of the two-step PS experiments.

## Practical Recommendation

Use the physics-kernel EIG list for the next round rather than the old EIG list. For inverse design, treat the new top conditions as conservative feasible candidates: they may not hit `recovery_index = 0.8` exactly, but their recommendations are physically more defensible under sparse data.
