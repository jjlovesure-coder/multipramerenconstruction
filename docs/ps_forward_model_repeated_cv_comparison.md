# PS Forward Model Repeated-CV Comparison

## Policy

Use repeated grouped CV as primary evidence; same-split n=6 results are diagnostic only.

## Repeated Grouped CV

| model | mean normalized score | std | n scores |
|---|---:|---:|---:|
| `raw_kernel` | 0.8509 | 0.7392 | 35 |
| `physics_informed_kernel` | 0.8181 | 0.29 | 35 |
| `tnm_calibrated` | 0.8034 | 0.4878 | 35 |

## Coverage

```json
{
  "n_samples": 108,
  "T1_bin": {
    "50-60": {
      "n": 33,
      "fraction": 0.3055555555555556,
      "is_low_support_region": false
    },
    "65-75": {
      "n": 27,
      "fraction": 0.25,
      "is_low_support_region": false
    },
    "80-90": {
      "n": 43,
      "fraction": 0.39814814814814814,
      "is_low_support_region": false
    },
    "95-100": {
      "n": 5,
      "fraction": 0.046296296296296294,
      "is_low_support_region": false
    },
    "other": {
      "n": 0,
      "fraction": 0.0,
      "is_low_support_region": true
    }
  },
  "T2_bin": {
    "T2 50-60": {
      "n": 20,
      "fraction": 0.18518518518518517,
      "is_low_support_region": false
    },
    "T2 65-75": {
      "n": 27,
      "fraction": 0.25,
      "is_low_support_region": false
    },
    "T2 80-90": {
      "n": 47,
      "fraction": 0.4351851851851852,
      "is_low_support_region": false
    },
    "T2 95-100": {
      "n": 14,
      "fraction": 0.12962962962962962,
      "is_low_support_region": false
    },
    "T2 other": {
      "n": 0,
      "fraction": 0.0,
      "is_low_support_region": true
    }
  },
  "path_class": {
    "single_step": {
      "n": 48,
      "fraction": 0.4444444444444444,
      "is_low_support_region": false
    },
    "isothermal": {
      "n": 1,
      "fraction": 0.009259259259259259,
      "is_low_support_region": true
    },
    "up-jump": {
      "n": 39,
      "fraction": 0.3611111111111111,
      "is_low_support_region": false
    },
    "down-jump": {
      "n": 20,
      "fraction": 0.18518518518518517,
      "is_low_support_region": false
    }
  },
  "time_bin": {
    "short_<=60s": {
      "n": 44,
      "fraction": 0.4074074074074074,
      "is_low_support_region": false
    },
    "medium_60-300s": {
      "n": 17,
      "fraction": 0.1574074074074074,
      "is_low_support_region": false
    },
    "long_300-900s": {
      "n": 25,
      "fraction": 0.23148148148148148,
      "is_low_support_region": false
    },
    "very_long_>900s": {
      "n": 22,
      "fraction": 0.2037037037037037,
      "is_low_support_region": false
    }
  }
}
```

## Interpretation

This comparison exists to prevent over-reading one small test split. Forward prediction claims should cite repeated grouped CV mean/std and then use same-split plots only as diagnostics.
