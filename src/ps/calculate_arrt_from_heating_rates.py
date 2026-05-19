"""Diagnose and calculate PS S*/H* from multi-heating-rate DSC peak positions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.physics.arrt_kissinger import PeakRatePoint, activation_from_peak_rates
from src.ps.dsc_processing import (
    RAW_FILES,
    ROOT,
    all_recovery_scan_indices,
    baseline_scan,
    corrected_signal,
    estimate_noise,
    extract_features,
    infer_condition,
    read_protocol,
    read_signal,
    scan_for_segment,
    segment_bounds,
)


OUT_DIR = ROOT / "results" / "ps" / "arrt_kissinger"
DOC_PATH = ROOT / "docs" / "ps_arrt_kissinger_calculation.md"


def finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def condition_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    def fmt(value: object) -> str:
        if not finite(value):
            return ""
        return f"{float(value):.6g}"

    return (
        str(row["mode"]),
        fmt(row["T1_C"]),
        fmt(row["t1_s"]),
        fmt(row.get("T2_C", "")),
        fmt(row.get("t2_s", "")),
    )


def scan_all_recovery_features() -> list[dict[str, object]]:
    """Extract final heating scans, including non-standard heating rates."""
    ref = baseline_scan(RAW_FILES["ref"])
    empty = baseline_scan(RAW_FILES["empty"])
    noise = estimate_noise(empty, ref)
    rows: list[dict[str, object]] = []

    for file_key, path in RAW_FILES.items():
        if file_key in {"ref", "empty"} or not path.exists():
            continue
        segments = read_protocol(path)
        bounds = segment_bounds(segments)
        signal = read_signal(path)
        previous_scan = -1
        cycle_id = 0
        for heat_idx in all_recovery_scan_indices(segments):
            cycle_segments = segments[previous_scan + 1: heat_idx + 1]
            previous_scan = heat_idx
            condition = infer_condition(file_key, cycle_segments)
            if condition["mode"] == "unknown":
                continue
            scan = scan_for_segment(signal, segments[heat_idx], bounds[heat_idx])
            if len(scan) < 20:
                continue
            cycle_id += 1
            corrected = corrected_signal(scan, ref)
            features = extract_features(corrected, noise, segments[heat_idx].rate_c_min)
            total_time = float(condition["t1_s"])
            if condition["mode"] == "two_step":
                total_time += float(condition["t2_s"])
            rows.append(
                {
                    "sample_id": f"{file_key}_{cycle_id:03d}",
                    "source_file": path.name,
                    "cycle_id": cycle_id,
                    "mode": condition["mode"],
                    "T1_C": condition["T1_C"],
                    "t1_s": condition["t1_s"],
                    "T2_C": condition["T2_C"],
                    "t2_s": condition["t2_s"],
                    "total_anneal_time_s": total_time,
                    "heating_rate_C_min": segments[heat_idx].rate_c_min,
                    **features,
                }
            )
    return rows


def calculate_group_results(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[condition_key(row)].append(row)

    calculated = []
    diagnostics = []
    for key, values in sorted(groups.items()):
        rates = sorted({float(v["heating_rate_C_min"]) for v in values if finite(v["heating_rate_C_min"])})
        diagnostic = {
            "condition_key": "|".join(key),
            "mode": key[0],
            "T1_C": key[1],
            "t1_s": key[2],
            "T2_C": key[3],
            "t2_s": key[4],
            "n_scans": len(values),
            "n_distinct_heating_rates": len(rates),
            "heating_rates_C_min": ";".join(f"{rate:g}" for rate in rates),
            "can_calculate_arrt_kissinger": len(rates) >= 3,
            "reason": "ok" if len(rates) >= 3 else "need at least three heating rates for the same annealed state",
        }
        diagnostics.append(diagnostic)
        if len(rates) < 3:
            continue

        points = []
        for row in values:
            if not finite(row["peak_temperature_Tp_C"]):
                continue
            points.append(
                PeakRatePoint(
                    heating_rate_k_s=float(row["heating_rate_C_min"]) / 60.0,
                    peak_temperature_k=float(row["peak_temperature_Tp_C"]) + 273.15,
                )
            )
        if len(points) < 3:
            diagnostic["can_calculate_arrt_kissinger"] = False
            diagnostic["reason"] = "not enough valid Tp values"
            continue
        result = activation_from_peak_rates(points)
        calculated.append(
            {
                **diagnostic,
                "activation_energy_E_kj_mol": result.activation_energy_kj_mol,
                "activation_enthalpy_H_star_kj_mol": result.activation_enthalpy_kj_mol,
                "activation_entropy_S_star_j_mol_K": result.activation_entropy_j_mol_k,
                "mean_peak_temperature_Tp_K": result.mean_peak_temperature_k,
                "kissinger_slope": result.slope,
                "kissinger_intercept": result.intercept,
                "kissinger_r2": result.r2,
            }
        )
    return calculated, diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, object]) -> None:
    text = f"""# PS ARRT/Kissinger 激活熵与激活焓计算诊断

## 1. 结论

可以按论文附录公式 5-9 先计算 `S*` 与 `H*`，但前提是：同一个退火态必须有多个升温速率下的 relaxation peak position `Tp`。

当前 PS 数据扫描了 `{summary['n_total_scans']}` 条 final heating 曲线，但没有任何一个退火态同时具备至少 3 个不同升温速率。因此严格的 ARRT/Kissinger 计算目前不能生成可训练的 `S* / H*` 标签。

这和上一版“由退火时间 + Tp 反推有效代理量”不同。上一版能给每条样本赋值，但属于模型代理；本诊断要求公式 5-9 的实验输入完整，因此宁可输出不可计算，也不把代理量伪装成实验量。

## 2. 使用公式

采用 Kissinger 关系：

```text
ln(beta / Tp^2) = C - E / (R Tp)
```

由斜率得到：

```text
E = -R * slope
```

附录公式 9 给出：

```text
E = H* + R Tp
H* = E - R Tp
```

再由绝对反应速率峰位近似：

```text
ln(beta / Tp^2) = -H* / (R Tp) + ln(kB R / (h H*)) + S* / R
```

求得：

```text
S* = R * [ln(beta / Tp^2) + H*/(R Tp) - ln(kB R / (h H*))]
```

## 3. 当前数据可计算性

- 退火态分组数：`{summary['n_condition_groups']}`
- 可严格计算的分组数：`{summary['n_calculable_groups']}`
- 最大升温速率种类数：`{summary['max_distinct_heating_rates_per_group']}`

## 4. 下一步实验建议

如果希望像论文一样先计算 `S* / H*` 再训练，建议为少量代表退火态增加多升温速率 DSC：

```text
beta = 5, 10, 20, 40 °C/min
```

每个退火态至少 3 个速率，最好 4 个速率以检查 Kissinger 线性。优先选择：

```text
single: 70 °C 300 s, 90 °C 100 s, 95 °C 100 s
two-step: 50 °C 10 s -> 80 °C 1800 s
two-step: 50 °C 10 s -> 100 °C 1800 s
two-step: 65 °C 10 s -> 100 °C 1200 s
```

这些点覆盖已有模型误差较高或 EIG 较高的区域。完成后，本脚本会自动输出可训练的 `activation_entropy_S_star_j_mol_K` 和 `activation_enthalpy_H_star_kj_mol`。

## 5. 输出

- 全部升温扫描特征：`results/ps/arrt_kissinger/all_recovery_scan_features.csv`
- 可计算性诊断：`results/ps/arrt_kissinger/condition_rate_diagnostics.csv`
- 严格公式计算结果：`results/ps/arrt_kissinger/arrt_kissinger_results.csv`
- 摘要：`results/ps/arrt_kissinger/summary.json`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scan_all_recovery_features()
    calculated, diagnostics = calculate_group_results(rows)
    write_csv(OUT_DIR / "all_recovery_scan_features.csv", rows)
    write_csv(OUT_DIR / "condition_rate_diagnostics.csv", diagnostics)
    write_csv(OUT_DIR / "arrt_kissinger_results.csv", calculated)
    max_rates = max((int(row["n_distinct_heating_rates"]) for row in diagnostics), default=0)
    summary = {
        "method": "Strict ARRT/Kissinger calculation from multi-heating-rate Tp data.",
        "n_total_scans": len(rows),
        "n_condition_groups": len(diagnostics),
        "n_calculable_groups": len(calculated),
        "max_distinct_heating_rates_per_group": max_rates,
        "minimum_required_distinct_heating_rates": 3,
        "can_train_with_strict_arrt_targets": len(calculated) >= 5,
        "outputs": {
            "all_recovery_scan_features": str((OUT_DIR / "all_recovery_scan_features.csv").relative_to(ROOT)),
            "condition_rate_diagnostics": str((OUT_DIR / "condition_rate_diagnostics.csv").relative_to(ROOT)),
            "arrt_kissinger_results": str((OUT_DIR / "arrt_kissinger_results.csv").relative_to(ROOT)),
            "report": str(DOC_PATH.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
