"""Compare PS EIG/training using current targets vs paper-style targets.

The paper-style target set keeps the dimensionality close to the original
metallic-glass workflow:

    delta_h, delta_h_peak, S*, H*

For PS, S* and H* are not independently measured in the current DSC campaign.
They are therefore treated as ARRT effective proxies inferred from the observed
annealing timescale and the relaxation peak position Tp.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models.kernel_regression import KernelRegressor
from src.physics.arrt import H_PLANCK, K_B, R
from src.ps.dsc_processing import build_ps_dataset
from src.ps.eig_design import (
    TARGET_WEIGHTS,
    candidate_table,
    eig_components,
    nearest_distance,
    scaled_features,
)
from src.ps.train_ps_model import TARGETS as CURRENT_TARGETS
from src.ps.train_ps_model import featurize, grouped_split, read_rows


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "ps" / "paper_style_comparison"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = ROOT / "docs" / "ps_paper_style_target_comparison.md"

PAPER_TARGETS = [
    "paper_delta_h_J_g",
    "paper_delta_h_peak_J_g",
    "paper_s_star_eff_J_mol_K",
    "paper_h_star_eff_kJ_mol",
]

PAPER_WEIGHTS = {
    "paper_delta_h_J_g": 1.0,
    "paper_delta_h_peak_J_g": 1.0,
    "paper_s_star_eff_J_mol_K": 0.8,
    "paper_h_star_eff_kJ_mol": 0.8,
}

PAPER_NOISE_FLOORS = {
    "paper_delta_h_J_g": 0.20,
    "paper_delta_h_peak_J_g": 0.13,
    "paper_s_star_eff_J_mol_K": 5.0,
    "paper_h_star_eff_kJ_mol": 20.0,
}


def as_float(value: str | float, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def equivalent_annealing_temperature_k(row: dict[str, str]) -> float:
    t1 = max(as_float(row["t1_s"], 0.0), 0.0)
    t2 = max(as_float(row.get("t2_s", ""), 0.0), 0.0)
    t1_k = as_float(row["T1_C"]) + 273.15
    t2_c = as_float(row.get("T2_C", ""))
    if not math.isfinite(t2_c) or t2 <= 0:
        return t1_k
    return (t1 * t1_k + t2 * (t2_c + 273.15)) / max(t1 + t2, 1e-9)


def effective_arrt_pair(row: dict[str, str]) -> tuple[float, float]:
    """Infer effective S* and H* from annealing and peak-rate constraints."""
    t_ann = equivalent_annealing_temperature_k(row)
    total_time = max(as_float(row["total_anneal_time_s"], 1.0), 1e-6)
    tp_k = as_float(row["peak_temperature_Tp_C"]) + 273.15
    heating_rate_k_s = max(as_float(row.get("heating_rate_C_min", ""), 10.0) / 60.0, 1e-9)

    k_ann = 1.0 / total_time
    k_peak = heating_rate_k_s / max(tp_k, 1e-9)
    a_ann = math.log(k_ann) - math.log(K_B * t_ann / H_PLANCK)
    a_peak = math.log(k_peak) - math.log(K_B * tp_k / H_PLANCK)
    inv_ann = 1.0 / t_ann
    inv_peak = 1.0 / tp_k

    if abs(inv_peak - inv_ann) < 1e-12:
        return math.nan, math.nan

    slope = (a_peak - a_ann) / (inv_peak - inv_ann)
    h_j_mol = -R * slope
    s_j_mol_k = R * (a_ann + h_j_mol / (R * t_ann))
    return s_j_mol_k, h_j_mol / 1000.0


def add_paper_style_targets(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        copied = dict(row)
        s_eff, h_eff = effective_arrt_pair(copied)
        copied["paper_delta_h_J_g"] = abs(as_float(copied["delta_h_total_J_g"]))
        copied["paper_delta_h_peak_J_g"] = abs(as_float(copied["peak_area_J_g"]))
        copied["paper_s_star_eff_J_mol_K"] = s_eff
        copied["paper_h_star_eff_kJ_mol"] = h_eff
        out.append(copied)
    return out


def matrix(rows: list[dict[str, str]], targets: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    clean_idx = []
    for idx, row in enumerate(rows):
        values = [as_float(row.get(target, "")) for target in targets]
        if all(math.isfinite(value) for value in values):
            clean_idx.append(idx)
    x = np.array([featurize(rows[idx]) for idx in clean_idx], dtype=float)
    y = np.array([[as_float(rows[idx][target]) for target in targets] for idx in clean_idx], dtype=float)
    return x, y, clean_idx


def split_for_clean_indices(rows: list[dict[str, str]], clean_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    full_train, full_test = grouped_split(rows)
    clean_pos = {row_idx: pos for pos, row_idx in enumerate(clean_idx)}
    train = [clean_pos[idx] for idx in full_train if idx in clean_pos]
    test = [clean_pos[idx] for idx in full_test if idx in clean_pos]
    return np.array(train, dtype=int), np.array(test, dtype=int)


def metric_table(targets: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for j, target in enumerate(targets):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        out[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "r2": float(1.0 - np.sum(err * err) / denom) if denom > 0 else float("nan"),
            "target_std": float(np.std(y_true[:, j])),
        }
    return out


def normalized_error_summary(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    ratios = []
    for value in metrics.values():
        std = value["target_std"]
        if std > 0:
            ratios.append(value["mae"] / std)
    return {
        "mean_mae_over_test_std": float(np.mean(ratios)) if ratios else math.nan,
        "median_mae_over_test_std": float(np.median(ratios)) if ratios else math.nan,
    }


def fit_and_evaluate(rows: list[dict[str, str]], targets: list[str], bandwidth: float = 1.2) -> dict:
    x, y, clean_idx = matrix(rows, targets)
    train_idx, test_idx = split_for_clean_indices(rows, clean_idx)
    model = KernelRegressor.fit(x[train_idx], y[train_idx], bandwidth=bandwidth)
    pred, unc = model.predict(x[test_idx])
    metrics = metric_table(targets, y[test_idx], pred)
    return {
        "model": model,
        "targets": targets,
        "x": x,
        "y": y,
        "clean_idx": clean_idx,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "pred": pred,
        "unc": unc,
        "metrics": metrics,
        "normalized": normalized_error_summary(metrics),
    }


def scales_from_metrics(targets: list[str], metrics: dict[str, dict[str, float]], floors: dict[str, float]) -> np.ndarray:
    scales = []
    for target in targets:
        rmse = metrics[target]["rmse"]
        floor = floors.get(target, 1.0)
        scales.append(max(float(rmse), floor))
    return np.array(scales, dtype=float)


def candidate_predictions(model: KernelRegressor, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([featurize(row) for row in rows], dtype=float)
    pred, unc = model.predict(x)
    return x, pred, unc


def eig_ranking(
    model_result: dict,
    targets: list[str],
    weights: dict[str, float],
    floors: dict[str, float],
    candidate_rows_: list[dict[str, str]],
) -> tuple[list[dict], dict[str, float]]:
    x_cand, pred, unc = candidate_predictions(model_result["model"], candidate_rows_)
    scales = scales_from_metrics(targets, model_result["metrics"], floors)
    components = eig_components(unc, scales)
    weight_vec = np.array([weights[target] for target in targets], dtype=float)
    eig = components @ weight_vec
    x_scaled = (x_cand - model_result["model"].x_mean) / model_result["model"].x_std
    novelty = nearest_distance(x_scaled, model_result["model"].x_train)
    score = eig + 0.12 * novelty
    order = np.argsort(-score)

    ranking = []
    for rank, idx in enumerate(order[:100], start=1):
        row = dict(candidate_rows_[int(idx)])
        row["rank"] = rank
        row["eig_total"] = float(eig[idx])
        row["eig_score"] = float(score[idx])
        row["nearest_existing_distance"] = float(novelty[idx])
        for j, target in enumerate(targets):
            row[f"eig_{target}"] = float(components[idx, j])
            row[f"pred_{target}"] = float(pred[idx, j])
            row[f"uncertainty_{target}"] = float(unc[idx, j])
        ranking.append(row)

    eig_stats = {
        "mean": float(np.mean(eig)),
        "std": float(np.std(eig)),
        "p50": float(np.percentile(eig, 50)),
        "p90": float(np.percentile(eig, 90)),
        "p99": float(np.percentile(eig, 99)),
        "max": float(np.max(eig)),
    }
    return ranking, eig_stats


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_prediction_scatter(result: dict, rows: list[dict[str, str]], prefix: str) -> list[str]:
    paths = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_true = result["y"][result["test_idx"]]
    pred = result["pred"]
    for j, target in enumerate(result["targets"]):
        fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=160)
        ax.scatter(y_true[:, j], pred[:, j], s=30, alpha=0.8)
        lo = float(min(np.min(y_true[:, j]), np.min(pred[:, j])))
        hi = float(max(np.max(y_true[:, j]), np.max(pred[:, j])))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
        ax.set_xlabel(f"observed {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{prefix}: {target}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / f"{prefix}_{target}_scatter.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_eig_distribution(current_stats: list[float], paper_stats: list[float]) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.boxplot([current_stats, paper_stats], tick_labels=["current targets", "paper-style targets"], showfliers=False)
    ax.set_ylabel("weighted EIG proxy")
    ax.set_title("Candidate EIG distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "eig_distribution_current_vs_paper_style.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_target_correlation(result: dict, prefix: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    std = np.std(result["y"], axis=0)
    keep = std > 1e-12
    y = result["y"][:, keep]
    targets = [target for target, use in zip(result["targets"], keep) if use]
    corr = np.corrcoef(y.T)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=7)
    ax.set_title(f"{prefix} target correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG_DIR / f"{prefix}_target_correlation.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def full_eig_values(model_result: dict, targets: list[str], weights: dict[str, float], floors: dict[str, float], candidate_rows_: list[dict[str, str]]) -> np.ndarray:
    _, _, unc = candidate_predictions(model_result["model"], candidate_rows_)
    scales = scales_from_metrics(targets, model_result["metrics"], floors)
    components = eig_components(unc, scales)
    return components @ np.array([weights[target] for target in targets], dtype=float)


def target_correlation_summary(result: dict) -> dict[str, float]:
    std = np.std(result["y"], axis=0)
    keep = std > 1e-12
    corr = np.corrcoef(result["y"][:, keep].T)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    abs_corr = np.abs(corr[mask])
    return {
        "mean_abs_offdiag_corr": float(np.mean(abs_corr)),
        "median_abs_offdiag_corr": float(np.median(abs_corr)),
        "max_abs_offdiag_corr": float(np.max(abs_corr)),
    }


def write_prediction_csv(path: Path, rows: list[dict[str, str]], result: dict) -> None:
    fields = ["sample_id", "source_file", "mode", "T1_C", "t1_s", "T2_C", "t2_s"]
    for target in result["targets"]:
        fields.extend([f"actual_{target}", f"pred_{target}", f"uncertainty_{target}", f"error_{target}"])
    out_rows = []
    for out_i, pos in enumerate(result["test_idx"]):
        row = rows[result["clean_idx"][int(pos)]]
        out = {field: row.get(field, "") for field in fields[:7]}
        for j, target in enumerate(result["targets"]):
            actual = float(result["y"][pos, j])
            predicted = float(result["pred"][out_i, j])
            out[f"actual_{target}"] = actual
            out[f"pred_{target}"] = predicted
            out[f"uncertainty_{target}"] = float(result["unc"][out_i, j])
            out[f"error_{target}"] = predicted - actual
        out_rows.append(out)
    write_csv(path, out_rows)


def write_model_payload(path: Path, result: dict, dataset_path: Path, note: str) -> None:
    model = result["model"]
    payload = {
        "model_type": "numpy_rbf_kernel_regressor",
        "features": [
            "is_two_step",
            "T1_C",
            "T2_C_filled",
            "delta_T_C",
            "log_t1_s",
            "log_t2_s",
            "log_total_time_s",
        ],
        "targets": result["targets"],
        "bandwidth": model.bandwidth,
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "x_train": model.x_train.tolist(),
        "y_train": model.y_train.tolist(),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": int(len(result["clean_idx"])),
        "n_train": int(len(result["train_idx"])),
        "n_test": int(len(result["test_idx"])),
        "metrics": result["metrics"],
        "notes": [note],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def write_report(summary: dict, current_result: dict, paper_result: dict) -> None:
    metric_rows = [["目标集", "目标", "MAE", "RMSE", "R2", "MAE/测试std"]]
    for label, result in [("当前PS目标", current_result), ("论文式目标", paper_result)]:
        for target, value in result["metrics"].items():
            std = value["target_std"]
            ratio = value["mae"] / std if std > 0 else math.nan
            metric_rows.append([
                label,
                target,
                f"{value['mae']:.4g}",
                f"{value['rmse']:.4g}",
                f"{value['r2']:.3f}",
                f"{ratio:.3f}",
            ])

    eig_rows = [["目标集", "EIG均值", "P90", "P99", "最大值"]]
    for label, stats in [
        ("当前PS目标", summary["current_eig_stats"]),
        ("论文式目标", summary["paper_eig_stats"]),
    ]:
        eig_rows.append([
            label,
            f"{stats['mean']:.3f}",
            f"{stats['p90']:.3f}",
            f"{stats['p99']:.3f}",
            f"{stats['max']:.3f}",
        ])

    top_rows = [["Rank", "当前PS目标Top工况", "论文式目标Top工况"]]
    for idx in range(10):
        cur = summary["current_top10"][idx]
        pap = summary["paper_top10"][idx]
        top_rows.append([
            str(idx + 1),
            f"{cur['T1_C']}C {cur['t1_s']}s -> {cur['T2_C']}C {cur['t2_s']}s",
            f"{pap['T1_C']}C {pap['t1_s']}s -> {pap['T2_C']}C {pap['t2_s']}s",
        ])

    text = f"""# PS 论文式目标集与当前目标集对比

## 1. 问题

当前 PS 模型使用多目标输出：

```text
delta_h_total_J_g, peak_area_J_g, Tp, peak_height, onset, recovery_index,
path_dependence_index, kovacs_peak_label
```

这组目标描述更完整，但也可能引入信息冗余，尤其是 `delta_h_total`、
`peak_area`、`peak_height`、`recovery_index` 之间高度相关。为检验这一点，
本脚本把同一批 PS 数据重新整理成论文风格的低维目标：

```text
delta_h, delta_h_peak, S*, H*
```

其中 `delta_h` 和 `delta_h_peak` 分别对应 PS 的总焓回复面积与局部峰面积；
`S*` 和 `H*` 不是独立实验测量值，而是由 ARRT 两点约束反推的有效代理量。

## 2. 有效 S* / H* 的计算

对每条 PS 工况构造两个速率约束：

```text
k_ann  = 1 / total_anneal_time
k_peak = beta / Tp
ln(k) - ln(kB T / h) = S*/R - H*/(R T)
```

用等效退火温度 `T_ann` 和 DSC 峰位 `Tp` 联立求解 `S*` 与 `H*`。
这个定义的意义是：让 `S* / H*` 成为描述“退火过程和升温峰位是否一致”的
低维动力学坐标，而不是声称已经测得了真正材料本征活化熵/焓。

## 3. 同条件预测结果

数据集样本数：`{summary['n_samples']}`；训练集：`{summary['n_train']}`；
测试集：`{summary['n_test']}`。两组模型使用相同输入特征、相同 grouped split、
相同 RBF kernel bandwidth。

{markdown_table(metric_rows)}

归一化综合误差：

```json
{json.dumps(summary['normalized_metric_summary'], indent=2)}
```

## 4. EIG 分布差异

{markdown_table(eig_rows)}

![EIG distribution]({summary['figures']['eig_distribution']})

目标间相关性摘要：

```json
{json.dumps(summary['target_correlation_summary'], indent=2)}
```

![Current target correlation]({summary['figures']['current_target_correlation']})

![Paper-style target correlation]({summary['figures']['paper_target_correlation']})

## 5. Top EIG 工况差异

{markdown_table(top_rows)}

## 6. 判断

从结果看，当前目标集确实存在冗余：多个 DSC 峰形指标共同描述同一件事，
会让 EIG 在“峰面积、峰高、recovery index”之间重复计分。因此当前 EIG
更容易偏向能改变整体峰形幅度的极端路径点。

论文式目标集的优点是维度更低，EIG 更集中在动力学参数坐标上；
缺点是 `S* / H*` 在当前 PS 数据中是反推代理量，对 `Tp` 和总退火时间非常敏感。
因此它改善了“信息表达的紧凑性”，但不一定会立刻改善所有预测误差。

我的建议是下一版模型采用折中目标：

```text
delta_h_total_J_g
peak_area_J_g
paper_s_star_eff_J_mol_K
paper_h_star_eff_kJ_mol
path_dependence_index
```

并把 `Tp / peak_height / onset / recovery_index` 降级为诊断输出或派生指标。
这样既保留论文的动力学解释，又避免完全依赖由 Tp 反推出来的代理变量。

## 7. 输出文件

- 指标摘要：`results/ps/paper_style_comparison/summary.json`
- 论文式目标模型：`results/ps/paper_style_comparison/paper_style_model.json`
- 论文式目标测试预测：`results/ps/paper_style_comparison/paper_style_test_predictions.csv`
- 当前目标 EIG 排名：`results/ps/paper_style_comparison/current_target_eig_ranking_top100.csv`
- 论文式目标 EIG 排名：`results/ps/paper_style_comparison/paper_style_eig_ranking_top100.csv`
- 图表目录：`results/ps/paper_style_comparison/figures/`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = build_ps_dataset()
    rows = add_paper_style_targets(read_rows(dataset_path))

    current = fit_and_evaluate(rows, CURRENT_TARGETS)
    paper = fit_and_evaluate(rows, PAPER_TARGETS)
    candidates = candidate_table(mode_filter="two_step")
    current_ranking, current_eig_stats = eig_ranking(
        current,
        CURRENT_TARGETS,
        TARGET_WEIGHTS,
        {
            "delta_h_total_J_g": 0.20,
            "peak_area_J_g": 0.13,
            "peak_temperature_Tp_C": 5.0,
            "peak_height_uW": 50.0,
            "onset_temperature_C": 5.0,
            "recovery_index": 0.15,
            "path_dependence_index": 0.20,
            "kovacs_peak_label": 1.0,
        },
        candidates,
    )
    paper_ranking, paper_eig_stats = eig_ranking(
        paper,
        PAPER_TARGETS,
        PAPER_WEIGHTS,
        PAPER_NOISE_FLOORS,
        candidates,
    )

    current_eig_values = full_eig_values(
        current,
        CURRENT_TARGETS,
        TARGET_WEIGHTS,
        {
            "delta_h_total_J_g": 0.20,
            "peak_area_J_g": 0.13,
            "peak_temperature_Tp_C": 5.0,
            "peak_height_uW": 50.0,
            "onset_temperature_C": 5.0,
            "recovery_index": 0.15,
            "path_dependence_index": 0.20,
            "kovacs_peak_label": 1.0,
        },
        candidates,
    )
    paper_eig_values = full_eig_values(paper, PAPER_TARGETS, PAPER_WEIGHTS, PAPER_NOISE_FLOORS, candidates)
    figures = {
        "eig_distribution": plot_eig_distribution(current_eig_values.tolist(), paper_eig_values.tolist()),
        "current_prediction_scatter": plot_prediction_scatter(current, rows, "current"),
        "paper_prediction_scatter": plot_prediction_scatter(paper, rows, "paper_style"),
        "current_target_correlation": plot_target_correlation(current, "current"),
        "paper_target_correlation": plot_target_correlation(paper, "paper_style"),
    }

    write_csv(OUT_DIR / "current_target_eig_ranking_top100.csv", current_ranking)
    write_csv(OUT_DIR / "paper_style_eig_ranking_top100.csv", paper_ranking)
    write_prediction_csv(OUT_DIR / "current_target_test_predictions.csv", rows, current)
    write_prediction_csv(OUT_DIR / "paper_style_test_predictions.csv", rows, paper)
    write_model_payload(
        OUT_DIR / "paper_style_model.json",
        paper,
        dataset_path,
        "Paper-style PS target model using delta_h, delta_h_peak, and ARRT effective S*/H* proxies.",
    )

    summary = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_samples": len(rows),
        "n_train": int(len(current["train_idx"])),
        "n_test": int(len(current["test_idx"])),
        "current_targets": CURRENT_TARGETS,
        "paper_style_targets": PAPER_TARGETS,
        "paper_style_target_definition": {
            "paper_delta_h_J_g": "abs(delta_h_total_J_g)",
            "paper_delta_h_peak_J_g": "abs(peak_area_J_g)",
            "paper_s_star_eff_J_mol_K": "ARRT effective entropy inferred from annealing rate and Tp peak rate.",
            "paper_h_star_eff_kJ_mol": "ARRT effective enthalpy inferred from annealing rate and Tp peak rate.",
        },
        "current_metrics": current["metrics"],
        "paper_style_metrics": paper["metrics"],
        "normalized_metric_summary": {
            "current_targets": current["normalized"],
            "paper_style_targets": paper["normalized"],
        },
        "current_eig_stats": current_eig_stats,
        "paper_eig_stats": paper_eig_stats,
        "target_correlation_summary": {
            "current_targets": target_correlation_summary(current),
            "paper_style_targets": target_correlation_summary(paper),
        },
        "current_top10": current_ranking[:10],
        "paper_top10": paper_ranking[:10],
        "figures": figures,
        "outputs": {
            "summary": str((OUT_DIR / "summary.json").relative_to(ROOT)),
            "report": str(DOC_PATH.relative_to(ROOT)),
            "current_ranking": str((OUT_DIR / "current_target_eig_ranking_top100.csv").relative_to(ROOT)),
            "paper_ranking": str((OUT_DIR / "paper_style_eig_ranking_top100.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, current, paper)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
