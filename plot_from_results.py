#!/usr/bin/env python3
"""
仅绘图脚本：从 results/*/metrics_and_curves.json 读取评测结果并生成图表。

输出：
- results/2/bar_metrics.png
- results/4/bar_metrics.png
- results/8/bar_metrics.png
- results/2/roc_curves_by_dataset.png
- results/4/roc_curves_by_dataset.png
- results/8/roc_curves_by_dataset.png
- results/dataset_grouped_auroc.png
- results/dataset_grouped_ap.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SHOTS = [2, 4, 8]
DATASETS = ["aitex", "elpv", "visa"]
DATASET_NAMES = ["AITEX", "ELPV", "VISA"]


def load_results(results_root: Path, shots):
    all_results = {}
    for shot in shots:
        shot_dir = results_root / str(shot)
        shot_json = shot_dir / "metrics_and_curves.json"
        if not shot_json.exists():
            raise FileNotFoundError(f"缺少结果文件: {shot_json}")

        with open(shot_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "results" not in payload:
            raise ValueError(f"JSON 格式错误，缺少 'results': {shot_json}")

        all_results[str(shot)] = payload["results"]
    return all_results


def plot_per_shot_bars(shot, shot_results, out_dir: Path):
    x = np.arange(len(DATASETS))
    width = 0.36

    baseline_auroc = [shot_results["Baseline"][d]["auroc"] or 0.0 for d in DATASETS]
    va_auroc = [shot_results["VA"][d]["auroc"] or 0.0 for d in DATASETS]
    baseline_ap = [shot_results["Baseline"][d]["ap"] or 0.0 for d in DATASETS]
    va_ap = [shot_results["VA"][d]["ap"] or 0.0 for d in DATASETS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)

    axes[0].bar(x - width / 2, baseline_auroc, width, label="Baseline")
    axes[0].bar(x + width / 2, va_auroc, width, label="VA")
    axes[0].set_title(f"{shot}-shot AUROC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(DATASET_NAMES)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, baseline_ap, width, label="Baseline")
    axes[1].bar(x + width / 2, va_ap, width, label="VA")
    axes[1].set_title(f"{shot}-shot AP")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DATASET_NAMES)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "bar_metrics.png")
    plt.close(fig)


def plot_per_shot_roc_curves(shot, shot_results, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for variant in ["Baseline", "VA"]:
            fpr = shot_results[variant][ds]["fpr"]
            tpr = shot_results[variant][ds]["tpr"]
            auroc = shot_results[variant][ds]["auroc"]
            if fpr is None or tpr is None or auroc is None:
                continue
            ax.plot(fpr, tpr, lw=2, label=f"{variant} (AUC={auroc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{DATASET_NAMES[idx]} | {shot}-shot")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves_by_dataset.png")
    plt.close(fig)


def plot_grouped_by_dataset_across_shots(all_results, metric_key, out_file: Path, shots):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)
    x = np.arange(len(shots))
    width = 0.36

    for i, ds in enumerate(DATASETS):
        baseline_vals = []
        va_vals = []
        for shot in shots:
            shot_key = str(shot)
            baseline_vals.append(all_results[shot_key]["Baseline"][ds][metric_key] or 0.0)
            va_vals.append(all_results[shot_key]["VA"][ds][metric_key] or 0.0)

        ax = axes[i]
        ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
        ax.bar(x + width / 2, va_vals, width, label="VA")
        ax.set_title(DATASET_NAMES[i])
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel("Shot")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    metric_title = "AUROC" if metric_key == "auroc" else "AP"
    fig.suptitle(f"Grouped by Dataset ({metric_title})", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot charts from cached evaluation JSON files.")
    parser.add_argument("--results-root", type=str, default="results", help="Path to results root folder.")
    parser.add_argument("--shots", type=str, default="2,4,8", help="Comma-separated shot values, e.g. 2,4,8")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    shots = [int(s.strip()) for s in args.shots.split(",") if s.strip()]

    all_results = load_results(results_root, shots)

    for shot in shots:
        shot_key = str(shot)
        shot_dir = results_root / shot_key
        shot_dir.mkdir(parents=True, exist_ok=True)

        plot_per_shot_bars(shot, all_results[shot_key], shot_dir)
        plot_per_shot_roc_curves(shot, all_results[shot_key], shot_dir)
        print(f"[INFO] 已生成 {shot}-shot 图表到: {shot_dir}")

    plot_grouped_by_dataset_across_shots(
        all_results,
        metric_key="auroc",
        out_file=results_root / "dataset_grouped_auroc.png",
        shots=shots,
    )
    plot_grouped_by_dataset_across_shots(
        all_results,
        metric_key="ap",
        out_file=results_root / "dataset_grouped_ap.png",
        shots=shots,
    )

    print(f"[INFO] 已生成全局分组图到: {results_root}")


if __name__ == "__main__":
    main()
