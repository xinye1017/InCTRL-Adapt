#!/usr/bin/env python3
"""
Visualize multi-seed evaluation results.

Reads from the extracted tar.gz data and generates:
  1. AUROC vs shot (line + error band) — per dataset, with baseline
  2. Per-category heatmap for VisA (mean across seeds)
  3. Seed variance scatter — which categories are most sensitive to sampling
  4. Delta vs baseline bar chart — all models at a glance
  5. Per-category box plot for VisA across shots

Output: reports/multi_seed_figures/
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────

SUMMARY_CSV = "/tmp/multi_seed_eval/multi_seed/summary.csv"
DETAIL_CSV = "/tmp/multi_seed_eval/multi_seed/all_models_multi_seed.csv"
OUT_DIR = Path("/Users/xinye/Desktop/InCTRL/reports/multi_seed_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "elpv": {0: 0.733, 2: 0.839, 4: 0.846, 8: 0.872},
    "aitex": {0: 0.733, 2: 0.761, 4: 0.790, 8: 0.806},
    "visa": {0: 0.781, 2: 0.858, 4: 0.877, 8: 0.887},
    "mvtec": {0: 0.912, 2: 0.940, 4: 0.945, 8: 0.953},
}

def load_summary():
    rows = []
    with open(SUMMARY_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["shot"] = int(r["shot"])
            r["auroc_mean"] = float(r["auroc_mean"])
            r["auroc_std"] = float(r["auroc_std"])
            r["aupr_mean"] = float(r["aupr_mean"])
            r["aupr_std"] = float(r["aupr_std"])
            r["baseline_auroc"] = float(r["baseline_auroc"]) if r["baseline_auroc"] else None
            r["delta_vs_baseline"] = float(r["delta_vs_baseline"]) if r["delta_vs_baseline"] else None
            rows.append(r)
    return rows

def load_detail():
    rows = []
    with open(DETAIL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["shot"] = int(r["shot"])
            r["seed"] = int(r["seed"])
            r["auroc"] = float(r["auroc"])
            r["aupr"] = float(r["aupr"])
            rows.append(r)
    return rows


# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "visa": "#2196F3",
    "aitex": "#FF5722",
    "elpv": "#4CAF50",
    "mvtec": "#9C27B0",
}
DATASET_LABELS = {"visa": "VisA", "aitex": "AITEX", "elpv": "ELPV", "mvtec": "MVTec AD"}


# ── Figure 1: AUROC vs Shot (line + error band + baseline) ──────────────────

def fig1_auroc_vs_shot(summary):
    # Group: train on MVTec → test on visa/aitex/elpv
    mvtec_rows = [r for r in summary if r["train_dataset"] == "mvtec"]
    # Group: train on VisA → test on mvtec
    visa_rows = [r for r in summary if r["train_dataset"] == "visa"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # ── Left panel: train on MVTec ──
    ax = axes[0]
    for ds in ["visa", "aitex", "elpv"]:
        ds_rows = sorted([r for r in mvtec_rows if r["test_dataset"] == ds], key=lambda r: r["shot"])
        shots = [r["shot"] for r in ds_rows]
        means = [r["auroc_mean"] for r in ds_rows]
        stds = [r["auroc_std"] for r in ds_rows]
        color = COLORS[ds]
        label = DATASET_LABELS[ds]

        ax.plot(shots, means, "o-", color=color, label=f"Ours → {label}", linewidth=2, markersize=7, zorder=5)
        ax.fill_between(shots,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

        # Baseline
        bl_shots = sorted(BASELINE[ds].keys())
        bl_vals = [BASELINE[ds][s] for s in bl_shots]
        ax.plot(bl_shots, bl_vals, "s--", color=color, alpha=0.45, markersize=5,
                label=f"InCTRL baseline → {label}")

    ax.set_xlabel("Shot")
    ax.set_ylabel("AUROC")
    ax.set_title("Train on MVTec AD → Cross-domain Test")
    ax.set_xticks([0, 2, 4, 8])
    ax.set_ylim(0.70, 0.95)
    ax.legend(fontsize=8.5, ncol=2, loc="lower right")

    # ── Right panel: train on VisA ──
    ax = axes[1]
    ds = "mvtec"
    ds_rows = sorted(visa_rows, key=lambda r: r["shot"])
    shots = [r["shot"] for r in ds_rows]
    means = [r["auroc_mean"] for r in ds_rows]
    stds = [r["auroc_std"] for r in ds_rows]
    color = COLORS[ds]
    label = DATASET_LABELS[ds]

    ax.plot(shots, means, "o-", color=color, label=f"Ours → {label}", linewidth=2, markersize=7, zorder=5)
    ax.fill_between(shots,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color=color)

    bl_shots = sorted(BASELINE[ds].keys())
    bl_vals = [BASELINE[ds][s] for s in bl_shots]
    ax.plot(bl_shots, bl_vals, "s--", color=color, alpha=0.45, markersize=5,
            label=f"InCTRL baseline → {label}")

    ax.set_xlabel("Shot")
    ax.set_ylabel("AUROC")
    ax.set_title("Train on VisA → Cross-domain Test")
    ax.set_xticks([0, 2, 4, 8])
    ax.set_ylim(0.92, 0.99)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Multi-Seed AUROC vs Shot Count (mean ± std, 3 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig1_auroc_vs_shot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 2: Per-category heatmap for VisA (mean across seeds) ─────────────

def fig2_visa_heatmap(detail):
    visa_rows = [r for r in detail if r["test_dataset"] == "visa"]
    categories = sorted(set(r["category"] for r in visa_rows))
    shots = sorted(set(r["shot"] for r in visa_rows))

    # Build mean matrix
    matrix = np.zeros((len(categories), len(shots)))
    for i, cat in enumerate(categories):
        for j, shot in enumerate(shots):
            vals = [r["auroc"] for r in visa_rows if r["category"] == cat and r["shot"] == shot]
            matrix[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.75, vmax=1.0)

    ax.set_xticks(range(len(shots)))
    ax.set_xticklabels([f"{s}-shot" for s in shots])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(shots)):
            val = matrix[i, j]
            color = "white" if val > 0.92 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("VisA Per-Category AUROC (mean over 3 seeds)\nTrain on MVTec AD", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.6, label="AUROC")
    fig.tight_layout()
    path = OUT_DIR / "fig2_visa_category_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 3: Seed variance scatter — which categories are most sensitive ───

def fig3_seed_sensitivity(detail):
    """Scatter: x=mean AUROC, y=std across seeds, per category per shot."""
    visa_rows = [r for r in detail if r["test_dataset"] == "visa"]
    categories = sorted(set(r["category"] for r in visa_rows))
    shots = sorted(set(r["shot"] for r in visa_rows))

    fig, ax = plt.subplots(figsize=(9, 6))
    shot_markers = {2: "o", 4: "s", 8: "D"}
    shot_colors = {2: "#1976D2", 4: "#F57C00", 8: "#388E3C"}

    for shot in shots:
        means_list = []
        stds_list = []
        labels_list = []
        for cat in categories:
            vals = [r["auroc"] for r in visa_rows if r["category"] == cat and r["shot"] == shot]
            if vals:
                means_list.append(np.mean(vals))
                stds_list.append(np.std(vals))
                labels_list.append(cat)

        ax.scatter(means_list, stds_list,
                   marker=shot_markers[shot], color=shot_colors[shot],
                   s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
                   label=f"{shot}-shot")

        # Annotate outliers (std > 0.01)
        for m, s, lbl in zip(means_list, stds_list, labels_list):
            if s > 0.01:
                ax.annotate(lbl, (m, s), fontsize=7.5, alpha=0.8,
                            xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Mean AUROC (across 3 seeds)")
    ax.set_ylabel("Std AUROC (across 3 seeds)")
    ax.set_title("VisA: Seed Sensitivity per Category\n(higher std = more sensitive to few-shot sampling)")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "fig3_seed_sensitivity_visa.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 4: Delta vs baseline — all models bar chart ──────────────────────

def fig4_delta_vs_baseline(summary):
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = []
    deltas = []
    colors = []
    errs = []

    for r in summary:
        ds = r["test_dataset"]
        label = f"{r['train_dataset'].upper()}→{DATASET_LABELS[ds]} {r['shot']}s"
        labels.append(label)
        deltas.append(r["delta_vs_baseline"] if r["delta_vs_baseline"] is not None else 0)
        errs.append(r["auroc_std"])
        colors.append(COLORS[ds])

    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, deltas, yerr=errs, fmt="none", ecolor="black", capsize=3, capthick=1, elinewidth=1)

    ax.axhline(y=0, color="gray", linewidth=1, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ΔAUROC vs InCTRL Baseline")
    ax.set_title("InCTRLAdapt vs InCTRL Baseline (mean ± std over 3 seeds)")

    # Annotate bars
    for bar, d, e in zip(bars, deltas, errs):
        sign = "+" if d >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, d + (0.002 if d >= 0 else -0.004),
                f"{sign}{d:.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "fig4_delta_vs_baseline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 5: Per-category box plot for MVTec (train=VisA) ──────────────────

def fig5_mvtec_category_boxplot(detail):
    mvtec_rows = [r for r in detail if r["test_dataset"] == "mvtec"]
    categories = sorted(set(r["category"] for r in mvtec_rows))
    shots = sorted(set(r["shot"] for r in mvtec_rows))

    fig, axes = plt.subplots(1, len(shots), figsize=(5.5 * len(shots), 6), sharey=True)
    if len(shots) == 1:
        axes = [axes]

    for ax, shot in zip(axes, shots):
        data = []
        cat_labels = []
        for cat in categories:
            vals = [r["auroc"] for r in mvtec_rows if r["category"] == cat and r["shot"] == shot]
            if vals:
                data.append(vals)
                cat_labels.append(cat)

        bp = ax.boxplot(data, tick_labels=cat_labels, patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor=COLORS["mvtec"], alpha=0.4),
                        medianprops=dict(color=COLORS["mvtec"], linewidth=2))

        # Overlay individual seed points
        for i, vals in enumerate(data):
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
            ax.scatter([i + 1 + j for j in jitter], vals, color=COLORS["mvtec"],
                       s=30, alpha=0.7, zorder=5, edgecolors="white", linewidth=0.5)

        ax.set_xticklabels(cat_labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"MVTec AD {shot}-shot", fontsize=11)
        ax.set_ylabel("AUROC" if ax == axes[0] else "")
        ax.set_ylim(0.6, 1.02)

    fig.suptitle("MVTec AD Per-Category AUROC (Train on VisA, 3 seeds)", fontsize=13, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "fig5_mvtec_category_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Figure 6: AUROC comparison table as figure ──────────────────────────────

def fig6_summary_table(summary):
    """Render summary as a clean table figure for quick reference."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    headers = ["Model", "Test Dataset", "Shot", "AUROC (mean±std)", "Baseline", "Δ"]
    cell_data = []
    cell_colors = []

    for r in summary:
        auroc_str = f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}"
        bl_str = f"{r['baseline_auroc']:.3f}" if r['baseline_auroc'] else "—"
        delta = r['delta_vs_baseline']
        delta_str = f"{delta:+.4f}" if delta is not None else "—"
        cell_data.append([
            r["model"].replace("_va_ta_pqa_", " "),
            DATASET_LABELS.get(r["test_dataset"], r["test_dataset"]),
            str(r["shot"]),
            auroc_str,
            bl_str,
            delta_str,
        ])
        if delta is not None:
            row_color = "#C8E6C9" if delta > 0 else "#FFCDD2"
        else:
            row_color = "white"
        cell_colors.append([row_color] * len(headers))

    table = ax.table(cellText=cell_data, colLabels=headers, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Multi-Seed Evaluation Summary (3 seeds: 42, 123, 7)", fontsize=13, pad=20)
    fig.tight_layout()
    path = OUT_DIR / "fig6_summary_table.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    summary = load_summary()
    detail = load_detail()

    print(f"Loaded {len(summary)} summary rows, {len(detail)} detail rows\n")

    fig1_auroc_vs_shot(summary)
    fig2_visa_heatmap(detail)
    fig3_seed_sensitivity(detail)
    fig4_delta_vs_baseline(summary)
    fig5_mvtec_category_boxplot(detail)
    fig6_summary_table(summary)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
