#!/usr/bin/env python3
"""
Generate pixel-level anomaly heatmap visualization.

Produces a figure like InCTRL Figure 3:
  - Left column: few-shot normal image prompts
  - Top row: query images (green border=normal, red=anomalous)
  - Bottom row: holistic in-context residual maps (15x15 patch grid)

Also generates an upsampled final_map overlay version.

Usage:
  python tools/visualize_anomaly_map.py \
    --checkpoint results/mvtec_va_ta_pqa_2shot_15ep/checkpoint_best.pyth \
    --dataset aitex --shot 2 --n_examples 6

  # Or specify a specific category for multi-category datasets:
  python tools/visualize_anomaly_map.py \
    --checkpoint results/mvtec_va_ta_pqa_4shot_15ep/checkpoint_best.pyth \
    --dataset visa --category pcb1 --shot 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from open_clip.config.defaults import assert_and_infer_cfg, get_cfg

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_CATEGORIES = {
    "aitex": ["AITEX"],
    "elpv": ["elpv"],
    "visa": [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    ],
    "mvtec": [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
}

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def build_cfg(shot: int, device: str):
    cfg = get_cfg()
    cfg.shot = shot
    cfg.image_size = 240
    cfg.NUM_GPUS = 1 if device == "cuda" else 0
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.TRAIN.SHOW_PROGRESS = False
    cfg = assert_and_infer_cfg(cfg)
    return cfg


def build_transform(image_size=240):
    def _convert_to_rgb(image):
        return image.convert("RGB")
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(_convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_model(cfg, checkpoint_path: str, device: str):
    from engine_IC import _build_active_model
    from open_clip.model import get_cast_dtype

    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = json.load(f)

    cast_dtype = get_cast_dtype("fp32")
    model = _build_active_model(cfg, model_cfg, cast_dtype, quick_gelu=False)

    ckpt_path = PROJECT_ROOT / checkpoint_path
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized [C, H, W] tensor back to displayable [H, W, 3] uint8."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def run_inference(model, tokenizer, query_tensor, prompt_tensors, device):
    """Run model on a single sample. Returns outputs dict."""
    query_image = query_tensor.unsqueeze(0).to(device)       # [1, C, H, W]
    prompt_images = prompt_tensors.unsqueeze(0).to(device)    # [1, shot, C, H, W]

    outputs = model(
        tokenizer=tokenizer,
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=None,
        return_aux=True,
        return_dict=True,
    )
    return outputs


def classify_sample(label: int, score: float, threshold: float = 0.5):
    """Return (type_str, border_color) for a sample."""
    predicted_anomaly = score >= threshold
    if label == 0 and not predicted_anomaly:
        return "True Negative", "#4CAF50"
    elif label == 0 and predicted_anomaly:
        return "False Positive", "#FF9800"
    elif label == 1 and predicted_anomaly:
        return "True Positive", "#F44336"
    else:
        return "False Negative", "#2196F3"


# ── Plotting ─────────────────────────────────────────────────────────────────

RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual",
    ["#FFFFCC", "#A1DAB4", "#41B6C4", "#2C7FB8", "#253494"],
    N=256,
)


def plot_residual_figure(
    prompt_images_display: list[np.ndarray],
    query_images_display: list[np.ndarray],
    residual_maps: list[np.ndarray],
    sample_types: list[str],
    border_colors: list[str],
    scores: list[float],
    title: str,
    save_path: str,
):
    """
    Plot the InCTRL-style figure:
      - Left column: prompt images
      - Top row: query images with colored borders
      - Bottom row: residual maps (15x15)
    """
    n_prompts = len(prompt_images_display)
    n_queries = len(query_images_display)

    fig, axes = plt.subplots(
        2, n_queries + 1,
        figsize=(2.5 + 3.2 * n_queries, 7.5),
        gridspec_kw={"width_ratios": [1.0] + [2.0] * n_queries,
                     "hspace": 0.25, "wspace": 0.15},
    )

    # ── Left column: prompt images ──
    ax = axes[0, 0]
    if n_prompts == 1:
        ax.imshow(prompt_images_display[0])
    else:
        # Stack prompts vertically
        stacked = np.concatenate(prompt_images_display, axis=0)
        ax.imshow(stacked)
    ax.set_title(f"{n_prompts}-shot Normal\nImage Prompts", fontsize=9, fontweight="bold")
    ax.axis("off")
    axes[1, 0].axis("off")

    # ── Top row: query images ──
    for j in range(n_queries):
        ax = axes[0, j + 1]
        ax.imshow(query_images_display[j])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[j])
            spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{sample_types[j]}\nscore={scores[j]:.3f}",
                      fontsize=9, fontweight="bold", color=border_colors[j])

    # ── Bottom row: residual maps ──
    all_vals = np.concatenate([rm.flatten() for rm in residual_maps])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))

    for j in range(n_queries):
        ax = axes[1, j + 1]
        rm = residual_maps[j]
        grid_size = rm.shape[0]
        im = ax.imshow(rm, cmap=RESIDUAL_CMAP, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels(range(grid_size), fontsize=5)
        ax.set_yticklabels(range(grid_size - 1, -1, -1), fontsize=5)
        ax.tick_params(length=0)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax, label="Residual Value")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_overlay_figure(
    query_images_display: list[np.ndarray],
    final_maps: list[np.ndarray],
    sample_types: list[str],
    border_colors: list[str],
    scores: list[float],
    title: str,
    save_path: str,
):
    """Plot query images with final_map heatmap overlay (upsampled to 240x240)."""
    n = len(query_images_display)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j in range(n):
        # Top: original image
        ax = axes[0, j]
        ax.imshow(query_images_display[j])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[j])
            spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{sample_types[j]}\nscore={scores[j]:.3f}", fontsize=9,
                      fontweight="bold", color=border_colors[j])

        # Bottom: overlay
        ax = axes[1, j]
        ax.imshow(query_images_display[j])
        fmap = final_maps[j]
        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax.imshow(fmap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("Query Image", fontsize=11)
    axes[1, 0].set_ylabel("Anomaly Map\nOverlay", fontsize=11)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anomaly heatmap visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CATEGORIES.keys()))
    parser.add_argument("--category", default=None, help="Specific category (default: first)")
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--n_examples", type=int, default=6,
                        help="Number of query examples to show")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    category = args.category or DATASET_CATEGORIES[args.dataset][0]
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "reports" / "anomaly_maps" / f"{args.dataset}_{category}_{args.shot}shot"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ──
    cfg = build_cfg(args.shot, device)
    transform = build_transform(240)

    print(f"Loading model: {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, device)

    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")

    # ── Build data loader ──
    json_dir = os.path.join("data", "AD_json", args.dataset.lower())
    cat_cfg = cfg.clone()
    cat_cfg.val_normal_json_path = [os.path.join(json_dir, f"{category}_val_normal.json")]
    cat_cfg.val_outlier_json_path = [os.path.join(json_dir, f"{category}_val_outlier.json")]
    cat_cfg.FEW_SHOT_SEED = args.seed

    from datasets import loader
    test_loader = loader.construct_loader(cat_cfg, "test", transform)

    print(f"Dataset: {args.dataset}/{category}, shot={args.shot}, seed={args.seed}")
    print(f"Samples in test set: {len(test_loader.dataset)}")

    # ── Run inference on all samples, collect results ──
    all_results = []
    for batch_idx, batch in enumerate(test_loader):
        if len(batch) >= 4:
            inputs, types, labels, masks = batch[0], batch[1], batch[2], batch[3]
        else:
            inputs, types, labels = batch[0], batch[1], batch[2]
            masks = None

        # inputs: list of [B, C, H, W] — inputs[0]=query, inputs[1:]=prompts
        if isinstance(inputs, (list, tuple)):
            query_tensor = inputs[0]
            prompt_list = inputs[1:]
            prompt_tensors = torch.stack(prompt_list, dim=1)
        elif isinstance(inputs, torch.Tensor) and inputs.dim() == 5:
            query_tensor = inputs[0]
            prompt_tensors = inputs[1:].permute(1, 0, 2, 3, 4)
        else:
            query_tensor = inputs
            prompt_tensors = None

        if prompt_tensors is None:
            continue

        for i in range(query_tensor.shape[0]):
            q = query_tensor[i]
            p = prompt_tensors[i]
            label = int(labels[i].item())

            outputs = run_inference(model, tokenizer, q, p, device)

            score = float(outputs["coupled_score"][0].cpu())
            residual_map = outputs["patch_residual_map"][0].cpu().numpy()
            final_map = outputs["final_map"][0, 0].cpu().numpy()

            grid = int(math.sqrt(residual_map.shape[0]))
            residual_grid = residual_map.reshape(grid, grid)

            stype, bcolor = classify_sample(label, score)

            all_results.append({
                "query_display": tensor_to_display(q),
                "prompt_displays": [tensor_to_display(p[s]) for s in range(p.shape[0])],
                "residual_grid": residual_grid,
                "final_map": final_map,
                "label": label,
                "score": score,
                "type": stype,
                "border_color": bcolor,
            })

    print(f"Collected {len(all_results)} results")

    # ── Select diverse examples ──
    by_type = {}
    for r in all_results:
        by_type.setdefault(r["type"], []).append(r)

    print("Distribution:", {k: len(v) for k, v in by_type.items()})

    selected = []
    for t in ["True Positive", "False Positive", "True Negative", "False Negative"]:
        if t in by_type:
            candidates = sorted(by_type[t], key=lambda r: r["score"],
                                reverse=(t in ["True Positive", "False Positive"]))
            for c in candidates[:max(1, args.n_examples // 4)]:
                if len(selected) < args.n_examples:
                    selected.append(c)

    rng = np.random.default_rng(args.seed)
    remaining = [r for r in all_results if r not in selected]
    while len(selected) < args.n_examples and remaining:
        idx = rng.integers(0, len(remaining))
        selected.append(remaining.pop(idx))

    n = len(selected)
    print(f"Selected {n} examples: {[s['type'] for s in selected]}")

    # ── Plot 1: InCTRL-style residual figure ──
    prompt_displays = selected[0]["prompt_displays"]
    plot_residual_figure(
        prompt_images_display=prompt_displays,
        query_images_display=[s["query_display"] for s in selected],
        residual_maps=[s["residual_grid"] for s in selected],
        sample_types=[s["type"] for s in selected],
        border_colors=[s["border_color"] for s in selected],
        scores=[s["score"] for s in selected],
        title=f"{args.dataset.upper()} / {category} — {args.shot}-shot Residual Map",
        save_path=str(output_dir / "residual_map.png"),
    )

    # ── Plot 2: Overlay heatmap ──
    plot_overlay_figure(
        query_images_display=[s["query_display"] for s in selected],
        final_maps=[s["final_map"] for s in selected],
        sample_types=[s["type"] for s in selected],
        border_colors=[s["border_color"] for s in selected],
        scores=[s["score"] for s in selected],
        title=f"{args.dataset.upper()} / {category} — {args.shot}-shot Anomaly Map Overlay",
        save_path=str(output_dir / "anomaly_overlay.png"),
    )

    # ── Plot 3: Individual high-res maps for each sample ──
    for idx, s in enumerate(selected):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original
        axes[0].imshow(s["query_display"])
        for spine in axes[0].spines.values():
            spine.set_edgecolor(s["border_color"])
            spine.set_linewidth(3)
        axes[0].set_title(f"Query ({s['type']})\nscore={s['score']:.3f}", fontsize=10)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Residual grid
        axes[1].imshow(s["residual_grid"], cmap=RESIDUAL_CMAP, origin="upper")
        axes[1].set_title("Patch Residual Map", fontsize=10)
        grid = s["residual_grid"].shape[0]
        axes[1].set_xticks(range(grid))
        axes[1].set_yticks(range(grid))
        axes[1].set_xticklabels(range(grid), fontsize=5)
        axes[1].set_yticklabels(range(grid), fontsize=5)

        # Overlay
        axes[2].imshow(s["query_display"])
        fmap = s["final_map"]
        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        axes[2].imshow(fmap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("Final Anomaly Map Overlay", fontsize=10)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.suptitle(f"Sample {idx} — label={'anomaly' if s['label'] else 'normal'}  "
                     f"pred={s['score']:.3f}", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(output_dir / f"sample_{idx}_{s['type'].replace(' ', '_').lower()}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll figures saved to {output_dir}/")
    print(f"  - residual_map.png       (InCTRL-style grid)")
    print(f"  - anomaly_overlay.png    (upsampled heatmap overlay)")
    print(f"  - sample_N_*.png         (individual high-res per sample)")


if __name__ == "__main__":
    main()
