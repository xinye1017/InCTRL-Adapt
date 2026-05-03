#!/usr/bin/env python3
"""Shared inference, selection, manifest, and plotting helpers for no-VA visualization."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(os.environ.get("INCTRL_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))

from open_clip.config.defaults import assert_and_infer_cfg, get_cfg

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
THRESHOLD = 0.5

RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual",
    ["#FFFFCC", "#A1DAB4", "#41B6C4", "#2C7FB8", "#253494"],
    N=256,
)


@dataclass
class GallerySample:
    model_name: str
    train_dataset: str
    test_dataset: str
    category: str
    shot: int
    seed: int
    sample_index: int
    query_path: str
    prompt_paths: list[str]
    label: int
    score: float
    prediction: int
    case_type: str
    border_color: str
    query_display: Any
    prompt_displays: list[Any]
    residual_grid: Any
    final_map: Any


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_output_dir(output_dir: str | Path) -> Path:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def safe_filename(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", value).strip("_") or "sample"


def build_cfg(shot: int, device: str):
    cfg = get_cfg()
    cfg.shot = int(shot)
    cfg.image_size = 240
    cfg.NUM_GPUS = 1 if device == "cuda" else 0
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.TRAIN.SHOW_PROGRESS = False
    return assert_and_infer_cfg(cfg)


def build_transform(image_size: int = 240):
    def _convert_to_rgb(image):
        return image.convert("RGB")

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_model(cfg, checkpoint_path: str, device: str):
    """Load the active InCTRL model. Checkpoint paths are loaded directly, not pre-validated."""
    from engine_IC import _build_active_model
    from open_clip.model import get_cast_dtype

    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as handle:
        model_cfg = json.load(handle)

    model = _build_active_model(cfg, model_cfg, get_cast_dtype("fp32"), quick_gelu=False)
    ckpt = torch.load(str(resolve_project_path(checkpoint_path)), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def run_inference(model, tokenizer, query_tensor: torch.Tensor, prompt_tensors: torch.Tensor, device: str):
    query_image = query_tensor.unsqueeze(0).to(device)
    prompt_images = prompt_tensors.unsqueeze(0).to(device)
    return model(
        tokenizer=tokenizer,
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=None,
        return_aux=True,
        return_dict=True,
    )


def classify_sample(label: int, score: float, threshold: float = THRESHOLD) -> tuple[str, str]:
    predicted_anomaly = score >= threshold
    if label == 0 and not predicted_anomaly:
        return "True Negative", "#4CAF50"
    if label == 0 and predicted_anomaly:
        return "False Positive", "#FF9800"
    if label == 1 and predicted_anomaly:
        return "True Positive", "#F44336"
    return "False Negative", "#2196F3"


def prediction_from_score(score: float, threshold: float = THRESHOLD) -> int:
    return int(score >= threshold)


def _batch_to_query_and_prompts(inputs):
    if isinstance(inputs, (list, tuple)):
        query_tensor = inputs[0]
        prompt_tensors = torch.stack(list(inputs[1:]), dim=1)
        return query_tensor, prompt_tensors
    if isinstance(inputs, torch.Tensor) and inputs.dim() == 5:
        return inputs[0], inputs[1:].permute(1, 0, 2, 3, 4)
    return inputs, None


def _sample_metadata(dataset, index: int) -> tuple[str, list[str]]:
    query_path = ""
    prompt_paths: list[str] = []
    samples = getattr(dataset, "image", None)
    shot_indices = getattr(dataset, "precomputed_shot_indices", None)
    normal_samples = getattr(dataset, "normal_samples", None)

    if samples and index < len(samples):
        query_path = str(samples[index].get("image_path", ""))
    if shot_indices and normal_samples and index < len(shot_indices):
        for normal_idx in shot_indices[index]:
            if normal_idx < len(normal_samples):
                prompt_paths.append(str(normal_samples[normal_idx].get("image_path", "")))
    return query_path, prompt_paths


def collect_category_samples(
    *,
    model,
    tokenizer,
    cfg,
    transform,
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    category: str,
    shot: int,
    seed: int,
    device: str,
) -> list[GallerySample]:
    """Run inference for one model/dataset/category/seed and return all sample artifacts."""
    from datasets import loader

    json_dir = os.path.join("data", "AD_json", test_dataset.lower())
    cat_cfg = cfg.clone()
    cat_cfg.val_normal_json_path = [os.path.join(json_dir, f"{category}_val_normal.json")]
    cat_cfg.val_outlier_json_path = [os.path.join(json_dir, f"{category}_val_outlier.json")]
    cat_cfg.FEW_SHOT_SEED = int(seed)

    test_loader = loader.construct_loader(cat_cfg, "test", transform)
    samples: list[GallerySample] = []
    global_index = 0

    for batch in test_loader:
        inputs, labels = batch[0], batch[2]
        query_tensor, prompt_tensors = _batch_to_query_and_prompts(inputs)
        if prompt_tensors is None:
            continue

        for batch_index in range(query_tensor.shape[0]):
            query = query_tensor[batch_index]
            prompts = prompt_tensors[batch_index]
            label = int(labels[batch_index].item())
            outputs = run_inference(model, tokenizer, query, prompts, device)

            score_tensor = outputs.get("coupled_score", outputs.get("final_score"))
            score = float(score_tensor[0].detach().cpu())
            prediction = prediction_from_score(score)
            case_type, border_color = classify_sample(label, score)

            residual_map = np.squeeze(outputs["patch_residual_map"][0].detach().cpu().numpy())
            if residual_map.ndim == 1:
                grid_size = int(math.sqrt(residual_map.shape[0]))
                residual_grid = residual_map.reshape(grid_size, grid_size)
            else:
                residual_grid = residual_map
            final_map = outputs["final_map"][0, 0].detach().cpu().numpy()

            query_path, prompt_paths = _sample_metadata(test_loader.dataset, global_index)
            samples.append(
                GallerySample(
                    model_name=model_name,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    category=category,
                    shot=shot,
                    seed=seed,
                    sample_index=global_index,
                    query_path=query_path,
                    prompt_paths=prompt_paths,
                    label=label,
                    score=score,
                    prediction=prediction,
                    case_type=case_type,
                    border_color=border_color,
                    query_display=tensor_to_display(query),
                    prompt_displays=[tensor_to_display(prompts[idx]) for idx in range(prompts.shape[0])],
                    residual_grid=residual_grid,
                    final_map=final_map,
                )
            )
            global_index += 1

    return samples


def select_representative_samples(
    samples: Iterable[GallerySample],
    n_examples: int,
    seed: int,
) -> list[GallerySample]:
    """Select TP/FP/TN/FN-oriented examples with deterministic fallback."""
    samples = list(samples)
    if n_examples <= 0 or not samples:
        return []

    for sample in samples:
        sample.prediction = prediction_from_score(sample.score)
        sample.case_type, sample.border_color = classify_sample(sample.label, sample.score)

    selected: list[GallerySample] = []
    used: set[int] = set()

    def add_first(candidates: list[GallerySample]) -> None:
        if len(selected) >= n_examples:
            return
        for candidate in candidates:
            if candidate.sample_index not in used:
                selected.append(candidate)
                used.add(candidate.sample_index)
                return

    case_order = [
        ("True Positive", True),
        ("False Positive", True),
        ("True Negative", False),
        ("False Negative", False),
    ]
    for case_type, reverse in case_order:
        candidates = [sample for sample in samples if sample.case_type == case_type]
        candidates.sort(key=lambda sample: sample.score, reverse=reverse)
        add_first(candidates)

    anomalous = sorted(
        [sample for sample in samples if sample.label == 1],
        key=lambda sample: sample.score,
        reverse=True,
    )
    add_first(anomalous)

    boundary = sorted(samples, key=lambda sample: abs(sample.score - THRESHOLD))
    add_first(boundary)

    rng = np.random.default_rng(seed)
    fallback = [sample for sample in samples if sample.sample_index not in used]
    fallback.sort(key=lambda sample: sample.score, reverse=True)
    while len(selected) < n_examples and fallback:
        if len(selected) % 2 == 0:
            candidate = fallback.pop(0)
        elif len(fallback) > 1:
            idx = int(rng.integers(0, len(fallback)))
            candidate = fallback.pop(idx)
        else:
            candidate = fallback.pop(0)
        selected.append(candidate)
        used.add(candidate.sample_index)

    return selected[:n_examples]


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def save_heatmap_overlay_grid(samples: list[GallerySample], title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(samples))
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, sample in enumerate(samples):
        axes[0, idx].imshow(sample.query_display)
        for spine in axes[0, idx].spines.values():
            spine.set_edgecolor(sample.border_color)
            spine.set_linewidth(4)
        axes[0, idx].set_title(
            f"{sample.case_type}\nscore={sample.score:.3f}",
            fontsize=9,
            fontweight="bold",
            color=sample.border_color,
        )
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])

        axes[1, idx].imshow(sample.query_display)
        axes[1, idx].imshow(normalize_map(sample.final_map), cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])

    axes[0, 0].set_ylabel("Query", fontsize=11)
    axes[1, 0].set_ylabel("Heatmap", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_residual_map_grid(samples: list[GallerySample], title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(samples))
    prompts = samples[0].prompt_displays if samples else []
    fig, axes = plt.subplots(
        2,
        n + 1,
        figsize=(2.5 + 3.2 * n, 7.4),
        gridspec_kw={"width_ratios": [1.0] + [2.0] * n, "hspace": 0.25, "wspace": 0.15},
    )
    if n == 1:
        axes = axes.reshape(2, 2)

    if prompts:
        axes[0, 0].imshow(np.concatenate(prompts, axis=0))
    axes[0, 0].set_title(f"{len(prompts)}-shot Normal\nReferences", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    residual_values = np.concatenate([np.asarray(sample.residual_grid).ravel() for sample in samples])
    vmin = float(np.percentile(residual_values, 2))
    vmax = float(np.percentile(residual_values, 98))
    image = None

    for idx, sample in enumerate(samples):
        query_ax = axes[0, idx + 1]
        query_ax.imshow(sample.query_display)
        for spine in query_ax.spines.values():
            spine.set_edgecolor(sample.border_color)
            spine.set_linewidth(4)
        query_ax.set_title(
            f"{sample.case_type}\nscore={sample.score:.3f}",
            fontsize=9,
            fontweight="bold",
            color=sample.border_color,
        )
        query_ax.set_xticks([])
        query_ax.set_yticks([])

        residual_ax = axes[1, idx + 1]
        image = residual_ax.imshow(sample.residual_grid, cmap=RESIDUAL_CMAP, vmin=vmin, vmax=vmax, origin="upper")
        grid_size = sample.residual_grid.shape[0]
        residual_ax.set_xticks(range(grid_size))
        residual_ax.set_yticks(range(grid_size))
        residual_ax.set_xticklabels(range(grid_size), fontsize=5)
        residual_ax.set_yticklabels(range(grid_size), fontsize=5)
        residual_ax.tick_params(length=0)

    if image is not None:
        colorbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.35])
        fig.colorbar(image, cax=colorbar_ax, label="Residual Value")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sample_panel(sample: GallerySample, title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(sample.query_display)
    for spine in axes[0].spines.values():
        spine.set_edgecolor(sample.border_color)
        spine.set_linewidth(3)
    axes[0].set_title(f"Query ({sample.case_type})\nscore={sample.score:.3f}", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(sample.residual_grid, cmap=RESIDUAL_CMAP, origin="upper")
    axes[1].set_title("Patch Residual Map", fontsize=10)
    grid_size = sample.residual_grid.shape[0]
    axes[1].set_xticks(range(grid_size))
    axes[1].set_yticks(range(grid_size))
    axes[1].set_xticklabels(range(grid_size), fontsize=5)
    axes[1].set_yticklabels(range(grid_size), fontsize=5)

    axes[2].imshow(sample.query_display)
    axes[2].imshow(normalize_map(sample.final_map), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Final Anomaly Map Overlay", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    label = "anomaly" if sample.label else "normal"
    pred = "anomaly" if sample.prediction else "normal"
    fig.suptitle(f"{title} | label={label} pred={pred} case={sample.case_type}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_manifest_row(sample: GallerySample, image_type: str, relative_path: Path) -> dict[str, Any]:
    return {
        "model_name": sample.model_name,
        "train_dataset": sample.train_dataset,
        "test_dataset": sample.test_dataset,
        "category": sample.category,
        "shot": sample.shot,
        "seed": sample.seed,
        "sample_index": sample.sample_index,
        "query_path": sample.query_path,
        "prompt_paths": "|".join(sample.prompt_paths),
        "label": sample.label,
        "prediction": sample.prediction,
        "score": f"{sample.score:.6f}",
        "case_type": sample.case_type,
        "image_type": image_type,
        "file": str(relative_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
