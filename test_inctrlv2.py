#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.inctrlv2_dataset import InCTRLv2DirectoryDataset, resolve_dataset_root
from models.inctrlv2 import build_inctrlv2_model
from models.inctrlv2.metrics import compute_image_metrics, compute_pixel_metrics

SUPPORTED_DATASETS = ["mvtec", "visa", "aitex", "elpv", "sdd"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate InCTRLv2.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/inctrlv2_eval")
    parser.add_argument("--metrics", type=str, nargs="+", default=["image_auroc", "image_ap", "pixel_auroc", "pixel_pro"])
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--clip_checkpoint", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--selected_layers", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--disable_dasl", action="store_true")
    parser.add_argument("--disable_oasl", action="store_true")
    parser.add_argument("--normal_json", type=str, nargs="*", default=None)
    parser.add_argument("--outlier_json", type=str, nargs="*", default=None)
    return parser.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _json_safe(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            result[key] = None
        else:
            result[key] = value
    return result


def _save_map(path: Path, anomaly_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    train_dataset = None
    if isinstance(checkpoint, dict):
        train_dataset = checkpoint.get("train_dataset") or ckpt_args.get("train_dataset")
    backbone = args.backbone or checkpoint.get("backbone") or ckpt_args.get("backbone") or "ViT-B-16-plus-240"
    input_size = args.input_size or ckpt_args.get("input_size") or 240
    selected_layers = args.selected_layers or checkpoint.get("selected_layers") or ckpt_args.get("selected_layers") or [7, 9, 11]
    alpha = args.alpha if args.alpha is not None else checkpoint.get("alpha", ckpt_args.get("alpha", 0.5))
    beta = args.beta if args.beta is not None else checkpoint.get("beta", ckpt_args.get("beta", 0.75))
    disable_dasl = args.disable_dasl or bool(ckpt_args.get("disable_dasl", False))
    disable_oasl = args.disable_oasl or bool(ckpt_args.get("disable_oasl", False))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = build_inctrlv2_model(
        backbone=backbone,
        clip_checkpoint=args.clip_checkpoint,
        device=device,
        selected_layers=selected_layers,
        alpha=alpha,
        beta=beta,
        disable_dasl=disable_dasl,
        disable_oasl=disable_oasl,
        allow_random_init=True,
    )
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset_root = resolve_dataset_root(args.data_root, args.test_dataset)
    dataset = InCTRLv2DirectoryDataset(
        dataset_root=dataset_root,
        split="test",
        shots=args.shot,
        input_size=input_size,
        seed=args.seed,
        normal_json_paths=args.normal_json,
        outlier_json_paths=args.outlier_json,
    )
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {"persistent_workers": args.num_workers > 1, "prefetch_factor": 1}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        **worker_kwargs,
    )

    scores = []
    labels = []
    pixel_maps = []
    masks = []
    rows = []
    out_root = Path(args.output_dir)
    if train_dataset:
        out_root = out_root / f"trained_on_{train_dataset}"
    out_dir = out_root / f"test_on_{args.test_dataset}" / f"shot_{args.shot}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"test_on_{args.test_dataset} shot={args.shot}"):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                query_image=batch["query_image"],
                prompt_images=batch["prompt_images"],
                class_name=batch["class_name"],
            )
            batch_scores = outputs["image_score"].detach().cpu().numpy()
            batch_maps = outputs["pixel_map"].detach().cpu().numpy()[:, 0]
            batch_masks = batch["mask"].detach().cpu().numpy()[:, 0]
            batch_labels = batch["label"].detach().cpu().numpy()
            scores.extend(batch_scores.tolist())
            labels.extend(batch_labels.tolist())
            pixel_maps.append(batch_maps)
            masks.append(batch_masks)
            for idx, image_path in enumerate(batch["image_path"]):
                rows.append(
                    {
                        "image_path": image_path,
                        "class_name": batch["class_name"][idx],
                        "label": int(batch_labels[idx]),
                        "score": float(batch_scores[idx]),
                    }
                )
                if args.save_maps:
                    safe_name = image_path.replace("/", "_").replace("\\", "_")
                    _save_map(out_dir / "visualizations" / f"{safe_name}.png", batch_maps[idx])

    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    pixel_maps_np = np.concatenate(pixel_maps, axis=0)
    masks_np = np.concatenate(masks, axis=0).astype(np.uint8)

    metrics = {
        "train_dataset": train_dataset,
        "test_dataset": args.test_dataset,
        "shot": args.shot,
        "seed": args.seed,
        "alpha": float(alpha),
        "beta": float(beta),
        "backbone": backbone,
    }
    image_metrics = compute_image_metrics(scores_np, labels_np)
    pixel_metrics = compute_pixel_metrics(pixel_maps_np, masks_np)
    metrics.update({key: value for key, value in image_metrics.items() if key in args.metrics})
    metrics.update({key: value for key, value in pixel_metrics.items() if key in args.metrics})

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2)
    with open(out_dir / "pixel_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(pixel_metrics), handle, indent=2)
    with open(out_dir / "anomaly_scores.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name", "label", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
