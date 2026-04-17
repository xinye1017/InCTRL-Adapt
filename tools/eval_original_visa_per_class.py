#!/usr/bin/env python3
"""Evaluate original InCTRL checkpoint on VisA per category.

This script intentionally imports model code from the original repository while
using the local extracted data/few-shot files from the active workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original InCTRL VisA per-class evaluation")
    parser.add_argument("--original-repo", type=Path, default=Path(r"D:\Data\Downloads\InCTRL原版"))
    parser.add_argument("--workspace", type=Path, default=Path(r"D:\Data\Downloads\InCTRL"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\Data\Downloads\InCTRL\checkpoints\InCTRL_trained_on_MVTec\4\checkpoint.pyth"),
    )
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def import_original_modules(original_repo: Path):
    sys.path.insert(0, str(original_repo))
    import open_clip  # type: ignore
    from open_clip.config.defaults import get_cfg  # type: ignore
    from open_clip.model import get_cast_dtype  # type: ignore

    return open_clip, get_cfg, get_cast_dtype


def build_transform():
    def _convert_to_rgb(image):
        return image.convert("RGB")

    return transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_image(path: str, transform) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image)


def load_category_samples(json_dir: Path, category: str) -> list[dict]:
    normal_path = json_dir / f"{category}_val_normal.json"
    outlier_path = json_dir / f"{category}_val_outlier.json"
    normal = json.loads(normal_path.read_text(encoding="utf-8"))
    outlier = json.loads(outlier_path.read_text(encoding="utf-8"))
    return normal + outlier


def discover_categories(json_dir: Path) -> list[str]:
    return sorted(
        path.name.replace("_val_normal.json", "")
        for path in json_dir.glob("*_val_normal.json")
    )


def build_original_model(args, open_clip, get_cfg, get_cast_dtype):
    cfg = get_cfg()
    cfg.NUM_GPUS = 1 if args.device == "cuda" else 0
    cfg.image_size = 240
    cfg.shot = args.shot

    model_config_path = args.original_repo / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model = open_clip.model.InCTRL(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
    return cfg, model


@torch.no_grad()
def build_prompt_cache(model, few_shot_path: Path, device: str) -> dict[str, torch.Tensor]:
    normal_list = torch.load(few_shot_path, map_location=device)
    normal_images = torch.stack(normal_list).to(device)
    token_n, fp_list_n, _ = model.encode_image(normal_images, normalize=False)
    fp_list_n = torch.stack(fp_list_n)[:, :, 1:, :]
    shot = normal_images.shape[0]
    return {
        "token_n_adapted_mean": model.adapter.forward(token_n).mean(dim=0, keepdim=True),
        "fp_list_n": fp_list_n.reshape(3, shot * 225, -1),
    }


@torch.no_grad()
def build_text_cache(model, tokenizer, category: str, device: str) -> torch.Tensor:
    from open_clip.model import get_texts  # imported from original repo via sys.path

    normal_texts, anomaly_texts = get_texts(category.replace("_", " "))
    pos_tokens = tokenizer(normal_texts).to(device)
    neg_tokens = tokenizer(anomaly_texts).to(device)
    pos_features = model.encode_text(pos_tokens)
    neg_features = model.encode_text(neg_tokens)
    pos_features = F.normalize(pos_features, dim=-1).mean(dim=0, keepdim=True)
    neg_features = F.normalize(neg_features, dim=-1).mean(dim=0, keepdim=True)
    pos_features = F.normalize(pos_features, dim=-1)
    neg_features = F.normalize(neg_features, dim=-1)
    return torch.cat([pos_features, neg_features], dim=0)


@torch.no_grad()
def predict_one(model, image: torch.Tensor, prompt_cache: dict[str, torch.Tensor], text_features: torch.Tensor, device: str) -> float:
    image = image.unsqueeze(0).to(device)
    token, fp_list, _ = model.encode_image(image, normalize=False)
    fp_list = torch.stack(fp_list)[:, :, 1:, :].reshape(3, 225, -1)

    token_ad = model.adapter.forward(token)
    token_ref = prompt_cache["token_n_adapted_mean"] - token_ad

    q = F.normalize(fp_list, dim=-1)
    n = F.normalize(prompt_cache["fp_list_n"], dim=-1)
    patch_dist = 0.5 * (1.0 - torch.matmul(q, n.transpose(-1, -2)))
    patch_ref_map = patch_dist.min(dim=-1).values.mean(dim=0)
    max_diff_score = patch_ref_map.max()

    image_feature = F.normalize(token, dim=-1)
    text_score = (100 * image_feature @ text_features.T).softmax(dim=-1)[:, 1:2]
    img_ref_score = model.diff_head_ref.forward(token_ref)
    holistic_map = text_score + img_ref_score + patch_ref_map.unsqueeze(0)
    hl_score = model.diff_head.forward(holistic_map).squeeze(1)
    final_score = (hl_score + max_diff_score.unsqueeze(0)) / 2
    return float(final_score.detach().cpu().item())


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Original InCTRL VisA Per-Class Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        f"Dataset: `{payload['dataset']}`",
        f"Shot: `{payload['shot']}`",
        f"Mean AUROC: `{payload['mean_auroc']:.6f}`",
        f"Mean AUPR: `{payload['mean_aupr']:.6f}`",
        "",
        "| Category | AUROC | AUPR | N |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in payload["categories"]:
        lines.append(
            f"| {item['category']} | {item['auroc']:.6f} | {item['aupr']:.6f} | {item['num_samples']} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    open_clip, get_cfg, get_cast_dtype = import_original_modules(args.original_repo)
    _, model = build_original_model(args, open_clip, get_cfg, get_cast_dtype)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    transform = build_transform()

    json_dir = args.workspace / "data" / "AD_json" / args.dataset
    few_shot_dir = args.workspace / "few-shot samples" / args.dataset / args.dataset / str(args.shot)
    categories = discover_categories(json_dir)
    print(f"[INFO] categories={len(categories)} device={args.device} shot={args.shot}", flush=True)

    category_results = []
    for category in categories:
        category_start = time.time()
        samples = load_category_samples(json_dir, category)
        few_shot_path = few_shot_dir / f"{category}.pt"
        prompt_cache = build_prompt_cache(model, few_shot_path, args.device)
        text_features = build_text_cache(model, tokenizer, category, args.device)

        scores = []
        labels = []
        for sample in tqdm(samples, desc=f"[VISA] {category}", leave=False):
            image = load_image(sample["image_path"], transform)
            scores.append(predict_one(model, image, prompt_cache, text_features, args.device))
            labels.append(int(sample["target"]))

        auroc = float(roc_auc_score(labels, scores))
        aupr = float(average_precision_score(labels, scores))
        elapsed = time.time() - category_start
        print(
            f"{category}: AUROC={auroc:.6f}, AUPR={aupr:.6f}, "
            f"N={len(labels)}, elapsed={elapsed:.1f}s",
            flush=True,
        )
        category_results.append({
            "category": category,
            "auroc": auroc,
            "aupr": aupr,
            "num_samples": len(labels),
            "elapsed_seconds": elapsed,
        })

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "shot": args.shot,
        "categories": category_results,
        "mean_auroc": float(np.mean([item["auroc"] for item in category_results])),
        "mean_aupr": float(np.mean([item["aupr"] for item in category_results])),
        "elapsed_seconds": time.time() - start_time,
    }

    output_json = args.output_json or args.workspace / "reports" / "original_inctrl_visa_4shot_per_class.json"
    output_md = args.output_md or args.workspace / "reports" / "original_inctrl_visa_4shot_per_class.md"
    write_outputs(payload, output_json, output_md)
    print(f"[INFO] saved json: {output_json}", flush=True)
    print(f"[INFO] saved md: {output_md}", flush=True)
    print(f"[INFO] mean AUROC={payload['mean_auroc']:.6f}, mean AUPR={payload['mean_aupr']:.6f}", flush=True)


if __name__ == "__main__":
    main()
