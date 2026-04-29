#!/usr/bin/env python3
"""
Multi-seed evaluation for all 6 InCTRLAdapt models.

Loads each trained checkpoint once, evaluates on target datasets with multiple
few-shot sampling seeds, and reports mean ± std AUROC/AUPR.

Output:
  results/multi_seed/all_models_multi_seed.csv   — per-model per-seed per-category
  results/multi_seed/summary.csv                 — mean±std aggregated per model×dataset
  results/multi_seed/all_results.json            — full structured results

Usage:
  python tools/eval_multi_seed.py
  python tools/eval_multi_seed.py --seeds 42 123 7 0 999
  python tools/eval_multi_seed.py --dry_run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from open_clip.config.defaults import assert_and_infer_cfg, get_cfg

# ── Model registry: 6 models ────────────────────────────────────────────────

MODELS = [
    {
        "name": "mvtec_va_ta_pqa_2shot",
        "checkpoint": "results/mvtec_va_ta_pqa_2shot_15ep/checkpoint_best.pyth",
        "train_dataset": "mvtec",
        "test_datasets": ["visa", "aitex", "elpv"],
        "shot": 2,
    },
    {
        "name": "mvtec_va_ta_pqa_4shot",
        "checkpoint": "results/mvtec_va_ta_pqa_4shot_15ep/checkpoint_best.pyth",
        "train_dataset": "mvtec",
        "test_datasets": ["visa", "aitex", "elpv"],
        "shot": 4,
    },
    {
        "name": "mvtec_va_ta_pqa_8shot",
        "checkpoint": "results/mvtec_va_ta_pqa_8shot_15ep/checkpoint_best.pyth",
        "train_dataset": "mvtec",
        "test_datasets": ["visa", "aitex", "elpv"],
        "shot": 8,
    },
    {
        "name": "visa_va_ta_pqa_2shot",
        "checkpoint": "results/visa_va_ta_pqa_2shot_15ep/checkpoint_best.pyth",
        "train_dataset": "visa",
        "test_datasets": ["mvtec"],
        "shot": 2,
    },
    {
        "name": "visa_va_ta_pqa_4shot",
        "checkpoint": "results/visa_va_ta_pqa_4shot_15ep/checkpoint_best.pyth",
        "train_dataset": "visa",
        "test_datasets": ["mvtec"],
        "shot": 4,
    },
    {
        "name": "visa_va_ta_pqa_8shot",
        "checkpoint": "results/visa_va_ta_pqa_8shot_15ep/checkpoint_best.pyth",
        "train_dataset": "visa",
        "test_datasets": ["mvtec"],
        "shot": 8,
    },
]

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

PUBLISHED_CROSS_DOMAIN_AUROC = {
    "elpv": {0: 0.733, 2: 0.839, 4: 0.846, 8: 0.872},
    "aitex": {0: 0.733, 2: 0.761, 4: 0.790, 8: 0.806},
    "visa": {0: 0.781, 2: 0.858, 4: 0.877, 8: 0.887},
    "mvtec": {0: 0.912, 2: 0.940, 4: 0.945, 8: 0.953},
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "multi_seed"


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_cfg(shot: int, device: str, few_shot_seed: int = 42):
    cfg = get_cfg()
    cfg.shot = shot
    cfg.image_size = 240
    cfg.NUM_GPUS = 1 if device == "cuda" else 0
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg.TEST.BATCH_SIZE = 16
    cfg.DATA_LOADER.NUM_WORKERS = 2
    cfg.DATA_LOADER.PIN_MEMORY = device == "cuda"
    cfg.TRAIN.SHOW_PROGRESS = True
    cfg.FEW_SHOT_SEED = few_shot_seed
    cfg = assert_and_infer_cfg(cfg)
    return cfg


def build_transform():
    from torchvision import transforms

    def _convert_to_rgb(image):
        return image.convert("RGB")

    return transforms.Compose([
        transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(240),
        transforms.Lambda(_convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
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
    info = model.load_state_dict(ckpt, strict=False)
    if info.missing_keys:
        print(f"  [WARN] missing keys: {info.missing_keys[:5]}...")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def eval_dataset_with_seed(model, tokenizer, transform, cfg, dataset_name: str, seed: int, device: str):
    """Evaluate one dataset with a specific few-shot seed. Returns per-category results."""
    from datasets import loader
    from engine_IC import eval_epoch

    categories = DATASET_CATEGORIES[dataset_name.lower()]
    json_dir = os.path.join("data", "AD_json", dataset_name.lower())
    results = []

    for cat in categories:
        cat_cfg = cfg.clone()
        cat_cfg.val_normal_json_path = [os.path.join(json_dir, f"{cat}_val_normal.json")]
        cat_cfg.val_outlier_json_path = [os.path.join(json_dir, f"{cat}_val_outlier.json")]
        cat_cfg.FEW_SHOT_SEED = seed

        test_loader = loader.construct_loader(cat_cfg, "test", transform)
        auroc, aupr = eval_epoch(test_loader, model, cat_cfg, tokenizer, f"test/{cat}")
        results.append({"category": cat, "auroc": float(auroc), "aupr": float(aupr)})

    mean_auroc = float(np.mean([r["auroc"] for r in results]))
    mean_aupr = float(np.mean([r["aupr"] for r in results]))
    return results, mean_auroc, mean_aupr


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed batch evaluation for 6 InCTRLAdapt models.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7],
                        help="Random seeds for few-shot sampling (default: 42 123 7)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the evaluation plan without running")
    parser.add_argument("--models", type=int, nargs="*", default=None,
                        help="0-indexed model indices to evaluate (default: all 6)")
    args = parser.parse_args()

    seeds = args.seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = args.models if args.models is not None else list(range(len(MODELS)))

    # ── Plan ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Multi-Seed Evaluation Plan")
    print("=" * 70)
    total_evals = 0
    for i in selected:
        m = MODELS[i]
        n_cats = sum(len(DATASET_CATEGORIES[ds]) for ds in m["test_datasets"])
        n_evals = len(seeds) * n_cats
        total_evals += n_evals
        ckpt_exists = (PROJECT_ROOT / m["checkpoint"]).exists()
        status = "✅" if ckpt_exists else "❌ NOT FOUND"
        print(f"  [{i}] {m['name']} ({m['shot']}-shot)")
        print(f"      checkpoint: {m['checkpoint']} {status}")
        print(f"      test on: {m['test_datasets']} × {len(seeds)} seeds = {n_evals} category evals")
    print(f"\n  Total: {len(selected)} models × {len(seeds)} seeds = {total_evals} category evaluations")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without evaluation.")
        return

    # ── Run ──────────────────────────────────────────────────────────────
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    transform = build_transform()

    all_detail_rows = []    # per-model, per-seed, per-category
    all_summary_rows = []   # per-model, per-seed, per-dataset (mean)
    all_agg_rows = []       # per-model, per-dataset (mean±std across seeds)
    full_results = {}       # JSON structure

    for i in selected:
        m = MODELS[i]
        model_name = m["name"]
        ckpt_path = PROJECT_ROOT / m["checkpoint"]
        if not ckpt_path.exists():
            print(f"\n[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f" Loading: {model_name} ({m['shot']}-shot, train={m['train_dataset']})")
        print(f"{'='*60}")

        cfg = build_cfg(m["shot"], device)
        model = load_model(cfg, m["checkpoint"], device)
        full_results[model_name] = {}

        for ds in m["test_datasets"]:
            ds_seed_aurocs = []
            ds_seed_auprs = []
            full_results[model_name][ds] = {}

            for seed in seeds:
                t0 = time.time()
                print(f"\n  [{ds.upper()}] seed={seed} ...")
                cat_results, mean_auroc, mean_aupr = eval_dataset_with_seed(
                    model, tokenizer, transform, cfg, ds, seed, device
                )
                elapsed = time.time() - t0
                ds_seed_aurocs.append(mean_auroc)
                ds_seed_auprs.append(mean_aupr)

                full_results[model_name][ds][str(seed)] = {
                    "categories": cat_results,
                    "mean_auroc": mean_auroc,
                    "mean_aupr": mean_aupr,
                }

                # Detail rows (per category)
                for cr in cat_results:
                    all_detail_rows.append({
                        "model": model_name,
                        "train_dataset": m["train_dataset"],
                        "test_dataset": ds,
                        "shot": m["shot"],
                        "seed": seed,
                        "category": cr["category"],
                        "auroc": round(cr["auroc"], 4),
                        "aupr": round(cr["aupr"], 4),
                    })

                # Summary row (dataset mean per seed)
                all_summary_rows.append({
                    "model": model_name,
                    "train_dataset": m["train_dataset"],
                    "test_dataset": ds,
                    "shot": m["shot"],
                    "seed": seed,
                    "mean_auroc": round(mean_auroc, 4),
                    "mean_aupr": round(mean_aupr, 4),
                    "elapsed_sec": round(elapsed, 1),
                })

                baseline = PUBLISHED_CROSS_DOMAIN_AUROC.get(ds, {}).get(m["shot"])
                delta_str = ""
                if baseline:
                    delta = mean_auroc - baseline
                    delta_str = f"  Δbaseline={delta:+.4f}"
                print(f"    -> AUROC={mean_auroc:.4f}  AUPR={mean_aupr:.4f}{delta_str}  ({elapsed:.1f}s)")

            # Aggregate row (mean±std across seeds)
            auroc_mean = float(np.mean(ds_seed_aurocs))
            auroc_std = float(np.std(ds_seed_aurocs))
            aupr_mean = float(np.mean(ds_seed_auprs))
            aupr_std = float(np.std(ds_seed_auprs))
            baseline = PUBLISHED_CROSS_DOMAIN_AUROC.get(ds, {}).get(m["shot"])
            all_agg_rows.append({
                "model": model_name,
                "train_dataset": m["train_dataset"],
                "test_dataset": ds,
                "shot": m["shot"],
                "n_seeds": len(seeds),
                "seeds": str(seeds),
                "auroc_mean": round(auroc_mean, 4),
                "auroc_std": round(auroc_std, 4),
                "aupr_mean": round(aupr_mean, 4),
                "aupr_std": round(aupr_std, 4),
                "baseline_auroc": baseline,
                "delta_vs_baseline": round(auroc_mean - baseline, 4) if baseline else None,
            })

        # Free GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Detail CSV (per category)
    detail_csv = OUTPUT_DIR / "all_models_multi_seed.csv"
    if all_detail_rows:
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_detail_rows)
        print(f"\n[SAVED] {detail_csv}")

    # 2. Summary CSV (mean±std per model×dataset)
    summary_csv = OUTPUT_DIR / "summary.csv"
    if all_agg_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_agg_rows)
        print(f"[SAVED] {summary_csv}")

    # 3. Full JSON
    results_json = OUTPUT_DIR / "all_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds,
            "models": {m["name"]: m for m in MODELS},
            "results": full_results,
            "aggregated": all_agg_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {results_json}")

    # ── Print final summary table ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f" Final Summary: AUROC mean±std  (seeds={seeds})")
    print(f"{'='*80}")
    print(f" {'Model':<28} {'Dataset':<8} {'Shot':>4}  {'AUROC':>14}  {'Baseline':>8}  {'Delta':>8}")
    print(f" {'-'*76}")
    for row in all_agg_rows:
        auroc_str = f"{row['auroc_mean']:.4f}±{row['auroc_std']:.4f}"
        baseline_str = f"{row['baseline_auroc']:.3f}" if row['baseline_auroc'] else "N/A"
        delta_str = f"{row['delta_vs_baseline']:+.4f}" if row['delta_vs_baseline'] is not None else "N/A"
        print(f" {row['model']:<28} {row['test_dataset']:<8} {row['shot']:>4}  {auroc_str:>14}  {baseline_str:>8}  {delta_str:>8}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
