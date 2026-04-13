#!/usr/bin/env python3
"""
使用已保存的检查点在 VisA 上进行测试
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip
from open_clip.model import get_cast_dtype
from open_clip.config.defaults import get_cfg
from datasets import loader as ds_loader
from binary_focal_loss import BinaryFocalLoss

# ============================================================================
# 配置
# ============================================================================

DATA_ROOT = PROJECT_ROOT / "data"
FEW_SHOT_ROOT = PROJECT_ROOT / "few-shot samples"
RESULTS_DIR = PROJECT_ROOT / "results"

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "inctrl_va_bs48_lr0.001_ep10_epoch_1.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOT_LIST = [2, 4, 8]
TEST_DATASETS = ["visa"]

# ============================================================================
# 工具函数
# ============================================================================

def get_transform():
    return transforms.Compose([
        transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(240),
        transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def prepare_dataset_paths():
    global DATASET_CATEGORIES, TYPE_TO_IDX

    json_mvtec = DATA_ROOT / "AD_json" / "mvtec"

    DATASET_CATEGORIES = {}
    for ds in TEST_DATASETS:
        cats = set()
        json_ds = DATA_ROOT / "AD_json" / ds
        for f in json_ds.glob("*_val_normal.json"):
            cats.add(f.name.replace("_val_normal.json", ""))
        DATASET_CATEGORIES[ds] = sorted(cats)

    # 构建类型名称到索引的映射
    all_types = set()
    for f in sorted(json_mvtec.glob("*_normal.json")):
        if "val_" not in f.name:
            data = json.load(open(f))
            for item in data:
                all_types.add(item['type'])
    TYPE_TO_IDX = {t: i for i, t in enumerate(sorted(all_types))}

    print(f"[INFO] VisA 测试类别 ({len(DATASET_CATEGORIES['visa'])}): {DATASET_CATEGORIES['visa']}")


def build_model(device):
    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    from open_clip import model as _model_mod

    model = _model_mod.InCTRL(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    print(f"[INFO] 加载检查点: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    return model.to(device)


def find_fs_pt(ds, cat, shot):
    """查找 few-shot pt 文件"""
    for path in FEW_SHOT_ROOT.iterdir():
        if path.name.lower() == ds.lower():
            for subpath in path.iterdir():
                if subpath.is_dir() and subpath.name.lower() == ds.lower():
                    shot_dir = subpath / str(shot)
                    if shot_dir.exists():
                        pt_path = shot_dir / f"{cat}.pt"
                        if pt_path.exists():
                            return pt_path
                        for candidate in shot_dir.glob("*.pt"):
                            if candidate.stem.lower() == cat.lower():
                                return candidate
    raise FileNotFoundError(f"Cannot find few-shot pt for {ds} - {cat} (shot={shot})")


def build_cached_normal_img_features(model, few_shot_path, device):
    """构建用于测试的 normal_list (返回 list of tensors)"""
    few_shot_list = torch.load(few_shot_path)
    # 返回 list of tensors，每个 tensor 是 [3, 240, 240]
    return [tensor.to(device) for tensor in few_shot_list]


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, cached_normal_list=None):
    model.eval()
    preds_all, labels_all = [], []

    for inputs, types, labels in tqdm(loader, desc="[TEST] Batch", leave=False):
        labels = labels.to(device)
        # cached_normal_list 直接作为 normal_list 参数传入
        preds, _ = model(
            tokenizer,
            inputs,
            types,
            cached_normal_list,  # 传入预计算的 normal images
        )
        preds_all.extend(preds.detach().cpu().float().numpy())
        labels_all.extend(labels.cpu().numpy())

    auroc = roc_auc_score(labels_all, preds_all)
    aupr = average_precision_score(labels_all, preds_all)
    return float(auroc), float(aupr)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print(f"[INFO] 设备: {DEVICE}")
    prepare_dataset_paths()

    model = build_model(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    transform = get_transform()
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for shot in SHOT_LIST:
        print(f"\n===== 测试 {shot}-shot =====")
        cfg.shot = shot
        shot_results = {}

        for ds in TEST_DATASETS:
            ds_res = []
            print(f"\n[INFO] 测试数据集: {ds.upper()}")

            for cat in DATASET_CATEGORIES[ds]:
                cfg.val_normal_json_path = [str(DATA_ROOT / "AD_json" / ds / f"{cat}_val_normal.json")]
                cfg.val_outlier_json_path = [str(DATA_ROOT / "AD_json" / ds / f"{cat}_val_outlier.json")]
                val_loader = ds_loader.construct_loader(cfg, "test", transform)

                fs_pt = find_fs_pt(ds, cat, shot)
                cached_normal_list = build_cached_normal_img_features(model, fs_pt, DEVICE)
                auroc, aupr = evaluate(
                    model,
                    tokenizer,
                    val_loader,
                    DEVICE,
                    cached_normal_list=cached_normal_list,
                )
                ds_res.append({"cat": cat, "auroc": auroc, "aupr": aupr})
                print(f"  {cat}: AUROC={auroc:.4f}, AUPR={aupr:.4f}")

            avg_auroc = float(np.mean([r["auroc"] for r in ds_res]))
            avg_aupr = float(np.mean([r["aupr"] for r in ds_res]))
            shot_results[ds] = {"auroc": avg_auroc, "aupr": avg_aupr, "details": ds_res}
            print(f"\n  >>> {ds.upper()} | {shot}-shot -> AUROC: {avg_auroc:.4f}, AUPR: {avg_aupr:.4f}")

        results[shot] = shot_results

    # 保存结果
    ckpt_name = CKPT_PATH.stem
    results_json_path = RESULTS_DIR / f"{ckpt_name}_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 结果已保存: {results_json_path}")

    # 打印汇总
    print("\n" + "=" * 72)
    print("测试结果汇总")
    print("=" * 72)
    for shot in SHOT_LIST:
        for ds in TEST_DATASETS:
            auroc = results[shot][ds]["auroc"]
            aupr = results[shot][ds]["aupr"]
            print(f"  {ds.upper()} | {shot}-shot -> AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
