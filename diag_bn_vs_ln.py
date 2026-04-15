#!/usr/bin/env python3
"""消融实验：验证 VA 中 LayerNorm 替换的影响

当前 VA checkpoint 是在 BatchNorm1d 下训练的，
推理时替换为 LayerNorm 会导致分布不匹配。
本脚本对比两种推理路径。
"""
import sys
sys.path.insert(0, r"d:/Data/Downloads/InCTRL")

import json
import torch
import numpy as np
from torchvision import transforms
import test_all_models as t
from open_clip.config.defaults import get_cfg
from open_clip.model import get_cast_dtype
from open_clip import model as _model_mod
from open_clip import visual_adapter as va_mod

DEVICE = t.DEVICE
SHOT = 4

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

def build_va_model(checkpoint_path, use_bn_in_inference=True, device=DEVICE):
    """构建 VA 模型，可选择推理时使用的归一化方式"""
    model_config_path = t.PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    cfg.VISUAL_ADAPTER.ENABLE = True

    # 临时修改 ResMLP 的归一化层类型
    old_norm = va_mod.ResMLP.__init__
    if use_bn_in_inference:
        # 让 ResMLP 用 BatchNorm1d（与训练一致）
        def patched_init(self, c_in, reduction=4):
            torch.nn.Module.__init__(self)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(c_in, c_in // reduction, bias=False),
                torch.nn.BatchNorm1d(c_in // reduction),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(c_in // reduction, c_in, bias=False),
            )
    else:
        # 用 LayerNorm（当前推理配置）
        def patched_init(self, c_in, reduction=4):
            torch.nn.Module.__init__(self)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(c_in, c_in // reduction, bias=False),
                torch.nn.LayerNorm(c_in // reduction),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(c_in // reduction, c_in, bias=False),
            )
    va_mod.ResMLP.__init__ = patched_init

    model = _model_mod.InCTRL(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    # 恢复原始 ResMLP
    va_mod.ResMLP.__init__ = old_norm

    return model.to(device)

def evaluate_model_simple(model, tokenizer, transform, shot, device):
    """在 VISA 上快速评估"""
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TEST.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = device == "cuda"
    cfg.shot = shot

    categories = sorted([
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ])

    json_dir = t.DATA_ROOT / "AD_json" / "visa"
    all_aurocs = []

    for cat in categories:
        cfg.val_normal_json_path = [str(json_dir / f"{cat}_val_normal.json")]
        cfg.val_outlier_json_path = [str(json_dir / f"{cat}_val_outlier.json")]
        val_loader = t.ds_loader.construct_loader(cfg, "test", transform)

        fs_pt = t.find_few_shot_pt("visa", cat, shot)
        cached_normal = t.build_cached_normal_list(fs_pt, device)

        labels_all, preds_all = t.collect_scores(model, tokenizer, val_loader, device, cached_normal)
        auroc, ap, _, _ = t.safe_binary_metrics(labels_all, preds_all)
        if auroc is not None:
            all_aurocs.append(auroc)
            print(f"  {cat:16s} AUROC={auroc:.4f}")

    return float(np.mean(all_aurocs)) if all_aurocs else None

if __name__ == "__main__":
    tokenizer = t.open_clip.get_tokenizer("ViT-B-16-plus-240")
    transform = get_transform()
    ckpt = t.resolve_checkpoint_file(SHOT, "VA")
    print(f"[INFO] 使用 checkpoint: {ckpt}")

    # 实验1: 当前推理配置 (LayerNorm)
    print("\n[EXP] 推理使用 LayerNorm (当前配置):")
    model_ln = build_va_model(ckpt, use_bn_in_inference=False)
    macro_auroc_ln = evaluate_model_simple(model_ln, tokenizer, transform, SHOT, DEVICE)
    print(f"  => macro AUROC={macro_auroc_ln:.4f}")
    del model_ln
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 实验2: 恢复 BatchNorm1d (与训练一致)
    print("\n[EXP] 推理使用 BatchNorm1d (与训练一致):")
    model_bn = build_va_model(ckpt, use_bn_in_inference=True)
    macro_auroc_bn = evaluate_model_simple(model_bn, tokenizer, transform, SHOT, DEVICE)
    print(f"  => macro AUROC={macro_auroc_bn:.4f}")
    del model_bn

    print(f"\n=== 消融结果 ===")
    print(f"  LayerNorm AUROC  = {macro_auroc_ln:.4f}")
    print(f"  BatchNorm1d AUROC = {macro_auroc_bn:.4f}")
    diff = macro_auroc_bn - macro_auroc_ln
    print(f"  Delta (BN - LN)  = {diff:+.4f}")

    if macro_auroc_bn > macro_auroc_ln:
        print("\n结论: 推理时用 BatchNorm1d 效果更好 -> 当前 LayerNorm 替换是 VA 退化的主因")
        print("建议: 重新训练 VA，从一开始就用 LayerNorm，或保持 BatchNorm1d 但增大推理 batch size")
    else:
        print("\n结论: LayerNorm 替换并非主因 -> VA 退化有其他原因")
