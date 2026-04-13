#!/usr/bin/env python3
"""
评估 6 个模型（Baseline/VA × 2/4/8-shot）在 AITEX、ELPV、VISA 上的表现，
并只保存结果 JSON（不绘图）。

输出目录：
- results/2/metrics_and_curves.json
- results/4/metrics_and_curves.json
- results/8/metrics_and_curves.json
- results/comparison_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip
from open_clip.config.defaults import get_cfg
from open_clip.model import get_cast_dtype
from datasets import loader as ds_loader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOTS = [2, 4, 8]
MODEL_VARIANTS = {
    "Baseline": {
        "checkpoint_root": PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec",
        "enable_va": False,
    },
    "VA": {
        "checkpoint_root": PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec_VA",
        "enable_va": True,
    },
}

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
FEW_SHOT_ROOT = PROJECT_ROOT / "few-shot samples"

DATASET_SPECS = {
    "aitex": {
        "json_dir": DATA_ROOT / "AD_json" / "aitex",
        "few_shot_dir": FEW_SHOT_ROOT / "AITEX" / "AITEX",
        "single_category": "AITEX",
    },
    "elpv": {
        "json_dir": DATA_ROOT / "AD_json" / "elpv",
        "few_shot_dir": FEW_SHOT_ROOT / "elpv" / "elpv",
        "single_category": "elpv",
    },
    "visa": {
        "json_dir": DATA_ROOT / "AD_json" / "visa",
        "few_shot_dir": FEW_SHOT_ROOT / "visa" / "visa",
        "single_category": None,
    },
}


def _convert_to_rgb(image):
    return image.convert("RGB") if hasattr(image, "convert") else image


def get_transform():
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


def find_categories(dataset_name):
    spec = DATASET_SPECS[dataset_name]
    if spec["single_category"] is not None:
        return [spec["single_category"]]

    cats = []
    # 【修改前】
    # for f in sorted(spec["json_dir"].glob("*_val_normal.json")):
    #     cats.append(f.name.replace("_val_normal.json", ""))
    
    # 【修改后】：读取真正的测试集文件，并排除掉名字里带有 "val_" 的小样本文件
    for f in sorted(spec["json_dir"].glob("*_normal.json")):
        if "val_" in f.name:
            continue
        cats.append(f.name.replace("_normal.json", ""))
        
    return cats


def resolve_checkpoint_file(shot, variant_name):
    root = MODEL_VARIANTS[variant_name]["checkpoint_root"] / str(shot)
    candidates = [
        root / "checkpoint",
        root / "checkpoint.pth",
        root / "checkpoint.pyth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 checkpoint: shot={shot}, variant={variant_name}, in {root}")


def build_model(enable_va, checkpoint_path, device):
    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    cfg.VISUAL_ADAPTER.ENABLE = bool(enable_va)

    from open_clip import model as _model_mod

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
    return model.to(device)


def find_few_shot_pt(dataset_name, category_name, shot):
    shot_dir = DATASET_SPECS[dataset_name]["few_shot_dir"] / str(shot)
    if not shot_dir.exists():
        raise FileNotFoundError(f"few-shot 目录不存在: {shot_dir}")

    direct = shot_dir / f"{category_name}.pt"
    if direct.exists():
        return direct

    for candidate in shot_dir.glob("*.pt"):
        if candidate.stem.lower() == category_name.lower():
            return candidate

    raise FileNotFoundError(
        f"few-shot pt 不存在: dataset={dataset_name}, category={category_name}, shot={shot}, dir={shot_dir}"
    )


def build_cached_normal_list(few_shot_pt, device):
    tensors = torch.load(few_shot_pt)
    return [t.to(device) for t in tensors]


@torch.no_grad()
def collect_scores(model, tokenizer, loader, device, cached_normal_list):
    model.eval()
    preds_all = []
    labels_all = []

    for inputs, types, labels in tqdm(loader, desc="[TEST] Batch", leave=False):
        labels = labels.to(device)
        preds, _ = model(tokenizer, inputs, types, cached_normal_list)
        preds_all.extend(preds.detach().cpu().float().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    return np.array(labels_all), np.array(preds_all)


def safe_binary_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return None, None, None, None

    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    return auroc, ap, fpr.tolist(), tpr.tolist()


def fmt_metric(v):
    return "N/A" if v is None else f"{v:.4f}"


def evaluate_model_on_dataset(model, tokenizer, transform, dataset_name, shot, device):
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = device == "cuda"
    cfg.shot = shot

    categories = find_categories(dataset_name)
    all_labels = []
    all_scores = []
    details = []

    for cat in categories:
        json_dir = DATASET_SPECS[dataset_name]["json_dir"]
        
        # 【修改前】
        # cfg.val_normal_json_path = [str(json_dir / f"{cat}_val_normal.json")]
        # cfg.val_outlier_json_path = [str(json_dir / f"{cat}_val_outlier.json")]
        
        # 【修改后】：直接读取 _normal.json 和 _outlier.json 完整测试集
        cfg.val_normal_json_path = [str(json_dir / f"{cat}_normal.json")]
        cfg.val_outlier_json_path = [str(json_dir / f"{cat}_outlier.json")]

        val_loader = ds_loader.construct_loader(cfg, "test", transform)
        fs_pt = find_few_shot_pt(dataset_name, cat, shot)
        cached_normal = build_cached_normal_list(fs_pt, device)

        labels, scores = collect_scores(model, tokenizer, val_loader, device, cached_normal)
        cat_auroc, cat_ap, _, _ = safe_binary_metrics(labels, scores)
        details.append(
            {
                "category": cat,
                "n_samples": int(labels.shape[0]),
                "auroc": cat_auroc,
                "ap": cat_ap,
            }
        )

        all_labels.append(labels)
        all_scores.append(scores)

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    ds_auroc, ds_ap, ds_fpr, ds_tpr = safe_binary_metrics(all_labels, all_scores)

    return {
        "auroc": ds_auroc,
        "ap": ds_ap,
        "fpr": ds_fpr,
        "tpr": ds_tpr,
        "n_samples": int(all_labels.shape[0]),
        "details": details,
    }


def evaluate_all_and_save_json():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    transform = get_transform()
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")

    all_results = {}

    for shot in SHOTS:
        shot_key = str(shot)
        print(f"\n{'=' * 80}")
        print(f"[INFO] 开始评估 {shot}-shot")
        print(f"{'=' * 80}")

        shot_dir = RESULTS_ROOT / shot_key
        shot_dir.mkdir(parents=True, exist_ok=True)

        shot_results = {}
        for variant_name, spec in MODEL_VARIANTS.items():
            ckpt_path = resolve_checkpoint_file(shot, variant_name)
            print(f"[INFO] 模型: {variant_name} | checkpoint: {ckpt_path}")

            model = build_model(spec["enable_va"], ckpt_path, DEVICE)
            variant_results = {}

            for ds in ["aitex", "elpv", "visa"]:
                print(f"[INFO] 评估: shot={shot}, variant={variant_name}, dataset={ds}")
                ds_result = evaluate_model_on_dataset(model, tokenizer, transform, ds, shot, DEVICE)
                variant_results[ds] = ds_result
                print(
                    f"  -> {ds.upper()}: AUROC={fmt_metric(ds_result['auroc'])} AP={fmt_metric(ds_result['ap'])} "
                    f"(n={ds_result['n_samples']})"
                )

            shot_results[variant_name] = variant_results
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        all_results[shot_key] = shot_results

        shot_json = shot_dir / "metrics_and_curves.json"
        with open(shot_json, "w", encoding="utf-8") as f:
            json.dump({"shot": shot, "results": shot_results}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 保存: {shot_json}")

        print(f"[INFO] 已保存 {shot}-shot JSON")

    merged_json = RESULTS_ROOT / "comparison_results.json"
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 保存汇总: {merged_json}")

    print("\n[INFO] 全部评估完成（仅JSON输出）")
    print(f"[INFO] 结果目录: {RESULTS_ROOT}")


if __name__ == "__main__":
    evaluate_all_and_save_json()
