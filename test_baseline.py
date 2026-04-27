#!/usr/bin/env python3
"""
评估 Baseline 模型在 AITEX、ELPV、VISA 上的跨 shot 表现。

只使用 checkpoints/InCTRL_trained_on_MVTec 下的 2/4/8-shot checkpoint。
每个 checkpoint 都会分别使用 2/4/8-shot few-shot samples 进行测试。

输出目录：
- results/baseline/metrics_and_curves.json
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
from datasets import loader as ds_loader
from open_clip.config.defaults import get_cfg
from open_clip.model import get_cast_dtype


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SHOTS = [2, 4, 8]
EVAL_SHOTS = [2, 4, 8]

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec"
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "baseline"
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
    return transforms.Compose(
        [
            transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(240),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def find_categories(dataset_name):
    spec = DATASET_SPECS[dataset_name]
    if spec["single_category"] is not None:
        return [spec["single_category"]]

    cats = []
    for f in sorted(spec["json_dir"].glob("*_val_normal.json")):
        cats.append(f.name.replace("_val_normal.json", ""))
    return cats


def resolve_checkpoint_file(model_shot):
    root = CHECKPOINT_ROOT / str(model_shot)
    candidates = [
        root / "checkpoint",
        root / "checkpoint.pth",
        root / "checkpoint.pyth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 checkpoint: model_shot={model_shot}, in {root}")


def build_baseline_model(checkpoint_path, device):
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

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(
            "Baseline checkpoint 与当前模型结构不完全匹配："
            f"missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
        )
    return model.to(device)


def find_few_shot_pt(dataset_name, category_name, eval_shot):
    shot_dir = DATASET_SPECS[dataset_name]["few_shot_dir"] / str(eval_shot)
    if not shot_dir.exists():
        raise FileNotFoundError(f"few-shot 目录不存在: {shot_dir}")

    direct = shot_dir / f"{category_name}.pt"
    if direct.exists():
        return direct

    for candidate in shot_dir.glob("*.pt"):
        if candidate.stem.lower() == category_name.lower():
            return candidate

    raise FileNotFoundError(
        "few-shot pt 不存在: "
        f"dataset={dataset_name}, category={category_name}, eval_shot={eval_shot}, dir={shot_dir}"
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


def evaluate_model_on_dataset(model, tokenizer, transform, dataset_name, eval_shot, device):
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TEST.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = device == "cuda"
    cfg.shot = eval_shot

    categories = find_categories(dataset_name)
    all_labels = []
    all_scores = []
    details = []

    for cat in categories:
        json_dir = DATASET_SPECS[dataset_name]["json_dir"]

        cfg.val_normal_json_path = [str(json_dir / f"{cat}_val_normal.json")]
        cfg.val_outlier_json_path = [str(json_dir / f"{cat}_val_outlier.json")]

        val_loader = ds_loader.construct_loader(cfg, "test", transform)
        fs_pt = find_few_shot_pt(dataset_name, cat, eval_shot)
        cached_normal = build_cached_normal_list(fs_pt, device)

        labels, scores = collect_scores(model, tokenizer, val_loader, device, cached_normal)
        cat_auroc, cat_ap, cat_fpr, cat_tpr = safe_binary_metrics(labels, scores)
        details.append(
            {
                "category": cat,
                "n_samples": int(labels.shape[0]),
                "auroc": cat_auroc,
                "ap": cat_ap,
                "fpr": cat_fpr,
                "tpr": cat_tpr,
            }
        )

        all_labels.append(labels)
        all_scores.append(scores)

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    micro_auroc, micro_ap, ds_fpr, ds_tpr = safe_binary_metrics(all_labels, all_scores)

    valid_aurocs = [d["auroc"] for d in details if d["auroc"] is not None]
    valid_aps = [d["ap"] for d in details if d["ap"] is not None]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None
    macro_ap = float(np.mean(valid_aps)) if valid_aps else None

    return {
        "auroc": macro_auroc,
        "ap": macro_ap,
        "fpr": ds_fpr,
        "tpr": ds_tpr,
        "micro_auroc": micro_auroc,
        "micro_ap": micro_ap,
        "n_samples": int(all_labels.shape[0]),
        "details": details,
    }


def evaluate_baseline_and_save_json():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    transform = get_transform()
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")

    all_results = {}

    for model_shot in MODEL_SHOTS:
        model_shot_key = str(model_shot)
        ckpt_path = resolve_checkpoint_file(model_shot)

        print(f"\n{'=' * 80}")
        print(f"[INFO] 加载 Baseline {model_shot}-shot checkpoint: {ckpt_path}")
        print(f"{'=' * 80}")

        model = build_baseline_model(ckpt_path, DEVICE)
        model_results = {}

        for eval_shot in EVAL_SHOTS:
            eval_shot_key = str(eval_shot)
            print(f"\n[INFO] 测试 few-shot prompts: {eval_shot}-shot")

            eval_results = {}
            for ds in ["aitex", "elpv", "visa"]:
                print(
                    f"[INFO] 评估: model_shot={model_shot}, eval_shot={eval_shot}, dataset={ds}"
                )
                ds_result = evaluate_model_on_dataset(
                    model, tokenizer, transform, ds, eval_shot, DEVICE
                )
                eval_results[ds] = ds_result
                print(
                    f"  -> {ds.upper()}: AUROC={fmt_metric(ds_result['auroc'])} "
                    f"AP={fmt_metric(ds_result['ap'])} (n={ds_result['n_samples']})"
                )

            model_results[eval_shot_key] = eval_results

        all_results[model_shot_key] = model_results

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    merged_json = RESULTS_ROOT / "metrics_and_curves.json"
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Baseline",
                "mode": "official_baseline_reproduction",
                "checkpoint_root": str(CHECKPOINT_ROOT),
                "model_type": "InCTRL",
                "model_shots": MODEL_SHOTS,
                "eval_shots": EVAL_SHOTS,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[INFO] 保存汇总: {merged_json}")
    print("[INFO] Baseline 全部评估完成（仅 JSON 输出）")


if __name__ == "__main__":
    evaluate_baseline_and_save_json()
