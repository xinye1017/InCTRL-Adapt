#!/usr/bin/env python3
"""
InCTRL 本地训练脚本
使用本地 GPU 和数据集进行训练
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip
from open_clip.model import get_cast_dtype
from open_clip.config.defaults import get_cfg
from open_clip.inctrl_pqa_fused import InCTRLPQA
from open_clip.inctrl_pqa_losses import compute_training_loss
from datasets import loader as ds_loader
from datasets.new_utlis import worker_init_fn_seed, BalancedBatchSampler
from binary_focal_loss import BinaryFocalLoss

# ============================================================================
# 配置
# ============================================================================

# 路径配置
DATA_ROOT = PROJECT_ROOT / "data"
FEW_SHOT_ROOT = PROJECT_ROOT / "few-shot samples"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 预训练模型路径
CKPT_NAME = "vit_b_16_plus_240-laion400m_e32-699c4b84.pt"
LOCAL_CKPT = PROJECT_ROOT / CKPT_NAME

# 训练配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 48
STEPS_PER_EPOCH = 100
LR = 1e-3
N_EPOCHS = 10
WEIGHT_DECAY = 0.0
TRAIN_SHOT = 4
EVAL_SHOTS = [2, 4, 8]
DEFAULT_TRAIN_DATASETS = ["mvtec", "visa"]
# Phase 1 defaults (2026-04-21): see open_clip/config/defaults.py::_C.PQA.*
IMAGE_LOSS_WEIGHT = 0.0
PQA_LOSS_WEIGHT = 0.5
MASK_LOSS_WEIGHT = 1.0
LOCAL_MIL_LOSS_WEIGHT = 0.0
LOCAL_MIL_TOPK_RATIO = 0.01
PRIOR_LOSS_WEIGHT = 0.0
TEXT_LOGIT_SCALE = 10.0


def get_default_num_workers():
    """Choose a conservative DataLoader default for local Windows and Linux CUDA runs."""
    cpu_count = os.cpu_count() or 1
    if os.name == "nt":
        return 0
    if DEVICE == "cuda":
        return min(cpu_count, max(2, min(4, cpu_count // 4)))
    return min(cpu_count, max(1, min(2, cpu_count // 4)))


DEFAULT_NUM_WORKERS = get_default_num_workers()

# 跨域测试映射。严格留一法:训练集域绝不出现在该集合对应的评估列表中,
# 从根上阻断同域泄漏。任何新增条目必须遵守 train_dataset not in value。
TEST_DATASETS_BY_TRAIN = {
    "mvtec": ["aitex", "elpv", "visa"],
    "visa": ["mvtec"],
}
# Static guard: catches accidental same-domain entries at import time.
for _train_ds, _test_dss in TEST_DATASETS_BY_TRAIN.items():
    if _train_ds in _test_dss:
        raise AssertionError(
            f"Same-domain evaluation leak detected for train={_train_ds}: "
            f"test list {_test_dss} contains the training dataset. "
            "Cross-domain protocol (InCTRL paper) forbids same-domain eval."
        )
del _train_ds, _test_dss

FEW_SHOT_DATASET_ALIASES = {
    "mvtec": "mvtecad",
    "aitex": "AITEX",
}

# Published InCTRL in-domain baselines (AUROC). Source:
# ``reports/original_inctrl_baseline.md``. Used only for automatic delta reporting
# in the cross-shot summary analytics block — NOT for any training decision.
# Missing entries simply disable delta reporting for that (dataset, shot) pair.
_PUBLISHED_BASELINE_AUROC = {
    "aitex": {0: 0.733, 2: 0.761, 4: 0.790, 8: 0.806},
    "elpv":  {0: 0.733, 2: 0.839, 4: 0.846, 8: 0.872},
    "visa":  {0: 0.781, 2: 0.858, 4: 0.877, 8: 0.887},
    "mvtec": {0: 0.912, 2: 0.940, 4: 0.945, 8: 0.953},
}
# Phase 1 (2026-04-21) cross-domain 4-shot exit thresholds for MVTec-trained runs.
# Derived from reports/original_inctrl_baseline.md and the Phase 1 plan: each
# target dataset should at minimum recover its published 0-shot number (or a
# modest discount) after MVTec-only cross-domain training.
_PHASE1_EXIT_THRESHOLDS_4SHOT = {
    "aitex": 0.73,
    "elpv":  0.82,
    "visa":  0.80,
}

# ============================================================================
# 工具函数
# ============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _convert_to_rgb(image):
    """将图像转换为RGB格式（兼容PIL）"""
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


# ============================================================================
# 数据集准备
# ============================================================================

def collect_training_jsons(dataset_name):
    json_dir = DATA_ROOT / "AD_json" / dataset_name
    if not json_dir.exists():
        raise FileNotFoundError(f"训练数据集目录不存在: {json_dir}")

    normal_jsons = [
        str(f) for f in sorted(json_dir.glob("*_normal.json"))
        if "val_" not in f.name
    ]
    outlier_jsons = [
        str(f) for f in sorted(json_dir.glob("*_outlier.json"))
        if "val_" not in f.name
    ]
    if not normal_jsons or not outlier_jsons:
        raise RuntimeError(
            f"{dataset_name} 缺少训练 JSON: normal={len(normal_jsons)}, "
            f"outlier={len(outlier_jsons)}"
        )
    return normal_jsons, outlier_jsons


def collect_test_categories(dataset_name):
    json_dir = DATA_ROOT / "AD_json" / dataset_name
    if not json_dir.exists():
        raise FileNotFoundError(f"测试数据集目录不存在: {json_dir}")
    categories = sorted(
        f.name.replace("_val_normal.json", "")
        for f in json_dir.glob("*_val_normal.json")
    )
    if not categories:
        raise RuntimeError(f"{dataset_name} 缺少 val_normal JSON")
    return categories


def resolve_few_shot_dataset_name(dataset_name):
    return FEW_SHOT_DATASET_ALIASES.get(dataset_name.lower(), dataset_name)


def prepare_dataset_registry(train_datasets):
    """只收集 JSON 路径，不原地改写 data/AD_json。"""
    registry = {
        "train_jsons": {},
        "test_categories": {},
    }
    all_test_datasets = []
    for train_dataset in train_datasets:
        all_test_datasets.extend(TEST_DATASETS_BY_TRAIN[train_dataset])
    all_test_datasets = list(dict.fromkeys(all_test_datasets))

    for train_dataset in train_datasets:
        normal_jsons, outlier_jsons = collect_training_jsons(train_dataset)
        registry["train_jsons"][train_dataset] = {
            "normal": normal_jsons,
            "outlier": outlier_jsons,
        }
        print(
            f"[INFO] {train_dataset.upper()} 训练 JSON: "
            f"正常={len(normal_jsons)}, 异常={len(outlier_jsons)}"
        )

    for test_dataset in all_test_datasets:
        categories = collect_test_categories(test_dataset)
        registry["test_categories"][test_dataset] = categories
        print(f"[INFO] {test_dataset.upper()} 测试类别 ({len(categories)}): {categories}")

    return registry


# ============================================================================
# 模型
# ============================================================================

def build_model(device, text_logit_scale=TEXT_LOGIT_SCALE):
    """构建 InCTRL 模型"""
    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    cfg.PQA.TEXT_LOGIT_SCALE = float(text_logit_scale)
    model = InCTRLPQA(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    if LOCAL_CKPT.exists():
        print(f"[INFO] 加载预训练权重: {LOCAL_CKPT}")
        checkpoint = torch.load(LOCAL_CKPT, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"[WARNING] 预训练权重不存在: {LOCAL_CKPT}")

    return model.to(device)


# ============================================================================
# 训练工具
# ============================================================================

def get_trainable_parameters(model):
    """获取 PQA-only 路径下可训练参数。"""
    params = model.get_trainable_parameters()
    params = [param for param in params if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for the PQA-only path.")
    return params


def build_optimizer(model, lr=1e-3, weight_decay=0.0):
    return torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, n_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )


# ============================================================================
# 测试相关
# ============================================================================

def find_fs_pt(ds, cat, shot):
    """查找 few-shot pt 文件
    路径结构: few-shot samples/{ds}/{ds}/{shot}/{cat}.pt
    例如: few-shot samples/visa/visa/2/candle.pt
    """
    ds_key = resolve_few_shot_dataset_name(ds)
    for path in FEW_SHOT_ROOT.iterdir():
        if path.name.lower() == ds_key.lower():
            # 在该目录下查找匹配 ds 名称的子目录
            for subpath in path.iterdir():
                if subpath.is_dir() and subpath.name.lower() == ds_key.lower():
                    shot_dir = subpath / str(shot)
                    if shot_dir.exists():
                        pt_path = shot_dir / f"{cat}.pt"
                        if pt_path.exists():
                            return pt_path
                        # 模糊匹配
                        for candidate in shot_dir.glob("*.pt"):
                            if candidate.stem.lower() == cat.lower():
                                return candidate
                    # 也尝试直接在 subpath/shot 目录下查找
                    direct_shot_dir = path / str(shot)
                    if direct_shot_dir.exists():
                        pt_path = direct_shot_dir / f"{cat}.pt"
                        if pt_path.exists():
                            return pt_path
                        for candidate in direct_shot_dir.glob("*.pt"):
                            if candidate.stem.lower() == cat.lower():
                                return candidate
    raise FileNotFoundError(f"Cannot find few-shot pt for {ds} - {cat} (shot={shot})")


def build_cached_normal_img_features(model, few_shot_path, device):
    """构建用于测试的 normal_list (返回 list of tensors)"""
    few_shot_list = torch.load(few_shot_path, map_location="cpu")
    # 返回 list of tensors，每个 tensor 是 [3, 240, 240]
    return [tensor.to(device) for tensor in few_shot_list]


@torch.no_grad()
def build_cached_prompt_features(model, few_shot_path, device, category_name=None):
    """预编码 few-shot prompt 特征，避免评估时重复跑 prompt visual tower。"""
    normal_list = build_cached_normal_img_features(model, few_shot_path, device)
    return model.build_prompt_feature_cache(normal_list=normal_list, category=category_name)


@torch.no_grad()
def build_cached_text_prototypes(model, category_name, device):
    """预编码类别文本原型，避免评估时每个 batch 重复跑 text tower。"""
    return model.build_text_prototype_cache(obj_types=[category_name], device=torch.device(device))


def split_query_prompt_inputs(inputs, device):
    query_image = inputs[0].to(device)
    prompt_images = torch.stack(inputs[1:], dim=1).to(device)
    return query_image, prompt_images


def unpack_batch(batch):
    if len(batch) == 4:
        inputs, types, labels, masks = batch
        return inputs, types, labels, masks
    inputs, types, labels = batch
    return inputs, types, labels, None


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    loader,
    device,
    cached_normal_list=None,
    prompt_feature_cache=None,
    text_prototype_cache=None,
    return_branch_metrics=False,
):
    """评估模型"""
    model.eval()
    preds_all, labels_all = [], []
    branch_preds = {
        "patch": [],
        "text": [],
        "pqa": [],
        "image": [],
    }

    for batch in tqdm(loader, desc="[TEST] Batch", leave=False):
        inputs, types, labels, _ = unpack_batch(batch)
        labels = labels.to(device)
        query_image, _ = split_query_prompt_inputs(inputs, device)
        outputs = model(
            query_image=query_image,
            normal_list=cached_normal_list,
            prompt_feature_cache=prompt_feature_cache,
            obj_types=types,
            text_prototype_cache=text_prototype_cache,
            return_aux=False,
            return_dict=True,
        )
        preds_all.extend(outputs["final_score"].detach().cpu().float().numpy())
        labels_all.extend(labels.cpu().numpy())
        for branch_name, output_key in [
            ("patch", "patch_score"),
            ("text", "text_score"),
            ("pqa", "pqa_score"),
            ("image", "image_score"),
        ]:
            branch_preds[branch_name].extend(outputs[output_key].detach().cpu().float().numpy())

    auroc = roc_auc_score(labels_all, preds_all)
    aupr = average_precision_score(labels_all, preds_all)
    if not return_branch_metrics:
        return float(auroc), float(aupr)

    branch_metrics = {}
    for branch_name, branch_values in branch_preds.items():
        if len(branch_values) != len(labels_all):
            continue
        branch_metrics[branch_name] = {
            "auroc": float(roc_auc_score(labels_all, branch_values)),
            "aupr": float(average_precision_score(labels_all, branch_values)),
        }
    branch_metrics["final"] = {"auroc": float(auroc), "aupr": float(aupr)}
    return float(auroc), float(aupr), branch_metrics


# ============================================================================
# 训练主函数
# ============================================================================

def run_experiment(
    dataset_registry,
    train_dataset,
    test_datasets,
    train_shot=TRAIN_SHOT,
    eval_shots=None,
    label=None,
    n_epochs=10,
    lr=1e-3,
    steps_per_epoch=100,
    batch_size=48,
    test_batch_size=1,
    num_workers=DEFAULT_NUM_WORKERS,
    max_test_categories=None,
    weight_decay=0.0,
    image_loss_weight=IMAGE_LOSS_WEIGHT,
    pqa_loss_weight=PQA_LOSS_WEIGHT,
    mask_loss_weight=MASK_LOSS_WEIGHT,
    local_mil_loss_weight=LOCAL_MIL_LOSS_WEIGHT,
    local_mil_topk_ratio=LOCAL_MIL_TOPK_RATIO,
    prior_loss_weight=PRIOR_LOSS_WEIGHT,
    text_logit_scale=TEXT_LOGIT_SCALE,
    resume_checkpoint=None,
    start_epoch=0,
):
    """运行一个训练域的 4-shot 模型，并用 cross-shots 评估。"""
    eval_shots = eval_shots or EVAL_SHOTS
    label = label or f"trained_on_{train_dataset}_shot_{train_shot}"
    # Runtime guard against accidental same-domain eval (defense in depth).
    if train_dataset in test_datasets:
        raise ValueError(
            f"Refusing to evaluate on same domain as training: "
            f"train_dataset={train_dataset!r} appears in test_datasets={test_datasets}. "
            "This would reintroduce in-domain leakage; use TEST_DATASETS_BY_TRAIN."
        )
    print(f"========== 实验 [{label}] ==========")
    print(
        f"配置: train_dataset={train_dataset}, test_datasets={test_datasets}, "
        f"train_shot={train_shot}, eval_shots={eval_shots}, n_epochs={n_epochs}, "
        f"lr={lr}, batch_size={batch_size}, num_workers={num_workers}, "
        f"weight_decay={weight_decay}, image_loss_weight={image_loss_weight}, "
        f"pqa_loss_weight={pqa_loss_weight}, mask_loss_weight={mask_loss_weight}, "
        f"local_mil_loss_weight={local_mil_loss_weight}, local_mil_topk_ratio={local_mil_topk_ratio}, "
        f"prior_loss_weight={prior_loss_weight}, text_logit_scale={text_logit_scale}"
    )
    print(f"设备: {DEVICE}")

    # 构建模型
    model = build_model(DEVICE, text_logit_scale=text_logit_scale)

    # 配置
    cfg = get_cfg()
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TEST.BATCH_SIZE = test_batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.MAX_EPOCH = n_epochs
    cfg.shot = train_shot
    cfg.steps_per_epoch = steps_per_epoch
    cfg.normal_json_path = dataset_registry["train_jsons"][train_dataset]["normal"]
    cfg.outlier_json_path = dataset_registry["train_jsons"][train_dataset]["outlier"]
    cfg.PQA.GLOBAL_LOSS_WEIGHT = pqa_loss_weight
    cfg.PQA.MASK_LOSS_WEIGHT = mask_loss_weight
    cfg.PQA.IMAGE_LOSS_WEIGHT = image_loss_weight
    cfg.PQA.LOCAL_MIL_LOSS_WEIGHT = local_mil_loss_weight
    cfg.PQA.LOCAL_MIL_TOPK_RATIO = local_mil_topk_ratio
    cfg.PQA.PRIOR_LOSS_WEIGHT = prior_loss_weight
    cfg.PQA.TEXT_LOGIT_SCALE = text_logit_scale
    cfg.DATA_LOADER.NUM_WORKERS = num_workers
    cfg.DATA_LOADER.PIN_MEMORY = DEVICE == "cuda"

    transform = get_transform()
    train_loader = ds_loader.construct_loader(cfg, "train", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    loss_fn = BinaryFocalLoss(logits=True).to(DEVICE)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, n_epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_dir = RESULTS_DIR / f"trained_on_{train_dataset}"
    checkpoint_dir = CHECKPOINT_DIR / f"trained_on_{train_dataset}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = result_dir / f"results_shot_{train_shot}.json"
    final_checkpoint_path = checkpoint_dir / f"final_model_shot_{train_shot}.pt"

    # 恢复训练
    resume_loss_history = []
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[INFO] 从检查点恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            elif "visual_optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["visual_optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif "visual_scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["visual_scheduler_state_dict"])
            if start_epoch == 0 and "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"])
            if "loss" in checkpoint:
                resume_loss_history = [float(x) for x in checkpoint["loss"]]
        else:
            model.load_state_dict(checkpoint, strict=False)
        model = model.to(DEVICE)
        # 调整学习率调度器的当前状态
        if not (
            isinstance(checkpoint, dict)
            and ("scheduler_state_dict" in checkpoint or "visual_scheduler_state_dict" in checkpoint)
        ):
            for _ in range(start_epoch):
                scheduler.step()
        print(f"[INFO] 从 epoch {start_epoch + 1} 继续训练")

    history_loss = []
    if resume_loss_history:
        history_loss = resume_loss_history
        print(f"[INFO] 已从 checkpoint 加载历史 loss 数据: {len(history_loss)} 个 epoch")

    # 训练循环
    completed_epoch = start_epoch
    for epoch in range(start_epoch, n_epochs):
        model.enable_pqa_training()
        trainable_params = get_trainable_parameters(model)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{n_epochs} | phase=pqa_fused | lr={current_lr:.8f}")

        model.train()
        epoch_loss = 0.0
        actual_steps = 0

        batch_pbar = tqdm(
            total=steps_per_epoch,
            desc=f"[TRAIN] Epoch {epoch + 1}/{n_epochs}",
            unit="batch",
            leave=False,
        )

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= steps_per_epoch:
                break

            inputs, types, labels, masks = unpack_batch(batch)
            labels = labels.to(DEVICE)
            if masks is not None:
                masks = masks.to(DEVICE, non_blocking=True)
            query_image, prompt_images = split_query_prompt_inputs(inputs, DEVICE)
            outputs = model(
                query_image=query_image,
                prompt_images=prompt_images,
                obj_types=types,
                return_aux=False,
                return_dict=True,
            )
            loss, loss_parts = compute_training_loss(
                outputs=outputs,
                labels=labels,
                loss_fn=loss_fn,
                masks=masks,
                image_loss_weight=image_loss_weight,
                pqa_loss_weight=pqa_loss_weight,
                mask_loss_weight=mask_loss_weight,
                local_mil_loss_weight=local_mil_loss_weight,
                local_mil_topk_ratio=local_mil_topk_ratio,
                prior_loss_weight=prior_loss_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            actual_steps += 1
            batch_pbar.update(1)
            batch_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "final": f"{loss_parts['final_loss'].item():.4f}",
                "image": f"{loss_parts['image_loss'].item():.4f}",
                "pqa": f"{loss_parts['pqa_loss'].item():.4f}",
                "mask": f"{loss_parts['pqa_mask_loss'].item():.4f}",
                "local_mil": f"{loss_parts['pqa_local_mil_loss'].item():.4f}",
                "prior": f"{loss_parts['prior_loss'].item():.4f}",
            })

        batch_pbar.close()
        scheduler.step()

        avg_loss = float(epoch_loss / max(actual_steps, 1))
        history_loss.append(avg_loss)
        completed_epoch = epoch + 1
        print(f"Epoch {epoch + 1} 完成 | avg_loss={avg_loss:.4f}")

    experiment_cfg = {
        "train_dataset": train_dataset,
        "test_datasets": test_datasets,
        "train_shot": train_shot,
        "eval_shots": eval_shots,
        "n_epochs": n_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "num_workers": num_workers,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": weight_decay,
        "image_loss_weight": image_loss_weight,
        "pqa_loss_weight": pqa_loss_weight,
        "mask_loss_weight": mask_loss_weight,
        "local_mil_loss_weight": local_mil_loss_weight,
        "local_mil_topk_ratio": local_mil_topk_ratio,
        "prior_loss_weight": prior_loss_weight,
        "text_logit_scale": text_logit_scale,
        "model_architecture": {
            "mode": "pqa_fused_loss_closure",
            "pqa_global_pooling": "learnable_gap_gmp",
            "decision_head": "softmax_convex_fusion",
        },
        "label": label,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg.dump(),
            "experiment_cfg": experiment_cfg,
            "epoch": completed_epoch,
            "loss": [float(x) for x in history_loss],
        },
        final_checkpoint_path,
    )
    print(f"[INFO] 最终检查点已保存: {final_checkpoint_path}")

    print("\n训练完成! 开始测试集评估...")

    # 测试评估
    results = {}
    for shot in eval_shots:
        print(f"\n===== 测试 {shot}-shot =====")
        cfg.shot = shot
        shot_results = {}

        for ds in test_datasets:
            ds_res = []
            print(f"\n[INFO] 测试数据集: {ds.upper()}")

            categories = dataset_registry["test_categories"][ds]
            if max_test_categories is not None:
                categories = categories[:max_test_categories]
            for cat in categories:
                cfg.val_normal_json_path = [str(DATA_ROOT / "AD_json" / ds / f"{cat}_val_normal.json")]
                cfg.val_outlier_json_path = [str(DATA_ROOT / "AD_json" / ds / f"{cat}_val_outlier.json")]
                val_loader = ds_loader.construct_loader(cfg, "test", transform)

                fs_pt = find_fs_pt(ds, cat, shot)
                prompt_feature_cache = build_cached_prompt_features(model, fs_pt, DEVICE, category_name=cat)
                text_prototype_cache = build_cached_text_prototypes(model, cat, DEVICE)
                eval_output = evaluate(
                    model,
                    tokenizer,
                    val_loader,
                    DEVICE,
                    prompt_feature_cache=prompt_feature_cache,
                    text_prototype_cache=text_prototype_cache,
                    return_branch_metrics=True,
                )
                auroc, aupr, branch_metrics = eval_output
                ds_res.append({
                    "cat": cat,
                    "auroc": auroc,
                    "aupr": aupr,
                    "branch_metrics": branch_metrics,
                })
                print(f"  {cat}: AUROC={auroc:.4f}, AUPR={aupr:.4f}")
                branch_summary = ", ".join(
                    f"{name}={metrics['auroc']:.4f}"
                    for name, metrics in branch_metrics.items()
                    if name != "final"
                )
                print(f"    branches AUROC: {branch_summary}")

            avg_auroc = float(np.mean([r["auroc"] for r in ds_res]))
            avg_aupr = float(np.mean([r["aupr"] for r in ds_res]))
            shot_results[ds] = {"auroc": avg_auroc, "aupr": avg_aupr, "details": ds_res}
            print(f"\n  >>> {ds.upper()} | {shot}-shot -> AUROC: {avg_auroc:.4f}, AUPR: {avg_aupr:.4f}")

        results[shot] = shot_results

    # 保存结果
    summary_data = {
        "label": label,
        "config": experiment_cfg,
        "loss": [float(x) for x in history_loss],
        "checkpoint_path": str(final_checkpoint_path.resolve()),
        "results": results,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 结果已保存: {results_json_path}")

    return {
        "label": label,
        "checkpoint_path": final_checkpoint_path,
        "results_json_path": results_json_path,
        "loss": history_loss,
        "results": results,
    }


def _build_summary_analytics(summary_rows, train_datasets):
    """Aggregate raw summary_rows into analyst-friendly views.

    Non-breaking: returns a dict that sits alongside the existing ``summary_rows``.
    Every value is plain JSON (no tensors) so the block is safe to serialize.

    Returned keys:
      - ``aggregated``: ``{pair: {"auroc": {shot: v}, "aupr": {shot: v}}}``
        One row per (train, test) pair, pivoted on shot. Removes the
        "scan 9 rows to build a table" friction.
      - ``baseline_deltas``: list of per-(pair, shot) rows containing ours,
        published in-domain baseline, delta vs in-domain, delta vs zero-shot.
      - ``phase1_exit``: only populated for MVTec-trained runs. Per-pair
        go/no-go status at shot=4 plus a ``_summary`` section describing
        whether *all* thresholds pass and which pairs fail.

    Missing published baselines degrade gracefully (delta becomes ``None``).
    """
    aggregated = {}
    for row in summary_rows:
        pair = f"{row['train_dataset']}->{row['test_dataset']}"
        entry = aggregated.setdefault(pair, {"auroc": {}, "aupr": {}})
        shot = int(row["eval_shot"])
        entry["auroc"][shot] = row["auroc"]
        entry["aupr"][shot] = row["aupr"]

    baseline_deltas = []
    for row in summary_rows:
        test_ds = row["test_dataset"]
        shot = int(row["eval_shot"])
        ref_in_domain = _PUBLISHED_BASELINE_AUROC.get(test_ds, {}).get(shot)
        ref_zero_shot = _PUBLISHED_BASELINE_AUROC.get(test_ds, {}).get(0)
        baseline_deltas.append({
            "pair": f"{row['train_dataset']}->{test_ds}",
            "shot": shot,
            "ours_auroc": row["auroc"],
            "published_in_domain_auroc": ref_in_domain,
            "delta_vs_in_domain": (
                round(row["auroc"] - ref_in_domain, 4) if ref_in_domain is not None else None
            ),
            "published_zero_shot_auroc": ref_zero_shot,
            "delta_vs_zero_shot": (
                round(row["auroc"] - ref_zero_shot, 4) if ref_zero_shot is not None else None
            ),
        })

    phase1_exit = {}
    if "mvtec" in train_datasets:
        for row in summary_rows:
            if row["train_dataset"] != "mvtec" or int(row["eval_shot"]) != 4:
                continue
            test_ds = row["test_dataset"]
            threshold = _PHASE1_EXIT_THRESHOLDS_4SHOT.get(test_ds)
            if threshold is None:
                continue
            phase1_exit[f"mvtec->{test_ds}"] = {
                "ours_4shot_auroc": row["auroc"],
                "exit_threshold": threshold,
                "margin": round(row["auroc"] - threshold, 4),
                "passes": row["auroc"] >= threshold,
            }
        per_pair = {k: v for k, v in phase1_exit.items() if k != "_summary"}
        if per_pair:
            phase1_exit["_summary"] = {
                "all_pass": all(v["passes"] for v in per_pair.values()),
                "failing_pairs": sorted(k for k, v in per_pair.items() if not v["passes"]),
            }

    return {
        "aggregated": aggregated,
        "baseline_deltas": baseline_deltas,
        "phase1_exit": phase1_exit,
    }


def run_all_experiments(
    train_datasets,
    train_shot=TRAIN_SHOT,
    eval_shots=None,
    n_epochs=10,
    lr=1e-3,
    steps_per_epoch=100,
    batch_size=48,
    test_batch_size=1,
    num_workers=DEFAULT_NUM_WORKERS,
    max_test_categories=None,
    weight_decay=0.0,
    image_loss_weight=IMAGE_LOSS_WEIGHT,
    pqa_loss_weight=PQA_LOSS_WEIGHT,
    mask_loss_weight=MASK_LOSS_WEIGHT,
    local_mil_loss_weight=LOCAL_MIL_LOSS_WEIGHT,
    local_mil_topk_ratio=LOCAL_MIL_TOPK_RATIO,
    prior_loss_weight=PRIOR_LOSS_WEIGHT,
    text_logit_scale=TEXT_LOGIT_SCALE,
    resume_checkpoint=None,
    start_epoch=0,
):
    eval_shots = eval_shots or EVAL_SHOTS
    train_datasets = [dataset.lower() for dataset in train_datasets]
    dataset_registry = prepare_dataset_registry(train_datasets)

    run_outputs = []
    summary_rows = []
    for train_dataset in train_datasets:
        test_datasets = TEST_DATASETS_BY_TRAIN[train_dataset]
        run_output = run_experiment(
            dataset_registry=dataset_registry,
            train_dataset=train_dataset,
            test_datasets=test_datasets,
            train_shot=train_shot,
            eval_shots=eval_shots,
            n_epochs=n_epochs,
            lr=lr,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            max_test_categories=max_test_categories,
            weight_decay=weight_decay,
            image_loss_weight=image_loss_weight,
            pqa_loss_weight=pqa_loss_weight,
            mask_loss_weight=mask_loss_weight,
            local_mil_loss_weight=local_mil_loss_weight,
            local_mil_topk_ratio=local_mil_topk_ratio,
            prior_loss_weight=prior_loss_weight,
            text_logit_scale=text_logit_scale,
            resume_checkpoint=resume_checkpoint,
            start_epoch=start_epoch,
        )
        run_outputs.append(run_output)
        for shot, shot_results in run_output["results"].items():
            for test_dataset, metrics in shot_results.items():
                if test_dataset == train_dataset:
                    # Defense in depth: never let a same-domain row slip into summary.
                    continue
                summary_rows.append({
                    "train_dataset": train_dataset,
                    "train_shot": train_shot,
                    "test_dataset": test_dataset,
                    "eval_shot": int(shot),
                    "auroc": metrics["auroc"],
                    "aupr": metrics["aupr"],
                    "checkpoint_path": str(run_output["checkpoint_path"].resolve()),
                    "results_json_path": str(run_output["results_json_path"].resolve()),
                })

    analytics = _build_summary_analytics(summary_rows, train_datasets)

    summary_path = RESULTS_DIR / f"cross_shot_train_shot_{train_shot}_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_datasets": train_datasets,
                "train_shot": train_shot,
                "eval_shots": eval_shots,
                "image_loss_weight": image_loss_weight,
                "pqa_loss_weight": pqa_loss_weight,
                "mask_loss_weight": mask_loss_weight,
                "local_mil_loss_weight": local_mil_loss_weight,
                "local_mil_topk_ratio": local_mil_topk_ratio,
                "prior_loss_weight": prior_loss_weight,
                "text_logit_scale": text_logit_scale,
                "protocol": "strict_cross_domain_only",
                "analytics": analytics,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[INFO] 跨 shot 汇总已保存: {summary_path}")
    phase1_exit = analytics.get("phase1_exit", {}).get("_summary")
    if phase1_exit:
        status = "PASS" if phase1_exit["all_pass"] else "FAIL"
        print(f"[INFO] Phase 1 退出状态: {status}")
        if not phase1_exit["all_pass"]:
            print(f"[INFO]   未达标对: {phase1_exit['failing_pairs']}")
    return {
        "runs": run_outputs,
        "summary_rows": summary_rows,
        "analytics": analytics,
        "summary_path": summary_path,
    }


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InCTRL 本地训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点路径，用于恢复训练")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="起始 epoch 编号（0-indexed，用于恢复训练）")
    parser.add_argument("--train-datasets", nargs="+", default=DEFAULT_TRAIN_DATASETS,
                        help="训练域，默认分别训练 mvtec 和 visa")
    parser.add_argument("--train-shot", type=int, default=TRAIN_SHOT,
                        help="训练时使用的 few-shot 数，默认 4")
    parser.add_argument("--eval-shots", nargs="+", type=int, default=EVAL_SHOTS,
                        help="测试时使用的 cross-shots，默认 2 4 8")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="训练 batch size")
    parser.add_argument("--test-batch-size", type=int, default=1,
                        help="测试 batch size；显存允许时可调大以加速 cross-shot 评估")
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH,
                        help="每个 epoch 的 batch 数")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help="训练 epoch 数")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"DataLoader worker 数，默认 {DEFAULT_NUM_WORKERS}；可按机器情况覆盖")
    parser.add_argument("--image-loss-weight", type=float, default=IMAGE_LOSS_WEIGHT,
                        help=f"图像级残差 LIRL 损失权重，默认 {IMAGE_LOSS_WEIGHT}")
    parser.add_argument("--pqa-loss-weight", type=float, default=PQA_LOSS_WEIGHT,
                        help=f"PQA 全局比较分支损失权重，默认 {PQA_LOSS_WEIGHT}")
    parser.add_argument("--mask-loss-weight", type=float, default=MASK_LOSS_WEIGHT,
                        help=f"PQA 局部 mask focal/dice 损失权重，默认 {MASK_LOSS_WEIGHT}")
    parser.add_argument("--local-mil-loss-weight", type=float, default=LOCAL_MIL_LOSS_WEIGHT,
                        help="无 mask 时 PQA 局部 MIL 弱监督权重，默认 0.0（关闭）")
    parser.add_argument("--local-mil-topk-ratio", type=float, default=LOCAL_MIL_TOPK_RATIO,
                        help="无 mask 局部 MIL top-k 比例，默认 0.01")
    parser.add_argument("--prior-loss-weight", type=float, default=PRIOR_LOSS_WEIGHT,
                        help="CLIP 文本零样本先验锚 KL 损失权重，默认 0.0（关闭）；跨域训练建议 0.1")
    parser.add_argument("--text-logit-scale", type=float, default=TEXT_LOGIT_SCALE,
                        help="scalar 融合中 text_logit 的温度，默认 10.0（text_score 仍用 100x 保持 CLIP 零样本概率）")
    parser.add_argument("--max-test-categories", type=int, default=None,
                        help="仅用于快速验证：限制每个测试域评估的类别数，默认评估全部类别")
    args = parser.parse_args()

    seed_everything(SEED)

    # 确认 GPU 可用
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA 不可用，将使用 CPU")

    # 实验配置
    print("\n实验配置:")
    print(f"TRAIN_DATASETS = {args.train_datasets}")
    print(f"TRAIN_SHOT = {args.train_shot}")
    print(f"EVAL_SHOTS = {args.eval_shots}")
    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"TEST_BATCH_SIZE = {args.test_batch_size}")
    print(f"STEPS_PER_EPOCH = {args.steps_per_epoch}")
    print(f"NUM_WORKERS = {args.num_workers}")
    print(f"IMAGE_LOSS_WEIGHT = {args.image_loss_weight}")
    print(f"PQA_LOSS_WEIGHT = {args.pqa_loss_weight}")
    print(f"MASK_LOSS_WEIGHT = {args.mask_loss_weight}")
    print(f"LOCAL_MIL_LOSS_WEIGHT = {args.local_mil_loss_weight}")
    print(f"LOCAL_MIL_TOPK_RATIO = {args.local_mil_topk_ratio}")
    print(f"PRIOR_LOSS_WEIGHT = {args.prior_loss_weight}")
    print(f"TEXT_LOGIT_SCALE = {args.text_logit_scale}")
    print(f"MAX_TEST_CATEGORIES = {args.max_test_categories}")
    print(f"LR = {LR}")
    print(f"N_EPOCHS = {args.epochs}")
    if args.resume:
        print(f"恢复检查点: {args.resume}")
        print(f"起始 epoch: {args.start_epoch + 1}")

    # 运行实验
    main_output = run_all_experiments(
        train_datasets=args.train_datasets,
        train_shot=args.train_shot,
        eval_shots=args.eval_shots,
        n_epochs=args.epochs,
        lr=LR,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        max_test_categories=args.max_test_categories,
        weight_decay=WEIGHT_DECAY,
        image_loss_weight=args.image_loss_weight,
        pqa_loss_weight=args.pqa_loss_weight,
        mask_loss_weight=args.mask_loss_weight,
        local_mil_loss_weight=args.local_mil_loss_weight,
        local_mil_topk_ratio=args.local_mil_topk_ratio,
        prior_loss_weight=args.prior_loss_weight,
        text_logit_scale=args.text_logit_scale,
        resume_checkpoint=args.resume,
        start_epoch=args.start_epoch,
    )

    print("\n" + "=" * 72)
    print("训练和评估完成!")
    print("=" * 72)
    print(f"汇总文件: {main_output['summary_path']}")

    # 打印最终结果
    print("\n最终结果汇总:")
    for row in main_output["summary_rows"]:
        print(
            f"  train={row['train_dataset'].upper()} "
            f"train_shot={row['train_shot']} "
            f"-> test={row['test_dataset'].upper()} "
            f"eval_shot={row['eval_shot']} "
            f"| AUROC={row['auroc']:.4f}, AUPR={row['aupr']:.4f}"
        )
