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
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip
from open_clip.model import get_cast_dtype
from open_clip.config.defaults import get_cfg
from open_clip.inctrl_three_adapters import InCTRLWithAdapters
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
IMAGE_LOSS_WEIGHT = 1.0
PQA_LOSS_WEIGHT = 1.0
TEXT_REG_WEIGHT = 0.01


def get_default_num_workers():
    """Choose a conservative DataLoader default for local Windows and Linux CUDA runs."""
    cpu_count = os.cpu_count() or 1
    if os.name == "nt":
        return 0
    if DEVICE == "cuda":
        return min(cpu_count, max(2, min(4, cpu_count // 4)))
    return min(cpu_count, max(1, min(2, cpu_count // 4)))


DEFAULT_NUM_WORKERS = get_default_num_workers()

# 跨域测试映射
TEST_DATASETS_BY_TRAIN = {
    "mvtec": ["visa"],
    "visa": ["mvtec"],
}

FEW_SHOT_DATASET_ALIASES = {
    "mvtec": "mvtecad",
    "aitex": "AITEX",
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

def build_model(device):
    """构建 InCTRL 模型"""
    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    model = InCTRLWithAdapters(
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

def get_trainable_parameters(model, phase):
    """获取当前 phase 下可训练参数"""
    if phase == "visual":
        params = model.get_visual_parameters()
    elif phase == "text":
        params = model.get_text_parameters()
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    params = [param for param in params if param.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters found for phase={phase}.")
    return params


def build_optimizers(model, lr=1e-3, weight_decay=0.0):
    visual_optimizer = torch.optim.AdamW(
        model.get_visual_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    text_optimizer = torch.optim.AdamW(
        model.get_text_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    return visual_optimizer, text_optimizer


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
    few_shot_list = torch.load(few_shot_path)
    # 返回 list of tensors，每个 tensor 是 [3, 240, 240]
    return [tensor.to(device) for tensor in few_shot_list]


@torch.no_grad()
def build_cached_prompt_features(model, few_shot_path, device):
    """预编码 few-shot prompt 特征，避免评估时重复跑 prompt visual tower。"""
    normal_list = build_cached_normal_img_features(model, few_shot_path, device)
    return model.build_prompt_feature_cache(normal_list=normal_list)


@torch.no_grad()
def build_cached_text_prototypes(model, category_name, device):
    """预编码类别文本原型，避免评估时每个 batch 重复跑 text tower。"""
    return model.build_text_prototype_cache(obj_types=[category_name], device=torch.device(device))


def split_query_prompt_inputs(inputs, device):
    query_image = inputs[0].to(device)
    prompt_images = torch.stack(inputs[1:], dim=1).to(device)
    return query_image, prompt_images


def compute_training_loss(
    outputs,
    labels,
    loss_fn,
    phase,
    image_loss_weight=IMAGE_LOSS_WEIGHT,
    pqa_loss_weight=PQA_LOSS_WEIGHT,
    text_reg_weight=TEXT_REG_WEIGHT,
):
    """Phase-specific hybrid InCTRL objective using logits for trainable branches."""
    labels = labels.float()
    zero = outputs["final_logit"].new_zeros(())
    final_loss = zero
    base_loss = zero
    image_loss = zero
    pqa_loss = zero
    text_loss = zero
    text_reg_loss = zero

    if phase == "visual":
        final_loss = loss_fn(outputs["final_logit"], labels)
        base_loss = loss_fn(outputs["base_logit"], labels)
        image_loss = loss_fn(outputs["image_logit"], labels) if image_loss_weight > 0 else zero
        pqa_loss = loss_fn(outputs["pqa_logit"], labels) if pqa_loss_weight > 0 else zero
        total_loss = final_loss + base_loss + image_loss_weight * image_loss + pqa_loss_weight * pqa_loss
    elif phase == "text":
        text_loss = loss_fn(outputs["text_logit"], labels)
        text_reg_loss = text_reg_weight * outputs.get("text_static_reg", zero)
        total_loss = text_loss + text_reg_loss
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    return total_loss, {
        "final_loss": final_loss.detach(),
        "base_loss": base_loss.detach(),
        "image_loss": image_loss.detach(),
        "pqa_loss": pqa_loss.detach(),
        "text_loss": text_loss.detach(),
        "text_reg_loss": text_reg_loss.detach(),
        "total_loss": total_loss.detach(),
    }


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
        "base": [],
        "text": [],
        "pqa": [],
        "image": [],
        "holistic": [],
        "max_patch": [],
    }

    for inputs, types, labels in tqdm(loader, desc="[TEST] Batch", leave=False):
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
            ("base", "base_score"),
            ("text", "text_score"),
            ("pqa", "pqa_score"),
            ("image", "image_score"),
            ("holistic", "holistic_score"),
            ("max_patch", "max_base_patch_score"),
        ]:
            if output_key in outputs:
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
    text_reg_weight=TEXT_REG_WEIGHT,
    resume_checkpoint=None,
    start_epoch=0,
):
    """运行一个训练域的 4-shot 模型，并用 cross-shots 评估。"""
    eval_shots = eval_shots or EVAL_SHOTS
    label = label or f"trained_on_{train_dataset}_shot_{train_shot}"
    print(f"========== 实验 [{label}] ==========")
    print(
        f"配置: train_dataset={train_dataset}, test_datasets={test_datasets}, "
        f"train_shot={train_shot}, eval_shots={eval_shots}, n_epochs={n_epochs}, "
        f"lr={lr}, batch_size={batch_size}, num_workers={num_workers}, "
        f"weight_decay={weight_decay}, image_loss_weight={image_loss_weight}, "
        f"pqa_loss_weight={pqa_loss_weight}, text_reg_weight={text_reg_weight}"
    )
    print(f"设备: {DEVICE}")

    # 构建模型
    model = build_model(DEVICE)

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
    cfg.DATA_LOADER.NUM_WORKERS = num_workers
    cfg.DATA_LOADER.PIN_MEMORY = DEVICE == "cuda"

    transform = get_transform()
    train_loader = ds_loader.construct_loader(cfg, "train", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    loss_fn = BinaryFocalLoss(logits=True).to(DEVICE)

    visual_optimizer, text_optimizer = build_optimizers(model, lr=lr, weight_decay=weight_decay)
    visual_scheduler = build_scheduler(visual_optimizer, n_epochs)
    text_scheduler = build_scheduler(text_optimizer, n_epochs)

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
            if "visual_optimizer_state_dict" in checkpoint:
                visual_optimizer.load_state_dict(checkpoint["visual_optimizer_state_dict"])
            if "text_optimizer_state_dict" in checkpoint:
                text_optimizer.load_state_dict(checkpoint["text_optimizer_state_dict"])
            if "visual_scheduler_state_dict" in checkpoint:
                visual_scheduler.load_state_dict(checkpoint["visual_scheduler_state_dict"])
            if "text_scheduler_state_dict" in checkpoint:
                text_scheduler.load_state_dict(checkpoint["text_scheduler_state_dict"])
            if start_epoch == 0 and "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"])
            if "loss" in checkpoint:
                resume_loss_history = [float(x) for x in checkpoint["loss"]]
        else:
            model.load_state_dict(checkpoint, strict=False)
        model = model.to(DEVICE)
        # 调整学习率调度器的当前状态
        if not (isinstance(checkpoint, dict) and "visual_scheduler_state_dict" in checkpoint):
            for _ in range(start_epoch):
                visual_scheduler.step()
                text_scheduler.step()
        print(f"[INFO] 从 epoch {start_epoch + 1} 继续训练")

    # 尝试加载已有的 loss 历史（恢复训练时需要合并）
    history_loss = []
    existing_results_path = results_json_path
    if existing_results_path.exists():
        try:
            with open(existing_results_path, encoding="utf-8") as f:
                existing_data = json.load(f)
            if "loss" in existing_data and len(existing_data["loss"]) > 0:
                history_loss = [float(x) for x in existing_data["loss"]]
                print(f"[INFO] 已加载历史 loss 数据: {len(history_loss)} 个 epoch")
        except Exception as e:
            print(f"[WARNING] 加载历史 loss 失败: {e}")
    if not history_loss and resume_loss_history:
        history_loss = resume_loss_history
        print(f"[INFO] 已从 checkpoint 加载历史 loss 数据: {len(history_loss)} 个 epoch")

    # 训练循环
    completed_epoch = start_epoch
    for epoch in range(start_epoch, n_epochs):
        phase = "visual" if epoch % 2 == 0 else "text"
        model.set_train_phase(phase)
        optimizer = visual_optimizer if phase == "visual" else text_optimizer
        scheduler = visual_scheduler if phase == "visual" else text_scheduler
        trainable_params = get_trainable_parameters(model, phase)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{n_epochs} | phase={phase} | lr={current_lr:.8f}")

        model.train()
        epoch_loss = 0.0
        actual_steps = 0

        batch_pbar = tqdm(
            total=steps_per_epoch,
            desc=f"[TRAIN] Epoch {epoch + 1}/{n_epochs}",
            unit="batch",
            leave=False,
        )

        for batch_idx, (inputs, types, labels) in enumerate(train_loader):
            if batch_idx >= steps_per_epoch:
                break

            labels = labels.to(DEVICE)
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
                phase=phase,
                image_loss_weight=image_loss_weight,
                pqa_loss_weight=pqa_loss_weight,
                text_reg_weight=text_reg_weight,
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
                "base": f"{loss_parts['base_loss'].item():.4f}",
                "image": f"{loss_parts['image_loss'].item():.4f}",
                "pqa": f"{loss_parts['pqa_loss'].item():.4f}",
                "text": f"{loss_parts['text_loss'].item():.4f}",
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
        "text_reg_weight": text_reg_weight,
        "model_architecture": {
            "fusion_mode": getattr(model, "fusion_mode", None),
            "use_text_adapter": getattr(model, "use_text_adapter", None),
            "use_visual_adapter": getattr(model, "use_visual_adapter", None),
            "use_prompt_query_adapter": getattr(model, "use_prompt_query_adapter", None),
            "use_pqa_in_final_map": getattr(model, "use_pqa_in_final_map", None),
            "use_branch_fusion": getattr(model, "use_branch_fusion", None),
        },
        "label": label,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "visual_optimizer_state_dict": visual_optimizer.state_dict(),
            "text_optimizer_state_dict": text_optimizer.state_dict(),
            "visual_scheduler_state_dict": visual_scheduler.state_dict(),
            "text_scheduler_state_dict": text_scheduler.state_dict(),
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
                prompt_feature_cache = build_cached_prompt_features(model, fs_pt, DEVICE)
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
    text_reg_weight=TEXT_REG_WEIGHT,
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
            text_reg_weight=text_reg_weight,
            resume_checkpoint=resume_checkpoint,
            start_epoch=start_epoch,
        )
        run_outputs.append(run_output)
        for shot, shot_results in run_output["results"].items():
            for test_dataset, metrics in shot_results.items():
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
                "text_reg_weight": text_reg_weight,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[INFO] 跨 shot 汇总已保存: {summary_path}")
    return {"runs": run_outputs, "summary_rows": summary_rows, "summary_path": summary_path}


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
                        help="图像级残差 LIRL 损失权重，默认 1.0")
    parser.add_argument("--pqa-loss-weight", type=float, default=PQA_LOSS_WEIGHT,
                        help="PQA 全局比较分支损失权重，默认 1.0")
    parser.add_argument("--text-reg-weight", type=float, default=TEXT_REG_WEIGHT,
                        help="TA 静态文本原型残差正则权重，默认 0.01")
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
    print(f"TEXT_REG_WEIGHT = {args.text_reg_weight}")
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
        text_reg_weight=args.text_reg_weight,
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
