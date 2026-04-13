#!/usr/bin/env python3
"""
使用 shot=8 训练 InCTRL 模型。
仅在每个 shot 的全部 epoch 完成后保存最终 checkpoint。
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
TRAINED_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec_VA"

# 预训练模型路径
CKPT_NAME = "vit_b_16_plus_240-laion400m_e32-699c4b84.pt"
LOCAL_CKPT = PROJECT_ROOT / CKPT_NAME

# 训练配置（保持与 train_local.py 一致）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 48
STEPS_PER_EPOCH = 100
LR = 1e-3
N_EPOCHS = 10
WEIGHT_DECAY = 0.0

# 仅训练 8-shot（4-shot 已完成）
TARGET_SHOTS = [8]

# DataLoader 优化：Windows 下使用更保守配置，避免 shared file mapping (1455) 错误。
if os.name == "nt":
    DATA_LOADER_WORKERS = 1
else:
    DATA_LOADER_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))

# 等算力模式：当 shot=8 时，按 batch*shot 近似恒定将 batch_size 减半。
ENABLE_EQUAL_COMPUTE_FOR_SHOT8 = True

# 数据集
TRAIN_DATASET_NAME = "mvtec"
TEST_DATASETS = ["visa"]

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

def remap_image_path(raw_path, dataset_name):
    """将旧路径重映射到本地路径"""
    normalized = str(raw_path).replace("\\", "/")
    idx = normalized.lower().find(f"/{dataset_name.lower()}/")
    if idx != -1:
        return DATA_ROOT / Path(normalized[idx + 1:])
    return Path(raw_path)


def normalize_dataset_json_paths():
    """修正 JSON 中的图像路径以适配本地环境。"""
    for ds in [TRAIN_DATASET_NAME] + TEST_DATASETS:
        json_dir = DATA_ROOT / "AD_json" / ds
        if not json_dir.exists():
            continue
        for json_file in json_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                modified = False
                for item in data:
                    old_path = item["image_path"]
                    new_path_str = str(remap_image_path(old_path, ds))
                    if new_path_str != old_path:
                        item["image_path"] = new_path_str
                        modified = True
                if modified:
                    json_file.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as e:
                print(f"[WARNING] 路径修正失败 {json_file}: {e}")


def prepare_dataset_paths():
    """准备训练和测试数据集的 JSON 路径"""
    global MVTEC_TRAIN_NORMAL, MVTEC_TRAIN_OUTLIER, DATASET_CATEGORIES, TYPE_TO_IDX

    # 修正 JSON 中的路径
    print("[INFO] 正在修正数据集路径...")
    normalize_dataset_json_paths()

    MVTEC_TRAIN_NORMAL = []
    MVTEC_TRAIN_OUTLIER = []
    json_mvtec = DATA_ROOT / "AD_json" / "mvtec"

    for f in sorted(json_mvtec.glob("*_normal.json")):
        if "val_" not in f.name:
            MVTEC_TRAIN_NORMAL.append(str(f))

    for f in sorted(json_mvtec.glob("*_outlier.json")):
        if "val_" not in f.name:
            MVTEC_TRAIN_OUTLIER.append(str(f))

    DATASET_CATEGORIES = {}
    for ds in TEST_DATASETS:
        cats = set()
        json_ds = DATA_ROOT / "AD_json" / ds
        for f in json_ds.glob("*_val_normal.json"):
            cats.add(f.name.replace("_val_normal.json", ""))
        DATASET_CATEGORIES[ds] = sorted(cats)

    # 构建类型名称到索引的映射（用于模型）
    all_types = set()
    for f in sorted(json_mvtec.glob("*_normal.json")):
        if "val_" not in f.name:
            data = json.load(open(f))
            for item in data:
                all_types.add(item["type"])
    TYPE_TO_IDX = {t: i for i, t in enumerate(sorted(all_types))}
    print(f"[INFO] TYPE_TO_IDX mapping: {TYPE_TO_IDX}")

    print(f"[INFO] MVTec 训练 JSON: 正常={len(MVTEC_TRAIN_NORMAL)}, 异常={len(MVTEC_TRAIN_OUTLIER)}")
    for ds, cats in DATASET_CATEGORIES.items():
        print(f"[INFO] {ds.upper()} 测试类别 ({len(cats)}): {cats}")


# ============================================================================
# 模型
# ============================================================================

def build_model(device):
    """构建 InCTRL 模型"""
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
    """获取所有可训练参数"""
    params = []
    params += list(model.adapter.parameters())
    params += list(model.diff_head.parameters())
    params += list(model.diff_head_ref.parameters())
    if hasattr(model, "visual_adapter") and model.visual_adapter is not None:
        params += list(model.visual_adapter.parameters())
    params = [param for param in params if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return params


def build_optimizer(model, lr=1e-3, weight_decay=0.0):
    trainable_params = get_trainable_parameters(model)
    return torch.optim.AdamW(
        trainable_params,
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
# 训练主函数（仅训练，不测试）
# ============================================================================

def train_model(
    shot,
    n_epochs=10,
    lr=1e-3,
    steps_per_epoch=100,
    batch_size=48,
    weight_decay=0.0,
):
    """训练模型

    Args:
        shot: 训练时使用的 shot 数
        其他参数与 train_local.py 保持一致
    """
    print(f"========== 训练实验 (shot={shot}) ==========")
    print(f"配置: n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, shot={shot}")
    print(f"设备: {DEVICE}")

    # 构建模型
    model = build_model(DEVICE)

    # 配置
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TEST.BATCH_SIZE = 1
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.MAX_EPOCH = n_epochs
    cfg.shot = shot  # 使用指定的 shot
    cfg.steps_per_epoch = steps_per_epoch
    cfg.normal_json_path = MVTEC_TRAIN_NORMAL
    cfg.outlier_json_path = MVTEC_TRAIN_OUTLIER
    effective_workers = DATA_LOADER_WORKERS
    if os.name == "nt" and shot >= 8:
        # 高 shot + 大 batch 时，限制 worker 数降低共享内存映射压力。
        effective_workers = min(effective_workers, 1)

    cfg.DATA_LOADER.NUM_WORKERS = effective_workers
    cfg.DATA_LOADER.PIN_MEMORY = DEVICE == "cuda"
    print(
        f"[INFO] DataLoader 配置: num_workers={cfg.DATA_LOADER.NUM_WORKERS}, "
        f"pin_memory={cfg.DATA_LOADER.PIN_MEMORY}"
    )

    transform = get_transform()
    train_loader = ds_loader.construct_loader(cfg, "train", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    loss_fn = BinaryFocalLoss(logits=False).to(DEVICE)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, n_epochs)

    trainable_params = get_trainable_parameters(model)
    print(f"[INFO] 可训练参数数量: {len(trainable_params)}")

    # 训练循环
    for epoch in range(n_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{n_epochs} | lr={current_lr:.8f}")

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
            preds, preds2 = model(tokenizer, inputs, types, None)
            loss = loss_fn(preds, labels.float()) + loss_fn(preds2, labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            actual_steps += 1
            batch_pbar.update(1)
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        batch_pbar.close()
        scheduler.step()

        avg_loss = float(epoch_loss / max(actual_steps, 1))
        print(f"Epoch {epoch + 1} 完成 | avg_loss={avg_loss:.4f}")

    print(f"\n训练完成! (shot={shot})")

    # 仅在全部 epoch 结束后保存一次最终 checkpoint
    final_ckpt_dir = TRAINED_CHECKPOINT_DIR / str(shot)
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = final_ckpt_dir / "checkpoint"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"[INFO] 最终检查点已保存: {final_ckpt_path}")

    return model


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    seed_everything(SEED)

    # 确认 GPU 可用
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA 不可用，将使用 CPU")

    # 准备数据集路径
    prepare_dataset_paths()

    # 创建主检查点目录
    TRAINED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 仅运行 8-shot 训练
    for shot in TARGET_SHOTS:
        effective_batch_size = BATCH_SIZE
        if ENABLE_EQUAL_COMPUTE_FOR_SHOT8 and shot == 8:
            effective_batch_size = max(2, BATCH_SIZE // 2)
            if effective_batch_size % 2 != 0:
                effective_batch_size -= 1
            print(
                f"[INFO] 等算力模式已启用: shot={shot}, batch_size={BATCH_SIZE} -> {effective_batch_size}"
            )

        train_model(
            shot=shot,
            n_epochs=N_EPOCHS,
            lr=LR,
            steps_per_epoch=STEPS_PER_EPOCH,
            batch_size=effective_batch_size,
            weight_decay=WEIGHT_DECAY,
        )
        print("\n" + "=" * 72)

    print("\n所有训练完成!")
    print(f"检查点保存在: {TRAINED_CHECKPOINT_DIR}")
    print("  - checkpoints/InCTRL_trained_on_MVTec_VA/8/checkpoint")
