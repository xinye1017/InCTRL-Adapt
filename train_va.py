#!/usr/bin/env python3
"""
使用 VA（Visual Adapter）训练 InCTRL 的 2/4/8-shot 消融模型。
默认运行 global_only / local_only / global_local 三组实验。
仅在每个 shot 的全部 epoch 完成后保存最终 checkpoint。
"""

import argparse
import json
import logging
import os
import random
import subprocess
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

# 设置文件日志
LOG_FILE = PROJECT_ROOT / "training_va_ablation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

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
TRAINED_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec_VA_ablation"

# 预训练模型路径
CKPT_NAME = "vit_b_16_plus_240-laion400m_e32-699c4b84.pt"
LOCAL_CKPT = PROJECT_ROOT / CKPT_NAME

# 训练配置（针对 24GB VRAM RTX 3090 云端实例）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 96
STEPS_PER_EPOCH = 100
LR = 1e-3
N_EPOCHS = 10
WEIGHT_DECAY = 0.0

# 默认完整消融矩阵：3 个 VA 路由 x 2/4/8-shot
TARGET_SHOTS = [2, 4, 8]
VA_ABLATION_MODES = ["global_only", "local_only", "global_local"]

# 从 CLIP backbone 初始化时，允许这些新训练模块缺失；其余缺失视为 checkpoint 不匹配。
TRAINABLE_PREFIXES = ("adapter.", "diff_head.", "diff_head_ref.", "visual_adapter.")
ALLOWED_BASE_UNEXPECTED_KEYS = {"logit_scale"}

# 严格等算力模式（按 batch_size * (1 + shot) 近似恒定）
SHOT_BATCH_SIZE = {
    2: 96,  # 96 * (1 + 2) = 288
    4: 56,  # 56 * (1 + 4) = 280
    8: 32,  # 32 * (1 + 8) = 288
}

# DataLoader 优化：Windows 下使用更保守配置，避免 shared file mapping (1455) 错误。
if os.name == "nt":
    DATA_LOADER_WORKERS = 1
else:
    DATA_LOADER_WORKERS = max(4, min(12, (os.cpu_count() or 8) // 2))

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
                logger.warning(f"路径修正失败 {json_file}: {e}")



def prepare_dataset_paths():
    """准备训练和测试数据集的 JSON 路径"""
    global MVTEC_TRAIN_NORMAL, MVTEC_TRAIN_OUTLIER, DATASET_CATEGORIES, TYPE_TO_IDX

    # 修正 JSON 中的路径
    logger.info("正在修正数据集路径...")
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
    logger.info(f"TYPE_TO_IDX mapping: {TYPE_TO_IDX}")

    logger.info(f"MVTec 训练 JSON: 正常={len(MVTEC_TRAIN_NORMAL)}, 异常={len(MVTEC_TRAIN_OUTLIER)}")
    for ds, cats in DATASET_CATEGORIES.items():
        logger.info(f"{ds.upper()} 测试类别 ({len(cats)}): {cats}")


# ============================================================================
# 模型
# ============================================================================


def extract_state_dict(checkpoint):
    """Extract and normalize a model state dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def is_allowed_missing_from_base(key):
    return key.startswith(TRAINABLE_PREFIXES)


def validate_base_checkpoint_load(load_info):
    """Fail fast when the base checkpoint does not cover frozen InCTRL/CLIP weights."""
    bad_missing = [
        key for key in load_info.missing_keys
        if not is_allowed_missing_from_base(key)
    ]
    bad_unexpected = [
        key for key in load_info.unexpected_keys
        if key not in ALLOWED_BASE_UNEXPECTED_KEYS
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "预训练 checkpoint 与当前 VA 训练结构不匹配；已停止以避免随机初始化进入实验。"
            f" missing={bad_missing}, unexpected={bad_unexpected}"
        )


def validate_visual_adapter_setup(model, va_mode):
    if getattr(model, "visual_adapter", None) is None:
        raise RuntimeError("VA 训练要求实例化 visual_adapter，但当前模型未启用。")
    if model.visual_adapter.mode != va_mode:
        raise RuntimeError(
            f"visual_adapter mode 不一致: model={model.visual_adapter.mode}, expected={va_mode}"
        )
    if not model.visual_adapter.zero_init:
        raise RuntimeError("VA 训练默认要求 ZERO_INIT=True，以保证 identity initialization。")


def load_base_checkpoint_for_va_training(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"预训练权重不存在: {checkpoint_path}")

    logger.info(f"加载并强校验预训练权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    load_info = model.load_state_dict(state_dict, strict=False)
    validate_base_checkpoint_load(load_info)

    allowed_missing = [
        key for key in load_info.missing_keys
        if is_allowed_missing_from_base(key)
    ]
    if allowed_missing:
        logger.info(
            "预训练 checkpoint 未包含新训练模块，将按当前初始化训练: "
            f"{allowed_missing}"
        )
    if load_info.unexpected_keys:
        logger.info(f"预训练 checkpoint 中已忽略兼容键: {load_info.unexpected_keys}")


def build_model(device, va_mode):
    """构建 InCTRL 模型（强制启用 VA）"""
    model_config_path = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"
    with open(model_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    cfg = get_cfg()
    cfg.VISUAL_ADAPTER.ENABLE = True
    cfg.VISUAL_ADAPTER.ZERO_INIT = True
    cfg.VISUAL_ADAPTER.MODE = va_mode
    from open_clip import model as _model_mod

    model = _model_mod.InCTRL(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    validate_visual_adapter_setup(model, va_mode)
    load_base_checkpoint_for_va_training(model, LOCAL_CKPT)

    return model.to(device)


# ============================================================================
# 训练工具
# ============================================================================


def get_trainable_named_parameters(model):
    """Return only parameters used by the VA ablation forward path."""
    allowed_prefixes = ("visual_adapter.", "diff_head.", "diff_head_ref.")
    params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(allowed_prefixes)
    ]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return params


def get_trainable_parameters(model):
    return [param for _, param in get_trainable_named_parameters(model)]



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
    va_mode,
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
    logger.info(f"========== 训练实验 (VA mode={va_mode}, shot={shot}) ==========")
    logger.info(
        f"配置: n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, "
        f"weight_decay={weight_decay}, shot={shot}, va_mode={va_mode}"
    )
    logger.info(f"设备: {DEVICE}")

    # 构建模型
    model = build_model(DEVICE, va_mode)

    # 配置
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TEST.BATCH_SIZE = 1
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.MAX_EPOCH = n_epochs
    cfg.shot = shot
    cfg.steps_per_epoch = steps_per_epoch
    cfg.normal_json_path = MVTEC_TRAIN_NORMAL
    cfg.outlier_json_path = MVTEC_TRAIN_OUTLIER
    effective_workers = DATA_LOADER_WORKERS
    if os.name == "nt" and shot >= 8:
        # 高 shot + 大 batch 时，限制 worker 数降低共享内存映射压力。
        effective_workers = min(effective_workers, 1)

    cfg.DATA_LOADER.NUM_WORKERS = effective_workers
    cfg.DATA_LOADER.PIN_MEMORY = DEVICE == "cuda"
    logger.info(
        f"DataLoader 配置: num_workers={cfg.DATA_LOADER.NUM_WORKERS}, "
        f"pin_memory={cfg.DATA_LOADER.PIN_MEMORY}"
    )

    transform = get_transform()
    train_loader = ds_loader.construct_loader(cfg, "train", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    loss_fn = BinaryFocalLoss(logits=False).to(DEVICE)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, n_epochs)

    trainable_named_params = get_trainable_named_parameters(model)
    trainable_params = [param for _, param in trainable_named_params]
    trainable_numel = sum(param.numel() for param in trainable_params)
    logger.info(f"可训练张量数量: {len(trainable_params)}")
    logger.info(f"可训练参数量: {trainable_numel:,}")
    logger.info(f"可训练参数前缀: {sorted({name.split('.')[0] for name, _ in trainable_named_params})}")

    # 训练循环
    for epoch in range(n_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{n_epochs} | lr={current_lr:.8f}")

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
        logger.info(f"Epoch {epoch + 1} 完成 | avg_loss={avg_loss:.4f}")

    logger.info(f"\n训练完成! (VA mode={va_mode}, shot={shot})")

    # 仅在全部 epoch 结束后保存一次最终 checkpoint
    final_ckpt_dir = TRAINED_CHECKPOINT_DIR / va_mode / str(shot)
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = final_ckpt_dir / "checkpoint"
    torch.save(model.state_dict(), final_ckpt_path)
    logger.info(f"最终检查点已保存: {final_ckpt_path}")

    metadata = {
        "train_dataset": TRAIN_DATASET_NAME,
        "shot": shot,
        "visual_adapter_mode": va_mode,
        "visual_adapter_zero_init": True,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "checkpoint": str(final_ckpt_path),
    }
    metadata_path = final_ckpt_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"实验元数据已保存: {metadata_path}")

    return model


# ============================================================================
# 主入口
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train InCTRL Visual Adapter ablations on MVTec normal prompts."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VA_ABLATION_MODES,
        default=VA_ABLATION_MODES,
        help="VA ablation modes to train.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        choices=TARGET_SHOTS,
        default=TARGET_SHOTS,
        help="Few-shot normal prompt counts to train.",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--run-post-eval",
        action="store_true",
        help="Run post-training evaluation after all requested runs finish.",
    )
    parser.add_argument(
        "--post-eval-selection-priority",
        choices=["highest_auroc", "lowest_fpr", "balanced"],
        default="balanced",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 72)
    logger.info("VA global/local 消融训练开始")
    logger.info("=" * 72)
    logger.info(f"训练模式: {args.modes}")
    logger.info(f"训练 shots: {args.shots}")

    seed_everything(SEED)

    # 确认 GPU 可用
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA 不可用，将使用 CPU")

    # 准备数据集路径
    prepare_dataset_paths()

    # 创建主检查点目录
    TRAINED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 依次运行 3 组 VA 消融 x 2/4/8-shot 训练。
    for va_mode in args.modes:
        for shot in args.shots:
            if shot not in SHOT_BATCH_SIZE:
                raise ValueError(f"SHOT_BATCH_SIZE 缺少 shot={shot} 的配置")

            effective_batch_size = SHOT_BATCH_SIZE[shot]
            if effective_batch_size % 2 != 0:
                # 与现有脚本保持一致，确保 batch 为偶数
                effective_batch_size -= 1

            logger.info(
                "严格等算力模式: "
                f"mode={va_mode}, shot={shot}, batch_size={BATCH_SIZE} -> {effective_batch_size}"
            )

            train_model(
                shot=shot,
                va_mode=va_mode,
                n_epochs=args.epochs,
                lr=args.lr,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=effective_batch_size,
                weight_decay=args.weight_decay,
            )
            logger.info("=" * 72)

    logger.info("\n所有训练完成!")
    logger.info(f"日志保存在: {LOG_FILE}")
    logger.info(f"检查点保存在: {TRAINED_CHECKPOINT_DIR}")
    for va_mode in args.modes:
        for shot in args.shots:
            logger.info(f"  - {TRAINED_CHECKPOINT_DIR / va_mode / str(shot) / 'checkpoint'}")

    if args.run_post_eval:
        post_eval_script = PROJECT_ROOT / "tools" / "post_train_evaluation.py"
        command = [
            sys.executable,
            str(post_eval_script),
            "--checkpoint-roots",
            str(TRAINED_CHECKPOINT_DIR),
            "--selection-priority",
            args.post_eval_selection_priority,
        ]
        logger.info("启动训练后自动评估: %s", " ".join(command))
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
