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
SHOT_LIST = [2, 4, 8]
SHOT = 2  # 主要测试 shot

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
                all_types.add(item['type'])
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
    for path in FEW_SHOT_ROOT.iterdir():
        if path.name.lower() == ds.lower():
            # 在该目录下查找匹配 ds 名称的子目录
            for subpath in path.iterdir():
                if subpath.is_dir() and subpath.name.lower() == ds.lower():
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
    raise FileNotFoundError(f"Cannot find few-shot pt for {ds} - {cat} (shot={shot})")


def build_cached_normal_img_features(model, few_shot_path, device):
    """构建用于测试的 normal_list (返回 list of tensors)"""
    few_shot_list = torch.load(few_shot_path)
    # 返回 list of tensors，每个 tensor 是 [3, 240, 240]
    return [tensor.to(device) for tensor in few_shot_list]


def split_query_prompt_inputs(inputs, device):
    query_image = inputs[0].to(device)
    prompt_images = torch.stack(inputs[1:], dim=1).to(device)
    return query_image, prompt_images


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, cached_normal_list=None):
    """评估模型"""
    model.eval()
    preds_all, labels_all = [], []

    for inputs, types, labels in tqdm(loader, desc="[TEST] Batch", leave=False):
        labels = labels.to(device)
        query_image, _ = split_query_prompt_inputs(inputs, device)
        outputs = model(
            query_image=query_image,
            normal_list=cached_normal_list,
            obj_types=types,
            return_aux=False,
            return_dict=True,
        )
        preds_all.extend(outputs["final_score"].detach().cpu().float().numpy())
        labels_all.extend(labels.cpu().numpy())

    auroc = roc_auc_score(labels_all, preds_all)
    aupr = average_precision_score(labels_all, preds_all)
    return float(auroc), float(aupr)


# ============================================================================
# 训练主函数
# ============================================================================

def run_experiment(
    label="experiment",
    n_epochs=10,
    lr=1e-3,
    steps_per_epoch=100,
    batch_size=48,
    weight_decay=0.0,
    resume_checkpoint=None,
    start_epoch=0,
):
    """运行训练实验

    Args:
        resume_checkpoint: 检查点路径，用于恢复训练
        start_epoch: 从第几个 epoch 开始训练（0-indexed，用于恢复训练）
    """
    print(f"========== 实验 [{label}] ==========")
    print(f"配置: n_epochs={n_epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
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
    cfg.shot = SHOT
    cfg.steps_per_epoch = steps_per_epoch
    cfg.normal_json_path = MVTEC_TRAIN_NORMAL
    cfg.outlier_json_path = MVTEC_TRAIN_OUTLIER
    cfg.DATA_LOADER.NUM_WORKERS = 0  # Windows 下设为 0 避免 multiprocessing pickle 问题
    cfg.DATA_LOADER.PIN_MEMORY = True

    transform = get_transform()
    train_loader = ds_loader.construct_loader(cfg, "train", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    loss_fn = BinaryFocalLoss(logits=False).to(DEVICE)

    visual_optimizer, text_optimizer = build_optimizers(model, lr=lr, weight_decay=weight_decay)
    visual_scheduler = build_scheduler(visual_optimizer, n_epochs)
    text_scheduler = build_scheduler(text_optimizer, n_epochs)

    # 创建检查点目录
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 恢复训练
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[INFO] 从检查点恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(DEVICE)
        # 调整学习率调度器的当前状态
        for _ in range(start_epoch):
            scheduler.step()
        print(f"[INFO] 从 epoch {start_epoch + 1} 继续训练")

    # 尝试加载已有的 loss 历史（恢复训练时需要合并）
    history_loss = []
    existing_results_path = RESULTS_DIR / f"{label}_results.json"
    if existing_results_path.exists():
        try:
            existing_data = json.load(open(existing_results_path))
            if "loss" in existing_data and len(existing_data["loss"]) > 0:
                history_loss = [float(x) for x in existing_data["loss"]]
                print(f"[INFO] 已加载历史 loss 数据: {len(history_loss)} 个 epoch")
        except Exception as e:
            print(f"[WARNING] 加载历史 loss 失败: {e}")

    # 训练循环
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
            loss = loss_fn(outputs["final_score"], labels.float())

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
        history_loss.append(avg_loss)
        print(f"Epoch {epoch + 1} 完成 | avg_loss={avg_loss:.4f}")

        # 保存检查点
        ckpt_path = CHECKPOINT_DIR / f"{label}_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] 检查点已保存: {ckpt_path}")

    print("\n训练完成! 开始测试集评估...")

    # 测试评估
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
    summary_data = {
        "label": label,
        "config": {
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "weight_decay": weight_decay,
        },
        "loss": [float(x) for x in history_loss],
        "results": results,
    }

    results_json_path = RESULTS_DIR / f"{label}_results.json"
    with open(results_json_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 结果已保存: {results_json_path}")

    return history_loss, results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InCTRL 本地训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点路径，用于恢复训练")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="起始 epoch 编号（0-indexed，用于恢复训练）")
    args = parser.parse_args()

    seed_everything(SEED)

    # 确认 GPU 可用
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA 不可用，将使用 CPU")

    # 准备数据集路径
    prepare_dataset_paths()

    # 实验配置
    EXPERIMENT_NAME = f"inctrl_va_bs{BATCH_SIZE}_lr{LR}_ep{N_EPOCHS}"

    SHARED_KWARGS = {
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "resume_checkpoint": args.resume,
        "start_epoch": args.start_epoch,
    }

    print(f"\n实验配置: {EXPERIMENT_NAME}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"STEPS_PER_EPOCH = {STEPS_PER_EPOCH}")
    print(f"LR = {LR}")
    print(f"N_EPOCHS = {N_EPOCHS}")
    if args.resume:
        print(f"恢复检查点: {args.resume}")
        print(f"起始 epoch: {args.start_epoch + 1}")

    # 运行实验
    loss_main, results_main = run_experiment(EXPERIMENT_NAME, **SHARED_KWARGS)

    print("\n" + "=" * 72)
    print("训练和评估完成!")
    print("=" * 72)

    # 打印最终结果
    print("\n最终结果汇总:")
    for shot in SHOT_LIST:
        for ds in TEST_DATASETS:
            auroc = results_main[shot][ds]["auroc"]
            aupr = results_main[shot][ds]["aupr"]
            print(f"  {ds.upper()} | {shot}-shot -> AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
