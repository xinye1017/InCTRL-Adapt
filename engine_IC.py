# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
import csv
import time
import copy
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
import torch.distributed as dist
import matplotlib.pyplot as plt
from collections import defaultdict
from open_clip.model import get_cast_dtype
from open_clip.inctrl_adapt import InCTRLAdapt
from open_clip.inctrl_pqa_losses import compute_inctrl_pqa_loss
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.get_logger(__name__)

TRAIN_HISTORY_FIELDS = [
    "epoch",
    "phase",
    "train_loss",
    "final_loss",
    "image_loss",
    "pqa_loss",
    "text_loss",
    "mask_loss",
    "text_mask_loss",
    "visual_loss",
    "visual_mask_loss",
    "val_auroc",
    "val_aupr",
    "best_val_auroc",
    "delta_vs_baseline",
    "elapsed_sec",
    "lr",
    "did_eval",
]


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _convert_to_rgb(image):
    return image.convert('RGB')


def _split_query_prompt_batch(inputs, device=None):
    query_image = inputs[0]
    prompt_images = torch.stack(inputs[1:], dim=1)
    if device is not None:
        query_image = query_image.to(device)
        prompt_images = prompt_images.to(device)
    return query_image, prompt_images


def _split_batch_with_optional_masks(batch, device=None):
    if len(batch) == 4:
        inputs, types, labels, masks = batch
    else:
        inputs, types, labels = batch
        masks = None
    query_image, prompt_images = _split_query_prompt_batch(inputs, device)
    if device is not None:
        labels = labels.to(device)
        if masks is not None:
            masks = masks.to(device)
    return query_image, prompt_images, types, labels, masks


def _get_base_model(model):
    return model.module if hasattr(model, "module") else model


def _build_active_model(cfg, model_cfg, cast_dtype, quick_gelu):
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    active_model = getattr(cfg.MODEL, "ACTIVE_MODEL", "InCTRL")
    if active_model in {"InCTRLAdapt", "InCTRLPQA"}:
        return InCTRLAdapt(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    return open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)


def _is_adapt_model(cfg):
    return getattr(cfg.MODEL, "ACTIVE_MODEL", "InCTRL") in {"InCTRLAdapt", "InCTRLPQA"}


def _resolve_max_epochs(cfg):
    max_epoch = int(getattr(cfg.SOLVER, "MAX_EPOCH", 400))
    return 10 if max_epoch == 400 else max_epoch


def _should_eval_epoch(cur_epoch, max_epoch, cfg):
    eval_period = max(1, int(getattr(cfg.TRAIN, "EVAL_PERIOD", 1)))
    return cur_epoch == 0 or (cur_epoch + 1) % eval_period == 0 or cur_epoch == max_epoch - 1


def _trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def _has_parameter_list(model, method_name):
    base_model = _get_base_model(model)
    if not hasattr(base_model, method_name):
        return False
    return len(getattr(base_model, method_name)()) > 0


def _visual_adapter_has_training_signal(cfg):
    visual_adapter_cfg = getattr(cfg, "VISUAL_ADAPTER", None)
    fusion_cfg = getattr(cfg, "FUSION", None)
    loss_cfg = getattr(cfg, "LOSS", None)
    if visual_adapter_cfg is not None and not bool(getattr(visual_adapter_cfg, "ENABLE", True)):
        return False
    if fusion_cfg is not None and not bool(getattr(fusion_cfg, "USE_VISUAL_BRANCH", True)):
        return False
    return any([
        _cfg_float(fusion_cfg, "VISUAL_WEIGHT", 0.0) > 0.0,
        _cfg_float(loss_cfg, "VISUAL_WEIGHT", 0.0) > 0.0,
        _cfg_float(loss_cfg, "VISUAL_MASK_WEIGHT", 0.0) > 0.0,
    ])


def _should_use_alternating_training(model, cfg):
    return (
        _has_parameter_list(model, "get_visual_parameters")
        and _has_parameter_list(model, "get_text_parameters")
        and _visual_adapter_has_training_signal(cfg)
    )


def _resolve_train_phase(cur_epoch, use_alternating, has_visual, has_text):
    if use_alternating:
        return "visual" if cur_epoch % 2 == 0 else "text"
    if has_visual and has_text:
        return "joint"
    return "visual" if has_visual else "text"


def _build_alternating_optimizers(model, lr=1e-3, use_alternating=True):
    base_model = _get_base_model(model)
    if not hasattr(base_model, "get_visual_parameters") or not hasattr(base_model, "get_text_parameters"):
        optimizer = torch.optim.AdamW(
            _trainable_parameters(base_model),
            lr=lr,
            betas=(0.9, 0.999),
        )
        return optimizer, optimizer

    if not use_alternating:
        if hasattr(base_model, "set_train_phase"):
            base_model.set_train_phase("joint")
        optimizer = torch.optim.AdamW(
            _trainable_parameters(base_model),
            lr=lr,
            betas=(0.9, 0.999),
        )
        return optimizer, optimizer

    visual_params = base_model.get_visual_parameters()
    text_params = base_model.get_text_parameters()

    if visual_params:
        visual_optimizer = torch.optim.AdamW(visual_params, lr=lr, betas=(0.9, 0.999))
    else:
        visual_optimizer = None

    if text_params:
        text_optimizer = torch.optim.AdamW(text_params, lr=lr, betas=(0.9, 0.999))
    else:
        text_optimizer = None

    # If one side is empty, both phases use the available optimizer
    if visual_optimizer is None and text_optimizer is None:
        optimizer = torch.optim.AdamW(
            _trainable_parameters(base_model), lr=lr, betas=(0.9, 0.999),
        )
        return optimizer, optimizer
    if visual_optimizer is None:
        return text_optimizer, text_optimizer
    if text_optimizer is None:
        return visual_optimizer, visual_optimizer

    return visual_optimizer, text_optimizer


def _select_adapt_score(outputs, cfg):
    score_key = str(getattr(cfg.FUSION, "SCORE_OUTPUT", "auto"))
    if score_key == "auto":
        score_key = (
            "coupled_score"
            if bool(getattr(cfg.FUSION, "IMAGE_PIXEL_COUPLING", True))
            else "final_score"
        )
    return outputs.get(score_key, outputs["final_score"])


def _optimizer_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _cfg_float(cfg, name, default=None):
    value = getattr(cfg, name, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _format_metric(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _build_epoch_record(
    epoch,
    phase,
    train_loss,
    loss_parts,
    val_auroc,
    val_aupr,
    best_val_auroc,
    baseline_auroc,
    elapsed_sec,
    lr,
    did_eval,
):
    if did_eval and val_auroc is not None:
        best_val_auroc = max(best_val_auroc or float("-inf"), float(val_auroc))
    best_value = None if best_val_auroc in (None, float("-inf")) else float(best_val_auroc)
    delta = None
    if did_eval and val_auroc is not None and baseline_auroc is not None and baseline_auroc >= 0:
        delta = float(val_auroc) - float(baseline_auroc)

    record = {
        "epoch": int(epoch),
        "phase": phase,
        "train_loss": _round_or_none(train_loss, 6),
        "final_loss": _round_or_none(loss_parts.get("final")),
        "image_loss": _round_or_none(loss_parts.get("image")),
        "pqa_loss": _round_or_none(loss_parts.get("pqa")),
        "text_loss": _round_or_none(loss_parts.get("text")),
        "mask_loss": _round_or_none(loss_parts.get("mask")),
        "text_mask_loss": _round_or_none(loss_parts.get("text_mask")),
        "visual_loss": _round_or_none(loss_parts.get("visual")),
        "visual_mask_loss": _round_or_none(loss_parts.get("visual_mask")),
        "val_auroc": _round_or_none(val_auroc),
        "val_aupr": _round_or_none(val_aupr),
        "best_val_auroc": _round_or_none(best_value),
        "delta_vs_baseline": _round_or_none(delta),
        "elapsed_sec": _round_or_none(elapsed_sec, 3),
        "lr": _round_or_none(lr, 8),
        "did_eval": bool(did_eval),
    }
    return record, best_value


def _format_epoch_summary(record):
    eval_state = "eval" if record.get("did_eval") else "skip"
    return (
        f"epoch {int(record['epoch']):03d} | "
        f"phase={record['phase']:<6} | "
        f"loss={_format_metric(record.get('train_loss'))} | "
        f"auroc={_format_metric(record.get('val_auroc'))} | "
        f"aupr={_format_metric(record.get('val_aupr'))} | "
        f"best={_format_metric(record.get('best_val_auroc'))} | "
        f"delta={_format_metric(record.get('delta_vs_baseline'))} | "
        f"{eval_state} | "
        f"{_format_metric(record.get('elapsed_sec'), digits=1)}s"
    )


def _latest_metrics_payload(output_dir, history_rows, checkpoint_path):
    latest = history_rows[-1] if history_rows else {}
    return {
        "output_dir": output_dir,
        "latest_epoch": latest.get("epoch"),
        "latest": latest,
        "best_val_auroc": latest.get("best_val_auroc"),
        "history_csv": os.path.join(output_dir, "train_history.csv"),
        "latest_metrics_json": os.path.join(output_dir, "latest_metrics.json"),
        "checkpoint_path": checkpoint_path,
    }


def _training_header_lines(cfg, max_epoch, train_loader, test_loader, use_alternating, baseline_auroc):
    train_ds = getattr(cfg, "train_dataset_name", "custom")
    eval_ds = getattr(cfg, "eval_dataset_name", "custom")
    is_cross_domain = train_ds != eval_ds
    eval_label = f"cross-domain ({train_ds}->{eval_ds})" if is_cross_domain else f"in-domain ({train_ds})"
    baseline_label = _format_metric(baseline_auroc) if baseline_auroc is not None and baseline_auroc >= 0 else "n/a (in-domain)"
    return [
        "",
        "=" * 78,
        "InCTRL local training",
        "=" * 78,
        f"model={cfg.MODEL.ACTIVE_MODEL} | shot={getattr(cfg, 'shot', '-')} | image_size={getattr(cfg, 'image_size', '-')}",
        f"epochs={max_epoch} | eval_period={getattr(cfg.TRAIN, 'EVAL_PERIOD', 1)} | steps_per_epoch={getattr(cfg, 'steps_per_epoch', '-')}",
        f"train_batches={len(train_loader)} | eval_batches={len(test_loader)} | alternating={use_alternating}",
        f"adapters: VA={cfg.VISUAL_ADAPTER.ENABLE} TA={cfg.TEXT_BRANCH.ENABLE} PQA={cfg.PQA.ENABLE}",
        f"score={cfg.FUSION.SCORE_OUTPUT} | pixel_fusion={cfg.FUSION.PIXEL_FUSION} | align_fusion={cfg.FUSION.ALIGN_FUSION}",
        f"lr=1e-3→{float(getattr(cfg.SOLVER, 'COSINE_MIN_LR', 1e-5)):.0e} (cosine) | early_stop_patience={int(getattr(cfg.TRAIN, 'EARLY_STOP_PATIENCE', 0))}",
        f"eval_mode={eval_label} | baseline_auroc={baseline_label} | output_dir={cfg.OUTPUT_DIR}",
        "-" * 78,
    ]


def _print_training_header(cfg, max_epoch, train_loader, test_loader, use_alternating, baseline_auroc):
    for line in _training_header_lines(cfg, max_epoch, train_loader, test_loader, use_alternating, baseline_auroc):
        print(line, flush=True)


def _progress_enabled(cfg):
    return bool(getattr(cfg.TRAIN, "SHOW_PROGRESS", False)) and tqdm is not None


def _iter_with_progress(iterable, cfg, total=None, desc=None, unit="batch"):
    if not _progress_enabled(cfg):
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
    )


def _train_progress_desc(phase, cur_epoch=None, max_epoch=None):
    if cur_epoch is None or max_epoch is None:
        return f"train {phase}"
    return f"train {phase} epoch {cur_epoch + 1}/{max_epoch}"

def train_epoch(
    train_loader,
    model,
    visual_optimizer,
    text_optimizer,
    tokenizer,
    cfg,
    phase,
    cur_epoch=None,
    max_epoch=None,
    return_details=False,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """
    # Enable train mode.
    model.train()
    base_model = _get_base_model(model)
    if hasattr(base_model, "set_train_phase"):
        base_model.set_train_phase(phase)
    optimizer = visual_optimizer if phase == "visual" else text_optimizer

    all_loss = 0.0
    loss_part_sums = defaultdict(float)
    loss_part_count = 0
    device = torch.device("cuda", torch.cuda.current_device()) if cfg.NUM_GPUS else None
    train_iter = _iter_with_progress(
        enumerate(train_loader),
        cfg,
        total=len(train_loader),
        desc=_train_progress_desc(phase, cur_epoch, max_epoch),
    )
    for cur_iter, batch in train_iter:
        query_image, prompt_images, types, labels, masks = _split_batch_with_optional_masks(batch, device=device)

        if _is_adapt_model(cfg):
            outputs = model(
                tokenizer=tokenizer,
                query_image=query_image,
                prompt_images=prompt_images,
                obj_types=types,
                return_aux=False,
                return_dict=True,
            )
            loss, loss_parts = compute_inctrl_pqa_loss(outputs, labels.float(), masks, cfg)
            for name, value in loss_parts.items():
                loss_part_sums[name] += float(value)
            loss_part_count += 1
        else:
            inputs, types, labels = batch[:3]
            if cfg.NUM_GPUS:
                labels = labels.cuda()
            preds, image_score = model(tokenizer, inputs, types)
            loss_fun = BinaryFocalLoss()
            if cfg.NUM_GPUS:
                loss_fun = loss_fun.cuda()
            loss = loss_fun(preds, labels.float()) + loss_fun(image_score, labels.float())

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # dist.all_reduce(loss)
        loss_value = loss.item()
        all_loss = all_loss + loss_value
        if hasattr(train_iter, "set_postfix"):
            avg_loss = all_loss / (cur_iter + 1)
            train_iter.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{avg_loss:.4f}",
                lr=f"{_optimizer_lr(optimizer):.1e}",
            )

    all_loss = all_loss / (cur_iter + 1)
    avg_parts = {
        name: value / loss_part_count
        for name, value in loss_part_sums.items()
    } if loss_part_count else {}
    if return_details:
        return all_loss, avg_parts
    return all_loss


@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, mode=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    metric_device = torch.device("cuda", torch.cuda.current_device()) if cfg.NUM_GPUS else torch.device("cpu")
    total_label = torch.Tensor([]).to(metric_device)
    total_pred = torch.Tensor([]).to(metric_device)

    device = metric_device if cfg.NUM_GPUS else None
    eval_iter = _iter_with_progress(
        enumerate(val_loader),
        cfg,
        total=len(val_loader),
        desc=f"eval {mode}" if mode else "eval",
    )
    for cur_iter, batch in eval_iter:
        query_image, prompt_images, types, labels, _ = _split_batch_with_optional_masks(batch, device=device)

        if _is_adapt_model(cfg):
            outputs = model(
                tokenizer=tokenizer,
                query_image=query_image,
                prompt_images=prompt_images,
                obj_types=types,
                return_aux=False,
                return_dict=True,
            )
            preds = _select_adapt_score(outputs, cfg)
        else:
            inputs, types, labels = batch[:3]
            if cfg.NUM_GPUS:
                labels = labels.cuda()
            preds, _ = model(tokenizer, inputs, types)

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)

    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()

    print("Predict " + mode + " set: ")
    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc, total_pr

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS:
        device = torch.cuda.current_device()

    # Build the model and print model statistics.
    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = _build_active_model(cfg, model_cfg, cast_dtype=cast_dtype, quick_gelu=quick_gelu)

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        # Transfer the model to the current GPU device
        model = model.cuda(device=device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device
        )

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Load a checkpoint to resume training if applicable.
    with pathmgr.open("./vit_b_16_plus_240-laion400m_e32-699c4b84.pt", "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    start_epoch = 0
    # model = model.module
    model.load_state_dict(checkpoint, strict=False)

    # Detect alternating training eligibility before building optimizers.
    # VA/TA alternation is only useful when VA contributes a score/loss signal.
    base_model = _get_base_model(model)
    has_visual = _has_parameter_list(base_model, "get_visual_parameters")
    has_text = _has_parameter_list(base_model, "get_text_parameters")
    use_alternating = _should_use_alternating_training(base_model, cfg)
    visual_optimizer, text_optimizer = _build_alternating_optimizers(
        model, lr=1e-3, use_alternating=use_alternating,
    )

    # Cosine LR schedule: decay from 1e-3 → COSINE_MIN_LR over max_epoch.
    _max_ep = _resolve_max_epochs(cfg)
    _min_lr = float(getattr(cfg.SOLVER, "COSINE_MIN_LR", 1e-5))
    visual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        visual_optimizer, T_max=_max_ep, eta_min=_min_lr,
    )
    text_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            text_optimizer, T_max=_max_ep, eta_min=_min_lr,
        )
        if text_optimizer is not visual_optimizer
        else visual_scheduler
    )

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train", transform)
    test_loader = loader.construct_loader(cfg, "test", transform)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    if use_alternating:
        logger.info("Alternating training: VA/TA phase switching enabled")
    else:
        phase_label = "joint" if has_visual and has_text else ("visual" if has_visual else "text")
        logger.info(f"Single-phase training: {phase_label} only (alternating disabled)")
    epoch_losses = []

    epoch_timer = EpochTimer()
    max_epoch = _resolve_max_epochs(cfg)
    history_rows = []
    best_val_auroc = None
    baseline_auroc = _cfg_float(cfg, "eval_baseline_auroc", default=None)
    history_path = os.path.join(cfg.OUTPUT_DIR, "train_history.csv")
    latest_path = os.path.join(cfg.OUTPUT_DIR, "latest_metrics.json")
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "checkpoint.pyth")
    best_checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "checkpoint_best.pyth")
    patience = int(getattr(cfg.TRAIN, "EARLY_STOP_PATIENCE", 0))
    epochs_without_improvement = 0
    _print_training_header(cfg, max_epoch, train_loader, test_loader, use_alternating, baseline_auroc)
    for cur_epoch in range(start_epoch, max_epoch):
        epoch_start = time.time()
        phase = _resolve_train_phase(cur_epoch, use_alternating, has_visual, has_text)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss, loss_parts = train_epoch(
            train_loader,
            model,
            visual_optimizer,
            text_optimizer,
            tokenizer,
            cfg,
            phase,
            cur_epoch=cur_epoch,
            max_epoch=max_epoch,
            return_details=True,
        )
        epoch_losses.append(epoch_loss)
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        did_eval = _should_eval_epoch(cur_epoch, max_epoch, cfg)
        if did_eval:
            test_roc, test_pr = eval_epoch(test_loader, model, cfg, tokenizer, "test")
        else:
            test_roc, test_pr = None, None
        record, best_val_auroc = _build_epoch_record(
            epoch=cur_epoch + 1,
            phase=phase,
            train_loss=epoch_loss,
            loss_parts=loss_parts,
            val_auroc=test_roc,
            val_aupr=test_pr,
            best_val_auroc=best_val_auroc,
            baseline_auroc=baseline_auroc,
            elapsed_sec=time.time() - epoch_start,
            lr=_optimizer_lr(visual_optimizer if phase == "visual" else text_optimizer),
            did_eval=did_eval,
        )
        history_rows.append(record)
        _write_csv(history_path, history_rows, TRAIN_HISTORY_FIELDS)
        _write_json(latest_path, _latest_metrics_payload(cfg.OUTPUT_DIR, history_rows, checkpoint_path))
        print(_format_epoch_summary(record), flush=True)

        # Save best checkpoint when val AUROC improves.
        if did_eval and test_roc is not None and best_val_auroc is not None:
            if float(test_roc) >= float(best_val_auroc):
                torch.save({
                    "epoch": cur_epoch + 1,
                    "model_state": model.state_dict(),
                    "cfg": cfg.dump(),
                }, best_checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Early stopping check.
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {cur_epoch + 1} (no improvement for {patience} eval epochs)", flush=True)
            break

        # Step LR schedulers.
        visual_scheduler.step()
        if text_scheduler is not visual_scheduler:
            text_scheduler.step()

    # Always save last checkpoint for resumability.
    torch.save({
        "epoch": cur_epoch + 1,
        "model_state": model.state_dict(),
        "cfg": cfg.dump(),
    }, checkpoint_path)
    _write_json(latest_path, _latest_metrics_payload(cfg.OUTPUT_DIR, history_rows, checkpoint_path))

    # Use best checkpoint for evaluation if it exists, otherwise fall back to last.
    if os.path.exists(best_checkpoint_path):
        cfg.TEST.CHECKPOINT_FILE_PATH = best_checkpoint_path
        # Reload best weights into the returned model.
        best_state = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(best_state["model_state"], strict=False)
        print(f"Loaded best checkpoint (epoch {best_state['epoch']}) for evaluation.", flush=True)
    else:
        cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path

    return model, tokenizer, transform


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


def eval_per_category(model, tokenizer, transform, cfg, dataset_name):
    """Evaluate model per-category on a dataset, return per-cat and mean metrics."""
    from train_local import DATASET_CATEGORIES

    categories = DATASET_CATEGORIES.get(dataset_name.lower(), [dataset_name])
    json_dir = os.path.join("data", "AD_json", dataset_name.lower())

    results = []
    for cat in categories:
        cat_cfg = cfg.clone()
        cat_cfg.val_normal_json_path = [os.path.join(json_dir, f"{cat}_val_normal.json")]
        cat_cfg.val_outlier_json_path = [os.path.join(json_dir, f"{cat}_val_outlier.json")]
        test_loader = loader.construct_loader(cat_cfg, "test", transform)
        auroc, aupr = eval_epoch(test_loader, model, cat_cfg, tokenizer, f"test/{cat}")
        results.append({"category": cat, "auroc": auroc, "aupr": aupr})

    mean_auroc = np.mean([r["auroc"] for r in results])
    mean_aupr = np.mean([r["aupr"] for r in results])

    print(f"\n{'='*50}")
    print(f" Per-category results on {dataset_name}")
    print(f"{'='*50}")
    print(f" {'Category':<15} {'AUROC':>10} {'AUPR':>10}")
    print(f" {'-'*35}")
    for r in results:
        print(f" {r['category']:<15} {r['auroc']:>10.4f} {r['aupr']:>10.4f}")
    print(f" {'-'*35}")
    print(f" {'MEAN':<15} {mean_auroc:>10.4f} {mean_aupr:>10.4f}")
    print(f"{'='*50}")

    return results, mean_auroc, mean_aupr


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    device = torch.cuda.current_device()

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = _build_active_model(cfg, model_cfg, cast_dtype=cast_dtype, quick_gelu=quick_gelu)
    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    # Create meters.
    total_roc, total_pr = eval_epoch(load, model, cfg, tokenizer, mode)

    return total_roc, total_pr
