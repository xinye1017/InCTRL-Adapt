# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
import csv
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
from open_clip.model import get_cast_dtype
from open_clip.inctrl_adapt import InCTRLAdapt
from open_clip.inctrl_pqa_losses import compute_inctrl_pqa_loss
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.get_logger(__name__)


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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


def _trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def _build_alternating_optimizers(model, lr=1e-3):
    base_model = _get_base_model(model)
    if not hasattr(base_model, "get_visual_parameters") or not hasattr(base_model, "get_text_parameters"):
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
            loss, _ = compute_inctrl_pqa_loss(outputs, labels.float(), masks, cfg)
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
            train_iter.set_postfix(loss=f"{loss_value:.4f}")

    all_loss = all_loss / (cur_iter + 1)
    print("train_loss: ", all_loss)
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
            score_key = str(getattr(cfg.FUSION, "SCORE_OUTPUT", "final_score"))
            preds = outputs.get(score_key, outputs["final_score"])
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

    visual_optimizer, text_optimizer = _build_alternating_optimizers(model, lr=1e-3)

    # Detect alternating training eligibility
    base_model = _get_base_model(model)
    has_visual = hasattr(base_model, "get_visual_parameters") and len(base_model.get_visual_parameters()) > 0
    has_text = hasattr(base_model, "get_text_parameters") and len(base_model.get_text_parameters()) > 0
    use_alternating = has_visual and has_text

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train", transform)
    test_loader = loader.construct_loader(cfg, "test", transform)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    if use_alternating:
        logger.info("Alternating training: VA/TA phase switching enabled")
    else:
        phase_label = "visual" if has_visual else "text"
        logger.info(f"Single-phase training: {phase_label} only (alternating disabled)")
    epoch_losses = []

    epoch_timer = EpochTimer()
    max_epoch = _resolve_max_epochs(cfg)
    history_rows = []
    for cur_epoch in range(start_epoch, max_epoch):
        print("Epoch: ", cur_epoch)
        if use_alternating:
            phase = "visual" if cur_epoch % 2 == 0 else "text"
        else:
            phase = "visual" if has_visual else "text"
        print("Train phase: ", phase)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss = train_epoch(
            train_loader,
            model,
            visual_optimizer,
            text_optimizer,
            tokenizer,
            cfg,
            phase,
            cur_epoch=cur_epoch,
            max_epoch=max_epoch,
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

        eval_period = max(1, max_epoch // 5)  # eval ~5 times total
        if cur_epoch == 0 or (cur_epoch + 1) % eval_period == 0 or cur_epoch == max_epoch - 1:
            test_roc, test_pr = eval_epoch(test_loader, model, cfg, tokenizer, "test")
        else:
            test_roc, test_pr = 0.0, 0.0
        history_rows.append({
            "epoch": cur_epoch + 1,
            "phase": phase,
            "train_loss": epoch_loss,
            "train_auroc": 0.0,
            "train_aupr": 0.0,
            "val_auroc": test_roc,
            "val_aupr": test_pr,
        })

    _write_csv(
        os.path.join(cfg.OUTPUT_DIR, "train_history.csv"),
        history_rows,
        ["epoch", "phase", "train_loss", "train_auroc", "train_aupr", "val_auroc", "val_aupr"],
    )
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "checkpoint.pyth")
    torch.save({
        "epoch": max_epoch,
        "model_state": model.state_dict(),
        "cfg": cfg.dump(),
    }, checkpoint_path)
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
