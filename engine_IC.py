# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
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
from open_clip.inctrl_pqa_fused import InCTRLPQA
from open_clip.inctrl_pqa_losses import compute_training_loss
from open_clip.utils.env import checkpoint_pathmgr as pathmgr

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')


def _split_query_prompt_batch(inputs, device=None):
    query_image = inputs[0]
    prompt_images = torch.stack(inputs[1:], dim=1)
    if device is not None:
        query_image = query_image.to(device)
        prompt_images = prompt_images.to(device)
    return query_image, prompt_images


def _unpack_batch(batch):
    if len(batch) == 4:
        inputs, types, labels, masks = batch
        return inputs, types, labels, masks
    inputs, types, labels = batch
    return inputs, types, labels, None


def _resolve_max_epochs(cfg, legacy_default=10):
    max_epochs = int(getattr(cfg.SOLVER, "MAX_EPOCH", legacy_default))
    return legacy_default if max_epochs == 400 else max_epochs


def _get_base_model(model):
    return model.module if hasattr(model, "module") else model


def _build_optimizer(model, lr=1e-3):
    base_model = _get_base_model(model)
    return torch.optim.AdamW(
        base_model.get_trainable_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
    )

def train_epoch(
    train_loader,
    model,
    optimizer,
    tokenizer,
    cfg,
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
    base_model.enable_pqa_training()

    all_loss = 0.0
    loss_sums = {
        "final_loss": 0.0,
        "pqa_loss": 0.0,
        "pqa_mask_loss": 0.0,
        "pqa_local_mil_loss": 0.0,
        "image_loss": 0.0,
    }
    loss_fun = BinaryFocalLoss(logits=True)
    if cfg.NUM_GPUS:
        loss_fun = loss_fun.cuda()
    pqa_loss_weight = float(getattr(cfg.PQA, "GLOBAL_LOSS_WEIGHT", 1.0))
    mask_loss_weight = float(getattr(cfg.PQA, "MASK_LOSS_WEIGHT", 1.0))
    image_loss_weight = float(getattr(cfg.PQA, "IMAGE_LOSS_WEIGHT", 0.0))
    local_mil_loss_weight = float(getattr(cfg.PQA, "LOCAL_MIL_LOSS_WEIGHT", 0.0))
    local_mil_topk_ratio = float(getattr(cfg.PQA, "LOCAL_MIL_TOPK_RATIO", 0.01))

    max_steps = int(getattr(cfg, "steps_per_epoch", 0) or 0)
    actual_steps = 0
    for cur_iter, batch in enumerate(train_loader):
        if max_steps > 0 and cur_iter >= max_steps:
            break
        inputs, types, labels, masks = _unpack_batch(batch)

        if cfg.NUM_GPUS:
            labels = labels.cuda()
            query_image, prompt_images = _split_query_prompt_batch(inputs, labels.device)
            if masks is not None:
                masks = masks.cuda(non_blocking=True)
        else:
            query_image, prompt_images = _split_query_prompt_batch(inputs)

        outputs = model(
            query_image=query_image,
            prompt_images=prompt_images,
            obj_types=types,
            return_aux=False,
            return_dict=True,
        )

        # Compute the loss.
        loss, loss_parts = compute_training_loss(
            outputs=outputs,
            labels=labels,
            loss_fn=loss_fun,
            masks=masks,
            pqa_loss_weight=pqa_loss_weight,
            mask_loss_weight=mask_loss_weight,
            image_loss_weight=image_loss_weight,
            local_mil_loss_weight=local_mil_loss_weight,
            local_mil_topk_ratio=local_mil_topk_ratio,
        )

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
        actual_steps += 1
        for loss_name in loss_sums:
            loss_sums[loss_name] += loss_parts[loss_name].item()

    all_loss = all_loss / max(actual_steps, 1)
    loss_avgs = {
        loss_name: loss_sum / max(actual_steps, 1)
        for loss_name, loss_sum in loss_sums.items()
    }
    print(
        "train_loss: "
        f"total={all_loss:.4f}, final={loss_avgs['final_loss']:.4f}, "
        f"pqa={loss_avgs['pqa_loss']:.4f}, mask={loss_avgs['pqa_mask_loss']:.4f}, "
        f"local_mil={loss_avgs['pqa_local_mil_loss']:.4f}, "
        f"image={loss_avgs['image_loss']:.4f}"
    )
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

    total_label = torch.Tensor([]).cuda()
    total_pred = torch.Tensor([]).cuda()

    for cur_iter, batch in enumerate(val_loader):
        inputs, types, labels, _ = _unpack_batch(batch)
        if cfg.NUM_GPUS:
            labels = labels.cuda()
            query_image, prompt_images = _split_query_prompt_batch(inputs, labels.device)
        else:
            query_image, prompt_images = _split_query_prompt_batch(inputs)

        outputs = model(
            query_image=query_image,
            prompt_images=prompt_images,
            obj_types=types,
            return_aux=False,
            return_dict=True,
        )
        preds = outputs["final_score"]

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)

    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()

    print("Predict " + mode + " set: ")
    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc

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

    model = InCTRLPQA(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)

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

    optimizer = _build_optimizer(model, lr=1e-3)

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train", transform)
    test_loader = loader.construct_loader(cfg, "test", transform)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_losses = []

    epoch_timer = EpochTimer()
    max_epochs = _resolve_max_epochs(cfg)
    for cur_epoch in range(start_epoch, max_epochs):
        print("Epoch: ", cur_epoch)
        print("Train phase: pqa_fused")
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            tokenizer,
            cfg,
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

        path = "./tmp/checkpoints/checkpoint_" + str(cur_epoch + 1) + ".pyth"
        torch.save(model.state_dict(), path)

        total_roc = eval_epoch(train_loader, model, cfg, tokenizer, "train")
        test_roc = eval_epoch(test_loader, model, cfg, tokenizer, "test")


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


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

    model = open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    # Create meters.
    total_roc = eval_epoch(load, model, cfg, tokenizer, mode)

    return total_roc
