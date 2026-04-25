from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossProb(nn.Module):
    """Binary focal loss for probability inputs."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.float().clamp(1e-7, 1.0 - 1e-7)
        targets = targets.float()
        bce = -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log())
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss with explicit all-zero-mask handling."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.float().reshape(pred.shape[0], -1)
        target_flat = target.float().reshape(target.shape[0], -1)
        losses = []
        for pred_row, target_row in zip(pred_flat, target_flat):
            if target_row.sum() == 0:
                losses.append(pred_row.mean())
                continue
            intersection = (pred_row * target_row).sum()
            union = pred_row.sum() + target_row.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            losses.append(1.0 - dice)
        return torch.stack(losses).mean()


def _mask_to_flat(mask: torch.Tensor, reference_flat: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.dim() == 2:
        return mask.float()
    if mask.dim() != 3:
        raise ValueError(f"mask must be [B,H,W], [B,1,H,W], or [B,N], got {tuple(mask.shape)}")
    batch_size, num_patches = reference_flat.shape
    grid = int(num_patches ** 0.5)
    resized = F.interpolate(mask.float().unsqueeze(1), size=(grid, grid), mode="nearest")
    return resized.squeeze(1).reshape(batch_size, num_patches)


def pixel_focal_loss_2class(normal_prob: torch.Tensor, abnormal_prob: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    focal = BinaryFocalLossProb()
    return focal(abnormal_prob, mask_flat) + focal(normal_prob, 1.0 - mask_flat)


def compute_dasl_loss(
    outputs: dict,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    lambda_image: float = 1.0,
    lambda_pixel: float = 1.0,
    disable_pixel_loss: bool = False,
) -> dict:
    focal = BinaryFocalLossProb()
    dice = DiceLoss()
    labels = labels.float()

    image_loss = focal(outputs["image_score"], labels)
    if masks is None or disable_pixel_loss:
        pixel_loss = image_loss.new_tensor(0.0)
    else:
        mask_flat = _mask_to_flat(masks, outputs["S_a"])
        semantic_focal = pixel_focal_loss_2class(outputs["S_n"], outputs["S_a"], mask_flat)
        dice_semantic = dice(outputs["S_a"], mask_flat)
        dice_fused = dice(outputs["pixel_map_dasl"], mask_flat)
        pixel_loss = semantic_focal + dice_semantic + dice_fused

    total = lambda_image * image_loss + lambda_pixel * pixel_loss
    return {
        "loss_image": image_loss,
        "loss_pixel_dasl": pixel_loss,
        "loss_dasl": total,
    }


def compute_oasl_loss(
    outputs: dict,
    masks: torch.Tensor | None = None,
    lambda_pixel: float = 1.0,
    disable_pixel_loss: bool = False,
) -> dict:
    dice = DiceLoss()
    if disable_pixel_loss:
        pixel_loss = outputs["S_hat_a"].new_tensor(0.0)
    else:
        if masks is None:
            mask_flat = torch.zeros_like(outputs["S_hat_a"])
        else:
            mask_flat = _mask_to_flat(masks, outputs["S_hat_a"])
        semantic_focal = pixel_focal_loss_2class(outputs["S_hat_n"], outputs["S_hat_a"], mask_flat)
        dice_semantic = dice(outputs["S_hat_a"], mask_flat)
        dice_fused = dice(outputs["pixel_map_oasl"], mask_flat)
        pixel_loss = semantic_focal + dice_semantic + dice_fused
    return {
        "loss_oasl": lambda_pixel * pixel_loss,
        "loss_pixel_oasl": pixel_loss,
    }


def compute_inctrlv2_loss(
    main_outputs: dict,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    oasl_outputs: dict | None = None,
    oasl_masks: torch.Tensor | None = None,
    lambda_image: float = 1.0,
    lambda_pixel: float = 1.0,
    lambda_oasl: float = 1.0,
    disable_dasl: bool = False,
    disable_oasl: bool = False,
    disable_pixel_loss: bool = False,
) -> dict:
    if disable_dasl:
        image_loss = BinaryFocalLossProb()(main_outputs["image_score"], labels.float())
        dasl = {
            "loss_image": image_loss,
            "loss_pixel_dasl": image_loss.new_tensor(0.0),
            "loss_dasl": lambda_image * image_loss,
        }
    else:
        dasl = compute_dasl_loss(
            main_outputs,
            labels,
            masks,
            lambda_image=lambda_image,
            lambda_pixel=lambda_pixel,
            disable_pixel_loss=disable_pixel_loss,
        )

    if disable_oasl or oasl_outputs is None:
        oasl = {
            "loss_oasl": dasl["loss_dasl"].new_tensor(0.0),
            "loss_pixel_oasl": dasl["loss_dasl"].new_tensor(0.0),
        }
    else:
        oasl = compute_oasl_loss(
            oasl_outputs,
            masks=oasl_masks,
            lambda_pixel=lambda_pixel,
            disable_pixel_loss=disable_pixel_loss,
        )

    total = dasl["loss_dasl"] + lambda_oasl * oasl["loss_oasl"]
    return {
        "loss": total,
        **dasl,
        **oasl,
    }
