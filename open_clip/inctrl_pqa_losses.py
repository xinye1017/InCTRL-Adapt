from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for binary or 2-class segmentation.

    For 2-channel logits (shape [..., 2, H, W]): applies softmax then focal
    on the anomaly channel against binary targets.
    For 1-channel logits: applies sigmoid then binary focal.
    """
    targets = targets.float()
    # Squeeze channel dim from masks if present: [B,1,H,W] → [B,H,W].
    if targets.dim() == logits.dim() and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if logits.shape[1] == 2:
        probs = logits.softmax(dim=1)
        target_indices = targets.long()
        pt = probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        base_loss = F.cross_entropy(logits, target_indices, reduction="none")
    else:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), probs, 1.0 - probs)
        logit_ch1 = logits.squeeze(1)
        base_loss = F.binary_cross_entropy_with_logits(logit_ch1, targets, reduction="none")
    pt = pt.clamp(min=1e-6)
    loss = alpha * (1.0 - pt) ** gamma * base_loss
    return loss.mean()


def dice_loss_2ch(
    logits: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice loss for 2-channel logits (softmax) or 1-channel (sigmoid)."""
    masks = masks.float()
    # Squeeze channel dim from masks if present: [B,1,H,W] → [B,H,W].
    if masks.dim() == logits.dim() and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if logits.shape[1] == 2:
        probs = logits.softmax(dim=1)[:, 1]  # [B, H, W]
    else:
        probs = torch.sigmoid(logits).squeeze(1)
    # Sum over spatial dims (H, W) per batch.
    spatial_dims = tuple(range(probs.dim() - 2))  # handles [B,H,W] and [B,H,W] equally
    intersection = (probs * masks).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + masks.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def dice_loss(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    masks = masks.float()
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Segmentation loss: FocalLoss + DiceLoss for 2-ch or 1-ch logits."""
    return focal_loss(logits, masks) + dice_loss_2ch(logits, masks)


def compute_inctrl_pqa_loss(
    outputs: dict,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    cfg,
):
    final_logit = outputs["final_logit"]
    labels = labels.to(device=final_logit.device, dtype=torch.float32)
    pqa_enabled = bool(getattr(cfg.PQA, "ENABLE", True)) if hasattr(cfg, "PQA") else True
    seg_head_enabled = bool(getattr(cfg.PQA, "ENABLE_SEG_HEAD", True)) if hasattr(cfg, "PQA") else True
    text_enabled = bool(getattr(cfg.TEXT_BRANCH, "ENABLE", True)) if hasattr(cfg, "TEXT_BRANCH") else True
    text_mask_weight = float(getattr(cfg.LOSS, "TEXT_MASK_WEIGHT", 0.0)) if hasattr(cfg, "LOSS") else 0.0
    visual_weight = float(getattr(cfg.LOSS, "VISUAL_WEIGHT", 0.0)) if hasattr(cfg, "LOSS") else 0.0
    visual_mask_weight = float(getattr(cfg.LOSS, "VISUAL_MASK_WEIGHT", 0.0)) if hasattr(cfg, "LOSS") else 0.0

    final = F.binary_cross_entropy_with_logits(final_logit, labels)
    image = F.binary_cross_entropy_with_logits(outputs["image_logit"], labels)

    # PQA global loss: use cross-entropy on 2-class logits if available.
    if pqa_enabled and outputs.get("pqa_global_logits") is not None:
        pqa = F.cross_entropy(outputs["pqa_global_logits"], labels.long())
    elif pqa_enabled:
        pqa = F.binary_cross_entropy_with_logits(outputs["pqa_logit"], labels)
    else:
        pqa = final_logit.sum() * 0.0

    if not text_enabled:
        text = final_logit.sum() * 0.0
    elif outputs.get("text_logits") is not None:
        text = F.cross_entropy(outputs["text_logits"], labels.long())
    else:
        text = F.binary_cross_entropy_with_logits(outputs["text_logit"], labels)

    # PQA segmentation loss: FocalLoss + DiceLoss on 2-channel logits.
    if masks is None or not pqa_enabled or not seg_head_enabled:
        mask = final_logit.sum() * 0.0
    else:
        seg_logits = outputs["pqa_seg_logits"]
        mask = segmentation_loss(seg_logits, masks.to(device=seg_logits.device))

    if masks is None or not text_enabled or text_mask_weight <= 0.0 or outputs.get("text_map_logits") is None:
        text_mask = final_logit.sum() * 0.0
    else:
        text_map_logits = outputs["text_map_logits"]
        text_mask = segmentation_loss(text_map_logits, masks.to(device=text_map_logits.device))

    # Visual branch (VA + static text) losses.
    if visual_weight <= 0.0 or outputs.get("visual_logits") is None:
        visual = final_logit.sum() * 0.0
    else:
        visual = F.cross_entropy(outputs["visual_logits"], labels.long())

    if masks is None or visual_mask_weight <= 0.0 or outputs.get("visual_map_logits") is None:
        visual_mask = final_logit.sum() * 0.0
    else:
        visual_map_logits = outputs["visual_map_logits"]
        visual_mask = segmentation_loss(visual_map_logits, masks.to(device=visual_map_logits.device))

    total = (
        final
        + float(cfg.LOSS.IMAGE_WEIGHT) * image
        + float(cfg.LOSS.PQA_WEIGHT) * pqa
        + float(cfg.LOSS.TEXT_WEIGHT) * text
        + float(cfg.LOSS.MASK_WEIGHT) * mask
        + text_mask_weight * text_mask
        + visual_weight * visual
        + visual_mask_weight * visual_mask
    )
    parts = {
        "final": final.item(),
        "image": image.item(),
        "pqa": pqa.item(),
        "text": text.item(),
        "mask": mask.item(),
        "text_mask": text_mask.item(),
        "visual": visual.item(),
        "visual_mask": visual_mask.item(),
        "total": total.item(),
    }
    return total, parts
