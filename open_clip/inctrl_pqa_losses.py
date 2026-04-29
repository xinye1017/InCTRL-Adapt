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
    if logits.shape[1] == 2:
        probs = logits.softmax(dim=1)
        pt = torch.where(targets.bool(), probs[:, 1], probs[:, 0])
    else:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), probs, 1.0 - probs)
    pt = pt.clamp(min=1e-6)
    bce = F.binary_cross_entropy_with_logits(
        logits[:, 1] if logits.shape[1] == 2 else logits.squeeze(1),
        targets,
        reduction="none",
    )
    loss = alpha * (1.0 - pt) ** gamma * bce
    return loss.mean()


def dice_loss_2ch(
    logits: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice loss for 2-channel logits (softmax) or 1-channel (sigmoid)."""
    masks = masks.float()
    if logits.shape[1] == 2:
        probs = logits.softmax(dim=1)[:, 1]
    else:
        probs = torch.sigmoid(logits).squeeze(1)
    intersection = (probs * masks).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
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

    total = (
        final
        + float(cfg.LOSS.IMAGE_WEIGHT) * image
        + float(cfg.LOSS.PQA_WEIGHT) * pqa
        + float(cfg.LOSS.TEXT_WEIGHT) * text
        + float(cfg.LOSS.MASK_WEIGHT) * mask
        + text_mask_weight * text_mask
    )
    parts = {
        "final": final.item(),
        "image": image.item(),
        "pqa": pqa.item(),
        "text": text.item(),
        "mask": mask.item(),
        "text_mask": text_mask.item(),
        "total": total.item(),
    }
    return total, parts
