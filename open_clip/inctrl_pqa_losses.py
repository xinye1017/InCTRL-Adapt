from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    masks = masks.float()
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.float()
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    return bce + dice_loss(logits, masks)


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

    final = F.binary_cross_entropy_with_logits(final_logit, labels)
    image = F.binary_cross_entropy_with_logits(outputs["image_logit"], labels)
    pqa = (
        F.binary_cross_entropy_with_logits(outputs["pqa_logit"], labels)
        if pqa_enabled
        else final_logit.sum() * 0.0
    )
    text = (
        F.binary_cross_entropy_with_logits(outputs["text_logit"], labels)
        if text_enabled
        else final_logit.sum() * 0.0
    )
    if masks is None or not pqa_enabled or not seg_head_enabled:
        mask = final_logit.sum() * 0.0
    else:
        seg_logits = outputs["pqa_seg_logits"]
        mask = segmentation_loss(seg_logits, masks.to(device=seg_logits.device))

    total = (
        final
        + float(cfg.LOSS.IMAGE_WEIGHT) * image
        + float(cfg.LOSS.PQA_WEIGHT) * pqa
        + float(cfg.LOSS.TEXT_WEIGHT) * text
        + float(cfg.LOSS.MASK_WEIGHT) * mask
    )
    parts = {
        "final": final.item(),
        "image": image.item(),
        "pqa": pqa.item(),
        "text": text.item(),
        "mask": mask.item(),
        "total": total.item(),
    }
    return total, parts
