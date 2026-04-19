from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def binary_dice_loss(probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.reshape(probabilities.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def multiclass_focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    return ((1.0 - pt) ** gamma * ce_loss).mean()


def compute_pqa_mask_loss(
    outputs: Dict[str, object],
    masks: Optional[torch.Tensor],
) -> torch.Tensor:
    local_logits = outputs.get("pqa_local_logits")
    if local_logits is None or masks is None:
        return outputs["final_logit"].new_zeros(())

    if isinstance(local_logits, torch.Tensor):
        local_logits = [local_logits]

    masks = masks.float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    losses = []
    for logits in local_logits:
        target_mask = F.interpolate(masks, size=logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
        target_labels = target_mask.squeeze(1).long()
        probabilities = torch.softmax(logits, dim=1)
        focal_loss = multiclass_focal_loss(logits, target_labels)
        anomaly_dice = binary_dice_loss(probabilities[:, 1], target_mask.squeeze(1))
        normal_dice = binary_dice_loss(probabilities[:, 0], 1.0 - target_mask.squeeze(1))
        losses.append(focal_loss + anomaly_dice + normal_dice)

    return torch.stack(losses).mean()


def compute_training_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    phase: Optional[str] = None,
    masks: Optional[torch.Tensor] = None,
    pqa_loss_weight: float = 1.0,
    mask_loss_weight: float = 1.0,
    image_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fused PQA objective: supervise final decision, PQA global, and optional PQA masks."""
    del phase
    labels = labels.float()
    zero = outputs["final_logit"].new_zeros(())
    final_loss = loss_fn(outputs["final_logit"], labels)
    image_loss = loss_fn(outputs["image_logit"], labels) if image_loss_weight > 0 else zero
    pqa_loss = loss_fn(outputs["pqa_logit"], labels) if pqa_loss_weight > 0 else zero
    pqa_mask_loss = compute_pqa_mask_loss(outputs, masks) if masks is not None and mask_loss_weight > 0 else zero
    total_loss = (
        final_loss
        + image_loss_weight * image_loss
        + pqa_loss_weight * pqa_loss
        + mask_loss_weight * pqa_mask_loss
    )

    return total_loss, {
        "final_loss": final_loss.detach(),
        "image_loss": image_loss.detach(),
        "pqa_loss": pqa_loss.detach(),
        "pqa_mask_loss": pqa_mask_loss.detach(),
        "total_loss": total_loss.detach(),
    }
