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
    if masks is None or local_logits is None:
        if isinstance(local_logits, torch.Tensor):
            return local_logits.new_zeros(())
        if isinstance(local_logits, (list, tuple)) and local_logits:
            return local_logits[0].new_zeros(())
        final_logit = outputs.get("final_logit")
        if isinstance(final_logit, torch.Tensor):
            return final_logit.new_zeros(())
        raise KeyError("outputs must contain 'pqa_local_logits' or 'final_logit'")

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


def compute_pqa_local_mil_loss(
    outputs: Dict[str, object],
    labels: torch.Tensor,
    topk_ratio: float = 0.01,
) -> torch.Tensor:
    local_logits = outputs.get("pqa_local_logits")
    if local_logits is None:
        final_logit = outputs.get("final_logit")
        if isinstance(final_logit, torch.Tensor):
            return final_logit.new_zeros(())
        raise KeyError("outputs must contain 'pqa_local_logits' or 'final_logit'")
    if isinstance(local_logits, torch.Tensor):
        local_logits = [local_logits]

    labels = labels.float()
    losses = []
    for logits in local_logits:
        anomaly_prob = torch.softmax(logits, dim=1)[:, 1].flatten(1)
        topk_count = max(1, int(anomaly_prob.shape[-1] * float(topk_ratio)))
        topk_count = min(topk_count, anomaly_prob.shape[-1])
        local_score = anomaly_prob.topk(topk_count, dim=-1).values.mean(dim=-1)
        losses.append(F.binary_cross_entropy(local_score.clamp(1e-6, 1.0 - 1e-6), labels))

    return torch.stack(losses).mean()


def compute_training_loss(
    outputs: Dict[str, object],
    labels: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    phase: Optional[str] = None,
    masks: Optional[torch.Tensor] = None,
    pqa_loss_weight: float = 1.0,
    mask_loss_weight: float = 1.0,
    image_loss_weight: float = 1.0,
    local_mil_loss_weight: float = 0.0,
    local_mil_topk_ratio: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute fused training loss using only final/image/PQA logits and optional PQA masks."""
    del phase
    labels = labels.float()

    final_logit = outputs["final_logit"]
    if not isinstance(final_logit, torch.Tensor):
        raise TypeError("outputs['final_logit'] must be a tensor")

    zero = final_logit.new_zeros(())
    final_loss = loss_fn(final_logit, labels)

    image_logit = outputs.get("image_logit")
    if image_loss_weight > 0 and not isinstance(image_logit, torch.Tensor):
        raise KeyError("outputs['image_logit'] is required when image_loss_weight > 0")
    image_loss = loss_fn(image_logit, labels) if image_loss_weight > 0 else zero

    pqa_logit = outputs.get("pqa_logit")
    if pqa_loss_weight > 0 and not isinstance(pqa_logit, torch.Tensor):
        raise KeyError("outputs['pqa_logit'] is required when pqa_loss_weight > 0")
    pqa_loss = loss_fn(pqa_logit, labels) if pqa_loss_weight > 0 else zero

    pqa_mask_loss = compute_pqa_mask_loss(outputs, masks) if masks is not None and mask_loss_weight > 0 else zero
    pqa_local_mil_loss = (
        compute_pqa_local_mil_loss(outputs, labels, topk_ratio=local_mil_topk_ratio)
        if masks is None and local_mil_loss_weight > 0
        else zero
    )

    total_loss = (
        final_loss
        + pqa_loss_weight * pqa_loss
        + image_loss_weight * image_loss
        + mask_loss_weight * pqa_mask_loss
        + local_mil_loss_weight * pqa_local_mil_loss
    )

    metrics = {
        "final_loss": final_loss.detach(),
        "image_loss": image_loss.detach(),
        "pqa_loss": pqa_loss.detach(),
        "pqa_mask_loss": pqa_mask_loss.detach(),
        "pqa_local_mil_loss": pqa_local_mil_loss.detach(),
        "total_loss": total_loss.detach(),
    }
    return total_loss, metrics
