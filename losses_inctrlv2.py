"""
InCTRLv2 losses: DASL (L_I + L_P) + OASL (L_OASL).

Paper-faithful reimplementation based on InCTRLv2 description.
Not official code.

Usage:
    outputs = model(tokenizer, image, text, normal_list, masks, mode='train')
    loss_dict = compute_inctrlv2_loss(outputs, labels, masks, cfg)
    loss_dict['loss'].backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Focal Loss (binary, works on probabilities)                                 #
# --------------------------------------------------------------------------- #
class BinaryFocalLossProb(nn.Module):
    """Focal loss that expects probability inputs (not logits)."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        """
        probs:   [*]  predicted probability of positive class, in (0,1)
        targets: [*]  ground-truth, 0 or 1
        """
        probs = probs.clamp(1e-7, 1 - 1e-7)
        bce = -(targets * probs.log() + (1 - targets) * (1 - probs).log())
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# --------------------------------------------------------------------------- #
#  Dice Loss                                                                   #
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   [B, H, W] or [B, N]  predicted anomaly probability
        target: [B, H, W] or [B, N]  ground-truth mask (0/1)
        """
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# --------------------------------------------------------------------------- #
#  Pixel-level focal loss (2-class: [Sn, Sa] vs mask)                         #
# --------------------------------------------------------------------------- #
def pixel_focal_loss_2class(Sn, Sa, mask, alpha=1.0, gamma=2.0):
    """
    2-class pixel focal loss as described in InCTRLv2.
    Sn: [B, N] or [B, H, W]  normality probability
    Sa: [B, N] or [B, H, W]  abnormality probability
    mask: same shape as Sn/Sa, binary ground truth (1=anomaly)
    """
    focal = BinaryFocalLossProb(alpha, gamma)
    loss_a = focal(Sa, mask)
    loss_n = focal(Sn, 1.0 - mask)
    return loss_a + loss_n


# --------------------------------------------------------------------------- #
#  Composite DASL loss                                                         #
# --------------------------------------------------------------------------- #
def compute_dasl_loss(outputs, labels, masks, cfg=None):
    """
    L_DASL = L_I + L_P

    L_I  = FocalLoss(score, label)             image-level
    L_P  = Focal([Sn, Sa], mask)               pixel semantic
         + Dice(Sa, mask)                      pixel semantic direct
         + Dice(Mp, mask)                      pixel fused

    Args:
        outputs: dict from InCTRLv2.forward()
        labels:  [B]  image-level labels (0=normal, 1=anomaly)
        masks:   [B, H, W] or [B, N]  pixel-level GT
        cfg:     optional config with loss weights
    """
    focal_img = BinaryFocalLossProb()
    dice = DiceLoss()

    # image-level
    L_I = focal_img(outputs['score'], labels.float())

    # decide mask resolution: if masks are full-res, downsample to patch grid
    Sa = outputs['Sa']       # [B, N]
    Sn = outputs['Sn']       # [B, N]
    Mp = outputs['Mp']       # [B, N]

    if masks is not None and masks.dim() == 3:
        # downsample mask to patch grid
        B, H, W = masks.shape
        gh = int(Sa.shape[-1] ** 0.5)
        mask_down = F.adaptive_avg_pool2d(
            masks.float().unsqueeze(1), (gh, gh)
        ).squeeze(1).reshape(B, -1)
        mask_down = (mask_down > 0.5).float()
    elif masks is not None:
        mask_down = masks.float()
    else:
        mask_down = torch.zeros_like(Sa)

    # pixel-level
    L_focal_px = pixel_focal_loss_2class(Sn, Sa, mask_down)
    L_dice_Sa = dice(Sa, mask_down)
    L_dice_Mp = dice(Mp, mask_down)

    L_P = L_focal_px + L_dice_Sa + L_dice_Mp

    return {
        'L_I': L_I,
        'L_P': L_P,
        'L_DASL': L_I + L_P,
    }


# --------------------------------------------------------------------------- #
#  Composite OASL loss                                                         #
# --------------------------------------------------------------------------- #
def compute_oasl_loss(outputs, cfg=None):
    """
    L_OASL = Focal([Shat_n, Shat_a], zero_mask)
           + Dice(Shat_a, zero_mask)
           + Dice(Mn, zero_mask)

    OASL is trained on normal-only data → mask is all zeros.
    """
    dice = DiceLoss()

    Shat_n = outputs['Shat_n']   # [B, N]
    Shat_a = outputs['Shat_a']   # [B, N]
    Mn = outputs['Mn']           # [B, N]
    zero_mask = torch.zeros_like(Shat_a)

    L_focal_px = pixel_focal_loss_2class(Shat_n, Shat_a, zero_mask)
    L_dice_Shat_a = dice(Shat_a, zero_mask)
    L_dice_Mn = dice(Mn, zero_mask)

    L_OASL = L_focal_px + L_dice_Shat_a + L_dice_Mn

    return {
        'L_OASL': L_OASL,
    }


# --------------------------------------------------------------------------- #
#  Total loss                                                                  #
# --------------------------------------------------------------------------- #
def compute_inctrlv2_loss(outputs, labels, masks, cfg=None):
    """
    L_total = lambda_dasl * L_DASL + lambda_oasl * L_OASL

    Args:
        outputs: dict from InCTRLv2.forward()
        labels:  [B] image-level labels
        masks:   [B, H, W] or [B, N] pixel GT, can be None
        cfg:     optional, expects .lambda_dasl, .lambda_oasl
    Returns:
        dict with 'loss', 'L_DASL', 'L_OASL', and sub-components
    """
    lambda_dasl = getattr(cfg, 'lambda_dasl', 1.0) if cfg else 1.0
    lambda_oasl = getattr(cfg, 'lambda_oasl', 1.0) if cfg else 1.0

    dasl = compute_dasl_loss(outputs, labels, masks, cfg)
    oasl = compute_oasl_loss(outputs, cfg)

    total = lambda_dasl * dasl['L_DASL'] + lambda_oasl * oasl['L_OASL']

    result = {
        'loss': total,
        **dasl,
        **oasl,
    }
    return result
