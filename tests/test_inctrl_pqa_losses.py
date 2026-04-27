from types import SimpleNamespace

import torch

from open_clip.inctrl_pqa_losses import compute_inctrl_pqa_loss, dice_loss


def _cfg():
    return SimpleNamespace(
        PQA=SimpleNamespace(ENABLE=True, ENABLE_SEG_HEAD=True),
        TEXT_BRANCH=SimpleNamespace(ENABLE=True),
        LOSS=SimpleNamespace(
            IMAGE_WEIGHT=1.0,
            PQA_WEIGHT=0.5,
            TEXT_WEIGHT=0.0,
            MASK_WEIGHT=1.0,
        )
    )


def test_dice_loss_is_near_zero_for_perfect_mask():
    logits = torch.full((1, 1, 4, 4), 20.0)
    masks = torch.ones(1, 1, 4, 4)

    loss = dice_loss(logits, masks)

    assert loss < 1e-4


def test_compute_inctrl_pqa_loss_uses_image_pqa_text_and_mask_terms():
    outputs = {
        "final_logit": torch.tensor([0.0, 1.0]),
        "image_logit": torch.tensor([0.0, 1.0]),
        "pqa_logit": torch.tensor([0.0, 1.0]),
        "text_logit": torch.tensor([0.0, 1.0]),
        "pqa_seg_logits": torch.randn(2, 1, 8, 8),
    }
    labels = torch.tensor([0, 1])
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 2:5, 2:5] = 1.0

    loss, parts = compute_inctrl_pqa_loss(outputs, labels, masks, _cfg())

    assert set(parts.keys()) == {"final", "image", "pqa", "text", "mask", "total"}
    assert parts["mask"] > 0.0
    assert parts["total"] == loss.item()


def test_compute_inctrl_pqa_loss_allows_missing_masks_with_zero_mask_term():
    outputs = {
        "final_logit": torch.tensor([0.0, 1.0]),
        "image_logit": torch.tensor([0.0, 1.0]),
        "pqa_logit": torch.tensor([0.0, 1.0]),
        "text_logit": torch.tensor([0.0, 1.0]),
        "pqa_seg_logits": torch.randn(2, 1, 8, 8),
    }
    labels = torch.tensor([0, 1])

    loss, parts = compute_inctrl_pqa_loss(outputs, labels, masks=None, cfg=_cfg())

    assert set(parts.keys()) == {"final", "image", "pqa", "text", "mask", "total"}
    assert parts["mask"] == 0.0
    assert parts["total"] == loss.item()


def test_compute_inctrl_pqa_loss_skips_pqa_and_mask_terms_when_pqa_disabled():
    cfg = _cfg()
    cfg.PQA.ENABLE = False
    outputs = {
        "final_logit": torch.tensor([0.0, 1.0]),
        "image_logit": torch.tensor([0.0, 1.0]),
        "pqa_logit": torch.tensor([0.0, 1.0]),
        "text_logit": torch.tensor([0.0, 1.0]),
        "pqa_seg_logits": torch.randn(2, 1, 8, 8),
    }
    labels = torch.tensor([0, 1])
    masks = torch.ones(2, 1, 8, 8)

    loss, parts = compute_inctrl_pqa_loss(outputs, labels, masks, cfg)

    assert parts["pqa"] == 0.0
    assert parts["mask"] == 0.0
    assert parts["total"] == loss.item()


def test_compute_inctrl_pqa_loss_skips_mask_term_when_seg_head_disabled():
    cfg = _cfg()
    cfg.PQA.ENABLE_SEG_HEAD = False
    outputs = {
        "final_logit": torch.tensor([0.0, 1.0]),
        "image_logit": torch.tensor([0.0, 1.0]),
        "pqa_logit": torch.tensor([0.0, 1.0]),
        "text_logit": torch.tensor([0.0, 1.0]),
        "pqa_seg_logits": torch.randn(2, 1, 8, 8),
    }
    labels = torch.tensor([0, 1])
    masks = torch.ones(2, 1, 8, 8)

    loss, parts = compute_inctrl_pqa_loss(outputs, labels, masks, cfg)

    # PQA logit loss still computed (alignment-only), but mask loss is zero
    assert parts["pqa"] > 0.0
    assert parts["mask"] == 0.0
    assert parts["total"] == loss.item()
