import torch

from open_clip.config.defaults import get_cfg
from open_clip.inctrl_pqa_fused import InCTRLPQA
from open_clip.inctrl_pqa_losses import compute_pqa_mask_loss


def _build_args(image_size=32, shot=2):
    cfg = get_cfg()
    cfg.image_size = image_size
    cfg.shot = shot
    return cfg


def _build_model():
    args = _build_args()
    vision_cfg = {
        "image_size": 32,
        "layers": 12,
        "width": 64,
        "patch_size": 16,
        "head_width": 32,
        "mlp_ratio": 4.0,
        "output_tokens": True,
    }
    text_cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 32,
        "heads": 4,
        "layers": 2,
    }
    return InCTRLPQA(
        args,
        embed_dim=32,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        patch_layers=(7, 9, 11),
        hidden_dim=16,
        feature_is_projected=False,
    )


def _forward(model):
    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    return model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )


def test_fused_forward_exposes_decision_and_pqa_outputs():
    torch.manual_seed(31)
    model = _build_model()
    model.eval()

    outputs = _forward(model)

    assert outputs["final_logit"].shape == (2,)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["pqa_local_logits"].shape == (2, 2, 32, 32)
    assert outputs["image_logit"].shape == (2,)
    assert outputs["text_logit"].shape == (2,)
    assert outputs["base_patch_map"].shape == (2, 4)
    assert outputs["hybrid_patch_map"].shape == (2, 4)
    assert outputs["aux"]["decision_input"].shape == (2, 5)


def test_final_logit_backprop_updates_decision_head():
    torch.manual_seed(37)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["final_logit"].sum().backward()

    assert any(parameter.grad is not None for parameter in model.decision_head.parameters())


def test_pqa_logit_backprop_updates_global_heads():
    torch.manual_seed(41)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["pqa_logit"].sum().backward()

    assert any(
        parameter.grad is not None
        for parameter in model.prompt_query_adapter.global_heads.parameters()
    )


def test_pqa_mask_loss_backprop_updates_local_heads():
    torch.manual_seed(43)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    masks = torch.zeros(2, 1, 32, 32)
    masks[1, :, 8:24, 8:24] = 1.0
    loss = compute_pqa_mask_loss(outputs, masks)
    loss.backward()

    assert loss > 0
    assert any(
        parameter.grad is not None
        for parameter in model.prompt_query_adapter.local_heads.parameters()
    )
