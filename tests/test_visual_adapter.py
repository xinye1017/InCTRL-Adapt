import json
from pathlib import Path

import torch

from open_clip.config.defaults import get_cfg
from open_clip.model import InCTRL


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"


def _load_model_config():
    with MODEL_CONFIG_PATH.open("r") as handle:
        return json.load(handle)


def _fake_tokenizer(texts, context_length=77):
    tokens = torch.zeros((len(texts), context_length), dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, -1] = 2
    return tokens


def test_visual_adapter_config_defaults_exist():
    cfg = get_cfg()

    assert cfg.VISUAL_ADAPTER.ENABLE is True
    assert cfg.VISUAL_ADAPTER.REDUCTION == 4
    assert cfg.VISUAL_ADAPTER.ZERO_INIT is True


def test_resmlp_supports_2d_and_3d_inputs():
    from open_clip.visual_adapter import ResMLP

    module = ResMLP(8, reduction=2)
    x2 = torch.randn(2, 8)
    x3 = torch.randn(2, 5, 8)

    y2 = module(x2)
    y3 = module(x3)

    assert y2.shape == x2.shape
    assert y3.shape == x3.shape


def test_visual_adapter_preserves_global_and_local_shapes():
    from open_clip.visual_adapter import VisualAdapter

    adapter = VisualAdapter(
        img_size=240,
        patch_size=16,
        global_input_dim=640,
        local_input_dim=896,
        reduction=4,
        zero_init=True,
    )

    global_tokens = torch.randn(2, 640)
    global_shots = torch.randn(2, 3, 640)
    patch_tokens = [torch.randn(2, 226, 896) for _ in range(3)]

    adapted_global = adapter.adapt_global(global_tokens)
    adapted_shots = adapter.adapt_global(global_shots)
    adapted_local = adapter.adapt_local(patch_tokens)

    assert adapted_global.shape == global_tokens.shape
    assert adapted_shots.shape == global_shots.shape
    assert len(adapted_local) == len(patch_tokens)
    assert [tensor.shape for tensor in adapted_local] == [tensor.shape for tensor in patch_tokens]


def test_inctrl_forward_keeps_outputs_and_trainable_paths():
    model_cfg = _load_model_config()
    args = get_cfg()
    args.NUM_GPUS = 0
    args.image_size = 240
    args.shot = 2
    args.VISUAL_ADAPTER.ENABLE = True
    model = InCTRL(
        args,
        model_cfg["embed_dim"],
        model_cfg["vision_cfg"],
        model_cfg["text_cfg"],
        quick_gelu=False,
        cast_dtype=None,
    )
    model.train()

    batch_size = 2
    query = torch.randn(batch_size, 3, 240, 240)
    normal_1 = torch.randn(batch_size, 3, 240, 240)
    normal_2 = torch.randn(batch_size, 3, 240, 240)

    final_score, img_ref_score = model(
        _fake_tokenizer,
        [query, normal_1, normal_2],
        ["candle", "candle"],
        None,
    )

    assert final_score.shape == (batch_size,)
    assert img_ref_score.shape == (batch_size,)

    loss = final_score.sum() + img_ref_score.sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in model.visual.parameters())
    assert all(parameter.grad is None for parameter in model.transformer.parameters())
    assert all(parameter.grad is None for parameter in model.token_embedding.parameters())
    assert any(parameter.grad is not None for parameter in model.visual_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.diff_head.parameters())
    assert any(parameter.grad is not None for parameter in model.diff_head_ref.parameters())
