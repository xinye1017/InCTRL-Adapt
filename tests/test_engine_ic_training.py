from types import SimpleNamespace

import torch

from engine_IC import (
    _build_active_model,
    _is_adapt_model,
    _resolve_max_epochs,
    _split_batch_with_optional_masks,
)
from open_clip.config.defaults import get_cfg
from train_local import _as_cfg_path_list


def test_resolve_max_epochs_uses_explicit_cfg_value():
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(MAX_EPOCH=1))

    assert _resolve_max_epochs(cfg) == 1


def test_resolve_max_epochs_keeps_legacy_default_for_generic_cfg_default():
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(MAX_EPOCH=400))

    assert _resolve_max_epochs(cfg) == 10


def test_pqa_lite_config_defaults_are_available():
    cfg = get_cfg()

    assert cfg.MODEL.ACTIVE_MODEL == "InCTRLAdapt"
    assert cfg.VISUAL_ADAPTER.ENABLE is True
    assert cfg.VISUAL_ADAPTER.REDUCTION == 4
    assert cfg.VISUAL_ADAPTER.ZERO_INIT is True
    assert cfg.TEXT_BRANCH.ENABLE is True
    assert cfg.TEXT_BRANCH.TEMPLATES == [
        "a photo of a normal object.",
        "a photo of a damaged object.",
    ]
    assert cfg.TEXT_BRANCH.LOGIT_SCALE == 100.0
    assert cfg.PQA.ENABLE is True
    assert cfg.PQA.ENABLE_SEG_HEAD is True
    assert cfg.PQA.PATCH_LAYERS == [7, 9, 11]
    assert cfg.PQA.CONTEXT_BETA == 1.0
    assert cfg.PQA.HIDDEN_DIM == 128
    assert cfg.PQA.GLOBAL_TOPK == 10
    assert cfg.FUSION.IMAGE_WEIGHT == 0.35
    assert cfg.FUSION.PATCH_WEIGHT == 0.25
    assert cfg.FUSION.PQA_WEIGHT == 0.25
    assert cfg.FUSION.TEXT_WEIGHT == 0.15
    assert cfg.FUSION.MAP_RES_WEIGHT == 0.4
    assert cfg.FUSION.MAP_PQA_WEIGHT == 0.4
    assert cfg.FUSION.MAP_TEXT_WEIGHT == 0.2
    assert cfg.LOSS.IMAGE_WEIGHT == 1.0
    assert cfg.LOSS.PQA_WEIGHT == 0.5
    assert cfg.LOSS.MASK_WEIGHT == 1.0
    assert cfg.LOSS.TEXT_WEIGHT == 0.0


def test_split_batch_with_optional_masks_supports_four_item_batch():
    inputs = [
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 3, 32, 32),
    ]
    types = ["candle", "candle"]
    labels = torch.tensor([0, 1])
    masks = torch.randn(2, 1, 32, 32)

    query, prompts, out_types, out_labels, out_masks = _split_batch_with_optional_masks(
        (inputs, types, labels, masks)
    )

    assert query.shape == (2, 3, 32, 32)
    assert prompts.shape == (2, 2, 3, 32, 32)
    assert out_types == types
    assert torch.equal(out_labels, labels)
    assert torch.equal(out_masks, masks)


def test_build_active_model_returns_inctrl_pqa_for_default_cfg():
    cfg = get_cfg()
    model_cfg = {
        "embed_dim": 8,
        "vision_cfg": {
            "image_size": 32,
            "layers": 2,
            "width": 8,
            "patch_size": 16,
            "head_width": 4,
            "mlp_ratio": 2.0,
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 8,
            "heads": 2,
            "layers": 1,
        },
    }

    model = _build_active_model(cfg, model_cfg, cast_dtype=None, quick_gelu=False)

    assert model.__class__.__name__ == "InCTRLAdapt"


def test_active_model_switch_identifies_legacy_model_path():
    cfg = get_cfg()
    assert _is_adapt_model(cfg) is True

    cfg.MODEL.ACTIVE_MODEL = "InCTRL"
    assert _is_adapt_model(cfg) is False


def test_train_local_wraps_single_json_path_for_dataset_constructor():
    assert _as_cfg_path_list("datasets/AD_json/visa/candle_train_normal.json") == [
        "datasets/AD_json/visa/candle_train_normal.json"
    ]
    assert _as_cfg_path_list(["a.json", "b.json"]) == ["a.json", "b.json"]
