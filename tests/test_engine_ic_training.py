import warnings
from types import SimpleNamespace

import torch

from engine_IC import (
    _build_active_model,
    eval_epoch,
    _iter_with_progress,
    _is_adapt_model,
    _progress_enabled,
    _resolve_max_epochs,
    _split_batch_with_optional_masks,
)
from open_clip.config.defaults import get_cfg
from train_local import _as_cfg_path_list, configure_train_local_output


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
    assert cfg.TEXT_BRANCH.TYPE == "adaptclip_prompt"
    assert cfg.TEXT_BRANCH.TEMPLATES == [
        "a photo of a normal object.",
        "a photo of a damaged object.",
    ]
    assert cfg.TEXT_BRANCH.LOGIT_SCALE == 100.0
    assert cfg.TEXT_BRANCH.N_CTX == 12
    assert cfg.TEXT_BRANCH.NORMAL_SUFFIX == "normal object."
    assert cfg.TEXT_BRANCH.ABNORMAL_SUFFIX == "damaged object."
    assert cfg.TEXT_BRANCH.CTX_INIT_STD == 0.02
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
    assert cfg.LOSS.TEXT_MASK_WEIGHT == 0.0


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


def test_progress_enabled_respects_train_config_flag():
    cfg = get_cfg()

    cfg.TRAIN.SHOW_PROGRESS = True
    assert _progress_enabled(cfg) is True

    cfg.TRAIN.SHOW_PROGRESS = False
    assert _progress_enabled(cfg) is False


def test_iter_with_progress_wraps_iterable_when_enabled(monkeypatch):
    cfg = get_cfg()
    cfg.TRAIN.SHOW_PROGRESS = True
    wrapped_calls = []

    def fake_tqdm(iterable, **kwargs):
        wrapped_calls.append(kwargs)
        return iterable

    monkeypatch.setattr("engine_IC.tqdm", fake_tqdm)

    iterable = [1, 2, 3]
    assert list(_iter_with_progress(iterable, cfg, total=3, desc="train")) == iterable
    assert wrapped_calls == [
        {
            "total": 3,
            "desc": "train",
            "unit": "batch",
            "dynamic_ncols": True,
            "leave": False,
        }
    ]


def test_eval_epoch_uses_progress_wrapper(monkeypatch):
    cfg = get_cfg()
    cfg.NUM_GPUS = 0
    cfg.TRAIN.SHOW_PROGRESS = True
    wrapped_calls = []

    def fake_iter_with_progress(iterable, cfg_arg, total=None, desc=None, unit="batch"):
        wrapped_calls.append({
            "cfg": cfg_arg,
            "total": total,
            "desc": desc,
            "unit": unit,
        })
        return iterable

    class FakeModel:
        def __init__(self):
            self.scores = iter([torch.tensor([0.1]), torch.tensor([0.9])])

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return {"final_score": next(self.scores)}

    def make_batch(label):
        inputs = [
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 3, 8, 8),
        ]
        return inputs, ["AITEX"], torch.tensor([label])

    monkeypatch.setattr("engine_IC._iter_with_progress", fake_iter_with_progress)

    auroc, aupr = eval_epoch(
        [make_batch(0), make_batch(1)],
        FakeModel(),
        cfg,
        tokenizer=None,
        mode="test/AITEX",
    )

    assert auroc == 1.0
    assert aupr == 1.0
    assert wrapped_calls == [
        {
            "cfg": cfg,
            "total": 2,
            "desc": "eval test/AITEX",
            "unit": "batch",
        }
    ]


def test_configure_train_local_output_suppresses_warnings():
    cfg = get_cfg()
    cfg.TRAIN.SUPPRESS_WARNINGS = True

    with warnings.catch_warnings(record=True) as caught:
        configure_train_local_output(cfg)
        warnings.warn("noisy training warning", UserWarning)

    assert caught == []


def test_train_local_wraps_single_json_path_for_dataset_constructor():
    assert _as_cfg_path_list("datasets/AD_json/visa/candle_train_normal.json") == [
        "datasets/AD_json/visa/candle_train_normal.json"
    ]
    assert _as_cfg_path_list(["a.json", "b.json"]) == ["a.json", "b.json"]
