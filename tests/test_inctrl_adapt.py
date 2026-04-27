import torch

from open_clip.inctrl_adapt import (
    InCTRLAdapt,
    _fuse_maps,
    _fuse_scores,
    _get_vision_width,
    _score_to_logit,
)


def _make_model_stub():
    """Create a lightweight stub that has the same interface as InCTRLAdapt."""
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    model.use_visual_adapter = True
    model.use_pqa = True
    model.use_seg_head = True
    model.use_text_branch = True
    # Monkeypatch get_visual_parameters / get_text_parameters to use plain tensors
    model._visual_params = [torch.randn(4, 4, requires_grad=True)]
    model._image_head_params = [torch.randn(4, 1, requires_grad=True)]
    model._pqa_params = [torch.randn(4, 4, requires_grad=True)]
    model._proj_params = [torch.randn(4, 4, requires_grad=True)]
    model._text_params = [torch.randn(4, 4, requires_grad=True)]
    return model


def _stub_get_visual_parameters(self):
    params = []
    if self.use_visual_adapter:
        params.extend(self._visual_params)
    params.extend(self._image_head_params)
    if self.use_pqa:
        params.extend(self._pqa_params)
    if self._proj_params:
        params.extend(self._proj_params)
    return params


def _stub_get_text_parameters(self):
    if not self.use_text_branch:
        return []
    return list(self._text_params)


def test_score_to_logit_is_finite_at_edges():
    scores = torch.tensor([0.0, 0.5, 1.0])
    logits = _score_to_logit(scores)

    assert torch.isfinite(logits).all()
    assert logits[0] < 0
    assert logits[1].abs() < 1e-6
    assert logits[2] > 0


def test_fuse_scores_uses_config_weights_without_learning_competition():
    fused = _fuse_scores(
        image_logit=torch.tensor([1.0]),
        patch_logit=torch.tensor([2.0]),
        pqa_logit=torch.tensor([3.0]),
        text_logit=torch.tensor([4.0]),
        weights=(0.35, 0.25, 0.25, 0.15),
    )

    assert torch.allclose(fused, torch.tensor([2.2]))


def test_fuse_maps_preserves_shape_and_weighting():
    residual = torch.ones(1, 1, 4, 4)
    pqa = torch.ones(1, 1, 4, 4) * 2
    text = torch.ones(1, 1, 4, 4) * 3

    fused = _fuse_maps(residual, pqa, text, weights=(0.4, 0.4, 0.2))

    assert fused.shape == (1, 1, 4, 4)
    assert torch.allclose(fused, torch.ones(1, 1, 4, 4) * 1.8)


def test_get_vision_width_reads_patch_token_dimension_from_config():
    assert _get_vision_width({"width": 896}, fallback=640) == 896
    assert _get_vision_width({}, fallback=640) == 640


def test_flatten_prompt_tokens_collapses_shot_and_patch_dimensions():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    prompts = torch.randn(2, 3, 4, 8)

    flat = InCTRLAdapt._flatten_prompt_tokens(model, prompts)

    assert flat.shape == (2, 12, 8)


def test_compute_patch_residual_returns_zero_for_identical_query_and_prompt_tokens():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    prompts = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    residual = InCTRLAdapt._compute_patch_residual(model, query, prompts)

    assert torch.allclose(residual, torch.zeros(1, 2), atol=1e-6)


def test_upsample_patch_map_expands_square_patch_grid_to_image_size():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    model.image_size = 32
    patch_map = torch.randn(2, 4)

    upsampled = InCTRLAdapt._upsample_patch_map(model, patch_map)

    assert upsampled.shape == (2, 1, 32, 32)


def test_disabled_text_branch_outputs_neutral_zero_logits_without_tokenizer():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    model.image_size = 32
    query_global = torch.randn(2, 8)
    patch_tokens = torch.randn(2, 4, 8)

    outputs = InCTRLAdapt._zero_text_outputs(model, query_global, patch_tokens)

    assert torch.equal(outputs["text_logit"], torch.zeros(2))
    assert torch.allclose(outputs["text_score"], torch.full((2,), 0.5))
    assert outputs["text_map"].shape == (2, 1, 32, 32)


def test_disabled_pqa_branch_outputs_neutral_zero_logits_and_map():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    model.image_size = 32
    query_tokens = torch.randn(2, 4, 8)

    outputs = InCTRLAdapt._zero_pqa_outputs(model, query_tokens)

    assert torch.equal(outputs["pqa_logit"], torch.zeros(2))
    assert torch.allclose(outputs["pqa_score"], torch.full((2,), 0.5))
    assert outputs["pqa_patch_map"].shape == (2, 4)
    assert outputs["pqa_seg_logits"].shape == (2, 1, 32, 32)
    assert torch.equal(outputs["pqa_patch_map"], torch.zeros(2, 4))
    assert torch.equal(outputs["pqa_seg_logits"], torch.zeros(2, 1, 32, 32))


def test_disabled_pqa_branch_contributes_zero_final_map_component():
    model = InCTRLAdapt.__new__(InCTRLAdapt)
    model.use_pqa = False
    logits = torch.zeros(2, 1, 32, 32)

    pqa_map = InCTRLAdapt._pqa_map_from_logits(model, logits)

    assert torch.equal(pqa_map, torch.zeros_like(logits))


def test_get_visual_parameters_returns_adapter_and_head_params():
    model = _make_model_stub()
    model.get_visual_parameters = lambda: _stub_get_visual_parameters(model)
    params = model.get_visual_parameters()
    assert len(params) == 4
    for p in params:
        assert p.requires_grad is True


def test_get_text_parameters_returns_text_branch_params():
    model = _make_model_stub()
    model.get_text_parameters = lambda: _stub_get_text_parameters(model)
    params = model.get_text_parameters()
    assert len(params) == 1
    assert params[0].requires_grad is True


def test_get_text_parameters_returns_empty_when_text_disabled():
    model = _make_model_stub()
    model.use_text_branch = False
    model.get_text_parameters = lambda: _stub_get_text_parameters(model)
    params = model.get_text_parameters()
    assert params == []


def test_set_train_phase_visual_enables_visual_and_disables_text():
    model = _make_model_stub()
    model.get_visual_parameters = lambda: _stub_get_visual_parameters(model)
    model.get_text_parameters = lambda: _stub_get_text_parameters(model)
    InCTRLAdapt.set_train_phase(model, "visual")

    for p in model.get_visual_parameters():
        assert p.requires_grad is True
    for p in model.get_text_parameters():
        assert p.requires_grad is False


def test_set_train_phase_text_enables_text_and_disables_visual():
    model = _make_model_stub()
    model.get_visual_parameters = lambda: _stub_get_visual_parameters(model)
    model.get_text_parameters = lambda: _stub_get_text_parameters(model)
    InCTRLAdapt.set_train_phase(model, "text")

    for p in model.get_visual_parameters():
        assert p.requires_grad is False
    for p in model.get_text_parameters():
        assert p.requires_grad is True
