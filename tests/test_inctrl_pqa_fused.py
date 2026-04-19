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


def test_fused_forward_exposes_simplified_contract():
    torch.manual_seed(31)
    model = _build_model()
    model.eval()

    outputs = _forward(model)

    required_keys = {
        "final_score",
        "final_logit",
        "patch_score",
        "patch_logit",
        "pqa_score",
        "pqa_logit",
        "image_score",
        "image_logit",
        "text_score",
        "text_logit",
        "fused_patch_map",
        "pqa_local_logits",
        "patch_map_fusion_weights",
        "aux",
    }
    deleted_legacy_keys = {
        "base_score",
        "base_logit",
        "holistic_score",
        "holistic_logit",
        "base_patch_map",
        "hybrid_patch_map",
        "raw_max_patch_score",
        "raw_max_patch_logit",
        "max_base_patch_score",
        "max_base_patch_logit",
        "max_hybrid_patch_score",
        "max_hybrid_patch_logit",
        "max_patch_score",
        "max_patch_logit",
        "patch_map",
        "raw_base_patch_map",
        "pqa_patch_score",
        "pqa_patch_logit",
        "pqa_local_scores",
    }

    assert required_keys.issubset(outputs.keys())
    assert deleted_legacy_keys.isdisjoint(outputs.keys())
    assert outputs["final_logit"].shape == (2,)
    assert outputs["patch_logit"].shape == (2,)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["pqa_local_logits"].shape == (2, 2, 32, 32)
    assert outputs["fused_patch_map"].shape == (2, 4)
    assert outputs["aux"]["decision_input"].shape == (2, 4)


def test_pqadapter_returns_coherent_patch_payload():
    torch.manual_seed(35)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)

    _, query_patch_tokens = model._encode_visual_features(query_image)
    query_patch_levels = model._as_query_level_list(
        model._prepare_patch_levels(query_patch_tokens, batch_size=2, num_shots=1)
    )

    _, prompt_patch_levels, _ = model._encode_prompt_features(
        query_image=query_image,
        prompt_images=prompt_images,
    )

    outputs = model.prompt_query_adapter(
        query_patch_levels=query_patch_levels,
        prompt_patch_levels=prompt_patch_levels,
        beta=torch.tensor(model.beta, device=query_patch_levels[0].device, dtype=query_patch_levels[0].dtype),
    )

    required_keys = {
        "inctrl_patch_maps",
        "pqa_patch_maps",
        "pqa_global_logits",
        "pqa_local_logits",
        "aligned_indices",
        "aligned_prompt_features",
        "layer_weights",
    }
    removed_keys = {
        "patch_logits",
        "patch_scores",
        "pqa_global_logits_2c",
        "pqa_global_scores",
        "pqa_local_scores",
        "residual_maps",
    }

    assert required_keys.issubset(outputs.keys())
    assert removed_keys.isdisjoint(outputs.keys())

    num_layers = len(model.patch_layers)
    assert len(outputs["inctrl_patch_maps"]) == num_layers
    assert len(outputs["pqa_patch_maps"]) == num_layers
    assert len(outputs["pqa_global_logits"]) == num_layers
    assert len(outputs["pqa_local_logits"]) == num_layers
    assert len(outputs["aligned_indices"]) == num_layers
    assert len(outputs["aligned_prompt_features"]) == num_layers

    assert outputs["layer_weights"].shape == (num_layers,)
    assert outputs["inctrl_patch_maps"][0].shape == (2, 4)
    assert outputs["pqa_patch_maps"][0].shape == (2, 4)
    assert outputs["pqa_global_logits"][0].shape == (2,)
    assert outputs["pqa_local_logits"][0].shape == (2, 2, 32, 32)
    assert outputs["aligned_indices"][0].shape == (2, 4)
    assert outputs["aligned_prompt_features"][0].shape == (2, 4, 32)


def test_forward_aux_keeps_patch_diagnostics_without_duplicates():
    torch.manual_seed(36)
    model = _build_model()
    model.eval()

    outputs = _forward(model)
    aux = outputs["aux"]

    expected_aux_keys = {
        "inctrl_patch_map",
        "pqa_patch_map",
        "aligned_indices",
        "aligned_prompt_features",
        "patch_fusion_weights",
        "per_layer_inctrl_patch_map",
        "per_layer_pqa_patch_map",
        "per_layer_pqa_global_logit",
    }
    banned_aux_keys = {
        "per_layer_raw_residual",
        "raw_base_patch_map_2d",
        "base_patch_map_2d",
        "hybrid_patch_map_2d",
    }

    assert expected_aux_keys.issubset(aux.keys())
    assert banned_aux_keys.isdisjoint(aux.keys())

    expected_inctrl_patch_map = sum(
        weight * patch_map
        for weight, patch_map in zip(aux["layer_weights"], aux["per_layer_inctrl_patch_map"])
    )
    expected_pqa_patch_map = sum(
        weight * patch_map
        for weight, patch_map in zip(aux["layer_weights"], aux["per_layer_pqa_patch_map"])
    )
    expected_fused_patch_map = (
        aux["patch_fusion_weights"][0] * aux["inctrl_patch_map"]
        + aux["patch_fusion_weights"][1] * aux["pqa_patch_map"]
    )

    assert torch.allclose(aux["inctrl_patch_map"], expected_inctrl_patch_map, atol=1e-6)
    assert torch.allclose(aux["pqa_patch_map"], expected_pqa_patch_map, atol=1e-6)
    assert torch.allclose(outputs["fused_patch_map"], expected_fused_patch_map, atol=1e-6)
    assert torch.allclose(outputs["patch_map_fusion_weights"], aux["patch_fusion_weights"], atol=1e-6)



def test_final_logit_backprop_updates_decision_head():
    torch.manual_seed(37)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["final_logit"].sum().backward()

    assert any(parameter.grad is not None for parameter in model.decision_head.parameters())


def test_patch_logit_backprop_updates_patch_map_fusion_logits():
    torch.manual_seed(41)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["patch_logit"].sum().backward()

    assert model.patch_map_fusion_logits.grad is not None


def test_pqa_logit_backprop_updates_global_heads():
    torch.manual_seed(47)
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


def test_return_dict_false_forces_tuple_output():
    torch.manual_seed(53)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    obj_types = ["candle", "candle"]

    dict_outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=obj_types,
        return_aux=True,
        return_dict=True,
    )
    tuple_outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=obj_types,
        return_aux=True,
        return_dict=False,
    )

    assert isinstance(tuple_outputs, tuple)
    assert len(tuple_outputs) == 2
    assert tuple_outputs[0].shape == (2,)
    assert tuple_outputs[1].shape == (2, 4)
    assert torch.allclose(tuple_outputs[0], dict_outputs["final_score"], atol=1e-6)
    assert torch.allclose(tuple_outputs[1], dict_outputs["fused_patch_map"], atol=1e-6)


def test_return_dict_none_uses_model_default_output_dict():
    torch.manual_seed(57)
    model = _build_model()
    model.eval()

    outputs = model(
        query_image=torch.randn(2, 3, 32, 32),
        prompt_images=torch.randn(2, 2, 3, 32, 32),
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=None,
    )

    assert isinstance(outputs, dict)
    assert outputs["final_logit"].shape == (2,)


def test_shared_5d_prompt_images_expand_to_query_batch():
    torch.manual_seed(59)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    shared_prompt_images = torch.randn(1, 2, 3, 32, 32)
    coerced = model._coerce_prompt_images(query_image, prompt_images=shared_prompt_images)

    assert coerced.shape == (2, 2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        prompt_images=shared_prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    assert outputs["final_logit"].shape == (2,)


def test_4d_prompt_images_raise_for_multi_query_batch():
    torch.manual_seed(61)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)

    try:
        model._coerce_prompt_images(query_image, prompt_images=prompt_images)
    except ValueError as error:
        assert "prompt_images with query batch size > 1" in str(error)
    else:
        raise AssertionError("Expected ValueError for ambiguous 4D prompt_images")


def test_prompt_feature_cache_round_trip_still_works():
    torch.manual_seed(67)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)
    cache = model.build_prompt_feature_cache(prompt_images=prompt_images)

    assert cache["prompt_global"].shape == (2, 32)
    assert len(cache["prompt_patch_levels"]) == 3
    assert cache["prompt_patch_levels"][0].shape == (2, 4, 32)
    assert cache["num_shots"] == 2

    direct_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    cached_outputs = model(
        query_image=query_image,
        prompt_feature_cache=cache,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )

    assert torch.allclose(cached_outputs["final_logit"], direct_outputs["final_logit"], atol=1e-6)
    assert torch.allclose(cached_outputs["fused_patch_map"], direct_outputs["fused_patch_map"], atol=1e-6)


def test_text_prototype_cache_round_trip_still_works():
    torch.manual_seed(71)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)
    cache = model.build_text_prototype_cache(
        obj_types=["candle", "candle"],
        device=torch.device("cpu"),
    )

    assert cache["normal_proto"].shape == (2, 32)
    assert cache["anomaly_proto"].shape == (2, 32)

    direct_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    cached_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        text_prototype_cache=cache,
        return_aux=True,
        return_dict=True,
    )

    assert torch.allclose(cached_outputs["text_logit"], direct_outputs["text_logit"], atol=1e-6)
    assert torch.allclose(cached_outputs["final_logit"], direct_outputs["final_logit"], atol=1e-6)
